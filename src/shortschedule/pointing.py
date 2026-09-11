"""Where the spacecraft points across a week, minute by minute.

Between observations the spacecraft goes to dark-idle.

Angles are computed from unit vectors in a common GCRS basis. The one
thing not derived here is the star tracker pointing during an
observation: that encodes the payload attitude convention, so it comes
from `pandoravisibility` rather than being restated.
"""

# Standard library
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Third-party
import numpy as np
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import GCRS, EarthLocation, SkyCoord, get_body
from astropy.time import Time
from pandoravisibility import Visibility

from .models import ScienceCalendar

# Label used for minutes with no observation running.  Reserved: plots
# always draw it black, so no target may use this name.
IDLE_LABEL = "IDLE"

# The three pointing directions and the three bodies every plot works
# over.  "earth" is the angle to the Earth *center* (nadir), matching
# the reference pointing analysis, not the angle above the limb that
# the keep-outs are tested against.
AXES = ("Boresight", "ST1", "ST2")
BODIES = ("sun", "earth", "moon")

# Body-axis unit vectors for PRI_CMD_DIR / SEC_CMD_DIR
# (138DOC0042 RevA, Table 6): 1..6 = +X, +Y, +Z, -X, -Y, -Z.
_BODY_AXIS = {
    1: np.array([1.0, 0.0, 0.0]),
    2: np.array([0.0, 1.0, 0.0]),
    3: np.array([0.0, 0.0, 1.0]),
    4: np.array([-1.0, 0.0, 0.0]),
    5: np.array([0.0, -1.0, 0.0]),
    6: np.array([0.0, 0.0, -1.0]),
}

# The dark-idle GOTO_TARGET, taken from the flown test sequence
# dark_idles/dark_idle_v4_2.seq:
#
#   PRI_REF_DIR 13 (nadir) -> REFERENCE +Z
#   SEC_REF_DIR  3 (Sun)   -> REFERENCE +X, best effort
#   PRI_CMD_DIR  6 (-Z)    -> COMMAND +Z
#   SEC_CMD_DIR  4 (-X)    -> COMMAND +X
#   ATT_INTERP 0 with Q_TARGE_TWRT_REF = (roll 0, pitch -45, yaw 180)
#
# Other dark-idle tests flew (0, -45, 0) and (0, 45, 0); the offset is
# the only thing that differs between them, so it is a parameter.
DARK_IDLE_EULER_DEG = (0.0, -45.0, 180.0)
DARK_IDLE_PRI_CMD = 6
DARK_IDLE_SEC_CMD = 4


def _unit_from_radec(ra_deg, dec_deg) -> np.ndarray:
    """Unit vector(s) from right ascension and declination in degrees.

    Parameters
    ----------
    ra_deg, dec_deg : array_like
        Angles in degrees, any common shape.

    Returns
    -------
    numpy.ndarray
        Cartesian unit vectors with the 3 components on axis 0.
    """
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    return np.stack(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    )


def _angle_between(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Angle in degrees between two sets of unit vectors.

    Parameters
    ----------
    first, second : numpy.ndarray
        Unit vectors with the 3 components on axis 0, broadcastable
        against one another.

    Returns
    -------
    numpy.ndarray
        Angles in degrees.
    """
    dot = np.sum(first * second, axis=0)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def _build_frame(primary_z: np.ndarray, secondary_x: np.ndarray):
    """Right-handed frame with +Z on the primary and +X near the secondary.

    This is the BCT REFERENCE/COMMAND frame construction: the primary
    direction is placed exactly, the secondary only as well as
    orthogonality allows (BCT GNC Users Guide 0DOC5120 RevC).

    Parameters
    ----------
    primary_z : numpy.ndarray, shape (n, 3)
        Direction forced onto the frame's +Z axis.
    secondary_x : numpy.ndarray, shape (n, 3)
        Direction the +X axis is driven toward.

    Returns
    -------
    numpy.ndarray, shape (n, 3, 3)
        Rotation matrices whose columns are the frame's x, y, z axes.
    """
    z_axis = primary_z / np.linalg.norm(primary_z, axis=1, keepdims=True)
    x_axis = (
        secondary_x
        - np.einsum("ij,ij->i", secondary_x, z_axis)[:, None] * z_axis
    )
    x_axis = x_axis / np.linalg.norm(x_axis, axis=1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=2)


def _euler_zyx_matrix(roll_deg, pitch_deg, yaw_deg) -> np.ndarray:
    """Rotation matrix for an ``ATT_INTERP 0`` offset.

    The commanded offset is applied to the REFERENCE axes as
    ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

    Parameters
    ----------
    roll_deg, pitch_deg, yaw_deg : float
        Euler angles in degrees, in the order the command carries them
        (``Q_TARGE_TWRT_REF1/2/3``).

    Returns
    -------
    numpy.ndarray, shape (3, 3)
    """
    roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ]
    )
    rot_y = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    rot_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rot_z @ rot_y @ rot_x


class PointingTimeline:
    """Per-minute pointing and keep-out angles for a whole calendar.

    Built by :func:`build_pointing_timeline`.  Every array is indexed by
    the same minute grid, so a mask taken from ``labels`` selects the
    matching entries of every angle series.

    Attributes
    ----------
    times : astropy.time.Time
        The minute grid, spanning the first observation start to the
        last observation stop.
    labels : numpy.ndarray of str
        Target observed at each minute, or ``IDLE_LABEL``.
    targets : list of str
        Unique target names in first-observed order.  Excludes idle.
    angles : dict
        ``(axis, body) -> ndarray`` of degrees, for every axis in
        :data:`AXES` and body in :data:`BODIES`.
    illumination : dict
        ``axis -> ndarray`` of degrees.  The Earth illumination angle at
        the reference point the visibility model uses
        (``daynight_mode``): the solar zenith angle at the sub-satellite
        point directly below the spacecraft, the library default and the
        same for every axis, or at the limb point that axis grazes.  0 is
        the sub-solar point, 90 the terminator, 180 the anti-solar point.
    earth_radius_deg : float
        Mean Earth angular radius over the span, in degrees, which turns
        a keep-out given above the limb into an angle from the Earth
        center.
    ra, dec : dict
        ``axis -> ndarray`` of degrees, where each axis points.
    segments : list of tuple
        ``(start_index, stop_index, label)`` for each run of constant
        label, so plots can draw one line per observation and per idle
        stretch without re-deriving the runs.
    unresolved : list of str
        Groups whose star tracker pointing could not be evaluated; their
        entries are NaN.
    """

    def __init__(
        self,
        times: Time,
        labels: np.ndarray,
        angles: Dict[Tuple[str, str], np.ndarray],
        illumination: Dict[str, np.ndarray],
        earth_radius_deg: float,
        ra: Dict[str, np.ndarray],
        dec: Dict[str, np.ndarray],
        unresolved: List[str],
    ) -> None:
        self.times = times
        self.labels = labels
        self.angles = angles
        self.illumination = illumination
        self.earth_radius_deg = earth_radius_deg
        self.ra = ra
        self.dec = dec
        self.unresolved = unresolved

        self.targets = []
        for label in labels:
            if label != IDLE_LABEL and label not in self.targets:
                self.targets.append(label)

        self.segments = []
        if len(labels):
            edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
            bounds = np.concatenate(([0], edges, [len(labels)]))
            for start, stop in zip(bounds[:-1], bounds[1:]):
                self.segments.append((int(start), int(stop), labels[start]))

    @property
    def observed_fraction(self) -> float:
        """Fraction of the spanned minutes that have an observation."""
        if not len(self.labels):
            return 0.0
        return float(np.mean(self.labels != IDLE_LABEL))


def build_pointing_timeline(
    calendar: ScienceCalendar,
    visibility: Any,
    step_minutes: int = 1,
    idle_euler_deg: Sequence[float] = DARK_IDLE_EULER_DEG,
    progress: Optional[Callable] = None,
    verbose: bool = False,
) -> PointingTimeline:
    """Reconstruct where the spacecraft points for every minute of a week.

    Observation minutes take the attitude implied by the observation's
    target and prescribed roll; the star tracker directions come from
    ``visibility.get_star_tracker_angles`` so the payload attitude
    convention is not restated here.  Idle minutes take the dark-idle
    command attitude, built from nadir and the Sun.

    Parameters
    ----------
    calendar : ScienceCalendar
        Calendar to walk.  Its observations define both the minute grid
        and which minutes are not idle.
    visibility : pandoravisibility.Visibility
        Supplies the orbit (``get_state``), the star tracker pointing and
        the Earth reference point for the illumination angle
        (``daynight_mode``).  Its ``roll`` attribute is set and restored
        around each group of observations.
    step_minutes : int, optional
        Grid spacing.  One minute matches the scheduler's own
        granularity; coarser is much faster on long spans.
    idle_euler_deg : sequence of float, optional
        ``(roll, pitch, yaw)`` offset of the dark-idle command, in
        degrees.  Defaults to :data:`DARK_IDLE_EULER_DEG`.
    progress : callable, optional
        ``progress(iterable, desc=..., total=...)`` returning the iterable
        wrapped in a progress bar, such as ``ScheduleProcessor._progress``.
        The per-group star tracker lookups are the slow part of a week.
    verbose : bool, optional
        Print progress and any groups that could not be resolved.

    Returns
    -------
    PointingTimeline
    """
    observations = sorted(
        (
            (visit.id, seq)
            for visit in calendar.visits
            for seq in visit.sequences
        ),
        key=lambda item: item[1].start_time,
    )
    if not observations:
        raise ValueError("Calendar has no observations to plot.")

    span_start = observations[0][1].start_time
    span_stop = max(seq.stop_time for _, seq in observations)
    n_steps = int(np.rint((span_stop - span_start).sec / 60.0 / step_minutes))
    if n_steps <= 0:
        raise ValueError("Calendar spans less than one grid step.")
    times = span_start + np.arange(n_steps) * step_minutes * u.min

    # Label every minute, and collect the observation minutes into
    # groups that share a pointing, so each group costs one call rather
    # than one call per observation.
    labels = np.full(n_steps, IDLE_LABEL, dtype=object)
    groups: Dict[Tuple[float, float, Optional[float]], List[int]] = {}
    for visit_id, seq in observations:
        first = int(
            np.rint((seq.start_time - span_start).sec / 60.0 / step_minutes)
        )
        last = int(
            np.rint((seq.stop_time - span_start).sec / 60.0 / step_minutes)
        )
        indices = [i for i in range(max(first, 0), min(last, n_steps))]
        if not indices:
            continue
        labels[indices] = seq.target

        key = (round(float(seq.ra), 6), round(float(seq.dec), 6), seq.roll)
        groups.setdefault(key, []).extend(indices)

    labels = np.array([str(label) for label in labels])

    # Orbit geometry, shared by every axis and every minute.
    state = visibility.get_state(times)
    location = EarthLocation.from_geocentric(state.x, state.y, state.z)
    observer_gcrs = location.get_gcrs(obstime=times)
    observer_eci = observer_gcrs.cartesian.xyz.to(u.m).value
    observer_distance = np.linalg.norm(observer_eci, axis=0)
    zenith_unit = observer_eci / observer_distance[np.newaxis, :]
    nadir_unit = -zenith_unit
    with np.errstate(invalid="ignore"):
        limb_angle_rad = np.arccos(R_earth.to(u.m).value / observer_distance)
    earth_radius_deg = float(np.nanmean(90.0 - np.degrees(limb_angle_rad)))

    sun_coord = get_body("sun", times, location=location)
    moon_coord = get_body("moon", times, location=location)
    sun_unit = _unit_from_radec(sun_coord.ra.deg, sun_coord.dec.deg)
    moon_unit = _unit_from_radec(moon_coord.ra.deg, moon_coord.dec.deg)
    body_units = {"sun": sun_unit, "earth": nadir_unit, "moon": moon_unit}

    pointing = {axis: np.full((3, n_steps), np.nan) for axis in AXES}
    unresolved: List[str] = []

    # Observation minutes
    original_roll = getattr(visibility, "roll", None)
    group_items = list(groups.items())
    if progress is not None:
        group_items = progress(
            group_items, desc="Reconstructing pointing", total=len(group_items)
        )
    try:
        for (ra_deg, dec_deg, roll), indices in group_items:
            index = np.asarray(sorted(indices))
            group_times = times[index]
            coord = SkyCoord(ra_deg, dec_deg, frame="icrs", unit="deg")

            target_gcrs = coord.transform_to(GCRS(obstime=group_times))
            target_xyz = target_gcrs.cartesian.xyz.value
            pointing["Boresight"][:, index] = target_xyz / np.linalg.norm(
                target_xyz, axis=0, keepdims=True
            )

            visibility.roll = None if roll is None else roll * u.deg
            for tracker, axis in ((1, "ST1"), (2, "ST2")):
                try:
                    tracker_angles = visibility.get_star_tracker_angles(
                        coord, group_times, tracker
                    )
                except Exception as error:
                    unresolved.append(
                        f"{axis} at RA {ra_deg:.4f} Dec {dec_deg:.4f} "
                        f"roll {roll}: {error}"
                    )
                    continue
                pointing[axis][:, index] = _unit_from_radec(
                    np.atleast_1d(tracker_angles["ra"].to(u.deg).value),
                    np.atleast_1d(tracker_angles["dec"].to(u.deg).value),
                )
    finally:
        visibility.roll = original_roll

    # Idle minutes
    idle = np.flatnonzero(labels == IDLE_LABEL)
    if idle.size:
        reference_to_eci = _build_frame(
            nadir_unit[:, idle].T, sun_unit[:, idle].T
        )
        target_to_eci = reference_to_eci @ _euler_zyx_matrix(*idle_euler_deg)
        command_to_body = _build_frame(
            _BODY_AXIS[DARK_IDLE_PRI_CMD][np.newaxis, :],
            _BODY_AXIS[DARK_IDLE_SEC_CMD][np.newaxis, :],
        )[0]
        body_to_eci = target_to_eci @ command_to_body.T

        pointing["Boresight"][:, idle] = body_to_eci[:, :, 2].T
        for tracker, axis in ((1, "ST1"), (2, "ST2")):
            tracker_body = np.array(
                Visibility._get_star_tracker_body_xyz(tracker)
            )
            pointing[axis][:, idle] = np.einsum(
                "nij,j->ni", body_to_eci, tracker_body
            ).T

    # Angles
    angles = {}
    illumination = {}
    right_ascension = {}
    declination = {}
    for axis in AXES:
        unit = pointing[axis]
        for body in BODIES:
            angles[(axis, body)] = _angle_between(unit, body_units[body])
        # The library's own illumination angle at its own reference point
        # (sub-satellite by default, or the grazed limb point), so the
        # scatter and the dynamic keep-out cannot disagree about either.
        # Vectors are (3, N), which is the layout it expects by default.
        illumination[axis] = visibility._daynight_illumination_angle(
            unit,
            zenith_unit,
            sun_unit,
            limb_angle_rad=limb_angle_rad,
        )
        right_ascension[axis] = (
            np.degrees(np.arctan2(unit[1], unit[0])) % 360.0
        )
        declination[axis] = np.degrees(np.arcsin(np.clip(unit[2], -1.0, 1.0)))

    if verbose:
        print(
            f"Pointing timeline: {n_steps} steps of {step_minutes} min, "
            f"{len(groups)} pointing groups, {idle.size} idle steps."
        )
        for problem in unresolved:
            print(f"  WARNING: could not resolve {problem}")

    return PointingTimeline(
        times=times,
        labels=labels,
        angles=angles,
        illumination=illumination,
        earth_radius_deg=earth_radius_deg,
        ra=right_ascension,
        dec=declination,
        unresolved=unresolved,
    )
