"""Spacecraft roll angle calculations for Pandora observations.

This module provides utilities to compute the spacecraft roll angle
required for each observation sequence. The roll angle depends on:
- Target coordinates (RA, Dec)
- Sun position at the time of observation

Roll must remain consistent for all
observations of the same target within a single visit.

When star-tracker keep-out constraints are active the module can
sweep candidate roll angles and select the one that maximises
visible time while respecting a minimum solar-power-fraction
threshold.

Notes
-----
Roll angles are expressed in degrees. The calculation ensures the
solar panels maintain proper sun exposure.
"""

# Standard library
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time


def calculate_roll(
    ra: float,
    dec: float,
    obs_time: Time,
) -> float:
    """Calculate the spacecraft roll angle for a given target and time.

    The roll angle is computed based on the position of the target
    relative to the Sun at the time of observation. This ensures
    proper solar panel orientation.

    Parameters
    ----------
    ra : float
        Right ascension of the target in degrees.
    dec : float
        Declination of the target in degrees.
    obs_time : Time
        The time of observation (used to determine Sun position).

    Returns
    -------
    float
        The spacecraft roll angle in degrees, in the range (-180, 180].
    """

    # Get target coordinates
    target_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    # Get Sun position at observation time (astropy >=5 returns GCRS)
    sun_coord = get_sun(obs_time)
    # Convert to ICRS if needed
    if not hasattr(sun_coord, "ra") or not hasattr(sun_coord, "dec"):
        sun_coord = sun_coord.transform_to("icrs")

    # Calculate roll angle
    roll, _ = _spacecraft_roll_from_radec(
        target_coord.ra.deg,
        target_coord.dec.deg,
        sun_coord.ra.deg,
        sun_coord.dec.deg,
    )
    roll = (roll + 180.0) % 360.0 - 180.0  # Normalize to (-180, 180]
    return roll


def _spacecraft_roll_from_radec(target_ra, target_dec, sun_ra, sun_dec):
    """
    Compute spacecraft roll about boresight defined by:
        zB = target direction
        yB = sun × target (or fallback for anti-Solar pointing)
        xB = yB × zB
    All in ECI.

    For when the target is nearly anti-Solar, we use celestial north projected onto the plane
    perpendicular to the boresight as the reference for yB.
    """
    # Convert to vectors
    t = radec_to_vector(target_ra, target_dec)  # boresight
    s = radec_to_vector(sun_ra, sun_dec)  # Sun direction

    # boresight body axes, we define the others later because of edge cases
    zB = normalize(t)

    # Celestial north in ECI (equatorial frame) is +Z_ECI
    north = np.array([0.0, 0.0, 1.0])
    north_proj = north - np.dot(north, zB) * zB

    # catch edge case where boresight is very close to celestial north
    # in practice this should never happen because of Sun constraints
    if np.linalg.norm(north_proj) < 1.0e-8:
        # boresight ≈ celestial north: pick arbitrary east-like vector
        east = np.array([1.0, 0.0, 0.0])
        north_proj = east - np.dot(east, zB) * zB

    # reference axes for roll angle calculation
    x_ref = normalize(north_proj)
    y_ref = normalize(np.cross(zB, x_ref))

    # Define spacecraft xB,yB using the Sun direction
    sun_cross = np.cross(s, zB)
    if np.linalg.norm(sun_cross) > 1e-6:
        yB = normalize(sun_cross)
        xB = normalize(np.cross(yB, zB))
    else:
        # we need to catch edge cases where Sun is very close to anti-boresight
        xB = x_ref
        yB = y_ref

    # Compute roll angle x_ref -> xB around zB
    cos_r = np.dot(x_ref, xB)
    sin_r = np.dot(y_ref, xB)
    roll = np.rad2deg(np.arctan2(sin_r, cos_r))  # degrees

    return roll, (xB, yB, zB)


def radec_to_vector(ra_deg, dec_deg):
    """Convert RA/Dec (deg) to an Earth-Centered Inertial (ECI) coordinate frame unit vector."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y, z])


def vector_to_radec(v):
    """Convert an ECI coordinate frame unit vector to RA/Dec (deg)."""
    x, y, z = v
    r = np.sqrt(x**2 + y**2 + z**2)
    dec = np.arcsin(z / r)
    ra = np.arctan2(y, x)
    ra_deg = np.rad2deg(ra) % 360.0
    dec_deg = np.rad2deg(dec)
    return ra_deg, dec_deg


def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n


# ------------------------------------------------------------------
# Solar power fraction helpers
# ------------------------------------------------------------------


def _payload_axes_from_roll(ra, dec, roll_deg):
    """Compute payload X, Y, Z body axes for a target and roll.

    Uses the same celestial-north-projection reference frame as
    ``_spacecraft_roll_from_radec`` and ``pandoravisibility``'s
    ``_roll_attitude``.

    Parameters
    ----------
    ra, dec : float
        Target right ascension and declination in degrees.
    roll_deg : float
        Spacecraft roll angle about boresight in degrees.

    Returns
    -------
    x_payload, y_payload, z_payload : np.ndarray
        (3,) unit vectors for the payload +X, +Y, +Z axes in ECI.
    """
    z_unit = normalize(radec_to_vector(ra, dec))

    north = np.array([0.0, 0.0, 1.0])
    north_proj = north - np.dot(north, z_unit) * z_unit
    if np.linalg.norm(north_proj) < 1.0e-8:
        east = np.array([1.0, 0.0, 0.0])
        north_proj = east - np.dot(east, z_unit) * z_unit
    x_ref = normalize(north_proj)
    y_ref = normalize(np.cross(z_unit, x_ref))

    roll_rad = np.deg2rad(roll_deg)
    cos_r = np.cos(roll_rad)
    sin_r = np.sin(roll_rad)
    x_payload = cos_r * x_ref + sin_r * y_ref
    y_payload = -sin_r * x_ref + cos_r * y_ref
    return x_payload, y_payload, z_unit


def compute_solar_power_fraction(
    ra: float,
    dec: float,
    roll_deg: float,
    obs_time: Time,
) -> float:
    """Solar-panel power fraction for a given target, roll, and time.

    The fraction is computed using Lambert's cosine law applied to the
    angle between the Sun direction and the solar-panel normal (the
    spacecraft +Y axis).  The convention matches
    ``pandoravisibility``'s ``get_visibility_best_roll`` output.

    Parameters
    ----------
    ra, dec : float
        Target RA / Dec in degrees.
    roll_deg : float
        Spacecraft roll angle about boresight in degrees.
    obs_time : Time
        Observation time (scalar).

    Returns
    -------
    float
        Power fraction in [0, 1].  1.0 = optimal illumination.
    """
    _, y_payload, _ = _payload_axes_from_roll(ra, dec, roll_deg)

    sun_coord = get_sun(obs_time)
    sun_vec = radec_to_vector(sun_coord.ra.deg, sun_coord.dec.deg)

    cos_sy = np.clip(np.dot(y_payload, sun_vec), -1.0, 1.0)
    theta_sy = np.arccos(np.abs(cos_sy))
    incidence = np.pi / 2 - theta_sy
    return float(np.cos(incidence))


def compute_mean_solar_power(
    ra: float,
    dec: float,
    roll_deg: float,
    times: Time,
) -> float:
    """Mean solar-panel power fraction over an array of times.

    Parameters
    ----------
    ra, dec : float
        Target RA / Dec in degrees.
    roll_deg : float
        Spacecraft roll angle about boresight in degrees.
    times : Time
        Array of observation times.

    Returns
    -------
    float
        Mean power fraction in [0, 1].
    """
    if times.isscalar:
        return compute_solar_power_fraction(ra, dec, roll_deg, times)

    _, y_payload, _ = _payload_axes_from_roll(ra, dec, roll_deg)

    total = 0.0
    for t in times:
        sun_coord = get_sun(t)
        sun_vec = radec_to_vector(sun_coord.ra.deg, sun_coord.dec.deg)
        cos_sy = np.clip(np.dot(y_payload, sun_vec), -1.0, 1.0)
        theta_sy = np.arccos(np.abs(cos_sy))
        incidence = np.pi / 2 - theta_sy
        total += np.cos(incidence)
    return float(total / len(times))


# ------------------------------------------------------------------
# Roll sweep / optimisation
# ------------------------------------------------------------------


def find_best_roll_for_target(
    visibility: Any,
    ra: float,
    dec: float,
    times: Time,
    roll_step: float = 2.0,
    min_power_frac: float = 0.8,
    sun_roll: Optional[float] = None,
) -> Optional[float]:
    """Find the roll angle that maximises visible time for a target.

    Sweeps candidate roll angles from 0° to 360° in *roll_step*
    increments and, for each candidate that meets the minimum solar
    power threshold, evaluates visibility via
    ``visibility.get_visibility(coord, times, roll=...)``.

    Parameters
    ----------
    visibility : pandoravisibility.Visibility
        Visibility instance (must support a ``roll`` keyword in
        ``get_visibility``).
    ra, dec : float
        Target RA / Dec in degrees.
    times : Time
        Minute-resolution observation times for this target.
    roll_step : float, optional
        Sweep resolution in degrees (default 2.0).
    min_power_frac : float, optional
        Minimum acceptable mean solar-panel power fraction.
        Candidates below this threshold are rejected (default 0.8).
    sun_roll : float or None, optional
        Sun-derived roll angle (from ``calculate_roll``) used as a
        tie-breaker when multiple candidates yield equal visible
        minutes.  If *None*, the first candidate wins ties.

    Returns
    -------
    float or None
        Best roll angle in degrees (normalised to [-180, 180]), or
        *None* if no candidate meets the power threshold *and* has at
        least one visible minute.
    """
    target_coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
    candidates: List[float] = list(np.arange(0, 360, roll_step))

    best_roll: Optional[float] = None
    best_vis_count: int = 0
    best_sun_dist: float = 360.0  # tiebreaker distance

    for cand in candidates:
        # --- power threshold ---
        power = compute_mean_solar_power(ra, dec, cand, times)
        if power < min_power_frac:
            continue

        # --- visibility ---
        vis = visibility.get_visibility(target_coord, times, roll=cand * u.deg)
        vis_count = int(np.sum(vis))

        if vis_count == 0:
            continue

        # --- tiebreaker: closeness to sun-derived roll ---
        if sun_roll is not None:
            sun_dist = abs(((cand - sun_roll + 180.0) % 360.0) - 180.0)
        else:
            sun_dist = 0.0

        if (vis_count > best_vis_count) or (
            vis_count == best_vis_count and sun_dist < best_sun_dist
        ):
            best_vis_count = vis_count
            best_roll = cand
            best_sun_dist = sun_dist

    if best_roll is not None:
        best_roll = (best_roll + 180.0) % 360.0 - 180.0
    return best_roll


def find_best_rolls_for_visit(
    visibility: Any,
    visit: Any,
    roll_step: float = 2.0,
    min_power_frac: float = 0.8,
) -> Dict[str, Optional[float]]:
    """Find the best roll angle for every target in a visit.

    For each unique target, gathers all minute-resolution times across
    its observation sequences and calls
    :func:`find_best_roll_for_target`.

    Parameters
    ----------
    visibility : pandoravisibility.Visibility
        Visibility instance.
    visit : Visit
        The visit to analyse.
    roll_step : float, optional
        Sweep resolution in degrees (default 2.0).
    min_power_frac : float, optional
        Minimum acceptable mean solar-panel power fraction.

    Returns
    -------
    dict
        Mapping of target name → optimal roll in degrees, or *None*
        when no valid roll was found for that target.
    """
    # Group sequences by target
    target_sequences: Dict[str, list] = {}
    for seq in visit.sequences:
        target_sequences.setdefault(seq.target, []).append(seq)

    result: Dict[str, Optional[float]] = {}
    for target, sequences in target_sequences.items():
        sequences_sorted = sorted(sequences, key=lambda s: s.start_time)
        ra = sequences_sorted[0].ra
        dec = sequences_sorted[0].dec

        # Build a combined minute-resolution time array
        all_times: List[Time] = []
        for seq in sequences_sorted:
            n_mins = max(1, int(np.rint(seq.duration.sec / 60.0)))
            deltas = np.arange(n_mins) * u.min
            all_times.extend([seq.start_time + dt for dt in deltas])
        if not all_times:
            result[target] = None
            continue
        combined_times = Time(all_times)

        # Sun-derived roll as tiebreaker reference
        sun_roll = calculate_roll(ra, dec, sequences_sorted[0].start_time)

        result[target] = find_best_roll_for_target(
            visibility,
            ra,
            dec,
            combined_times,
            roll_step=roll_step,
            min_power_frac=min_power_frac,
            sun_roll=sun_roll,
        )
    return result


def calculate_visit_rolls(
    visit,
    reference_time: Optional[Time] = None,
) -> Dict[str, float]:
    """Calculate roll angles for all unique targets in a visit.

    This function ensures that the same roll angle is used for all
    observation sequences of the same target within a visit, as
    required by spacecraft pointing constraints.

    Parameters
    ----------
    visit : Visit
        The visit containing observation sequences.
    reference_time : Time, optional
        Reference time to use for roll calculation. If None, uses the
        start time of the first sequence for each target.

    Returns
    -------
    dict
        Mapping of target name to roll angle in degrees.
        Example: {"WASP-18": 45.2, "K2-18": 15.1}

    Notes
    -----

    """
    target_rolls: Dict[str, float] = {}

    # Group sequences by target
    target_sequences: Dict[str, list] = {}
    for seq in visit.sequences:
        if seq.target not in target_sequences:
            target_sequences[seq.target] = []
        target_sequences[seq.target].append(seq)

    # Calculate roll for each unique target
    for target, sequences in target_sequences.items():
        # Sort sequences by start time to use the earliest one
        sequences_sorted = sorted(sequences, key=lambda s: s.start_time)

        # Use provided reference time or earliest sequence start time
        if reference_time is not None:
            calc_time = reference_time
        else:
            calc_time = sequences_sorted[0].start_time

        # Get coordinates from first sequence (should be same for all)
        ra = sequences_sorted[0].ra
        dec = sequences_sorted[0].dec

        # Calculate roll
        roll = calculate_roll(ra, dec, calc_time)
        target_rolls[target] = roll

    return target_rolls


def apply_rolls_to_visit(
    visit,
    target_rolls: Optional[Dict[str, float]] = None,
    precomputed_rolls: Optional[Dict[str, float]] = None,
) -> None:
    """Apply roll angles to all observation sequences in a visit.

    This function modifies the visit's sequences in place, setting
    the roll attribute on each ObservationSequence.

    Parameters
    ----------
    visit : Visit
        The visit to update. Modified in place.
    target_rolls : dict, optional
        Pre-calculated mapping of target name to roll angle.
        If None, rolls will be calculated using calculate_visit_rolls().
    precomputed_rolls : dict, optional
        Visibility-aware mapping of target name to roll angle.
        Entries in this dict take precedence over *target_rolls*.
        Targets whose value is *None* fall through to
        *target_rolls* / ``calculate_roll``.

    Returns
    -------
    None
        The visit is modified in place.
    """
    if target_rolls is None:
        target_rolls = calculate_visit_rolls(visit)

    # Merge: precomputed_rolls overrides target_rolls when present
    merged: Dict[str, float] = dict(target_rolls)
    if precomputed_rolls is not None:
        for tgt, roll_val in precomputed_rolls.items():
            if roll_val is not None:
                merged[tgt] = roll_val

    for seq in visit.sequences:
        if seq.target in merged:
            seq.roll = merged[seq.target]
        else:
            # Target not found - calculate individually
            seq.roll = calculate_roll(seq.ra, seq.dec, seq.start_time)


def apply_rolls_to_calendar(
    calendar,
    verbose: bool = False,
    precomputed_rolls: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Apply roll angles to all visits in a science calendar.

    This is a convenience function that applies roll angles to all
    visits in a calendar, ensuring consistency within each visit.

    Parameters
    ----------
    calendar : ScienceCalendar
        The calendar to update. Modified in place.
    verbose : bool, optional
        If True, print progress information.
    precomputed_rolls : dict, optional
        Nested mapping ``{visit_id: {target: roll_deg}}`` of
        visibility-aware roll angles.  These override the
        sun-derived rolls for any target where a valid value was
        found.  Targets with *None* values fall through to the
        sun-derived calculation.

    Returns
    -------
    None
        The calendar is modified in place.
    """
    for visit in calendar.visits:
        if verbose:
            print(f"Calculating rolls for visit {visit.id}")

        target_rolls = calculate_visit_rolls(visit)
        visit_precomputed = None
        if precomputed_rolls is not None:
            visit_precomputed = precomputed_rolls.get(visit.id)
        apply_rolls_to_visit(
            visit,
            target_rolls,
            precomputed_rolls=visit_precomputed,
        )

        if verbose:
            for target, roll in target_rolls.items():
                final = (
                    visit_precomputed.get(target)
                    if visit_precomputed
                    else None
                )
                display = final if final is not None else roll
                print(f"  {target}: {display:.2f} deg")
