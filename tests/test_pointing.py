"""Tests for the per-minute pointing reconstruction and its plots.

The geometry here is the part with no second opinion: the scheduler
never computes an idle attitude, so nothing else in the package would
notice if the frame construction silently went wrong. These tests pin it
against quantities that are fixed by the command definition rather than
by this implementation:

- the dark-idle boresight sits at a constant, known angle to nadir,
  which follows from the pitch offset and the -Z command axis alone
- a star tracker is bolted to the bus, so its angle to the boresight is
  the same constant during an observation and during idle
- the Sun/Moon/Earth angles agree with pandoravisibility's own

They use the real ``Visibility`` for the same reason
``test_keepout_plumb_through.py`` does: a mock cannot catch a divergence
from the library.
"""

# Third-party
import matplotlib
import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time
from pandoravisibility import Visibility

# First-party/Local
from shortschedule.models import ObservationSequence, ScienceCalendar, Visit
from shortschedule.pointing import (
    AXES,
    BODIES,
    DARK_IDLE_EULER_DEG,
    IDLE_LABEL,
    _euler_zyx_matrix,
    build_pointing_timeline,
)
from shortschedule.visualizer import ScheduleVisualizer

matplotlib.use("Agg")

TLE1 = "1 67395U 80229J   26196.69732639  .00000000  00000-0  37770-3 0    00"
TLE2 = "2 67395  97.8056 194.9117 0006480  50.2285  39.6294 14.88117629    09"

T0 = Time("2026-07-16T00:00:00", scale="utc")


def _make_calendar():
    """Two targets with a deliberate 30 min idle gap between them."""
    sequences = [
        ObservationSequence(
            id="s1",
            target="TARGET_A",
            priority=1,
            start_time=T0,
            stop_time=T0 + 40 * u.min,
            ra=15.0,
            dec=-20.0,
            roll=30.0,
            payload_params={},
        ),
        ObservationSequence(
            id="s2",
            target="TARGET_B",
            priority=2,
            start_time=T0 + 70 * u.min,
            stop_time=T0 + 110 * u.min,
            ra=250.0,
            dec=45.0,
            roll=-60.0,
            payload_params={},
        ),
    ]
    return ScienceCalendar(
        metadata={}, visits=[Visit(id="v1", sequences=sequences)]
    )


def _visibility():
    return Visibility(
        TLE1,
        TLE2,
        sun_min=91 * u.deg,
        moon_min=25 * u.deg,
        st_sun_min=50 * u.deg,
        st_moon_min=20 * u.deg,
        st_earthlimb_min=30 * u.deg,
    )


@pytest.fixture(scope="module")
def timeline():
    return build_pointing_timeline(_make_calendar(), _visibility())


# ================================================================
# Tests: the minute grid and labelling
# ================================================================


class TestTimelineLayout:
    """Every minute of the span is labelled, once."""

    def test_grid_spans_first_start_to_last_stop(self, timeline):
        assert len(timeline.times) == 110
        assert timeline.times[0].isot == T0.isot

    def test_observed_and_idle_minutes_are_labelled(self, timeline):
        assert list(timeline.labels[:40]) == ["TARGET_A"] * 40
        assert list(timeline.labels[40:70]) == [IDLE_LABEL] * 30
        assert list(timeline.labels[70:]) == ["TARGET_B"] * 40

    def test_targets_exclude_idle(self, timeline):
        assert sorted(timeline.targets) == ["TARGET_A", "TARGET_B"]

    def test_segments_reconstruct_the_labels(self, timeline):
        rebuilt = []
        for start, stop, label in timeline.segments:
            rebuilt.extend([label] * (stop - start))
        assert rebuilt == list(timeline.labels)

    def test_observed_fraction(self, timeline):
        assert timeline.observed_fraction == pytest.approx(80 / 110)

    def test_every_series_is_populated(self, timeline):
        for axis in AXES:
            for body in BODIES:
                values = timeline.angles[(axis, body)]
                assert len(values) == len(timeline.times)
                assert not np.isnan(values).any()
            assert not np.isnan(timeline.illumination[axis]).any()

    def test_angles_are_physical(self, timeline):
        for axis in AXES:
            for body in BODIES:
                values = timeline.angles[(axis, body)]
                assert values.min() >= 0.0
                assert values.max() <= 180.0
            illumination = timeline.illumination[axis]
            assert illumination.min() >= 0.0
            assert illumination.max() <= 180.0

    def test_calendar_with_no_observations_is_rejected(self):
        empty = ScienceCalendar(metadata={}, visits=[])
        with pytest.raises(ValueError, match="no observations"):
            build_pointing_timeline(empty, _visibility())


# ================================================================
# Tests: attitude geometry
# ================================================================


class TestIdleAttitude:
    """The dark-idle attitude is fixed relative to nadir and the Sun."""

    def test_boresight_holds_a_constant_angle_to_nadir(self, timeline):
        """135 deg, and constant.

        The command puts nadir on REFERENCE +Z, pitches 45 deg off it,
        then maps body -Z onto that. The boresight is body +Z, so it
        ends up 180 - 45 deg from nadir at every point in the orbit.
        Nothing about the target or the epoch can change this, which is
        what makes it a usable check on the frame construction.
        """
        idle = timeline.labels == IDLE_LABEL
        earth_angle = timeline.angles[("Boresight", "earth")][idle]

        assert earth_angle == pytest.approx(135.0, abs=1e-6)

    def test_the_euler_offset_is_what_sets_that_angle(self):
        """Change the pitch, and the idle boresight moves with it."""
        timeline = build_pointing_timeline(
            _make_calendar(), _visibility(), idle_euler_deg=(0.0, -20.0, 180.0)
        )
        idle = timeline.labels == IDLE_LABEL

        assert timeline.angles[("Boresight", "earth")][idle] == pytest.approx(
            160.0, abs=1e-6
        )

    def test_idle_default_matches_the_flown_command(self):
        assert DARK_IDLE_EULER_DEG == (0.0, -45.0, 180.0)


class TestStarTrackerGeometry:
    """The trackers are bolted to the bus, in both regimes."""

    @pytest.mark.parametrize("tracker", ["ST1", "ST2"])
    def test_tracker_to_boresight_angle_is_rigid(self, timeline, tracker):
        """A tracker cannot move relative to the boresight.

        Its body-frame vector fixes the angle, so observation minutes
        and idle minutes must both show it. If the idle frame math or
        the roll handling were wrong, this would drift.
        """
        from shortschedule.pointing import _unit_from_radec

        expected = np.degrees(
            np.arccos(np.array(Visibility._get_star_tracker_body_xyz(1))[2])
        )
        boresight = _unit_from_radec(
            timeline.ra["Boresight"], timeline.dec["Boresight"]
        )
        tracker_unit = _unit_from_radec(
            timeline.ra[tracker], timeline.dec[tracker]
        )
        separation = np.degrees(
            np.arccos(
                np.clip(np.sum(boresight * tracker_unit, axis=0), -1.0, 1.0)
            )
        )

        assert separation == pytest.approx(expected, abs=0.01)

    def test_boresight_follows_the_target_during_an_observation(
        self, timeline
    ):
        """Within aberration, the boresight is the target direction."""
        from astropy.coordinates import SkyCoord

        target = SkyCoord(250.0, 45.0, frame="icrs", unit="deg")
        pointed = SkyCoord(
            timeline.ra["Boresight"][80],
            timeline.dec["Boresight"][80],
            unit="deg",
        )

        assert target.separation(pointed).arcsec < 60.0


class TestAgainstTheLibrary:
    """Angles must match what pandoravisibility computes itself."""

    def test_star_tracker_angles_match(self, timeline):
        from astropy.coordinates import SkyCoord

        visibility = _visibility()
        index = np.arange(70, 110)
        times = timeline.times[index]
        coord = SkyCoord(250.0, 45.0, frame="icrs", unit="deg")

        visibility.roll = -60.0 * u.deg
        for tracker, axis in ((1, "ST1"), (2, "ST2")):
            library = visibility.get_star_tracker_angles(coord, times, tracker)
            for body in BODIES:
                assert timeline.angles[(axis, body)][index] == pytest.approx(
                    library[f"{body}_angle"].to_value(u.deg), abs=1e-6
                )

    def test_boresight_angles_match(self, timeline):
        from astropy.coordinates import SkyCoord

        visibility = _visibility()
        index = np.arange(0, 40)
        separations = visibility.get_separations(
            SkyCoord(15.0, -20.0, frame="icrs", unit="deg"),
            timeline.times[index],
        )

        for body in ("sun", "moon"):
            assert timeline.angles[("Boresight", body)][
                index
            ] == pytest.approx(separations[body].to_value(u.deg), abs=1e-6)


class TestEulerMatrix:
    """The ATT_INTERP 0 offset is Rz(yaw) Ry(pitch) Rx(roll)."""

    def test_zero_offset_is_the_identity(self):
        assert _euler_zyx_matrix(0.0, 0.0, 0.0) == pytest.approx(np.eye(3))

    def test_yaw_rotates_about_z(self):
        rotated = _euler_zyx_matrix(0.0, 0.0, 90.0) @ np.array([1.0, 0.0, 0.0])

        assert rotated == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)

    def test_pitch_rotates_about_y(self):
        rotated = _euler_zyx_matrix(0.0, 90.0, 0.0) @ np.array([0.0, 0.0, 1.0])

        assert rotated == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)


# ================================================================
# Tests: side effects and failure handling
# ================================================================


class TestSideEffects:
    """Building a timeline must not disturb the caller's Visibility."""

    @pytest.mark.parametrize("roll", [None, 12.0 * u.deg])
    def test_instance_roll_is_restored(self, roll):
        """The roll is set per pointing group and must be put back.

        The scheduler shares one Visibility instance across the whole
        run, so leaving a roll behind would silently change every later
        visibility call.
        """
        visibility = _visibility()
        visibility.roll = roll

        build_pointing_timeline(_make_calendar(), visibility)

        assert visibility.roll == roll

    def test_unresolvable_tracker_is_recorded_not_raised(self):
        """A tracker that cannot be evaluated leaves NaN and a note.

        Reporting beats guessing: the plot shows a hole, and the caller
        is told which group it belongs to.
        """
        visibility = _visibility()

        def _fail(*args, **kwargs):
            raise ValueError("degenerate attitude")

        visibility.get_star_tracker_angles = _fail
        timeline = build_pointing_timeline(_make_calendar(), visibility)

        assert len(timeline.unresolved) == 4  # two groups, two trackers
        assert "degenerate attitude" in timeline.unresolved[0]
        observed = timeline.labels != IDLE_LABEL
        assert np.isnan(timeline.angles[("ST1", "sun")][observed]).all()
        # The boresight and the idle minutes are unaffected.
        assert not np.isnan(timeline.angles[("Boresight", "sun")]).any()
        assert not np.isnan(timeline.angles[("ST1", "sun")][~observed]).any()


# ================================================================
# Tests: the plots
# ================================================================


class _StubScheduler:
    """Carries only what the pointing plots read off a scheduler."""

    def __init__(self, visibility):
        self.visibility = visibility

    def get_gap_report(self):
        return {}


@pytest.fixture
def visualizer():
    return ScheduleVisualizer(_StubScheduler(_visibility()))


class TestPointingPlots:
    """Each entry point renders, and they share one timeline."""

    @pytest.mark.parametrize(
        "method",
        [
            "plot_pointing_timeline",
            "plot_keepout_angles",
            "plot_earth_illumination",
        ],
    )
    def test_plot_renders(self, visualizer, method):
        figure = getattr(visualizer, method)(_make_calendar())

        assert figure is not None
        assert len(figure.axes) > 0

    def test_timeline_is_built_once_per_calendar(self, visualizer):
        """The timeline is the expensive part; three plots, one build."""
        calendar = _make_calendar()

        first = visualizer.get_pointing_timeline(calendar)
        visualizer.plot_keepout_angles(calendar)
        second = visualizer.get_pointing_timeline(calendar)

        assert first is second

    def test_changing_the_idle_attitude_rebuilds(self, visualizer):
        calendar = _make_calendar()

        first = visualizer.get_pointing_timeline(calendar)
        second = visualizer.get_pointing_timeline(
            calendar, idle_euler_deg=(0.0, 10.0, 0.0)
        )

        assert first is not second

    def test_idle_never_takes_a_target_color(self, visualizer):
        """Black is reserved, so no target may be given a grey either."""
        colors = visualizer._get_pointing_colors(
            [f"TARGET_{index}" for index in range(30)]
        )

        assert len(set(colors.values())) == 30
        for color in colors.values():
            assert max(color[:3]) - min(color[:3]) > 0.08

    def test_colors_are_stable_across_calls(self, visualizer):
        first = visualizer._get_pointing_colors(["B", "A"])
        second = visualizer._get_pointing_colors(["A", "B"])

        assert first == second

    def test_a_line_only_ever_draws_its_own_steps(self, visualizer):
        """No line may include a step belonging to another label."""
        import matplotlib.pyplot as plt

        from shortschedule.pointing import BODIES

        calendar = _make_calendar()
        timeline = visualizer.get_pointing_timeline(calendar)
        colors = visualizer._get_pointing_colors(timeline.targets)
        figure, ax = plt.subplots()
        visualizer._draw_pointing_series(
            ax,
            timeline,
            timeline.angles[("Boresight", BODIES[0])],
            colors,
            1.0,
        )

        drawn = sorted(len(line.get_xdata()) for line in ax.lines)
        expected = sorted(stop - start for start, stop, _ in timeline.segments)

        assert drawn == expected
        assert sum(drawn) == len(timeline.times)


def test_illumination_follows_the_model_reference_point():
    """The illumination angle is taken where the model measures it.

    Sub-satellite (the library default) is target independent, so every
    axis reads the same value; the grazed-limb mode differs per axis.
    """
    calendar = _make_calendar()
    subsatellite = build_pointing_timeline(calendar, _visibility())
    np.testing.assert_array_equal(
        subsatellite.illumination["Boresight"],
        subsatellite.illumination["ST1"],
    )
    assert 64.0 < subsatellite.earth_radius_deg < 68.0

    limb = build_pointing_timeline(
        calendar, Visibility(TLE1, TLE2, daynight_mode="limb")
    )
    assert not np.allclose(
        limb.illumination["Boresight"], limb.illumination["ST1"]
    )


def test_visibility_gantt_paints_rolls_only_on_request():
    """Off by default the bar is all priority and there is no colorbar."""
    from shortschedule.scheduler import ScheduleProcessor

    visualizer = ScheduleVisualizer(ScheduleProcessor(TLE1, TLE2))
    calendar = _make_calendar()

    plain = visualizer.plot_gantt_with_visibility(calendar)
    assert len(plain.axes) == 1

    with_rolls = visualizer.plot_gantt_with_visibility(
        calendar, plot_rolls=True
    )
    assert len(with_rolls.axes) == 2
