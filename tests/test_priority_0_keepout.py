"""A stricter Earth-limb keepout for priority-0 observations only.

Priority-0 targets may need to be held further off the Earth so the
spacecraft can dissipate more heat. ``priority_0_earthlimb_min`` is a
second, stricter visibility model rather than a tweak to the first: a
flat limb angle with the day/night split and the dynamic DPC wedge both
switched off, leaving every other keepout alone.
"""

# Third-party
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pandoravisibility import Visibility

# First-party/Local
from shortschedule.models import ObservationSequence, ScienceCalendar, Visit
from shortschedule.scheduler import ScheduleProcessor

TLE1 = "1 67395U 80229J   26196.69732639  .00000000  00000-0  37770-3 0    00"
TLE2 = "2 67395  97.8056 194.9117 0006480  50.2285  39.6294 14.88117629    09"

T0 = Time("2026-07-16T00:00:00", scale="utc")

# Operationally-shaped configuration: a dynamic wedge and a day/night
# split on the nominal model, so a test can tell whether the priority-0
# model really dropped both.
NOMINAL = dict(
    earthlimb_day_min=44,
    earthlimb_night_min=13,
    sun_min=91,
    moon_min=25,
    st_sun_min=50,
    st_moon_min=20,
    st_earthlimb_min=30,
    use_dynamic_earthlimb=True,
)


# The two observations sit at different right ascensions so a recorded
# visibility lookup can be attributed to one of them.
PRIORITY_0_RA = 15.0
PRIORITY_1_RA = 16.0


def _make_seq(sid, priority, start_min=0, duration_min=45, ra=None, dec=-20.0):
    if ra is None:
        ra = PRIORITY_0_RA if priority == 0 else PRIORITY_1_RA
    start = T0 + start_min * u.min
    return ObservationSequence(
        id=sid,
        target=f"TARGET_P{priority}",
        priority=priority,
        start_time=start,
        stop_time=start + duration_min * u.min,
        ra=ra,
        dec=dec,
        payload_params={},
    )


class TestModelConstruction:
    """What the second Visibility is built with."""

    def test_absent_by_default(self):
        scheduler = ScheduleProcessor(TLE1, TLE2, **NOMINAL)

        assert scheduler.priority_0_visibility is None
        assert scheduler.priority_0_earthlimb_min is None

    def test_built_when_a_number_is_given(self):
        scheduler = ScheduleProcessor(
            TLE1, TLE2, priority_0_earthlimb_min=54, **NOMINAL
        )

        assert scheduler.priority_0_visibility is not None
        assert scheduler.priority_0_visibility.earthlimb_min.to_value(
            u.deg
        ) == pytest.approx(54.0)

    def test_the_limb_angle_is_flat(self):
        """No day/night split and no dynamic wedge, whatever the rest says.

        The nominal model here has both. If either leaked through, the
        angle actually applied would vary with illumination instead of
        being the number the caller gave.
        """
        scheduler = ScheduleProcessor(
            TLE1, TLE2, priority_0_earthlimb_min=54, **NOMINAL
        )
        strict = scheduler.priority_0_visibility

        assert strict.use_dynamic_earthlimb is False
        assert scheduler.visibility.use_dynamic_earthlimb is True
        for attribute in ("earthlimb_day_min", "earthlimb_night_min"):
            nominal = getattr(scheduler.visibility, attribute)
            assert nominal is not None
            assert getattr(strict, attribute) in (None, strict.earthlimb_min)

    def test_every_other_keepout_is_untouched(self):
        """Only the boresight Earth limb differs between the two models."""
        scheduler = ScheduleProcessor(
            TLE1, TLE2, priority_0_earthlimb_min=54, **NOMINAL
        )

        for attribute in (
            "sun_min",
            "moon_min",
            "st_sun_min",
            "st_moon_min",
            "st_earthlimb_min",
            "mars_min",
            "jupiter_min",
            "st_required",
        ):
            assert getattr(scheduler.priority_0_visibility, attribute) == (
                getattr(scheduler.visibility, attribute)
            ), attribute

    def test_it_is_stricter_in_practice(self):
        """The stricter model must actually reject more minutes.

        Construction assertions alone would pass even if the angle were
        wired to something the library ignores.
        """
        scheduler = ScheduleProcessor(
            TLE1, TLE2, priority_0_earthlimb_min=70, **NOMINAL
        )
        coord = SkyCoord(15.0, -20.0, frame="icrs", unit="deg")
        times = T0 + np.arange(200) * u.min

        nominal = np.asarray(scheduler.visibility.get_visibility(coord, times))
        strict = np.asarray(
            scheduler.priority_0_visibility.get_visibility(coord, times)
        )

        assert strict.sum() < nominal.sum()
        # Stricter means a subset: nothing may become visible.
        assert not (strict & ~nominal).any()


class TestModelSelection:
    """Which model each observation is judged by."""

    def test_priority_0_gets_the_strict_model(self):
        scheduler = ScheduleProcessor(
            TLE1, TLE2, priority_0_earthlimb_min=54, **NOMINAL
        )

        assert (
            scheduler._visibility_for_priority(0)
            is scheduler.priority_0_visibility
        )

    @pytest.mark.parametrize("priority", [1, 2, 3, None])
    def test_everything_else_gets_the_nominal_model(self, priority):
        scheduler = ScheduleProcessor(
            TLE1, TLE2, priority_0_earthlimb_min=54, **NOMINAL
        )

        assert (
            scheduler._visibility_for_priority(priority)
            is scheduler.visibility
        )

    @pytest.mark.parametrize("priority", [0, 1, 2])
    def test_unconfigured_means_one_model_for_everyone(self, priority):
        """With the parameter unset, priority 0 is judged like anything else."""
        scheduler = ScheduleProcessor(TLE1, TLE2, **NOMINAL)

        assert (
            scheduler._visibility_for_priority(priority)
            is scheduler.visibility
        )


class TestEveryPassAgrees:
    """No pass may judge an observation by a model another pass did not."""

    def _calendar(self):
        return ScienceCalendar(
            metadata={},
            visits=[
                Visit(
                    id="v1",
                    sequences=[
                        _make_seq("s0", 0, start_min=60, duration_min=50),
                        _make_seq("s1", 1, start_min=180, duration_min=50),
                    ],
                )
            ],
        )

    def test_no_pass_judges_priority_0_by_the_nominal_model(self):
        """Record which model each pass asked for, per priority.

        This is the check that catches a pass left reading
        ``self.visibility`` directly: such a pass would never appear in
        the recorded calls, and the priority-0 observation would be
        judged leniently without anything else noticing.
        """
        scheduler = ScheduleProcessor(
            TLE1, TLE2, priority_0_earthlimb_min=54, **NOMINAL
        )
        scheduler._print = lambda message: None
        asked = []

        class _Recorder:
            """Forwards to a real model, noting which target it was asked about."""

            def __init__(self, name, inner):
                self._name = name
                self._inner = inner

            def __getattr__(self, attribute):
                inner = getattr(self._inner, attribute)
                if not callable(inner):
                    return inner

                def _call(*args, **kwargs):
                    for arg in args:
                        if isinstance(arg, SkyCoord):
                            asked.append((self._name, round(arg.ra.deg, 4)))
                            break
                    return inner(*args, **kwargs)

                return _call

        scheduler.visibility = _Recorder("nominal", scheduler.visibility)
        scheduler.priority_0_visibility = _Recorder(
            "strict", scheduler.priority_0_visibility
        )

        calendar = self._calendar()
        calendar = scheduler._trim_non_visible_tails(calendar)
        calendar = scheduler._trim_to_longest_visible_block(calendar)
        calendar = scheduler._enforce_start_buffers(calendar)
        scheduler.validate_visibility(calendar, report_issues=False)

        assert asked, "no pass consulted a visibility model"
        # The two observations sit at different RA, so a lookup can be
        # attributed to one of them. A pass that still reads
        # self.visibility directly shows up here as the nominal model
        # being asked about the priority-0 target.
        assert ("nominal", PRIORITY_0_RA) not in asked
        assert ("strict", PRIORITY_0_RA) in asked
        assert ("nominal", PRIORITY_1_RA) in asked
        assert ("strict", PRIORITY_1_RA) not in asked

    def test_delivered_priority_0_respects_the_strict_limb(self):
        """End to end: what survives must satisfy the model it was judged by.

        The window is chosen so the strict model leaves a run long enough
        to survive ``min_sequence_duration`` (24 visible minutes nominal,
        18 strict). An observation the strict model rejects outright has
        nothing to trim to and is correctly left alone and reported, so it
        would not exercise this at all.
        """
        scheduler = ScheduleProcessor(
            TLE1,
            TLE2,
            priority_0_earthlimb_min=30,
            earthlimb_gap_tolerance=0,
            st_gap_tolerance=0,
            st_gap_tolerance_start_buffer=0,
            earthlimb_gap_tolerance_start_buffer=0,
            **NOMINAL,
        )
        scheduler._print = lambda message: None
        # These passes return a new calendar rather than mutating the
        # one given, so the result has to be carried forward.
        calendar = scheduler._trim_non_visible_tails(self._calendar())
        calendar = scheduler._trim_to_longest_visible_block(calendar)

        for seq in calendar.visits[0].sequences:
            n_mins = int(np.rint(seq.duration.sec / 60.0))
            if n_mins <= 0:
                continue
            model = scheduler._visibility_for_priority(seq.priority)
            visible = np.atleast_1d(
                np.asarray(
                    model.get_visibility(
                        SkyCoord(seq.ra, seq.dec, frame="icrs", unit="deg"),
                        seq.start_time + np.arange(n_mins) * u.min,
                    )
                )
            )
            assert visible.all(), (
                f"{seq.id} (priority {seq.priority}) keeps "
                f"{int((~visible).sum())} minutes its own model rejects"
            )


class TestRollSweep:
    """A priority-0 target is scored under the keepouts it will fly."""

    def test_all_priority_0_target_uses_the_strict_model(self):
        from shortschedule.roll import find_best_rolls_for_visit

        nominal = Visibility(TLE1, TLE2)
        strict = Visibility(TLE1, TLE2, earthlimb_min=70 * u.deg)
        used = []

        class _Recorder:
            def __init__(self, name, inner):
                self.name = name
                self._inner = inner

            def __getattr__(self, attribute):
                used.append(self.name)
                return getattr(self._inner, attribute)

        visit = Visit(
            id="v1",
            sequences=[_make_seq("s0", 0), _make_seq("s0b", 0, start_min=60)],
        )
        find_best_rolls_for_visit(
            _Recorder("nominal", nominal),
            visit,
            roll_step=90.0,
            priority_0_visibility=_Recorder("strict", strict),
        )

        assert "strict" in used
        assert "nominal" not in used

    def test_mixed_priority_target_keeps_the_nominal_model(self):
        """One roll serves the whole visit, so the strictest must not win.

        Scoring a mixed-priority target against the priority-0 model
        would over-constrain its priority-1 observations, which are not
        subject to the stricter limb at all.
        """
        from shortschedule.roll import find_best_rolls_for_visit

        nominal = Visibility(TLE1, TLE2)
        strict = Visibility(TLE1, TLE2, earthlimb_min=70 * u.deg)
        used = []

        class _Recorder:
            def __init__(self, name, inner):
                self.name = name
                self._inner = inner

            def __getattr__(self, attribute):
                used.append(self.name)
                return getattr(self._inner, attribute)

        mixed = _make_seq("s1", 1, start_min=60)
        mixed.target = "TARGET_P0"  # same target, different priority
        visit = Visit(id="v1", sequences=[_make_seq("s0", 0), mixed])
        find_best_rolls_for_visit(
            _Recorder("nominal", nominal),
            visit,
            roll_step=90.0,
            priority_0_visibility=_Recorder("strict", strict),
        )

        assert "nominal" in used
        assert "strict" not in used
