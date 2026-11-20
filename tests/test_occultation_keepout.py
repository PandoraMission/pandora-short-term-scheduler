"""A stricter Earth-limb keepout for every target but each visit's main one.

The spacecraft may need to be held further off the Earth so it can
dissipate more heat. ``occultation_earthlimb_min`` is a second, stricter
visibility model rather than a tweak to the first: a flat limb angle with
the day/night split and the dynamic DPC wedge both switched off, leaving
every other keepout alone. Each visit's main target keeps the standard
keepouts whatever its priority; everything else flies the stricter one.
"""

# Standard library
import xml.etree.ElementTree as ET

# Third-party
import matplotlib
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pandoravisibility import Visibility

# First-party/Local
from shortschedule.models import ObservationSequence, ScienceCalendar, Visit
from shortschedule.roll import get_best_roll_per_visit
from shortschedule.scheduler import ScheduleProcessor
from shortschedule.visualizer import ScheduleVisualizer

matplotlib.use("Agg")

TLE1 = "1 67395U 80229J   26196.69732639  .00000000  00000-0  37770-3 0    00"
TLE2 = "2 67395  97.8056 194.9117 0006480  50.2285  39.6294 14.88117629    09"

T0 = Time("2026-07-16T00:00:00", scale="utc")

# Operationally-shaped configuration: a dynamic wedge and a day/night
# split on the nominal model, so a test can tell whether the occultation
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

GAIA = "G2545911550549266944"
OTHER_GAIA = "G2368552456609447168"

# The main and non-main observations sit at different right ascensions so
# a recorded visibility lookup can be attributed to one of them.
MAIN_RA = 16.0
OTHER_RA = 15.0


def _seq(sid, target, priority, start_min=0, duration_min=45, ra=None):
    start = T0 + start_min * u.min
    return ObservationSequence(
        id=sid,
        target=target,
        priority=priority,
        start_time=start,
        stop_time=start + duration_min * u.min,
        ra=OTHER_RA if ra is None else ra,
        dec=-20.0,
        payload_params={},
    )


def _visit(*specs, vid="v1"):
    """``specs`` are (target, priority, minutes), laid end to end."""
    sequences, start = [], 0
    for i, (target, priority, minutes) in enumerate(specs):
        sequences.append(_seq(f"s{i}", target, priority, start, minutes))
        start += minutes
    return Visit(id=vid, sequences=sequences)


def _processor(**kwargs):
    proc = ScheduleProcessor(TLE1, TLE2, **NOMINAL, **kwargs)
    proc.printed = []
    proc._print = lambda message, *a, **kw: proc.printed.append(message)
    return proc


class TestChoosingTheMainTarget:
    def test_the_priority_1_or_2_target_is_main(self):
        proc = _processor()
        visit = _visit(("TOI-5349", 0, 300), ("HD_29037", 1, 60))

        assert proc._mark_main_targets(
            ScienceCalendar(metadata={}, visits=[visit])
        ) == {"v1": ["HD_29037"]}
        # Chosen only; the long-term calendar is not flagged.
        assert not any(s.is_main for s in visit.sequences)

    def test_priority_1_and_2_of_one_target_is_still_one_target(self):
        proc = _processor()
        visit = _visit(("TOI-544b", 1, 30), (GAIA, 0, 90), ("TOI-544b", 2, 30))

        main = proc._mark_main_targets(
            ScienceCalendar(metadata={}, visits=[visit])
        )

        assert main == {"v1": ["TOI-544b"]}
        assert not any(m.startswith("ERROR") for m in proc.printed)

    def test_two_priority_1_or_2_targets_is_an_error_and_no_main(self):
        proc = _processor()
        visit = _visit(("TOI-544b", 1, 30), ("HD_29037", 2, 30))

        main = proc._mark_main_targets(
            ScienceCalendar(metadata={}, visits=[visit])
        )

        assert main == {"v1": []}
        assert not any(s.is_main for s in visit.sequences)
        errors = [m for m in proc.printed if m.startswith("ERROR")]
        assert len(errors) == 1
        assert "TOI-544b" in errors[0] and "HD_29037" in errors[0]

    def test_without_priority_1_or_2_the_most_minutes_is_main(self):
        proc = _processor()
        visit = _visit(
            ("TOI-3714", 0, 40), ("TOI-5349", 0, 50), ("TOI-3714", 0, 20)
        )

        main = proc._mark_main_targets(
            ScienceCalendar(metadata={}, visits=[visit])
        )

        # TOI-3714 has 60 minutes over two observations.
        assert main == {"v1": ["TOI-3714"]}
        assert not any(m.startswith("WARNING") for m in proc.printed)

    def test_a_gaia_target_with_the_most_minutes_is_skipped(self):
        proc = _processor()
        visit = _visit((GAIA, 0, 200), ("TOI-5349", 0, 50), ("HD_1", 0, 30))

        main = proc._mark_main_targets(
            ScienceCalendar(metadata={}, visits=[visit])
        )

        assert main == {"v1": ["TOI-5349"]}
        warnings = [m for m in proc.printed if m.startswith("WARNING")]
        assert len(warnings) == 1
        assert GAIA in warnings[0] and "TOI-5349" in warnings[0]

    def test_only_gaia_targets_means_no_main(self):
        proc = _processor()
        visit = _visit((GAIA, 0, 200), (OTHER_GAIA, 0, 50))

        main = proc._mark_main_targets(
            ScienceCalendar(metadata={}, visits=[visit])
        )

        assert main == {"v1": []}
        assert not any(s.is_main for s in visit.sequences)
        assert any(
            m.startswith("WARNING") and "no main target" in m
            for m in proc.printed
        )

    def test_a_tie_goes_to_the_first_listed(self):
        proc = _processor()
        visit = _visit(("TOI-5349", 0, 50), ("HD_29037", 0, 50))

        assert proc._mark_main_targets(
            ScienceCalendar(metadata={}, visits=[visit])
        ) == {"v1": ["TOI-5349"]}

    def test_each_visit_has_its_own(self):
        proc = _processor()
        cal = ScienceCalendar(
            metadata={},
            visits=[
                _visit(("A", 1, 30), ("B", 0, 90), vid="v1"),
                _visit(("A", 0, 30), ("B", 0, 90), vid="v2"),
            ],
        )

        assert proc._mark_main_targets(cal) == {"v1": ["A"], "v2": ["B"]}
        assert proc.main_targets == {"v1": ["A"], "v2": ["B"]}

    @pytest.mark.parametrize(
        "name, gaia",
        [
            (GAIA, True),
            ("G123456", True),
            ("GJ 1214", False),
            ("G", False),
            ("G12a456789", False),
            ("HD_29037", False),
            ("TOI-5349", False),
        ],
    )
    def test_what_counts_as_gaia(self, name, gaia):
        assert ScheduleProcessor._is_gaia(name) is gaia

    def test_chosen_on_the_long_term_calendar_before_the_window(
        self, monkeypatch
    ):
        """A target mostly outside the window still wins on its minutes,
        and the flag lands on the windowed copies, not the input."""
        proc = _processor()
        visit = _visit(("A", 0, 100), ("B", 0, 40))
        cal = ScienceCalendar(metadata={}, visits=[visit])
        seen = {}

        def _stop(calendar, *args, **kwargs):
            seen.update(
                {s.target: s.is_main for s in calendar.visits[0].sequences}
            )
            raise RuntimeError("stop here")

        monkeypatch.setattr(proc, "_analyze_original_calendar", _stop)
        monkeypatch.setattr(proc, "_setup_run_logging", lambda *a: None)

        with pytest.raises(RuntimeError, match="stop here"):
            proc.process_calendar(cal, window_start=T0 + 90 * u.min)

        # A keeps only 10 of its 100 minutes inside the window.
        assert seen == {"A": True, "B": False}
        assert not any(s.is_main for s in visit.sequences)

    def test_processing_leaves_the_long_term_calendar_unchanged(
        self, monkeypatch
    ):
        """Window cut, flags and name cleaning all act on copies."""
        proc = _processor(fix_bad_data=True)
        visit = _visit(("HD 1+2", 1, 100), ("B", 0, 40))
        section = ET.Element("AcquireVisCamScienceData")
        ET.SubElement(section, "TargetID").text = "HD 1+2"
        visit.sequences[0].payload_params = {
            "AcquireVisCamScienceData": section
        }
        cal = ScienceCalendar(metadata={"keep": 1}, visits=[visit])
        before = [
            (s.target, s.start_time.isot, s.stop_time.isot, s.is_main)
            for s in visit.sequences
        ]
        seen = {}

        def _stop(calendar, *args, **kwargs):
            seen.update({s.target: s for s in calendar.visits[0].sequences})
            raise RuntimeError("stop here")

        monkeypatch.setattr(proc, "_analyze_original_visibility", _stop)
        monkeypatch.setattr(proc, "_setup_run_logging", lambda *a: None)

        with pytest.raises(RuntimeError, match="stop here"):
            proc.process_calendar(cal, window_start=T0 + 90 * u.min)

        # The copies were cut, flagged and renamed...
        assert set(seen) == {"HD_1_2", "B"}
        assert seen["HD_1_2"].is_main
        # ...and the input was not.
        assert [
            (s.target, s.start_time.isot, s.stop_time.isot, s.is_main)
            for s in visit.sequences
        ] == before
        assert section.find("TargetID").text == "HD 1+2"
        assert cal.metadata == {"keep": 1}

    def test_the_flag_survives_a_copy(self):
        seq = _seq("s0", "A", 0)
        seq.is_main = True

        assert seq.copy().is_main is True


class TestModelConstruction:
    """What the second Visibility is built with."""

    def test_absent_by_default(self):
        scheduler = ScheduleProcessor(TLE1, TLE2, **NOMINAL)

        assert scheduler.occultation_visibility is None
        assert scheduler.occultation_earthlimb_min is None

    def test_the_old_name_is_gone(self):
        with pytest.raises(TypeError):
            ScheduleProcessor(TLE1, TLE2, priority_0_earthlimb_min=54)

    def test_built_when_a_number_is_given(self):
        scheduler = ScheduleProcessor(
            TLE1, TLE2, occultation_earthlimb_min=54, **NOMINAL
        )

        assert scheduler.occultation_visibility is not None
        assert scheduler.occultation_visibility.earthlimb_min.to_value(
            u.deg
        ) == pytest.approx(54.0)

    def test_the_limb_angle_is_flat(self):
        """No day/night split and no dynamic wedge, whatever the rest says.

        The nominal model here has both. If either leaked through, the
        angle actually applied would vary with illumination instead of
        being the number the caller gave.
        """
        scheduler = ScheduleProcessor(
            TLE1, TLE2, occultation_earthlimb_min=54, **NOMINAL
        )
        strict = scheduler.occultation_visibility

        assert strict.use_dynamic_earthlimb is False
        assert scheduler.visibility.use_dynamic_earthlimb is True
        for attribute in ("earthlimb_day_min", "earthlimb_night_min"):
            nominal = getattr(scheduler.visibility, attribute)
            assert nominal is not None
            assert getattr(strict, attribute) in (None, strict.earthlimb_min)

    def test_every_other_keepout_is_untouched(self):
        """Only the boresight Earth limb differs between the two models."""
        scheduler = ScheduleProcessor(
            TLE1, TLE2, occultation_earthlimb_min=54, **NOMINAL
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
            assert getattr(scheduler.occultation_visibility, attribute) == (
                getattr(scheduler.visibility, attribute)
            ), attribute

    def test_it_is_stricter_in_practice(self):
        """The stricter model must actually reject more minutes.

        Construction assertions alone would pass even if the angle were
        wired to something the library ignores.
        """
        scheduler = ScheduleProcessor(
            TLE1, TLE2, occultation_earthlimb_min=70, **NOMINAL
        )
        coord = SkyCoord(15.0, -20.0, frame="icrs", unit="deg")
        times = T0 + np.arange(200) * u.min

        nominal = np.asarray(
            scheduler.visibility.get_visibility(coord, times)["visible"]
        )
        strict = np.asarray(
            scheduler.occultation_visibility.get_visibility(coord, times)[
                "visible"
            ]
        )

        assert strict.sum() < nominal.sum()
        # Stricter means a subset: nothing may become visible.
        assert not (strict & ~nominal).any()


class TestModelSelection:
    """Which model each observation is judged by: main or not, never priority."""

    @pytest.mark.parametrize("priority", [0, 1, 2])
    def test_the_main_target_gets_the_nominal_model(self, priority):
        scheduler = ScheduleProcessor(
            TLE1, TLE2, occultation_earthlimb_min=54, **NOMINAL
        )
        seq = _seq("s0", "A", priority)
        seq.is_main = True

        assert scheduler._visibility_for(seq) is scheduler.visibility

    @pytest.mark.parametrize("priority", [0, 1, 2])
    def test_everything_else_gets_the_occultation_model(self, priority):
        scheduler = ScheduleProcessor(
            TLE1, TLE2, occultation_earthlimb_min=54, **NOMINAL
        )
        seq = _seq("s0", "A", priority)

        assert (
            scheduler._visibility_for(seq) is scheduler.occultation_visibility
        )

    @pytest.mark.parametrize("is_main", [True, False])
    def test_unconfigured_means_one_model_for_everyone(self, is_main):
        scheduler = ScheduleProcessor(TLE1, TLE2, **NOMINAL)
        seq = _seq("s0", "A", 0)
        seq.is_main = is_main

        assert scheduler._visibility_for(seq) is scheduler.visibility


class _Recorder:
    """Forwards to a real model, noting which target it was asked about."""

    def __init__(self, name, inner, asked):
        self._name = name
        self._inner = inner
        self._asked = asked

    def __getattr__(self, attribute):
        inner = getattr(self._inner, attribute)
        if not callable(inner):
            return inner

        def _call(*args, **kwargs):
            for arg in args:
                if isinstance(arg, SkyCoord):
                    self._asked.append((self._name, round(arg.ra.deg, 4)))
                    break
            return inner(*args, **kwargs)

        return _call


class TestEveryPassAgrees:
    """No pass may judge an observation by a model another pass did not."""

    def _calendar(self):
        # A priority 0 main target and a priority 1 one that is not, so a
        # pass still choosing by priority gets both wrong.
        main = _seq("s0", "MAIN", 0, start_min=60, duration_min=50, ra=MAIN_RA)
        main.is_main = True
        other = _seq("s1", "OTHER", 1, start_min=180, duration_min=50)
        return ScienceCalendar(
            metadata={}, visits=[Visit(id="v1", sequences=[main, other])]
        )

    def test_no_pass_judges_a_non_main_target_by_the_nominal_model(self):
        """Record which model each pass asked for, per target.

        This is the check that catches a pass left reading
        ``self.visibility`` directly, or still choosing by priority.
        """
        scheduler = ScheduleProcessor(
            TLE1, TLE2, occultation_earthlimb_min=54, **NOMINAL
        )
        scheduler._print = lambda message: None
        asked = []
        scheduler.visibility = _Recorder(
            "nominal", scheduler.visibility, asked
        )
        scheduler.occultation_visibility = _Recorder(
            "strict", scheduler.occultation_visibility, asked
        )

        calendar = self._calendar()
        calendar = scheduler._trim_non_visible_tails(calendar)
        calendar = scheduler._trim_to_longest_visible_block(calendar)
        calendar = scheduler._enforce_start_buffers(calendar)
        scheduler.validate_visibility(calendar, report_issues=False)

        assert asked, "no pass consulted a visibility model"
        assert ("nominal", MAIN_RA) in asked
        assert ("strict", MAIN_RA) not in asked
        assert ("strict", OTHER_RA) in asked
        assert ("nominal", OTHER_RA) not in asked

    def test_delivered_observations_respect_their_own_limb(self):
        """End to end: what survives must satisfy the model it was judged by."""
        scheduler = ScheduleProcessor(
            TLE1,
            TLE2,
            occultation_earthlimb_min=30,
            earthlimb_gap_tolerance=0,
            st_gap_tolerance=0,
            st_gap_tolerance_start_buffer=0,
            earthlimb_gap_tolerance_start_buffer=0,
            **NOMINAL,
        )
        scheduler._print = lambda message: None
        calendar = scheduler._trim_non_visible_tails(self._calendar())
        calendar = scheduler._trim_to_longest_visible_block(calendar)

        for seq in calendar.visits[0].sequences:
            n_mins = int(np.rint(seq.duration.sec / 60.0))
            if n_mins <= 0:
                continue
            visible = np.atleast_1d(
                np.asarray(
                    scheduler._visibility_for(seq).get_visibility(
                        SkyCoord(seq.ra, seq.dec, frame="icrs", unit="deg"),
                        seq.start_time + np.arange(n_mins) * u.min,
                    )["visible"]
                )
            )
            assert visible.all(), (
                f"{seq.id} ({seq.target}) keeps "
                f"{int((~visible).sum())} minutes its own model rejects"
            )


class TestRollSweep:
    """A target is scored under the keepouts it will fly."""

    def _sweep(self, visit, overrides=()):
        used = []

        class _Named:
            def __init__(self, name, inner):
                self.name = name
                self._inner = inner

            def __getattr__(self, attribute):
                used.append(self.name)
                return getattr(self._inner, attribute)

        get_best_roll_per_visit(
            visit,
            _Named("nominal", Visibility(TLE1, TLE2)),
            roll_step=90.0,
            occultation_visibility=_Named(
                "strict", Visibility(TLE1, TLE2, earthlimb_min=70 * u.deg)
            ),
            occultation_overrides=overrides,
        )
        return used

    def test_a_non_main_target_uses_the_occultation_model(self):
        visit = Visit(
            id="v1",
            sequences=[_seq("s0", "A", 1), _seq("s1", "A", 1, start_min=60)],
        )

        used = self._sweep(visit)

        assert "strict" in used
        assert "nominal" not in used

    def test_the_main_target_uses_the_nominal_model(self):
        sequences = [_seq("s0", "A", 0), _seq("s1", "A", 0, start_min=60)]
        for seq in sequences:
            seq.is_main = True

        used = self._sweep(Visit(id="v1", sequences=sequences))

        assert "nominal" in used
        assert "strict" not in used

    def test_an_override_uses_the_nominal_model(self):
        visit = Visit(
            id="v1",
            sequences=[_seq("s0", "A", 0), _seq("s1", "A", 0, start_min=60)],
        )

        used = self._sweep(visit, overrides=frozenset({"A"}))

        assert "nominal" in used
        assert "strict" not in used

    @pytest.mark.parametrize(
        "given, target", [("HD_1", "HD 1"), ("HD 1", "HD_1"), ("HD+1", "HD 1")]
    )
    def test_an_override_matches_on_cleaned_names(self, given, target):
        """As in the scheduler, so the roll sweep and the timing passes
        agree on the keepout even with fix_bad_data off."""
        visit = Visit(
            id="v1",
            sequences=[
                _seq("s0", target, 0),
                _seq("s1", target, 0, start_min=60),
            ],
        )

        used = self._sweep(visit, overrides=[given])

        assert "nominal" in used
        assert "strict" not in used


class TestOverrides:
    """occultation_overrides: named targets keep the standard keepouts."""

    def _scheduler(self, overrides):
        return ScheduleProcessor(
            TLE1,
            TLE2,
            occultation_earthlimb_min=54,
            occultation_overrides=overrides,
            **NOMINAL,
        )

    @pytest.mark.parametrize("priority", [0, 1, 2])
    def test_a_non_main_override_gets_the_nominal_model(self, priority):
        scheduler = self._scheduler(["A"])

        assert (
            scheduler._visibility_for(_seq("s0", "A", priority))
            is scheduler.visibility
        )
        assert (
            scheduler._visibility_for(_seq("s1", "B", priority))
            is scheduler.occultation_visibility
        )

    def test_default_is_no_overrides(self):
        scheduler = ScheduleProcessor(
            TLE1, TLE2, occultation_earthlimb_min=54, **NOMINAL
        )

        assert scheduler.occultation_overrides is None
        assert not scheduler._overrides_occultation("A")

    @pytest.mark.parametrize(
        "given, target",
        [
            ("HD 260655c", "HD 260655c"),
            ("HD 260655c", "HD_260655c"),
            ("HD_260655c", "HD 260655c"),
            (" TOI-544b ", "TOI-544b"),
        ],
    )
    def test_either_spelling_of_a_name_matches(self, given, target):
        """Names are cleaned (space and + to _) after the window cut, so
        an override must match the target before and after."""
        assert self._scheduler([given])._overrides_occultation(target)

    def test_a_single_name_is_not_split_into_letters(self):
        scheduler = self._scheduler("TOI-544b")

        assert scheduler.occultation_overrides == ["TOI-544b"]
        assert scheduler._overrides_occultation("TOI-544b")
        assert not scheduler._overrides_occultation("T")

    def test_an_override_is_not_made_main(self):
        proc = _processor(occultation_overrides=["B"])
        visit = _visit(("A", 1, 30), ("B", 0, 90))

        assert proc._mark_main_targets(
            ScienceCalendar(metadata={}, visits=[visit])
        ) == {"v1": ["A"]}

    def test_a_priority_1_or_2_override_is_main_alongside_the_other(self):
        """No error: the override and the visit's own priority 1/2 target
        are both main."""
        proc = _processor(occultation_overrides=["Titan"])
        visit = _visit(
            ("TOI-544b", 1, 60), ("Titan", 2, 20), ("G1234567", 0, 30)
        )

        main = proc._mark_main_targets(
            ScienceCalendar(metadata={}, visits=[visit])
        )

        assert main == {"v1": ["TOI-544b", "Titan"]}
        assert not any(m.startswith("ERROR") for m in proc.printed)

    def test_two_other_priority_1_or_2_targets_still_an_error(self):
        proc = _processor(occultation_overrides=["Titan"])
        visit = _visit(("A", 1, 60), ("B", 2, 20), ("Titan", 2, 20))

        main = proc._mark_main_targets(
            ScienceCalendar(metadata={}, visits=[visit])
        )

        assert main == {"v1": ["Titan"]}
        errors = [m for m in proc.printed if m.startswith("ERROR")]
        assert len(errors) == 1 and "A, B" in errors[0]

    def test_a_priority_1_or_2_override_alone_is_main(self):
        proc = _processor(occultation_overrides=["Titan"])
        visit = _visit(("HD_1", 0, 90), ("Titan", 2, 20))

        assert proc._mark_main_targets(
            ScienceCalendar(metadata={}, visits=[visit])
        ) == {"v1": ["Titan"]}

    def test_a_priority_0_override_is_not_main(self):
        proc = _processor(occultation_overrides=["HD_29037"])
        visit = _visit(("TOI-544b", 1, 30), ("HD_29037", 0, 90))

        assert proc._mark_main_targets(
            ScienceCalendar(metadata={}, visits=[visit])
        ) == {"v1": ["TOI-544b"]}

    def test_every_main_target_is_flagged_on_the_copies(self, monkeypatch):
        proc = _processor(occultation_overrides=["Titan"])
        visit = _visit(("TOI-544b", 1, 60), ("Titan", 2, 20), ("HD_1", 0, 30))
        cal = ScienceCalendar(metadata={}, visits=[visit])
        seen = {}

        def _stop(calendar, *args, **kwargs):
            seen.update(
                {s.target: s.is_main for s in calendar.visits[0].sequences}
            )
            raise RuntimeError("stop here")

        monkeypatch.setattr(proc, "_analyze_original_calendar", _stop)
        monkeypatch.setattr(proc, "_setup_run_logging", lambda *a: None)

        with pytest.raises(RuntimeError, match="stop here"):
            proc.process_calendar(cal, window_start=T0, window_duration_days=1)

        assert seen == {"TOI-544b": True, "Titan": True, "HD_1": False}

    def test_a_name_matching_no_target_is_a_warning(self):
        proc = _processor(occultation_overrides=["B", "TYPO", "HD 1"])
        visit = _visit(("A", 1, 30), ("B", 0, 90), ("HD_1", 0, 10))

        proc._mark_main_targets(ScienceCalendar(metadata={}, visits=[visit]))

        warnings = [m for m in proc.printed if m.startswith("WARNING")]
        assert len(warnings) == 1
        assert "TYPO" in warnings[0]


class TestRowsAreShaded:
    """The two Gantt plots put a light band behind every row: blue for
    the main target, magenta for an override, green for the rest."""

    def _calendar(self):
        main = _seq("s0", "MAIN", 0, start_min=0, duration_min=20, ra=MAIN_RA)
        main.is_main = True
        override = _seq("s1", "OVR", 0, start_min=20, duration_min=20)
        other = _seq("s2", "OTHER", 0, start_min=40, duration_min=20, ra=17.0)
        return ScienceCalendar(
            metadata={},
            visits=[Visit(id="v1", sequences=[main, override, other])],
        )

    def _shaded(self, monkeypatch, plot):
        visualizer = ScheduleVisualizer(
            ScheduleProcessor(TLE1, TLE2, occultation_overrides=["OVR"])
        )
        rows = []
        original = visualizer._shade_row

        def _spy(ax, y, kind):
            rows.append((y, kind))
            original(ax, y, kind)

        monkeypatch.setattr(visualizer, "_shade_row", _spy)
        fig = getattr(visualizer, plot)(self._calendar())
        ax = fig.axes[0]
        labels = dict(
            zip(
                ax.get_yticks(),
                (t.get_text() for t in ax.get_yticklabels()),
            )
        )
        legend = [
            t.get_text()
            for legend in [ax.get_legend(), *fig.legends]
            if legend is not None
            for t in legend.get_texts()
        ]
        return {labels[y]: kind for y, kind in rows}, legend

    @pytest.mark.parametrize(
        "plot",
        ["plot_gantt_timeline_by_priority", "plot_gantt_with_visibility"],
    )
    def test_each_row_gets_its_band(self, monkeypatch, plot):
        shaded, legend = self._shaded(monkeypatch, plot)

        kinds = {
            next(
                name for name in ("MAIN", "OVR", "OTHER") if name in label
            ): kind
            for label, kind in shaded.items()
        }
        assert kinds == {"MAIN": "main", "OVR": "override", "OTHER": "other"}
        assert [t for t in legend if "target" in t or "override" in t] == [
            "Main target",
            "Occultation override",
            "Non-main target",
        ]

    def test_the_band_colors(self):
        bands = ScheduleVisualizer._ROW_BANDS

        assert bands["main"][0] == "tab:blue"
        assert bands["override"][0] == "magenta"
        assert bands["other"][0] == "tab:green"

    def test_visibility_plot_separates_visits_and_colors_priority_0(self):
        # Two visits of two rows each: one line, between rows 1 and 2.
        calendar = ScienceCalendar(
            metadata={},
            visits=[
                _visit(("A", 1, 20), ("B", 0, 20), vid="v1"),
                Visit(
                    id="v2",
                    sequences=[
                        _seq("s0", "C", 1, start_min=40, duration_min=20),
                        _seq("s1", "D", 0, start_min=60, duration_min=20),
                    ],
                ),
            ],
        )
        visualizer = ScheduleVisualizer(ScheduleProcessor(TLE1, TLE2))
        ax = visualizer.plot_gantt_with_visibility(calendar).axes[0]

        separators = [
            line.get_ydata()[0]
            for line in ax.get_lines()
            if line.get_color() == "black"
        ]
        assert separators == [1.5]
        priority_0 = matplotlib.colors.to_rgba(
            visualizer._get_priority_colors([0])[0]
        )
        assert priority_0 != matplotlib.colors.to_rgba("lightgray")
        assert any(patch.get_facecolor() == priority_0 for patch in ax.patches)
