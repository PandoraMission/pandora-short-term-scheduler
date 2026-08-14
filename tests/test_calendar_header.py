"""What the delivered calendar's Meta header says about the run.

Two things are pinned here.

``Calendar_Status`` reflects whether the run failed, not whether a
validator had something to say. Under the gap tolerances a healthy
calendar legitimately contains keepout violations, and merging can
absorb one deliberately, so `validate_visibility` reporting entries is
expected rather than disqualifying. Only an error logged during the run
makes the calendar invalid.

The keepouts and tolerances that were applied are recorded on the header
too, read back off the ``Visibility`` instances rather than off the
constructor arguments, so a keepout the caller left unset is reported as
the value the library actually supplied.
"""

# Standard library
import xml.etree.ElementTree as ET

# Third-party
import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time
from pandoravisibility import Visibility

# First-party/Local
from shortschedule.models import ObservationSequence, ScienceCalendar, Visit
from shortschedule.scheduler import ScheduleProcessor
from shortschedule.writer import XMLWriter

TLE1 = "1 67395U 80229J   26196.69732639  .00000000  00000-0  37770-3 0    00"
TLE2 = "2 67395  97.8056 194.9117 0006480  50.2285  39.6294 14.88117629    09"

T0 = Time("2026-07-16T00:00:00", scale="utc")

NAMESPACE = "{/pandora/calendar/}"


class _AllVisible:
    """Visibility that never objects, so nothing logs an error of its own."""

    _st_constraint_active = False

    def get_visibility(self, coord, times, roll=None):
        return np.ones(np.atleast_1d(times).size, dtype=bool)

    def get_all_constraints(self, coord, time):
        return {"moon": True, "sun": True, "earthlimb": True}

    def get_separations(self, coord, time):
        return {}


def _calendar():
    sequences = [
        ObservationSequence(
            id="s1",
            target="TARGET_A",
            priority=1,
            start_time=T0,
            stop_time=T0 + 45 * u.min,
            ra=15.0,
            dec=-20.0,
            payload_params={},
        ),
        ObservationSequence(
            id="s2",
            target="TARGET_B",
            priority=0,
            start_time=T0 + 120 * u.min,
            stop_time=T0 + 165 * u.min,
            ra=250.0,
            dec=45.0,
            payload_params={},
        ),
    ]
    return ScienceCalendar(
        metadata={}, visits=[Visit(id="v1", sequences=sequences)]
    )


def _process(tmp_path, scheduler, stub_visibility=True):
    # The stub keeps the status tests free of visibility-driven errors.
    # The header tests must NOT use it: the settings are read back off the
    # Visibility instance, which is the whole point of recording them.
    if stub_visibility:
        scheduler.visibility = _AllVisible()
    return scheduler.process_calendar(
        _calendar(),
        window_start=T0,
        window_duration_days=1,
        log_path=tmp_path / "run",
    )


class TestErrorCounting:
    """``run_error_count`` counts errors and nothing else."""

    def _prepared(self, tmp_path):
        scheduler = ScheduleProcessor(TLE1, TLE2)
        scheduler._setup_run_logging(_calendar(), False, tmp_path / "run")
        return scheduler

    def test_starts_at_zero(self, tmp_path):
        assert self._prepared(tmp_path).run_error_count == 0

    def test_info_and_warnings_do_not_count(self, tmp_path):
        """The whole point: a noisy but successful run is still successful."""
        scheduler = self._prepared(tmp_path)

        scheduler._print("Processing calendar with TLE:")
        scheduler._print("  Visit 1 / TARGET_A: best roll = 12.0")
        scheduler._print("WARNING: BAD DATA: something looked odd")
        scheduler._print("WARNING: SINGLE-ROI: no usable target RA/Dec")

        assert scheduler.run_error_count == 0

    def test_errors_count(self, tmp_path):
        scheduler = self._prepared(tmp_path)

        scheduler._print("ERROR: OVERLAP: needs a manual fix")
        scheduler._print("ERROR: START BUFFER: left unchanged")

        assert scheduler.run_error_count == 2

    def test_a_processor_that_never_ran_reports_zero(self):
        assert ScheduleProcessor(TLE1, TLE2).run_error_count == 0

    def test_the_count_resets_between_runs(self, tmp_path):
        """Otherwise one bad run would poison every later one."""
        scheduler = self._prepared(tmp_path)
        scheduler._print("ERROR: something went wrong")
        assert scheduler.run_error_count == 1

        scheduler._setup_run_logging(_calendar(), False, tmp_path / "run2")

        assert scheduler.run_error_count == 0


class TestCalendarStatus:
    """``Calendar_Status`` follows errors, not validator chatter."""

    def test_a_clean_run_is_valid(self, tmp_path):
        processed = _process(tmp_path, ScheduleProcessor(TLE1, TLE2))

        assert processed.metadata["calendar_status"] == "VALID"

    def test_validator_issues_alone_do_not_invalidate(self, tmp_path):
        """A reported issue with no error keeps the calendar valid.

        This is the behaviour change: tolerated keepout violations are
        reported by ``validate_visibility`` on a perfectly good week, and
        used to mark the delivered calendar INVALID.
        """
        scheduler = ScheduleProcessor(TLE1, TLE2)
        scheduler.validate_visibility = lambda calendar, report_issues=True: [
            {"sequence_id": "s1", "message": "tolerated dip"}
        ]
        processed = _process(tmp_path, scheduler)

        assert scheduler.run_error_count == 0
        assert processed.metadata["calendar_status"] == "VALID"

    def test_an_error_during_the_run_invalidates(self, tmp_path):
        """Anything logged at ERROR level is enough, wherever it came from."""
        scheduler = ScheduleProcessor(TLE1, TLE2)
        original = scheduler._trim_non_visible_tails

        def _erroring(calendar):
            scheduler._print("ERROR: OVERLAP: still present after repair")
            return original(calendar)

        scheduler._trim_non_visible_tails = _erroring
        processed = _process(tmp_path, scheduler)

        assert scheduler.run_error_count >= 1
        assert processed.metadata["calendar_status"] == "INVALID"


class TestSettingsInTheHeader:
    """The configuration the run applied is recorded on ``Meta``."""

    def _meta(self, tmp_path, scheduler):
        processed = _process(tmp_path, scheduler, stub_visibility=False)
        path = tmp_path / "out.xml"
        XMLWriter().write_calendar(processed, output_path=str(path))
        return ET.parse(path).getroot().find(f"{NAMESPACE}Meta").attrib

    def test_tolerances_and_keepouts_are_written(self, tmp_path):
        scheduler = ScheduleProcessor(
            TLE1,
            TLE2,
            earthlimb_gap_tolerance=6,
            st_gap_tolerance=12,
            st_gap_tolerance_start_buffer=11,
            earthlimb_gap_tolerance_start_buffer=9,
            earthlimb_day_min=44,
            earthlimb_night_min=13,
            sun_min=91,
            moon_min=20,
            st_sun_min=50,
            st_moon_min=20,
            st_earthlimb_min=30,
            use_dynamic_earthlimb=True,
            roll_step=1.0,
            min_power_frac=0.68,
        )
        meta = self._meta(tmp_path, scheduler)

        assert meta["Earthlimb_Gap_Tolerance_Min"] == "6"
        assert meta["ST_Gap_Tolerance_Min"] == "12"
        assert meta["ST_Gap_Tolerance_Start_Buffer_Min"] == "11"
        assert meta["Earthlimb_Gap_Tolerance_Start_Buffer_Min"] == "9"
        assert meta["Roll_Step_Deg"] == "1"
        assert meta["Min_Power_Frac"] == "0.68"
        assert meta["Max_Movement_Min"] == "45"

    def test_an_unset_keepout_reports_the_library_default(self, tmp_path):
        """Not the blank the caller passed.

        All keepouts default to ``None`` here so ``pandoravisibility``
        stays the single source of truth. The header has to say what was
        actually applied, or it records a configuration nobody flew.
        """
        scheduler = ScheduleProcessor(TLE1, TLE2)
        meta = self._meta(tmp_path, scheduler)

        library_moon = Visibility(TLE1, TLE2).moon_min.to_value(u.deg)

        assert meta["Moon_Min_Deg"] == f"{library_moon:g}"
        assert float(meta["Moon_Min_Deg"]) == pytest.approx(25.0)

    def test_priority_0_limb_is_absent_when_unused(self, tmp_path):
        meta = self._meta(tmp_path, ScheduleProcessor(TLE1, TLE2))

        assert "Priority_0_Earthlimb_Min_Deg" not in meta

    def test_priority_0_limb_is_written_when_set(self, tmp_path):
        meta = self._meta(
            tmp_path, ScheduleProcessor(TLE1, TLE2, priority_0_earthlimb_min=54)
        )

        assert meta["Priority_0_Earthlimb_Min_Deg"] == "54"

    def test_status_reaches_the_header(self, tmp_path):
        scheduler = ScheduleProcessor(TLE1, TLE2)
        processed = _process(tmp_path, scheduler)
        path = tmp_path / "status.xml"
        XMLWriter().write_calendar(processed, output_path=str(path))
        meta = ET.parse(path).getroot().find(f"{NAMESPACE}Meta").attrib

        assert meta["Calendar_Status"] == "VALID"

    def test_a_calendar_with_no_settings_still_writes(self, tmp_path):
        """The writer must not require the new metadata key.

        Calendars built in memory, or parsed from an older delivery, have
        no ``scheduler_settings`` at all.
        """
        path = tmp_path / "bare.xml"
        XMLWriter().write_calendar(_calendar(), output_path=str(path))
        meta = ET.parse(path).getroot().find(f"{NAMESPACE}Meta").attrib

        assert "Sun_Min_Deg" not in meta
