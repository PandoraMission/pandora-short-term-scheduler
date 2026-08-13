"""Tests for in-place adjustment: growth, the movement limit, overlaps.

The scheduler absorbs a stale TLE by nudging observations around the times
the long-term calendar chose. It must not slide them off to a different
orbit, and it must never deliver an overlap.
"""

# Third-party
import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time, TimeDelta

# First-party/Local
from shortschedule.models import ObservationSequence, ScienceCalendar, Visit
from shortschedule.scheduler import ScheduleProcessor

T0 = Time("2026-01-01T00:00:00", scale="utc")


def _make_seq(sid, target, start_min, duration_min, ra=10.0, dec=20.0):
    start = T0 + start_min * u.min
    return ObservationSequence(
        id=sid,
        target=target,
        priority=1,
        start_time=start,
        stop_time=start + duration_min * u.min,
        ra=ra,
        dec=dec,
        payload_params={},
    )


def _make_calendar(sequences, visit_id="v1"):
    return ScienceCalendar(
        metadata={}, visits=[Visit(id=visit_id, sequences=sequences)]
    )


def _timing(calendar):
    """Snapshot the long-term timing the movement limit is measured from."""
    return {
        (visit.id, seq.id): (seq.start_time, seq.stop_time)
        for visit in calendar.visits
        for seq in visit.sequences
    }


class _PatternVis:
    """Visibility driven by a boolean array indexed in minutes from T0.

    Dark minutes are modelled as an Earth-limb failure with the star
    trackers clear, so the gap tolerances can be exercised.
    """

    def __init__(self, pattern):
        self.pattern = np.asarray(pattern, dtype=bool)

    def _lit(self, time):
        index = int(np.rint((time - T0).sec / 60.0))
        return (
            bool(self.pattern[index])
            if 0 <= index < len(self.pattern)
            else False
        )

    def get_visibility(self, coord, times, roll=None):
        indices = np.rint((times - T0).sec / 60.0).astype(int)
        return np.array(
            [
                bool(self.pattern[i]) if 0 <= i < len(self.pattern) else False
                for i in np.atleast_1d(indices)
            ],
            dtype=bool,
        )

    def get_all_constraints(self, coord, time):
        return {"moon": True, "sun": True, "earthlimb": self._lit(time)}

    def get_star_tracker_breakdown(self, coord, time, roll=None, pre=None):
        return {"passed": {"combined": True}}


def _processor(visibility=None, limit=45, earthlimb_gap_tolerance=0):
    proc = ScheduleProcessor.__new__(ScheduleProcessor)
    proc.visibility = visibility
    proc.max_movement_minutes = limit
    proc.min_sequence_duration = TimeDelta(8 * 60 * u.s)
    proc._roll_sweep_enabled = False
    proc._computed_target_rolls = {}
    # Zero tolerance means growth stops at the first dark minute, which
    # keeps most of these tests about bounds rather than about tolerances.
    proc.earthlimb_gap_tolerance = earthlimb_gap_tolerance
    proc.st_gap_tolerance = 0
    # The passes record what they did, so give the double the real report
    # schema rather than a hand-rolled stand-in that could drift from it.
    proc._initialize_gap_report()
    return proc


# ================================================================
# Growth into idle time
# ================================================================


class TestGrowIntoFreeTime:
    def test_grows_while_the_target_stays_visible(self):
        """An isolated observation expands over its visible window."""
        pattern = np.zeros(120, dtype=bool)
        pattern[20:80] = True  # visible 20-79
        proc = _processor(_PatternVis(pattern))
        seq = _make_seq("s1", "T", start_min=40, duration_min=10)
        cal = _make_calendar([seq])

        proc._grow_into_free_time(cal, _timing(cal))

        out = cal.visits[0].sequences[0]
        assert abs((out.start_time - (T0 + 20 * u.min)).sec) < 1
        assert abs((out.stop_time - (T0 + 80 * u.min)).sec) < 1

    def test_stops_at_the_neighbour_so_no_overlap_is_created(self):
        """Growth is bounded by the surrounding observations."""
        proc = _processor(_PatternVis(np.ones(200, dtype=bool)))
        first = _make_seq("s1", "A", start_min=0, duration_min=20)
        second = _make_seq("s2", "B", start_min=60, duration_min=20)
        cal = _make_calendar([first, second])

        proc._grow_into_free_time(cal, _timing(cal))

        a, b = cal.visits[0].sequences
        assert b.start_time >= a.stop_time
        assert not proc.validate_no_overlaps_astropy(cal, report_issues=False)

    def test_stops_at_the_movement_limit(self):
        """Growth cannot push a boundary past the limit."""
        proc = _processor(_PatternVis(np.ones(600, dtype=bool)), limit=45)
        seq = _make_seq("s1", "T", start_min=200, duration_min=20)
        cal = _make_calendar([seq])

        proc._grow_into_free_time(cal, _timing(cal))

        out = cal.visits[0].sequences[0]
        assert abs((out.start_time - (T0 + 155 * u.min)).sec) < 1
        assert abs((out.stop_time - (T0 + 265 * u.min)).sec) < 1

    def test_does_not_grow_across_a_dark_minute(self):
        """With no tolerance, a single dark minute stops the growth."""
        pattern = np.ones(120, dtype=bool)
        pattern[35] = False
        pattern[70] = False
        proc = _processor(_PatternVis(pattern))
        seq = _make_seq("s1", "T", start_min=40, duration_min=20)
        cal = _make_calendar([seq])

        proc._grow_into_free_time(cal, _timing(cal))

        out = cal.visits[0].sequences[0]
        assert abs((out.start_time - (T0 + 36 * u.min)).sec) < 1
        assert abs((out.stop_time - (T0 + 70 * u.min)).sec) < 1

    def test_grows_through_a_dark_run_the_tolerance_accepts(self):
        """A dip short enough to tolerate must not block growth past it.

        A dark stretch inside an observation is kept when the tolerance
        accepts it, so the same stretch sitting just outside must not stop
        the observation growing through it. Raising the tolerance otherwise
        has no effect on how far an observation can reach.
        """
        pattern = np.ones(120, dtype=bool)
        pattern[62:66] = False  # 4 dark minutes, then visible again
        proc = _processor(
            _PatternVis(pattern), earthlimb_gap_tolerance=6
        )
        seq = _make_seq("s1", "T", start_min=40, duration_min=20)
        cal = _make_calendar([seq])

        proc._grow_into_free_time(cal, _timing(cal))

        # Reaches the 45 min limit past the planned stop rather than
        # stopping at minute 62.
        out = cal.visits[0].sequences[0]
        assert abs((out.stop_time - (T0 + 105 * u.min)).sec) < 1

    def test_stops_at_a_dark_run_the_tolerance_rejects(self):
        """A dip longer than the tolerance still stops the walk."""
        pattern = np.ones(120, dtype=bool)
        pattern[62:75] = False  # 13 dark minutes against a 6 min tolerance
        proc = _processor(
            _PatternVis(pattern), earthlimb_gap_tolerance=6
        )
        seq = _make_seq("s1", "T", start_min=40, duration_min=20)
        cal = _make_calendar([seq])

        proc._grow_into_free_time(cal, _timing(cal))

        out = cal.visits[0].sequences[0]
        assert abs((out.stop_time - (T0 + 62 * u.min)).sec) < 1

    def test_growth_never_ends_on_a_dark_minute(self):
        """Growth stops on a visible minute even when a dip is tolerated."""
        pattern = np.ones(120, dtype=bool)
        pattern[62:66] = False
        pattern[70:] = False  # nothing visible beyond minute 70
        proc = _processor(
            _PatternVis(pattern), earthlimb_gap_tolerance=6
        )
        seq = _make_seq("s1", "T", start_min=40, duration_min=20)
        cal = _make_calendar([seq])

        proc._grow_into_free_time(cal, _timing(cal))

        out = cal.visits[0].sequences[0]
        assert abs((out.stop_time - (T0 + 70 * u.min)).sec) < 1


# ================================================================
# The movement limit
# ================================================================


class TestClampMovement:
    def test_within_the_limit_is_untouched(self):
        proc = _processor()
        seq = _make_seq("s1", "T", start_min=0, duration_min=30)
        original = _timing(cal := _make_calendar([seq]))
        seq.start_time = seq.start_time + 20 * u.min

        proc._clamp_movement(cal, original)

        assert abs((seq.start_time - (T0 + 20 * u.min)).sec) < 1

    def test_a_start_that_drifted_too_far_is_clamped_and_reported(self, capsys):
        """The orbit-scale shift this todo exists to stop."""
        proc = _processor()
        seq = _make_seq("s1", "T", start_min=200, duration_min=30)
        original = _timing(cal := _make_calendar([seq]))
        seq.start_time = seq.start_time - 97 * u.min  # a whole orbit earlier

        proc._clamp_movement(cal, original)

        assert abs((seq.start_time - (T0 + 155 * u.min)).sec) < 1
        assert "MOVED TOO FAR" in capsys.readouterr().out

    def test_a_stop_that_drifted_too_far_is_clamped(self, capsys):
        proc = _processor()
        seq = _make_seq("s1", "T", start_min=100, duration_min=30)
        original = _timing(cal := _make_calendar([seq]))
        seq.stop_time = seq.stop_time + 200 * u.min

        proc._clamp_movement(cal, original)

        assert abs((seq.stop_time - (T0 + 175 * u.min)).sec) < 1
        assert "MOVED TOO FAR" in capsys.readouterr().out

    def test_clamp_is_refused_when_it_would_leave_too_short(self, capsys):
        """A clamp that would breach the minimum duration is not applied.

        Pulling the start back to the limit normally lengthens an
        observation, so this is a defensive guard: it needs a start that
        drifted past the limit while the observation was also cut down to
        almost nothing.
        """
        proc = _processor()
        seq = _make_seq("s1", "T", start_min=0, duration_min=60)
        original = _timing(cal := _make_calendar([seq]))
        # Starts 50 min late (over the limit) and only 2 minutes long, so
        # clamping the start to +45 would leave 7 minutes.
        seq.start_time = T0 + 50 * u.min
        seq.stop_time = T0 + 52 * u.min

        proc._clamp_movement(cal, original)

        assert seq.start_time == T0 + 50 * u.min
        assert seq.stop_time == T0 + 52 * u.min
        assert "manual review" in capsys.readouterr().out

    def test_a_zero_limit_disables_the_check(self):
        proc = _processor(limit=0)
        seq = _make_seq("s1", "T", start_min=200, duration_min=30)
        original = _timing(cal := _make_calendar([seq]))
        seq.start_time = seq.start_time - 97 * u.min

        proc._clamp_movement(cal, original)

        assert abs((seq.start_time - (T0 + 103 * u.min)).sec) < 1


# ================================================================
# The overlap guarantee
# ================================================================


class TestGapReportRecordsWhatHappened:
    """The report must describe the passes that actually run now.

    The old filled-versus-remaining gap counters were left behind when gap
    filling was removed, and reported zero forever.
    """

    def test_growth_is_recorded(self):
        pattern = np.zeros(120, dtype=bool)
        pattern[20:80] = True
        proc = _processor(_PatternVis(pattern))
        seq = _make_seq("s1", "T", start_min=40, duration_min=10)
        cal = _make_calendar([seq])

        proc._grow_into_free_time(cal, _timing(cal))

        summary = proc.gap_report["processing_summary"]
        assert summary["minutes_grown_at_starts"] == 20
        assert summary["minutes_grown_at_stops"] == 30

    def test_clamping_is_recorded(self):
        proc = _processor()
        seq = _make_seq("s1", "T", start_min=200, duration_min=30)
        original = _timing(cal := _make_calendar([seq]))
        seq.start_time = seq.start_time - 97 * u.min

        proc._clamp_movement(cal, original)

        assert proc.gap_report["processing_summary"]["boundaries_clamped"] == 1

    def test_overlap_repairs_are_recorded(self):
        proc = _processor()
        cal = _make_calendar(
            [
                _make_seq("s1", "A", start_min=0, duration_min=40),
                _make_seq("s2", "B", start_min=30, duration_min=20),
            ]
        )

        proc._repair_overlaps(cal)

        assert proc.gap_report["processing_summary"]["overlaps_repaired"] == 1

    def test_modifications_are_tallied(self):
        proc = _processor()
        grown = _make_seq("s1", "A", start_min=0, duration_min=20)
        trimmed = _make_seq("s2", "B", start_min=60, duration_min=20)
        untouched = _make_seq("s3", "C", start_min=120, duration_min=20)
        original = _timing(cal := _make_calendar([grown, trimmed, untouched]))
        grown.stop_time = grown.stop_time + 10 * u.min
        trimmed.stop_time = trimmed.stop_time - 5 * u.min

        proc._log_timing_changes(cal, original)

        summary = proc.gap_report["processing_summary"]
        assert summary["sequences_lengthened"] == 1
        assert summary["sequences_shortened"] == 1
        assert summary["sequences_modified"] == 2
        modifications = proc.gap_report["sequence_modifications"]
        assert len(modifications["unchanged_sequences"]) == 1


class TestRepairOverlaps:
    def test_a_clean_calendar_is_untouched(self):
        proc = _processor()
        cal = _make_calendar(
            [
                _make_seq("s1", "A", start_min=0, duration_min=20),
                _make_seq("s2", "B", start_min=30, duration_min=20),
            ]
        )

        proc._repair_overlaps(cal)

        a, b = cal.visits[0].sequences
        assert abs((a.stop_time - (T0 + 20 * u.min)).sec) < 1
        assert abs((b.start_time - (T0 + 30 * u.min)).sec) < 1

    def test_the_earlier_stop_is_pulled_back_and_reported(self, capsys):
        proc = _processor()
        first = _make_seq("s1", "A", start_min=0, duration_min=40)
        second = _make_seq("s2", "B", start_min=30, duration_min=20)
        cal = _make_calendar([first, second])

        proc._repair_overlaps(cal)

        assert abs((first.stop_time - second.start_time).sec) < 1
        assert not proc.validate_no_overlaps_astropy(cal, report_issues=False)
        assert "OVERLAP" in capsys.readouterr().out

    def test_an_unrepairable_overlap_is_left_and_reported(self, capsys):
        """Truncating below the minimum duration is worse than reporting."""
        proc = _processor()
        first = _make_seq("s1", "A", start_min=0, duration_min=40)
        second = _make_seq("s2", "B", start_min=3, duration_min=40)
        cal = _make_calendar([first, second])

        proc._repair_overlaps(cal)

        assert abs((first.stop_time - (T0 + 40 * u.min)).sec) < 1
        output = capsys.readouterr().out
        assert "OVERLAP" in output
        assert "manual fix" in output

    def test_one_observation_overlapping_two_others(self, capsys):
        """Truncating to the next start resolves the whole run."""
        proc = _processor()
        cal = _make_calendar(
            [
                _make_seq("s1", "A", start_min=0, duration_min=120),
                _make_seq("s2", "B", start_min=40, duration_min=20),
                _make_seq("s3", "C", start_min=70, duration_min=20),
            ]
        )

        proc._repair_overlaps(cal)

        assert not proc.validate_no_overlaps_astropy(cal, report_issues=False)


# ================================================================
# The pipeline no longer slides observations to close gaps
# ================================================================


def test_pipeline_keeps_observations_at_their_planned_time(monkeypatch):
    """Idle time between observations is left alone.

    Previously every observation had its start dragged backwards to abut
    the one before it, which is what produced the orbit-scale shifts.
    """
    monkeypatch.setattr(
        "shortschedule.scheduler.Visibility",
        lambda *a, **kw: _PatternVis(np.ones(400, dtype=bool)),
    )
    proc = ScheduleProcessor("L1", "L2", max_movement_minutes=45)
    proc.earthlimb_gap_tolerance = 0
    proc.st_gap_tolerance = 0
    proc.st_gap_tolerance_start_buffer = 0

    first = _make_seq("s1", "A", start_min=0, duration_min=20)
    second = _make_seq("s2", "B", start_min=120, duration_min=20)
    cal = _make_calendar([first, second])

    result = proc._process_all_sequences(cal)

    a, b = result.visits[0].sequences
    # B grows backwards only as far as its own 45 minute allowance, and
    # never back to A's stop the way blind gap filling used to drag it.
    assert (b.start_time - (T0 + 75 * u.min)).sec == pytest.approx(0, abs=60)
    assert not proc.validate_no_overlaps_astropy(result, report_issues=False)
