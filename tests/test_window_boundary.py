"""Tests for the schedule window as a hard boundary.

The delivered calendar must sit inside ``[window_start, window_end)``. An
observation the long-term calendar runs across an edge is cut there, growth
stops there (with a warning if the target would have stayed visible), and
the timing validator flags anything that still reaches outside.
"""

# Third-party
import pytest
from astropy import units as u
from astropy.time import TimeDelta

# First-party/Local
from shortschedule.models import ScienceCalendar, Visit
from shortschedule.scheduler import ScheduleProcessor
from tests.test_edge_refinement import T0, _at, _off_by, _SecondsVis, _seq

# A one-hour window, 01:00 to 02:00 after T0.
START, DAYS = _at(60), 1.0 / 24.0
END = _at(120)


def _processor(monkeypatch, *spans, **kwargs):
    monkeypatch.setattr(
        "shortschedule.scheduler.Visibility",
        lambda *a, **kw: _SecondsVis(*spans),
    )
    kwargs.setdefault("st_gap_tolerance_start_buffer", 0)
    kwargs.setdefault("earthlimb_gap_tolerance_start_buffer", 0)
    proc = ScheduleProcessor("L1", "L2", **kwargs)
    proc._initialize_gap_report()
    return proc


def _visits(*groups):
    return ScienceCalendar(
        metadata={},
        visits=[
            Visit(id=f"v{i}", sequences=list(group))
            for i, group in enumerate(groups, start=1)
        ],
    )


def _run(proc, calendar):
    """Extract the window and process it, as process_calendar does."""
    windowed = proc._extract_time_window(calendar, START, DAYS, False)
    return proc._process_all_sequences(windowed)


def _all(calendar):
    return [seq for visit in calendar.visits for seq in visit.sequences]


class TestExtraction:
    def test_observations_across_either_edge_are_cut_to_the_window(
        self, monkeypatch, capsys
    ):
        proc = _processor(monkeypatch, (0, 3 * 3600))
        before = _seq(1, _at(50), _at(75))
        inside = _seq(2, _at(80), _at(100))
        after = _seq(3, _at(110), _at(130))
        cal = _visits([before, inside, after])

        windowed = proc._extract_time_window(cal, START, DAYS, False)

        kept = _all(windowed)
        assert [seq.id for seq in kept] == [1, 2, 3]
        assert _off_by(kept[0].start_time, START) < 0.01
        assert _off_by(kept[0].stop_time, _at(75)) < 0.01
        # A copy, so later passes cannot change the long-term calendar.
        assert kept[1] is not inside
        assert _off_by(kept[1].start_time, inside.start_time) < 0.01
        assert _off_by(kept[1].stop_time, inside.stop_time) < 0.01
        assert _off_by(kept[2].stop_time, END) < 0.01
        out = capsys.readouterr().out
        assert out.count("WARNING") == 2 and "WINDOW" in out
        assert proc.gap_report["processing_summary"]["window_clipped"] == 2

    def test_the_input_calendar_keeps_its_times(self, monkeypatch):
        proc = _processor(monkeypatch, (0, 3 * 3600))
        after = _seq(1, _at(110), _at(130))

        proc._extract_time_window(_visits([after]), START, DAYS, False)

        assert _off_by(after.stop_time, _at(130)) < 0.01

    def test_observations_wholly_outside_are_left_out(self, monkeypatch):
        proc = _processor(monkeypatch, (0, 3 * 3600))
        cal = _visits([_seq(1, _at(30), _at(60)), _seq(2, _at(120), _at(140))])

        assert _all(proc._extract_time_window(cal, START, DAYS, False)) == []

    @pytest.mark.parametrize(
        "priority, drop, kept",
        [(0, True, False), (0, False, True), (1, True, True)],
    )
    def test_a_cut_under_the_minimum(
        self, monkeypatch, capsys, priority, drop, kept
    ):
        """Dropped only for a priority 0 with drop_priority0."""
        proc = _processor(monkeypatch, (0, 3 * 3600), drop_priority0=drop)
        short = _seq(1, _at(115), _at(130))
        short.priority = priority

        windowed = proc._extract_time_window(
            _visits([short]), START, DAYS, False
        )

        out = capsys.readouterr().out
        assert "WARNING" in out and "under 8 min" in out
        assert bool(_all(windowed)) == kept
        assert ("Dropped (drop_priority0)" in out) != kept
        summary = proc.gap_report["processing_summary"]
        assert summary["priority_0_dropped"] == (0 if kept else 1)


class TestGrowth:
    def test_stop_held_at_the_window_end_with_a_warning(
        self, monkeypatch, capsys
    ):
        proc = _processor(monkeypatch, (0, 3 * 3600))
        seq = _seq(1, _at(90), _at(110))

        result = _run(proc, _visits([seq]))

        (grown,) = _all(result)
        assert _off_by(grown.stop_time, END) < 0.01
        out = capsys.readouterr().out
        assert "more min after its stop, past the window end" in out
        assert out.count("past the window end") == 1

    def test_start_held_at_the_window_start_with_a_warning(
        self, monkeypatch, capsys
    ):
        proc = _processor(monkeypatch, (0, 3 * 3600))
        seq = _seq(1, _at(70), _at(90))

        result = _run(proc, _visits([seq]))

        (grown,) = _all(result)
        assert _off_by(grown.start_time, START) < 0.01
        assert "before its start, past the window start" in (
            capsys.readouterr().out
        )

    def test_already_on_the_edge_still_warns(self, monkeypatch, capsys):
        """No room to grow, but the target would have stayed visible."""
        proc = _processor(monkeypatch, (0, 3 * 3600))
        seq = _seq(1, _at(100), END)

        result = _run(proc, _visits([seq]))

        assert _off_by(_all(result)[0].stop_time, END) < 0.01
        assert "past the window end" in capsys.readouterr().out

    def test_no_warning_when_the_target_sets_at_the_edge(
        self, monkeypatch, capsys
    ):
        proc = _processor(monkeypatch, (60 * 60, 120 * 60))
        seq = _seq(1, _at(90), _at(110))

        result = _run(proc, _visits([seq]))

        assert _off_by(_all(result)[0].stop_time, END) < 0.01
        assert "WINDOW" not in capsys.readouterr().out

    def test_no_warning_for_growth_stopped_by_a_neighbor(
        self, monkeypatch, capsys
    ):
        proc = _processor(monkeypatch, (0, 3 * 3600))
        first = _seq(1, _at(90), _at(100))
        last = _seq(2, _at(105), END)

        result = _run(proc, _visits([first, last]))

        assert _off_by(_all(result)[0].stop_time, _at(105)) < 0.01
        out = capsys.readouterr().out
        # Only the last observation, on the edge, is warned about.
        assert out.count("past the window end") == 1
        assert "-v1-2 | WINDOW" in out


def test_the_1s_pass_does_not_cross_the_edge(monkeypatch):
    """Visible past the edge by a few seconds: the stop still stays put."""
    proc = _processor(monkeypatch, (0, 120 * 60 + 40))
    seq = _seq(1, _at(90), _at(110))

    result = _run(proc, _visits([seq]))

    assert _off_by(_all(result)[0].stop_time, END) < 0.01


def test_wasp52_and_toi5293_cases(monkeypatch):
    """The WY46 calendar: one grew before the start, one merged past the end."""
    proc = _processor(monkeypatch, (0, 3 * 3600))
    opening = _seq(1, _at(60), _at(68))
    opening.priority = 0
    closing = _seq(2, _at(102), _at(114))
    tail = _seq(3, _at(116), _at(132))
    cal = _visits([opening], [closing, tail])

    result = _run(proc, cal)

    for seq in _all(result):
        assert seq.start_time >= START - TimeDelta(1e-3 * u.s)
        assert seq.stop_time <= END + TimeDelta(1e-3 * u.s)


def test_validator_flags_an_observation_outside_the_window(capsys):
    proc = ScheduleProcessor.__new__(ScheduleProcessor)
    proc.min_sequence_duration = TimeDelta(8 * 60 * u.s)
    proc.window_start, proc.window_end = START, END
    cal = _visits([_seq(1, _at(100), _at(125))])

    issues = proc.validate_sequence_timing(cal, report_issues=False)

    assert len(issues["outside_window"]) == 1
    assert issues["timing_summary"]["outside_window_found"] == 1
    assert "ERROR" in capsys.readouterr().out


def test_validator_is_quiet_without_a_window():
    proc = ScheduleProcessor.__new__(ScheduleProcessor)
    proc.min_sequence_duration = TimeDelta(8 * 60 * u.s)
    cal = _visits([_seq(1, T0, _at(20))])

    issues = proc.validate_sequence_timing(cal, report_issues=False)

    assert issues["outside_window"] == []
