"""Tests for settling observation boundaries to the second.

Every other pass samples visibility once a minute, so a target setting
toward the Earth limb can leave the keepouts up to a minute before the stop
the minute grid gives it. The last pass rechecks the minute either side of
each boundary at 1 s.
"""

# Third-party
import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time, TimeDelta

# First-party/Local
from shortschedule.models import ObservationSequence, ScienceCalendar, Visit
from shortschedule.scheduler import ScheduleProcessor
from tests.doubles import answers_visibility

T0 = Time("2026-01-01T00:00:00", scale="utc")


def _at(minutes, seconds=0):
    return T0 + (minutes * 60 + seconds) * u.s


class _SecondsVis:
    """Visibility true inside the given ``[start, stop)`` second spans of T0."""

    def __init__(self, *spans):
        self.spans = spans

    def _visible(self, times):
        seconds = np.round(np.atleast_1d((times - T0).sec), 3)
        out = np.zeros(seconds.shape, dtype=bool)
        for start, stop in self.spans:
            out |= (seconds >= start) & (seconds < stop)
        return out

    @answers_visibility
    def get_visibility(self, coord, times, roll=None):
        out = self._visible(times)
        return out if np.ndim((times - T0).sec) else bool(out[0])

    def get_constraint(self, coord, body, times):
        return self._visible(times)

    def get_star_tracker_breakdown(self, coord, times, roll=None, pre=None):
        return {"passed": {"combined": np.ones(np.size(times), dtype=bool)}}


def _seq(sid, start, stop):
    return ObservationSequence(
        id=sid,
        target=f"T{sid}",
        priority=1,
        start_time=start,
        stop_time=stop,
        ra=10.0,
        dec=20.0,
        payload_params={},
    )


def _calendar(*sequences):
    return ScienceCalendar(
        metadata={}, visits=[Visit(id="v1", sequences=list(sequences))]
    )


def _timing(calendar):
    return {
        (visit.id, seq.id): (seq.start_time, seq.stop_time)
        for visit in calendar.visits
        for seq in visit.sequences
    }


def _processor(visibility, limit=45):
    proc = ScheduleProcessor.__new__(ScheduleProcessor)
    proc.visibility = visibility
    proc.max_movement_minutes = limit
    proc.min_sequence_duration = TimeDelta(8 * 60 * u.s)
    proc._initialize_gap_report()
    return proc


def _refine(proc, calendar, original=None):
    proc._refine_edges_to_second(calendar, original or _timing(calendar))


def _off_by(time, expected):
    return abs((time - expected).sec)


class TestStop:
    def test_pulled_back_to_where_the_target_sets(self):
        """The WASP-52 case: dark from 30:25, the minute grid stopped at 31."""
        proc = _processor(_SecondsVis((10 * 60, 30 * 60 + 25)))
        seq = _seq(1, _at(10), _at(31))
        cal = _calendar(seq)

        _refine(proc, cal)

        assert _off_by(seq.stop_time, _at(30, 25)) < 0.01
        assert _off_by(seq.start_time, _at(10)) < 0.01

    def test_grows_over_the_seconds_still_visible(self):
        proc = _processor(_SecondsVis((0, 30 * 60 + 40)))
        seq = _seq(1, _at(10), _at(30))
        cal = _calendar(seq)

        _refine(proc, cal)

        assert _off_by(seq.stop_time, _at(30, 40)) < 0.01

    def test_growth_stops_at_the_next_observation(self):
        proc = _processor(_SecondsVis((0, 3600)))
        first = _seq(1, _at(10), _at(30))
        second = _seq(2, _at(30, 20), _at(45))
        cal = _calendar(first, second)

        _refine(proc, cal)

        assert _off_by(first.stop_time, _at(30, 20)) < 0.01
        assert first.stop_time <= second.start_time

    def test_growth_stops_at_the_movement_limit(self):
        proc = _processor(_SecondsVis((0, 30 * 60 + 40)), limit=1)
        seq = _seq(1, _at(10), _at(30))
        cal = _calendar(seq)
        original = {("v1", 1): (_at(10), _at(29, 30))}

        _refine(proc, cal, original)

        assert _off_by(seq.stop_time, _at(30, 30)) < 0.01

    def test_cut_below_the_minimum_is_applied_with_a_warning(self, capsys):
        """Seconds the target cannot be seen are worth less than the minimum."""
        proc = _processor(_SecondsVis((10 * 60, 17 * 60 + 30)))
        seq = _seq(1, _at(10), _at(18))
        cal = _calendar(seq)

        _refine(proc, cal)

        assert _off_by(seq.stop_time, _at(17, 30)) < 0.01
        assert "WARNING" in capsys.readouterr().out


class TestUnderTheMinimum:
    """A cut under ``min_sequence_duration``: drop a priority 0 only with
    ``drop_priority0``; otherwise keep it short with a warning."""

    @staticmethod
    def _short_calendar(priority):
        # 10:00-18:00, dark from 17:30: the cut leaves 7.5 min. A neighbor
        # starts at 18:30 so a dropped observation's room is not taken.
        short = _seq(1, _at(10), _at(18))
        short.priority = priority
        after = _seq(2, _at(18, 30), _at(40))
        return _calendar(short, after), short

    @pytest.mark.parametrize(
        "priority, drop, kept",
        [(0, True, False), (0, False, True), (1, True, True), (2, True, True)],
    )
    def test_dropped_only_for_priority_0_with_drop_priority0(
        self, capsys, priority, drop, kept
    ):
        proc = _processor(
            _SecondsVis((10 * 60, 17 * 60 + 30), (18 * 60 + 30, 3600))
        )
        proc.drop_priority0 = drop
        cal, short = self._short_calendar(priority)

        _refine(proc, cal)

        remaining = [seq.id for v in cal.visits for seq in v.sequences]
        out = capsys.readouterr().out
        assert "WARNING" in out and "EDGES" in out
        if kept:
            assert remaining == [1, 2]
            assert _off_by(short.stop_time, _at(17, 30)) < 0.01
            assert "Dropped" not in out
        else:
            assert remaining == [2]
            assert "Dropped (drop_priority0)" in out
            assert proc.gap_report["processing_summary"]["priority_0_dropped"] == 1

    def test_a_visit_emptied_by_the_drop_goes_too(self):
        proc = _processor(_SecondsVis((10 * 60, 17 * 60 + 30)))
        proc.drop_priority0 = True
        short = _seq(1, _at(10), _at(18))
        short.priority = 0
        other = _seq(2, _at(30), _at(45))
        cal = ScienceCalendar(
            metadata={},
            visits=[
                Visit(id="v1", sequences=[short]),
                Visit(id="v2", sequences=[other]),
            ],
        )

        _refine(proc, cal)

        assert [v.id for v in cal.visits] == ["v2"]

    def test_the_next_start_skips_a_dropped_neighbor(self):
        """Growth looks past a dropped observation to the one before it."""
        proc = _processor(
            _SecondsVis((10 * 60, 17 * 60 + 30), (17 * 60 + 45, 3600))
        )
        proc.drop_priority0 = True
        short = _seq(1, _at(10), _at(18))
        short.priority = 0
        after = _seq(2, _at(18, 0), _at(40))
        cal = _calendar(short, after)

        _refine(proc, cal)

        # The dropped observation's stop (18:00) no longer bounds it; the
        # target is visible from 17:45, so the start grows 15 s.
        assert [seq.id for v in cal.visits for seq in v.sequences] == [2]
        assert _off_by(after.start_time, _at(17, 45)) < 0.01


class TestStart:
    def test_grows_earlier_over_the_seconds_already_visible(self):
        proc = _processor(_SecondsVis((9 * 60 + 35, 3600)))
        seq = _seq(1, _at(10), _at(30))
        cal = _calendar(seq)

        _refine(proc, cal)

        assert _off_by(seq.start_time, _at(9, 35)) < 0.01

    def test_growth_stops_at_the_previous_observation(self):
        # The first observation's last minute opens dark at 8:50, so its
        # stop stays put and the 10 s between them are the second's to take.
        proc = _processor(_SecondsVis((0, 8 * 60 + 50), (9 * 60, 3600)))
        first = _seq(1, _at(0), _at(9, 50))
        second = _seq(2, _at(10), _at(30))
        cal = _calendar(first, second)

        _refine(proc, cal)

        assert _off_by(second.start_time, _at(9, 50)) < 0.01
        assert first.stop_time <= second.start_time

    def test_pushed_past_a_dark_patch_in_the_first_minute(self):
        proc = _processor(
            _SecondsVis((0, 10 * 60 + 5), (10 * 60 + 12, 3600))
        )
        seq = _seq(1, _at(10), _at(30))
        cal = _calendar(seq)

        _refine(proc, cal)

        assert _off_by(seq.start_time, _at(10, 12)) < 0.01


class TestLeftAlone:
    def test_a_boundary_on_a_dark_minute_sample_is_not_touched(self):
        """An earlier pass kept it there; this pass does not second-guess it."""
        proc = _processor(_SecondsVis((10 * 60 + 20, 29 * 60 + 50)))
        seq = _seq(1, _at(10), _at(31))
        cal = _calendar(seq)

        _refine(proc, cal)

        assert _off_by(seq.start_time, _at(10)) < 0.01
        # The last minute opens dark at 30:00, so the stop stays too.
        assert _off_by(seq.stop_time, _at(31)) < 0.01

    def test_abutting_observations_with_no_free_time_are_unchanged(self):
        """Visible throughout, but no free second to grow into."""
        proc = _processor(_SecondsVis((0, 30 * 60)))
        cal = _calendar(
            _seq(1, _at(0), _at(10)),
            _seq(2, _at(10), _at(20)),
            _seq(3, _at(20), _at(30)),
        )
        before = _timing(cal)

        _refine(proc, cal)

        for key, (start, stop) in _timing(cal).items():
            assert _off_by(start, before[key][0]) < 0.01
            assert _off_by(stop, before[key][1]) < 0.01
        assert proc.gap_report["processing_summary"]["edge_seconds_grown"] == 0


def test_report_and_notes():
    proc = _processor(_SecondsVis((9 * 60 + 35, 30 * 60 + 25)))
    seq = _seq(1, _at(10), _at(31))
    cal = _calendar(seq)

    _refine(proc, cal)

    summary = proc.gap_report["processing_summary"]
    assert summary["edge_seconds_cut"] == 35
    assert summary["edge_seconds_grown"] == 25
    notes = proc._timing_notes[("v1", 1)]
    assert any(note.startswith("start: 25 s earlier") for note in notes)
    assert any(note.startswith("stop: 35 s earlier") for note in notes)


@pytest.mark.parametrize("refine", [True, False])
def test_pipeline_switch(monkeypatch, refine):
    """On by default; off leaves the minute-grid stop."""
    monkeypatch.setattr(
        "shortschedule.scheduler.Visibility",
        lambda *a, **kw: _SecondsVis((0, 30 * 60 + 25)),
    )
    kwargs = {} if refine else {"refine_edges_to_second": False}
    proc = ScheduleProcessor(
        "L1",
        "L2",
        st_gap_tolerance_start_buffer=0,
        earthlimb_gap_tolerance_start_buffer=0,
        **kwargs,
    )
    cal = _calendar(_seq(1, _at(10), _at(40)))

    result = proc._process_all_sequences(cal)

    stop = result.visits[0].sequences[0].stop_time
    assert _off_by(stop, _at(30, 25) if refine else _at(31)) < 0.01
