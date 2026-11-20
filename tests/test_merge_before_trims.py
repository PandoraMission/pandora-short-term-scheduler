"""Tests for merging same-target observations before the trims.

The WY47 long-term calendar gave G2545911550549266944 two observations
per orbit across a minute of Free Time: one visible, one dark from start to
finish. Neither trim can touch an observation with nothing visible, so the
dark one survived and the end-of-run merge joined it to the visible one.
Merging on the long-term times first turns the dark one into a tail that
the trims cut off.
"""

# Third-party
import numpy as np
from astropy import units as u

# First-party/Local
from shortschedule.models import ScienceCalendar, Visit
from shortschedule.scheduler import ScheduleProcessor
from tests.doubles import answers_visibility
from tests.test_edge_refinement import T0, _at, _off_by, _seq

OTHER_RA = 50.0


class _LimbVis:
    """Visible inside ``[start, stop)`` minute spans after T0, per RA.

    Dark time is the Earth limb, so it is tolerable up to
    ``earthlimb_gap_tolerance``; the star trackers always pass.
    """

    _st_constraint_active = True

    def __init__(self, spans_by_ra):
        self.spans_by_ra = spans_by_ra

    def _visible(self, coord, times):
        seconds = np.round(np.atleast_1d((times - T0).sec), 3)
        out = np.zeros(seconds.shape, dtype=bool)
        for start, stop in self.spans_by_ra.get(round(coord.ra.deg, 3), ()):
            out |= (seconds >= start * 60) & (seconds < stop * 60)
        return out

    @answers_visibility
    def get_visibility(self, coord, times, roll=None):
        out = self._visible(coord, times)
        return out if np.ndim((times - T0).sec) else bool(out[0])

    def get_constraint(self, coord, body, times):
        return self._visible(coord, times)

    def get_all_constraints(self, coord, time, roll=None):
        return {
            "moon": True,
            "sun": True,
            "earthlimb": bool(self._visible(coord, time)[0]),
        }

    def get_star_tracker_breakdown(self, coord, times, roll=None, pre=None):
        passed = np.ones(np.size(times), dtype=bool)
        return {
            "passed": {"combined": passed if np.ndim(times.jd) else passed[0]}
        }


def _processor(monkeypatch, spans_by_ra, **kwargs):
    monkeypatch.setattr(
        "shortschedule.scheduler.Visibility",
        lambda *a, **kw: _LimbVis(spans_by_ra),
    )
    kwargs.setdefault("st_gap_tolerance_start_buffer", 0)
    kwargs.setdefault("earthlimb_gap_tolerance_start_buffer", 0)
    # The pointing pad has its own tests (test_pointing_pad.py).
    kwargs.setdefault("pointing_pad_before_start", 0 * u.s)
    kwargs.setdefault("pointing_pad_after_stop", 0 * u.s)
    proc = ScheduleProcessor(
        "L1", "L2", earthlimb_gap_tolerance=6, st_gap_tolerance=16, **kwargs
    )
    proc._initialize_gap_report()
    return proc


def _pair(first_span, second_span, priority=0):
    """Two observations of one target, in one visit."""
    first = _seq(1, _at(first_span[0]), _at(first_span[1]))
    second = _seq(2, _at(second_span[0]), _at(second_span[1]))
    first.target = second.target = "G2545"
    first.priority = second.priority = priority
    return first, second


def _calendar(*sequences):
    return ScienceCalendar(
        metadata={}, visits=[Visit(id="v1", sequences=list(sequences))]
    )


def _spans(calendar):
    return [
        (
            seq.id,
            round((seq.start_time - T0).sec / 60, 2),
            round((seq.stop_time - T0).sec / 60, 2),
        )
        for visit in calendar.visits
        for seq in sorted(visit.sequences, key=lambda s: s.start_time)
    ]


class TestTheWY47Case:
    """Visible 0-12, a minute of Free Time, dark 13-30."""

    def test_the_dark_observation_is_merged_and_trimmed(
        self, monkeypatch, capsys
    ):
        proc = _processor(monkeypatch, {10.0: [(0, 12)]})
        cal = _calendar(*_pair((0, 12), (13, 30)))

        result = proc._process_all_sequences(cal, merge_first=True)

        assert _spans(result) == [(1, 0.0, 12.0)]
        out = capsys.readouterr().out
        assert "MERGE (before trims): absorbing sequence 2" in out
        summary = proc.gap_report["processing_summary"]
        assert summary["merged_before_trims"] == 1
        assert summary["priority_0_dropped"] == 0

    def test_without_merging_first_the_dark_one_survives(self, monkeypatch):
        """What the WY47 run did before this change."""
        proc = _processor(monkeypatch, {10.0: [(0, 12)]})
        cal = _calendar(*_pair((0, 12), (13, 30)))

        result = proc._process_all_sequences(cal)

        assert _spans(result) == [(1, 0.0, 12.0), (2, 13.0, 30.0)]

    def test_trimmed_when_the_next_observation_cannot_take_the_time(
        self, monkeypatch
    ):
        """A neighbor after it, dark over the freed time: the tail trim
        refuses, and the longest-visible-block trim cuts it instead."""
        proc = _processor(monkeypatch, {10.0: [(0, 12)], OTHER_RA: [(30, 60)]})
        first, second = _pair((0, 12), (13, 30))
        after = _seq(3, _at(30), _at(60))
        after.target, after.ra = "TOI-544b", OTHER_RA
        after.priority = 2
        cal = ScienceCalendar(
            metadata={},
            visits=[
                Visit(id="v1", sequences=[first, second]),
                Visit(id="v2", sequences=[after]),
            ],
        )

        result = proc._process_all_sequences(cal, merge_first=True)

        assert _spans(result) == [(1, 0.0, 12.0), (3, 30.0, 60.0)]


class TestTheJoin:
    def test_a_tolerable_dark_gap_merges_into_one(self, monkeypatch):
        proc = _processor(monkeypatch, {10.0: [(0, 15), (18, 40)]})
        cal = _calendar(*_pair((0, 15), (18, 40)))

        result = proc._process_all_sequences(cal, merge_first=True)

        assert _spans(result) == [(1, 0.0, 40.0)]

    def test_a_long_dark_stretch_across_the_join_is_not_merged(
        self, monkeypatch
    ):
        """Dark 15-35, both sides visible beyond it: merged, the trims would
        keep only one side. Each side is trimmed on its own instead."""
        proc = _processor(monkeypatch, {10.0: [(0, 15), (35, 50)]})
        cal = _calendar(*_pair((0, 24), (26, 50)))

        result = proc._process_all_sequences(cal, merge_first=True)

        assert _spans(result) == [(1, 0.0, 15.0), (2, 35.0, 50.0)]
        assert (
            proc.gap_report["processing_summary"]["merged_before_trims"] == 0
        )

    def test_a_dark_head_is_merged_and_trimmed(self, monkeypatch):
        proc = _processor(monkeypatch, {10.0: [(18, 40)]})
        cal = _calendar(*_pair((0, 16), (17, 40)))

        result = proc._process_all_sequences(cal, merge_first=True)

        assert _spans(result) == [(1, 18.0, 40.0)]

    def test_not_merged_when_the_trim_would_leave_under_the_minimum(
        self, monkeypatch
    ):
        """Six visible minutes: the trim would refuse to cut the tail."""
        proc = _processor(monkeypatch, {10.0: [(0, 6)]})
        cal = _calendar(*_pair((0, 10), (11, 30)))

        merged = proc._merge_similar_observations(cal, before_trims=True)

        assert len(merged.visits[0].sequences) == 2

    def test_contiguous_pairs_are_judged_too(self, monkeypatch):
        proc = _processor(monkeypatch, {10.0: [(0, 15), (27, 45)]})
        cal = _calendar(*_pair((0, 21), (21, 45)))

        merged = proc._merge_similar_observations(cal, before_trims=True)

        assert len(merged.visits[0].sequences) == 2

    def test_the_end_of_run_merge_still_joins_contiguous_pairs(
        self, monkeypatch
    ):
        proc = _processor(monkeypatch, {10.0: [(0, 15), (27, 45)]})
        cal = _calendar(*_pair((0, 21), (21, 45)))

        merged = proc._merge_similar_observations(cal)

        assert len(merged.visits[0].sequences) == 1


def test_nothing_is_dropped_before_the_trims(monkeypatch, capsys):
    """drop_priority0's filler drop belongs to the end-of-run merge."""
    proc = _processor(
        monkeypatch,
        {10.0: [(0, 60)], OTHER_RA: [(0, 60)]},
        drop_priority0=True,
    )
    first, second = _pair((0, 20), (25, 45), priority=1)
    filler = _seq(3, _at(20), _at(25))
    filler.ra, filler.priority = OTHER_RA, 0
    cal = _calendar(first, filler, second)

    early = proc._merge_similar_observations(cal, before_trims=True)
    late = proc._merge_similar_observations(cal)

    assert len(early.visits[0].sequences) == 3
    assert (
        "dropped priority-0" not in capsys.readouterr().out.split("Merged")[0]
    )
    assert len(late.visits[0].sequences) == 1


def test_the_merged_span_is_the_long_term_time(monkeypatch, capsys):
    """The timing log measures the trim from the merged span."""
    proc = _processor(monkeypatch, {10.0: [(0, 12)]})
    cal = _calendar(*_pair((0, 12), (13, 30)))

    proc._process_all_sequences(cal, merge_first=True)

    out = capsys.readouterr().out
    assert "SHRANK: stop earlier 18.0 min (duration 30.0 -> 12.0 min)" in out
    assert _off_by(cal.visits[0].sequences[1].stop_time, _at(30)) < 0.01


def test_the_main_target_flag_survives_the_trims(monkeypatch):
    """The passes rebuild the observations they move; each rebuild must
    keep its main-target flag, or the next pass judges it by the wrong
    keepouts. Here a dark tail is cut and the next observation takes the
    freed minutes."""
    proc = _processor(monkeypatch, {10.0: [(0, 12)], OTHER_RA: [(10, 60)]})
    first = _seq(1, _at(0), _at(20))
    after = _seq(2, _at(20), _at(60))
    after.ra = OTHER_RA
    first.is_main = after.is_main = True
    cal = ScienceCalendar(
        metadata={},
        visits=[
            Visit(id="v1", sequences=[first]),
            Visit(id="v2", sequences=[after]),
        ],
    )

    result = proc._process_all_sequences(cal)

    assert _spans(result) == [(1, 0.0, 12.0), (2, 12.0, 60.0)]
    assert all(
        seq.is_main for visit in result.visits for seq in visit.sequences
    )


def test_the_flag_survives_the_neighbors_of_a_block_trim(monkeypatch):
    """Cut to its longest visible block, an observation hands its dark
    head and tail to the neighbors, which are rebuilt longer."""
    proc = _processor(
        monkeypatch,
        {OTHER_RA: [(0, 30)], 10.0: [(30, 50)], 70.0: [(45, 90)]},
    )
    before = _seq(1, _at(0), _at(20))
    middle = _seq(2, _at(20), _at(60))
    after = _seq(3, _at(60), _at(90))
    before.ra, after.ra = OTHER_RA, 70.0
    cal = ScienceCalendar(
        metadata={},
        visits=[
            Visit(id=f"v{seq.id}", sequences=[seq])
            for seq in (before, middle, after)
        ],
    )
    for seq in (before, middle, after):
        seq.is_main = True

    result = proc._trim_to_longest_visible_block(cal)

    assert _spans(result) == [(1, 0.0, 30.0), (2, 30.0, 50.0), (3, 50.0, 90.0)]
    assert all(
        seq.is_main for visit in result.visits for seq in visit.sequences
    )
