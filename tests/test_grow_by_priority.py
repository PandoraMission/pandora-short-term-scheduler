"""Tests for grow_by_priority.

Higher priorities grow first and may take minutes from an adjacent
lower-priority observation, down to that observation's floor. The doubles
and helpers come from the movement-limit tests, whose growth pass this
extends.
"""

# Third-party
import numpy as np

# First-party/Local
from tests.test_movement_limit import (
    T0,
    _make_calendar,
    _make_seq,
    _PatternVis,
    _processor,
    _timing,
)


def _seq(sid, target, start_min, duration_min, priority):
    seq = _make_seq(
        sid, target, start_min=start_min, duration_min=duration_min
    )
    seq.priority = priority
    return seq


def _proc(pattern=None, by_priority=True, limit=45):
    if pattern is None:
        pattern = np.ones(300, dtype=bool)
    proc = _processor(_PatternVis(pattern), limit=limit)
    proc.grow_by_priority = by_priority
    return proc


def _minute(time):
    return int(np.rint((time - T0).sec / 60.0))


def _grow(proc, sequences):
    cal = _make_calendar(sequences)
    proc._grow_into_free_time(cal, _timing(cal))
    return cal.visits[0].sequences


def test_higher_priority_takes_from_a_lower_neighbor_down_to_its_floor(
    capsys,
):
    """Priority 2 eats into the priority 1 after it until that one is at
    the minimum duration; the refusal beyond that is reported."""
    proc = _proc()
    high, low = _grow(
        proc, [_seq("s1", "A", 0, 20, 2), _seq("s2", "B", 20, 40, 1)]
    )

    # B kept 8 min (its floor); A took the other 32 and stopped there, though
    # visibility and its own 45 min limit would have allowed 65.
    assert (_minute(high.stop_time), _minute(low.start_time)) == (52, 52)
    assert _minute(low.stop_time) == 105  # B still grew on its free side
    summary = proc.gap_report["processing_summary"]
    assert summary["minutes_taken_from_lower_priority"] == 32
    out = capsys.readouterr().out
    assert "GROWTH: took 32 min from lower-priority B" in out
    assert "ERROR" in out and "GROWTH BLOCKED: visibility allows 13" in out


def test_walk_order_is_priority_then_start_time(monkeypatch):
    proc = _proc()
    seen = []
    monkeypatch.setattr(
        proc,
        "_grow_one_side",
        lambda ordered, index, direction, timing: (
            seen.append((ordered[index][1].id, direction)) or (0, 0)
        ),
    )
    _grow(
        proc,
        [
            _seq("p0", "A", 0, 10, 0),
            _seq("p2b", "B", 20, 10, 2),
            _seq("p1", "C", 40, 10, 1),
            _seq("p2a", "D", 60, 10, 2),
        ],
    )
    assert [sid for sid, _ in seen[::2]] == ["p2b", "p2a", "p1", "p0"]
    assert [direction for _, direction in seen] == [-1, 1] * 4


def test_equal_priority_neighbor_is_a_hard_bound(capsys):
    proc = _proc()
    first, second = _grow(
        proc, [_seq("s1", "A", 0, 20, 1), _seq("s2", "B", 20, 40, 1)]
    )
    assert (_minute(first.stop_time), _minute(second.start_time)) == (20, 20)
    assert (
        proc.gap_report["processing_summary"][
            "minutes_taken_from_lower_priority"
        ]
        == 0
    )
    assert "GROWTH" not in capsys.readouterr().out


def test_flag_off_keeps_hard_bounds(capsys):
    proc = _proc(by_priority=False)
    high, low = _grow(
        proc, [_seq("s1", "A", 0, 20, 2), _seq("s2", "B", 20, 40, 1)]
    )
    assert (_minute(high.stop_time), _minute(low.start_time)) == (20, 20)
    assert "GROWTH" not in capsys.readouterr().out


def test_visibility_still_limits_the_take(capsys):
    """Dark minutes stop growth before the neighbor's floor does, and
    that is not a blocked-growth error."""
    pattern = np.zeros(300, dtype=bool)
    pattern[:30] = True
    proc = _proc(pattern)
    high, low = _grow(
        proc, [_seq("s1", "A", 0, 20, 2), _seq("s2", "B", 20, 40, 1)]
    )
    assert (_minute(high.stop_time), _minute(low.start_time)) == (30, 30)
    assert "GROWTH BLOCKED" not in capsys.readouterr().out


def test_own_movement_limit_binding_is_not_reported(capsys):
    proc = _proc(limit=10)
    high, low = _grow(
        proc, [_seq("s1", "A", 0, 20, 2), _seq("s2", "B", 20, 100, 1)]
    )
    assert (_minute(high.stop_time), _minute(low.start_time)) == (30, 30)
    assert "GROWTH BLOCKED" not in capsys.readouterr().out


def test_taking_from_the_previous_neighbor_moves_its_stop():
    proc = _proc()
    low, high = _grow(
        proc, [_seq("s1", "A", 0, 40, 0), _seq("s2", "B", 40, 20, 2)]
    )
    # A keeps its 8 min floor; B's start moved 32 min earlier onto it.
    assert (_minute(low.stop_time), _minute(high.start_time)) == (8, 8)
    assert _minute(low.start_time) == 0
    assert not proc._below_minimum_duration(low.stop_time - low.start_time)


def test_take_leaves_room_for_the_neighbor_to_clean_its_opening():
    """A neighbor that opens dark keeps its minimum measured from where
    the start-buffer pass will move its start, not from its dark start."""
    pattern = np.ones(300, dtype=bool)
    pattern[0] = False  # A's first minute is dark
    proc = _proc(pattern)
    proc.earthlimb_gap_tolerance_start_buffer = 5
    low, high = _grow(
        proc, [_seq("s1", "A", 0, 40, 0), _seq("s2", "B", 40, 20, 2)]
    )
    # The buffer pass will trim A's dark minute, so A keeps 9 to hold 8.
    assert (_minute(low.stop_time), _minute(high.start_time)) == (9, 9)


def test_take_at_a_start_backs_off_until_the_opening_can_be_cleaned():
    """Pushing a neighbor's start onto a dark stretch would have the
    buffer pass move it past that stretch, so the take stops short of it."""
    pattern = np.ones(300, dtype=bool)
    pattern[36:40] = False  # a dark stretch inside B
    proc = _proc(pattern)
    proc.earthlimb_gap_tolerance_start_buffer = 5
    high, low = _grow(
        proc, [_seq("s1", "A", 0, 20, 2), _seq("s2", "B", 20, 25, 1)]
    )
    # From a start at 32 or later the 5 min window touches the dark
    # stretch and the buffer pass would land at 40, leaving B 5 min. From
    # 31 the window is clean and B keeps 14, so that is where the take
    # stops, though B's bare floor would have allowed 37.
    assert (_minute(high.stop_time), _minute(low.start_time)) == (31, 31)


def test_timing_log_says_why_each_boundary_moved(capsys):
    proc = _proc()
    cal = _make_calendar(
        [_seq("s1", "A", 0, 20, 2), _seq("s2", "B", 20, 40, 1)]
    )
    original = _timing(cal)
    proc._grow_into_free_time(cal, original)
    capsys.readouterr()

    proc._log_timing_changes(cal, original)
    out = capsys.readouterr().out
    assert (
        "ELONGATED: stop later 32.0 min (duration 20.0 -> 52.0 min): "
        "stop: grew 32 min, 32 of them taken from lower-priority B"
    ) in out
    assert (
        "ELONGATED: start later 32.0 min, stop later 45.0 min "
        "(duration 40.0 -> 53.0 min): start: gave 32 min to "
        "higher-priority A; stop: grew 45 min into idle time"
    ) in out
