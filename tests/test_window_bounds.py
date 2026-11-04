"""Tests for the processing window's edges.

``window_start`` is hard: the time before it belongs to the previous
calendar, so nothing that started before it is scheduled and nothing grows
back into it. ``window_end`` is soft: an observation starting before it keeps
its full length, and growth may carry it into the next calendar.
"""

# Third-party
import numpy as np
from astropy import units as u

# First-party/Local
from tests.test_movement_limit import (
    T0,
    _make_calendar,
    _make_seq,
    _PatternVis,
    _processor,
    _timing,
)


def _minutes(time):
    return int(np.rint((time - T0).sec / 60.0))


class TestExtractTimeWindow:
    def test_kept_whole_when_it_starts_inside_the_window(self):
        """A start straddler is left out; an end straddler is kept whole."""
        proc = _processor()
        before = _make_seq("s0", "A", start_min=0, duration_min=20)
        start = _make_seq("s1", "B", start_min=50, duration_min=30)
        inside = _make_seq("s2", "C", start_min=100, duration_min=20)
        end = _make_seq("s3", "D", start_min=1480, duration_min=40)
        after = _make_seq("s4", "E", start_min=1500, duration_min=20)
        cal = _make_calendar([before, start, inside, end, after])

        # Window is minutes 60 to 1500.
        out = proc._extract_time_window(cal, T0 + 60 * u.min, 1, False)

        spans = [
            (seq.id, _minutes(seq.start_time), _minutes(seq.stop_time))
            for seq in out.visits[0].sequences
        ]
        assert spans == [("s2", 100, 120), ("s3", 1480, 1520)]


class TestGrowthAtTheWindowEdges:
    def test_start_is_a_hard_bound_and_end_is_not(self):
        """Visibility and the movement limit allow growth either way; only
        the start of the window stops it."""
        proc = _processor(_PatternVis(np.ones(400, dtype=bool)), limit=45)
        proc.window_start = T0 + 100 * u.min
        proc.window_end = T0 + 300 * u.min
        first = _make_seq("s1", "A", start_min=110, duration_min=20)
        last = _make_seq("s2", "B", start_min=270, duration_min=20)
        cal = _make_calendar([first, last])

        proc._grow_into_free_time(cal, _timing(cal))

        a, b = cal.visits[0].sequences
        assert _minutes(a.start_time) == 100
        # 45 min past its long-term stop at 290, beyond the window end.
        assert _minutes(b.stop_time) == 335
