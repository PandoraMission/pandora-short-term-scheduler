"""Tests for the trimming passes and roll-aware visibility.

Covers:
- _find_false_blocks helper
- trimming non-visible tails, heads, and mid-observation dark stretches
- the gap tolerances, judged at the roll the observation will fly
- the star-tracker and Earth-limb start buffers
"""

# Third-party
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

# First-party/Local
from shortschedule.models import ObservationSequence, ScienceCalendar, Visit
from shortschedule.scheduler import ScheduleProcessor, _find_false_blocks

# ================================================================
# Helpers
# ================================================================

T0 = Time("2026-01-01T00:00:00", scale="utc")


def _make_seq(sid, target, start_min, duration_min, ra=10.0, dec=20.0):
    """Create an ObservationSequence starting *start_min* after T0."""
    start = T0 + start_min * u.min
    stop = start + duration_min * u.min
    return ObservationSequence(
        id=sid,
        target=target,
        priority=1,
        start_time=start,
        stop_time=stop,
        ra=ra,
        dec=dec,
        payload_params={},
    )


def _make_calendar(sequences, visit_id="v1"):
    """Wrap a list of sequences into a single-visit calendar."""
    visit = Visit(id=visit_id, sequences=sequences)
    return ScienceCalendar(metadata={}, visits=[visit])


def _make_time_grid(n_minutes):
    """Minute-resolution time grid starting at T0."""
    return T0 + np.arange(n_minutes) * u.min


# ================================================================
# DummyVisibility mocks
# ================================================================


class DummyVisibilityAllTrue:
    """Visibility mock — always visible, ignores roll."""

    def __init__(self, l1, l2, **kwargs):
        pass

    def get_visibility(self, coord, times, roll=None):
        try:
            n = len(times)
        except Exception:
            return np.array([True], dtype=bool)
        return np.ones(n, dtype=bool)


class DummyVisibilityPattern:
    """Visibility mock driven by a minute-indexed boolean array.

    Parameters
    ----------
    pattern : np.ndarray[bool]
        One bool per minute starting at T0.  Minutes outside the
        array are treated as ``True``.
    """

    def __init__(self, l1, l2, pattern=None, **kwargs):
        self._pattern = (
            pattern if pattern is not None else np.array([], dtype=bool)
        )

    def get_visibility(self, coord, times, roll=None):
        try:
            n = len(times)
        except Exception:
            n = 1
            times = Time([times])
        result = np.ones(n, dtype=bool)
        for i, t in enumerate(times):
            idx = int(np.rint((t - T0).sec / 60.0))
            if 0 <= idx < len(self._pattern):
                result[i] = self._pattern[idx]
        return result


class TestFindFalseBlocks:
    """Unit tests for the _find_false_blocks helper."""

    def test_all_true(self):
        grid = _make_time_grid(5)
        blocks = _find_false_blocks(np.ones(5, dtype=bool), grid)
        assert blocks == []

    def test_all_false(self):
        grid = _make_time_grid(5)
        blocks, idx = _find_false_blocks(
            np.zeros(5, dtype=bool), grid, return_index=True
        )
        assert len(blocks) == 1
        assert idx == [(0, 5)]

    def test_single_false_middle(self):
        vis = np.array([True, True, False, False, True, True], dtype=bool)
        grid = _make_time_grid(6)
        blocks, idx = _find_false_blocks(vis, grid, return_index=True)
        assert len(idx) == 1
        assert idx[0] == (2, 4)

    def test_multiple_disjoint(self):
        vis = np.array(
            [False, True, True, False, False, True, False],
            dtype=bool,
        )
        grid = _make_time_grid(7)
        _, idx = _find_false_blocks(vis, grid, return_index=True)
        assert len(idx) == 3
        assert idx[0] == (0, 1)
        assert idx[1] == (3, 5)
        assert idx[2] == (6, 7)

    def test_false_at_start(self):
        vis = np.array([False, False, True, True], dtype=bool)
        grid = _make_time_grid(4)
        _, idx = _find_false_blocks(vis, grid, return_index=True)
        assert idx[0] == (0, 2)

    def test_false_at_end(self):
        vis = np.array([True, True, False, False], dtype=bool)
        grid = _make_time_grid(4)
        _, idx = _find_false_blocks(vis, grid, return_index=True)
        assert idx[0] == (2, 4)

    def test_single_element_true(self):
        grid = _make_time_grid(1)
        blocks = _find_false_blocks(np.array([True], dtype=bool), grid)
        assert blocks == []

    def test_single_element_false(self):
        grid = _make_time_grid(1)
        _, idx = _find_false_blocks(
            np.array([False], dtype=bool), grid, return_index=True
        )
        assert idx == [(0, 1)]

    def test_empty_array(self):
        result = _find_false_blocks(np.array([], dtype=bool), [])
        assert result == []


class TestTrimNonVisibleTails:
    """Unit tests for ScheduleProcessor._trim_non_visible_tails."""

    def _make_processor(self, visibility_cls):
        proc = ScheduleProcessor.__new__(ScheduleProcessor)
        proc.min_sequence_duration = TimeDelta(8 * 60 * u.s)
        proc._roll_sweep_enabled = False
        proc._computed_target_rolls = {}
        proc.visibility = visibility_cls
        proc.earthlimb_gap_tolerance = 0
        proc.st_gap_tolerance = 0
        proc.gap_report = {
            "visibility_gaps": [],
            "processing_summary": {},
        }
        return proc

    def test_no_tail_no_change(self):
        """All-visible sequence is untouched."""
        dummy = DummyVisibilityAllTrue("L1", "L2")
        proc = self._make_processor(dummy)

        seq = _make_seq("s1", "T", start_min=0, duration_min=20)
        cal = _make_calendar([seq])
        result = proc._trim_non_visible_tails(cal)

        out = result.visits[0].sequences[0]
        assert out.stop_time == seq.stop_time

    def test_tail_trimmed(self):
        """Non-visible tail is trimmed to last visible minute +1."""
        # Minutes 0-17 visible, 18-19 non-visible
        pattern = np.ones(30, dtype=bool)
        pattern[18:20] = False
        dummy = DummyVisibilityPattern("L1", "L2", pattern=pattern)
        proc = self._make_processor(dummy)

        seq = _make_seq("s1", "T", start_min=0, duration_min=20)
        cal = _make_calendar([seq])
        result = proc._trim_non_visible_tails(cal)

        out = result.visits[0].sequences[0]
        expected_stop = T0 + 18 * u.min
        assert abs((out.stop_time - expected_stop).sec) < 1

    def test_next_sequence_extended_backward(self):
        """After trimming, next seq extends backward if visible."""
        # Seq A: minutes 0-19, RA=10 → tail at 18-19 non-visible
        # Seq B: minutes 20-39, RA=50 → all visible
        # Per-target mock: False at 18-19 only for RA≈10
        pattern_a = np.ones(40, dtype=bool)
        pattern_a[18:20] = False

        class _PerTargetVis:
            def __init__(self, *a, **kw):
                pass

            def get_visibility(self, coord, times, roll=None):
                n = len(times)
                if abs(coord.ra.deg - 10.0) < 1.0:
                    result = np.ones(n, dtype=bool)
                    for i, t in enumerate(times):
                        idx = int(np.rint((t - T0).sec / 60.0))
                        if 0 <= idx < len(pattern_a):
                            result[i] = pattern_a[idx]
                    return result
                return np.ones(n, dtype=bool)

        proc = self._make_processor(_PerTargetVis("L1", "L2"))

        seqA = _make_seq(
            "sA", "TargetA", start_min=0, duration_min=20, ra=10.0
        )
        seqB = _make_seq(
            "sB", "TargetB", start_min=20, duration_min=20, ra=50.0
        )
        cal = _make_calendar([seqA, seqB])
        result = proc._trim_non_visible_tails(cal)

        outB = result.visits[0].sequences[1]
        # Seq B should extend backward to fill the 2-minute gap
        expected_start = T0 + 18 * u.min
        assert abs((outB.start_time - expected_start).sec) < 1

    def test_skip_if_trimming_too_short(self):
        """Sequence not trimmed if result would be < min_sequence_duration."""
        # 10-minute seq, minutes 2-9 non-visible (only minutes 0-1 visible)
        # Trimming would leave 2 minutes < 8 min minimum
        pattern = np.ones(20, dtype=bool)
        pattern[2:10] = False
        dummy = DummyVisibilityPattern("L1", "L2", pattern=pattern)
        proc = self._make_processor(dummy)

        seq = _make_seq("s1", "T", start_min=0, duration_min=10)
        cal = _make_calendar([seq])
        result = proc._trim_non_visible_tails(cal)

        out = result.visits[0].sequences[0]
        # Should remain unchanged
        assert out.stop_time == seq.stop_time

    def test_entirely_non_visible_skipped(self):
        """Entirely non-visible sequence is not modified."""
        pattern = np.zeros(20, dtype=bool)
        dummy = DummyVisibilityPattern("L1", "L2", pattern=pattern)
        proc = self._make_processor(dummy)

        seq = _make_seq("s1", "T", start_min=0, duration_min=20)
        cal = _make_calendar([seq])
        result = proc._trim_non_visible_tails(cal)

        out = result.visits[0].sequences[0]
        assert out.stop_time == seq.stop_time

    def test_trim_no_gap_when_next_cannot_absorb(self):
        """Tail trim must not create a gap when next target can't absorb."""
        # Seq A: minutes 0-19, tail at 15-19 non-visible
        # Seq B: minutes 20-39, but next target NOT visible for gap
        # Expected: A should NOT be trimmed (would create gap)
        pattern_a = np.ones(40, dtype=bool)
        pattern_a[15:20] = False

        class _NeitherVisibleInGap:
            def __init__(self, *a, **kw):
                pass

            def get_visibility(self, coord, times, roll=None):
                n = len(times)
                if abs(coord.ra.deg - 10.0) < 1.0:
                    result = np.ones(n, dtype=bool)
                    for i, t in enumerate(times):
                        idx = int(np.rint((t - T0).sec / 60.0))
                        if 0 <= idx < len(pattern_a):
                            result[i] = pattern_a[idx]
                    return result
                # Next target also NOT visible in gap region
                result = np.ones(n, dtype=bool)
                for i, t in enumerate(times):
                    idx = int(np.rint((t - T0).sec / 60.0))
                    if 15 <= idx < 20:
                        result[i] = False
                return result

        proc = self._make_processor(_NeitherVisibleInGap("L1", "L2"))
        seqA = _make_seq(
            "sA", "TargetA", start_min=0, duration_min=20, ra=10.0
        )
        seqB = _make_seq(
            "sB", "TargetB", start_min=20, duration_min=20, ra=50.0
        )
        cal = _make_calendar([seqA, seqB])
        result = proc._trim_non_visible_tails(cal)

        outA = result.visits[0].sequences[0]
        outB = result.visits[0].sequences[1]
        # No gap: A.stop must equal B.start
        gap_sec = abs((outB.start_time - outA.stop_time).sec)
        assert gap_sec < 1, f"Gap of {gap_sec:.0f}s created"


class TestTrimToLongestVisibleBlock:
    """Unit tests for ScheduleProcessor._trim_to_longest_visible_block."""

    def _make_processor(self, visibility_cls):
        proc = ScheduleProcessor.__new__(ScheduleProcessor)
        proc.min_sequence_duration = TimeDelta(8 * 60 * u.s)
        proc._roll_sweep_enabled = False
        proc._computed_target_rolls = {}
        proc.visibility = visibility_cls
        proc.earthlimb_gap_tolerance = 0
        proc.st_gap_tolerance = 0
        proc.gap_report = {
            "visibility_gaps": [],
            "processing_summary": {},
        }
        return proc

    def test_dark_head_trimmed(self):
        """A dark head is trimmed away.

        This pass is the only thing that trims a non-visible head now that
        the dedicated head pass is gone, because the span it selects has
        its leading dark minutes stripped.
        """
        pattern = np.ones(40, dtype=bool)
        pattern[0:6] = False

        class _HeadDarkVis:
            def __init__(self, *a, **kw):
                pass

            def get_visibility(self, coord, times, roll=None):
                indices = np.rint((times - T0).sec / 60.0).astype(int)
                return np.array(
                    [
                        bool(pattern[i]) if 0 <= i < len(pattern) else True
                        for i in np.atleast_1d(indices)
                    ],
                    dtype=bool,
                )

            def get_all_constraints(self, coord, time, roll=None):
                return {"moon": True, "sun": True, "earthlimb": False}

        proc = self._make_processor(_HeadDarkVis("L1", "L2"))

        seq = _make_seq("s1", "T", start_min=0, duration_min=30)
        cal = _make_calendar([seq])
        result = proc._trim_to_longest_visible_block(cal)

        out = result.visits[0].sequences[0]
        assert abs((out.start_time - (T0 + 6 * u.min)).sec) < 1
        assert abs((out.stop_time - (T0 + 30 * u.min)).sec) < 1

    def test_all_visible_no_change(self):
        """All-visible sequence is untouched."""
        dummy = DummyVisibilityAllTrue("L1", "L2")
        proc = self._make_processor(dummy)

        seq = _make_seq("s1", "T", start_min=0, duration_min=30)
        cal = _make_calendar([seq])
        result = proc._trim_to_longest_visible_block(cal)

        out = result.visits[0].sequences[0]
        assert out.start_time == seq.start_time
        assert out.stop_time == seq.stop_time

    def test_mid_sequence_gap_trimmed(self):
        """Non-visible minutes in the middle are removed.

        Pattern: 0-19 visible, 20-24 NOT visible, 25-39 visible.
        Longest block is 0-19 (20 min) vs 25-39 (15 min) -> keep 0-19.
        """
        pattern = np.ones(40, dtype=bool)
        pattern[20:25] = False
        dummy = DummyVisibilityPattern("L1", "L2", pattern=pattern)
        proc = self._make_processor(dummy)

        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])
        result = proc._trim_to_longest_visible_block(cal)

        out = result.visits[0].sequences[0]
        expected_start = T0
        expected_stop = T0 + 20 * u.min
        assert abs((out.start_time - expected_start).sec) < 1
        assert abs((out.stop_time - expected_stop).sec) < 1

    def test_selects_longest_block(self):
        """When multiple visible blocks exist, longest is selected.

        Pattern: 0-9 visible, 10-14 dark, 15-34 visible, 35-39 dark.
        Longest block is 15-34 (20 min).
        """
        pattern = np.ones(40, dtype=bool)
        pattern[10:15] = False
        pattern[35:40] = False
        dummy = DummyVisibilityPattern("L1", "L2", pattern=pattern)
        proc = self._make_processor(dummy)

        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])
        result = proc._trim_to_longest_visible_block(cal)

        out = result.visits[0].sequences[0]
        expected_start = T0 + 15 * u.min
        expected_stop = T0 + 35 * u.min
        assert abs((out.start_time - expected_start).sec) < 1
        assert abs((out.stop_time - expected_stop).sec) < 1

    def test_entirely_non_visible_skipped(self):
        """Entirely non-visible sequence is left as-is."""
        pattern = np.zeros(30, dtype=bool)
        dummy = DummyVisibilityPattern("L1", "L2", pattern=pattern)
        proc = self._make_processor(dummy)

        seq = _make_seq("s1", "T", start_min=0, duration_min=30)
        cal = _make_calendar([seq])
        result = proc._trim_to_longest_visible_block(cal)

        out = result.visits[0].sequences[0]
        assert out.start_time == seq.start_time
        assert out.stop_time == seq.stop_time

    def test_skip_if_longest_block_too_short(self):
        """If longest visible block < min_sequence_duration, skip."""
        # 20-min seq, only 5 min visible -> too short (min is 8)
        pattern = np.zeros(20, dtype=bool)
        pattern[5:10] = True  # 5-min visible block
        dummy = DummyVisibilityPattern("L1", "L2", pattern=pattern)
        proc = self._make_processor(dummy)

        seq = _make_seq("s1", "T", start_min=0, duration_min=20)
        cal = _make_calendar([seq])
        result = proc._trim_to_longest_visible_block(cal)

        out = result.visits[0].sequences[0]
        # Should be unchanged since longest visible block < 8 min
        assert out.start_time == seq.start_time
        assert out.stop_time == seq.stop_time

    def test_prev_seq_extended_forward(self):
        """Previous sequence extends into freed gap at start."""

        # Seq A: minutes 0-19, RA=10 (all visible)
        # Seq B: minutes 20-49, RA=50
        #   B pattern: 20-24 dark, 25-44 visible, 45-49 dark
        #   -> trimmed to 25-44
        # Prev (A at RA=10) should extend forward into 20-24
        class _PerTargetVis:
            def __init__(self, *a, **kw):
                pass

            def get_visibility(self, coord, times, roll=None):
                n = len(times)
                result = np.ones(n, dtype=bool)
                if abs(coord.ra.deg - 50.0) < 1.0:
                    # Target B: dark at 20-24, 45-49
                    for i, t in enumerate(times):
                        idx = int(np.rint((t - T0).sec / 60.0))
                        if 20 <= idx < 25 or 45 <= idx < 50:
                            result[i] = False
                # Target A: always visible
                return result

        proc = self._make_processor(_PerTargetVis("L1", "L2"))
        seqA = _make_seq(
            "sA", "TargetA", start_min=0, duration_min=20, ra=10.0
        )
        seqB = _make_seq(
            "sB", "TargetB", start_min=20, duration_min=30, ra=50.0
        )
        cal = _make_calendar([seqA, seqB])
        result = proc._trim_to_longest_visible_block(cal)

        outA = result.visits[0].sequences[0]
        outB = result.visits[0].sequences[1]

        # B should be trimmed to 25-45
        assert abs((outB.start_time - (T0 + 25 * u.min)).sec) < 1
        # A should extend forward (into 20-24 since A is visible)
        assert outA.stop_time > T0 + 20 * u.min

    def test_next_seq_extended_backward(self):
        """Next sequence extends backward into freed gap at tail."""

        # Seq A: minutes 0-29, RA=10
        #   A pattern: 0-19 visible, 20-24 dark, 25-29 visible
        #   -> trimmed to 0-19 (longest block = 20 min)
        # Seq B: minutes 30-49, RA=50 (all visible)
        # B should extend backward into 20-29 if B is visible
        class _PerTargetVis:
            def __init__(self, *a, **kw):
                pass

            def get_visibility(self, coord, times, roll=None):
                n = len(times)
                result = np.ones(n, dtype=bool)
                if abs(coord.ra.deg - 10.0) < 1.0:
                    # Target A: dark at 20-24
                    for i, t in enumerate(times):
                        idx = int(np.rint((t - T0).sec / 60.0))
                        if 20 <= idx < 25:
                            result[i] = False
                # Target B (RA=50): always visible
                return result

        proc = self._make_processor(_PerTargetVis("L1", "L2"))
        seqA = _make_seq(
            "sA", "TargetA", start_min=0, duration_min=30, ra=10.0
        )
        seqB = _make_seq(
            "sB", "TargetB", start_min=30, duration_min=20, ra=50.0
        )
        cal = _make_calendar([seqA, seqB])
        result = proc._trim_to_longest_visible_block(cal)

        outA = result.visits[0].sequences[0]
        outB = result.visits[0].sequences[1]

        # A should be trimmed to 0-19 (longest visible block)
        assert abs((outA.stop_time - (T0 + 20 * u.min)).sec) < 1
        # B should extend backward (B visible at 20-29)
        assert outB.start_time < T0 + 30 * u.min

    def test_tolerable_gap_kept(self):
        """Short gap within tolerance is kept — no trimming."""
        # 40-min seq, 2-min dark gap at 20-21 (earthlimb failure)
        pattern = np.ones(40, dtype=bool)
        pattern[20:22] = False

        class _EarthlimbFailVis:
            def __init__(self, *a, **kw):
                pass

            def get_visibility(self, coord, times, roll=None):
                n = len(times)
                result = np.ones(n, dtype=bool)
                for i, t in enumerate(times):
                    idx = int(np.rint((t - T0).sec / 60.0))
                    if 0 <= idx < len(pattern):
                        result[i] = pattern[idx]
                return result

            def get_all_constraints(self, coord, time, roll=None):
                return {
                    "moon": True,
                    "sun": True,
                    "earthlimb": False,
                }

        proc = self._make_processor(_EarthlimbFailVis("L1", "L2"))
        proc.earthlimb_gap_tolerance = 2  # tolerate up to 2 min

        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])
        result = proc._trim_to_longest_visible_block(cal)

        out = result.visits[0].sequences[0]
        # Sequence should remain unchanged — gap is tolerable
        assert out.start_time == seq.start_time
        assert out.stop_time == seq.stop_time

    def test_intolerable_gap_trimmed(self):
        """Gap exceeding tolerance is trimmed despite tolerance > 0."""
        # 40-min seq, 5-min dark gap at 15-19 (earthlimb)
        pattern = np.ones(40, dtype=bool)
        pattern[15:20] = False

        class _EarthlimbFailVis:
            def __init__(self, *a, **kw):
                pass

            def get_visibility(self, coord, times, roll=None):
                n = len(times)
                result = np.ones(n, dtype=bool)
                for i, t in enumerate(times):
                    idx = int(np.rint((t - T0).sec / 60.0))
                    if 0 <= idx < len(pattern):
                        result[i] = pattern[idx]
                return result

            def get_all_constraints(self, coord, time, roll=None):
                return {
                    "moon": True,
                    "sun": True,
                    "earthlimb": False,
                }

        proc = self._make_processor(_EarthlimbFailVis("L1", "L2"))
        proc.earthlimb_gap_tolerance = 2  # 5-min gap > 2 min tol

        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])
        result = proc._trim_to_longest_visible_block(cal)

        out = result.visits[0].sequences[0]
        # Should be trimmed — 5-min gap exceeds 2-min tolerance
        # Longest segment: 20-39 (20 min) vs 0-14 (15 min)
        assert abs((out.start_time - (T0 + 20 * u.min)).sec) < 1
        assert abs((out.stop_time - (T0 + 40 * u.min)).sec) < 1

    def test_st_gap_tolerance(self):
        """Star-tracker gap within st_gap_tolerance is tolerated."""
        # 30-min seq, 2-min dark at 10-11 (star tracker failure)
        pattern = np.ones(30, dtype=bool)
        pattern[10:12] = False

        class _STFailVis:
            def __init__(self, *a, **kw):
                pass

            def get_visibility(self, coord, times, roll=None):
                n = len(times)
                result = np.ones(n, dtype=bool)
                for i, t in enumerate(times):
                    idx = int(np.rint((t - T0).sec / 60.0))
                    if 0 <= idx < len(pattern):
                        result[i] = pattern[idx]
                return result

            def get_all_constraints(self, coord, time, roll=None):
                # Boresight constraints pass; ST fails
                return {
                    "moon": True,
                    "sun": True,
                    "earthlimb": True,
                }

            def get_star_tracker_breakdown(
                self, coord, time, roll=None, pre=None
            ):
                # The dark minutes are exactly the tracker failures.
                idx = int(np.rint((time - T0).sec / 60.0))
                ok = pattern[idx] if 0 <= idx < len(pattern) else True
                return {"passed": {"combined": bool(ok)}}

        proc = self._make_processor(_STFailVis("L1", "L2"))
        proc.st_gap_tolerance = 2

        seq = _make_seq("s1", "T", start_min=0, duration_min=30)
        cal = _make_calendar([seq])
        result = proc._trim_to_longest_visible_block(cal)

        out = result.visits[0].sequences[0]
        # Gap is ST-only, 2 min <= 2 min tol → tolerated
        assert out.start_time == seq.start_time
        assert out.stop_time == seq.stop_time

    def test_mixed_tolerable_and_intolerable(self):
        """Sequence with one tolerable and one intolerable gap."""
        # 60-min seq:
        #   0-19 visible, 20-21 dark (tolerable, earthlimb, 2 min)
        #   22-44 visible, 45-49 dark (intolerable, earthlimb, 5 min)
        #   50-59 visible
        pattern = np.ones(60, dtype=bool)
        pattern[20:22] = False
        pattern[45:50] = False

        class _EarthlimbFailVis:
            def __init__(self, *a, **kw):
                pass

            def get_visibility(self, coord, times, roll=None):
                n = len(times)
                result = np.ones(n, dtype=bool)
                for i, t in enumerate(times):
                    idx = int(np.rint((t - T0).sec / 60.0))
                    if 0 <= idx < len(pattern):
                        result[i] = pattern[idx]
                return result

            def get_all_constraints(self, coord, time, roll=None):
                return {
                    "moon": True,
                    "sun": True,
                    "earthlimb": False,
                }

        proc = self._make_processor(_EarthlimbFailVis("L1", "L2"))
        proc.earthlimb_gap_tolerance = 2

        seq = _make_seq("s1", "T", start_min=0, duration_min=60)
        cal = _make_calendar([seq])
        result = proc._trim_to_longest_visible_block(cal)

        out = result.visits[0].sequences[0]
        # The first segment (0-44, containing only the tolerable gap)
        # is longer (45 min) than the second segment (50-59, 10 min).
        # Trimmed to visible bounds within that segment: 0-44.
        assert abs((out.start_time - T0).sec) < 1
        assert abs((out.stop_time - (T0 + 45 * u.min)).sec) < 1


# ================================================================
# Tests: tolerance at sequence tails
# ================================================================


class TestToleranceAtHeadsAndTails:
    """Verify gap tolerances are applied at sequence boundaries."""

    def _make_processor(self, visibility_cls):
        proc = ScheduleProcessor.__new__(ScheduleProcessor)
        proc.min_sequence_duration = TimeDelta(8 * 60 * u.s)
        proc._roll_sweep_enabled = False
        proc._computed_target_rolls = {}
        proc.visibility = visibility_cls
        proc.earthlimb_gap_tolerance = 0
        proc.st_gap_tolerance = 0
        proc.gap_report = {
            "visibility_gaps": [],
            "processing_summary": {},
        }
        return proc

    def test_tail_within_tolerance_not_trimmed(self):
        """Trailing non-visible minutes within tolerance are kept."""
        # 20-min seq, last 2 min non-visible (earthlimb)
        pattern = np.ones(30, dtype=bool)
        pattern[18:20] = False

        class _ELVis:
            def __init__(self, *a, **kw):
                pass

            def get_visibility(self, coord, times, roll=None):
                n = len(times)
                result = np.ones(n, dtype=bool)
                for i, t in enumerate(times):
                    idx = int(np.rint((t - T0).sec / 60.0))
                    if 0 <= idx < len(pattern):
                        result[i] = pattern[idx]
                return result

            def get_all_constraints(self, coord, time, roll=None):
                return {
                    "moon": True,
                    "sun": True,
                    "earthlimb": False,
                }

        proc = self._make_processor(_ELVis("L1", "L2"))
        proc.earthlimb_gap_tolerance = 2  # 2-min tail <= 2 tol

        seq = _make_seq("s1", "T", start_min=0, duration_min=20)
        cal = _make_calendar([seq])
        result = proc._trim_non_visible_tails(cal)

        out = result.visits[0].sequences[0]
        # Tail should NOT be trimmed — within tolerance
        assert out.stop_time == seq.stop_time

    def test_tail_exceeding_tolerance_trimmed(self):
        """Trailing non-visible minutes exceeding tolerance are trimmed."""
        # 20-min seq, last 5 min non-visible (earthlimb)
        pattern = np.ones(30, dtype=bool)
        pattern[15:20] = False

        class _ELVis:
            def __init__(self, *a, **kw):
                pass

            def get_visibility(self, coord, times, roll=None):
                n = len(times)
                result = np.ones(n, dtype=bool)
                for i, t in enumerate(times):
                    idx = int(np.rint((t - T0).sec / 60.0))
                    if 0 <= idx < len(pattern):
                        result[i] = pattern[idx]
                return result

            def get_all_constraints(self, coord, time, roll=None):
                return {
                    "moon": True,
                    "sun": True,
                    "earthlimb": False,
                }

        proc = self._make_processor(_ELVis("L1", "L2"))
        proc.earthlimb_gap_tolerance = 2  # 5-min tail > 2 tol

        seq = _make_seq("s1", "T", start_min=0, duration_min=20)
        cal = _make_calendar([seq])
        result = proc._trim_non_visible_tails(cal)

        out = result.visits[0].sequences[0]
        # Tail should be trimmed
        expected_stop = T0 + 15 * u.min
        assert abs((out.stop_time - expected_stop).sec) < 1


class _STBreakdownVis:
    """Visibility mock exposing a star-tracker breakdown from a mask.

    st_mask is indexed in minutes since T0. ``roll_masks`` maps a rounded
    roll in degrees to an alternative mask, so a test can show the verdict
    actually depends on the roll it was asked about.
    """

    _st_constraint_active = True

    def __init__(self, st_mask, roll_masks=None):
        self.st_mask = np.asarray(st_mask, dtype=bool)
        self.roll_masks = roll_masks or {}
        self.rolls_seen = []

    def get_visibility(self, coord, times, roll=None):
        return np.ones(len(times), dtype=bool)

    def get_all_constraints(self, coord, time, roll=None):
        return {"moon": True, "sun": True, "earthlimb": True}

    def get_star_tracker_breakdown(self, coord, time, roll=None, pre=None):
        roll_deg = None if roll is None else float(roll.to(u.deg).value)
        self.rolls_seen.append(roll_deg)
        mask = self.st_mask
        if roll_deg is not None:
            mask = self.roll_masks.get(round(roll_deg), mask)

        def _lookup(index):
            return bool(mask[index]) if 0 <= index < len(mask) else True

        if time.isscalar:
            index = int(np.rint((time - T0).sec / 60.0))
            return {"passed": {"combined": _lookup(index)}}
        indices = np.rint((time - T0).sec / 60.0).astype(int)
        return {
            "passed": {
                "combined": np.array([_lookup(i) for i in indices], dtype=bool)
            }
        }


class TestSTStartBuffer:
    """The opening minutes of an observation must be star-tracker visible."""

    def _make_processor(self, visibility, buffer_minutes=12):
        proc = ScheduleProcessor.__new__(ScheduleProcessor)
        proc.visibility = visibility
        proc.min_sequence_duration = TimeDelta(8 * 60 * u.s)
        proc._roll_sweep_enabled = False
        proc._computed_target_rolls = {}
        proc.st_gap_tolerance_start_buffer = buffer_minutes
        # These tests are about the star-tracker buffer specifically, so
        # the Earth-limb one is left off.
        proc.earthlimb_gap_tolerance_start_buffer = 0
        return proc

    def test_clear_start_left_alone(self):
        """A start that already clears the buffer is untouched."""
        proc = self._make_processor(_STBreakdownVis(np.ones(60, dtype=bool)))
        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        assert cal.visits[0].sequences[0].start_time == T0

    def test_dark_start_trimmed_forward(self):
        """A tracker dropout at the start moves start_time forward."""
        mask = np.ones(60, dtype=bool)
        mask[0:5] = False
        proc = self._make_processor(_STBreakdownVis(mask))
        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        out = cal.visits[0].sequences[0]
        assert abs((out.start_time - (T0 + 5 * u.min)).sec) < 1
        # Only the start moves; the stop is left where it was.
        assert abs((out.stop_time - (T0 + 40 * u.min)).sec) < 1

    def test_dropout_inside_buffer_pushes_past_it(self):
        """A dropout inside the buffer moves past the dropout, not past 0."""
        mask = np.ones(60, dtype=bool)
        mask[3:6] = False  # minute 0 alone looks fine; the buffer does not
        proc = self._make_processor(_STBreakdownVis(mask), buffer_minutes=12)
        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        out = cal.visits[0].sequences[0]
        assert abs((out.start_time - (T0 + 6 * u.min)).sec) < 1

    def test_no_clear_run_logs_error_and_keeps_sequence(self, capsys):
        """No qualifying run anywhere → error logged, sequence untouched."""
        proc = self._make_processor(_STBreakdownVis(np.zeros(60, dtype=bool)))
        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        out = cal.visits[0].sequences[0]
        assert out.start_time == T0
        assert out.stop_time == T0 + 40 * u.min
        assert "START BUFFER" in capsys.readouterr().out

    def test_trim_below_minimum_duration_is_refused(self, capsys):
        """Trimming that would leave under min_sequence_duration is refused."""
        mask = np.ones(60, dtype=bool)
        mask[0:14] = False
        proc = self._make_processor(_STBreakdownVis(mask), buffer_minutes=5)
        seq = _make_seq("s1", "T", start_min=0, duration_min=20)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        # Would have to start 14 min in, leaving 6 min < the 8 min minimum.
        assert cal.visits[0].sequences[0].start_time == T0
        assert "START BUFFER" in capsys.readouterr().out

    def test_buffer_longer_than_observation_must_be_clear_throughout(self):
        """A buffer past the stop time means the whole observation is clear."""
        mask = np.ones(60, dtype=bool)
        mask[0:2] = False
        proc = self._make_processor(_STBreakdownVis(mask), buffer_minutes=30)
        seq = _make_seq("s1", "T", start_min=0, duration_min=12)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        # The buffer outruns the 12 min observation, so the requirement is
        # "clear to the stop"; trimming the two dark minutes achieves that.
        out = cal.visits[0].sequences[0]
        assert abs((out.start_time - (T0 + 2 * u.min)).sec) < 1

    def test_dropout_after_the_buffer_is_left_to_the_gap_tolerance(self):
        """Dropouts beyond the buffer are not this pass's concern."""
        mask = np.ones(60, dtype=bool)
        mask[20:24] = False
        proc = self._make_processor(_STBreakdownVis(mask), buffer_minutes=12)
        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        assert cal.visits[0].sequences[0].start_time == T0

    def test_disabled_by_zero_buffer(self):
        """A zero buffer skips the check entirely."""
        proc = self._make_processor(
            _STBreakdownVis(np.zeros(60, dtype=bool)), buffer_minutes=0
        )
        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        assert cal.visits[0].sequences[0].start_time == T0

    def test_skipped_when_star_trackers_inactive(self):
        """No star-tracker constraints configured: nothing to enforce."""
        vis = _STBreakdownVis(np.zeros(60, dtype=bool))
        vis._st_constraint_active = False
        proc = self._make_processor(vis)
        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        assert cal.visits[0].sequences[0].start_time == T0
        assert vis.rolls_seen == []

    def test_uses_the_swept_roll(self):
        """The observation's swept roll is what the trackers are checked at."""
        vis = _STBreakdownVis(
            np.ones(60, dtype=bool),
            roll_masks={137: np.zeros(60, dtype=bool)},
        )
        proc = self._make_processor(vis)
        proc._roll_sweep_enabled = True
        proc._computed_target_rolls = {"v1": {"T": 137.0}}
        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        assert vis.rolls_seen == [137.0]
        # At roll 137 the trackers never pass, so nothing can be trimmed.
        assert cal.visits[0].sequences[0].start_time == T0


class _EarthlimbPatternVis:
    """Boresight Earth-limb clearance from a minute-indexed mask.

    Star trackers are always clear, so these tests isolate the Earth-limb
    start buffer.
    """

    _st_constraint_active = False

    def __init__(self, clear_mask):
        self.clear_mask = np.asarray(clear_mask, dtype=bool)

    def _lookup(self, times):
        indices = np.rint((times - T0).sec / 60.0).astype(int)
        return np.array(
            [
                (
                    bool(self.clear_mask[i])
                    if 0 <= i < len(self.clear_mask)
                    else True
                )
                for i in np.atleast_1d(indices)
            ],
            dtype=bool,
        )

    def get_visibility(self, coord, times, roll=None):
        return self._lookup(times)

    def get_constraint(self, coord, body, time, pre=None):
        assert body == "earthlimb"
        return self._lookup(time)


class TestEarthlimbStartBuffer:
    """The boresight must be clear of the Earth over the opening minutes."""

    def _make_processor(self, visibility, buffer_minutes=12):
        proc = ScheduleProcessor.__new__(ScheduleProcessor)
        proc.visibility = visibility
        proc.min_sequence_duration = TimeDelta(8 * 60 * u.s)
        proc._roll_sweep_enabled = False
        proc._computed_target_rolls = {}
        proc.st_gap_tolerance_start_buffer = 0
        proc.earthlimb_gap_tolerance_start_buffer = buffer_minutes
        return proc

    def test_clear_start_left_alone(self):
        proc = self._make_processor(
            _EarthlimbPatternVis(np.ones(60, dtype=bool))
        )
        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        assert cal.visits[0].sequences[0].start_time == T0

    def test_start_inside_the_earth_limb_is_trimmed_forward(self):
        """An observation opening with the boresight in the Earth moves."""
        mask = np.ones(60, dtype=bool)
        mask[0:7] = False
        proc = self._make_processor(_EarthlimbPatternVis(mask))
        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        out = cal.visits[0].sequences[0]
        assert abs((out.start_time - (T0 + 7 * u.min)).sec) < 1
        assert abs((out.stop_time - (T0 + 40 * u.min)).sec) < 1

    def test_dip_inside_the_buffer_pushes_past_it(self):
        """A dip the gap tolerance would accept still moves the start."""
        mask = np.ones(60, dtype=bool)
        mask[4:9] = False
        proc = self._make_processor(
            _EarthlimbPatternVis(mask), buffer_minutes=12
        )
        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        out = cal.visits[0].sequences[0]
        assert abs((out.start_time - (T0 + 9 * u.min)).sec) < 1

    def test_dip_after_the_buffer_is_left_alone(self):
        """Beyond the buffer the gap tolerance takes over again."""
        mask = np.ones(60, dtype=bool)
        mask[20:24] = False
        proc = self._make_processor(
            _EarthlimbPatternVis(mask), buffer_minutes=12
        )
        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        assert cal.visits[0].sequences[0].start_time == T0

    def test_never_clear_logs_error_and_keeps_sequence(self, capsys):
        proc = self._make_processor(
            _EarthlimbPatternVis(np.zeros(60, dtype=bool))
        )
        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        assert cal.visits[0].sequences[0].start_time == T0
        assert "START BUFFER" in capsys.readouterr().out

    def test_disabled_by_zero_buffer(self):
        proc = self._make_processor(
            _EarthlimbPatternVis(np.zeros(60, dtype=bool)), buffer_minutes=0
        )
        seq = _make_seq("s1", "T", start_min=0, duration_min=40)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        assert cal.visits[0].sequences[0].start_time == T0

    def test_both_buffers_are_satisfied_together(self):
        """Clearing one constraint must not land the start inside the other.

        The trackers are dark over 0-5 and the boresight over 8-12, so
        neither requirement alone gives the right answer: satisfying the
        trackers would start at 6, which puts the Earth-limb violation
        straight back inside the buffer.
        """
        tracker_ok = np.ones(60, dtype=bool)
        tracker_ok[0:6] = False
        limb_clear = np.ones(60, dtype=bool)
        limb_clear[8:13] = False

        class _BothVis(_EarthlimbPatternVis):
            _st_constraint_active = True

            def get_star_tracker_breakdown(
                self, coord, time, roll=None, pre=None
            ):
                indices = np.rint((time - T0).sec / 60.0).astype(int)
                return {
                    "passed": {
                        "combined": np.array(
                            [
                                (
                                    bool(tracker_ok[i])
                                    if 0 <= i < len(tracker_ok)
                                    else True
                                )
                                for i in np.atleast_1d(indices)
                            ],
                            dtype=bool,
                        )
                    }
                }

        proc = self._make_processor(_BothVis(limb_clear), buffer_minutes=12)
        proc.st_gap_tolerance_start_buffer = 12
        seq = _make_seq("s1", "T", start_min=0, duration_min=50)
        cal = _make_calendar([seq])

        proc._enforce_start_buffers(cal)

        out = cal.visits[0].sequences[0]
        assert abs((out.start_time - (T0 + 13 * u.min)).sec) < 1


class TestGapToleranceUsesObservationRoll:
    """``_is_gap_tolerable`` must judge the trackers at the flown roll."""

    def _make_processor(self, visibility):
        proc = ScheduleProcessor.__new__(ScheduleProcessor)
        proc.visibility = visibility
        proc.earthlimb_gap_tolerance = 0
        proc.st_gap_tolerance = 2
        return proc

    def test_roll_is_forwarded_to_the_tracker_check(self):
        """The roll handed in reaches get_star_tracker_breakdown."""
        vis = _STBreakdownVis(np.zeros(4, dtype=bool))
        proc = self._make_processor(vis)
        coord = SkyCoord(10, 20, frame="icrs", unit="deg")

        tolerable = proc._is_gap_tolerable(
            coord, _make_time_grid(4), 0, 2, roll=137.0
        )

        assert tolerable is True
        assert vis.rolls_seen == [137.0]

    def test_trackers_clear_at_this_roll_needs_no_tolerance(self):
        """Boresight and trackers both clear → the gap is not ridden out."""
        vis = _STBreakdownVis(np.ones(4, dtype=bool))
        proc = self._make_processor(vis)
        coord = SkyCoord(10, 20, frame="icrs", unit="deg")

        result = proc._is_gap_tolerable(
            coord, _make_time_grid(4), 0, 2, roll=137.0
        )

        assert result is False

    def test_sun_or_moon_failure_is_never_tolerable(self):
        """Only earth-limb and star-tracker failures have tolerances."""

        class _SunFailVis(_STBreakdownVis):
            def get_all_constraints(self, coord, time, roll=None):
                return {"moon": True, "sun": False, "earthlimb": True}

        proc = self._make_processor(_SunFailVis(np.zeros(4, dtype=bool)))
        proc.earthlimb_gap_tolerance = 30
        proc.st_gap_tolerance = 30
        coord = SkyCoord(10, 20, frame="icrs", unit="deg")

        result = proc._is_gap_tolerable(coord, _make_time_grid(4), 0, 2)

        assert result is False

    def test_unevaluable_tracker_check_is_reported_not_guessed(self, capsys):
        """A tracker check that blows up is logged and the gap is trimmed."""

        class _BrokenVis(_STBreakdownVis):
            def get_star_tracker_breakdown(
                self, coord, time, roll=None, pre=None
            ):
                raise RuntimeError("ephemeris unavailable")

        proc = self._make_processor(_BrokenVis(np.zeros(4, dtype=bool)))
        coord = SkyCoord(10, 20, frame="icrs", unit="deg")

        result = proc._is_gap_tolerable(coord, _make_time_grid(4), 0, 2)

        # Keeping dark minutes on the strength of a verdict we never got
        # would be the unsafe guess, so the gap is treated as intolerable.
        assert result is False
        assert "star-tracker check failed" in capsys.readouterr().out
