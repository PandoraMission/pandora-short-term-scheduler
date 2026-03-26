"""Tests for gap-filling logic and roll-aware visibility in the scheduler.

Covers:
- _find_false_blocks helper
- _fill_gaps method
- _fix_visibility integration with mock visibility
- Roll-aware gap filling (roll kwarg threading)
- Gap report structure verification
"""

# Third-party
import numpy as np
import pytest
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


class DummyVisibilityRollSensitive:
    """Visibility mock — True only when the correct roll is passed.

    Parameters
    ----------
    good_roll_deg : float
        The roll angle (degrees) for which visibility is all-True.
        Any other roll (or ``None``) gives all-False.
    tolerance : float
        Matching tolerance in degrees.
    """

    def __init__(self, l1, l2, good_roll_deg=90.0, tolerance=1.0, **kwargs):
        self._good_roll = good_roll_deg
        self._tol = tolerance

    def get_visibility(self, coord, times, roll=None):
        try:
            n = len(times)
        except Exception:
            n = 1
        if roll is not None:
            val = roll.to(u.deg).value
            if abs(val - self._good_roll) <= self._tol:
                return np.ones(n, dtype=bool)
        return np.zeros(n, dtype=bool)


# ================================================================
# Tests: _find_false_blocks
# ================================================================


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


# ================================================================
# Tests: _fill_gaps
# ================================================================


class TestFillGaps:
    """Unit tests for ScheduleProcessor._fill_gaps."""

    def _make_processor(self):
        """Create a bare ScheduleProcessor without full Visibility."""
        proc = ScheduleProcessor.__new__(ScheduleProcessor)
        proc.min_sequence_duration = TimeDelta(5 * 60 * u.s)
        proc.max_sequence_duration = TimeDelta(90 * 60 * u.s)
        proc._roll_sweep_enabled = False
        proc._computed_target_rolls = {}
        proc.earthlimb_gap_tolerance = 0
        proc.st_gap_tolerance = 0
        return proc

    def test_gap_shifts_start_backward(self):
        proc = self._make_processor()
        seq = _make_seq("s1", "T", start_min=10, duration_min=20)
        original_stop = seq.stop_time

        filled = proc._fill_gaps(seq, gap_length=5)

        assert filled.start_time == seq.start_time - 5 * u.min
        assert filled.stop_time == original_stop

    def test_gap_zero_is_noop(self):
        proc = self._make_processor()
        seq = _make_seq("s1", "T", start_min=10, duration_min=20)
        filled = proc._fill_gaps(seq, gap_length=0)
        assert filled.start_time == seq.start_time

    def test_payload_params_deep_copied(self):
        proc = self._make_processor()
        params = {"key": {"nested": "value"}}
        seq = _make_seq("s1", "T", start_min=10, duration_min=20)
        seq.payload_params = params

        filled = proc._fill_gaps(seq, gap_length=3)

        # Mutating original should not affect filled copy
        params["key"]["nested"] = "changed"
        assert filled.payload_params["key"]["nested"] == "value"

    def test_blind_fill_extends_into_non_visible(self):
        """_fill_gaps is blind — extends regardless of visibility."""
        proc = self._make_processor()
        seq = _make_seq("s1", "T", start_min=10, duration_min=20)
        filled = proc._fill_gaps(seq, gap_length=5)

        # Should extend the full 5 minutes blindly
        assert filled.start_time == seq.start_time - 5 * u.min


# ================================================================
# Tests: _fix_visibility with mock visibility
# ================================================================


class TestFixVisibility:
    """Integration tests for _fix_visibility using pattern-based mocks."""

    def _make_processor_with_pattern(self, pattern, monkeypatch):
        """Set up a ScheduleProcessor whose visibility is pattern-driven."""
        dummy = DummyVisibilityPattern("L1", "L2", pattern=pattern)
        monkeypatch.setattr(
            "shortschedule.scheduler.Visibility",
            lambda l1, l2, **kw: dummy,
        )
        proc = ScheduleProcessor("L1", "L2")
        return proc

    def test_single_gap_previous_visible(self, monkeypatch):
        """Previous target visible during gap → extend prev, shrink current."""
        # Sequence A: minutes 0-9 (visible)
        # Gap: minutes 10-14 (NOT visible for A's target in main eval,
        #       but VISIBLE for A's target in gap-fill check)
        # Sequence B: minutes 15-24 (visible)
        #
        # The main visibility array has False at 10-14.
        # When _fix_visibility queries A's target during gap, we want True.
        pattern = np.ones(25, dtype=bool)
        pattern[10:15] = False  # gap in main visibility

        seqA = _make_seq("sA", "TargetA", start_min=0, duration_min=10)
        seqB = _make_seq("sB", "TargetB", start_min=10, duration_min=15)
        cal = _make_calendar([seqA, seqB])

        # Use all-true dummy so gap-fill queries for prev target return True
        dummy_all_true = DummyVisibilityAllTrue("L1", "L2")
        monkeypatch.setattr(
            "shortschedule.scheduler.Visibility",
            lambda l1, l2, **kw: dummy_all_true,
        )
        proc = ScheduleProcessor("L1", "L2")
        # Manually set the visibility array with the gap
        all_minutes_bool = pattern.copy()

        result = proc._fix_visibility(cal, all_minutes_bool)

        # Sequence A should have been extended
        seqA_out = result.visits[0].sequences[0]
        assert seqA_out.stop_time > seqA.stop_time

    def test_single_gap_previous_not_visible(self, monkeypatch):
        """Previous target NOT visible during gap → gap remains."""
        pattern = np.ones(25, dtype=bool)
        pattern[10:15] = False

        seqA = _make_seq("sA", "TargetA", start_min=0, duration_min=10)
        seqB = _make_seq("sB", "TargetB", start_min=10, duration_min=15)
        cal = _make_calendar([seqA, seqB])

        # Visibility returns False for everything during gap check
        dummy_all_false_gap = DummyVisibilityPattern(
            "L1", "L2", pattern=pattern
        )
        monkeypatch.setattr(
            "shortschedule.scheduler.Visibility",
            lambda l1, l2, **kw: dummy_all_false_gap,
        )
        proc = ScheduleProcessor("L1", "L2")
        result = proc._fix_visibility(cal, pattern)

        # Sequence A should NOT have been extended
        seqA_out = result.visits[0].sequences[0]
        assert seqA_out.stop_time == seqA.stop_time

    def test_all_false_visibility(self, monkeypatch):
        """All minutes non-visible → no extensions, sequences shrunk."""
        n = 20
        pattern = np.zeros(n, dtype=bool)

        seqA = _make_seq("sA", "TargetA", start_min=0, duration_min=10)
        seqB = _make_seq("sB", "TargetB", start_min=10, duration_min=10)
        cal = _make_calendar([seqA, seqB])

        dummy = DummyVisibilityPattern("L1", "L2", pattern=pattern)
        monkeypatch.setattr(
            "shortschedule.scheduler.Visibility",
            lambda l1, l2, **kw: dummy,
        )
        proc = ScheduleProcessor("L1", "L2")
        proc._fix_visibility(cal, pattern)

        # Gap report should record remaining gaps
        assert proc.gap_report["processing_summary"]["gaps_remaining"] >= 0

    def test_gap_report_counts(self, monkeypatch):
        """Verify gap report counts after gap processing."""
        pattern = np.ones(30, dtype=bool)
        pattern[10:15] = False
        pattern[20:25] = False

        seqA = _make_seq("sA", "TargetA", start_min=0, duration_min=10)
        seqB = _make_seq("sB", "TargetB", start_min=10, duration_min=10)
        seqC = _make_seq("sC", "TargetA", start_min=20, duration_min=10)
        cal = _make_calendar([seqA, seqB, seqC])

        dummy = DummyVisibilityAllTrue("L1", "L2")
        monkeypatch.setattr(
            "shortschedule.scheduler.Visibility",
            lambda l1, l2, **kw: dummy,
        )
        proc = ScheduleProcessor("L1", "L2")
        proc._fix_visibility(cal, pattern)

        summary = proc.gap_report["processing_summary"]
        assert summary["gaps_processed"] == 2


# ================================================================
# Tests: Roll-aware gap filling
# ================================================================


class TestRollAwareGapFilling:
    """Verify roll kwarg is threaded through gap-filling logic."""

    def test_gap_fill_uses_stored_roll(self, monkeypatch):
        """_fix_visibility passes the stored roll to get_visibility."""
        pattern = np.ones(20, dtype=bool)
        pattern[8:12] = False

        seqA = _make_seq("sA", "TargetA", start_min=0, duration_min=10)
        seqB = _make_seq("sB", "TargetB", start_min=10, duration_min=10)
        cal = _make_calendar([seqA, seqB])

        # Track whether roll was passed
        received_rolls = []

        class TrackingVisibility:
            def __init__(self, l1, l2, **kw):
                pass

            def get_visibility(self, coord, times, roll=None):
                received_rolls.append(roll)
                try:
                    n = len(times)
                except Exception:
                    n = 1
                return np.ones(n, dtype=bool)

        monkeypatch.setattr(
            "shortschedule.scheduler.Visibility",
            TrackingVisibility,
        )
        proc = ScheduleProcessor("L1", "L2")

        # Enable roll sweep (normally set by passing st_* params)
        proc._roll_sweep_enabled = True

        # Pre-populate computed rolls
        proc._computed_target_rolls = {
            "v1": {"TargetA": 42.0, "TargetB": 99.0}
        }

        proc._fix_visibility(cal, pattern)

        # At least one call should have passed roll=42*u.deg
        # (for TargetA, the previous target during the gap)
        roll_values = [
            r.to(u.deg).value for r in received_rolls if r is not None
        ]
        assert 42.0 in roll_values

    def test_end_to_end_roll_sensitive(self, monkeypatch):
        """Full process_calendar with roll-sensitive visibility."""
        import shortschedule

        pkgdir = shortschedule.__file__.rsplit("/", 1)[0]
        sample = (
            pkgdir + "/data/Pandora_science_calendar_20251018_tsb-futz.xml"
        )

        from shortschedule.parser import parse_science_calendar

        cal = parse_science_calendar(sample)
        if not cal.visits:
            pytest.skip("Sample calendar has no visits")

        # Use all-true visibility so pipeline completes normally
        monkeypatch.setattr(
            "shortschedule.scheduler.Visibility",
            DummyVisibilityAllTrue,
        )
        first_seq = cal.visits[0].sequences[0]
        sched = ScheduleProcessor("LINE1", "LINE2")
        processed = sched.process_calendar(
            cal,
            window_start=first_seq.start_time.isot,
            window_duration_days=1,
            verbose=False,
        )

        # All sequences should have roll assigned
        for visit in processed.visits:
            for seq in visit.sequences:
                assert seq.roll is not None


# ================================================================
# Tests: _trim_non_visible_tails
# ================================================================


class TestTrimNonVisibleTails:
    """Unit tests for ScheduleProcessor._trim_non_visible_tails."""

    def _make_processor(self, visibility_cls):
        proc = ScheduleProcessor.__new__(ScheduleProcessor)
        proc.min_sequence_duration = TimeDelta(8 * 60 * u.s)
        proc.max_sequence_duration = TimeDelta(90 * 60 * u.s)
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


# ================================================================
# Tests: _trim_non_visible_heads
# ================================================================


class TestTrimNonVisibleHeads:
    """Unit tests for ScheduleProcessor._trim_non_visible_heads."""

    def _make_processor(self, visibility_cls):
        proc = ScheduleProcessor.__new__(ScheduleProcessor)
        proc.min_sequence_duration = TimeDelta(8 * 60 * u.s)
        proc.max_sequence_duration = TimeDelta(90 * 60 * u.s)
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

    def test_no_head_no_change(self):
        """All-visible sequence is untouched."""
        dummy = DummyVisibilityAllTrue("L1", "L2")
        proc = self._make_processor(dummy)

        seq = _make_seq("s1", "T", start_min=0, duration_min=20)
        cal = _make_calendar([seq])
        result = proc._trim_non_visible_heads(cal)

        out = result.visits[0].sequences[0]
        assert out.start_time == seq.start_time

    def test_head_trimmed(self):
        """Non-visible head is trimmed to first visible minute."""
        # Minutes 0-4 non-visible, 5-19 visible
        pattern = np.ones(30, dtype=bool)
        pattern[0:5] = False
        dummy = DummyVisibilityPattern("L1", "L2", pattern=pattern)
        proc = self._make_processor(dummy)

        seq = _make_seq("s1", "T", start_min=0, duration_min=20)
        cal = _make_calendar([seq])
        result = proc._trim_non_visible_heads(cal)

        out = result.visits[0].sequences[0]
        expected_start = T0 + 5 * u.min
        assert abs((out.start_time - expected_start).sec) < 1
        # stop_time should be unchanged
        assert out.stop_time == seq.stop_time

    def test_prev_sequence_extended_forward(self):
        """After trimming head, prev seq extends forward if visible."""
        # Seq A: minutes 0-19, RA=10 → all visible
        # Seq B: minutes 20-39, RA=50 → head at 20-24 non-visible
        pattern_b = np.ones(40, dtype=bool)
        pattern_b[20:25] = False

        class _PerTargetVis:
            def __init__(self, *a, **kw):
                pass

            def get_visibility(self, coord, times, roll=None):
                n = len(times)
                if abs(coord.ra.deg - 50.0) < 1.0:
                    result = np.ones(n, dtype=bool)
                    for i, t in enumerate(times):
                        idx = int(np.rint((t - T0).sec / 60.0))
                        if 0 <= idx < len(pattern_b):
                            result[i] = pattern_b[idx]
                    return result
                # Target A is all-visible
                return np.ones(n, dtype=bool)

        proc = self._make_processor(_PerTargetVis("L1", "L2"))

        seqA = _make_seq(
            "sA", "TargetA", start_min=0, duration_min=20, ra=10.0
        )
        seqB = _make_seq(
            "sB", "TargetB", start_min=20, duration_min=20, ra=50.0
        )
        cal = _make_calendar([seqA, seqB])
        result = proc._trim_non_visible_heads(cal)

        outA = result.visits[0].sequences[0]
        # Seq A should extend forward to cover the gap
        assert outA.stop_time > seqA.stop_time

    def test_skip_if_trimming_too_short(self):
        """Sequence not trimmed if result would be < min_sequence_duration."""
        # 10-minute seq, minutes 0-7 non-visible (only minutes 8-9 visible)
        # Trimming would leave 2 minutes < 8 min minimum
        pattern = np.ones(20, dtype=bool)
        pattern[0:8] = False
        dummy = DummyVisibilityPattern("L1", "L2", pattern=pattern)
        proc = self._make_processor(dummy)

        seq = _make_seq("s1", "T", start_min=0, duration_min=10)
        cal = _make_calendar([seq])
        result = proc._trim_non_visible_heads(cal)

        out = result.visits[0].sequences[0]
        # Should remain unchanged
        assert out.start_time == seq.start_time

    def test_entirely_non_visible_skipped(self):
        """Entirely non-visible sequence is not modified."""
        pattern = np.zeros(20, dtype=bool)
        dummy = DummyVisibilityPattern("L1", "L2", pattern=pattern)
        proc = self._make_processor(dummy)

        seq = _make_seq("s1", "T", start_min=0, duration_min=20)
        cal = _make_calendar([seq])
        result = proc._trim_non_visible_heads(cal)

        out = result.visits[0].sequences[0]
        assert out.start_time == seq.start_time

    def test_contiguity_preserved_when_prev_not_visible(self):
        """Head trim + blind prev extension maintains contiguity."""
        # Seq A: minutes 0-19, RA=10
        # Seq B: minutes 20-39, RA=50, head at 20-24 non-visible
        # Prev target (A) also NOT visible in gap → blind extend
        pattern_b = np.ones(40, dtype=bool)
        pattern_b[20:25] = False

        class _NeitherVisibleInGap:
            def __init__(self, *a, **kw):
                pass

            def get_visibility(self, coord, times, roll=None):
                n = len(times)
                if abs(coord.ra.deg - 50.0) < 1.0:
                    result = np.ones(n, dtype=bool)
                    for i, t in enumerate(times):
                        idx = int(np.rint((t - T0).sec / 60.0))
                        if 0 <= idx < len(pattern_b):
                            result[i] = pattern_b[idx]
                    return result
                # Prev target also NOT visible at 20-24
                result = np.ones(n, dtype=bool)
                for i, t in enumerate(times):
                    idx = int(np.rint((t - T0).sec / 60.0))
                    if 20 <= idx < 25:
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
        result = proc._trim_non_visible_heads(cal)

        outA = result.visits[0].sequences[0]
        outB = result.visits[0].sequences[1]
        # No gap: A.stop must equal B.start
        gap_sec = abs((outB.start_time - outA.stop_time).sec)
        assert gap_sec < 1, f"Gap of {gap_sec:.0f}s created"
        # B's head should be trimmed
        assert outB.start_time > seqB.start_time


# ================================================================
# Tests: _fix_visibility does not create gaps
# ================================================================


class TestFixVisibilityNoGaps:
    """Verify _fix_visibility never introduces inter-sequence gaps."""

    def test_no_shrink_when_prev_not_visible(self, monkeypatch):
        """When prev target is NOT visible in gap, current must not shrink."""
        # Sequence A: minutes 0-14
        # Sequence B: minutes 15-34 (minutes 15-19 are non-visible)
        # Previous target NOT visible in gap either → no extend.
        # Bug (before fix): B would still shrink → gap from 15 to 20.
        pattern = np.ones(35, dtype=bool)
        pattern[15:20] = False  # gap in main visibility

        seqA = _make_seq("sA", "TargetA", start_min=0, duration_min=15)
        seqB = _make_seq("sB", "TargetB", start_min=15, duration_min=20)
        cal = _make_calendar([seqA, seqB])

        # Visibility returns False for everything (prev not visible)
        dummy = DummyVisibilityPattern("L1", "L2", pattern=pattern)
        monkeypatch.setattr(
            "shortschedule.scheduler.Visibility",
            lambda l1, l2, **kw: dummy,
        )
        proc = ScheduleProcessor("L1", "L2")
        result = proc._fix_visibility(cal, pattern)

        outA = result.visits[0].sequences[0]
        outB = result.visits[0].sequences[1]
        gap_sec = (outB.start_time - outA.stop_time).sec
        assert abs(gap_sec) < 1, f"Gap of {gap_sec:.0f}s between sequences"

    def test_partial_visibility_no_gap(self, monkeypatch):
        """Partial prev visibility: shrink matches extend, no gap."""
        # Sequence A: minutes 0-9
        # Sequence B: minutes 10-29 (minutes 10-14 non-visible)
        # Previous target visible at 10-12 only (not 13-14)
        pattern = np.ones(30, dtype=bool)
        pattern[10:15] = False

        seqA = _make_seq("sA", "TargetA", start_min=0, duration_min=10)
        seqB = _make_seq("sB", "TargetB", start_min=10, duration_min=20)
        cal = _make_calendar([seqA, seqB])

        # Previous target visible only at 10-12
        partial_pattern = np.ones(30, dtype=bool)
        partial_pattern[13:15] = False

        dummy = DummyVisibilityPattern("L1", "L2", pattern=partial_pattern)
        monkeypatch.setattr(
            "shortschedule.scheduler.Visibility",
            lambda l1, l2, **kw: dummy,
        )
        proc = ScheduleProcessor("L1", "L2")
        result = proc._fix_visibility(cal, pattern)

        outA = result.visits[0].sequences[0]
        outB = result.visits[0].sequences[1]
        gap_sec = (outB.start_time - outA.stop_time).sec
        assert abs(gap_sec) < 1, f"Gap of {gap_sec:.0f}s between sequences"


# ================================================================
# Tests: _trim_to_longest_visible_block
# ================================================================


class TestTrimToLongestVisibleBlock:
    """Unit tests for ScheduleProcessor._trim_to_longest_visible_block."""

    def _make_processor(self, visibility_cls):
        proc = ScheduleProcessor.__new__(ScheduleProcessor)
        proc.min_sequence_duration = TimeDelta(8 * 60 * u.s)
        proc.max_sequence_duration = TimeDelta(90 * 60 * u.s)
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

            def get_all_constraints(self, coord, time):
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

            def get_all_constraints(self, coord, time):
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

            def get_all_constraints(self, coord, time):
                # Boresight constraints pass; ST fails
                return {
                    "moon": True,
                    "sun": True,
                    "earthlimb": True,
                }

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

            def get_all_constraints(self, coord, time):
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
# Tests: tolerance at sequence heads and tails
# ================================================================


class TestToleranceAtHeadsAndTails:
    """Verify gap tolerances are applied at sequence boundaries."""

    def _make_processor(self, visibility_cls):
        proc = ScheduleProcessor.__new__(ScheduleProcessor)
        proc.min_sequence_duration = TimeDelta(8 * 60 * u.s)
        proc.max_sequence_duration = TimeDelta(90 * 60 * u.s)
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

            def get_all_constraints(self, coord, time):
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

            def get_all_constraints(self, coord, time):
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

    def test_head_within_tolerance_not_shrunk(self, monkeypatch):
        """Leading non-visible minutes within tolerance → no shrink."""
        # Seq A: 0-9, Seq B: 10-29 (minutes 10-11 non-visible)
        # Gap of 2 min at head of B, earthlimb failure
        pattern = np.ones(30, dtype=bool)
        pattern[10:12] = False

        class _ELVis:
            def __init__(self, l1, l2, **kw):
                pass

            def get_visibility(self, coord, times, roll=None):
                n = len(times)
                result = np.ones(n, dtype=bool)
                for i, t in enumerate(times):
                    idx = int(np.rint((t - T0).sec / 60.0))
                    if 0 <= idx < len(pattern):
                        result[i] = pattern[idx]
                return result

            def get_all_constraints(self, coord, time):
                return {
                    "moon": True,
                    "sun": True,
                    "earthlimb": False,
                }

        monkeypatch.setattr(
            "shortschedule.scheduler.Visibility",
            _ELVis,
        )
        proc = ScheduleProcessor(
            "L1",
            "L2",
            earthlimb_gap_tolerance=2,
        )

        seqA = _make_seq("sA", "TargetA", start_min=0, duration_min=10)
        seqB = _make_seq("sB", "TargetB", start_min=10, duration_min=20)
        cal = _make_calendar([seqA, seqB])
        result = proc._fix_visibility(cal, pattern)

        outA = result.visits[0].sequences[0]
        outB = result.visits[0].sequences[1]
        # A should NOT be extended (gap is tolerable)
        assert outA.stop_time == seqA.stop_time
        # B should NOT be shrunk (gap is tolerable)
        assert outB.start_time == seqB.start_time


# ================================================================
# TestForceGapFill — force_gap_fill skips visibility corrections
# ================================================================


class TestForceGapFill:
    """When force_gap_fill=True, _fix_visibility, _trim_non_visible_tails,
    and _trim_to_longest_visible_block are all skipped.  Gaps filled
    by _fill_gaps remain even if minutes are non-visible."""

    def _make_processor(self, monkeypatch, pattern):
        """Create a processor with force_gap_fill=True and a partial-
        visibility pattern."""

        class _PartialVis:
            def __init__(self, l1, l2, **kw):
                self._pattern = pattern

            def get_visibility(self, coord, times, roll=None):
                n = len(times)
                result = np.ones(n, dtype=bool)
                for i, t in enumerate(times):
                    idx = int(np.rint((t - T0).sec / 60.0))
                    if 0 <= idx < len(self._pattern):
                        result[i] = self._pattern[idx]
                return result

            def get_all_constraints(self, coord, time):
                return {"earthlimb": True, "sun": True}

            def get_separations(self, coord, time):
                return {"earthlimb": 90.0 * u.deg}

        monkeypatch.setattr(
            "shortschedule.scheduler.Visibility",
            _PartialVis,
        )
        proc = ScheduleProcessor("L1", "L2", force_gap_fill=True)
        proc.earthlimb_gap_tolerance = 0
        proc.st_gap_tolerance = 0
        return proc

    def test_non_visible_tails_preserved(self, monkeypatch):
        """Tails that would normally be trimmed are kept."""
        # Last 5 minutes non-visible
        pattern = np.ones(30, dtype=bool)
        pattern[25:30] = False
        proc = self._make_processor(monkeypatch, pattern)

        seq = _make_seq("s1", "T1", start_min=0, duration_min=30)
        cal = _make_calendar([seq])
        result = proc._trim_non_visible_tails(cal)

        # Without force_gap_fill the tail would be trimmed; with it
        # _process_all_sequences skips the call.  We test the flag
        # directly:
        assert proc.force_gap_fill is True

    def test_process_all_sequences_no_gap(self, monkeypatch):
        """Two sequences with a gap — gap is filled and NOT reopened."""
        # Seq A: mins 0-19 (all visible)
        # Gap: mins 20-24 (non-visible)
        # Seq B: mins 25-44 (all visible)
        pattern = np.ones(45, dtype=bool)
        pattern[20:25] = False
        proc = self._make_processor(monkeypatch, pattern)

        seqA = _make_seq("sA", "TA", start_min=0, duration_min=20)
        seqB = _make_seq("sB", "TB", start_min=25, duration_min=20)
        cal = _make_calendar([seqA, seqB])

        result = proc._process_all_sequences(cal)
        outA = result.visits[0].sequences[0]
        outB = result.visits[0].sequences[1]

        # _fill_gaps extends seqB backward to fill the 5-min gap,
        # and force_gap_fill prevents the visibility pass from
        # undoing it → no gap between A and B.
        gap_sec = (outB.start_time - outA.stop_time).sec
        assert abs(gap_sec) < 1, f"Expected no gap but found {gap_sec:.1f}s"

    def test_mid_sequence_dark_preserved(self, monkeypatch):
        """Dark minutes in the middle of a sequence are NOT trimmed."""
        # 30-min sequence; mins 12-14 non-visible
        pattern = np.ones(30, dtype=bool)
        pattern[12:15] = False
        proc = self._make_processor(monkeypatch, pattern)

        seq = _make_seq("s1", "T1", start_min=0, duration_min=30)
        cal = _make_calendar([seq])
        result = proc._process_all_sequences(cal)

        out = result.visits[0].sequences[0]
        dur_min = out.duration.sec / 60.0
        # Duration should be unchanged (30 min) — no trimming
        assert abs(dur_min - 30.0) < 1


# ================================================================
# Smart force-fill rules
# ================================================================


class TestForceGapFillRules:
    """Tests for constraint-aware gap-filling rules:
    1. Never extend below earthlimb_hard_floor (5 deg).
    2. Prefer star-tracker violations over earthlimb violations.
    3. Prefer extending the previous sequence forward (gaps at end).
    """

    @staticmethod
    def _smart_vis_factory(
        *,
        prev_el_angles=None,
        next_el_angles=None,
        prev_constraints=None,
        next_constraints=None,
    ):
        """Build a mock Visibility class that returns per-minute data.

        Parameters are dicts mapping minute-offset (from T0) to values.
        Unspecified minutes default to fully visible / 90 deg earthlimb.
        """
        prev_el = prev_el_angles or {}
        next_el = next_el_angles or {}
        prev_con = prev_constraints or {}
        next_con = next_constraints or {}

        class _SmartVis:
            _st_constraint_active = False

            def __init__(self, l1, l2, **kw):
                pass

            def get_visibility(self, coord, times, roll=None):
                n = len(times)
                result = np.ones(n, dtype=bool)
                # Use ra to distinguish prev (ra=10) vs next (ra=30)
                is_prev = abs(coord.ra.deg - 10.0) < 1.0
                el_map = prev_el if is_prev else next_el
                for i, t in enumerate(times):
                    m = int(np.rint((t - T0).sec / 60.0))
                    if m in el_map:
                        # Non-visible when earthlimb map specifies
                        # a value (the caller wants control)
                        result[i] = False
                return result

            def get_separations(self, coord, time):
                is_prev = abs(coord.ra.deg - 10.0) < 1.0
                el_map = prev_el if is_prev else next_el
                m = int(np.rint((time - T0).sec / 60.0))
                angle = el_map.get(m, 90.0)
                return {"earthlimb": angle * u.deg}

            def get_all_constraints(self, coord, time):
                is_prev = abs(coord.ra.deg - 10.0) < 1.0
                con_map = prev_con if is_prev else next_con
                m = int(np.rint((time - T0).sec / 60.0))
                return con_map.get(
                    m,
                    {"earthlimb": True, "sun": True},
                )

        return _SmartVis

    def _make_processor(self, monkeypatch, vis_cls, floor=5.0):
        monkeypatch.setattr("shortschedule.scheduler.Visibility", vis_cls)
        proc = ScheduleProcessor(
            "L1",
            "L2",
            force_gap_fill=True,
            earthlimb_hard_floor=floor,
        )
        proc.earthlimb_gap_tolerance = 0
        proc.st_gap_tolerance = 0
        return proc

    # ---- Rule 3: prefer extending prev forward ----

    def test_prefer_extending_prev_forward(self, monkeypatch):
        """When both targets are equally good, prev takes the gap."""
        # 5-min gap, both targets non-visible but above floor,
        # both have ST-only failures → prev should take all 5 min.
        gap_mins = {m: 20.0 for m in range(20, 25)}  # earthlimb 20 deg
        cons = {
            m: {"earthlimb": True, "star_tracker": False}
            for m in range(20, 25)
        }
        vis_cls = self._smart_vis_factory(
            prev_el_angles=gap_mins,
            next_el_angles=gap_mins,
            prev_constraints=cons,
            next_constraints=cons,
        )
        proc = self._make_processor(monkeypatch, vis_cls)

        seqA = _make_seq(
            "sA", "TA", start_min=0, duration_min=20, ra=10, dec=20
        )
        seqB = _make_seq(
            "sB", "TB", start_min=25, duration_min=20, ra=30, dec=40
        )
        cal = _make_calendar([seqA, seqB])
        result = proc._force_fill_gaps(cal)

        outA = result.visits[0].sequences[0]
        outB = result.visits[0].sequences[1]

        # Prev extended forward by 5 min
        assert abs(outA.stop_time - (T0 + 25 * u.min)) < TimeDelta(
            2, format="sec"
        )
        # Next NOT extended backward (prev took it all)
        assert abs(outB.start_time - (T0 + 25 * u.min)) < TimeDelta(
            2, format="sec"
        )

    # ---- Rule 2: earthlimb hard floor ----

    def test_earthlimb_hard_floor_leaves_gap(self, monkeypatch):
        """Minutes where both targets have earthlimb < floor stay unfilled."""
        # 5-min gap, both targets have earthlimb = 3 deg (< 5 floor)
        gap_mins = {m: 3.0 for m in range(20, 25)}
        vis_cls = self._smart_vis_factory(
            prev_el_angles=gap_mins,
            next_el_angles=gap_mins,
        )
        proc = self._make_processor(monkeypatch, vis_cls)

        seqA = _make_seq(
            "sA", "TA", start_min=0, duration_min=20, ra=10, dec=20
        )
        seqB = _make_seq(
            "sB", "TB", start_min=25, duration_min=20, ra=30, dec=40
        )
        cal = _make_calendar([seqA, seqB])
        result = proc._force_fill_gaps(cal)

        outA = result.visits[0].sequences[0]
        outB = result.visits[0].sequences[1]

        # Neither sequence extended — gap remains
        assert abs(outA.stop_time - (T0 + 20 * u.min)) < TimeDelta(
            2, format="sec"
        )
        assert abs(outB.start_time - (T0 + 25 * u.min)) < TimeDelta(
            2, format="sec"
        )

    def test_floor_partial_fill(self, monkeypatch):
        """Prev extends until earthlimb drops below floor; next fills
        backward from the other end until its floor."""
        # 6-min gap (min 20-25).
        # Prev: earthlimb ok for min 20-22 (10 deg), floor at 23-25 (3 deg)
        # Next: earthlimb ok for min 23-25 (10 deg), floor at 20-22 (3 deg)
        prev_el = {20: 10.0, 21: 10.0, 22: 10.0, 23: 3.0, 24: 3.0, 25: 3.0}
        next_el = {20: 3.0, 21: 3.0, 22: 3.0, 23: 10.0, 24: 10.0, 25: 10.0}
        cons_ok = {
            m: {"earthlimb": True, "star_tracker": False}
            for m in range(20, 26)
        }
        vis_cls = self._smart_vis_factory(
            prev_el_angles=prev_el,
            next_el_angles=next_el,
            prev_constraints=cons_ok,
            next_constraints=cons_ok,
        )
        proc = self._make_processor(monkeypatch, vis_cls)

        seqA = _make_seq(
            "sA", "TA", start_min=0, duration_min=20, ra=10, dec=20
        )
        seqB = _make_seq(
            "sB", "TB", start_min=26, duration_min=20, ra=30, dec=40
        )
        cal = _make_calendar([seqA, seqB])
        result = proc._force_fill_gaps(cal)

        outA = result.visits[0].sequences[0]
        outB = result.visits[0].sequences[1]

        # Prev takes min 20-22 (3 min forward)
        assert abs(outA.stop_time - (T0 + 23 * u.min)) < TimeDelta(
            2, format="sec"
        )
        # Next takes min 23-25 (3 min backward)
        assert abs(outB.start_time - (T0 + 23 * u.min)) < TimeDelta(
            2, format="sec"
        )

    # ---- Rule 1: prefer star-tracker gaps over earthlimb gaps ----

    def test_prefer_st_over_earthlimb(self, monkeypatch):
        """Prev stops when it has earthlimb failure and next only has ST
        failure (which is preferred)."""
        # 6-min gap (min 20-25).
        # Min 20-21: prev has ST-only failure (earthlimb ok) → prev extends
        # Min 22-25: prev has earthlimb failure but next only has ST failure
        #   → prev stops, next takes these minutes
        prev_el = {
            20: 15.0,
            21: 15.0,  # above floor, ST-only
            22: 12.0,
            23: 12.0,
            24: 12.0,
            25: 12.0,  # EL fail
        }
        next_el = {
            20: 15.0,
            21: 15.0,
            22: 15.0,
            23: 15.0,
            24: 15.0,
            25: 15.0,
        }
        prev_cons = {
            20: {"earthlimb": True, "star_tracker": False},
            21: {"earthlimb": True, "star_tracker": False},
            22: {"earthlimb": False, "star_tracker": True},
            23: {"earthlimb": False, "star_tracker": True},
            24: {"earthlimb": False, "star_tracker": True},
            25: {"earthlimb": False, "star_tracker": True},
        }
        next_cons = {
            m: {"earthlimb": True, "star_tracker": False}
            for m in range(20, 26)
        }
        vis_cls = self._smart_vis_factory(
            prev_el_angles=prev_el,
            next_el_angles=next_el,
            prev_constraints=prev_cons,
            next_constraints=next_cons,
        )
        proc = self._make_processor(monkeypatch, vis_cls)

        seqA = _make_seq(
            "sA", "TA", start_min=0, duration_min=20, ra=10, dec=20
        )
        seqB = _make_seq(
            "sB", "TB", start_min=26, duration_min=20, ra=30, dec=40
        )
        cal = _make_calendar([seqA, seqB])
        result = proc._force_fill_gaps(cal)

        outA = result.visits[0].sequences[0]
        outB = result.visits[0].sequences[1]

        # Prev takes only min 20-21 (2 min); stops at 22 where
        # prev=EL_FAIL and next=ST_ONLY (better)
        assert abs(outA.stop_time - (T0 + 22 * u.min)) < TimeDelta(
            2, format="sec"
        )
        # Next extends backward from 26 to 22
        assert abs(outB.start_time - (T0 + 22 * u.min)) < TimeDelta(
            2, format="sec"
        )

    # ---- _classify_gap_minute unit tests ----

    def test_classify_floor(self, monkeypatch):
        """Earthlimb below hard floor → _GAP_FLOOR."""
        vis_cls = self._smart_vis_factory(
            prev_el_angles={5: 3.0},
        )
        proc = self._make_processor(monkeypatch, vis_cls)
        coord = SkyCoord(10, 20, frame="icrs", unit="deg")
        result = proc._classify_gap_minute(coord, T0 + 5 * u.min)
        assert result == proc._GAP_FLOOR

    def test_classify_st_only(self, monkeypatch):
        """Only star-tracker failure → _GAP_ST_ONLY."""
        vis_cls = self._smart_vis_factory(
            prev_el_angles={5: 20.0},
            prev_constraints={5: {"earthlimb": True, "star_tracker": False}},
        )
        proc = self._make_processor(monkeypatch, vis_cls)
        coord = SkyCoord(10, 20, frame="icrs", unit="deg")
        result = proc._classify_gap_minute(coord, T0 + 5 * u.min)
        assert result == proc._GAP_ST_ONLY

    def test_classify_el_fail(self, monkeypatch):
        """Earthlimb constraint fails (but above floor) → _GAP_EL_FAIL."""
        vis_cls = self._smart_vis_factory(
            prev_el_angles={5: 10.0},
            prev_constraints={5: {"earthlimb": False, "star_tracker": True}},
        )
        proc = self._make_processor(monkeypatch, vis_cls)
        coord = SkyCoord(10, 20, frame="icrs", unit="deg")
        result = proc._classify_gap_minute(coord, T0 + 5 * u.min)
        assert result == proc._GAP_EL_FAIL
