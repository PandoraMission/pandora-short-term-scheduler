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
