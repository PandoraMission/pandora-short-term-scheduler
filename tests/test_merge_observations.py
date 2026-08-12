"""Tests for merging adjacent same-target observations in the scheduler.

Covers ScheduleProcessor._merge_similar_observations and its integration
into process_calendar via the merge_similar_observations kwarg:
- Adjacent same-target sequences in a visit are merged
- Transitive merging of three or more contiguous sequences
- Different targets / pointings are not merged
- Non-contiguous (gapped) sequences are not merged
- Merges never cross visit boundaries
- The merged sequence keeps the first sequence's identity/payload and the
  second sequence's stop time
- The input calendar is not mutated
- End-to-end wiring through process_calendar
"""

# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

# First-party/Local
from shortschedule.models import ObservationSequence, ScienceCalendar, Visit
from shortschedule.scheduler import ScheduleProcessor

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


def _bare_processor():
    """A ScheduleProcessor with no Visibility; merge needs no other state."""
    return ScheduleProcessor.__new__(ScheduleProcessor)


def _seq_by_id(visit, sid):
    return next((s for s in visit.sequences if s.id == sid), None)


# ================================================================
# Tests: _merge_similar_observations
# ================================================================


class TestMergeSimilarObservations:
    """Unit tests for ScheduleProcessor._merge_similar_observations."""

    def test_adjacent_same_target_merged(self):
        """Two back-to-back same-target sequences collapse into one."""
        proc = _bare_processor()
        seqA = _make_seq("s1", "TargetA", start_min=0, duration_min=20)
        seqB = _make_seq("s2", "TargetA", start_min=20, duration_min=30)
        cal = _make_calendar([seqA, seqB])

        result = proc._merge_similar_observations(cal)

        seqs = result.visits[0].sequences
        assert len(seqs) == 1
        merged = seqs[0]
        # Keeps the first sequence's identity, spans both durations.
        assert merged.id == "s1"
        assert merged.start_time == seqA.start_time
        assert merged.stop_time == seqB.stop_time

    def test_three_contiguous_merge_transitively(self):
        """A run of three contiguous sequences collapses to one."""
        proc = _bare_processor()
        seqs_in = [
            _make_seq("s1", "TargetA", start_min=0, duration_min=10),
            _make_seq("s2", "TargetA", start_min=10, duration_min=10),
            _make_seq("s3", "TargetA", start_min=20, duration_min=10),
        ]
        cal = _make_calendar(seqs_in)

        result = proc._merge_similar_observations(cal)

        seqs = result.visits[0].sequences
        assert len(seqs) == 1
        assert seqs[0].id == "s1"
        assert seqs[0].stop_time == seqs_in[-1].stop_time

    def test_different_targets_not_merged(self):
        """Adjacent sequences with different targets are left alone."""
        proc = _bare_processor()
        seqA = _make_seq("s1", "TargetA", start_min=0, duration_min=20)
        seqB = _make_seq("s2", "TargetB", start_min=20, duration_min=20)
        cal = _make_calendar([seqA, seqB])

        result = proc._merge_similar_observations(cal)

        assert len(result.visits[0].sequences) == 2

    def test_same_target_different_pointing_not_merged(self):
        """Same target name but different RA/Dec is not merged."""
        proc = _bare_processor()
        seqA = _make_seq(
            "s1", "TargetA", start_min=0, duration_min=20, ra=10.0
        )
        seqB = _make_seq(
            "s2", "TargetA", start_min=20, duration_min=20, ra=42.0
        )
        cal = _make_calendar([seqA, seqB])

        result = proc._merge_similar_observations(cal)

        assert len(result.visits[0].sequences) == 2

    def test_gapped_sequences_not_merged(self):
        """A time gap between same-target sequences prevents merging."""
        proc = _bare_processor()
        seqA = _make_seq("s1", "TargetA", start_min=0, duration_min=20)
        # Starts 10 min after seqA stops.
        seqB = _make_seq("s2", "TargetA", start_min=30, duration_min=20)
        cal = _make_calendar([seqA, seqB])

        result = proc._merge_similar_observations(cal)

        assert len(result.visits[0].sequences) == 2

    def test_no_merge_across_visits(self):
        """Identical adjacent sequences in different visits stay separate."""
        proc = _bare_processor()
        seqA = _make_seq("s1", "TargetA", start_min=0, duration_min=20)
        seqB = _make_seq("s1", "TargetA", start_min=20, duration_min=20)
        cal = ScienceCalendar(
            metadata={},
            visits=[
                Visit(id="v1", sequences=[seqA]),
                Visit(id="v2", sequences=[seqB]),
            ],
        )

        result = proc._merge_similar_observations(cal)

        assert len(result.visits) == 2
        assert len(result.visits[0].sequences) == 1
        assert len(result.visits[1].sequences) == 1

    def test_merge_mixed_run(self):
        """Only the contiguous same-target prefix merges; rest preserved."""
        proc = _bare_processor()
        seqs_in = [
            _make_seq("s1", "TargetA", start_min=0, duration_min=10),
            _make_seq("s2", "TargetA", start_min=10, duration_min=10),
            _make_seq("s3", "TargetB", start_min=20, duration_min=10),
            _make_seq("s4", "TargetB", start_min=30, duration_min=10),
        ]
        cal = _make_calendar(seqs_in)

        result = proc._merge_similar_observations(cal)

        seqs = result.visits[0].sequences
        assert [s.id for s in seqs] == ["s1", "s3"]
        assert seqs[0].stop_time == seqs_in[1].stop_time  # s1+s2
        assert seqs[1].stop_time == seqs_in[3].stop_time  # s3+s4

    def test_unsorted_input_is_ordered_before_merge(self):
        """Out-of-order sequences are merged by chronological adjacency."""
        proc = _bare_processor()
        seqA = _make_seq("s1", "TargetA", start_min=0, duration_min=20)
        seqB = _make_seq("s2", "TargetA", start_min=20, duration_min=20)
        cal = _make_calendar([seqB, seqA])  # reversed order

        result = proc._merge_similar_observations(cal)

        seqs = result.visits[0].sequences
        assert len(seqs) == 1
        assert seqs[0].id == "s1"
        assert seqs[0].start_time == seqA.start_time
        assert seqs[0].stop_time == seqB.stop_time

    def test_input_calendar_not_mutated(self):
        """The original calendar/sequences are left untouched."""
        proc = _bare_processor()
        seqA = _make_seq("s1", "TargetA", start_min=0, duration_min=20)
        seqB = _make_seq("s2", "TargetA", start_min=20, duration_min=30)
        original_stop = seqA.stop_time
        cal = _make_calendar([seqA, seqB])

        proc._merge_similar_observations(cal)

        # Original visit still has both sequences, unchanged.
        assert len(cal.visits[0].sequences) == 2
        assert cal.visits[0].sequences[0].stop_time == original_stop


# ================================================================
# Tests: integration via process_calendar
# ================================================================


class _DummyVisibilityAllTrue:
    """Visibility mock — always visible, ignores roll."""

    def __init__(self, l1, l2, **kwargs):
        pass

    def get_visibility(self, coord, times, roll=None):
        try:
            n = len(times)
        except Exception:
            return np.array([True], dtype=bool)
        return np.ones(n, dtype=bool)


class TestProcessCalendarMergeKwarg:
    """process_calendar(merge_similar_observations=...) wiring."""

    def _load_sample(self):
        import shortschedule

        sample = (
            Path(shortschedule.__file__).parent
            / "data"
            / "Pandora_science_calendar_20251018_tsb-futz.xml"
        )
        from shortschedule.parser import parse_science_calendar

        cal = parse_science_calendar(sample)
        if not cal.visits:
            pytest.skip("Sample calendar has no visits")
        return cal

    @pytest.mark.slow
    def test_merge_reduces_sequence_count(self, monkeypatch, tmp_path):
        """Enabling the merge never increases the sequence count and
        produces no zero-length result."""
        monkeypatch.setattr(
            "shortschedule.scheduler.Visibility",
            _DummyVisibilityAllTrue,
        )
        cal = self._load_sample()
        first_seq = cal.visits[0].sequences[0]

        sched_off = ScheduleProcessor("L1", "L2")
        off = sched_off.process_calendar(
            cal.copy(),
            window_start=first_seq.start_time.isot,
            window_duration_days=1,
            merge_similar_observations=False,
            log_path=tmp_path / "off",
        )

        sched_on = ScheduleProcessor("L1", "L2")
        on = sched_on.process_calendar(
            cal.copy(),
            window_start=first_seq.start_time.isot,
            window_duration_days=1,
            merge_similar_observations=True,
            log_path=tmp_path / "on",
        )

        n_off = sum(len(v.sequences) for v in off.visits)
        n_on = sum(len(v.sequences) for v in on.visits)
        assert n_on <= n_off
        assert n_on > 0

    @pytest.mark.slow
    def test_merge_disabled_by_default(self, monkeypatch, tmp_path):
        """Omitting the kwarg leaves merging off (no error, processes)."""
        monkeypatch.setattr(
            "shortschedule.scheduler.Visibility",
            _DummyVisibilityAllTrue,
        )
        cal = self._load_sample()
        first_seq = cal.visits[0].sequences[0]
        sched = ScheduleProcessor("L1", "L2")
        processed = sched.process_calendar(
            cal,
            window_start=first_seq.start_time.isot,
            window_duration_days=1,
            log_path=tmp_path / "run",
        )
        assert processed is not None
