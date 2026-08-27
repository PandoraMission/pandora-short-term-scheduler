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


def _bare_processor(earthlimb_gap_tolerance=0, st_gap_tolerance=0):
    """A ScheduleProcessor carrying only what merging looks at.

    The tolerances default to zero, so a gap between two observations
    prevents merging unless a test opts into bridging one.
    """
    proc = ScheduleProcessor.__new__(ScheduleProcessor)
    proc.visibility = None
    proc._computed_target_rolls = {}
    proc.earthlimb_gap_tolerance = earthlimb_gap_tolerance
    proc.st_gap_tolerance = st_gap_tolerance
    return proc


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
    def test_merge_enabled_by_default(self, monkeypatch, tmp_path):
        """Omitting the kwarg merges: back-to-back same-target observations
        are an artifact of how the long-term calendar splits visits, so the
        delivered calendar should join them unless asked not to."""
        monkeypatch.setattr(
            "shortschedule.scheduler.Visibility",
            _DummyVisibilityAllTrue,
        )
        cal = self._load_sample()
        first_seq = cal.visits[0].sequences[0]
        sched = ScheduleProcessor("L1", "L2")

        merged = []
        original = sched._merge_similar_observations

        def _spy(calendar, verbose=False):
            merged.append(calendar)
            return original(calendar, verbose)

        monkeypatch.setattr(sched, "_merge_similar_observations", _spy)

        processed = sched.process_calendar(
            cal,
            window_start=first_seq.start_time.isot,
            window_duration_days=1,
            log_path=tmp_path / "run",
        )
        assert merged, "merging should run when the kwarg is omitted"
        assert processed is not None


# ================================================================
# Tests: bridging a tolerable keepout gap between two observations
# ================================================================


class _DarkGapVis:
    """Visibility where a named span is dark for a star-tracker reason.

    A brief tracker dropout between two observations of one target is the
    case this exists for: the same dropout inside an observation would be
    ridden out under ``st_gap_tolerance``, so an observation boundary
    landing on it should not change the outcome.
    """

    _st_constraint_active = True

    def __init__(self, dark_from_min, dark_to_min):
        self.dark = range(dark_from_min, dark_to_min)

    def _minutes(self, times):
        return np.rint((times - T0).sec / 60.0).astype(int)

    def get_visibility(self, coord, times, roll=None):
        return np.array(
            [i not in self.dark for i in np.atleast_1d(self._minutes(times))],
            dtype=bool,
        )

    def get_all_constraints(self, coord, time, roll=None):
        # Boresight is clear; the trackers are what drop out.
        return {"moon": True, "sun": True, "earthlimb": True}

    def get_star_tracker_breakdown(self, coord, time, roll=None, pre=None):
        minutes = np.atleast_1d(self._minutes(time))
        combined = np.array([i not in self.dark for i in minutes], dtype=bool)
        return {
            "passed": {"combined": combined[0] if time.isscalar else combined}
        }


def _pair_across_gap(gap_minutes=2, roll_a=30.0, roll_b=30.0):
    """Two observations of one target separated by *gap_minutes*."""
    first = _make_seq("s1", "TargetA", start_min=0, duration_min=20)
    second = _make_seq(
        "s2", "TargetA", start_min=20 + gap_minutes, duration_min=20
    )
    first.roll = roll_a
    second.roll = roll_b
    return first, second


class TestBridgingATolerableGap:
    def test_tolerable_tracker_dropout_is_absorbed(self):
        """A dropout short enough to ride out joins the two observations."""
        first, second = _pair_across_gap(gap_minutes=2)
        proc = _bare_processor(st_gap_tolerance=12)
        proc.visibility = _DarkGapVis(20, 22)
        cal = _make_calendar([first, second])

        result = proc._merge_similar_observations(cal)

        merged = result.visits[0].sequences
        assert len(merged) == 1
        assert abs((merged[0].stop_time - second.stop_time).sec) < 1

    def test_gap_longer_than_the_tolerance_is_not_absorbed(self):
        first, second = _pair_across_gap(gap_minutes=20)
        proc = _bare_processor(st_gap_tolerance=12)
        proc.visibility = _DarkGapVis(20, 40)
        cal = _make_calendar([first, second])

        result = proc._merge_similar_observations(cal)

        assert len(result.visits[0].sequences) == 2

    def test_different_rolls_are_never_joined(self):
        """One observation flies one attitude."""
        first, second = _pair_across_gap(
            gap_minutes=2, roll_a=30.0, roll_b=95.0
        )
        proc = _bare_processor(st_gap_tolerance=12)
        proc.visibility = _DarkGapVis(20, 22)
        cal = _make_calendar([first, second])

        result = proc._merge_similar_observations(cal)

        assert len(result.visits[0].sequences) == 2

    def test_equivalent_rolls_across_the_wrap_are_joined(self):
        """-180 and +180 are the same attitude."""
        first, second = _pair_across_gap(
            gap_minutes=2, roll_a=180.0, roll_b=-180.0
        )
        proc = _bare_processor(st_gap_tolerance=12)
        proc.visibility = _DarkGapVis(20, 22)
        cal = _make_calendar([first, second])

        result = proc._merge_similar_observations(cal)

        assert len(result.visits[0].sequences) == 1

    def test_an_observation_inside_the_gap_blocks_the_merge(self):
        """Even one in another visit, since visits interleave in time."""
        first, second = _pair_across_gap(gap_minutes=4)
        intruder = _make_seq("x1", "Other", start_min=21, duration_min=2)
        intruder.roll = 30.0
        proc = _bare_processor(st_gap_tolerance=12)
        proc.visibility = _DarkGapVis(20, 24)
        cal = ScienceCalendar(
            metadata={},
            visits=[
                Visit(id="v1", sequences=[first, second]),
                Visit(id="v2", sequences=[intruder]),
            ],
        )

        result = proc._merge_similar_observations(cal)

        assert len(result.visits[0].sequences) == 2

    def test_gap_with_no_tolerance_configured_is_not_absorbed(self):
        first, second = _pair_across_gap(gap_minutes=2)
        proc = _bare_processor()  # both tolerances zero
        proc.visibility = _DarkGapVis(20, 22)
        cal = _make_calendar([first, second])

        result = proc._merge_similar_observations(cal)

        assert len(result.visits[0].sequences) == 2
