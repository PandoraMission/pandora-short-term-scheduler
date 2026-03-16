"""Tests for enhanced validation reporting.

Covers the new fields, messages, and suggested fixes added to:
- validate_visibility
- validate_no_overlaps_astropy
- validate_sequence_timing
- validate_payload_exposures
- validate_roll_consistency
- print_validation_summary
"""

# Standard library
import xml.etree.ElementTree as ET

# Third-party
import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time, TimeDelta

# First-party/Local
from shortschedule.models import (
    ObservationSequence,
    ScienceCalendar,
    Visit,
)
from shortschedule.scheduler import ScheduleProcessor

# ================================================================
# Helpers
# ================================================================

T0 = Time("2026-01-01T00:00:00", scale="utc")


def _seq(
    sid,
    target,
    start_min,
    dur_min,
    ra=10.0,
    dec=20.0,
    roll=None,
    payload=None,
):
    """Create an ObservationSequence relative to T0."""
    start = T0 + start_min * u.min
    stop = start + dur_min * u.min
    return ObservationSequence(
        id=sid,
        target=target,
        priority=1,
        start_time=start,
        stop_time=stop,
        ra=ra,
        dec=dec,
        payload_params=payload or {},
        roll=roll,
    )


def _cal(sequences, visit_id="v1"):
    visit = Visit(id=visit_id, sequences=sequences)
    return ScienceCalendar(metadata={}, visits=[visit])


def _cal_multi(visit_seq_pairs):
    """Create a calendar with multiple visits.

    Parameters
    ----------
    visit_seq_pairs : list of (str, list[ObservationSequence])
    """
    visits = [Visit(id=vid, sequences=seqs) for vid, seqs in visit_seq_pairs]
    return ScienceCalendar(metadata={}, visits=visits)


def _bare_sched(**attrs):
    """Create a ScheduleProcessor without __init__ (no Visibility).

    Sets useful defaults for attributes used by validators.
    """
    sched = ScheduleProcessor.__new__(ScheduleProcessor)
    sched.min_sequence_duration = TimeDelta(120, format="sec")
    sched._computed_target_rolls = {}
    sched._roll_sweep_enabled = False
    for k, v in attrs.items():
        setattr(sched, k, v)
    return sched


def _make_vda_element(exposure_us, num_frames=None, frames_per_coadd=None):
    root = ET.Element("AcquireVisCamScienceData")
    e = ET.SubElement(root, "ExposureTime_us")
    e.text = str(exposure_us)
    if num_frames is not None:
        nf = ET.SubElement(root, "NumTotalFramesRequested")
        nf.text = str(num_frames)
    if frames_per_coadd is not None:
        fpc = ET.SubElement(root, "FramesPerCoadd")
        fpc.text = str(frames_per_coadd)
    return root


# ================================================================
# Visibility mocks
# ================================================================


class _VisAllTrue:
    """Always visible."""

    def get_visibility(self, coord, times, roll=None):
        try:
            n = len(times)
        except Exception:
            n = 1
        return np.ones(n, dtype=bool)


class _VisHalfFalse:
    """First half of minutes not visible, second half visible.

    Also stores constraint failure data accessible via
    ``get_all_constraints``.
    """

    def get_visibility(self, coord, times, roll=None):
        try:
            n = len(times)
        except Exception:
            n = 1
        vis = np.ones(n, dtype=bool)
        vis[: n // 2] = False
        return vis

    def get_all_constraints(self, coord, time):
        return {
            "moon": True,
            "sun": False,
            "earthlimb": True,
            "star_tracker": False,
        }


class _VisAllFalse:
    """Never visible."""

    def get_visibility(self, coord, times, roll=None):
        try:
            n = len(times)
        except Exception:
            n = 1
        return np.zeros(n, dtype=bool)

    def get_all_constraints(self, coord, time):
        return {
            "moon": False,
            "sun": False,
            "earthlimb": False,
            "star_tracker": False,
        }


class _VisRollFails:
    """Boresight constraints all pass, but roll-aware visibility fails.

    Simulates the case where get_all_constraints returns all-True
    yet get_visibility (called with a roll) returns partially False.
    """

    def get_visibility(self, coord, times, roll=None):
        try:
            n = len(times)
        except Exception:
            n = 1
        vis = np.ones(n, dtype=bool)
        vis[:2] = False  # first 2 minutes fail
        return vis

    def get_all_constraints(self, coord, time):
        return {
            "moon": True,
            "sun": True,
            "earthlimb": True,
            "star_tracker": True,
        }


# ================================================================
# Tests: validate_visibility
# ================================================================


class TestValidateVisibility:
    def test_no_issues_when_fully_visible(self):
        sched = _bare_sched(visibility=_VisAllTrue())
        cal = _cal([_seq("s1", "StarA", 0, 60)])
        issues = sched.validate_visibility(cal, report_issues=False)
        assert issues == []

    def test_partial_visibility_returns_issue_with_all_fields(self):
        sched = _bare_sched(visibility=_VisHalfFalse())
        cal = _cal([_seq("s1", "StarA", 0, 60, ra=45.0, dec=-30.0)])
        issues = sched.validate_visibility(cal, report_issues=False)

        assert len(issues) == 1
        iss = issues[0]

        # Identity fields
        assert iss["sequence_id"] == "s1"
        assert iss["visit_id"] == "v1"
        assert iss["target"] == "StarA"
        assert iss["ra"] == 45.0
        assert iss["dec"] == -30.0

        # Visibility stats
        assert iss["total_minutes"] == 60
        assert iss["non_visible_minutes"] == 30
        assert 0.49 < iss["visibility_fraction"] < 0.51

        # Gap bounds
        assert isinstance(iss["first_gap_start"], Time)
        assert isinstance(iss["last_gap_end"], Time)

        # Constraint breakdown
        assert isinstance(iss["constraint_failures"], dict)
        assert iss["constraint_failures"]["sun"] is False
        assert iss["constraint_failures"]["star_tracker"] is False
        assert "sun" in iss["constraint_summary"]

        # Actionable message
        assert "StarA" in iss["message"]
        assert "s1" in iss["message"]
        assert "v1" in iss["message"]
        assert "50%" in iss["message"]

    def test_fully_dark_all_constraints_fail(self):
        sched = _bare_sched(visibility=_VisAllFalse())
        cal = _cal([_seq("s1", "T", 0, 10)])
        issues = sched.validate_visibility(cal, report_issues=False)
        assert len(issues) == 1
        iss = issues[0]
        assert iss["non_visible_minutes"] == 10
        assert iss["visibility_fraction"] == 0.0
        # All four constraints fail
        for k in ("moon", "sun", "earthlimb", "star_tracker"):
            assert iss["constraint_failures"][k] is False

    def test_roll_included_when_roll_sweep_enabled(self):
        sched = _bare_sched(
            visibility=_VisHalfFalse(),
            _roll_sweep_enabled=True,
            _computed_target_rolls={"v1": {"StarA": 42.0}},
        )
        cal = _cal([_seq("s1", "StarA", 0, 20)])
        issues = sched.validate_visibility(cal, report_issues=False)
        assert len(issues) == 1
        assert issues[0]["roll"] == 42.0

    def test_roll_dependent_constraint_detected(self):
        """When boresight constraints all pass but roll-aware
        visibility fails, the summary should indicate the star
        tracker constraint at the specific roll angle."""
        sched = _bare_sched(
            visibility=_VisRollFails(),
            _roll_sweep_enabled=True,
            _computed_target_rolls={"v1": {"StarA": 170.0}},
        )
        cal = _cal([_seq("s1", "StarA", 0, 20)])
        issues = sched.validate_visibility(cal, report_issues=False)
        assert len(issues) == 1
        iss = issues[0]
        assert "star_tracker" in iss["constraint_summary"]
        assert "170.0" in iss["constraint_summary"]
        assert iss["constraint_failures"].get("star_tracker_at_roll") is False
        assert "star_tracker" in iss["message"]

    def test_roll_none_when_sweep_disabled(self):
        sched = _bare_sched(visibility=_VisHalfFalse())
        cal = _cal([_seq("s1", "StarA", 0, 20)])
        issues = sched.validate_visibility(cal, report_issues=False)
        assert issues[0]["roll"] is None


# ================================================================
# Tests: validate_no_overlaps_astropy
# ================================================================


class TestValidateNoOverlaps:
    def test_no_overlaps_when_sequences_are_disjoint(self):
        sched = _bare_sched()
        cal = _cal(
            [
                _seq("a", "T1", 0, 10),
                _seq("b", "T2", 10, 10),
            ]
        )
        assert (
            sched.validate_no_overlaps_astropy(cal, report_issues=False) == []
        )

    def test_overlap_returns_all_new_fields(self):
        sched = _bare_sched()
        # seq1 ends at minute 12, seq2 starts at minute 10 → 2 min overlap
        cal = _cal(
            [
                _seq("a1", "T1", 0, 12),
                _seq("a2", "T2", 10, 10),
            ]
        )
        overlaps = sched.validate_no_overlaps_astropy(cal, report_issues=False)
        assert len(overlaps) == 1
        ov = overlaps[0]

        # Original keys preserved
        assert ov["sequence1_id"] == "a1"
        assert ov["sequence2_id"] == "a2"
        assert ov["sequence1_target"] == "T1"
        assert ov["sequence2_target"] == "T2"

        # New keys
        assert ov["visit1_id"] == "v1"
        assert ov["visit2_id"] == "v1"
        assert isinstance(ov["overlap_duration_minutes"], float)
        assert ov["overlap_duration_minutes"] > 1.5

        # Suggested fix
        assert "a2" in ov["suggested_fix"]
        assert "a1" in ov["suggested_fix"]

        # Actionable message
        assert "overlap" in ov["message"].lower()
        assert "a1" in ov["message"]
        assert "a2" in ov["message"]

    def test_overlap_across_visits(self):
        sched = _bare_sched()
        s1 = _seq("x", "T1", 0, 15)
        s2 = _seq("y", "T2", 10, 15)
        cal = _cal_multi([("v1", [s1]), ("v2", [s2])])
        overlaps = sched.validate_no_overlaps_astropy(cal, report_issues=False)
        assert len(overlaps) == 1
        assert overlaps[0]["visit1_id"] == "v1"
        assert overlaps[0]["visit2_id"] == "v2"


# ================================================================
# Tests: validate_sequence_timing
# ================================================================


class TestValidateSequenceTiming:
    def test_clean_calendar_has_zero_issues(self):
        sched = _bare_sched()
        cal = _cal(
            [
                _seq("s1", "T", 0, 10),
                _seq("s2", "T", 10, 10),
            ]
        )
        result = sched.validate_sequence_timing(cal, report_issues=False)
        assert result["timing_summary"]["total_issues"] == 0
        assert result["overlaps"] == []
        assert result["short_sequences"] == []
        assert result["large_gaps"] == []

    def test_short_sequence_returns_new_fields(self):
        sched = _bare_sched(
            min_sequence_duration=TimeDelta(5 * 60, format="sec"),
        )
        # 1-minute sequence  — well under 5-minute minimum
        cal = _cal([_seq("sh", "T", 0, 1)])
        result = sched.validate_sequence_timing(cal, report_issues=False)
        shorts = result["short_sequences"]
        assert len(shorts) == 1
        s = shorts[0]

        assert s["sequence_id"] == "sh"
        assert s["visit_id"] == "v1"
        assert s["target"] == "T"
        assert isinstance(s["duration_minutes"], float)
        assert s["duration_minutes"] < 2.0
        assert isinstance(s["minimum_required_minutes"], float)
        assert s["minimum_required_minutes"] == 5.0

        # Suggested fix and message
        assert "stop_time" in s["suggested_fix"]
        assert "sh" in s["message"]
        assert "T" in s["message"]

    def test_large_gap_returns_new_fields(self):
        sched = _bare_sched()
        # 5-minute gap between two 5-minute sequences
        cal = _cal(
            [
                _seq("s1", "A", 0, 5),
                _seq("s2", "B", 10, 5),
            ]
        )
        result = sched.validate_sequence_timing(cal, report_issues=False)
        gaps = result["large_gaps"]
        assert len(gaps) == 1
        g = gaps[0]

        assert g["after_sequence"] == "s1"
        assert g["before_sequence"] == "s2"
        assert g["after_visit_id"] == "v1"
        assert g["before_visit_id"] == "v1"
        assert isinstance(g["gap_duration_minutes"], float)
        assert g["gap_duration_minutes"] == pytest.approx(5.0, abs=0.1)

        # Times are astropy Time objects
        assert isinstance(g["gap_start"], Time)
        assert isinstance(g["gap_end"], Time)

        # Message
        assert "s1" in g["message"]
        assert "s2" in g["message"]

    def test_timing_summary_counts(self):
        sched = _bare_sched(
            min_sequence_duration=TimeDelta(5 * 60, format="sec"),
        )
        cal = _cal(
            [
                _seq("s1", "T", 0, 12),  # overlaps s2
                _seq("s2", "T", 10, 1),  # short
                # 20-minute gap
                _seq("s3", "T", 31, 5),
            ]
        )
        result = sched.validate_sequence_timing(cal, report_issues=False)
        summary = result["timing_summary"]
        assert summary["overlaps_found"] >= 1
        assert summary["short_sequences_found"] >= 1
        assert summary["large_gaps_found"] >= 1
        assert summary["total_issues"] == (
            summary["overlaps_found"]
            + summary["short_sequences_found"]
            + summary["large_gaps_found"]
        )


# ================================================================
# Tests: validate_payload_exposures
# ================================================================


class TestValidatePayloadExposures:
    def _sched_with_overheads(self):
        sched = _bare_sched()
        sched.vda_pre_sequence_overhead = 260 * u.s
        sched.vda_post_sequence_overhead = 120 * u.s
        sched.nirda_pre_sequence_overhead = 258 * u.s
        sched.nirda_post_sequence_overhead = 120 * u.s
        return sched

    def test_no_issues_when_exposure_fits(self):
        sched = self._sched_with_overheads()
        # 600s sequence, 380s overhead → 220s effective; 100s exposure OK
        payload = {
            "AcquireVisCamScienceData": _make_vda_element(
                100_000_000, num_frames=2
            ),
        }
        cal = _cal([_seq("s1", "T", 0, 10, payload=payload)])
        issues = sched.validate_payload_exposures(cal, report_issues=False)
        assert issues == []

    def test_single_exposure_over_effective_duration(self):
        sched = self._sched_with_overheads()
        # 600s sequence, 380s overhead → 220s effective
        # single exposure = 300s (> 220s effective)
        payload = {
            "AcquireVisCamScienceData": _make_vda_element(300_000_000),
        }
        cal = _cal([_seq("s1", "T", 0, 10, payload=payload)])
        issues = sched.validate_payload_exposures(cal, report_issues=False)
        assert len(issues) >= 1
        iss = issues[0]
        assert iss["problem"] == "single_exposure_longer_than_sequence"
        assert iss["visit_id"] == "v1"
        assert iss["sequence_id"] == "s1"
        assert iss["target"] == "T"
        assert iss["exposure_seconds"] == 300.0
        assert iss["sequence_duration_seconds"] == pytest.approx(
            600.0, abs=0.1
        )
        assert iss["effective_duration_seconds"] == pytest.approx(
            220.0, abs=0.1
        )
        assert iss["overhead_seconds"] == pytest.approx(380.0)
        assert "s1" in iss["message"]
        assert "suggested_fix" in iss

    def test_total_exposure_over_effective_duration(self):
        sched = self._sched_with_overheads()
        # effective = 220s; 50s × 5 frames = 250s > 220s
        payload = {
            "AcquireVisCamScienceData": _make_vda_element(
                50_000_000, num_frames=5
            ),
        }
        cal = _cal([_seq("s1", "T", 0, 10, payload=payload)])
        issues = sched.validate_payload_exposures(cal, report_issues=False)
        total_issues = [
            i
            for i in issues
            if i["problem"] == "total_exposure_longer_than_sequence"
        ]
        assert len(total_issues) >= 1
        iss = total_issues[0]
        assert iss["effective_duration_seconds"] == pytest.approx(220.0)
        assert iss["overhead_seconds"] == pytest.approx(380.0)
        assert iss["suggested_max_frames"] == 4  # 220 / 50 = 4
        assert "5" not in iss["suggested_fix"]  # should suggest <= 4
        assert "s1" in iss["message"]

    def test_coadd_exposure_returns_new_fields(self):
        sched = self._sched_with_overheads()
        # effective = 220s; 100s × 3 coadd = 300s > 220s
        payload = {
            "AcquireVisCamScienceData": _make_vda_element(
                100_000_000, frames_per_coadd=3
            ),
        }
        cal = _cal([_seq("s1", "T", 0, 10, payload=payload)])
        issues = sched.validate_payload_exposures(cal, report_issues=False)
        coadd = [
            i
            for i in issues
            if i["problem"] == "coadd_exposure_longer_than_sequence"
        ]
        assert len(coadd) >= 1
        assert coadd[0]["effective_duration_seconds"] == pytest.approx(220.0)
        assert "message" in coadd[0]
        assert "suggested_fix" in coadd[0]

    def test_heuristic_field_returns_new_fields(self):
        sched = self._sched_with_overheads()
        root = ET.Element("AcquireInfCamImages")
        sub = ET.SubElement(root, "SomeExposure")
        sub.text = "300"  # 300s > effective 220s
        payload = {"AcquireInfCamImages": root}
        cal = _cal([_seq("s1", "T", 0, 10, payload=payload)])
        issues = sched.validate_payload_exposures(cal, report_issues=False)
        heuristic = [
            i
            for i in issues
            if i["problem"] == "payload_exposure_field_longer_than_sequence"
        ]
        assert len(heuristic) >= 1
        h = heuristic[0]
        assert h["field"] == "AcquireInfCamImages.SomeExposure"
        assert h["effective_duration_seconds"] == pytest.approx(220.0)
        assert "message" in h
        assert "suggested_fix" in h

    def test_zero_overhead_when_attrs_missing(self):
        """Bare sched without overhead attrs -> 0s overhead."""
        sched = ScheduleProcessor.__new__(ScheduleProcessor)
        # 3s sequence, 6s exposure -> flagged with 0 overhead
        payload = {
            "AcquireVisCamScienceData": _make_vda_element(6_000_000),  # 6s
        }
        cal = _cal([_seq("s1", "T", 0, 0.05, payload=payload)])
        issues = sched.validate_payload_exposures(cal, report_issues=False)
        assert len(issues) >= 1
        assert issues[0]["overhead_seconds"] == 0.0


# ================================================================
# Tests: validate_roll_consistency
# ================================================================


class TestValidateRollConsistency:
    def test_no_issues_when_rolls_consistent(self):
        sched = _bare_sched()
        cal = _cal(
            [
                _seq("s1", "StarA", 0, 10, roll=45.0),
                _seq("s2", "StarA", 10, 10, roll=45.0),
            ]
        )
        issues = sched.validate_roll_consistency(cal, report_issues=False)
        assert issues == []

    def test_single_sequence_per_target_no_issue(self):
        sched = _bare_sched()
        cal = _cal(
            [
                _seq("s1", "StarA", 0, 10, roll=45.0),
                _seq("s2", "StarB", 10, 10, roll=90.0),
            ]
        )
        issues = sched.validate_roll_consistency(cal, report_issues=False)
        assert issues == []

    def test_inconsistent_rolls_returns_all_fields(self):
        sched = _bare_sched()
        cal = _cal(
            [
                _seq("s1", "StarA", 0, 10, roll=45.0),
                _seq("s2", "StarA", 10, 10, roll=50.0),
                _seq("s3", "StarA", 20, 10, roll=45.0),
            ]
        )
        issues = sched.validate_roll_consistency(cal, report_issues=False)
        assert len(issues) == 1
        iss = issues[0]

        assert iss["visit_id"] == "v1"
        assert iss["target"] == "StarA"
        assert set(iss["sequence_ids"]) == {"s1", "s2", "s3"}
        assert len(iss["roll_values"]) == 3
        assert iss["max_difference_deg"] == pytest.approx(5.0)

        # New fields
        assert isinstance(iss["roll_map"], dict)
        assert iss["roll_map"]["s1"] == 45.0
        assert iss["roll_map"]["s2"] == 50.0
        assert isinstance(iss["suggested_roll"], float)
        assert iss["suggested_roll"] == pytest.approx(45.0)

        # Suggested fix and message
        assert "StarA" in iss["suggested_fix"]
        assert "v1" in iss["suggested_fix"]
        assert "StarA" in iss["message"]
        assert "5.000" in iss["message"]

    def test_none_rolls_skipped(self):
        sched = _bare_sched()
        cal = _cal(
            [
                _seq("s1", "StarA", 0, 10, roll=None),
                _seq("s2", "StarA", 10, 10, roll=45.0),
            ]
        )
        issues = sched.validate_roll_consistency(cal, report_issues=False)
        assert issues == []


# ================================================================
# Tests: print_validation_summary
# ================================================================


class TestPrintValidationSummary:
    def test_valid_calendar_returns_valid(self):
        sched = _bare_sched(visibility=_VisAllTrue())
        cal = _cal(
            [
                _seq("s1", "T", 0, 10),
                _seq("s2", "T", 10, 10),
            ]
        )
        result = sched.print_validation_summary(cal)
        assert result["status"] == "VALID"
        assert sum(result["counts"].values()) == 0
        assert result["details"] == {}

    def test_invalid_calendar_returns_counts(self):
        sched = _bare_sched(visibility=_VisHalfFalse())
        # overlapping + visibility issues
        cal = _cal(
            [
                _seq("s1", "T1", 0, 15),
                _seq("s2", "T2", 10, 10),
            ]
        )
        result = sched.print_validation_summary(cal)
        assert result["status"] == "INVALID"
        assert sum(result["counts"].values()) > 0
        # Should have at least visibility and overlap categories
        assert "visibility" in result["counts"]
        assert "overlap" in result["counts"]

    def test_details_contain_raw_issue_lists(self):
        sched = _bare_sched(visibility=_VisAllFalse())
        cal = _cal([_seq("s1", "T", 0, 10)])
        result = sched.print_validation_summary(cal)
        assert "visibility" in result["details"]
        vis_issues = result["details"]["visibility"]
        assert isinstance(vis_issues, list)
        assert len(vis_issues) == 1
        assert vis_issues[0]["sequence_id"] == "s1"

    def test_timing_issues_in_summary(self):
        sched = _bare_sched(
            visibility=_VisAllTrue(),
            min_sequence_duration=TimeDelta(600, format="sec"),
        )
        # Short sequence triggers timing issue
        cal = _cal([_seq("s1", "T", 0, 1)])
        result = sched.print_validation_summary(cal)
        assert "sequence_timing" in result["counts"]
        assert result["counts"]["sequence_timing"] >= 1
