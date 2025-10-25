# Third-party
from astropy.time import Time, TimeDelta

# First-party/Local
from shortschedule.models import ObservationSequence, ScienceCalendar, Visit
from shortschedule.scheduler import ScheduleProcessor


def test_validate_no_overlaps_astropy_detects_overlap():
    start = Time("2025-01-01T00:00:00", scale="utc")
    # seq1: 0 -> 120s
    seq1 = ObservationSequence(
        id="a1",
        target="A",
        priority=1,
        start_time=start,
        stop_time=start + TimeDelta(120, format="sec"),
        ra=0.0,
        dec=0.0,
        payload_params={},
    )
    # seq2: starts at 60s -> overlap with seq1
    seq2 = ObservationSequence(
        id="a2",
        target="B",
        priority=1,
        start_time=start + TimeDelta(60, format="sec"),
        stop_time=start + TimeDelta(180, format="sec"),
        ra=0.0,
        dec=0.0,
        payload_params={},
    )

    visit = Visit(id="v1", sequences=[seq1, seq2])
    cal = ScienceCalendar(metadata={}, visits=[visit])

    sched = ScheduleProcessor.__new__(ScheduleProcessor)
    overlaps = sched.validate_no_overlaps_astropy(cal, report_issues=False)
    assert len(overlaps) >= 1
    assert overlaps[0]["sequence1_id"] == "a1"
    assert overlaps[0]["sequence2_id"] == "a2"


def test_get_minute_by_minute_assignments_correctness():
    # Two back-to-back sequences of 2 minutes each
    start = Time("2025-01-01T00:00:00", scale="utc")
    s1 = ObservationSequence(
        id="s1",
        target="T1",
        priority=1,
        start_time=start,
        stop_time=start + TimeDelta(2 * 60, format="sec"),
        ra=0.0,
        dec=0.0,
        payload_params={},
    )
    s2 = ObservationSequence(
        id="s2",
        target="T2",
        priority=1,
        start_time=start + TimeDelta(2 * 60, format="sec"),
        stop_time=start + TimeDelta(4 * 60, format="sec"),
        ra=0.0,
        dec=0.0,
        payload_params={},
    )

    visit = Visit(id="vX", sequences=[s1, s2])
    cal = ScienceCalendar(metadata={}, visits=[visit])

    sched = ScheduleProcessor.__new__(ScheduleProcessor)
    result = sched.get_minute_by_minute_assignments(cal)
    assignments = result["assignments"]

    # Total minutes should be 4
    assert len(assignments) == 4
    # First two minutes assigned to s1, next two to s2
    assert assignments[0]["sequence_id"] == "s1"
    assert assignments[1]["sequence_id"] == "s1"
    assert assignments[2]["sequence_id"] == "s2"
    assert assignments[3]["sequence_id"] == "s2"


def test_validate_sequence_timing_flags_short_sequence():
    # Sequence shorter than min_sequence_duration should be flagged
    start = Time("2025-01-01T00:00:00", scale="utc")
    short_seq = ObservationSequence(
        id="short",
        target="S",
        priority=1,
        start_time=start,
        stop_time=start + TimeDelta(30, format="sec"),
        ra=0.0,
        dec=0.0,
        payload_params={},
    )
    visit = Visit(id="vshort", sequences=[short_seq])
    cal = ScienceCalendar(metadata={}, visits=[visit])

    sched = ScheduleProcessor.__new__(ScheduleProcessor)
    # set minimum sequence duration to 2 minutes (as typical constructor does)
    sched.min_sequence_duration = TimeDelta(2 * 60, format="sec")

    issues = sched.validate_sequence_timing(cal, report_issues=False)
    assert issues[
        "short_sequences"
    ], "Expected short_sequences to be non-empty"
    assert issues["short_sequences"][0]["sequence_id"] == "short"
