# Standard library
import xml.etree.ElementTree as ET

# Third-party
from astropy.time import Time, TimeDelta

# First-party/Local
from shortschedule.models import ObservationSequence, ScienceCalendar, Visit
from shortschedule.scheduler import ScheduleProcessor


def make_vda_element(exposure_us, num_frames=None, frames_per_coadd=None):
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


def test_validator_detects_total_exposure_over_duration():
    # create a short sequence (10 seconds long)
    start = Time("2025-01-01T00:00:00", scale="utc")
    stop = start + TimeDelta(10, format="sec")
    payload = {
        "AcquireVisCamScienceData": make_vda_element(1_000_000, num_frames=20)
    }
    seq = ObservationSequence(
        id="s1",
        target="T",
        priority=1,
        start_time=start,
        stop_time=stop,
        ra=0.0,
        dec=0.0,
        payload_params=payload,
    )
    visit = Visit(id="v1", sequences=[seq])
    cal = ScienceCalendar(metadata={}, visits=[visit])

    # Instantiate ScheduleProcessor without calling __init__ to avoid Visibility side-effects
    sched = ScheduleProcessor.__new__(ScheduleProcessor)
    issues = sched.validate_payload_exposures(cal, report_issues=False)
    assert any(
        i.get("problem") == "total_exposure_longer_than_sequence"
        for i in issues
    )


def test_validator_detects_single_exposure_over_duration():
    # sequence duration 5 seconds, single exposure 12 seconds
    start = Time("2025-01-01T00:00:00", scale="utc")
    stop = start + TimeDelta(5, format="sec")
    payload = {"AcquireVisCamScienceData": make_vda_element(12_000_000)}
    seq = ObservationSequence(
        id="s2",
        target="U",
        priority=1,
        start_time=start,
        stop_time=stop,
        ra=0.0,
        dec=0.0,
        payload_params=payload,
    )
    visit = Visit(id="v2", sequences=[seq])
    cal = ScienceCalendar(metadata={}, visits=[visit])

    sched = ScheduleProcessor.__new__(ScheduleProcessor)
    issues = sched.validate_payload_exposures(cal, report_issues=False)
    assert any(
        i.get("problem") == "single_exposure_longer_than_sequence"
        for i in issues
    )


def test_validator_accepts_valid_exposure():
    # sequence duration 30 seconds, total exposure 15 seconds -> OK
    start = Time("2025-01-01T00:00:00", scale="utc")
    stop = start + TimeDelta(30, format="sec")
    payload = {
        "AcquireVisCamScienceData": make_vda_element(5_000_000, num_frames=3)
    }
    seq = ObservationSequence(
        id="s3",
        target="V",
        priority=1,
        start_time=start,
        stop_time=stop,
        ra=0.0,
        dec=0.0,
        payload_params=payload,
    )
    visit = Visit(id="v3", sequences=[seq])
    cal = ScienceCalendar(metadata={}, visits=[visit])

    sched = ScheduleProcessor.__new__(ScheduleProcessor)
    issues = sched.validate_payload_exposures(cal, report_issues=False)
    assert len(issues) == 0


def test_validator_checks_frames_per_coadd_when_no_total_frames():
    # No NumTotalFramesRequested, FramesPerCoadd present -> coadd exposure check
    start = Time("2025-01-01T00:00:00", scale="utc")
    stop = start + TimeDelta(4, format="sec")
    # exposure 2s, frames per coadd 3 -> coadd exposure 6s > 4s
    payload = {
        "AcquireVisCamScienceData": make_vda_element(
            2_000_000, frames_per_coadd=3
        )
    }
    seq = ObservationSequence(
        id="s4",
        target="W",
        priority=1,
        start_time=start,
        stop_time=stop,
        ra=0.0,
        dec=0.0,
        payload_params=payload,
    )
    visit = Visit(id="v4", sequences=[seq])
    cal = ScienceCalendar(metadata={}, visits=[visit])

    sched = ScheduleProcessor.__new__(ScheduleProcessor)
    issues = sched.validate_payload_exposures(cal, report_issues=False)
    assert any(
        i.get("problem") == "coadd_exposure_longer_than_sequence"
        for i in issues
    )


def test_validator_heuristic_flat_field_detection_and_non_numeric_robustness():
    # Create a payload with other category that flattens to keys containing 'exposure'
    start = Time("2025-01-01T00:00:00", scale="utc")
    stop = start + TimeDelta(5, format="sec")

    root = ET.Element("AcquireInfCamImages")
    a = ET.SubElement(root, "SomeExposure")
    a.text = "6"  # seconds -> should be flagged
    b = ET.SubElement(root, "OtherExposure_us")
    b.text = "7000000"  # 7s -> should be flagged
    c = ET.SubElement(root, "BadExposure_us")
    c.text = "not_a_number"  # should be ignored (no crash)

    payload = {"AcquireInfCamImages": root}
    seq = ObservationSequence(
        id="s5",
        target="X",
        priority=1,
        start_time=start,
        stop_time=stop,
        ra=0.0,
        dec=0.0,
        payload_params=payload,
    )
    visit = Visit(id="v5", sequences=[seq])
    cal = ScienceCalendar(metadata={}, visits=[visit])

    sched = ScheduleProcessor.__new__(ScheduleProcessor)
    issues = sched.validate_payload_exposures(cal, report_issues=False)

    # Expect two field-level exposure issues (SomeExposure and OtherExposure_us)
    field_issues = [
        i
        for i in issues
        if i.get("problem") == "payload_exposure_field_longer_than_sequence"
    ]
    assert len(field_issues) >= 2
