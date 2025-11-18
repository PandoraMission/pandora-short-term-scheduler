"""Tests for star ROI validation and generation."""

# Standard library
import xml.etree.ElementTree as ET

# Third-party
from astropy.time import Time, TimeDelta

# First-party/Local
from shortschedule.models import ObservationSequence, ScienceCalendar, Visit
from shortschedule.scheduler import ScheduleProcessor
from shortschedule.writer import XMLWriter


def test_validate_star_roi_consistency_detects_mismatch():
    """Test that validation detects when MaxNumStarRois != numPredefinedStarRois."""
    start = Time("2025-01-01T00:00:00", scale="utc")

    # Create payload with mismatched values
    payload_xml = ET.Element("AcquireVisCamScienceData")
    num_predefined = ET.SubElement(payload_xml, "numPredefinedStarRois")
    num_predefined.text = "9"
    max_num = ET.SubElement(payload_xml, "MaxNumStarRois")
    max_num.text = "0"  # Mismatch!

    seq = ObservationSequence(
        id="seq1",
        target="TestTarget",
        priority=1,
        start_time=start,
        stop_time=start + TimeDelta(120, format="sec"),
        ra=0.0,
        dec=0.0,
        payload_params={"AcquireVisCamScienceData": payload_xml},
    )

    visit = Visit(id="v1", sequences=[seq])
    cal = ScienceCalendar(metadata={}, visits=[visit])

    sched = ScheduleProcessor.__new__(ScheduleProcessor)
    issues = sched.validate_star_roi_consistency(cal, report_issues=False)

    assert len(issues) == 1
    assert (
        issues[0]["problem"]
        == "MaxNumStarRois_not_equal_to_numPredefinedStarRois"
    )
    assert issues[0]["numPredefinedStarRois"] == 9
    assert issues[0]["MaxNumStarRois"] == 0


def test_validate_star_roi_consistency_accepts_matching():
    """Test that validation passes when MaxNumStarRois == numPredefinedStarRois."""
    start = Time("2025-01-01T00:00:00", scale="utc")

    # Create payload with matching values
    payload_xml = ET.Element("AcquireVisCamScienceData")
    num_predefined = ET.SubElement(payload_xml, "numPredefinedStarRois")
    num_predefined.text = "9"
    max_num = ET.SubElement(payload_xml, "MaxNumStarRois")
    max_num.text = "9"  # Matching!

    seq = ObservationSequence(
        id="seq1",
        target="TestTarget",
        priority=1,
        start_time=start,
        stop_time=start + TimeDelta(120, format="sec"),
        ra=0.0,
        dec=0.0,
        payload_params={"AcquireVisCamScienceData": payload_xml},
    )

    visit = Visit(id="v1", sequences=[seq])
    cal = ScienceCalendar(metadata={}, visits=[visit])

    sched = ScheduleProcessor.__new__(ScheduleProcessor)
    issues = sched.validate_star_roi_consistency(cal, report_issues=False)

    assert len(issues) == 0


def test_validate_star_roi_consistency_handles_missing_values():
    """Test that validation handles missing values gracefully."""
    start = Time("2025-01-01T00:00:00", scale="utc")

    # Create payload with only numPredefinedStarRois
    payload_xml = ET.Element("AcquireVisCamScienceData")
    num_predefined = ET.SubElement(payload_xml, "numPredefinedStarRois")
    num_predefined.text = "9"
    # MaxNumStarRois is missing

    seq = ObservationSequence(
        id="seq1",
        target="TestTarget",
        priority=1,
        start_time=start,
        stop_time=start + TimeDelta(120, format="sec"),
        ra=0.0,
        dec=0.0,
        payload_params={"AcquireVisCamScienceData": payload_xml},
    )

    visit = Visit(id="v1", sequences=[seq])
    cal = ScienceCalendar(metadata={}, visits=[visit])

    sched = ScheduleProcessor.__new__(ScheduleProcessor)
    issues = sched.validate_star_roi_consistency(cal, report_issues=False)

    # Should not raise an error, just skip validation
    assert len(issues) == 0


def test_writer_ensures_star_roi_consistency():
    """Test that writer ensures MaxNumStarRois equals numPredefinedStarRois."""
    # Create payload with mismatched values
    payload_xml = ET.Element("AcquireVisCamScienceData")
    num_predefined = ET.SubElement(payload_xml, "numPredefinedStarRois")
    num_predefined.text = "9"
    max_num = ET.SubElement(payload_xml, "MaxNumStarRois")
    max_num.text = "0"  # Mismatch that should be fixed!

    # Create a simple calendar
    start = Time("2025-01-01T00:00:00", scale="utc")
    seq = ObservationSequence(
        id="seq1",
        target="TestTarget",
        priority=1,
        start_time=start,
        stop_time=start + TimeDelta(120, format="sec"),
        ra=100.0,
        dec=-20.0,
        payload_params={"AcquireVisCamScienceData": payload_xml},
    )

    visit = Visit(id="v1", sequences=[seq])
    metadata = {
        "valid_from": "2025-01-01 00:00:00",
        "expires": "2025-01-22 00:00:00",
    }
    cal = ScienceCalendar(metadata=metadata, visits=[visit])

    # Write the calendar
    writer = XMLWriter()
    # Standard library
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False
    ) as f:
        output_path = f.name

    writer.write_calendar(cal, output_path=output_path)

    # Parse the written file and check
    tree = ET.parse(output_path)
    root = tree.getroot()
    namespace = {"pandora": "/pandora/calendar/"}

    # Find the AcquireVisCamScienceData element
    vis_cam_elem = root.find(".//pandora:AcquireVisCamScienceData", namespace)
    assert vis_cam_elem is not None

    # Remove namespace from tags for easier checking
    for elem in vis_cam_elem.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    # Check that MaxNumStarRois was set to match numPredefinedStarRois
    num_predefined_elem = vis_cam_elem.find("numPredefinedStarRois")
    max_num_elem = vis_cam_elem.find("MaxNumStarRois")

    assert num_predefined_elem is not None
    assert max_num_elem is not None
    assert num_predefined_elem.text == "9"
    assert max_num_elem.text == "9"  # Should be corrected to 9

    # Clean up
    # Standard library
    import os

    os.unlink(output_path)


def test_writer_creates_max_num_star_rois_if_missing():
    """Test that writer creates MaxNumStarRois if it doesn't exist."""
    # Create payload with only numPredefinedStarRois
    payload_xml = ET.Element("AcquireVisCamScienceData")
    num_predefined = ET.SubElement(payload_xml, "numPredefinedStarRois")
    num_predefined.text = "8"
    # MaxNumStarRois is missing

    # Create a simple calendar
    start = Time("2025-01-01T00:00:00", scale="utc")
    seq = ObservationSequence(
        id="seq1",
        target="TestTarget",
        priority=1,
        start_time=start,
        stop_time=start + TimeDelta(120, format="sec"),
        ra=100.0,
        dec=-20.0,
        payload_params={"AcquireVisCamScienceData": payload_xml},
    )

    visit = Visit(id="v1", sequences=[seq])
    metadata = {
        "valid_from": "2025-01-01 00:00:00",
        "expires": "2025-01-22 00:00:00",
    }
    cal = ScienceCalendar(metadata=metadata, visits=[visit])

    # Write the calendar
    writer = XMLWriter()
    # Standard library
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False
    ) as f:
        output_path = f.name

    writer.write_calendar(cal, output_path=output_path)

    # Parse the written file and check
    tree = ET.parse(output_path)
    root = tree.getroot()
    namespace = {"pandora": "/pandora/calendar/"}

    # Find the AcquireVisCamScienceData element
    vis_cam_elem = root.find(".//pandora:AcquireVisCamScienceData", namespace)
    assert vis_cam_elem is not None

    # Remove namespace from tags for easier checking
    for elem in vis_cam_elem.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    # Check that MaxNumStarRois was created and set correctly
    num_predefined_elem = vis_cam_elem.find("numPredefinedStarRois")
    max_num_elem = vis_cam_elem.find("MaxNumStarRois")

    assert num_predefined_elem is not None
    assert max_num_elem is not None
    assert num_predefined_elem.text == "8"
    assert max_num_elem.text == "8"  # Should be created and set to 8

    # Clean up
    # Standard library
    import os

    os.unlink(output_path)
