"""Tests for star ROI validation and generation."""

# Standard library
import os
import tempfile
import xml.etree.ElementTree as ET

# Third-party
from astropy.time import Time, TimeDelta

# First-party/Local
from shortschedule.models import ObservationSequence, ScienceCalendar, Visit
from shortschedule.scheduler import ScheduleProcessor
from shortschedule.writer import XMLWriter


def test_validate_star_roi_consistency_detects_mismatch():
    """Test that validation detects when MaxNumStarRois != numPredefinedStarRois for method 1."""
    start = Time("2025-01-01T00:00:00", scale="utc")

    # Create payload with mismatched values (method 1)
    payload_xml = ET.Element("AcquireVisCamScienceData")
    star_roi_det = ET.SubElement(payload_xml, "StarRoiDetMethod")
    star_roi_det.text = "1"
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
    """Test that validation passes when MaxNumStarRois == numPredefinedStarRois for method 1."""
    start = Time("2025-01-01T00:00:00", scale="utc")

    # Create payload with matching values (method 1)
    payload_xml = ET.Element("AcquireVisCamScienceData")
    star_roi_det = ET.SubElement(payload_xml, "StarRoiDetMethod")
    star_roi_det.text = "1"
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


def test_validate_star_roi_consistency_method_2_nonzero_predefined():
    """Test that validation detects when numPredefinedStarRois != 0 for method 2."""
    start = Time("2025-01-01T00:00:00", scale="utc")

    # Create payload with method 2 but numPredefinedStarRois is not 0
    payload_xml = ET.Element("AcquireVisCamScienceData")
    star_roi_det = ET.SubElement(payload_xml, "StarRoiDetMethod")
    star_roi_det.text = "2"
    num_predefined = ET.SubElement(payload_xml, "numPredefinedStarRois")
    num_predefined.text = "9"  # Should be 0 for method 2!
    max_num = ET.SubElement(payload_xml, "MaxNumStarRois")
    max_num.text = "10"

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
        == "numPredefinedStarRois_should_be_0_for_method_2"
    )
    assert issues[0]["numPredefinedStarRois"] == 9


def test_validate_star_roi_consistency_method_2_accepts_zero():
    """Test that validation passes when numPredefinedStarRois == 0 for method 2."""
    start = Time("2025-01-01T00:00:00", scale="utc")

    # Create payload with method 2 and numPredefinedStarRois = 0
    payload_xml = ET.Element("AcquireVisCamScienceData")
    star_roi_det = ET.SubElement(payload_xml, "StarRoiDetMethod")
    star_roi_det.text = "2"
    num_predefined = ET.SubElement(payload_xml, "numPredefinedStarRois")
    num_predefined.text = "0"  # Correct for method 2
    max_num = ET.SubElement(payload_xml, "MaxNumStarRois")
    max_num.text = "10"

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


def test_validate_star_roi_consistency_method_2_rejects_zero_max():
    """Test that validation detects when MaxNumStarRois == 0 for method 2."""
    start = Time("2025-01-01T00:00:00", scale="utc")

    # Create payload with method 2 but MaxNumStarRois = 0
    payload_xml = ET.Element("AcquireVisCamScienceData")
    star_roi_det = ET.SubElement(payload_xml, "StarRoiDetMethod")
    star_roi_det.text = "2"
    num_predefined = ET.SubElement(payload_xml, "numPredefinedStarRois")
    num_predefined.text = "0"  # Correct for method 2
    max_num = ET.SubElement(payload_xml, "MaxNumStarRois")
    max_num.text = "0"  # Invalid for method 2!

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
        issues[0]["problem"] == "MaxNumStarRois_should_not_be_0_for_method_2"
    )
    assert issues[0]["MaxNumStarRois"] == 0


def test_validate_star_roi_consistency_handles_missing_values():
    """Test that validation handles missing values gracefully."""
    start = Time("2025-01-01T00:00:00", scale="utc")

    # Create payload with only numPredefinedStarRois and explicit StarRoiDetMethod=1
    payload_xml = ET.Element("AcquireVisCamScienceData")
    star_roi_det = ET.SubElement(payload_xml, "StarRoiDetMethod")
    star_roi_det.text = "1"
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


def test_writer_ensures_star_roi_consistency_method_1():
    """Test that writer ensures MaxNumStarRois equals numPredefinedStarRois for method 1."""
    # Create payload with mismatched values (method 1)
    payload_xml = ET.Element("AcquireVisCamScienceData")
    star_roi_det = ET.SubElement(payload_xml, "StarRoiDetMethod")
    star_roi_det.text = "1"
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
    os.unlink(output_path)


def test_writer_ensures_star_roi_consistency_method_2():
    """Test that writer sets numPredefinedStarRois to 0 for method 2."""
    # Create payload with method 2 but wrong numPredefinedStarRois
    payload_xml = ET.Element("AcquireVisCamScienceData")
    star_roi_det = ET.SubElement(payload_xml, "StarRoiDetMethod")
    star_roi_det.text = "2"
    num_predefined = ET.SubElement(payload_xml, "numPredefinedStarRois")
    num_predefined.text = "9"  # Should be set to 0 for method 2
    max_num = ET.SubElement(payload_xml, "MaxNumStarRois")
    max_num.text = "10"  # Should keep this value

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

    # Check that numPredefinedStarRois was set to 0
    num_predefined_elem = vis_cam_elem.find("numPredefinedStarRois")
    max_num_elem = vis_cam_elem.find("MaxNumStarRois")

    assert num_predefined_elem is not None
    assert max_num_elem is not None
    assert num_predefined_elem.text == "0"  # Should be corrected to 0
    assert max_num_elem.text == "10"  # Should keep original value

    # Clean up
    os.unlink(output_path)


def test_writer_creates_max_num_star_rois_if_missing():
    """Test that writer creates MaxNumStarRois if it doesn't exist for method 1."""
    # Create payload with only numPredefinedStarRois (method 1)
    payload_xml = ET.Element("AcquireVisCamScienceData")
    star_roi_det = ET.SubElement(payload_xml, "StarRoiDetMethod")
    star_roi_det.text = "1"
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
    os.unlink(output_path)


def test_validate_star_roi_consistency_detects_unparseable_values():
    """Test that validation detects when values cannot be parsed as integers for methods 0, 1, 3."""
    start = Time("2025-01-01T00:00:00", scale="utc")

    # Create payload with unparseable values (method 1)
    payload_xml = ET.Element("AcquireVisCamScienceData")
    star_roi_det = ET.SubElement(payload_xml, "StarRoiDetMethod")
    star_roi_det.text = "1"
    num_predefined = ET.SubElement(payload_xml, "numPredefinedStarRois")
    num_predefined.text = "not_a_number"  # Unparseable!
    max_num = ET.SubElement(payload_xml, "MaxNumStarRois")
    max_num.text = "9"

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
    assert issues[0]["problem"] == "star_roi_values_not_parseable_as_integers"
    assert issues[0]["numPredefinedStarRois"] == "not_a_number"


def test_validate_star_roi_consistency_method_2_detects_unparseable_predefined():
    """Test that validation detects unparseable numPredefinedStarRois for method 2."""
    start = Time("2025-01-01T00:00:00", scale="utc")

    # Create payload with method 2 and unparseable numPredefinedStarRois
    payload_xml = ET.Element("AcquireVisCamScienceData")
    star_roi_det = ET.SubElement(payload_xml, "StarRoiDetMethod")
    star_roi_det.text = "2"
    num_predefined = ET.SubElement(payload_xml, "numPredefinedStarRois")
    num_predefined.text = "invalid"  # Unparseable!
    max_num = ET.SubElement(payload_xml, "MaxNumStarRois")
    max_num.text = "10"

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
        == "numPredefinedStarRois_not_parseable_as_integer"
    )
    assert issues[0]["numPredefinedStarRois"] == "invalid"


def test_validate_star_roi_consistency_method_2_detects_unparseable_max():
    """Test that validation detects unparseable MaxNumStarRois for method 2."""
    start = Time("2025-01-01T00:00:00", scale="utc")

    # Create payload with method 2 and unparseable MaxNumStarRois
    payload_xml = ET.Element("AcquireVisCamScienceData")
    star_roi_det = ET.SubElement(payload_xml, "StarRoiDetMethod")
    star_roi_det.text = "2"
    num_predefined = ET.SubElement(payload_xml, "numPredefinedStarRois")
    num_predefined.text = "0"  # Correct for method 2
    max_num = ET.SubElement(payload_xml, "MaxNumStarRois")
    max_num.text = "bad_value"  # Unparseable!

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
    assert issues[0]["problem"] == "MaxNumStarRois_not_parseable_as_integer"
    assert issues[0]["MaxNumStarRois"] == "bad_value"
