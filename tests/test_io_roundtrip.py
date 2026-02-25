# Standard library
import os
import xml.etree.ElementTree as ET

# First-party/Local
import shortschedule
from shortschedule.parser import parse_science_calendar, parse_xml_element
from shortschedule.writer import XMLWriter


def get_sample_calendar_path():
    pkgdir = os.path.dirname(shortschedule.__file__)
    return os.path.join(
        pkgdir, "data", "Pandora_science_calendar_20251018_tsb-futz.xml"
    )


def test_parse_sample_calendar_and_roundtrip(tmp_path):
    sample = get_sample_calendar_path()
    assert os.path.exists(sample), f"Sample calendar not found at {sample}"

    cal = parse_science_calendar(sample)
    # Basic sanity checks
    assert hasattr(cal, "visits")
    assert len(cal.visits) > 0
    total_sequences = sum(len(v.sequences) for v in cal.visits)
    assert total_sequences > 0

    # Write calendar out and parse back
    out_file = tmp_path / "roundtrip.xml"
    writer = XMLWriter()
    written = writer.write_calendar(cal, str(out_file))
    assert os.path.exists(written)

    cal2 = parse_science_calendar(written)
    total_sequences2 = sum(len(v.sequences) for v in cal2.visits)
    assert total_sequences == total_sequences2


def test_boresight_contains_pri_cmd_dir(tmp_path):
    """Written XML should include PRI_CMD_DIR element in each Boresight."""
    sample = get_sample_calendar_path()
    cal = parse_science_calendar(sample)

    out_file = tmp_path / "pri_cmd_dir.xml"
    writer = XMLWriter()
    writer.write_calendar(cal, str(out_file))

    tree = ET.parse(str(out_file))
    root = tree.getroot()
    ns = (
        {"pandora": root.tag.split("}")[0].strip("{")}
        if "}" in root.tag
        else {}
    )
    prefix = "pandora:" if ns else ""

    boresights = root.findall(f".//{prefix}Boresight", ns)
    assert len(boresights) > 0, "No Boresight elements found in output"
    for bs in boresights:
        pri = bs.find(f"{prefix}PRI_CMD_DIR", ns)
        assert pri is not None, "PRI_CMD_DIR missing from Boresight"
        assert pri.text == "10", f"PRI_CMD_DIR should be 10, got {pri.text}"


def test_payload_element_roundtrip_preserves_tags():
    sample = get_sample_calendar_path()
    cal = parse_science_calendar(sample)

    # Look for the first sequence with payload parameters
    found_elem = None
    for visit in cal.visits:
        for seq in visit.sequences:
            if seq.payload_params:
                # get first payload element
                first_key = next(iter(seq.payload_params))
                elem = seq.payload_params[first_key]
                found_elem = elem
                break
        if found_elem is not None:
            break

    assert (
        found_elem is not None
    ), "No payload parameter element found in sample calendar"

    # Convert to string and parse via parser.parse_xml_element
    xml_str = ET.tostring(found_elem, encoding="unicode")
    parsed = parse_xml_element(xml_str)
    # Ensure tag names are preserved
    assert parsed.tag == found_elem.tag or parsed.tag.endswith(found_elem.tag)
