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
