import os
import xml.etree.ElementTree as ET

import numpy as np
from astropy.time import Time

import shortschedule
from shortschedule.parser import parse_science_calendar
from shortschedule.scheduler import ScheduleProcessor
from shortschedule.writer import XMLWriter


class DummyVisibilityAllTrue:
    def __init__(self, l1, l2):
        pass

    def get_visibility(self, coord, times):
        try:
            length = len(times)
        except Exception:
            return np.array([True], dtype=bool)
        return np.ones(length, dtype=bool)


def get_sample_calendar_path():
    pkgdir = os.path.dirname(shortschedule.__file__)
    return os.path.join(pkgdir, 'data', 'Pandora_science_calendar_20251018_tsb-futz.xml')


def test_processed_calendar_metadata_written(monkeypatch, tmp_path):
    # Replace Visibility with a dummy to avoid external TLE work
    monkeypatch.setattr('shortschedule.scheduler.Visibility', DummyVisibilityAllTrue)

    sample = get_sample_calendar_path()
    assert os.path.exists(sample), f"Sample calendar not found at {sample}"

    cal = parse_science_calendar(sample)
    assert len(cal.visits) > 0

    first_seq = cal.visits[0].sequences[0]
    window_start = first_seq.start_time.isot

    sched = ScheduleProcessor('LINE1_EXAMPLE', 'LINE2_EXAMPLE')
    processed = sched.process_calendar(cal, window_start=window_start, window_duration_days=1, verbose=False)

    out_file = tmp_path / 'processed_meta.xml'
    XMLWriter().write_calendar(processed, str(out_file))

    assert os.path.exists(out_file), "Output file was not created"

    root = ET.parse(str(out_file)).getroot()
    # ElementTree places elements in the default namespace if one is set on
    # the root. Search for the Meta element by local-name to be robust.
    meta = None
    for child in root:
        if child.tag.endswith('Meta') or child.tag == 'Meta':
            meta = child
            break
    assert meta is not None, "Meta element missing from written XML"

    attrs = {k.lower(): v for k, v in meta.attrib.items()}

    # Ensure TLEs and processing timestamp are present in some variant
    assert 'tle_line1' in attrs or 'tle_line1'.lower() in attrs
    assert 'tle_line2' in attrs or 'tle_line2'.lower() in attrs
    assert 'processed_datetime' in attrs or 'processed_datetime'.lower() in attrs

    # Gap_Report should have been serialized
    assert 'gap_report' in attrs or 'gap_report'.lower() in attrs or 'gap-report' in attrs
