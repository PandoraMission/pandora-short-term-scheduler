# Standard library
import xml.etree.ElementTree as ET
from pathlib import Path

# Third-party
import numpy as np

# First-party/Local
import shortschedule
from shortschedule.parser import parse_science_calendar
from shortschedule.scheduler import ScheduleProcessor
from shortschedule.writer import XMLWriter


class DummyVisibilityAllTrue:
    def __init__(self, l1, l2, **kwargs):
        pass

    def get_visibility(self, coord, times, roll=None):
        try:
            length = len(times)
        except Exception:
            return np.array([True], dtype=bool)
        return np.ones(length, dtype=bool)


def get_sample_calendar_path():
    return (
        Path(shortschedule.__file__).parent
        / "data"
        / "Pandora_science_calendar_20251018_tsb-futz.xml"
    )


def test_processed_calendar_metadata_written(monkeypatch, tmp_path):
    # Replace Visibility with a dummy to avoid external TLE work
    monkeypatch.setattr(
        "shortschedule.scheduler.Visibility", DummyVisibilityAllTrue
    )

    sample = get_sample_calendar_path()
    assert sample.exists(), f"Sample calendar not found at {sample}"

    cal = parse_science_calendar(sample)
    assert len(cal.visits) > 0

    first_seq = cal.visits[0].sequences[0]
    window_start = first_seq.start_time.isot

    sched = ScheduleProcessor("LINE1_EXAMPLE", "LINE2_EXAMPLE")
    processed = sched.process_calendar(
        cal,
        window_start=window_start,
        window_duration_days=1,
        log_path=tmp_path / "run",
        verbose=False,
    )

    out_file = tmp_path / "processed_meta.xml"
    XMLWriter().write_calendar(processed, str(out_file))

    assert out_file.exists(), "Output file was not created"

    root = ET.parse(str(out_file)).getroot()
    # ElementTree places elements in the default namespace if one is set on
    # the root. Search for the Meta element by local-name to be robust.
    meta = None
    for child in root:
        if child.tag.endswith("Meta") or child.tag == "Meta":
            meta = child
            break
    assert meta is not None, "Meta element missing from written XML"

    attrs = {k.lower(): v for k, v in meta.attrib.items()}

    # Ensure TLEs and processing timestamp are present in some variant
    assert "tle_line1" in attrs or "tle_line1".lower() in attrs
    assert "tle_line2" in attrs or "tle_line2".lower() in attrs
    assert "created" in attrs or "created".lower() in attrs

    # The short-term scheduler version must be stamped into the calendar.
    assert "short_term_scheduler_version" in attrs
    assert attrs["short_term_scheduler_version"] == shortschedule.get_version()


def test_version_written_with_default_metadata(tmp_path):
    """The version is stamped even for a calendar with no source metadata."""
    from astropy.time import Time, TimeDelta

    from shortschedule.models import (
        ObservationSequence,
        ScienceCalendar,
        Visit,
    )

    start = Time("2026-03-01T00:00:00", scale="utc")
    seq = ObservationSequence(
        id="001",
        target="T",
        priority=0,
        start_time=start,
        stop_time=start + TimeDelta(3600, format="sec"),
        ra=1.0,
        dec=2.0,
        payload_params={},
    )
    cal = ScienceCalendar(metadata={}, visits=[Visit("0001", [seq])])
    out_file = tmp_path / "v.xml"
    XMLWriter().write_calendar(cal, str(out_file))

    root = ET.parse(str(out_file)).getroot()
    meta = next(c for c in root if c.tag.endswith("Meta"))
    attrs = {k.lower(): v for k, v in meta.attrib.items()}
    assert attrs["short_term_scheduler_version"] == shortschedule.get_version()
