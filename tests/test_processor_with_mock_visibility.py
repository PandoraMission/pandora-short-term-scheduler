# Third-party
import numpy as np

# First-party/Local
import shortschedule
from shortschedule.parser import parse_science_calendar
from shortschedule.scheduler import ScheduleProcessor


class DummyVisibilityAllTrue:
    def __init__(self, l1, l2):
        pass

    def get_visibility(self, coord, times):
        # times may be an astropy Time or array-like; return True for each entry
        try:
            length = len(times)
        except Exception:
            # single Time -> treat as length 1
            return np.array([True], dtype=bool)
        return np.ones(length, dtype=bool)


def test_process_calendar_with_mocked_visibility(monkeypatch, tmp_path):
    # Monkeypatch the Visibility used in the scheduler to avoid external TLE work
    monkeypatch.setattr(
        "shortschedule.scheduler.Visibility", DummyVisibilityAllTrue
    )

    # Load sample calendar
    pkgdir = shortschedule.__file__.rsplit("/", 1)[0]
    sample = pkgdir + "/data/Pandora_science_calendar_20251018_tsb-futz.xml"

    cal = parse_science_calendar(sample)
    assert len(cal.visits) > 0

    # Use the first sequence start time as window_start
    first_seq = cal.visits[0].sequences[0]
    window_start = first_seq.start_time.isot

    # Create scheduler with dummy TLEs (strings required)
    sched = ScheduleProcessor("LINE1", "LINE2")

    processed = sched.process_calendar(
        cal, window_start=window_start, window_duration_days=1, verbose=False
    )

    # Basic assertions: processed_calendar object and gap_report present
    assert hasattr(processed, "visits")
    gr = sched.get_gap_report()
    assert isinstance(gr, dict)
    assert "processing_summary" in gr
