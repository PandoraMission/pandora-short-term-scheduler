"""Every product a run makes belongs beside the long-term calendar.

The delivered XML, the run logs, the diagnostics and the plots all used to
be able to land in whatever directory the run happened to start in, which
scattered a week's products across the filesystem.
"""

# Standard library
import shutil
from pathlib import Path

# Third-party
import numpy as np
import pytest

# First-party/Local
import shortschedule
from shortschedule.models import ScienceCalendar
from shortschedule.parser import parse_science_calendar
from shortschedule.scheduler import ScheduleProcessor
from shortschedule.writer import XMLWriter

SAMPLE = (
    Path(shortschedule.__file__).parent
    / "data"
    / "Pandora_science_calendar_20251018_tsb-futz.xml"
)


@pytest.fixture
def calendar_in(tmp_path, monkeypatch):
    """A calendar in its own directory, with the run started elsewhere."""
    calendar_dir = tmp_path / "long-term"
    working_dir = tmp_path / "somewhere-else"
    calendar_dir.mkdir()
    working_dir.mkdir()
    path = calendar_dir / SAMPLE.name
    shutil.copy(SAMPLE, path)
    monkeypatch.chdir(working_dir)
    return path


def test_source_path_points_at_the_parsed_file(calendar_in):
    calendar = parse_science_calendar(str(calendar_in))

    assert calendar.source_path == calendar_in


def test_source_path_is_none_without_a_file():
    """A calendar built in memory has nowhere of its own to write to."""
    assert ScienceCalendar(metadata={}, visits=[]).source_path is None


def test_written_calendar_lands_beside_its_source(calendar_in):
    """With no output_path, the delivered XML goes next to the calendar."""
    calendar = parse_science_calendar(str(calendar_in))

    written = Path(XMLWriter().write_calendar(calendar, mission_phase="COM"))

    assert written.parent == calendar_in.parent
    assert written.exists()
    assert not list(Path.cwd().iterdir()), "wrote into the working directory"


def test_an_explicit_output_path_still_wins(calendar_in, tmp_path):
    chosen = tmp_path / "chosen.xml"
    calendar = parse_science_calendar(str(calendar_in))

    written = Path(
        XMLWriter().write_calendar(calendar, output_path=str(chosen))
    )

    assert written == chosen
    assert chosen.exists()


def test_logs_and_diagnostics_land_beside_the_source(calendar_in, monkeypatch):
    """The whole run leaves its products in one directory."""
    monkeypatch.setattr(
        "shortschedule.scheduler.Visibility",
        lambda *a, **kw: _AlwaysVisible(),
    )
    processor = ScheduleProcessor("L1", "L2")
    calendar = parse_science_calendar(str(calendar_in))
    window_start = calendar.visits[0].sequences[0].start_time.isot

    processed = processor.process_calendar(
        calendar, window_start=window_start, window_duration_days=1
    )
    processor.generate_diagnostics(processed)
    XMLWriter().write_calendar(processed, mission_phase="COM")

    produced = {p.name for p in calendar_in.parent.iterdir()}
    assert f"{calendar_in.stem}.log" in produced
    assert f"{calendar_in.stem}.diag" in produced
    assert any(name.startswith("PAN-SCICAL-") for name in produced)
    assert not list(Path.cwd().iterdir()), "wrote into the working directory"


class _AlwaysVisible:
    """Minimal visibility stand-in so the run does no ephemeris work."""

    def get_visibility(self, coord, times, roll=None):
        try:
            return np.ones(len(times), dtype=bool)
        except TypeError:
            return True
