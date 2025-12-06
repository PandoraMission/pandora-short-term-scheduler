# Pandora Short-Term Scheduler (shortschedule)

[![Tests](https://github.com/PandoraMission/pandora-short-term-scheduler/actions/workflows/ci.yml/badge.svg)](https://github.com/PandoraMission/pandora-short-term-scheduler/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**shortschedule** is a compact toolkit for parsing, processing, and visualizing Pandora
science calendars (PAN-SCICAL XML). It adjusts observation timing and payload
parameters when updated spacecraft TLEs are available, producing processed
calendars that can be written back to PAN-compliant XML.

## Highlights

- **Parse** PAN-SCICAL XML into typed Python models (`ScienceCalendar`, `Visit`, `ObservationSequence`)
- **Process** calendars using updated TLEs to fill visibility gaps and recompute payload integrations
- **Export** processed calendars as PAN-compliant XML with `XMLWriter`
- **Visualize** Gantt/timeline plots to compare original and processed schedules
- **Validate** payload exposure times and sequence timing constraints

---

## Codebase Overview

The package is organized under `src/shortschedule/` and exposes a small, focused public API:

### Core Modules

| Module | Purpose |
|--------|---------|
| **`models.py`** | Data classes for the in-memory calendar representation: `ObservationSequence`, `Visit`, and `ScienceCalendar`. Times use `astropy.time.Time`; payload parameters are stored as `ElementTree` elements for lossless XML round-tripping. |
| **`parser.py`** | Parse PAN-SCICAL XML files into `ScienceCalendar` objects. Entry point: `parse_science_calendar(xml_path)`. |
| **`scheduler.py`** | `ScheduleProcessor` class that adjusts a calendar given new TLE data. It computes minute-by-minute visibility via `pandoravisibility`, fills gaps, updates payload integration parameters, and produces a detailed `gap_report`. |
| **`writer.py`** | `XMLWriter` class to serialize a `ScienceCalendar` back to PAN-compliant XML, preserving payload structure and attaching processing metadata. |
| **`visualizer.py`** | `ScheduleVisualizer` class for generating Gantt charts, visit summaries, and priority statistics. Handles large time spans by automatically windowing plots. |

### Data Flow

```
PAN-SCICAL XML ──▶ parser.parse_science_calendar() ──▶ ScienceCalendar
                                                              │
                                                              ▼
                              ScheduleProcessor.process_calendar()
                                                              │
                                                              ▼
                                                 Processed ScienceCalendar
                                                              │
                         ┌────────────────────────────────────┴────────────────────────────────────┐
                         ▼                                                                          ▼
           XMLWriter.write_calendar()                                          ScheduleVisualizer.plot_*()
                         │                                                                          │
                         ▼                                                                          ▼
              Processed PAN-SCICAL XML                                               Gantt/Timeline Figures
```

### Key Design Decisions

- **Astropy for time handling**: All times are `astropy.time.Time` objects; durations are `TimeDelta`. This ensures precision and avoids timezone pitfalls.
- **Payload preservation**: Payload parameters are stored as raw `ElementTree` elements so nested structures survive round-trip serialization without loss.
- **Visibility via `pandoravisibility`**: The scheduler delegates visibility calculations to the external `pandoravisibility` package. Tests mock `Visibility.get_visibility()` for determinism.
- **Metadata round-trip**: Processing metadata (TLE lines, `processed_datetime`, `gap_report`, `calendar_status`) is attached to processed calendars and written to XML for traceability.

### Tests

Unit tests live in `tests/` and cover:

| Test File | Coverage |
|-----------|----------|
| `test_io_roundtrip.py` | XML parse → write → re-parse consistency |
| `test_metadata_roundtrip.py` | Metadata survives serialization |
| `test_payload_validator.py` | Payload exposure fits within sequence duration |
| `test_scheduler_validators.py` | Sequence timing and visibility validation |
| `test_processor_with_mock_visibility.py` | End-to-end processing with mocked visibility |
| `test_star_roi_validation.py` | Star/ROI parameter validation |
| `test_version.py` | Package version accessibility |

Run tests with `make pytest` or `poetry run pytest -q`.

Install (developer flow)
------------------------
We use Poetry for dependency and environment management. Recommended steps:

```bash
poetry install --with dev
```

To add or change dependencies use `poetry add` or edit `pyproject.toml` and run `poetry install`.

Running tests and linters
-------------------------
Use the Makefile wrappers (these call Poetry in CI):

```bash
make flake8   # lint
make black    # format (Black line length = 79)
make isort    # sort imports
make pytest   # run tests
```

You can also run pytest directly if your environment is active:

```bash
pytest -q
```

Quick start examples
--------------------
Parse a calendar:

```python
from shortschedule import parse_science_calendar

xml_file_path = "../src/shortschedule/data/Pandora_science_calendar_20251018_tsb-futz.xml"
cal = parse_science_calendar(xml_file_path)
```

Process with updated TLEs:

```python
from shortschedule import ScheduleProcessor

tle_line1 = "1 99152U 26011B  26005.66013674 +.00000000 +00000-0 +00000-0 0   16"
new_tltle_line2e2 = "2 99152 97.6750 17.6690 0000000 328.8990 20.9640 14.865"

proc = ScheduleProcessor(tle_line1, tle_line2)
processed = proc.process_calendar(cal, window_start='2026-10-01T00:00:00Z', window_duration_days=3)
```

Write processed calendar to XML:

```python
from shortschedule import XMLWriter

XMLWriter().write_calendar(processed, output_path='processed.xml')
```

Notes for contributors
----------------------
- Time handling: use `astropy.time.Time` and `TimeDelta` for all time arithmetic and return `Time` objects.
- Visibility: `pandoravisibility.Visibility` is used for visibility arrays; tests that exercise visibility must
  mock `Visibility.get_visibility(...)` to return deterministic boolean numpy arrays.
- Metadata: the scheduler attaches processing metadata (TLE lines, `processed_datetime`, `gap_report`, and
  `calendar_status`) to the processed `ScienceCalendar`. If you change metadata keys, update
  `src/shortschedule/writer.py` and the metadata round-trip tests.

Formatting and CI
-----------------
We require Black formatting (line length = 79). Run `make black` locally before committing; CI enforces
formatting. Consider adding a local pre-commit hook to run Black and isort.

Further development and testing
-------------------------------
- Unit tests live in `tests/`. Add tests for new behavior and keep them deterministic (no network or time
  variability). Use `monkeypatch` to stub `pandoravisibility.Visibility.get_visibility` in tests.
- If you add a new public API or change critical interfaces (scheduler, models, writer, parser), add tests that
  exercise those changes and update the README or docs as needed.

Contact / License
-----------------
See the repository `LICENSE` file for terms. For questions about the scheduler or repository conventions,
open an issue or PR in this repository.
