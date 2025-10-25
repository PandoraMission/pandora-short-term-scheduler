# Copilot Instructions — pandora-short-term-scheduler

## Purpose
These instructions guide automated agents (e.g. Copilot, ChatGPT, or other LLM-based tools)
to make safe, testable, and convention-aligned edits to the `pandora-short-term-scheduler` repository.

The goal is to preserve scientific correctness, test integrity, and developer workflow consistency.


---

## Repository overview

**Package root:** `src/shortschedule/`  
**Package name (pyproject):** `shortschedule` — use this in all imports and tests.

**Core modules:**
- `parser.py` — XML parsing of science calendars
- `models.py` — `ScienceCalendar`, `Visit`, `ObservationSequence` data classes
- `scheduler.py` — main algorithm (`ScheduleProcessor`)
- `writer.py` — XML writing and metadata round-trip logic
- `visualizer.py` — visualization helpers

---

## Developer workflow

**Use Poetry.**
```bash
poetry install --with dev
```

**Run checks via Makefile:**
- `make pytest` — run tests (`src` and `tests`)
- `make flake8`, `make black`, `make isort` — linting/formatting (Black line length = 79)
- `make serve` or `poetry run mkdocs serve` — build documentation locally

**Before committing or pushing:**
```bash
make flake8
make black
make pytest
```

**Formatting requirement:** All code must be formatted with Black (line length = 79). Run `make black`
before committing; CI will enforce formatting.

---

## Key patterns and conventions

### Time handling
- Use `astropy.time.Time` and `astropy.time.TimeDelta` for all time operations.
- Prefer returning `Time` objects; do not use naive `datetime` unless required by an external library.

### Data model
- Always use classes from `src/shortschedule/models.py`:
  - `ScienceCalendar`
  - `Visit`
  - `ObservationSequence`
- Avoid ad-hoc dicts for data interchange unless explicitly required by external code.

### Visibility
- `pandoravisibility.Visibility.get_visibility(...)` provides external visibility arrays.
- **In tests:** mock this method to return deterministic boolean numpy arrays.

### Metadata
- Metadata keys attached to calendars may include:
  - `tle_line1`, `tle_line2`
  - `processed_datetime` (ISO string)
  - `gap_report` (JSON string)
- If any metadata shape or name changes, update:
  - `src/shortschedule/writer.py`
  - `tests/test_metadata_roundtrip.py`

---

## Allowed edits
Automated edits **may include**:
- Small, localized changes that:
  - Fix bugs
  - Improve readability or type hints
  - Add or refine tests
  - Adjust function signatures **only if all callers are updated**
  - Update comments, docstrings, or formatting

- Adding or modifying test files under `tests/`, provided tests remain deterministic.

- Refactoring internal helper functions (non-public) **within the same module**.

- Adding lightweight developer tooling (e.g., Makefile targets or CI config improvements)
  if consistent with existing conventions.

---

## Disallowed edits
Automated agents **must not**:
- Delete or rename public classes/functions without updating all references and tests.
- Modify APIs in `scheduler.py`, `models.py`, or `writer.py` without explicit test coverage.
- Remove or rewrite files in `src/shortschedule/old/` without checking `__init__.py` guarded imports.
- Introduce external dependencies not declared in `pyproject.toml`.
- Commit credentials, tokens, or external data.

---

## Critical interfaces
If you modify any of the following, corresponding tests **must** be added or updated:

| Module | Critical Interface | Notes |
|--------|-------------------|-------|
| `scheduler.py` | `ScheduleProcessor`, `gap_report`, `processing_log` | Central scheduling logic |
| `models.py` | `ScienceCalendar`, `Visit`, `ObservationSequence` | Core data model |
| `writer.py` | `write_calendar_xml`, metadata serialization | Controls round-trip consistency |
| `parser.py` | `parse_calendar_xml` | XML input interface |
| `pandoravisibility.Visibility` | `get_visibility` | External dependency — must be mocked in tests |

---

## Testing expectations

**All tests must:**
- Import `shortschedule` (installed package)
- Be deterministic and self-contained (no network or external time sources)
- Run successfully with:
  ```bash
  poetry run pytest -q
  ```

**To mock visibility:**
```python
monkeypatch.setattr(
    "pandoravisibility.Visibility.get_visibility",
    lambda *_, **__: np.ones(60, dtype=bool)
)
```

---

## Safe automation checklist (for Copilot or AI agents)
1. Confirm the edit is small, local, and covered by tests.  
2. Run formatting and lint checks (`make black`, `make flake8`).  
3. Run `make pytest` locally to verify correctness.  
4. Update documentation if public functions or metadata change.  
5. Do not modify or bypass `pyproject.toml` dependencies.  
6. Prefer conservative, minimal-diff edits that preserve test determinism.  

---

## Reference files (read first)
- `src/shortschedule/scheduler.py`
- `src/shortschedule/models.py`
- `src/shortschedule/writer.py`
- `src/shortschedule/parser.py`
- `Makefile`, `pyproject.toml`, `.github/workflows/tests.yml`
- `docs/README.md`

---

## When unsure
- Prefer small, test-covered edits.
- For major changes or API adjustments, open a **draft PR** and run CI before merging.

---

This document defines the operational and behavioral constraints for automated code modification in
`pandora-short-term-scheduler`.  
It is concise, authoritative, and safe for LLM-based code editing.