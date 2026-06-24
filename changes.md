## v1.2.0 (2026-xxx)

- Adds NIRDA and VISDA classes which contain accurate and up to date parameters to perform timing and data volume calculations.
- Adds overhead class which accounts for pre- and post- overhead timings for both VISDA and NIRDA
- Adds baseline short-term calendar runner script to docs/
- Adds ability to merge back-to-back observations of the same target.
- Adds ability to override payload parameters set by original long-term calendar on a per-priority type basis.
- Adds warnings if single NIRDA or VISDA data file exceeds payload limits.
- Adds dependence on NIRDA reset1 for VITL settling time. These parameters are all adjustable.
- Adds helper that renumbers both visits and sequencies to fix any misnumbering after merges.
- Adds log file to track changes, info, and warnings raised by the short term scheduler.
- Fixes minute-by-minute parsing to improve processing time.
- Adds several progress bars during various slower processing sections.
- Adds tests for all of these changes.
