## v1.2.0 (2026-xxx)

- Adds NIRDA and VISDA classes which contain accurate and up to date parameters to perform timing and data volume calculations.
  - Adds tests for these new classes.
- Adds overhead class which accounts for pre- and post- overhead timings for both VISDA and NIRDA
- Adds baseline short-term calendar runner script to docs/
- Adds ability to merge back-to-back observations of the same target.
- Adds ability to override payload parameters set by original long-term calendar on a per-priority type basis.
- Adds warnings if single NIRDA or VISDA data file exceeds payload limits.