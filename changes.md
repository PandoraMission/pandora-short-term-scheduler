## v1.2.2 (2026-xxx)

- Lance noted that our nirda size was not divisible by 1024 which may lead to edge case problems that could be causing nirda crashes.
  - Changes y_size from 250 to 256 and y_start from 962 to 959.

## v1.2.1 (2026-xxx)

- Adds in the ability to use the dynamic Earth limb keepout.

## v1.2.0 (2026-xxx)

- Adds NIRDA and VISDA classes which contain accurate and up to date parameters to perform timing and data volume calculations.
- Adds overhead class which accounts for pre- and post- overhead timings for both VISDA and NIRDA.
- Adds baseline short-term calendar runner script to docs/
- Adds ability to merge back-to-back observations of the same target.
- Adds ability to override payload parameters set by original long-term calendar on a per-priority type basis.
  - Overrides can be taken from user provided dict or from the visda/nirda class defaults.
- Adds warnings if single NIRDA or VISDA data file exceeds payload limits.
- Adds dependence on NIRDA reset1 for VITL settling time. These parameters are all adjustable.
- Adds helper that renumbers both visits and sequencies to fix any misnumbering after merges.
- Adds log file to track changes, info, and warnings raised by the short term scheduler.
- Fixes minute-by-minute parsing to improve processing time.
- Adds several progress bars during various slower processing sections.
- Adds helper method to generate diagnostic data file.
  - Diag file contains a observation file manifest including compressed fits file names.
- Adds short term scheduler to processed calendar meta data.
- Adds override for PRI_CMD_DIR -> 9.
- Adds ability to convert det method 2 to 1 and adds pre-defined RA/DEC for the single ROI for observations that have max_num_rois = 1.
- Adds ability to clean bad symbols (like "+" and spaces " ") and other unsupported words (like "nan") in target IDs.
- Adds data volume exploration jupyter notebook to docs/
- Adds tests for all of these changes.
