## v1.2.3 (2026-xxx)
- Adds `st_gap_tolerance_start_buffer` (default 12 min). The star trackers must be visible for that many minutes at the beginning of every observation, measured from its start time, with no gap tolerance applied; without it the spacecraft cannot acquire good pointing. Observations that open with a tracker dropout have their start trimmed forward to the first minute that clears the buffer. Ones that cannot be fixed, because no stretch of the observation clears it or because trimming would drop below the minimum duration, are left alone and reported in the error log.
- Fixes gap tolerance being judged at the wrong roll. `_is_gap_tolerable` took its star-tracker verdict from `get_all_constraints`, which accepts no roll argument and so always evaluated the trackers at the `Visibility` instance's roll rather than the roll the observation actually flies. The tracker check now goes through `get_star_tracker_breakdown` at the swept roll. A sun/moon/planet keepout failure is now also explicitly never tolerable, rather than falling through the classification.
- A star-tracker check that cannot be evaluated is now reported to the error log instead of being inferred from whether the boresight was clear. The gap is then treated as intolerable and trimmed away.

## v1.2.2 (2026-xxx)

- Lance noted that our nirda size was not divisible by 1024 which may lead to edge case problems that could be causing nirda crashes.
  - Changes y_size from 250 to 256 and y_start from 962 to 959.
- Fixes issue where the gnatt plot would break if the calendar was too long

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
