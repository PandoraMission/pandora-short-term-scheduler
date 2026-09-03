## v1.4.1 (2026-09-03)
- The delivered calendar header now records the `pandoravisibility` version the run was built against (`Pandora_Visibility_Version`), alongside the existing `Short_Term_Scheduler_Version`, since that package's keepouts drive every visibility decision in the calendar.

## v1.4.0 (2026-08-27)
- Picks up the `pandoravisibility` v1.3.0 defaults, which are Pandora's flight keepouts rather than a loose starting point.
  - Fixes the roll sweep switching itself off. `_roll_sweep_enabled` was derived from the constructor arguments, so a scheduler that inherited the library's star-tracker keepouts applied them while never sweeping for a roll that could satisfy them. It now reads `Visibility._st_constraint_active`, falling back to the arguments only for a duck-typed visibility object that has no such attribute.
  - Note for callers: `None` no longer means "switch this keepout off". It means "defer to `pandoravisibility`", which now supplies a real angle for every keepout including the star-tracker ones. Pass `0` to disable one. `docs/roll-aware-visibility-example.ipynb` built its no-star-tracker comparison arm with `st_sun_min=None` and friends, which now gives that arm the full tracker keepouts, so it passes zeros instead.
  - Fixes `priority_0_earthlimb_min` no longer being a flat angle. The day/night pair was removed from the priority-0 keyword dict with `pop`, which only means "do not pass it", so `Visibility` fell back to its own defaults. Those are real angles since v1.3.0 and would have set the threshold instead of the flat angle the caller asked for. They are now sent as an explicit `None`.
- `validate_visibility` asks for its constraint breakdown at the roll the observation will actually fly. `get_all_constraints` gained a `roll` argument in `pandoravisibility` v1.3.0; without it the `star_tracker` verdict described the model's own attitude and could contradict the visibility result it was reporting on.
  - The per-tracker rows come from `get_star_tracker_breakdown`, which shares its geometry with that verdict, rather than being rebuilt from `get_star_tracker_angles`. The old path also measured the Earth limb from the geodetic horizon instead of the geocentric one the keepout is tested against.
  - The reported Earth-limb threshold reads the observer geometry from `Visibility._precompute` rather than rebuilding it.
  - The `side` label reports the Earth illumination angle while the dynamic wedge is in use. It was inferred by matching the effective threshold against `earthlimb_day_min`, which a continuous wedge never equals, so every step read as "night".

## v1.3.1 (2026-08-19)
- Fixed an issue when an observation is below the minimum observing time then it was not triggering an error.

## v1.3.0 (2026-08-19)
- `Calendar_Status` in the delivered XML header now reflects whether the run failed, not whether a validator just marked something as not visible. 
- Records the configuration the run applied on the XML header, so a delivered calendar says what it was built under instead of leaving it to be reconstructed from a log: the gap tolerances and their start buffers, `Max_Movement_Min`, `Roll_Step_Deg`, `Min_Power_Frac`, and every keepout in degrees including `Priority_0_Earthlimb_Min_Deg` (written only when in use) and `Use_Dynamic_Earthlimb`.
- Adds `priority_0_earthlimb_min` (default `None`), a stricter boresight Earth-limb keepout applied to priority-0 observations only, so they can be held further off the Earth to dissipate more heat. Every other keepout, star trackers included, is unchanged. `None` leaves the scheduler behaving exactly as before.
- Replaces blind gap filling with in-place adjustment. `_fill_gaps` dragged every observation's start back right at the one before it, unchecked and unbounded, and `_fix_visibility` then cleaned up the dark minutes that were created; between them they moved many observations by significant fractions since calendars now carry a lot of free time. Neither is called any more. Idle time is expected under the current conops, so observations keep the times the long-term calendar gave them and are only trimmed and grown in place.
  - `gap_report` now records what the passes did rather than how many gaps were closed, which is no longer something the scheduler attempts: `sequences_modified` split into grown and trimmed, `minutes_grown_at_starts`/`_at_stops`, `boundaries_clamped` and `overlaps_repaired`. `print_gap_summary` and the comparison plot report those. This also fixes `Sequences Modified`, which had been printing 0 on every run since long before this release because nothing ever wrote it.
- Adds `_grow_into_free_time` to gain back the observing time the removed gap filling used to achieve. Each observation expands outwards into adjacent idle time while the target stays visible at its scheduled roll, bounded by its neighbors and by `max_movement_minutes`, stepping over any dip the gap tolerances accept and always stopping on a visible minute.
- Adds `max_movement_minutes` (default 45): neither boundary of an observation may end up further than this from where the long-term calendar put it, or it is clamped there and reported in the error log, since a target needing to move that far is not really visible near its planned time or the long-term scheduler is not calculating visibility correctly.
- Adds an overlap guard that runs last and in both modes. The passes above cannot produce an overlap, but if one appears the earlier observation's stop is truncated to the later one's start and the repair logged, and anything still overlapping is reported for a manual fix.
- Merging can now absorb a short keepout violation between two observations of the same target, instead of letting it split them.
- `process_calendar(merge_similar_observations=...)` now defaults to `True`.
- Every product of a run now lands beside the long-term calendar it came from, rather than in whatever directory the run was launched from.
- Removes dead code left over from the gap-filling design. Gone: `_fill_gaps` and `_fix_visibility` (no longer called), `_trim_non_visible_heads` (never called; a dark head is trimmed by `_trim_to_longest_visible_block`, whose selected span has its leading dark minutes stripped), `max_sequence_duration` (set but never read, so growth was never capped by it), and the whole `force_gap_fill` mode with `_force_fill_gaps`, `_classify_gap_minute` and `earthlimb_hard_floor`. `force_gap_fill` was a constructor argument, so passing it is now an error; `docs/run_scheduler_example.py` no longer does. Output on the real week is unchanged.
- Adds `earthlimb_gap_tolerance_start_buffer` (default 12 min), the boresight counterpart to `st_gap_tolerance_start_buffer`. The gap tolerance may not be spent at the very beginning of an observation: an observation that opens with the boresight inside the Earth-limb keepout is not worth starting. Both buffers are enforced together, because moving the start to clear one can push it into a violation of the other. `_enforce_st_start_buffer` is accordingly now `_enforce_start_buffers`.
- Fixes keepout defaults silently diverging from `pandoravisibility`. `moon_min`, `sun_min`, and `earthlimb_min` restated the library's defaults in `ScheduleProcessor.__init__`, and `moon_min` had drifted to 20 deg against the library's 25 deg, so a scheduler built without an explicit moon keepout quietly used a looser one. All keepouts now default to `None`, meaning the constraint is left out of the `Visibility` call and that package's own default applies. Configurations that pass their keepouts explicitly, including `docs/run_scheduler_example.py`, are unaffected.
- Fixes the roll sweep running when no star-tracker constraint is active. The sweep was gated on whether a star-tracker argument had been *passed*, but a limit of zero disables that keepout, so `st_sun_min=0` switched on a full roll sweep for constraints that were never applied. The gate now tests for a limit greater than zero.
- Folds the day/night Earth-limb keepouts into the single forwarding dict rather than a second special-cased path, so a keepout cannot reach `Visibility` on one path and not the other.
- Fixes an observation exactly `min_sequence_duration` long being rejected as too short. Subtracting two `Time` objects an exact 8 minutes apart does not give 480 s: it gives 479.9999999999983 s at some epochs and 480.0000000000079 s at others, so the six passes that shorten an observation and then check the result were deciding on the date rather than on the schedule.
- Adds pointing plots, Every target gets its own color across all three, with black reserved for idle.
  - `plot_pointing_timeline` — boresight right ascension and declination across the week.
  - `plot_keepout_angles` — 3x3 grid of Sun, Earth and Moon angle for the boresight and each star tracker, with the configured Sun and Moon keep-outs drawn. The Earth row is the angle to the Earth centre, so no keep-out line is drawn on it; the Earth keep-outs are limb relative.
  - `plot_earth_illumination` — Earth-centre angle against how sunlit the limb point each axis grazes is, which is the angle the dynamic DPC wedge is keyed on.
  - All three share one `PointingTimeline` per calendar, so a full set costs about what one costs (~10 s on a week). `docs/run_scheduler_example.py` saves all three beside the calendar.
- Visibility plot improvements:
  - Fixes the visibility Gantt evaluating visibility at the wrong roll. It read the scheduler's swept-roll cache, which is only populated when that same processor instance built the schedule, so plotting a calendar loaded from XML judged the star trackers at the default attitude: 1066 min painted non-visible against a true 2. It now uses the roll written onto each observation, which is the one that will be flown.
  - Splits each bar in the visibility Gantt: the upper half keeps the priority color, the lower half shows the prescribed roll.
  - Adds a duty-cycle line to the visibility Gantt title: observed minutes against the wall-clock span they cover, so idle time counts against it.

## v1.2.3 (2026-08-19)
- Adds `st_gap_tolerance_start_buffer` (default 12 min). The star trackers must be visible for that many minutes at the beginning of every observation, measured from its start time, with no gap tolerance applied; without it the spacecraft cannot acquire good pointing. Observations that open with a tracker dropout have their start trimmed forward to the first minute that clears the buffer. Ones that cannot be fixed, because no stretch of the observation clears it or because trimming would drop below the minimum duration, are left alone and reported in the error log.
- Fixes gap tolerance being judged at the wrong roll. `_is_gap_tolerable` took its star-tracker verdict from `get_all_constraints`, which accepts no roll argument and so always evaluated the trackers at the `Visibility` instance's roll rather than the roll the observation actually flies. The tracker check now goes through `get_star_tracker_breakdown` at the swept roll. A sun/moon/planet keepout failure is now also explicitly never tolerable, rather than falling through the classification.
- A star-tracker check that cannot be evaluated is now reported to the error log instead of being inferred from whether the boresight was clear. The gap is then treated as intolerable and trimmed away.

## v1.2.2 (2026-08-12)

- Lance noted that our nirda size was not divisible by 1024 which may lead to edge case problems that could be causing nirda crashes.
  - Changes y_size from 250 to 256 and y_start from 962 to 959.
- Fixes issue where the gnatt plot would break if the calendar was too long

## v1.2.1 (2026-08-12)

- Adds in the ability to use the dynamic Earth limb keepout.

## v1.2.0 (2026-08-12)

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
