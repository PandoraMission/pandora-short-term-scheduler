"""Schedule processing utilities.

This module implements the ScheduleProcessor which is responsible for
adjusting a `ScienceCalendar` when updated spacecraft ephemerides (TLEs)
are provided. The processor performs these high-level steps:

- extract a time window to process
- compute minute-by-minute visibility using `pandoravisibility.Visibility`
- identify visibility gaps and attempt to extend previous sequences or
  shrink following sequences to reduce unobserved time
- update payload integration parameters (VIS/NIR) to fit the new timing
- assemble a comprehensive gap/processing report
"""

# Standard library
import copy
import logging
import uuid
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - zoneinfo is stdlib on 3.9+
    ZoneInfo = None
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, SkyCoord
from astropy.coordinates import get_body
from astropy.time import Time, TimeDelta
from pandoravisibility import Visibility

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is an optional
    tqdm = None


class _NullProgress:
    """No-op stand-in for a tqdm bar (used when tqdm is unavailable)."""

    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


from .models import ObservationSequence, ScienceCalendar, Visit
from .nirda import NirdaData
from .overhead import OverheadTiming
from .roll import apply_rolls_to_calendar, find_best_rolls_for_visit
from .visda import VisdaData


# Characters that are not allowed in target name fields (``Target`` /
# ``TargetID``) because they break downstream filename and identifier
# handling. Each bad symbol is mapped to its safe replacement and substituted
# by the ``fix_bad_data`` pass.
BAD_NAME_SYMBOLS = {
    "+": "_",
    " ": "_",
}

# Tag names whose values are expected to be non-numeric (names, IDs,
# timestamps, TLE lines). They are excluded from the NaN-like value scan run
# by the ``fix_bad_data`` pass.
NON_NUMERIC_TAGS = frozenset(
    {
        "Target",
        "TargetID",
        "ID",
        "Start",
        "Stop",
        "TLE_Line1",
        "TLE_Line2",
        "Calendar_Status",
    }
)


class ScheduleProcessor:
    """Main class for processing and adjusting science calendars with updated TLE.

    Public methods
    --------------
    - process_calendar(calendar, window_start=None, window_duration_days=21, verbose=False)
        Process a calendar and return an updated ScienceCalendar.
    - get_gap_report()
        Return a structured report summarizing visibility gaps and actions taken.

    The class expects `Visibility(tle1, tle2)` to offer `get_visibility(coord, times)`
    returning a boolean array of the same length as `times`.
    """

    @staticmethod
    def _to_deg(
        value: Optional[float],
    ) -> Optional[u.Quantity]:
        """Convert a plain float (degrees) to an astropy Quantity.

        Returns *None* unchanged so callers can use ``None`` to fall back
        to the ``Visibility`` class default for that constraint.
        """
        if value is None:
            return None
        return value * u.deg

    def __init__(
        self,
        tle_line1: str,
        tle_line2: str,
        vda_pre_sequence_overhead: u.Quantity | None = None,
        vda_post_sequence_overhead: u.Quantity | None = None,
        nirda_pre_sequence_overhead: u.Quantity | None = None,
        nirda_post_sequence_overhead: u.Quantity | None = None,
        override_nirda_parameters: Optional[Dict[int, Dict[str, Any]]] = None,
        override_visda_parameters: Optional[Dict[int, Dict[str, Any]]] = None,
        override_payload_parameters: Optional[Dict[Any, Dict[str, Dict[str, Any]]]] = None,
        max_file_size_uncompressed: u.Quantity = 830.0 * 1000 * 1000 * u.byte,
        max_file_size_compressed: u.Quantity = 255.0 * 1000 * 1000 * u.byte,
        update_nirda_reset1_for_vitl: bool = True,
        vitl_settling_time: u.Quantity = 60.0 * u.s,
        convert_single_roi_to_predefined: bool = True,
        fix_bad_data: bool = True,
        moon_min: Optional[float] = 20.0,
        sun_min: Optional[float] = 91.0,
        earthlimb_min: Optional[float] = 20.0,
        earthlimb_day_min: Optional[float] = None,
        earthlimb_night_min: Optional[float] = None,
        mars_min: Optional[float] = None,
        jupiter_min: Optional[float] = None,
        st_sun_min: Optional[float] = None,
        st_moon_min: Optional[float] = None,
        st_earthlimb_min: Optional[float] = None,
        st1_earthlimb_min: Optional[float] = None,
        st2_earthlimb_min: Optional[float] = None,
        roll_step: float = 2.0,
        min_power_frac: float = 0.7,
        earthlimb_gap_tolerance: int = 0,
        st_gap_tolerance: int = 0,
        st_gap_tolerance_start_buffer: int = 12,
        use_dynamic_earthlimb: bool = False,
        force_gap_fill: bool = False,
        earthlimb_hard_floor: float = 5.0,
    ) -> None:
        """
        Initialize the scheduler with TLE and parameters.

        Parameters:
        -----------
        tle_line1, tle_line2 : str
            TLE lines for satellite
        vda_pre_sequence_overhead : Quantity, optional
            VDA pre-sequence overhead (default is None which will use the overhead defaults).
        vda_post_sequence_overhead : Quantity, optional
            VDA post-sequence overhead (default is None which will use the overhead defaults).
        nirda_pre_sequence_overhead : Quantity, optional
            NIRDA pre-sequence overhead (default is None which will use the overhead defaults).
        nirda_post_sequence_overhead : Quantity, optional
            NIRDA post-sequence overhead (default is None which will use the overhead defaults).
        override_nirda_parameters : dict, optional
            Per-priority NIRDA payload overrides applied during the
            payload-update step. Maps an observation priority to a mapping of
            ``NirdaData`` field names to the values to force; a value of
            ``None`` means "use the ``NirdaData`` default". For example:
                ``{0: {'drop_frames_1': 2, 'drop_frames_3': None},
                   1: {'reset_frames_1': 30}}``
                    means: for every priority-0 observation set
                    ``drop_frames_1`` to 2 and ``drop_frames_3`` to the class
                    default; for priority 1 set ``reset_frames_1`` to 30. The
                    overridden values are written back onto the observation
                    before recomputing SC_Integrations.
            Field names are ``NirdaData`` attribute names; the corresponding
            XML tags are updated automatically. An iterable of field names is
            also accepted (treated as all-default, e.g.
            ``{0: ['drop_frames_1']}``). Defaults to no overrides.
        override_visda_parameters : dict, optional
            Per-priority VISDA payload overrides, structured identically to
            ``override_nirda_parameters`` but using ``VisdaData`` field names
            (e.g. ``{0: {'frames_per_coadd': 5}}``). Defaults to no
            overrides.
        override_payload_parameters : dict, optional
            General per-priority overrides written directly onto the payload
            XML by tag name (the CalendarCleaner ``config.json`` format).
            Structure: ``{priority: {section: {xml_tag: value}}}`` where
            *section* is e.g. ``'AcquireInfCamImages'`` or
            ``'AcquireVisCamScienceData'`` and *xml_tag* is a literal payload
            tag (``ROI_StartX``, ``ROI_SizeX``, ``SC_Resets2``,
            ``FramesPerCoadd``, ``RiceX`` ...). Tags missing from an
            observation are created. Priority keys may be ints (``0``) or
            ``'Priority_0'`` strings. These are applied *before* the
            integration counts are recomputed, so size/coadd/reset changes
            flow through. Free-time observations are skipped. Defaults to no
            overrides.
        max_file_size_uncompressed : Quantity[byte], optional
            Maximum allowed *uncompressed* data volume per detector per
            sequence. A warning is raised during the payload-update step if a
            sequence's computed NIRDA or VISDA data exceeds this.
            Defaults to 830 MB.
        max_file_size_compressed : Quantity[byte], optional
            Maximum allowed *compressed* data volume per detector per
            sequence. A warning is raised if a sequence's computed compressed
            NIRDA or VISDA data exceeds this.
            Defaults to 255 MB.
        update_nirda_reset1_for_vitl : bool, optional
            When True (default), each NIRDA observation's ``reset_frames_1``
            is adjusted via ``NirdaData.update_for_vitl`` to cover
            ``vitl_settling_time`` before its integration count is computed,
            and the resulting ``SC_Resets1`` is written back to the payload.
        vitl_settling_time : Quantity[second], optional
            Minimum VITL detector settling time used when
            ``update_nirda_reset1_for_vitl`` is True. Defaults to 60 s.
        convert_single_roi_to_predefined : bool, optional
            When True (default), any observation whose VIS section requests a
            single brightest-star auto-detect ROI (``MaxNumStarRois == 1`` and
            ``StarRoiDetMethod == 2``) is converted to the predefined-ROI
            method (``StarRoiDetMethod == 1``) with the target RA/Dec written
            as the single predefined ROI (``numPredefinedStarRois == 1``,
            ``PredefinedStarRoiRa/RA1``, ``PredefinedStarRoiDec/Dec1``).
        fix_bad_data : bool, optional
            When True (default), each observation's target name fields
            (``Target``/``TargetID``) have invalid symbols replaced per
            ``BAD_NAME_SYMBOLS`` (e.g. ``+`` -> ``_``), and all other
            (numeric) fields are scanned for NaN-like values, which are logged
            as warnings. Free-time observations are exempt from the NaN scan
            because their RA/Dec are expected to be NaN.
        moon_min, sun_min, earthlimb_min, mars_min, jupiter_min : float, optional
            Minimum angular separations (degrees) for visibility constraints.
        earthlimb_day_min : float, optional
            Earth-limb keepout angle (degrees) on the **day** side of the
            terminator.  When ``None`` (default), ``earthlimb_min`` is used
            for both day and night sides (``Visibility`` default behaviour).
        earthlimb_night_min : float, optional
            Earth-limb keepout angle (degrees) on the **night** side of the
            terminator.  When ``None`` (default), ``earthlimb_min`` is used
            for both day and night sides (``Visibility`` default behaviour).
        st_sun_min, st_moon_min, st_earthlimb_min, st1_earthlimb_min,
        st2_earthlimb_min : float, optional
            Additional constraints for star trackers.
        roll_step : float, optional
            Roll-angle sweep resolution in degrees (default 2.0).
        min_power_frac : float, optional
            Minimum acceptable solar-panel power fraction (0-1).
            Roll candidates below this are rejected (default 0.7).
        earthlimb_gap_tolerance : int, optional
            Maximum number of contiguous minutes of earth-limb
            visibility violations to tolerate within a sequence
            (default 0).  Short dips are kept; longer gaps trigger
            trimming.
        st_gap_tolerance : int, optional
            Maximum number of contiguous minutes of star-tracker
            visibility violations to tolerate within a sequence
            (default 0).
        st_gap_tolerance_start_buffer : int, optional
            Minutes of uninterrupted star-tracker visibility required at
            the beginning of every observation, measured from its start
            time (default 12). ``st_gap_tolerance`` lets a tracker dropout
            be tolerated mid-observation, but the spacecraft cannot
            acquire good pointing without the trackers at the start, so no
            tolerance is applied inside this buffer. Sequences that open
            dark are trimmed forward to the first minute that clears it;
            set to 0 to disable the check.
        use_dynamic_earthlimb : bool, default=true
            If True, then uses the dynamic DPC boresight Earth limb.
            This is the wedge shape keepout based on the Earth illumination.
        force_gap_fill : bool, optional
            When True, fill all gaps between sequences even if the
            extended time violates keepout constraints.  The
            visibility-fixing, tail-trimming, and mid-sequence
            trimming passes are skipped so the schedule has no
            temporal gaps.  Validation will still report keepout
            violations (default False).
        earthlimb_hard_floor : float, optional
            Absolute minimum earth-limb angle (degrees) allowed when
            force-filling gaps.  Even in force mode the scheduler
            will not extend a sequence into minutes where the
            earth-limb separation drops below this value
            (default 5.0).
        """
        # Validate TLE format
        if not isinstance(tle_line1, str):
            raise ValueError("Invalid TLE line 1 format")
        if not isinstance(tle_line2, str):
            raise ValueError("Invalid TLE line 2 format")
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2

        _kw: Dict[str, Any] = dict(
            moon_min=self._to_deg(moon_min),
            sun_min=self._to_deg(sun_min),
            earthlimb_min=self._to_deg(earthlimb_min),
            mars_min=self._to_deg(mars_min),
            jupiter_min=self._to_deg(jupiter_min),
            st_sun_min=self._to_deg(st_sun_min),
            st_moon_min=self._to_deg(st_moon_min),
            st_earthlimb_min=self._to_deg(st_earthlimb_min),
            st1_earthlimb_min=self._to_deg(st1_earthlimb_min),
            st2_earthlimb_min=self._to_deg(st2_earthlimb_min),
            use_dynamic_earthlimb=use_dynamic_earthlimb
        )
        # Only forward day/night earthlimb keepouts when explicitly set so that
        # Visibility falls back to earthlimb_min for whichever side is None.
        if earthlimb_day_min is not None:
            _kw["earthlimb_day_min"] = self._to_deg(earthlimb_day_min)
        if earthlimb_night_min is not None:
            _kw["earthlimb_night_min"] = self._to_deg(earthlimb_night_min)
        # Strip None entries so Visibility uses its own class-level defaults
        # for any constraint the caller left unset.
        _kw = {k: v for k, v in _kw.items() if v is not None}
        self.visibility = Visibility(tle_line1, tle_line2, **_kw)

        self.min_sequence_duration = TimeDelta(8 * 60 * u.s)
        self.max_sequence_duration = TimeDelta(90 * 60 * u.s)

        # Gap tolerance: maximum contiguous non-visible minutes to allow
        self.earthlimb_gap_tolerance = earthlimb_gap_tolerance
        self.st_gap_tolerance = st_gap_tolerance
        self.st_gap_tolerance_start_buffer = st_gap_tolerance_start_buffer
        self.force_gap_fill = force_gap_fill
        self.earthlimb_hard_floor = earthlimb_hard_floor

        # Roll sweep configuration
        self.roll_step = roll_step
        self.min_power_frac = min_power_frac
        # Roll sweep is only meaningful when star-tracker constraints are
        # active (those constraints depend on roll; boresight constraints
        # do not).  Disable the sweep when no ST parameters were given so
        # that vanilla ScheduleProcessor(tle1, tle2) behaves as before.
        _st_params = (
            st_sun_min,
            st_moon_min,
            st_earthlimb_min,
            st1_earthlimb_min,
            st2_earthlimb_min,
        )
        self._roll_sweep_enabled: bool = any(p is not None for p in _st_params)

        # Per-visit, per-target precomputed rolls populated during
        # _process_all_sequences.  Structure:
        #   { visit_id: { target_name: roll_deg_or_None } }
        self._computed_target_rolls: Dict[str, Dict[str, Optional[float]]] = {}

        # Validate any explicitly supplied overheads carry time units so that
        # downstream .to(u.s) / .to(u.us) calls succeed. ``None`` means "use
        # the OverheadTiming default derived from the modelled MOC sequence".
        _overhead_params = {
            "vda_pre_sequence_overhead": vda_pre_sequence_overhead,
            "vda_post_sequence_overhead": vda_post_sequence_overhead,
            "nirda_pre_sequence_overhead": nirda_pre_sequence_overhead,
            "nirda_post_sequence_overhead": nirda_post_sequence_overhead,
        }
        for _name, _val in _overhead_params.items():
            if _val is None or isinstance(_val, TimeDelta):
                continue
            if isinstance(_val, u.Quantity):
                try:
                    _val.to(u.s)
                except u.UnitConversionError:
                    raise ValueError(
                        f"{_name} must have time units; "
                        f"got unit '{_val.unit}'"
                    )
            else:
                raise TypeError(
                    f"{_name} must be an astropy Quantity or TimeDelta "
                    f"with time units; got {type(_val).__name__!r}"
                )

        # Collect the overheads into a single OverheadTiming so the
        # payload-update step can hand it straight to the NIRDA/VISDA data
        # classes. Note OverheadTiming uses "visda" naming for the VDA fields
        # and fills any None with its modelled default.
        self.overhead = OverheadTiming(
            visda_pre_overhead_time=vda_pre_sequence_overhead,
            visda_post_overhead_time=vda_post_sequence_overhead,
            nirda_pre_overhead_time=nirda_pre_sequence_overhead,
            nirda_post_overhead_time=nirda_post_sequence_overhead,
        )

        # Per-priority payload overrides applied during the payload-update
        # step. Structure: { priority: {field_name: value-or-None} }, where a
        # None value means "use the data-class default". An iterable of field
        # names is also accepted (treated as all-default).
        self._override_nirda_parameters: Dict[int, Any] = (
            override_nirda_parameters or {}
        )
        self._override_visda_parameters: Dict[int, Any] = (
            override_visda_parameters or {}
        )
        # General per-priority XML-tag payload overrides (cleaner config.json
        # format). Normalized to int priority keys.
        self._override_payload_parameters: Dict[int, Any] = (
            self._normalize_priority_keys(override_payload_parameters)
        )

        # Per-observation data-volume limits. A warning is raised during the
        # payload-update step if a sequence's computed NIRDA/VISDA data
        # exceeds these.
        self.max_file_size_uncompressed = max_file_size_uncompressed
        self.max_file_size_compressed = max_file_size_compressed

        # VITL settling: when enabled, each NIRDA observation has its
        # reset_frames_1 adjusted to cover vitl_settling_time before its
        # integration count is computed.
        self.update_nirda_reset1_for_vitl = update_nirda_reset1_for_vitl
        self.vitl_settling_time = vitl_settling_time

        # Single-ROI conversion: when enabled, any observation whose VIS
        # section requests exactly one star ROI via the brightest-star
        # auto-detect method (MaxNumStarRois == 1, StarRoiDetMethod == 2) is
        # converted to the predefined-ROI method (StarRoiDetMethod == 1) with
        # the target RA/Dec supplied as the single predefined ROI.
        self.convert_single_roi_to_predefined = convert_single_roi_to_predefined

        # Bad-data fixes: when enabled, target name fields have invalid
        # symbols (see BAD_NAME_SYMBOLS) replaced, and all other fields are
        # scanned for NaN-like values (reported as warnings).
        self.fix_bad_data = fix_bad_data

        # Enhanced gap tracking with before/after comparison
        self.gap_report = {
            "original_calendar_stats": {},
            "processed_calendar_stats": {},
            "visibility_analysis": {
                "original_gaps": [],
                "filled_gaps": [],
                "remaining_gaps": [],
                "unfillable_gaps": [],
            },
            "sequence_modifications": {
                "extended_sequences": [],
                "shortened_sequences": [],
                "unchanged_sequences": [],
            },
            "processing_summary": {
                "total_gaps_found": 0,
                "gaps_filled": 0,
                "gaps_remaining": 0,
                "total_time_recovered_minutes": 0,
                "sequences_modified": 0,
            },
        }

    def process_calendar(
        self,
        calendar: ScienceCalendar,
        window_start: Optional[Any] = None,
        window_duration_days: int = 21,
        merge_similar_observations: bool = False,
        log_path: Optional[Any] = None,
        verbose: bool = False,
    ) -> ScienceCalendar:
        """Process a `ScienceCalendar` and return an updated calendar.

        The processor performs a time-window extraction, computes
        minute-by-minute visibility using the configured TLEs, identifies
        visibility gaps, attempts to fill gaps by extending previous
        sequences (and shrinking following sequences), updates payload
        integration parameters, and produces a `gap_report` summary.

        Side effects
        -----------
        - The returned `ScienceCalendar` will have its `.metadata` updated
          to include the TLE lines, a `processed_datetime` and the
          generated `gap_report` to aid downstream writing and analysis.

        Parameters
        ----------
        calendar : ScienceCalendar
            Input calendar to process.
        window_start : str or astropy.time.Time, optional
            ISO string or Time object indicating the window start.
        window_duration_days : int, optional
            Number of days to include in the processing window.
        merge_similar_observations : bool, optional
            When True, adjacent observation sequences in the same visit that
            share the same target and pointing are merged into a single
            longer sequence.
            Defaults to False.
        log_path : str or pathlib.Path, optional
            Base path for the run log files. The ".log" (everything) and
            ".errors.log" (warnings/errors only, created lazily) files are
            named after this path's stem. When omitted, the input calendar's
            ``source_path`` is used; if that is unavailable, only console
            logging is produced.
        verbose : bool, optional
            When True, INFO-level diagnostics are echoed to the console; the
            ".log" file always captures them regardless. Warnings and
            errors are shown on the console either way.

        Returns
        -------
        ScienceCalendar
            Processed calendar with updated sequences and metadata.
        """

        # Configure logging for this run (console + per-calendar log files).
        self._setup_run_logging(calendar, verbose, log_path)

        # Clear previous gap report
        self._initialize_gap_report()

        self._print("Processing calendar with TLE:")
        self._print(f"  Line 1: {self.tle_line1}")
        self._print(f"  Line 2: {self.tle_line2}")

        # Extract windowed calendar FIRST
        windowed_calendar = self._extract_time_window(
            calendar, window_start, window_duration_days, verbose
        )

        # Normalize target names up front (before the roll sweep). The roll
        # sweep keys its results by target name, so any +/space -> _ rename
        # must happen before the sweep or the renamed targets lose their
        # swept roll and fall back to the sun-derived value.
        if getattr(self, "fix_bad_data", False):
            self._normalize_target_names(windowed_calendar, verbose)

        # Capture windowed calendar statistics (not original full calendar)
        self._analyze_original_calendar(
            windowed_calendar
        )  # Use windowed version

        # Analyze original visibility gaps in the windowed calendar
        self._analyze_original_visibility(windowed_calendar, verbose)

        # Process sequences
        processed_calendar = self._process_all_sequences(
            windowed_calendar, verbose
        )

        # Calculate and apply roll angles to all sequences
        # This ensures all sequences of the same target within a visit
        # have the same roll angle.  Precomputed visibility-aware rolls
        # take precedence over the sun-derived default.
        apply_rolls_to_calendar(
            processed_calendar,
            verbose=verbose,
            precomputed_rolls=self._computed_target_rolls,
        )

        # Optionally merge back-to-back same-target sequences within each
        # visit. This runs *after* all gap-filling, trimming, and other
        # duration/timing adjustments so it operates on the final scheduled
        # boundaries. Because merging extends a sequence over its neighbor,
        # the merged sequences' payload integration counts are recomputed
        # for their new combined durations.
        if merge_similar_observations:
            processed_calendar = self._merge_similar_observations(
                processed_calendar, verbose
            )
            processed_calendar = self._update_payload_parameters(
                processed_calendar
            )

        # Renumber visit and observation IDs sequentially. This runs last,
        # after any merges/time changes that may have dropped IDs, so the
        # delivered calendar always has contiguous, ordered identifiers.
        self._renumber_ids(processed_calendar, verbose)

        # Analyze processed calendar
        self._analyze_processed_calendar(processed_calendar)

        # Generate comprehensive report
        self._finalize_gap_report()

        calendar_status = "VALID"
        validation_counts: Dict[str, int] = {}

        target_issues = self.validate_target_names(
            processed_calendar, report_issues=False
        )
        if target_issues:
            validation_counts["target_name"] = len(target_issues)
            calendar_status = "INVALID"

        vis_issues = self.validate_visibility(
            processed_calendar, report_issues=False
        )
        if vis_issues:
            validation_counts["visibility"] = len(vis_issues)
            calendar_status = "INVALID"

        payload_issues = self.validate_payload_exposures(
            processed_calendar, report_issues=False
        )
        if payload_issues:
            validation_counts["payload_exposure"] = len(payload_issues)
            calendar_status = "INVALID"

        overlap_issues = self.validate_no_overlaps_astropy(
            processed_calendar, report_issues=False
        )
        if overlap_issues:
            validation_counts["overlap"] = len(overlap_issues)
            calendar_status = "INVALID"

        timing_result = self.validate_sequence_timing(
            processed_calendar, report_issues=False
        )
        timing_total = timing_result["timing_summary"]["total_issues"]
        if timing_total > 0:
            validation_counts["sequence_timing"] = timing_total
            calendar_status = "INVALID"

        roll_issues = self.validate_roll_consistency(
            processed_calendar, report_issues=False
        )
        if roll_issues:
            validation_counts["roll_consistency"] = len(roll_issues)
            calendar_status = "INVALID"

        # Print compact validation summary
        if validation_counts:
            self._print(
                f"\n--- Validation: {calendar_status} "
                f"({sum(validation_counts.values())} issues) ---"
            )
            for cat, cnt in validation_counts.items():
                self._print(f"  {cat}: {cnt}")
            self._print(
                "Run print_validation_summary(calendar) "
                "for actionable details.\n"
            )
        else:
            self._print(f"\n--- Validation: {calendar_status} " f"(0 issues) ---\n")

        new_metadata = copy.deepcopy(processed_calendar.metadata)
        new_metadata.update(
            {
                "valid_from": self.window_start.isot,
                "expires": self.window_end.isot,
                "tle_line1": self.tle_line1,
                "tle_line2": self.tle_line2,
                "created": Time.now().isot,
                "delivery_id": str(uuid.uuid4()),
                "total_visits": len(processed_calendar.visits),
                "total_sequences": sum(
                    len(visit.sequences) for visit in processed_calendar.visits
                ),
                "calendar_status": calendar_status,
            }
        )

        # Attach updated metadata to the processed calendar
        processed_calendar.metadata = new_metadata

        return processed_calendar

    def _extract_time_window(
        self,
        calendar: ScienceCalendar,
        window_start: Optional[Any],
        window_duration_days: int,
        verbose: bool,
    ) -> ScienceCalendar:
        """Extract time-based window from calendar."""
        if isinstance(window_start, str):
            window_start = Time(window_start, format="isot", scale="utc")

        window_end = window_start + TimeDelta(
            window_duration_days, format="jd"
        )

        self.window_start = window_start
        self.window_end = window_end

        self._print(f"Extracting window: {window_start} to {window_end}")

        # Find sequences within window
        windowed_visits = []
        for visit in calendar.visits:
            # complain if there are empty visits
            if not visit.sequences:
                self._print(f"Warning: Empty sequence list for visit {visit.id}")
            windowed_sequences = []
            for seq in visit.sequences:
                seq_start = seq.start_time
                seq_stop = seq.stop_time

                # Include sequence if it overlaps with window. First complete sequence.
                if (
                    seq_start < window_end
                    and seq_stop > window_start
                    and seq_start >= window_start
                ):
                    windowed_sequences.append(seq)

            if windowed_sequences:
                windowed_visits.append(
                    Visit(id=visit.id, sequences=windowed_sequences)
                )

        return ScienceCalendar(
            metadata=calendar.metadata, visits=windowed_visits
        )

    # Tolerances used when deciding whether two sequences can be merged.
    _MERGE_ADJACENCY_TOL_SEC = 1.0  # max stop-to-start gap (seconds)
    _MERGE_POINTING_TOL_DEG = 1e-6  # max RA/Dec difference (degrees)

    def _renumber_ids(
        self, calendar: ScienceCalendar, verbose: bool = False
    ) -> ScienceCalendar:
        """Renumber visit and observation IDs to be sequential.

        Visits are numbered ``0001``, ``0002``, ... in their current order,
        and within each visit the observation sequences are numbered
        ``001``, ``002``, ... in their current order. This is run after all
        time changes and merges so that any IDs dropped along the way are
        replaced with a contiguous, ordered set.

        Parameters
        ----------
        calendar : ScienceCalendar
            Calendar whose IDs are renumbered in place.
        verbose : bool, optional
            Print the number of IDs changed when True.

        Returns
        -------
        ScienceCalendar
            The same calendar instance, with IDs renumbered.
        """
        changed = 0
        visit_id_map: Dict[str, str] = {}
        for visit_index, visit in enumerate(calendar.visits, start=1):
            new_visit_id = f"{visit_index:04d}"
            if visit.id is not None:
                visit_id_map[visit.id] = new_visit_id
            if visit.id != new_visit_id:
                self._print(
                    f"RENUMBER visit ID '{visit.id}' -> '{new_visit_id}'"
                )
                visit.id = new_visit_id
                changed += 1

            for seq_index, seq in enumerate(visit.sequences, start=1):
                new_seq_id = f"{seq_index:03d}"
                if seq.id != new_seq_id:
                    self._print(
                        f"{self._seq_prefix(visit.id, seq)} | RENUMBER "
                        f"observation ID '{seq.id}' -> '{new_seq_id}'"
                    )
                    seq.id = new_seq_id
                    changed += 1

        # Re-key the precomputed roll cache onto the new visit IDs.  Anything
        # that reads it after this point (validate_visibility, the visibility
        # Gantt plot) looks up by the *renumbered* visit ID, so leaving the
        # cache on the old IDs silently returns a neighbouring visit's rolls
        # -- or none at all -- and reports bogus keepout violations.
        cached_rolls = getattr(self, "_computed_target_rolls", None)
        if visit_id_map and cached_rolls:
            self._computed_target_rolls = {
                visit_id_map.get(old_id, old_id): rolls
                for old_id, rolls in cached_rolls.items()
            }

        self._print(f"Renumbered IDs: {changed} identifier(s) updated.")

        return calendar

    def _merge_similar_observations(
        self, calendar: ScienceCalendar, verbose: bool = False
    ) -> ScienceCalendar:
        """Merge back-to-back same-target sequences within each visit.

        Two consecutive sequences (in start-time order) inside the same
        visit are merged when they:

        1. belong to the same visit (sequences are grouped per visit),
        2. observe the same target with the same pointing (RA/Dec), and
        3. are contiguous in time, the second sequence starts at (within
           a tolerance of) the first sequence's stop time.

        The merged sequence keeps the first sequence's identity, priority,
        and payload parameters, and extends its ``stop_time`` to the second
        sequence's ``stop_time``. Merging is applied transitively, so a run
        of three or more contiguous same-target sequences collapses into a
        single sequence.

        Parameters
        ----------
        calendar : ScienceCalendar
            Calendar to merge in place-safe fashion (a new calendar with
            new visits/sequences is returned; the input is not mutated).
        verbose : bool, optional
            If True, print a line for each merge performed.

        Returns
        -------
        ScienceCalendar
            Calendar with eligible sequences merged.
        """
        merged_count = 0
        new_visits: List[Visit] = []

        for visit in calendar.visits:
            # Process sequences in chronological order so "right after each
            # other" is well defined regardless of input ordering.
            ordered = sorted(visit.sequences, key=lambda s: s.start_time)

            merged_sequences: List[ObservationSequence] = []
            for seq in ordered:
                if merged_sequences and self._can_merge(merged_sequences[-1], seq):
                    # Extend the previous (kept) sequence over this one.
                    previous = merged_sequences[-1]
                    self._print(
                        f"{self._seq_prefix(visit.id, previous)} | MERGE: "
                        f"absorbing sequence {seq.id} "
                        f"({self._seq_prefix(visit.id, seq)}); stop "
                        f"{previous.stop_time_str} -> {seq.stop_time_str}"
                    )
                    previous.stop_time = seq.stop_time
                    merged_count += 1
                else:
                    # Copy so the returned calendar never aliases the input.
                    merged_sequences.append(seq.copy())

            new_visits.append(Visit(id=visit.id, sequences=merged_sequences))

        self._print(
            f"Merged {merged_count} similar observation sequence(s) "
            f"across {len(calendar.visits)} visit(s)."
        )

        return ScienceCalendar(
            metadata=calendar.metadata,
            visits=new_visits,
            visibility=calendar.visibility,
        )

    def _can_merge(
        self, first: ObservationSequence, second: ObservationSequence
    ) -> bool:
        """Return True if ``second`` can be merged into ``first``.

        See :meth:`_merge_similar_observations` for the merge criteria.
        """
        # Same target (case-insensitive, whitespace-insensitive).
        if (first.target or "").strip().lower() != (second.target or "").strip().lower():
            return False

        # Same pointing.
        if (
            abs(first.ra - second.ra) > self._MERGE_POINTING_TOL_DEG
            or abs(first.dec - second.dec) > self._MERGE_POINTING_TOL_DEG
        ):
            return False

        # Contiguous in time: second starts when first stops.
        gap_sec = (second.start_time - first.stop_time).sec
        return abs(gap_sec) <= self._MERGE_ADJACENCY_TOL_SEC

    def _process_all_sequences(
        self, calendar: ScienceCalendar, verbose: bool = False
    ) -> ScienceCalendar:
        """Iterate through sequences and build minute-resolution visibility.

        This internal routine constructs a synchronized time grid for the
        windowed calendar, queries visibility for each sequence target and
        accumulates a boolean minute-array (`all_minutes_bool`) describing
        which minutes are visible. It then calls the visibility-fixing
        and payload-update steps to produce the final calendar.

        Parameters
        ----------
        calendar : ScienceCalendar
            Windowed calendar to operate on.
        verbose : bool, optional
            If True, print progress messages.

        Returns
        -------
        ScienceCalendar
            Calendar with adjusted sequences and updated payload parameters.
        """

        working_calendar = deepcopy(calendar)

        # Snapshot each sequence's original timing so any shrink/elongate
        # performed by the gap-fill / trim passes below can be logged.
        original_timing = {
            (visit.id, seq.id): (seq.start_time, seq.stop_time)
            for visit in working_calendar.visits
            for seq in visit.sequences
        }

        # ── Pre-compute best roll per target per visit ──────────
        # Only run the sweep when star-tracker constraints are active;
        # boresight-only constraints are roll-independent.
        if self._roll_sweep_enabled:
            for visit in self._progress(
                working_calendar.visits,
                desc="Roll sweep",
                total=len(working_calendar.visits),
            ):
                visit_rolls = find_best_rolls_for_visit(
                    self.visibility,
                    visit,
                    roll_step=self.roll_step,
                    min_power_frac=self.min_power_frac,
                )
                self._computed_target_rolls[visit.id] = visit_rolls
                for tgt, r in visit_rolls.items():
                    self._print(f"  Visit {visit.id} / {tgt}: best roll = {r}")

        # Use initial time grid for processing
        total_minutes, start_time, end_time, time_grid = (
            self._get_synchronized_time_grid(working_calendar)
        )
        all_minutes_bool = np.zeros(total_minutes, dtype=bool)

        i = 0
        last_stop = deepcopy(start_time)

        vis_bar = self._progress_bar(
            sum(len(v.sequences) for v in working_calendar.visits),
            desc="Computing visibility",
        )
        for visit in working_calendar.visits:
            visit_rolls = self._computed_target_rolls.get(visit.id, {})

            for j, seq in enumerate(visit.sequences):

                # Compute gap since last sequence stop
                gap_length = int(
                    np.rint((seq.start_time - last_stop).sec / 60.0)
                )
                if gap_length > 0:
                    self._print(
                        f"{self._seq_prefix(visit.id, seq)} | GAP-FILL: "
                        f"extending start earlier by {gap_length} min to "
                        f"fill gap before this sequence"
                    )

                    if not self.force_gap_fill:
                        seq = self._fill_gaps(
                            seq, gap_length, visit_id=visit.id
                        )
                        visit.sequences[j] = seq  # persist change

                # Evaluate visibility for this sequence
                n_mins = int(np.rint(seq.duration.sec / 60.0))
                ra, dec = seq.ra, seq.dec
                target_coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
                deltas = np.arange(n_mins) * u.min
                times = seq.start_time + deltas

                # Use precomputed roll if sweep was enabled and roll found
                target_roll = visit_rolls.get(seq.target)
                if self._roll_sweep_enabled and target_roll is not None:
                    vis = self.visibility.get_visibility(
                        target_coord,
                        times,
                        roll=target_roll * u.deg,
                    )
                else:
                    vis = self.visibility.get_visibility(target_coord, times)
                end_index = min(i + len(vis), total_minutes)
                all_minutes_bool[i:end_index] = vis[: end_index - i]

                i += len(vis)
                last_stop = seq.stop_time
                vis_bar.update(1)
        vis_bar.close()

        # Fill remaining time after last sequence
        if i < total_minutes:
            all_minutes_bool[i:] = False
            self._print(f"Filled trailing {total_minutes - i} minutes as False")

        self.all_minutes_bool = (
            all_minutes_bool  # this is only necessary for testing.
        )

        if self.force_gap_fill:
            working_calendar = self._force_fill_gaps(working_calendar)
        else:
            working_calendar = self._fix_visibility(
                working_calendar, all_minutes_bool
            )

            # Trim non-visible tails that _fix_visibility cannot handle
            # (it only shrinks starts forward; tails need stop_time
            # shrunk).
            working_calendar = self._trim_non_visible_tails(working_calendar)

            # Trim sequences to their longest contiguous visible
            # block so that no sequence contains mid-observation
            # dark minutes (e.g. negative earth-limb angles where
            # the instrument would look at the Earth).
            working_calendar = self._trim_to_longest_visible_block(
                working_calendar
            )

            # Finally require star-tracker visibility over each
            # observation's opening minutes; without it the spacecraft
            # cannot acquire good pointing. This runs last so it has the
            # final say on every start time.
            working_calendar = self._enforce_st_start_buffer(working_calendar)

        # Report any timing changes (shrink/elongate) made above.
        self._log_timing_changes(working_calendar, original_timing)

        # last thing is to update all the payload parameters
        working_calendar = self._update_payload_parameters(working_calendar)

        return working_calendar

    def _log_timing_changes(
        self, calendar: ScienceCalendar, original_timing: Dict[Any, Any]
    ) -> None:
        """Log per-sequence shrink/elongate vs the snapshot in
        *original_timing* (keyed by ``(visit_id, sequence_id)``)."""
        for visit in calendar.visits:
            for seq in visit.sequences:
                orig = original_timing.get((visit.id, seq.id))
                if orig is None:
                    continue
                old_start, old_stop = orig
                d_start = (seq.start_time - old_start).sec
                d_stop = (seq.stop_time - old_stop).sec
                if abs(d_start) < 1.0 and abs(d_stop) < 1.0:
                    continue

                parts = []
                if abs(d_start) >= 1.0:
                    where = "earlier" if d_start < 0 else "later"
                    parts.append(f"start {where} {abs(d_start) / 60:.1f} min")
                if abs(d_stop) >= 1.0:
                    where = "later" if d_stop > 0 else "earlier"
                    parts.append(f"stop {where} {abs(d_stop) / 60:.1f} min")

                old_dur = (old_stop - old_start).sec / 60.0
                new_dur = seq.duration.sec / 60.0
                verb = "ELONGATED" if new_dur > old_dur else "SHRANK"
                self._print(
                    f"{self._seq_prefix(visit.id, seq)} | {verb}: "
                    + ", ".join(parts)
                    + f" (duration {old_dur:.1f} -> {new_dur:.1f} min)"
                )

    def _fill_gaps(
        self,
        sequence: ObservationSequence,
        gap_length: int,
        visit_id: Optional[str] = None,
    ) -> ObservationSequence:
        """Extend the start of a sequence backward in time to fill a gap.

        This extension is intentionally **blind** (no visibility check).
        Its sole purpose is to maintain schedule contiguity — every
        minute between the first and last sequence must be assigned.
        Non-visible minutes introduced here are cleaned up downstream
        by ``_fix_visibility`` (heads) and ``_trim_non_visible_tails``
        (tails).

        Parameters
        ----------
        sequence : ObservationSequence
            The sequence to adjust.
        gap_length : int
            Gap length in minutes.
        visit_id : str, optional
            Reserved for future use.

        Returns
        -------
        ObservationSequence
            A new ObservationSequence with start time shifted earlier.
        """
        new_start = sequence.start_time - gap_length * u.min
        return ObservationSequence(
            id=sequence.id,
            target=sequence.target,
            priority=sequence.priority,
            start_time=new_start,
            stop_time=sequence.stop_time,
            ra=sequence.ra,
            dec=sequence.dec,
            payload_params=deepcopy(sequence.payload_params),
        )

    def _get_synchronized_time_grid(
        self, calendar: ScienceCalendar
    ) -> Tuple[int, Optional[Time], Optional[Time], Any]:
        """Create a minute-resolution time grid covering all sequences.

        Returns a tuple (total_minutes, start_time, end_time, time_grid)
        where `time_grid` is an array of Astropy Time objects spaced by
        one minute. If the calendar contains no sequences, returns
        (0, None, None, []).
        """
        all_sequences = []
        for visit in calendar.visits:
            for seq in visit.sequences:
                all_sequences.append(seq)

        if not all_sequences:
            return 0, None, None, []

        all_sequences.sort(key=lambda s: s.start_time)
        start_time = all_sequences[0].start_time
        end_time = all_sequences[-1].stop_time

        # Calculate total minutes
        duration = end_time - start_time
        total_minutes = int(np.ceil(duration.sec / 60.0))

        # Create time grid
        time_grid = start_time + np.arange(total_minutes) * u.min

        return total_minutes, start_time, end_time, time_grid

    def _trim_non_visible_heads(
        self, calendar: ScienceCalendar
    ) -> ScienceCalendar:
        """Trim non-visible heads from sequences.

        For each sequence whose first minute(s) are not visible, shrink
        ``start_time`` forward to the first visible minute.  Then
        attempt to extend the *previous* sequence forward to absorb
        the freed time (only where that target is visible).

        This is the complement of ``_trim_non_visible_tails``.
        """
        working_cal = deepcopy(calendar)

        all_sequences: List[Tuple[str, ObservationSequence]] = []
        for visit in working_cal.visits:
            for seq in visit.sequences:
                all_sequences.append((visit.id, seq))
        all_sequences.sort(key=lambda x: x[1].start_time)

        for idx, (visit_id, seq) in self._progress(
            list(enumerate(all_sequences)),
            desc="Trimming non-visible heads",
            total=len(all_sequences),
        ):
            n_mins = int(np.rint(seq.duration.sec / 60.0))
            if n_mins <= 0:
                continue

            target_coord = SkyCoord(seq.ra, seq.dec, frame="icrs", unit="deg")
            deltas = np.arange(n_mins) * u.min
            times = seq.start_time + deltas

            target_roll = self._computed_target_rolls.get(visit_id, {}).get(
                seq.target
            )
            if self._roll_sweep_enabled and target_roll is not None:
                vis = self.visibility.get_visibility(
                    target_coord,
                    times,
                    roll=target_roll * u.deg,
                )
            else:
                vis = self.visibility.get_visibility(target_coord, times)

            vis_arr = np.asarray(vis)

            # Nothing to do if the first minute is already visible
            if len(vis_arr) == 0 or vis_arr[0]:
                continue

            visible_indices = np.where(vis_arr)[0]
            if len(visible_indices) == 0:
                continue  # entirely non-visible — skip

            first_visible_idx = visible_indices[0]

            new_start = seq.start_time + first_visible_idx * u.min
            if (seq.stop_time - new_start) < self.min_sequence_duration:
                continue  # trimming would make sequence too short

            trimmed = ObservationSequence(
                id=seq.id,
                target=seq.target,
                priority=seq.priority,
                start_time=new_start,
                stop_time=seq.stop_time,
                ra=seq.ra,
                dec=seq.dec,
                payload_params=deepcopy(seq.payload_params),
            )
            working_cal.replace_sequence(visit_id, seq.id, trimmed)
            all_sequences[idx] = (visit_id, trimmed)

            # Try extending the previous sequence forward to fill
            # the gap.
            if idx == 0:
                continue

            prev_visit_id, prev_seq = all_sequences[idx - 1]
            gap_minutes = int(
                np.rint((new_start - prev_seq.stop_time).sec / 60.0)
            )
            if gap_minutes <= 0:
                continue

            prev_coord = SkyCoord(
                prev_seq.ra,
                prev_seq.dec,
                frame="icrs",
                unit="deg",
            )
            gap_deltas = np.arange(gap_minutes) * u.min
            gap_times = prev_seq.stop_time + gap_deltas

            prev_roll = self._computed_target_rolls.get(prev_visit_id, {}).get(
                prev_seq.target
            )
            if self._roll_sweep_enabled and prev_roll is not None:
                prev_vis = self.visibility.get_visibility(
                    prev_coord,
                    gap_times,
                    roll=prev_roll * u.deg,
                )
            else:
                prev_vis = self.visibility.get_visibility(
                    prev_coord, gap_times
                )
            prev_vis_arr = np.asarray(prev_vis)

            # Walk forward from prev.stop_time to find the last
            # contiguous visible minute.
            if len(prev_vis_arr) == 0 or not prev_vis_arr[0]:
                # Prev not visible at gap start — extend blindly
                # to maintain contiguity.
                new_prev_stop = new_start
            else:
                last_contiguous = 0
                while (
                    last_contiguous + 1 < len(prev_vis_arr)
                    and prev_vis_arr[last_contiguous + 1]
                ):
                    last_contiguous += 1
                new_prev_stop = gap_times[last_contiguous] + 1 * u.min

            extended_prev = ObservationSequence(
                id=prev_seq.id,
                target=prev_seq.target,
                priority=prev_seq.priority,
                start_time=prev_seq.start_time,
                stop_time=new_prev_stop,
                ra=prev_seq.ra,
                dec=prev_seq.dec,
                payload_params=deepcopy(prev_seq.payload_params),
            )
            working_cal.replace_sequence(
                prev_visit_id, prev_seq.id, extended_prev
            )
            all_sequences[idx - 1] = (prev_visit_id, extended_prev)

        return working_cal

    def _trim_non_visible_tails(
        self, calendar: ScienceCalendar
    ) -> ScienceCalendar:
        """Trim non-visible tails from sequences.

        For each sequence whose last minute(s) are not visible, shrink
        ``stop_time`` to the last visible minute + 1.  Then attempt to
        extend the *next* sequence backward to absorb the freed time
        (only where that target is visible).

        This is the complement of ``_fix_visibility`` which handles
        non-visible *heads* by extending the previous sequence forward
        and shrinking the current sequence's start.
        """
        working_cal = deepcopy(calendar)

        # Collect all sequences globally, sorted by start_time
        all_sequences: List[Tuple[str, ObservationSequence]] = []
        for visit in working_cal.visits:
            for seq in visit.sequences:
                all_sequences.append((visit.id, seq))
        all_sequences.sort(key=lambda x: x[1].start_time)

        for idx, (visit_id, seq) in self._progress(
            list(enumerate(all_sequences)),
            desc="Trimming non-visible tails",
            total=len(all_sequences),
        ):
            n_mins = int(np.rint(seq.duration.sec / 60.0))
            if n_mins <= 0:
                continue

            target_coord = SkyCoord(seq.ra, seq.dec, frame="icrs", unit="deg")
            deltas = np.arange(n_mins) * u.min
            times = seq.start_time + deltas

            target_roll = self._computed_target_rolls.get(visit_id, {}).get(
                seq.target
            )
            if self._roll_sweep_enabled and target_roll is not None:
                vis = self.visibility.get_visibility(
                    target_coord, times, roll=target_roll * u.deg
                )
            else:
                vis = self.visibility.get_visibility(target_coord, times)

            vis_arr = np.asarray(vis)

            # Nothing to do if last minute is visible
            if len(vis_arr) == 0 or vis_arr[-1]:
                continue

            visible_indices = np.where(vis_arr)[0]
            if len(visible_indices) == 0:
                continue  # entirely non-visible — skip

            # Check whether the trailing non-visible run is short
            # enough to tolerate (e.g. a brief earthlimb dip).
            last_visible_idx = visible_indices[-1]
            tail_length = len(vis_arr) - (last_visible_idx + 1)
            if tail_length > 0 and self._is_gap_tolerable(
                target_coord,
                times,
                last_visible_idx + 1,
                tail_length,
                roll=target_roll if self._roll_sweep_enabled else None,
            ):
                continue  # tolerable tail — leave it

            new_stop = seq.start_time + (last_visible_idx + 1) * u.min

            if (new_stop - seq.start_time) < self.min_sequence_duration:
                continue  # trimming would make sequence too short

            # Check whether the next sequence can absorb the freed
            # time.  If not, trimming would create a gap — skip.
            can_absorb = False
            if idx + 1 < len(all_sequences):
                next_visit_id, next_seq = all_sequences[idx + 1]
                gap_minutes = int(
                    np.rint((next_seq.start_time - new_stop).sec / 60.0)
                )
                if gap_minutes > 0:
                    next_coord = SkyCoord(
                        next_seq.ra,
                        next_seq.dec,
                        frame="icrs",
                        unit="deg",
                    )
                    gap_deltas = np.arange(gap_minutes) * u.min
                    gap_times = new_stop + gap_deltas

                    next_roll = self._computed_target_rolls.get(
                        next_visit_id, {}
                    ).get(next_seq.target)
                    if self._roll_sweep_enabled and next_roll is not None:
                        next_vis = self.visibility.get_visibility(
                            next_coord,
                            gap_times,
                            roll=next_roll * u.deg,
                        )
                    else:
                        next_vis = self.visibility.get_visibility(
                            next_coord, gap_times
                        )
                    next_vis_arr = np.asarray(next_vis)

                    # Next can absorb only if the last gap minute
                    # (adjacent to its original start) is visible
                    # and we can walk backward to new_stop.
                    if len(next_vis_arr) > 0 and next_vis_arr[-1]:
                        first_contiguous = len(next_vis_arr) - 1
                        while (
                            first_contiguous > 0
                            and next_vis_arr[first_contiguous - 1]
                        ):
                            first_contiguous -= 1
                        if first_contiguous == 0:
                            can_absorb = True
                else:
                    # No gap between trim point and next → ok
                    can_absorb = True
            else:
                # Last sequence — trimming tail is fine (no gap to
                # worry about).
                can_absorb = True

            if not can_absorb:
                continue

            trimmed = ObservationSequence(
                id=seq.id,
                target=seq.target,
                priority=seq.priority,
                start_time=seq.start_time,
                stop_time=new_stop,
                ra=seq.ra,
                dec=seq.dec,
                payload_params=deepcopy(seq.payload_params),
            )
            working_cal.replace_sequence(visit_id, seq.id, trimmed)

            # Extend the next sequence backward to fill the gap
            if idx + 1 >= len(all_sequences):
                continue

            next_visit_id, next_seq = all_sequences[idx + 1]
            gap_minutes = int(
                np.rint((next_seq.start_time - new_stop).sec / 60.0)
            )
            if gap_minutes <= 0:
                continue

            next_coord = SkyCoord(
                next_seq.ra, next_seq.dec, frame="icrs", unit="deg"
            )
            gap_deltas = np.arange(gap_minutes) * u.min
            gap_times = new_stop + gap_deltas

            next_roll = self._computed_target_rolls.get(next_visit_id, {}).get(
                next_seq.target
            )
            if self._roll_sweep_enabled and next_roll is not None:
                next_vis = self.visibility.get_visibility(
                    next_coord,
                    gap_times,
                    roll=next_roll * u.deg,
                )
            else:
                next_vis = self.visibility.get_visibility(
                    next_coord, gap_times
                )
            next_vis_arr = np.asarray(next_vis)

            # Walk backward from the original next start to find the
            # earliest contiguous visible minute.
            last_idx = len(next_vis_arr) - 1
            if not next_vis_arr[last_idx]:
                continue  # next target also not visible here

            first_contiguous = last_idx
            while first_contiguous > 0 and next_vis_arr[first_contiguous - 1]:
                first_contiguous -= 1

            new_next_start = gap_times[first_contiguous]
            extended_next = ObservationSequence(
                id=next_seq.id,
                target=next_seq.target,
                priority=next_seq.priority,
                start_time=new_next_start,
                stop_time=next_seq.stop_time,
                ra=next_seq.ra,
                dec=next_seq.dec,
                payload_params=deepcopy(next_seq.payload_params),
            )
            working_cal.replace_sequence(
                next_visit_id, next_seq.id, extended_next
            )
            # Update local list so subsequent iterations see
            # the modified next sequence.
            all_sequences[idx + 1] = (next_visit_id, extended_next)

        return working_cal

    def _star_tracker_failed(
        self,
        target_coord: SkyCoord,
        time: Time,
        roll: Optional[float],
    ) -> bool:
        """Whether the star-tracker keepout fails at ``time`` for this roll.

        ``get_all_constraints`` cannot answer this. It takes no ``roll``
        argument, so its ``star_tracker`` verdict is always evaluated at the
        ``Visibility`` instance's own roll rather than the roll the
        observation will actually fly, which is exactly the roll the sweep
        chose to keep the trackers clear. ``get_star_tracker_breakdown``
        does accept a roll, so it is used instead.

        A failure to evaluate the trackers is reported and answered "not a
        tracker failure", which leaves the caller treating the gap as
        intolerable and trimming it away. Guessing the other way would keep
        dark minutes in the schedule on the strength of a verdict we never
        actually got.
        """
        try:
            breakdown = self.visibility.get_star_tracker_breakdown(
                target_coord,
                time,
                roll=None if roll is None else roll * u.deg,
            )
            return not bool(breakdown["passed"]["combined"])
        except Exception as exc:
            self._print(
                f"ERROR: star-tracker check failed at {time.isot} for "
                f"RA/Dec {target_coord.ra.deg:.4f}/"
                f"{target_coord.dec.deg:.4f}, roll {roll}: {exc}"
            )
            return False

    def _is_gap_tolerable(
        self,
        target_coord: SkyCoord,
        times: Any,
        gap_start: int,
        gap_length: int,
        roll: Optional[float] = None,
    ) -> bool:
        """Check whether a non-visible gap is short enough to tolerate.

        Uses ``get_all_constraints`` at the first non-visible minute to
        identify which boresight constraint(s) failed, checks the star
        tracker separately at *roll* (see :meth:`_star_tracker_failed`),
        then compares the gap length against the matching tolerance
        (``earthlimb_gap_tolerance`` or ``st_gap_tolerance``).

        If both tolerances are zero (the default), every gap is
        intolerable and this returns False immediately.
        """
        el_tol = self.earthlimb_gap_tolerance
        st_tol = self.st_gap_tolerance

        if el_tol == 0 and st_tol == 0:
            return False

        try:
            constraints = self.visibility.get_all_constraints(
                target_coord, times[gap_start]
            )
        except Exception:
            return False

        # Drop the star-tracker verdict: it was computed at the wrong roll
        # and is recomputed below. What is left is roll-independent.
        failed = {k for k, v in constraints.items() if not v}
        failed.discard("star_tracker")

        earthlimb_failed = "earthlimb" in failed
        st_failed = self._star_tracker_failed(
            target_coord, times[gap_start], roll
        )

        if failed - {"earthlimb"}:
            # A sun/moon/planet keepout failed — never tolerable.
            return False
        if earthlimb_failed and st_failed:
            return gap_length <= min(el_tol, st_tol)
        if earthlimb_failed:
            return gap_length <= el_tol
        if st_failed:
            return gap_length <= st_tol

        return False

    def _trim_to_longest_visible_block(
        self, calendar: ScienceCalendar
    ) -> ScienceCalendar:
        """Trim sequences to remove intolerable mid-observation dark gaps.

        After ``_fix_visibility`` and ``_trim_non_visible_tails`` have
        handled leading and trailing non-visible minutes, sequences can
        still contain non-visible minutes in the **middle** (e.g. when
        the target dips below the Earth-limb keepout during an orbit).

        Short gaps are tolerated when their duration does not exceed the
        configured tolerances (``earthlimb_gap_tolerance`` and
        ``st_gap_tolerance``).  Gaps exceeding those limits cause the
        sequence to be trimmed to its longest acceptable span — the
        longest contiguous window that contains only tolerable gaps.

        After trimming, the method attempts to extend neighbouring
        sequences to reclaim the freed time (where those neighbours
        are visible).
        """
        working_cal = deepcopy(calendar)

        # Collect all sequences globally, sorted by start_time.
        all_sequences: List[Tuple[str, ObservationSequence]] = []
        for visit in working_cal.visits:
            for seq in visit.sequences:
                all_sequences.append((visit.id, seq))
        all_sequences.sort(key=lambda x: x[1].start_time)

        for idx, (visit_id, seq) in self._progress(
            list(enumerate(all_sequences)),
            desc="Trimming to longest visible block",
            total=len(all_sequences),
        ):
            analysis = self._analyze_mid_sequence_visibility(visit_id, seq)
            if analysis is None:
                continue

            target_coord, times, vis_arr = analysis
            gaps = self._find_nonvisible_gaps(vis_arr)
            if not gaps:
                continue

            seq_roll = (
                self._computed_target_rolls.get(visit_id, {}).get(seq.target)
                if self._roll_sweep_enabled
                else None
            )
            gap_tolerable = [
                self._is_gap_tolerable(
                    target_coord,
                    times,
                    gap_start,
                    gap_end - gap_start,
                    roll=seq_roll,
                )
                for gap_start, gap_end in gaps
            ]
            if all(gap_tolerable):
                continue

            best_window = self._best_tolerable_segment(
                vis_arr,
                gaps,
                gap_tolerable,
            )
            if best_window is None:
                continue

            trimmed = self._build_trimmed_sequence(seq, *best_window)
            if trimmed is None:
                continue

            working_cal.replace_sequence(visit_id, seq.id, trimmed)
            all_sequences[idx] = (visit_id, trimmed)

            self._extend_previous_after_mid_trim(
                working_cal,
                all_sequences,
                idx,
                seq.start_time,
                trimmed.start_time,
            )
            self._extend_next_after_mid_trim(
                working_cal,
                all_sequences,
                idx,
                trimmed.stop_time,
                seq.stop_time,
            )

        return working_cal

    def _enforce_st_start_buffer(
        self, calendar: ScienceCalendar
    ) -> ScienceCalendar:
        """Require star-tracker visibility over each observation's opening.

        ``st_gap_tolerance`` lets a star-tracker dropout be tolerated in
        the middle of an observation, but the spacecraft cannot acquire
        good pointing without the trackers at the start. So the first
        ``st_gap_tolerance_start_buffer`` minutes, measured from the
        observation's start time, not from when science begins after the
        pre-observation overhead -- must be star-tracker visible with no
        tolerance applied.

        Sequences that open with a tracker dropout have their
        ``start_time`` moved forward (in place) to the earliest minute that
        clears the buffer.  When no minute in the sequence clears it, or
        trimming there would leave the sequence shorter than
        ``min_sequence_duration``, the sequence is left alone and the
        problem is written to the error log.

        This runs last among the passes that move a start time, so it has
        the final say.  It only ever moves a start later, so it cannot
        create an overlap.
        """
        buffer_minutes = int(
            getattr(self, "st_gap_tolerance_start_buffer", 0) or 0
        )
        if buffer_minutes <= 0:
            return calendar
        if not getattr(self.visibility, "_st_constraint_active", False):
            return calendar

        for visit in calendar.visits:
            visit_rolls = self._computed_target_rolls.get(visit.id, {})
            for seq in visit.sequences:
                n_mins = int(np.rint(seq.duration.sec / 60.0))
                if n_mins <= 0:
                    continue

                target_coord = SkyCoord(
                    seq.ra, seq.dec, frame="icrs", unit="deg"
                )
                times = seq.start_time + np.arange(n_mins) * u.min
                target_roll = visit_rolls.get(seq.target)
                try:
                    breakdown = (
                        self.visibility.get_star_tracker_breakdown(
                            target_coord,
                            times,
                            roll=(
                                target_roll * u.deg
                                if self._roll_sweep_enabled
                                and target_roll is not None
                                else None
                            ),
                        )
                    )
                except Exception:
                    continue
                st_ok = np.atleast_1d(
                    np.asarray(breakdown["passed"]["combined"])
                )

                # Walk the start forward past every tracker dropout that
                # lands inside the buffer window. Dropouts are visited in
                # order, so once one sits beyond the current window all the
                # later ones do too. An observation shorter than the buffer
                # simply has to be clear all the way to its stop.
                offset = 0
                for dark_minute in np.flatnonzero(~st_ok):
                    if dark_minute >= offset + buffer_minutes:
                        break
                    if dark_minute >= offset:
                        offset = int(dark_minute) + 1
                if offset == 0:
                    continue

                prefix = self._seq_prefix(visit.id, seq)
                if offset >= len(st_ok):
                    self._print(
                        f"ERROR: {prefix} | ST START BUFFER: star trackers "
                        f"are never visible long enough anywhere in this "
                        f"observation; pointing acquisition will be "
                        f"unreliable. Left unchanged."
                    )
                    continue

                new_start = seq.start_time + offset * u.min
                if (seq.stop_time - new_start) < self.min_sequence_duration:
                    self._print(
                        f"ERROR: {prefix} | ST START BUFFER: star trackers "
                        f"do not settle until {offset} min in, and trimming "
                        f"there would leave under "
                        f"{self.min_sequence_duration.sec / 60:.0f} min. "
                        f"Left unchanged."
                    )
                    continue

                self._print(
                    f"{prefix} | ST START BUFFER: start moved later by "
                    f"{offset} min to secure up to {buffer_minutes} min of "
                    f"star-tracker visibility at the start."
                )
                seq.start_time = new_start

        return calendar

    def _analyze_mid_sequence_visibility(
        self,
        visit_id: str,
        seq: ObservationSequence,
    ) -> Optional[Tuple[SkyCoord, Any, np.ndarray]]:
        """Return per-minute visibility for one sequence, if useful."""
        n_mins = int(np.rint(seq.duration.sec / 60.0))
        if n_mins <= 0:
            return None

        target_coord = SkyCoord(seq.ra, seq.dec, frame="icrs", unit="deg")
        deltas = np.arange(n_mins) * u.min
        times = seq.start_time + deltas
        vis_arr = self._visibility_for_sequence(
            visit_id, seq, target_coord, times
        )
        if np.all(vis_arr):
            return None
        return target_coord, times, vis_arr

    def _visibility_for_sequence(
        self,
        visit_id: str,
        seq: ObservationSequence,
        target_coord: SkyCoord,
        times: Any,
    ) -> np.ndarray:
        """Get visibility array for a sequence with roll-aware lookup."""
        target_roll = self._computed_target_rolls.get(visit_id, {}).get(
            seq.target
        )
        if self._roll_sweep_enabled and target_roll is not None:
            vis = self.visibility.get_visibility(
                target_coord,
                times,
                roll=target_roll * u.deg,
            )
        else:
            vis = self.visibility.get_visibility(target_coord, times)
        return np.asarray(vis)

    def _find_nonvisible_gaps(
        self,
        vis_arr: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Find contiguous non-visible runs as half-open index ranges."""
        gaps: List[Tuple[int, int]] = []
        gap_start = None
        for i, visible in enumerate(vis_arr):
            if not visible:
                if gap_start is None:
                    gap_start = i
            elif gap_start is not None:
                gaps.append((gap_start, i))
                gap_start = None

        if gap_start is not None:
            gaps.append((gap_start, len(vis_arr)))
        return gaps

    def _best_tolerable_segment(
        self,
        vis_arr: np.ndarray,
        gaps: List[Tuple[int, int]],
        gap_tolerable: List[bool],
    ) -> Optional[Tuple[int, int]]:
        """Return best [start, end) span separated by intolerable gaps."""
        segment_bounds: List[Tuple[int, int]] = []
        seg_start = 0
        for i, (gap_start, gap_end) in enumerate(gaps):
            if not gap_tolerable[i]:
                if gap_start > seg_start:
                    segment_bounds.append((seg_start, gap_start))
                seg_start = gap_end

        if seg_start < len(vis_arr):
            segment_bounds.append((seg_start, len(vis_arr)))
        if not segment_bounds:
            return None

        best_start, best_end = max(
            segment_bounds,
            key=lambda bounds: bounds[1] - bounds[0],
        )

        while best_start < best_end and not vis_arr[best_start]:
            best_start += 1
        while best_end > best_start and not vis_arr[best_end - 1]:
            best_end -= 1

        if best_end <= best_start:
            return None
        return best_start, best_end

    def _build_trimmed_sequence(
        self,
        seq: ObservationSequence,
        best_start: int,
        best_end: int,
    ) -> Optional[ObservationSequence]:
        """Create trimmed sequence if valid and changed from input."""
        new_start = seq.start_time + best_start * u.min
        new_stop = seq.start_time + best_end * u.min

        if (new_stop - new_start) < self.min_sequence_duration:
            return None
        if new_start == seq.start_time and new_stop == seq.stop_time:
            return None

        return ObservationSequence(
            id=seq.id,
            target=seq.target,
            priority=seq.priority,
            start_time=new_start,
            stop_time=new_stop,
            ra=seq.ra,
            dec=seq.dec,
            payload_params=deepcopy(seq.payload_params),
        )

    def _extend_previous_after_mid_trim(
        self,
        working_cal: ScienceCalendar,
        all_sequences: List[Tuple[str, ObservationSequence]],
        idx: int,
        old_start: Time,
        new_start: Time,
    ) -> None:
        """Extend previous sequence forward into newly freed leading time."""
        if idx <= 0 or new_start <= old_start:
            return

        prev_visit_id, prev_seq = all_sequences[idx - 1]
        freed_mins = int(np.rint((new_start - prev_seq.stop_time).sec / 60.0))
        if freed_mins <= 0:
            return

        prev_coord = SkyCoord(
            prev_seq.ra, prev_seq.dec, frame="icrs", unit="deg"
        )
        gap_deltas = np.arange(freed_mins) * u.min
        gap_times = prev_seq.stop_time + gap_deltas
        prev_vis_arr = self._visibility_for_sequence(
            prev_visit_id,
            prev_seq,
            prev_coord,
            gap_times,
        )

        extend_end = 0
        while extend_end < len(prev_vis_arr) and prev_vis_arr[extend_end]:
            extend_end += 1

        if extend_end <= 0:
            return

        new_prev_stop = prev_seq.stop_time + extend_end * u.min
        extended_prev = ObservationSequence(
            id=prev_seq.id,
            target=prev_seq.target,
            priority=prev_seq.priority,
            start_time=prev_seq.start_time,
            stop_time=new_prev_stop,
            ra=prev_seq.ra,
            dec=prev_seq.dec,
            payload_params=deepcopy(prev_seq.payload_params),
        )
        working_cal.replace_sequence(prev_visit_id, prev_seq.id, extended_prev)
        all_sequences[idx - 1] = (prev_visit_id, extended_prev)

    def _extend_next_after_mid_trim(
        self,
        working_cal: ScienceCalendar,
        all_sequences: List[Tuple[str, ObservationSequence]],
        idx: int,
        new_stop: Time,
        old_stop: Time,
    ) -> None:
        """Extend next sequence backward into newly freed trailing time."""
        if idx + 1 >= len(all_sequences) or new_stop >= old_stop:
            return

        next_visit_id, next_seq = all_sequences[idx + 1]
        freed_mins = int(np.rint((next_seq.start_time - new_stop).sec / 60.0))
        if freed_mins <= 0:
            return

        next_coord = SkyCoord(
            next_seq.ra, next_seq.dec, frame="icrs", unit="deg"
        )
        gap_deltas = np.arange(freed_mins) * u.min
        gap_times = new_stop + gap_deltas
        next_vis_arr = self._visibility_for_sequence(
            next_visit_id,
            next_seq,
            next_coord,
            gap_times,
        )

        last_idx = len(next_vis_arr) - 1
        if last_idx < 0 or not next_vis_arr[last_idx]:
            return

        first_contiguous = last_idx
        while first_contiguous > 0 and next_vis_arr[first_contiguous - 1]:
            first_contiguous -= 1

        new_next_start = gap_times[first_contiguous]
        extended_next = ObservationSequence(
            id=next_seq.id,
            target=next_seq.target,
            priority=next_seq.priority,
            start_time=new_next_start,
            stop_time=next_seq.stop_time,
            ra=next_seq.ra,
            dec=next_seq.dec,
            payload_params=deepcopy(next_seq.payload_params),
        )
        working_cal.replace_sequence(next_visit_id, next_seq.id, extended_next)
        all_sequences[idx + 1] = (next_visit_id, extended_next)

    # ── Force gap-fill helpers ─────────────────────────────────

    # Numeric scores for gap-minute classification (higher = better)
    _GAP_FLOOR = 0  # earthlimb < hard floor — never fill
    _GAP_EL_FAIL = 1  # earthlimb constraint fail (>= floor)
    _GAP_ST_ONLY = 2  # only star-tracker constraints fail
    _GAP_VISIBLE = 3  # fully visible

    def _classify_gap_minute(self, coord: SkyCoord, time: Time) -> int:
        """Score a non-visible gap minute for one target.

        Returns one of ``_GAP_FLOOR``, ``_GAP_EL_FAIL``,
        ``_GAP_ST_ONLY``, or ``_GAP_VISIBLE``.
        """
        try:
            seps = self.visibility.get_separations(coord, time)
            el = seps.get("earthlimb", 90 * u.deg)
            if el.to(u.deg).value < self.earthlimb_hard_floor:
                return self._GAP_FLOOR
        except Exception:
            return self._GAP_FLOOR

        try:
            constraints = self.visibility.get_all_constraints(coord, time)
            failed = {k for k, v in constraints.items() if not v}
            if not failed:
                # Boresight constraints pass — likely a
                # roll-dependent star-tracker failure.
                return self._GAP_ST_ONLY
            el_failed = "earthlimb" in failed
            st_failed = any(
                k.startswith("st") or k == "star_tracker" for k in failed
            )
            if st_failed and not el_failed:
                return self._GAP_ST_ONLY
            return self._GAP_EL_FAIL
        except Exception:
            return self._GAP_EL_FAIL

    def _force_fill_gaps(self, calendar: ScienceCalendar) -> ScienceCalendar:
        """Fill gaps between sequences with constraint-aware rules.

        Rules applied (in priority order):

        1. **Earth-limb hard floor** — never extend into a minute
           where the earth-limb separation is below
           ``earthlimb_hard_floor`` (default 5°).
        2. **Prefer star-tracker violations** — when choosing which
           neighbour to extend into a gap, prefer the direction
           whose only constraint violation is star-tracker rather
           than earth-limb.
        3. **Prefer gaps at end** — extend the *previous* sequence
           forward first (placing any remaining non-visible time at
           its tail) before extending the *next* sequence backward.
        """
        working_cal = deepcopy(calendar)

        # Collect all sequences globally, sorted by start_time
        all_sequences: List[Tuple[str, ObservationSequence]] = []
        for visit in working_cal.visits:
            for seq in visit.sequences:
                all_sequences.append((visit.id, seq))
        all_sequences.sort(key=lambda x: x[1].start_time)

        for idx in self._progress(
            range(len(all_sequences) - 1),
            desc="Force-filling gaps",
            total=len(all_sequences) - 1,
        ):
            prev_vid, prev_seq = all_sequences[idx]
            next_vid, next_seq = all_sequences[idx + 1]

            gap_start = prev_seq.stop_time
            gap_end = next_seq.start_time
            gap_mins = int(np.rint((gap_end - gap_start).sec / 60.0))
            if gap_mins <= 0:
                continue

            prev_coord = SkyCoord(
                prev_seq.ra,
                prev_seq.dec,
                frame="icrs",
                unit="deg",
            )
            next_coord = SkyCoord(
                next_seq.ra,
                next_seq.dec,
                frame="icrs",
                unit="deg",
            )
            gap_deltas = np.arange(gap_mins) * u.min
            gap_times = gap_start + gap_deltas

            # Batch visibility for both targets
            prev_roll = self._computed_target_rolls.get(prev_vid, {}).get(
                prev_seq.target
            )
            next_roll = self._computed_target_rolls.get(next_vid, {}).get(
                next_seq.target
            )

            if self._roll_sweep_enabled and prev_roll is not None:
                prev_vis = np.asarray(
                    self.visibility.get_visibility(
                        prev_coord,
                        gap_times,
                        roll=prev_roll * u.deg,
                    )
                )
            else:
                prev_vis = np.asarray(
                    self.visibility.get_visibility(prev_coord, gap_times)
                )

            if self._roll_sweep_enabled and next_roll is not None:
                next_vis = np.asarray(
                    self.visibility.get_visibility(
                        next_coord,
                        gap_times,
                        roll=next_roll * u.deg,
                    )
                )
            else:
                next_vis = np.asarray(
                    self.visibility.get_visibility(next_coord, gap_times)
                )

            # Classify each gap minute for both targets.
            # Visible minutes get the top score automatically;
            # non-visible minutes are scored via _classify_gap_minute.
            prev_score = np.full(gap_mins, self._GAP_VISIBLE)
            next_score = np.full(gap_mins, self._GAP_VISIBLE)
            for i in range(gap_mins):
                if not prev_vis[i]:
                    prev_score[i] = self._classify_gap_minute(
                        prev_coord, gap_times[i]
                    )
                if not next_vis[i]:
                    next_score[i] = self._classify_gap_minute(
                        next_coord, gap_times[i]
                    )

            # --- Assign minutes ---
            # Walk forward from gap start extending *prev* (rule 3).
            # Stop when prev hits the hard floor or when prev has an
            # earth-limb failure while next is strictly better
            # (rule 2 / rule 1).
            prev_extend = 0
            for i in range(gap_mins):
                ps = prev_score[i]
                ns = next_score[i]
                if ps == self._GAP_FLOOR:
                    break
                if ps == self._GAP_EL_FAIL and ns > ps:
                    break
                prev_extend = i + 1

            # Walk backward from gap end extending *next*.
            next_extend_start = gap_mins
            for i in range(gap_mins - 1, prev_extend - 1, -1):
                if next_score[i] == self._GAP_FLOOR:
                    break
                next_extend_start = i

            # Apply prev-forward extension
            if prev_extend > 0:
                new_stop = gap_start + prev_extend * u.min
                extended_prev = ObservationSequence(
                    id=prev_seq.id,
                    target=prev_seq.target,
                    priority=prev_seq.priority,
                    start_time=prev_seq.start_time,
                    stop_time=new_stop,
                    ra=prev_seq.ra,
                    dec=prev_seq.dec,
                    payload_params=deepcopy(prev_seq.payload_params),
                )
                working_cal.replace_sequence(
                    prev_vid, prev_seq.id, extended_prev
                )
                all_sequences[idx] = (prev_vid, extended_prev)

            # Apply next-backward extension
            if next_extend_start < gap_mins:
                new_start = gap_start + next_extend_start * u.min
                extended_next = ObservationSequence(
                    id=next_seq.id,
                    target=next_seq.target,
                    priority=next_seq.priority,
                    start_time=new_start,
                    stop_time=next_seq.stop_time,
                    ra=next_seq.ra,
                    dec=next_seq.dec,
                    payload_params=deepcopy(next_seq.payload_params),
                )
                working_cal.replace_sequence(
                    next_vid, next_seq.id, extended_next
                )
                all_sequences[idx + 1] = (next_vid, extended_next)

        return working_cal

    def _fix_visibility(
        self, calendar: ScienceCalendar, all_minutes_bool: Any
    ) -> ScienceCalendar:
        """
        Fix visibility gaps by extending previous sequences and shrinking current sequences.
        """
        working_cal = deepcopy(calendar)

        # Get synchronized time grid
        total_minutes, start_time, end_time, time_grid = (
            self._get_synchronized_time_grid(working_cal)
        )
        assignments_result = self.get_minute_by_minute_assignments(working_cal)
        assignments = assignments_result["assignments"]

        # Find visibility gaps
        false_blocks, false_idx = _find_false_blocks(
            all_minutes_bool, time_grid, return_index=True
        )

        # Track gaps for reporting
        visibility_gaps = []
        gaps_filled = 0
        gaps_total = len(false_idx)

        if not false_idx:
            # No visibility gaps found
            return working_cal

        # Define helper functions for assignment access
        def get_previous(j, target):
            while j > 0 and assignments[j]["target"] == target:
                j -= 1
            return j if j >= 0 else None

        def get_ra_dec(idx):
            return assignments[idx]["ra"], assignments[idx]["dec"]

        # Process each visibility gap
        for gap_start_idx, gap_end_idx in self._progress(
            false_idx, desc="Fixing visibility gaps", total=gaps_total
        ):
            # Get times for this gap
            gap_times = []
            for x in range(0, gap_end_idx - gap_start_idx):
                if gap_start_idx + x < len(assignments):
                    gap_times.append(assignments[gap_start_idx + x]["time"])

            if not gap_times:
                continue

            # Check if this gap is short enough to tolerate for the
            # current sequence's target (the one whose head is
            # non-visible).  If so, skip — no extend/shrink needed.
            gap_len = len(gap_times)
            if gap_start_idx < len(assignments):
                cur_ra = assignments[gap_start_idx]["ra"]
                cur_dec = assignments[gap_start_idx]["dec"]
                if cur_ra is not None and cur_dec is not None:
                    cur_coord = SkyCoord(
                        cur_ra, cur_dec, frame="icrs", unit="deg"
                    )
                    cur_roll = (
                        self._computed_target_rolls.get(
                            assignments[gap_start_idx]["visit_id"], {}
                        ).get(assignments[gap_start_idx]["target"])
                        if self._roll_sweep_enabled
                        else None
                    )
                    if self._is_gap_tolerable(
                        cur_coord,
                        Time(gap_times),
                        0,
                        gap_len,
                        roll=cur_roll,
                    ):
                        continue

            # Get previous sequence's target coordinates
            prev_idx = get_previous(
                gap_start_idx, assignments[gap_start_idx]["target"]
            )
            if prev_idx is None:
                continue

            ra, dec = get_ra_dec(prev_idx)
            if ra is None or dec is None:
                continue
            target_coord = SkyCoord(ra, dec, frame="icrs", unit="deg")

            # Look up precomputed roll for the previous target
            prev_target = assignments[prev_idx]["target"]
            prev_visit_id = assignments[prev_idx]["visit_id"]
            prev_roll = self._computed_target_rolls.get(prev_visit_id, {}).get(
                prev_target
            )

            # Check visibility of previous target during gap
            if self._roll_sweep_enabled and prev_roll is not None:
                vis = self.visibility.get_visibility(
                    target_coord,
                    Time(gap_times),
                    roll=prev_roll * u.deg,
                )
            else:
                vis = self.visibility.get_visibility(
                    target_coord, Time(gap_times)
                )

            # ── Pre-check: is shrinking the following sequence feasible? ──
            # We must decide this BEFORE extending the previous sequence so
            # that an extend followed by a failed shrink cannot create an
            # overlap between the two adjacent sequences.
            shrink_feasible = False
            seq_to_shrink = None
            current_visit_id = None
            current_sequence_id = None
            gap_end_time = None

            if gap_start_idx < len(assignments):
                cur_asgn = assignments[gap_start_idx]
                current_visit_id = cur_asgn["visit_id"]
                current_sequence_id = cur_asgn["sequence_id"]
                seq_to_shrink = working_cal.get_sequence(
                    current_visit_id, current_sequence_id
                )
                if seq_to_shrink:
                    gap_end_time = Time(gap_times[-1]) + 1 * u.minute
                    remaining_duration = seq_to_shrink.stop_time - gap_end_time
                    shrink_feasible = (
                        remaining_duration >= self.min_sequence_duration
                    )

            # Extend previous sequence only when the following sequence can
            # also be shrunk — keeping extend and shrink atomic.
            did_extend = False
            extend_stop_time = None
            if np.any(vis) and shrink_feasible:
                visible_times = np.array(gap_times)[vis]
                if len(visible_times) > 0:
                    last_visible_time = visible_times[-1]
                    prev_assignment = assignments[prev_idx]
                    visit_id = prev_assignment["visit_id"]
                    sequence_id = prev_assignment["sequence_id"]

                    seq_to_extend = working_cal.get_sequence(
                        visit_id, sequence_id
                    )
                    if seq_to_extend:
                        new_stop_time = Time(last_visible_time) + 1 * u.minute
                        extended_seq = ObservationSequence(
                            id=seq_to_extend.id,
                            target=seq_to_extend.target,
                            priority=seq_to_extend.priority,
                            start_time=seq_to_extend.start_time,
                            stop_time=new_stop_time,
                            ra=seq_to_extend.ra,
                            dec=seq_to_extend.dec,
                            payload_params=deepcopy(
                                seq_to_extend.payload_params
                            ),
                        )
                        working_cal.replace_sequence(
                            visit_id, sequence_id, extended_seq
                        )
                        gaps_filled += 1
                        did_extend = True
                        extend_stop_time = new_stop_time

            # Track remaining visibility gaps
            if np.any(~vis):
                non_visible_times = np.array(gap_times)[~vis]
                if len(non_visible_times) > 0:
                    first_false_time = non_visible_times[0]
                    last_false_time = non_visible_times[-1]
                    visibility_gaps.append(
                        (
                            Time(first_false_time),
                            Time(last_false_time) + 1 * u.minute,
                        )
                    )

            # Shrink the current sequence (only when extend succeeded).
            # Use extend_stop_time so B starts exactly where A ended —
            # no gap.
            if did_extend and shrink_feasible and seq_to_shrink is not None:
                shrunk_seq = ObservationSequence(
                    id=seq_to_shrink.id,
                    target=seq_to_shrink.target,
                    priority=seq_to_shrink.priority,
                    start_time=extend_stop_time,
                    stop_time=seq_to_shrink.stop_time,
                    ra=seq_to_shrink.ra,
                    dec=seq_to_shrink.dec,
                    payload_params=deepcopy(seq_to_shrink.payload_params),
                )
                working_cal.replace_sequence(
                    current_visit_id, current_sequence_id, shrunk_seq
                )

        # Update gap report
        self.gap_report["visibility_gaps"] = visibility_gaps
        self.gap_report["processing_summary"].update(
            {
                "gaps_processed": gaps_total,
                "gaps_filled": gaps_filled,
                "gaps_remaining": len(visibility_gaps),
            }
        )

        return working_cal

    def get_minute_by_minute_assignments(
        self, calendar: ScienceCalendar
    ) -> Dict[str, Any]:
        """Generate assignments using synchronized time grid."""
        # Use synchronized time grid
        total_minutes, start_time, end_time, time_grid = (
            self._get_synchronized_time_grid(calendar)
        )

        if total_minutes == 0:
            return {"times": [], "assignments": [], "summary": {}}

        # Time tolerance for comparisons (1 second)
        tol = 1.0  # seconds

        # Build the minute grid once (vectorised). Both materialising scalar
        # Time objects (``list(times)``) and per-scalar ``isot`` formatting
        # are very slow, so keep ``times`` as a single Time array and format
        # all ISOT strings in one vectorised call.
        times = start_time + np.arange(total_minutes) * u.min
        isot_values = np.atleast_1d(times.isot)

        # Pre-compute each sequence's [start, stop) window in seconds relative
        # to ``start_time``, sorted by start. Doing the (slow) astropy Time
        # subtraction once per sequence — rather than once per (minute,
        # sequence) — and then walking a forward pointer keeps this O(minutes
        # + sequences) instead of O(minutes * sequences). The latter is why
        # the per-minute scan slowed down as it advanced through the window:
        # each later minute had to skip every already-finished sequence
        # before reaching its owner.
        intervals = []
        for visit in calendar.visits:
            for seq in visit.sequences:
                s0 = (seq.start_time - start_time).to(u.s).value
                s1 = (seq.stop_time - start_time).to(u.s).value
                intervals.append((s0, s1, seq, visit.id))
        intervals.sort(key=lambda iv: iv[0])
        n_intervals = len(intervals)

        assignments = []
        lo = 0  # index of the earliest sequence that may still own a minute

        for minute_idx in self._progress(
            range(total_minutes),
            desc="Mapping minute assignments",
            total=total_minutes,
        ):
            current = float(minute_idx) * 60.0

            # Retire sequences whose window has ended: once a minute is at or
            # past stop - tol, that sequence (and, by sort order, none before
            # it) can own this or any later minute.
            while lo < n_intervals and current >= intervals[lo][1] - tol:
                lo += 1

            assignment = {
                "time": isot_values[minute_idx],
                "minute_index": minute_idx,
                "sequence_id": None,
                "target": None,
                "visit_id": None,
                "ra": None,
                "dec": None,
                "priority": None,
                "status": "unassigned",
            }

            if lo < n_intervals:
                s0, s1, seq, visit_id = intervals[lo]
                starts_at_or_after = current >= s0 - tol
                ends_before = current < s1 - tol
                starts_exactly = abs(current - s0) <= tol
                if (starts_at_or_after and ends_before) or starts_exactly:
                    assignment.update(
                        {
                            "sequence_id": seq.id,
                            "target": seq.target,
                            "visit_id": visit_id,
                            "ra": seq.ra,
                            "dec": seq.dec,
                            "priority": seq.priority,
                            "status": "assigned",
                        }
                    )

            assignments.append(assignment)

        return {"times": times, "assignments": assignments}

    def _update_payload_parameters(
        self, calendar: ScienceCalendar
    ) -> ScienceCalendar:
        """Adjust payload parameters based on observation duration."""
        for visit in calendar.visits:
            visit_id = visit.id
            for seq in visit.sequences:
                sequence_id = seq.id
                new_sequence = self._update_payload_parameters_sequence(
                    seq, visit_id=visit_id
                )
                calendar.replace_sequence(visit_id, sequence_id, new_sequence)

        return calendar

    def _build_payload_data(
        self,
        sequence: ObservationSequence,
        override_fields: Any,
        data_cls: Any,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Build a NirdaData/VisdaData object from a sequence's payload.

        The payload section, field<->XML mapping, and required fields are
        taken from the data class itself (``PAYLOAD_SECTION``,
        ``CONFIG_SPEC``, ``REQUIRED_CONFIG_FIELDS``). For each config field
        the value is read from the observation's payload XML and converted
        to the data-class field. Fields in *override_fields* are instead
        forced and queued to be written back to the observation so the
        calendar reflects the override.

        *override_fields* may be either a mapping ``{field_name: value}``
        (a non-``None`` value is used directly; ``None`` means "use the
        ``data_cls`` default") or an iterable of field names (treated as
        "use the default" for each).

        Returns
        -------
        (data_obj, writeback) on success, where ``writeback`` maps XML tags
        to the string values that should be written back to the payload for
        overridden fields.  Returns ``(None, missing_tags)`` if any required
        field is absent (e.g. a sequence with no payload for this section).
        """
        section = data_cls.PAYLOAD_SECTION
        spec = data_cls.CONFIG_SPEC
        required_fields = data_cls.REQUIRED_CONFIG_FIELDS

        # Normalize override_fields to a {field: value-or-None} mapping. A
        # dict supplies explicit values (None -> class default); an iterable
        # of names means "use the default" for each.
        if isinstance(override_fields, dict):
            overrides = dict(override_fields)
        else:
            overrides = {field: None for field in (override_fields or ())}

        default_config = data_cls().get_config()
        kwargs: Dict[str, Any] = dict(extra_kwargs or {})
        # Share the run logger so the data class's own warnings (zero frame
        # time, oversize, VITL fallback, ...) land in the same log.
        kwargs.setdefault("logger", getattr(self, "logger", None))
        writeback: Dict[str, str] = {}
        missing: List[str] = []

        for field, (tag, from_xml, to_xml) in spec.items():
            if field in overrides:
                # None -> class default; otherwise use the supplied value.
                ov = overrides[field]
                value = default_config[field] if ov is None else ov
                kwargs[field] = value
                writeback[tag] = to_xml(value)
                continue

            raw = sequence.get_payload_parameter(section, tag)
            if raw is None or raw == "":
                if field in required_fields:
                    missing.append(tag)
                continue
            try:
                kwargs[field] = from_xml(raw)
            except (ValueError, TypeError):
                missing.append(tag)

        if missing:
            return None, missing
        return data_cls(**kwargs), writeback

    def _warn_if_data_exceeds_limits(
        self,
        sequence: ObservationSequence,
        detector: str,
        data: u.Quantity,
        data_compressed: u.Quantity,
        visit_id: Any = None,
    ) -> None:
        """Warn if a sequence's computed data volume exceeds the limits.

        Compares the *uncompressed* ``data`` against
        ``max_file_size_uncompressed`` and the *compressed*
        ``data_compressed`` against ``max_file_size_compressed``. Each
        breach is emitted both as a ``UserWarning`` (for programmatic
        consumers) and through the run logger so it lands in the console and
        the ``.errors.log``.
        """
        max_uncompressed = getattr(self, "max_file_size_uncompressed", None)
        max_compressed = getattr(self, "max_file_size_compressed", None)

        prefix = self._seq_prefix(visit_id, sequence)

        if (
            max_uncompressed is not None
            and data.to(u.byte).value > max_uncompressed.to(u.byte).value
        ):
            msg = (
                f"{prefix} | {detector} uncompressed data "
                f"{data.to(u.byte).value / 1e6:.1f} MB exceeds limit "
                f"{max_uncompressed.to(u.byte).value / 1e6:.1f} MB"
            )
            self._print(f"Warning: {msg}")
            warnings.warn(msg, stacklevel=2)

        if (
            max_compressed is not None
            and data_compressed.to(u.byte).value
            > max_compressed.to(u.byte).value
        ):
            msg = (
                f"{prefix} | {detector} compressed data "
                f"{data_compressed.to(u.byte).value / 1e6:.1f} MB exceeds "
                f"limit {max_compressed.to(u.byte).value / 1e6:.1f} MB"
            )
            self._print(f"Warning: {msg}")
            warnings.warn(msg, stacklevel=2)

    @staticmethod
    def _normalize_priority_keys(raw: Optional[Dict[Any, Any]]) -> Dict[int, Any]:
        """Coerce override priority keys to ints (accepts 'Priority_0', '0')."""
        out: Dict[int, Any] = {}
        for key, value in (raw or {}).items():
            if isinstance(key, bool):
                # bool is an int subclass; reject to avoid surprises.
                raise ValueError(f"Invalid priority key: {key!r}")
            if isinstance(key, int):
                priority = key
            elif isinstance(key, str):
                token = key.strip()
                if token.lower().startswith("priority_"):
                    token = token.split("_", 1)[1]
                priority = int(token)
            else:
                raise ValueError(f"Invalid priority key: {key!r}")
            out[priority] = value
        return out

    @staticmethod
    def _format_payload_value(value: Any) -> str:
        """Format a Python value as payload XML text (cleaner-compatible)."""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            if value.is_integer():
                return str(int(value))
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value)

    def _set_override_element(
        self,
        parent: "ET.Element",
        mapping: Dict[str, Any],
        prefix: str,
        path: str,
    ) -> None:
        """Recursively force *mapping* onto *parent*, creating missing tags.

        Scalar values are written as element text; nested dicts create (or
        descend into) child elements, supporting structures like
        ``{'Boresight': {'PRI_CMD_DIR': 9}}``.
        """
        for tag, value in mapping.items():
            child = parent.find(tag)
            if child is None:
                child = ET.SubElement(parent, tag)
            if isinstance(value, dict):
                self._set_override_element(
                    child, value, prefix, f"{path}/{tag}"
                )
            else:
                old = child.text
                child.text = self._format_payload_value(value)
                # Only report when the value actually changed.
                old_norm = old.strip() if old is not None else None
                if old_norm != child.text:
                    self._print(
                        f"{prefix} | PAYLOAD OVERRIDE: {path}/{tag} "
                        f"'{old}' -> '{child.text}'"
                    )

    def _apply_payload_overrides(
        self, sequence: ObservationSequence, visit_id: Any = None
    ) -> None:
        """Force per-priority XML overrides onto a sequence.

        Writes ``override_payload_parameters[priority][section][...]`` onto
        the observation, creating any missing tag (or section). Values may be
        nested dicts (e.g. ``Observational_Parameters -> Boresight ->
        PRI_CMD_DIR``). The payload detector sections
        (``AcquireInfCamImages`` / ``AcquireVisCamScienceData``) and an
        ``Observational_Parameters`` override are all stored on
        ``payload_params``; the writer merges the latter into the
        Observational_Parameters block it builds. Free-time observations are
        skipped. Runs before the integration recompute so size/coadd/reset
        changes take effect.
        """
        overrides = getattr(self, "_override_payload_parameters", {}) or {}
        if not overrides:
            return
        entry = overrides.get(sequence.priority)
        if not entry:
            return
        if (sequence.target or "").strip().lower() in (
            "free time",
            "freetime",
            "free_time",
            "free-time",
        ):
            return

        prefix = self._seq_prefix(visit_id, sequence)
        for section, mapping in entry.items():
            section_elem = sequence.payload_params.get(section)
            if section_elem is None:
                section_elem = ET.Element(section)
                sequence.payload_params[section] = section_elem
            self._set_override_element(section_elem, mapping, prefix, section)

    def _update_payload_parameters_sequence(
        self, sequence: ObservationSequence, visit_id: Any = None
    ) -> ObservationSequence:
        # Pass sequence.duration (TimeDelta) so both helpers receive the
        # correct type and the overhead subtraction uses a consistent unit.
        duration = sequence.duration

        # General XML-tag payload overrides first, so subsequent integration
        # recomputation sees the forced ROI/coadd/reset values.
        self._apply_payload_overrides(sequence, visit_id=visit_id)

        # Per-priority parameter overrides (see process_calendar). Falls back
        # to no overrides when the attributes are unset (e.g. when a bare
        # ScheduleProcessor is constructed in tests).
        nirda_overrides = getattr(self, "_override_nirda_parameters", {}) or {}
        visda_overrides = getattr(self, "_override_visda_parameters", {}) or {}
        nirda_fields = nirda_overrides.get(sequence.priority, ())
        visda_fields = visda_overrides.get(sequence.priority, ())

        overhead = getattr(self, "overhead", None)

        sequence = self._update_VDA_integrations(
            sequence,
            duration,
            overhead=overhead,
            override_fields=visda_fields,
            visit_id=visit_id,
        )
        sequence = self._update_NIRDA_integrations(
            sequence,
            duration,
            overhead=overhead,
            override_fields=nirda_fields,
            visit_id=visit_id,
        )

        # Convert single-ROI auto-detect observations to the predefined-ROI
        # method. Runs after the overrides above so a forced MaxNumStarRois of
        # 1 is taken into account; the conversion does not change timing or
        # data volume, so its position relative to the integration recompute
        # is immaterial.
        if getattr(self, "convert_single_roi_to_predefined", False):
            self._convert_single_roi_to_predefined(sequence, visit_id=visit_id)

        # Fix bad data (invalid name symbols + NaN-like value reporting).
        if getattr(self, "fix_bad_data", False):
            self._fix_bad_data(sequence, visit_id=visit_id)

        return sequence

    @staticmethod
    def _clean_name(name: str) -> str:
        """Replace every :data:`BAD_NAME_SYMBOLS` character in *name*."""
        for bad, good in BAD_NAME_SYMBOLS.items():
            name = name.replace(bad, good)
        return name

    def _normalize_sequence_names(
        self, sequence: ObservationSequence, visit_id: Any = None
    ) -> bool:
        """Replace invalid symbols in a sequence's target name fields.

        Substitutes every :data:`BAD_NAME_SYMBOLS` character (e.g. ``+`` and
        space -> ``_``) in the sequence's ``target`` attribute and any
        ``Target``/``TargetID`` payload tags. Returns ``True`` if anything
        changed. Idempotent: re-running on an already-clean sequence is a
        no-op. This runs up front (before the roll sweep) so the swept rolls,
        which are keyed by target name, are not orphaned by a later rename.
        """
        prefix = self._seq_prefix(visit_id, sequence)
        changed = False

        # The sequence's target name attribute.
        if sequence.target:
            fixed = self._clean_name(sequence.target)
            if fixed != sequence.target:
                self._print(
                    f"{prefix} | BAD DATA: Target "
                    f"'{sequence.target}' -> '{fixed}'"
                )
                sequence.target = fixed
                changed = True

        # Any Target/TargetID payload tags.
        for section_elem in sequence.payload_params.values():
            if not isinstance(section_elem, ET.Element):
                continue
            for elem in section_elem.iter():
                tag = elem.tag.rsplit("}", 1)[-1]
                if tag not in ("Target", "TargetID") or not elem.text:
                    continue
                fixed = self._clean_name(elem.text)
                if fixed != elem.text:
                    self._print(
                        f"{prefix} | BAD DATA: {tag} "
                        f"'{elem.text}' -> '{fixed}'"
                    )
                    elem.text = fixed
                    changed = True
        return changed

    def _normalize_target_names(
        self, calendar: ScienceCalendar, verbose: bool = False
    ) -> None:
        """Normalize target name fields across the whole calendar up front.

        Runs immediately after windowing -- before the roll sweep -- so the
        target names the roll sweep keys on match the names present when the
        precomputed rolls are applied. Without this, a later ``+``/space ->
        ``_`` rename would orphan a target's swept roll, dropping it back to
        the sun-derived fallback.
        """
        n_changed = 0
        for visit in calendar.visits:
            for seq in visit.sequences:
                if self._normalize_sequence_names(seq, visit_id=visit.id):
                    n_changed += 1
        if verbose and n_changed:
            self._print(
                f"Normalized invalid symbols in {n_changed} target name(s)."
            )

    def _fix_bad_data(
        self, sequence: ObservationSequence, visit_id: Any = None
    ) -> None:
        """Replace invalid name symbols and report NaN-like field values.

        Mirrors the CalendarCleaner ``Fix_Bad_Data`` step:

        - ``Target``/``TargetID`` fields (the sequence's ``target`` attribute
          and any ``Target``/``TargetID`` payload tags) have each symbol in
          ``BAD_NAME_SYMBOLS`` replaced by its safe substitute (see
          :meth:`_normalize_sequence_names`; normally already applied up front
          by :meth:`_normalize_target_names`, so this is a safety net).
        - Every other field is scanned for NaN-like text; matches in tags not
          listed in ``NON_NUMERIC_TAGS`` are logged as warnings. Free-time
          observations are skipped here because their RA/Dec are expected to
          be NaN.
        """
        prefix = self._seq_prefix(visit_id, sequence)

        # 1+2) Replace invalid symbols in the target name fields.
        self._normalize_sequence_names(sequence, visit_id=visit_id)

        # 3) Scan numeric fields for NaN-like values (report only). Free-time
        # observations legitimately carry NaN RA/Dec, so skip them.
        if (sequence.target or "").strip().lower() in (
            "free time",
            "freetime",
            "free_time",
            "free-time",
        ):
            return
        for section, section_elem in sequence.payload_params.items():
            if not isinstance(section_elem, ET.Element):
                continue
            for elem in section_elem.iter():
                tag = elem.tag.rsplit("}", 1)[-1]
                if tag in NON_NUMERIC_TAGS or not elem.text:
                    continue
                if elem.text.strip().lower() == "nan":
                    self._print(
                        f"WARNING: {prefix} | BAD DATA: {section}/{tag} "
                        f"has NaN-like value '{elem.text.strip()}'"
                    )

    def _convert_single_roi_to_predefined(
        self, sequence: ObservationSequence, visit_id: Any = None
    ) -> bool:
        """Convert a single-ROI auto-detect VIS section to predefined-ROI.

        Mirrors the CalendarCleaner ``Fix_Single_ROI_Det`` step: when the
        ``AcquireVisCamScienceData`` section requests exactly one star ROI via
        the brightest-star auto-detect method (``MaxNumStarRois == 1`` and
        ``StarRoiDetMethod == 2``), switch it to the predefined-ROI method
        (``StarRoiDetMethod == 1``) and supply the target RA/Dec as the single
        predefined ROI.

        The target RA/Dec is resolved verbatim, preferring the VIS section's
        ``TargetRA``/``TargetDEC``, then the sequence's ``ra``/``dec``. The
        conversion is idempotent: a section already carrying ``RA1``/``Dec1``
        predefined children is left untouched. Returns True if a conversion
        was made.
        """
        if (sequence.target or "").strip().lower() in (
            "free time",
            "freetime",
            "free_time",
            "free-time",
        ):
            return False

        vis_section = sequence.payload_params.get("AcquireVisCamScienceData")
        if vis_section is None:
            return False

        def _to_int(elem):
            if elem is None or elem.text is None:
                return None
            try:
                return int(float(elem.text))
            except (ValueError, TypeError):
                return None

        max_rois = _to_int(vis_section.find("MaxNumStarRois"))
        det_method = _to_int(vis_section.find("StarRoiDetMethod"))
        if max_rois != 1 or det_method != 2:
            return False

        # Idempotency: only skip when an actual predefined ROI (RA1/Dec1) is
        # already present, not a bare placeholder parent.
        ra_parent = vis_section.find("PredefinedStarRoiRa")
        dec_parent = vis_section.find("PredefinedStarRoiDec")
        has_ra1 = ra_parent is not None and ra_parent.find("RA1") is not None
        has_dec1 = dec_parent is not None and dec_parent.find("Dec1") is not None
        if has_ra1 and has_dec1:
            return False

        # Resolve the target RA/Dec verbatim: prefer the VIS-section values,
        # then fall back to the sequence's own coordinates.
        def _usable(value):
            return (
                value is not None
                and str(value).strip() != ""
                and str(value).strip().lower() != "nan"
            )

        ra_elem = vis_section.find("TargetRA")
        dec_elem = vis_section.find("TargetDEC")
        ra = ra_elem.text if ra_elem is not None else None
        dec = dec_elem.text if dec_elem is not None else None
        if not _usable(ra) and sequence.ra is not None:
            ra = self._format_payload_value(sequence.ra)
        if not _usable(dec) and sequence.dec is not None:
            dec = self._format_payload_value(sequence.dec)

        prefix = self._seq_prefix(visit_id, sequence)
        if not _usable(ra) or not _usable(dec):
            self._print(
                f"WARNING: {prefix} | SINGLE-ROI: no usable target RA/Dec "
                f"(RA={ra!r}, Dec={dec!r}); left unchanged."
            )
            return False

        ra = str(ra).strip()
        dec = str(dec).strip()

        # Switch to predefined-ROI method with a single ROI.
        det_elem = vis_section.find("StarRoiDetMethod")
        det_elem.text = "1"

        num_elem = vis_section.find("numPredefinedStarRois")
        if num_elem is None:
            num_elem = ET.SubElement(vis_section, "numPredefinedStarRois")
        num_elem.text = "1"

        if ra_parent is None:
            ra_parent = ET.SubElement(vis_section, "PredefinedStarRoiRa")
        for stale in list(ra_parent):
            ra_parent.remove(stale)
        ET.SubElement(ra_parent, "RA1").text = ra

        if dec_parent is None:
            dec_parent = ET.SubElement(vis_section, "PredefinedStarRoiDec")
        for stale in list(dec_parent):
            dec_parent.remove(stale)
        ET.SubElement(dec_parent, "Dec1").text = dec

        self._print(
            f"{prefix} | SINGLE-ROI: StarRoiDetMethod 2 -> 1, "
            f"numPredefinedStarRois=1, RA1={ra}, Dec1={dec}"
        )
        return True

    def _update_VDA_integrations(
        self,
        sequence: ObservationSequence,
        duration: TimeDelta,
        overhead: Optional[OverheadTiming] = None,
        override_fields: Any = (),
        visit_id: Any = None,
    ) -> ObservationSequence:
        """Set NumTotalFramesRequested using a ``VisdaData`` model.

        The VISDA detector configuration is built from the sequence's
        ``AcquireVisCamScienceData`` payload (or, for any field listed in
        *override_fields*, from the ``VisdaData`` defaults), and the frame
        count that fits the sequence duration -- net of the pre/post
        overheads in *overhead* -- is computed by
        ``VisdaData.solve_integrations``.

        Parameters
        ----------
        overhead : OverheadTiming, optional
            Overhead timings to apply. Defaults to ``self.overhead`` (built
            once at construction); a bare ``OverheadTiming`` is used only if
            the processor has none.
        """
        if overhead is None:
            overhead = getattr(self, "overhead", None) or OverheadTiming()

        # Detector read time is not represented in the payload; the existing
        # scheduling math treats a frame as taking exactly the exposure time.
        visda, info = self._build_payload_data(
            sequence,
            override_fields,
            VisdaData,
            extra_kwargs={"read_time_per_frame_s": 0 * u.s},
        )
        prefix = self._seq_prefix(visit_id, sequence)
        if visda is None:
            self._print(
                f"Warning: {prefix} | Missing VDA parameters: "
                f"{', '.join(info)}"
            )
            return sequence

        # Write any overridden parameters back onto the observation, logging
        # each forced change.
        for tag, text in info.items():
            old = sequence.get_payload_parameter(
                "AcquireVisCamScienceData", tag
            )
            sequence.set_payload_parameter(
                "AcquireVisCamScienceData", tag, text
            )
            # Only report when the value actually changed.
            if (old if old is None else str(old)) != text:
                self._print(
                    f"{prefix} | VISDA OVERRIDE: {tag} '{old}' -> '{text}'"
                )

        old_frames = sequence.get_payload_parameter(
            "AcquireVisCamScienceData", "NumTotalFramesRequested"
        )
        frames, data, data_compressed = visda.solve_integrations(
            duration.to(u.s), overhead
        )
        self._warn_if_data_exceeds_limits(
            sequence, "VISDA", data, data_compressed, visit_id=visit_id
        )

        success = sequence.set_payload_parameter(
            "AcquireVisCamScienceData",
            "NumTotalFramesRequested",
            str(int(frames)),
        )
        if success:
            self._print(
                f"{prefix} | VISDA NumTotalFramesRequested "
                f"'{old_frames}' -> '{int(frames)}'"
            )
        else:
            self._print(
                f"Warning: {prefix} | Failed to update "
                f"NumTotalFramesRequested"
            )
        return sequence

    def _update_NIRDA_integrations(
        self,
        sequence: ObservationSequence,
        duration: TimeDelta,
        overhead: Optional[OverheadTiming] = None,
        override_fields: Any = (),
        visit_id: Any = None,
    ) -> ObservationSequence:
        """Set SC_Integrations using a ``NirdaData`` model.

        The NIRDA detector configuration is built from the sequence's
        ``AcquireInfCamImages`` payload (or, for any field listed in
        *override_fields*, from the ``NirdaData`` defaults), and the number
        of integrations that fit the sequence duration -- net of the
        pre/post overheads in *overhead* -- is computed by
        ``NirdaData.solve_integrations``.

        Parameters
        ----------
        overhead : OverheadTiming, optional
            Overhead timings to apply. Defaults to ``self.overhead`` (built
            once at construction); a bare ``OverheadTiming`` is used only if
            the processor has none.
        """
        if overhead is None:
            overhead = getattr(self, "overhead", None) or OverheadTiming()

        prefix = self._seq_prefix(visit_id, sequence)

        nirda, info = self._build_payload_data(
            sequence,
            override_fields,
            NirdaData,
        )
        if nirda is None:
            self._print(
                f"Warning: {prefix} | Missing NIRDA parameters: "
                f"{', '.join(info)}"
            )
            return sequence

        # Write any overridden parameters back onto the observation, logging
        # each forced change.
        for tag, text in info.items():
            old = sequence.get_payload_parameter("AcquireInfCamImages", tag)
            sequence.set_payload_parameter("AcquireInfCamImages", tag, text)
            # Only report when the value actually changed.
            if (old if old is None else str(old)) != text:
                self._print(
                    f"{prefix} | NIRDA OVERRIDE: {tag} '{old}' -> '{text}'"
                )

        # Optionally adjust reset_frames_1 to cover the VITL settling time
        # before computing integrations, and persist the new SC_Resets1.
        # This affects number of integrations so needs to be set before
        # those are calculated.
        if getattr(self, "update_nirda_reset1_for_vitl", False):
            vitl_settling_time = getattr(
                self, "vitl_settling_time", 60.0 * u.s
            )
            old_resets1 = sequence.get_payload_parameter(
                "AcquireInfCamImages", "SC_Resets1"
            )
            nirda.update_for_vitl(vitl_settling_time)
            new_resets1 = str(int(nirda.reset_frames_1))
            sequence.set_payload_parameter(
                "AcquireInfCamImages", "SC_Resets1", new_resets1
            )
            if str(old_resets1) != new_resets1:
                self._print(
                    f"{prefix} | VITL: SC_Resets1 '{old_resets1}' -> "
                    f"'{new_resets1}' to cover "
                    f"{vitl_settling_time.to(u.s).value:.1f} s settling"
                )

        old_integrations = sequence.get_payload_parameter(
            "AcquireInfCamImages", "SC_Integrations"
        )
        integrations, data, data_compressed = nirda.solve_integrations(
            duration.to(u.s), overhead
        )
        self._warn_if_data_exceeds_limits(
            sequence, "NIRDA", data, data_compressed, visit_id=visit_id
        )

        success = sequence.set_payload_parameter(
            "AcquireInfCamImages", "SC_Integrations", str(int(integrations))
        )
        if success:
            self._print(
                f"{prefix} | NIRDA SC_Integrations "
                f"'{old_integrations}' -> '{int(integrations)}'"
            )
        else:
            self._print(
                f"Warning: {prefix} | Failed to update SC_Integrations"
            )

        return sequence

    def validate_visibility(
        self, calendar: ScienceCalendar, report_issues: bool = True
    ) -> List[Dict[str, Any]]:
        """Validate that all sequences have good visibility.

        Returns a list of issue dicts. Each dict contains:

        - ``sequence_id``, ``visit_id``, ``target``
        - ``ra``, ``dec`` (degrees)
        - ``roll`` used (degrees) or *None*
        - ``start_time``, ``stop_time``
        - ``total_minutes``, ``non_visible_minutes``
        - ``visibility_fraction`` (0–1)
        - ``first_gap_start``, ``last_gap_end`` – Time bounds of
          non-visible spans
        - ``constraint_failures`` – dict from
          ``Visibility.get_all_constraints`` at the first
          non-visible minute (keys: moon, sun, earthlimb,
          star_tracker; values: bool)
        - ``constraint_summary`` – human-readable string
          listing which constraints failed
        - ``message`` – one-line actionable description
        """
        issues = []

        vis_bar = self._progress_bar(
            sum(len(v.sequences) for v in calendar.visits),
            desc="Validating visibility",
        )
        for visit in calendar.visits:
            visit_rolls = self._computed_target_rolls.get(visit.id, {})
            for seq in visit.sequences:
                n_mins = int(np.rint(seq.duration.sec / 60.0))
                target_coord = SkyCoord(
                    seq.ra, seq.dec, frame="icrs", unit="deg"
                )
                deltas = np.arange(n_mins) * u.min
                times = seq.start_time + deltas

                target_roll = visit_rolls.get(seq.target)
                if self._roll_sweep_enabled and target_roll is not None:
                    vis = self.visibility.get_visibility(
                        target_coord,
                        times,
                        roll=target_roll * u.deg,
                    )
                else:
                    vis = self.visibility.get_visibility(target_coord, times)

                if not np.all(vis):
                    vis_arr = np.asarray(vis)
                    non_vis_mask = ~vis_arr
                    non_vis_indices = np.where(non_vis_mask)[0]
                    first_gap_start = times[non_vis_indices[0]]
                    last_gap_end = times[non_vis_indices[-1]] + 1 * u.min
                    non_visible_minutes = int(np.sum(non_vis_mask))

                    # Constraint breakdown at first non-visible minute
                    constraint_failures = {}
                    constraint_summary = ""
                    roll_used = (
                        target_roll
                        if (
                            self._roll_sweep_enabled
                            and target_roll is not None
                        )
                        else None
                    )
                    constraint_details = {}
                    try:
                        fail_time = times[non_vis_indices[0]]
                        constraint_failures = (
                            self.visibility.get_all_constraints(
                                target_coord,
                                fail_time,
                            )
                        )
                        # Capture actual separations and limits
                        try:
                            seps = self.visibility.get_separations(
                                target_coord, fail_time
                            )
                            vis_obj = self.visibility
                            for body in [
                                "moon",
                                "sun",
                                "earthlimb",
                                "mars",
                                "jupiter",
                            ]:
                                if body not in constraint_failures:
                                    continue
                                actual = seps.get(body)
                                if actual is None:
                                    continue
                                # Determine the effective limit
                                if body == "earthlimb" and (
                                    vis_obj.earthlimb_day_min is not None
                                    or vis_obj.earthlimb_night_min is not None
                                ):
                                    # Day/night mode: compute
                                    # effective threshold at this
                                    # time using the same geometry
                                    # as the constraint check.
                                    try:
                                        obs_loc = (
                                            vis_obj._get_observer_location(
                                                fail_time
                                            )
                                        )
                                        obs_gcrs = obs_loc.get_gcrs(
                                            obstime=fail_time
                                        )
                                        obs_xyz = obs_gcrs.cartesian.xyz.to(
                                            u.m
                                        ).value
                                        zenith_u = obs_xyz / np.linalg.norm(
                                            obs_xyz
                                        )
                                        tgt_gcrs = target_coord.transform_to(
                                            GCRS(obstime=fail_time)
                                        )
                                        tgt_u = tgt_gcrs.cartesian.xyz.value
                                        tgt_u = tgt_u / np.linalg.norm(tgt_u)
                                        sun_body = get_body(
                                            "sun",
                                            time=fail_time,
                                            location=obs_loc,
                                        )
                                        sun_u = sun_body.cartesian.xyz.value
                                        sun_u = sun_u / np.linalg.norm(sun_u)
                                        obs_dist = np.linalg.norm(obs_xyz)
                                        la_rad = np.arccos(
                                            6371000.0 / obs_dist
                                        )
                                        eff_deg = float(
                                            vis_obj._effective_earthlimb_min_deg(
                                                tgt_u,
                                                zenith_u,
                                                sun_u,
                                                limb_angle_rad=la_rad,
                                            )
                                        )
                                        is_day = bool(
                                            eff_deg
                                            == (
                                                vis_obj.earthlimb_day_min.to(
                                                    u.deg
                                                ).value
                                                if vis_obj.earthlimb_day_min
                                                is not None
                                                else vis_obj.earthlimb_min.to(
                                                    u.deg
                                                ).value
                                            )
                                        )
                                        side = "day" if is_day else "night"
                                        limit_deg = eff_deg
                                        constraint_details[body] = {
                                            "passes": bool(
                                                constraint_failures[body]
                                            ),
                                            "required_deg": limit_deg,
                                            "actual_deg": float(
                                                actual.to(u.deg).value
                                            ),
                                            "side": side,
                                        }
                                    except Exception:
                                        # Fall back to simple limit
                                        limit = getattr(
                                            vis_obj,
                                            "earthlimb_min",
                                            None,
                                        )
                                        if limit is not None:
                                            constraint_details[body] = {
                                                "passes": bool(
                                                    constraint_failures[body]
                                                ),
                                                "required_deg": float(
                                                    limit.to(u.deg).value
                                                ),
                                                "actual_deg": float(
                                                    actual.to(u.deg).value
                                                ),
                                            }
                                else:
                                    limit = getattr(
                                        vis_obj,
                                        f"{body}_min",
                                        None,
                                    )
                                    if limit is not None:
                                        constraint_details[body] = {
                                            "passes": bool(
                                                constraint_failures[body]
                                            ),
                                            "required_deg": float(
                                                limit.to(u.deg).value
                                            ),
                                            "actual_deg": float(
                                                actual.to(u.deg).value
                                            ),
                                        }
                            # Star tracker details
                            if vis_obj._st_constraint_active:
                                for tracker in [1, 2]:
                                    try:
                                        angles = (
                                            vis_obj.get_star_tracker_angles(
                                                target_coord,
                                                fail_time,
                                                tracker,
                                            )
                                        )
                                        checks = vis_obj._st_checks_for(
                                            tracker
                                        )
                                        for name, limit, key in checks:
                                            actual_val = angles[key]
                                            ok = bool(actual_val >= limit)
                                            label = f"st{tracker}_{name}"
                                            constraint_details[label] = {
                                                "passes": ok,
                                                "required_deg": float(
                                                    limit.to(u.deg).value
                                                ),
                                                "actual_deg": float(
                                                    actual_val.to(u.deg).value
                                                ),
                                            }
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        failed = [
                            k for k, v in constraint_failures.items() if not v
                        ]
                        if failed:
                            constraint_summary = ", ".join(failed)
                        elif roll_used is not None:
                            # Boresight constraints all pass but
                            # roll-aware visibility still fails →
                            # the star-tracker keepout at this
                            # roll is the culprit.
                            constraint_summary = (
                                f"star_tracker at " f"roll={roll_used:.1f}°"
                            )
                            constraint_failures["star_tracker_at_roll"] = False
                        else:
                            constraint_summary = "unknown"
                    except Exception:
                        constraint_summary = "(unable to determine)"

                    vis_frac = float(np.sum(vis_arr) / len(vis_arr))

                    message = (
                        f"Seq {seq.id} ({seq.target}) in visit "
                        f"{visit.id}: {vis_frac:.0%} visible "
                        f"({non_visible_minutes}/{n_mins} min "
                        f"dark). Failed: {constraint_summary}. "
                        f"First gap at {first_gap_start.isot}."
                    )

                    issue = {
                        "sequence_id": seq.id,
                        "visit_id": visit.id,
                        "target": seq.target,
                        "ra": seq.ra,
                        "dec": seq.dec,
                        "roll": roll_used,
                        "start_time": seq.start_time,
                        "stop_time": seq.stop_time,
                        "total_minutes": n_mins,
                        "non_visible_minutes": non_visible_minutes,
                        "visibility_fraction": vis_frac,
                        "first_gap_start": first_gap_start,
                        "last_gap_end": last_gap_end,
                        "constraint_failures": constraint_failures,
                        "constraint_details": constraint_details,
                        "constraint_summary": constraint_summary,
                        "message": message,
                    }
                    issues.append(issue)

                    if report_issues:
                        self._print(message)

                vis_bar.update(1)
        vis_bar.close()

        return issues

    # ------------------------------------------------------------------
    # Per-day diagnostics (.diag)
    # ------------------------------------------------------------------
    @staticmethod
    def _diag_choose_day(start_dt: datetime, stop_dt: datetime) -> str:
        """Assign an observation to the UTC day with the largest overlap."""
        if stop_dt <= start_dt:
            return start_dt.date().isoformat()
        day_seconds: Dict[str, float] = defaultdict(float)
        cursor = start_dt
        while cursor < stop_dt:
            next_midnight = datetime(
                cursor.year, cursor.month, cursor.day
            ) + timedelta(days=1)
            chunk_end = min(next_midnight, stop_dt)
            day_seconds[cursor.date().isoformat()] += (
                chunk_end - cursor
            ).total_seconds()
            cursor = chunk_end
        return max(day_seconds.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def _fits_files_for_sequence(
        seq: ObservationSequence,
        overhead: OverheadTiming,
        nirda: Optional[NirdaData],
        sc_integrations: int,
        visda: Optional[VisdaData],
        num_total_frames: int,
    ) -> List[str]:
        """Return the FITS product filenames packaged in a sequence's ``.bin``.

        Names follow the Pandora flight-software conventions (InfImg from
        ``MACIEMain.cpp``, VisSci from ``PCOCameraMain.cpp``) built from the
        observation's payload parameters. The per-detector capture-start
        timestamps are the sequence start plus the detector pre-overhead.
        """
        files: List[str] = []
        target = seq.target

        # NIRDA: InfImg cube (only if integrations were scheduled).
        if nirda is not None and sc_integrations > 0:
            nir_start = (
                seq.start_time + overhead.nirda_pre_overhead_time.to(u.s)
            ).datetime
            nir_date = nir_start.strftime("%Y-%m-%d__%H-%M-%S")
            cube_depth = sc_integrations * nirda.groups
            files.append(
                f"{nir_date}_InfImg_{target}_"
                f"d{nirda.roi_x_size:04d}x{nirda.roi_y_size:04d}"
                f"x{cube_depth:04d}_b1_e01"
                f"_i{sc_integrations:02d}_g{nirda.groups:02d}"
                f"_d{nirda.drop_frames_2:02d}_r{nirda.read_frames:02d}.fits"
            )

        # VISDA: VisSci cube (only if frames were requested).
        if visda is not None and num_total_frames > 0:
            vis_start = (
                seq.start_time + overhead.visda_pre_overhead_time.to(u.s)
            ).datetime
            vis_date = vis_start.strftime("%Y-%m-%d__%H-%M-%S")
            exp_us = int(visda.exposure_time_s.to(u.us).value)
            files.append(
                f"{vis_date}_VisSci_{target}_"
                f"d{visda.roi_dimension:03d}_n{visda.num_rois:03d}"
                f"_f{num_total_frames:05d}_e{exp_us:09d}us.fits"
            )

        # Engineering housekeeping file (timestamp only; downlink time is not
        # known here, so the sequence start is used as a best-effort stamp).
        eng_date = seq.start_time.datetime.strftime("%Y_%m_%dT%H_%M_%S")
        files.append(f"{eng_date}_engineering.fits")

        return files

    def generate_diagnostics(
        self,
        calendar: ScienceCalendar,
        output_path: Optional[Any] = None,
        pass_data_volume_mb: Optional[float] = None,
    ) -> str:
        """Build a per-day ``.diag`` report and (optionally) write it.

        Mirrors the legacy CalendarCleaner diagnostic: a week summary
        followed by a per-day breakdown of observation counts (by priority),
        unique targets, NIR/VIS frame and data totals (compressed and
        uncompressed), observing/gap minutes (with percentages), and a
        per-day file manifest. Data volumes are computed from the
        ``NirdaData``/``VisdaData`` models built from each observation's
        payload, so they stay consistent with the scheduler.

        Parameters
        ----------
        calendar : ScienceCalendar
            Calendar to summarize (typically the processed calendar).
        output_path : str or pathlib.Path, optional
            Where to write the ``.diag`` file (its suffix is forced to
            ``.diag``). If omitted, the calendar's ``source_path`` metadata
            is used; if that is also missing, nothing is written and only
            the text is returned.
        pass_data_volume_mb : float, optional
            Downlink volume of a single pass (MB). When given, "Required
            Passes" is reported; otherwise it shows "N/A".

        Returns
        -------
        str
            The full diagnostic text.
        """
        mib = 1024.0 * 1024.0

        def fmt(value: float) -> str:
            value = float(value)
            if value.is_integer():
                return str(int(value))
            return f"{value:.3f}".rstrip("0").rstrip(".")

        def data_str(data_bytes: float) -> str:
            data_mb = data_bytes / mib
            if pass_data_volume_mb and pass_data_volume_mb > 0:
                passes = data_mb / pass_data_volume_mb
                return f"{fmt(data_mb)} MB (Required Passes: {fmt(passes)})"
            return f"{fmt(data_mb)} MB (Required Passes: N/A)"

        def to_int(value: Any) -> int:
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return 0

        def new_bucket() -> Dict[str, Any]:
            return {
                "count": 0,
                "priority_counts": {0: 0, 1: 0, 2: 0},
                "targets": set(),
                "nir_frames": 0,
                "vis_frames": 0,
                "nir_data": 0.0,
                "vis_data": 0.0,
                "nir_data_unc": 0.0,
                "vis_data_unc": 0.0,
                "timelines": [],
                "manifest": [],
            }

        daily: Dict[str, Dict[str, Any]] = {}

        # Per-detector capture-start offsets for FITS filenames.
        overhead = getattr(self, "overhead", None) or OverheadTiming()

        for visit in calendar.visits:
            for seq in visit.sequences:
                start_dt = seq.start_time.datetime
                stop_dt = seq.stop_time.datetime
                day = self._diag_choose_day(start_dt, stop_dt)
                bucket = daily.setdefault(day, new_bucket())

                bucket["count"] += 1
                if seq.priority in bucket["priority_counts"]:
                    bucket["priority_counts"][seq.priority] += 1
                bucket["targets"].add(seq.target)
                bucket["timelines"].append((start_dt, stop_dt))

                # NIRDA frames + data from the NirdaData model.
                nirda, _ = self._build_payload_data(seq, (), NirdaData)
                sc_integrations = to_int(
                    seq.get_payload_parameter(
                        "AcquireInfCamImages", "SC_Integrations"
                    )
                )
                if nirda is not None and sc_integrations > 0:
                    nir_frames = (
                        sc_integrations
                        * nirda.other_integration_saved_frames
                    )
                    nir_unc = (
                        sc_integrations * nirda.integration_data
                    ).to(u.byte).value
                    bucket["nir_frames"] += int(nir_frames)
                    bucket["nir_data_unc"] += nir_unc
                    bucket["nir_data"] += nir_unc * nirda.compression_ratio

                # VISDA frames (coadds) + data from the VisdaData model.
                visda, _ = self._build_payload_data(
                    seq,
                    (),
                    VisdaData,
                    extra_kwargs={"read_time_per_frame_s": 0 * u.s},
                )
                num_total_frames = to_int(
                    seq.get_payload_parameter(
                        "AcquireVisCamScienceData", "NumTotalFramesRequested"
                    )
                )
                if visda is not None and num_total_frames > 0:
                    coadds = (
                        num_total_frames // visda.frames_per_coadd
                        if visda.frames_per_coadd > 0
                        else num_total_frames
                    )
                    vis_unc = (coadds * visda.frame_bytes).to(u.byte).value
                    bucket["vis_frames"] += int(coadds)
                    bucket["vis_data_unc"] += vis_unc
                    bucket["vis_data"] += vis_unc * visda.compression_ratio

                # Manifest: the downlinked .bin plus the individual FITS
                # products it contains (named from payload parameters).
                stamp = start_dt.strftime("%Y%m%dT%H%M%S")
                bin_path = f"/mnt/data/sci/{stamp}_{seq.target}.bin"
                fits_files = self._fits_files_for_sequence(
                    seq,
                    overhead,
                    nirda,
                    sc_integrations,
                    visda,
                    num_total_frames,
                )
                bucket["manifest"].append((bin_path, fits_files))

        text = self._render_diagnostics(daily, fmt, data_str)

        # Resolve where to write the .diag file.
        base = output_path
        if base is None:
            base = (calendar.metadata or {}).get("source_path")
        if base is not None:
            diag_path = Path(base).with_suffix(".diag")
            diag_path.write_text(text, encoding="utf-8")
            self._print(f"Wrote diagnostics to {diag_path}")

        return text

    @staticmethod
    def _diag_observing_and_gaps(timelines: List[Tuple]) -> Tuple[float, float]:
        """Return (observing_minutes, gap_minutes) for a day's timelines."""
        observing = 0.0
        gaps = 0.0
        previous_stop = None
        for start_dt, stop_dt in sorted(timelines, key=lambda t: (t[0], t[1])):
            observing += max(0.0, (stop_dt - start_dt).total_seconds() / 60.0)
            if previous_stop is not None and start_dt > previous_stop:
                gaps += (start_dt - previous_stop).total_seconds() / 60.0
            previous_stop = (
                stop_dt
                if previous_stop is None
                else max(previous_stop, stop_dt)
            )
        return observing, gaps

    def _render_diagnostics(self, daily, fmt, data_str) -> str:
        """Render the diagnostic text from the per-day buckets."""
        sorted_days = sorted(daily.keys())
        if not sorted_days:
            return "No observations were available for diagnostic generation.\n"

        def pct(part: float, whole: float) -> str:
            return f"{(100.0 * part / whole):.1f}" if whole > 0 else "0.0"

        # Finalize per-day observing/gap minutes and accumulate week totals.
        summary = {
            "count": 0,
            "priority_counts": {0: 0, 1: 0, 2: 0},
            "nir_data": 0.0,
            "vis_data": 0.0,
            "nir_data_unc": 0.0,
            "vis_data_unc": 0.0,
            "observing": 0.0,
            "gaps": 0.0,
        }
        for day in sorted_days:
            item = daily[day]
            observing, gaps = self._diag_observing_and_gaps(item["timelines"])
            item["observing"] = observing
            item["gaps"] = gaps
            summary["count"] += item["count"]
            for p in (0, 1, 2):
                summary["priority_counts"][p] += item["priority_counts"][p]
            summary["nir_data"] += item["nir_data"]
            summary["vis_data"] += item["vis_data"]
            summary["nir_data_unc"] += item["nir_data_unc"]
            summary["vis_data_unc"] += item["vis_data_unc"]
            summary["observing"] += observing
            summary["gaps"] += gaps

        lines: List[str] = []

        # ── Week summary ───────────────────────────────────────────
        sum_total = summary["nir_data"] + summary["vis_data"]
        sum_total_unc = summary["nir_data_unc"] + summary["vis_data_unc"]
        sum_span = summary["observing"] + summary["gaps"]
        lines.append(
            f"Calendar Summary {sorted_days[0]} : {sorted_days[-1]}"
        )
        lines.append(f"Total Observations: {summary['count']}")
        lines.append(f"  - Priority 0 = {summary['priority_counts'][0]}")
        lines.append(f"  - Priority 1 = {summary['priority_counts'][1]}")
        lines.append(f"  - Priority 2 = {summary['priority_counts'][2]}")
        lines.append(
            f"Total Gaps: {fmt(summary['gaps'])} Mins "
            f"({pct(summary['gaps'], sum_span)}%)"
        )
        lines.append(
            f"Total Observing: {fmt(summary['observing'])} Mins "
            f"({pct(summary['observing'], sum_span)}%)"
        )
        lines.append("Total NIR Data = " + data_str(summary["nir_data"]))
        lines.append("Total Vis Data = " + data_str(summary["vis_data"]))
        lines.append("Total Data = " + data_str(sum_total))
        lines.append("Uncompressed Data")
        lines.append("Total NIR Data = " + data_str(summary["nir_data_unc"]))
        lines.append("Total Vis Data = " + data_str(summary["vis_data_unc"]))
        lines.append("Total Data = " + data_str(sum_total_unc))
        lines.append("")

        # ── Per-day breakdown ──────────────────────────────────────
        for day in sorted_days:
            item = daily[day]
            day_total = item["nir_data"] + item["vis_data"]
            day_total_unc = item["nir_data_unc"] + item["vis_data_unc"]
            span = item["observing"] + item["gaps"]

            lines.append(day)
            lines.append(f"Number of Observations: {item['count']}")
            lines.append(f"  - Priority 0 = {item['priority_counts'][0]}")
            lines.append(f"  - Priority 1 = {item['priority_counts'][1]}")
            lines.append(f"  - Priority 2 = {item['priority_counts'][2]}")
            lines.append("List of Unique Targets:")
            for target in sorted(item["targets"]):
                lines.append(f"  - {target}")
            lines.append(f"Total NIR Frames = {fmt(item['nir_frames'])}")
            lines.append(f"Total Vis Frames = {fmt(item['vis_frames'])}")
            lines.append(
                f"Total Gaps: {fmt(item['gaps'])} Mins "
                f"({pct(item['gaps'], span)}%)"
            )
            lines.append(
                f"Total Observing: {fmt(item['observing'])} Mins "
                f"({pct(item['observing'], span)}%)"
            )
            lines.append("Total NIR Data = " + data_str(item["nir_data"]))
            lines.append("Total Vis Data = " + data_str(item["vis_data"]))
            lines.append("Total Data = " + data_str(day_total))
            lines.append("Uncompressed Data")
            lines.append("Total NIR Data = " + data_str(item["nir_data_unc"]))
            lines.append("Total Vis Data = " + data_str(item["vis_data_unc"]))
            lines.append("Total Data = " + data_str(day_total_unc))
            lines.append("")
            lines.append("Manifest of Files for the Day:")
            for bin_path, fits_files in sorted(
                item["manifest"], key=lambda entry: entry[0]
            ):
                lines.append(f"- {bin_path}")
                for fits_name in fits_files:
                    lines.append(f"\t- {fits_name}")
            lines.append("")
            lines.append("----")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def validate_target_names(
        self, calendar: ScienceCalendar, report_issues: bool = True
    ) -> List[Dict[str, Any]]:
        """Validate that all target names do not contain spaces.

        Parameters
        ----------
        calendar : ScienceCalendar
            The calendar to validate.
        report_issues : bool, optional
            If True, print issues to stdout.

        Returns
        -------
        List[Dict[str, Any]]
            List of issues found. Each issue is a dict with:
            - sequence_id: str
            - target: str
            - visit_id: str
        """
        issues = []

        for visit in calendar.visits:
            for seq in visit.sequences:
                if seq.target and " " in seq.target:
                    issue = {
                        "sequence_id": seq.id,
                        "target": seq.target,
                        "visit_id": visit.id,
                    }
                    issues.append(issue)

                    if report_issues:
                        self._print(
                            f"Target name issue: '{seq.target}' contains spaces (sequence {seq.id}, visit {visit.id})"
                        )

        return issues

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _print(self, *args, **kwargs) -> None:
        """Route a ``print``-style call through the run logger.

        Joins *args* like ``print`` and logs the result. Messages whose
        (stripped) text begins with "warning" or "error" are logged at
        WARNING/ERROR level so they reach the console and the
        ``.errors.log`` file; everything else is logged at INFO and only
        reaches the console when ``verbose`` was set. If no run logger has
        been configured (e.g. a bare processor in a unit test), falls back
        to the builtin ``print``.
        """
        sep = kwargs.get("sep", " ")
        message = sep.join(str(a) for a in args)

        logger = getattr(self, "logger", None)
        if logger is None:
            print(message)
            return

        head = message.lstrip().lower()
        if head.startswith("error"):
            logger.error(message)
        elif head.startswith("warning"):
            logger.warning(message)
        else:
            logger.info(message)

    def _setup_run_logging(
        self,
        calendar: ScienceCalendar,
        verbose: bool,
        log_path: Optional[Any] = None,
    ) -> None:
        """Configure ``self.logger`` for a processing run.

        Two log files are written alongside (and named after) the input
        calendar: ``<stem>.log`` captures everything, and
        ``<stem>.errors.log`` captures only warnings/errors and is created
        lazily (so it never appears when the run is clean).

        Parameters
        ----------
        calendar : ScienceCalendar
            Used to discover the source calendar path via
            ``metadata['source_path']`` when *log_path* is not given.
        verbose : bool
            When True the console receives INFO and above; otherwise the
            console receives only WARNING and above. The ``.log`` file
            always receives INFO and above.
        log_path : str or pathlib.Path, optional
            Explicit base path for the log file. Its suffix is replaced with
            ``.log``. If omitted, the calendar's ``source_path`` is used.
            If neither is available, only console logging is configured.
        """
        logger = logging.getLogger(f"shortschedule.run.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        # Drop any handlers from a previous run on this processor.
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        # No per-entry timestamps; the run start time is recorded once in the
        # log file header instead (see below).
        fmt = logging.Formatter("%(message)s")

        # Console handler: gated by verbose for INFO, always shows warnings.
        console = logging.StreamHandler()
        console.setLevel(logging.INFO if verbose else logging.WARNING)
        console.setFormatter(fmt)
        logger.addHandler(console)

        # Resolve the log file base path.
        base = log_path
        if base is None:
            source = (calendar.metadata or {}).get("source_path")
            base = source
        if base is not None:
            base = Path(base)
            log_file = base.with_suffix(".log")
            errors_file = base.with_suffix(".errors.log")

            # Write a header recording the run start time in UTC and US
            # Eastern, so individual entries need not carry timestamps. The
            # FileHandler below then appends to this file.
            now_utc = datetime.now(timezone.utc)
            utc_str = now_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
            if ZoneInfo is not None:
                eastern = now_utc.astimezone(ZoneInfo("America/New_York"))
                eastern_str = eastern.strftime("%Y-%m-%d %H:%M:%S %Z")
            else:  # pragma: no cover - zoneinfo missing
                eastern_str = "unavailable (zoneinfo not installed)"
            with open(log_file, "w", encoding="utf-8") as handle:
                handle.write("=" * 70 + "\n")
                handle.write("Short-term scheduler run log\n")
                handle.write(f"Run start (UTC):     {utc_str}\n")
                handle.write(f"Run start (Eastern): {eastern_str}\n")
                handle.write("=" * 70 + "\n\n")

            file_handler = logging.FileHandler(
                log_file, mode="a", encoding="utf-8"
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

            # delay=True means the file is only created on first emit, so a
            # clean run leaves no (empty) errors log behind.
            errors_handler = logging.FileHandler(
                errors_file, mode="w", encoding="utf-8", delay=True
            )
            errors_handler.setLevel(logging.WARNING)
            errors_handler.setFormatter(fmt)
            logger.addHandler(errors_handler)

            self.logger = logger
            self.logger.info(f"Logging to {log_file}")
        else:
            self.logger = logger

    @staticmethod
    def _seq_prefix(visit_id: Any, seq: ObservationSequence) -> str:
        """Return the standard log prefix for an observation.

        Format: ``<start datetime>-<target id>-<visit id>-<observation id>``.
        """
        try:
            start = seq.start_time.isot
        except Exception:
            start = str(getattr(seq, "start_time", "?"))
        return f"{start}-{seq.target}-{visit_id}-{seq.id}"

    def _progress(self, iterable, desc: str, total: Optional[int] = None):
        """Wraps iterables in a tqdm progress bar when available.

        Falls back to the plain iterable if ``tqdm`` is not installed. The
        bar auto-disables on non-interactive streams (``disable=None``), so
        it shows during real runs but stays silent under pytest/CI.
        """
        if tqdm is None:
            return iterable
        return tqdm(
            iterable, desc=desc, total=total, disable=None, leave=False
        )

    def _progress_bar(self, total: int, desc: str):
        """Return a manually-updated progress bar (or a no-op fallback).

        Use when a single bar must span a nested loop: call ``.update()``
        per item and ``.close()`` when done.
        """
        if tqdm is None:
            return _NullProgress()
        return tqdm(total=total, desc=desc, disable=None, leave=False)

    def _initialize_gap_report(self) -> None:
        """Initialize/reset the gap report structure."""
        self.gap_report = {
            "original_calendar_stats": {},
            "processed_calendar_stats": {},
            "visibility_analysis": {
                "original_gaps": [],
                "filled_gaps": [],
                "remaining_gaps": [],
                "unfillable_gaps": [],
            },
            "sequence_modifications": {
                "extended_sequences": [],
                "shortened_sequences": [],
                "unchanged_sequences": [],
            },
            "processing_summary": {
                "total_gaps_found": 0,
                "gaps_filled": 0,
                "gaps_remaining": 0,
                "total_time_recovered_minutes": 0,
                "sequences_modified": 0,
                "original_gap_time_minutes": 0,
                "duty_cycle_improvement_percent": 0,
                "duration_improvement_minutes": 0,
                "duration_improvement_hours": 0,
                "sequences_added": 0,
            },
        }

    def _analyze_original_calendar(self, calendar: ScienceCalendar) -> None:
        """Analyze original calendar before processing."""
        stats = calendar.get_summary_stats()

        self.gap_report["original_calendar_stats"] = {
            "total_sequences": stats["total_sequences"],
            "total_duration_minutes": stats["total_duration_minutes"],
            "total_duration_hours": stats["total_duration_hours"],
            "calendar_span_days": stats["calendar_span_days"],
            "duty_cycle_percent": stats["duty_cycle_percent"],
            "priority_breakdown": stats["priority_breakdown"],
        }

    def _analyze_original_visibility(
        self, calendar: ScienceCalendar, verbose: bool = False
    ) -> None:
        """Analyze visibility gaps in original calendar."""
        original_gaps = []
        total_gap_time = 0

        # Get all sequences chronologically
        all_sequences = []
        for visit in calendar.visits:
            for seq in visit.sequences:
                all_sequences.append(seq)
        all_sequences.sort(key=lambda s: s.start_time)

        # Check for gaps between sequences
        for i in range(len(all_sequences) - 1):
            current_seq = all_sequences[i]
            next_seq = all_sequences[i + 1]

            gap_start = current_seq.stop_time
            gap_end = next_seq.start_time
            gap_duration = (gap_end - gap_start).sec / 60.0

            if gap_duration > 0:
                gap_info = {
                    "gap_start": gap_start,
                    "gap_end": gap_end,
                    "duration_minutes": gap_duration,
                    "before_sequence": current_seq.id,
                    "after_sequence": next_seq.id,
                    "before_target": current_seq.target,
                    "after_target": next_seq.target,
                }
                original_gaps.append(gap_info)
                total_gap_time += gap_duration

                self._print(
                    f"Original gap: {gap_duration:.1f} min between "
                    f"{current_seq.id} and {next_seq.id}"
                )

        self.gap_report["visibility_analysis"]["original_gaps"] = original_gaps
        self.gap_report["processing_summary"][
            "original_gap_time_minutes"
        ] = total_gap_time

    def _analyze_processed_calendar(self, calendar: ScienceCalendar) -> None:
        """Analyze processed calendar and compare to original."""
        stats = calendar.get_summary_stats()

        self.gap_report["processed_calendar_stats"] = {
            "total_sequences": stats["total_sequences"],
            "total_duration_minutes": stats["total_duration_minutes"],
            "total_duration_hours": stats["total_duration_hours"],
            "calendar_span_days": stats["calendar_span_days"],
            "duty_cycle_percent": stats["duty_cycle_percent"],
            "priority_breakdown": stats["priority_breakdown"],
        }

    def _finalize_gap_report(self) -> None:
        """Generate final summary statistics."""
        original = self.gap_report["original_calendar_stats"]
        processed = self.gap_report["processed_calendar_stats"]

        # Calculate improvements
        duty_cycle_improvement = (
            processed["duty_cycle_percent"] - original["duty_cycle_percent"]
        )

        duration_improvement = (
            processed["total_duration_minutes"]
            - original["total_duration_minutes"]
        )

        self.gap_report["processing_summary"].update(
            {
                "duty_cycle_improvement_percent": duty_cycle_improvement,
                "duration_improvement_minutes": duration_improvement,
                "duration_improvement_hours": duration_improvement / 60,
                "sequences_added": processed["total_sequences"]
                - original["total_sequences"],
            }
        )

    def get_gap_report(self) -> Dict[str, Any]:
        """Return comprehensive gap analysis report."""
        return self.gap_report

    def print_gap_summary(self):
        """Print a human-readable summary of gap analysis."""
        report = self.gap_report
        summary = report["processing_summary"]

        self._print("\n" + "=" * 60)
        self._print("VISIBILITY GAP ANALYSIS SUMMARY")
        self._print("=" * 60)

        self._print("\nORIGINAL CALENDAR:")
        self._print(
            f"  Total Sequences: {report['original_calendar_stats']['total_sequences']}"
        )
        self._print(
            f"  Total Duration: {report['original_calendar_stats']['total_duration_hours']:.1f} hours"
        )
        self._print(
            f"  Duty Cycle: {report['original_calendar_stats']['duty_cycle_percent']:.1f}%"
        )

        self._print("\nPROCESSED CALENDAR:")
        self._print(
            f"  Total Sequences: {report['processed_calendar_stats']['total_sequences']}"
        )
        self._print(
            f"  Total Duration: {report['processed_calendar_stats']['total_duration_hours']:.1f} hours"
        )
        self._print(
            f"  Duty Cycle: {report['processed_calendar_stats']['duty_cycle_percent']:.1f}%"
        )

        self._print("\nIMPROVEMENTS:")
        self._print(
            f"  Duration Gained: {summary.get('duration_improvement_hours', 0):.1f} hours"
        )
        self._print(
            f"  Duty Cycle Improved: {summary.get('duty_cycle_improvement_percent', 0):.1f}%"
        )
        self._print(f"  Sequences Modified: {summary.get('sequences_modified', 0)}")

        if "gaps_filled" in summary:
            self._print(
                f"  Gaps Filled: {summary['gaps_filled']}/{summary['gaps_filled'] + summary['gaps_remaining']}"
            )

    def debug_sequence_visibility(
        self,
        calendar: ScienceCalendar,
        sequence_id: str,
        target_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Debug visibility for a specific sequence."""
        # Find the sequence
        target_seq = None
        target_visit_id = None

        for visit in calendar.visits:
            for seq in visit.sequences:
                if seq.id == sequence_id and (
                    target_name is None or seq.target == target_name
                ):
                    target_seq = seq
                    target_visit_id = visit.id
                    break
            if target_seq:
                break

        if not target_seq:
            self._print(f"Sequence {sequence_id} not found")
            return

        self._print(f"\n{'='*60}")
        self._print(f"DEBUGGING SEQUENCE {sequence_id}: {target_seq.target}")
        self._print(f"{'='*60}")
        self._print(f"Visit ID: {target_visit_id}")
        self._print(f"Start Time: {target_seq.start_time}")
        self._print(f"Stop Time: {target_seq.stop_time}")
        self._print(f"Duration: {target_seq.duration.sec/60:.1f} minutes")
        self._print(f"Target: {target_seq.target}")
        self._print(f"RA/Dec: {target_seq.ra:.3f}, {target_seq.dec:.3f}")

        # Check visibility minute by minute
        n_mins = int(np.rint(target_seq.duration.sec / 60.0))
        target_coord = SkyCoord(
            target_seq.ra, target_seq.dec, frame="icrs", unit="deg"
        )
        deltas = np.arange(n_mins) * u.min
        times = target_seq.start_time + deltas

        vis = self.visibility.get_visibility(target_coord, times)

        self._print("\nMinute-by-minute visibility:")
        for i, (time, visible) in enumerate(zip(times, vis)):
            status = "✓ VISIBLE" if visible else "✗ NOT VISIBLE"
            self._print(f"  Minute {i+1}: {time.isot} - {status}")

        self._print("\nVisibility Summary:")
        self._print(f"  Total minutes: {len(vis)}")
        self._print(f"  Visible minutes: {np.sum(vis)}")
        self._print(f"  Visibility fraction: {np.sum(vis)/len(vis):.3f}")

        return {
            "sequence": target_seq,
            "times": times,
            "visibility": vis,
            "visibility_fraction": np.sum(vis) / len(vis),
        }

    def validate_no_overlaps_astropy(
        self, calendar: ScienceCalendar, report_issues: bool = True
    ) -> List[Dict[str, Any]]:
        """Detect overlapping sequences using Astropy time comparison.

        Returns a list of overlap dicts containing:

        - ``sequence1_id``, ``sequence1_target``, ``visit1_id``
        - ``sequence2_id``, ``sequence2_target``, ``visit2_id``
        - ``sequence1_start``, ``sequence1_stop``
        - ``sequence2_start``, ``sequence2_stop``
        - ``overlap_duration_minutes``
        - ``suggested_fix`` – actionable string
        - ``message`` – one-line summary
        """

        overlaps = []
        tolerance = TimeDelta(1.0 * u.s)  # 1 second tolerance - correct way

        # Get all sequences sorted by time
        all_sequences = []
        for visit in calendar.visits:
            for seq in visit.sequences:
                all_sequences.append({"visit_id": visit.id, "sequence": seq})

        all_sequences.sort(key=lambda x: x["sequence"].start_time)

        # Check for overlaps
        for i in range(len(all_sequences) - 1):
            entry1 = all_sequences[i]
            entry2 = all_sequences[i + 1]
            seq1 = entry1["sequence"]
            seq2 = entry2["sequence"]

            # Check if seq1 ends significantly after seq2 starts
            if seq1.stop_time > (seq2.start_time + tolerance):
                overlap_duration = (
                    (seq1.stop_time - seq2.start_time).to(u.min).value
                )

                suggested_fix = (
                    f"Delay sequence {seq2.id} start to "
                    f"{seq1.stop_time.isot} or shorten "
                    f"sequence {seq1.id} stop by "
                    f"{overlap_duration:.1f} min."
                )
                message = (
                    f"Overlap: seq {seq1.id} ({seq1.target}, "
                    f"visit {entry1['visit_id']}) ends at "
                    f"{seq1.stop_time.isot} but seq {seq2.id} "
                    f"({seq2.target}, visit {entry2['visit_id']}) "
                    f"starts at {seq2.start_time.isot} "
                    f"({overlap_duration:.1f} min overlap). "
                    f"Fix: {suggested_fix}"
                )

                overlap_issue = {
                    "sequence1_id": seq1.id,
                    "sequence1_target": seq1.target,
                    "visit1_id": entry1["visit_id"],
                    "sequence1_start": seq1.start_time,
                    "sequence1_stop": seq1.stop_time,
                    "sequence2_id": seq2.id,
                    "sequence2_target": seq2.target,
                    "visit2_id": entry2["visit_id"],
                    "sequence2_start": seq2.start_time,
                    "sequence2_stop": seq2.stop_time,
                    "overlap_duration_minutes": overlap_duration,
                    "suggested_fix": suggested_fix,
                    "message": message,
                }
                overlaps.append(overlap_issue)

                if report_issues:
                    self._print(message)

        return overlaps

    def validate_sequence_timing(
        self, calendar: ScienceCalendar, report_issues: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive timing validation.

        Checks for overlaps, short sequences, and large gaps.
        Each sub-issue includes a ``message`` with actionable detail.

        Returns
        -------
        dict
            Keys: ``overlaps``, ``short_sequences``, ``large_gaps``,
            ``timing_summary``.
        """
        issues: Dict[str, Any] = {
            "overlaps": [],
            "short_sequences": [],
            "large_gaps": [],
            "timing_summary": {},
        }

        # Check for overlaps (already enhanced with message)
        issues["overlaps"] = self.validate_no_overlaps_astropy(
            calendar, report_issues=False
        )

        # Get all sequences sorted by time
        all_sequences = []
        for visit in calendar.visits:
            for seq in visit.sequences:
                all_sequences.append(
                    {
                        "visit_id": visit.id,
                        "sequence": seq,
                        "start_time": seq.start_time,
                        "stop_time": seq.stop_time,
                        "duration_minutes": seq.duration.sec / 60.0,
                    }
                )

        all_sequences.sort(key=lambda x: x["start_time"])

        # Check for sequences shorter than minimum duration
        min_duration = self.min_sequence_duration
        min_dur_min = min_duration.sec / 60.0
        for seq_info in all_sequences:
            dur_min = seq_info["duration_minutes"]
            if seq_info["sequence"].duration < min_duration:
                seq = seq_info["sequence"]
                message = (
                    f"Seq {seq.id} ({seq.target}) in visit "
                    f"{seq_info['visit_id']}: duration "
                    f"{dur_min:.1f} min < minimum "
                    f"{min_dur_min:.0f} min. "
                    f"Extend stop_time to at least "
                    f"{(seq.start_time + min_duration).isot}."
                )
                short_issue = {
                    "sequence_id": seq.id,
                    "target": seq.target,
                    "visit_id": seq_info["visit_id"],
                    "start_time": seq_info["start_time"],
                    "stop_time": seq_info["stop_time"],
                    "duration_minutes": dur_min,
                    "minimum_required_minutes": min_dur_min,
                    "suggested_fix": (
                        f"Extend stop_time to "
                        f"{(seq.start_time + min_duration).isot}"
                    ),
                    "message": message,
                }
                issues["short_sequences"].append(short_issue)

        # Check for large gaps between sequences
        max_acceptable_gap = 2.0 * u.minute  # 2 minutes
        for i in range(len(all_sequences) - 1):
            s1 = all_sequences[i]
            s2 = all_sequences[i + 1]
            gap_td = s2["start_time"] - s1["stop_time"]

            if gap_td > max_acceptable_gap:
                gap_min = gap_td.sec / 60.0
                message = (
                    f"Gap of {gap_min:.1f} min between "
                    f"seq {s1['sequence'].id} ({s1['sequence'].target}, "
                    f"visit {s1['visit_id']}) and "
                    f"seq {s2['sequence'].id} ({s2['sequence'].target}, "
                    f"visit {s2['visit_id']}): "
                    f"{s1['stop_time'].isot} \u2192 "
                    f"{s2['start_time'].isot}. "
                    f"Consider extending seq {s1['sequence'].id} "
                    f"stop or advancing seq {s2['sequence'].id} "
                    f"start."
                )
                gap_issue = {
                    "after_sequence": s1["sequence"].id,
                    "after_target": s1["sequence"].target,
                    "after_visit_id": s1["visit_id"],
                    "before_sequence": s2["sequence"].id,
                    "before_target": s2["sequence"].target,
                    "before_visit_id": s2["visit_id"],
                    "gap_start": s1["stop_time"],
                    "gap_end": s2["start_time"],
                    "gap_duration_minutes": gap_min,
                    "message": message,
                }
                issues["large_gaps"].append(gap_issue)

        # Generate summary
        issues["timing_summary"] = {
            "total_sequences": len(all_sequences),
            "overlaps_found": len(issues["overlaps"]),
            "short_sequences_found": len(issues["short_sequences"]),
            "large_gaps_found": len(issues["large_gaps"]),
            "total_issues": len(issues["overlaps"])
            + len(issues["short_sequences"])
            + len(issues["large_gaps"]),
        }

        # Report issues if requested
        if report_issues:
            self._print("\n" + "=" * 60)
            self._print("SEQUENCE TIMING VALIDATION REPORT")
            self._print("=" * 60)

            summary = issues["timing_summary"]
            self._print(
                f"Total sequences analyzed: " f"{summary['total_sequences']}"
            )
            self._print(f"Total timing issues found: " f"{summary['total_issues']}")
            self._print()

            if issues["overlaps"]:
                self._print(f"OVERLAPS ({len(issues['overlaps'])} found):")
                for i, ov in enumerate(issues["overlaps"]):
                    self._print(f"  {i+1}. {ov['message']}")
            else:
                self._print("\u2713 OVERLAPS: None found")

            self._print()

            if issues["short_sequences"]:
                self._print(
                    f"SHORT SEQUENCES "
                    f"({len(issues['short_sequences'])} found, "
                    f"< {min_dur_min:.0f} min):"
                )
                for i, sh in enumerate(issues["short_sequences"]):
                    self._print(f"  {i+1}. {sh['message']}")
            else:
                self._print("\u2713 SHORT SEQUENCES: None found")

            self._print()

            if issues["large_gaps"]:
                self._print(
                    f"LARGE GAPS ({len(issues['large_gaps'])} "
                    f"found, > 2 min):"
                )
                for i, gap in enumerate(issues["large_gaps"][:5]):
                    self._print(f"  {i+1}. {gap['message']}")
                if len(issues["large_gaps"]) > 5:
                    self._print(
                        f"     ... and "
                        f"{len(issues['large_gaps']) - 5} more"
                    )
            else:
                self._print("\u2713 LARGE GAPS: None found")

        return issues

    def validate_payload_exposures(
        self, calendar: ScienceCalendar, report_issues: bool = True
    ) -> List[Dict[str, Any]]:
        """Validate payload exposure times against sequence duration.

        Checks that single-frame exposure, total-frame exposure, and
        coadd exposure fit within the sequence duration *after*
        subtracting pre/post overheads.  Each issue dict includes a
        ``message`` with actionable detail and a ``suggested_fix``.

        Returns
        -------
        list of dict
            Issue dicts. Empty list when everything is valid.
        """
        issues = []

        # Compute effective overhead budget (max of VDA/NIRDA). A bare
        # processor without an OverheadTiming falls back to zero overhead.
        overhead = getattr(self, "overhead", None)
        if overhead is None:
            pre_oh_sec = 0.0
            post_oh_sec = 0.0
        else:
            pre_oh_sec = max(
                overhead.visda_pre_overhead_time.to(u.s).value,
                overhead.nirda_pre_overhead_time.to(u.s).value,
            )
            post_oh_sec = max(
                overhead.visda_post_overhead_time.to(u.s).value,
                overhead.nirda_post_overhead_time.to(u.s).value,
            )
        total_oh_sec = pre_oh_sec + post_oh_sec

        for visit in calendar.visits:
            for seq in visit.sequences:
                seq_dur_sec = seq.duration.sec
                effective_sec = seq_dur_sec - total_oh_sec

                # 1) VDA camera
                exposure_us = seq.get_payload_parameter(
                    "AcquireVisCamScienceData", "ExposureTime_us"
                )
                num_frames = seq.get_payload_parameter(
                    "AcquireVisCamScienceData",
                    "NumTotalFramesRequested",
                )
                frames_per_coadd = seq.get_payload_parameter(
                    "AcquireVisCamScienceData", "FramesPerCoadd"
                )

                if exposure_us is not None:
                    try:
                        exp_us_val = float(exposure_us)
                    except (ValueError, TypeError):
                        exp_us_val = None

                    if exp_us_val is not None:
                        single_sec = exp_us_val / 1e6
                        if single_sec > effective_sec:
                            msg = (
                                f"Seq {seq.id} ({seq.target}, "
                                f"visit {visit.id}): single VDA "
                                f"exposure {single_sec:.3f}s > "
                                f"effective duration "
                                f"{effective_sec:.1f}s "
                                f"(sequence {seq_dur_sec:.1f}s "
                                f"- overhead "
                                f"{total_oh_sec:.0f}s)."
                            )
                            issues.append(
                                {
                                    "visit_id": visit.id,
                                    "sequence_id": seq.id,
                                    "target": seq.target,
                                    "problem": (
                                        "single_exposure_longer"
                                        "_than_sequence"
                                    ),
                                    "exposure_seconds": single_sec,
                                    "sequence_duration_seconds": (seq_dur_sec),
                                    "effective_duration_seconds": (
                                        effective_sec
                                    ),
                                    "overhead_seconds": total_oh_sec,
                                    "suggested_fix": (
                                        f"Reduce ExposureTime_us "
                                        f"to <= "
                                        f"{int(effective_sec*1e6)}"
                                    ),
                                    "message": msg,
                                }
                            )
                            if report_issues:
                                self._print(msg)

                        if num_frames is not None:
                            try:
                                tf = int(num_frames)
                                tot_sec = (exp_us_val * tf) / 1e6
                                if tot_sec > effective_sec:
                                    max_f = int(
                                        effective_sec / (exp_us_val / 1e6)
                                    )
                                    msg = (
                                        f"Seq {seq.id} "
                                        f"({seq.target}, visit "
                                        f"{visit.id}): total VDA "
                                        f"exposure {tot_sec:.1f}s "
                                        f"({tf} frames) > "
                                        f"effective "
                                        f"{effective_sec:.1f}s. "
                                        f"Max frames that fit: "
                                        f"{max_f}."
                                    )
                                    issues.append(
                                        {
                                            "visit_id": visit.id,
                                            "sequence_id": seq.id,
                                            "target": seq.target,
                                            "problem": (
                                                "total_exposure_"
                                                "longer_than_"
                                                "sequence"
                                            ),
                                            "total_exposure_seconds": (
                                                tot_sec
                                            ),
                                            "sequence_duration_seconds": (
                                                seq_dur_sec
                                            ),
                                            "effective_duration_seconds": (
                                                effective_sec
                                            ),
                                            "overhead_seconds": (total_oh_sec),
                                            "suggested_max_frames": (max_f),
                                            "suggested_fix": (
                                                f"Set "
                                                f"NumTotalFrames"
                                                f"Requested "
                                                f"<= {max_f}"
                                            ),
                                            "message": msg,
                                        }
                                    )
                                    if report_issues:
                                        self._print(msg)
                            except (ValueError, TypeError):
                                pass

                        if num_frames is None and frames_per_coadd is not None:
                            try:
                                fpc = int(frames_per_coadd)
                                tot_sec = (exp_us_val * fpc) / 1e6
                                if tot_sec > effective_sec:
                                    msg = (
                                        f"Seq {seq.id} "
                                        f"({seq.target}, visit "
                                        f"{visit.id}): coadd "
                                        f"exposure {tot_sec:.1f}s "
                                        f"> effective "
                                        f"{effective_sec:.1f}s."
                                    )
                                    issues.append(
                                        {
                                            "visit_id": visit.id,
                                            "sequence_id": seq.id,
                                            "target": seq.target,
                                            "problem": (
                                                "coadd_exposure_"
                                                "longer_than_"
                                                "sequence"
                                            ),
                                            "coadd_exposure_seconds": (
                                                tot_sec
                                            ),
                                            "sequence_duration_seconds": (
                                                seq_dur_sec
                                            ),
                                            "effective_duration_seconds": (
                                                effective_sec
                                            ),
                                            "overhead_seconds": (total_oh_sec),
                                            "suggested_fix": (
                                                "Reduce "
                                                "FramesPerCoadd or "
                                                "ExposureTime_us"
                                            ),
                                            "message": msg,
                                        }
                                    )
                                    if report_issues:
                                        self._print(msg)
                            except (ValueError, TypeError):
                                pass

                # 2) Heuristic scan: any flattened key with 'exposure'
                flat = seq.get_flat_payload_parameters()
                for key, val in flat.items():
                    if "exposure" in key.lower() and val is not None:
                        if key.startswith("AcquireVisCamScienceData"):
                            continue
                        try:
                            v = float(val)
                        except (ValueError, TypeError):
                            continue

                        val_sec = v / 1e6 if key.lower().endswith("_us") else v

                        if val_sec > effective_sec:
                            msg = (
                                f"Seq {seq.id} ({seq.target}, "
                                f"visit {visit.id}): payload "
                                f"field {key} = {val_sec:.3f}s "
                                f"> effective "
                                f"{effective_sec:.1f}s."
                            )
                            issues.append(
                                {
                                    "visit_id": visit.id,
                                    "sequence_id": seq.id,
                                    "target": seq.target,
                                    "problem": (
                                        "payload_exposure_field_"
                                        "longer_than_sequence"
                                    ),
                                    "field": key,
                                    "value_seconds": val_sec,
                                    "sequence_duration_seconds": (seq_dur_sec),
                                    "effective_duration_seconds": (
                                        effective_sec
                                    ),
                                    "overhead_seconds": total_oh_sec,
                                    "suggested_fix": (
                                        f"Reduce {key} to fit "
                                        f"within "
                                        f"{effective_sec:.1f}s"
                                    ),
                                    "message": msg,
                                }
                            )
                            if report_issues:
                                self._print(msg)

        return issues

    def validate_star_roi_consistency(
        self, calendar: ScienceCalendar, report_issues: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Validate MaxNumStarRois/numPredefinedStarRois consistency.

        According to flight software requirements:
        - Method 0, 1, 3: MaxNumStarRois should equal numPredefinedStarRois
        - Method 2: numPredefinedStarRois should be 0, MaxNumStarRois should be > 0

        Parameters
        ----------
        calendar : ScienceCalendar
            The science calendar to validate.
        report_issues : bool, optional
            If True (default), issues are reported in the returned list. If False,
            the function still performs validation but does not print or log issues.

        Returns
        -------
        list of dict
            A list of issue dictionaries found. Each dictionary contains:
                - 'visit_id': The visit ID where the issue was found.
                - 'sequence_id': The sequence ID where the issue was found.
                - 'problem': A string describing the type of problem.
                - 'StarRoiDetMethod': The value of StarRoiDetMethod.
                - 'numPredefinedStarRois': The value of numPredefinedStarRois.
                - 'MaxNumStarRois': The value of MaxNumStarRois.
            Returns an empty list if no issues are found.

        Problem Types
        -------------
        The 'problem' key in each issue dict can have values such as:
            - "MaxNumStarRois != numPredefinedStarRois for method 0/1/3"
            - "numPredefinedStarRois != 0 for method 2"
            - "MaxNumStarRois <= 0 for method 2"

        Examples
        --------
        >>> issues = processor.validate_star_roi_consistency(calendar)
        >>> issues[0]
        {
            'visit_id': 'V001',
            'sequence_id': 'S001',
            'problem': 'MaxNumStarRois != numPredefinedStarRois for method 0/1/3',
            'star_roi_det_method': 1,
            'num_predefined': 3,
            'max_num': 2
        }
        """
        issues = []

        for visit in calendar.visits:
            for seq in visit.sequences:
                # Check AcquireVisCamScienceData payload
                star_roi_det_method = seq.get_payload_parameter(
                    "AcquireVisCamScienceData", "StarRoiDetMethod"
                )
                num_predefined = seq.get_payload_parameter(
                    "AcquireVisCamScienceData", "numPredefinedStarRois"
                )
                max_num = seq.get_payload_parameter(
                    "AcquireVisCamScienceData", "MaxNumStarRois"
                )

                # Parse StarRoiDetMethod (default to 2 if not present)
                method = 2
                if star_roi_det_method is not None:
                    try:
                        method = int(star_roi_det_method)
                    except (ValueError, TypeError):
                        method = 2

                # Validate based on method
                if method == 2:
                    # Method 2: numPredefinedStarRois should be 0
                    # and MaxNumStarRois should not be 0
                    if num_predefined is not None:
                        try:
                            num_predefined_val = int(num_predefined)
                            if num_predefined_val != 0:
                                issue = {
                                    "visit_id": visit.id,
                                    "sequence_id": seq.id,
                                    "target": seq.target,
                                    "problem": "numPredefinedStarRois_should_be_0_for_method_2",
                                    "StarRoiDetMethod": method,
                                    "numPredefinedStarRois": num_predefined_val,
                                }
                                issues.append(issue)
                                if report_issues:
                                    self._print(
                                        f"STAR ROI ISSUE: sequence {seq.id} "
                                        f"StarRoiDetMethod=2 but "
                                        f"numPredefinedStarRois={num_predefined_val} (should be 0)"
                                    )
                        except (ValueError, TypeError):
                            issue = {
                                "visit_id": visit.id,
                                "sequence_id": seq.id,
                                "target": seq.target,
                                "problem": "numPredefinedStarRois_not_parseable_as_integer",
                                "StarRoiDetMethod": method,
                                "numPredefinedStarRois": str(num_predefined),
                            }
                            issues.append(issue)
                            if report_issues:
                                self._print(
                                    f"STAR ROI ISSUE: sequence {seq.id} "
                                    f"numPredefinedStarRois='{num_predefined}' cannot be parsed as integer"
                                )
                    # Also check that MaxNumStarRois is not 0 for method 2
                    if max_num is not None:
                        try:
                            max_num_val = int(max_num)
                            if max_num_val == 0:
                                issue = {
                                    "visit_id": visit.id,
                                    "sequence_id": seq.id,
                                    "target": seq.target,
                                    "problem": "MaxNumStarRois_should_not_be_0_for_method_2",
                                    "StarRoiDetMethod": method,
                                    "MaxNumStarRois": max_num_val,
                                }
                                issues.append(issue)
                                if report_issues:
                                    self._print(
                                        f"STAR ROI ISSUE: sequence {seq.id} "
                                        f"StarRoiDetMethod=2 but "
                                        f"MaxNumStarRois={max_num_val} (should be > 0)"
                                    )
                        except (ValueError, TypeError):
                            issue = {
                                "visit_id": visit.id,
                                "sequence_id": seq.id,
                                "target": seq.target,
                                "problem": "MaxNumStarRois_not_parseable_as_integer",
                                "StarRoiDetMethod": method,
                                "MaxNumStarRois": str(max_num),
                            }
                            issues.append(issue)
                            if report_issues:
                                self._print(
                                    f"STAR ROI ISSUE: sequence {seq.id} "
                                    f"MaxNumStarRois='{max_num}' cannot be parsed as integer"
                                )
                else:
                    # Methods 0, 1, 3: MaxNumStarRois should equal numPredefinedStarRois
                    if num_predefined is not None and max_num is not None:
                        try:
                            num_predefined_val = int(num_predefined)
                            max_num_val = int(max_num)

                            if num_predefined_val != max_num_val:
                                issue = {
                                    "visit_id": visit.id,
                                    "sequence_id": seq.id,
                                    "target": seq.target,
                                    "problem": "MaxNumStarRois_not_equal_to_numPredefinedStarRois",
                                    "StarRoiDetMethod": method,
                                    "numPredefinedStarRois": num_predefined_val,
                                    "MaxNumStarRois": max_num_val,
                                }
                                issues.append(issue)
                                if report_issues:
                                    self._print(
                                        f"STAR ROI ISSUE: sequence {seq.id} "
                                        f"StarRoiDetMethod={method}, "
                                        f"MaxNumStarRois ({max_num_val}) != "
                                        f"numPredefinedStarRois ({num_predefined_val})"
                                    )
                        except (ValueError, TypeError):
                            # If we can't parse as integers, flag as an issue
                            issue = {
                                "visit_id": visit.id,
                                "sequence_id": seq.id,
                                "target": seq.target,
                                "problem": "star_roi_values_not_parseable_as_integers",
                                "StarRoiDetMethod": method,
                                "numPredefinedStarRois": str(num_predefined),
                                "MaxNumStarRois": str(max_num),
                            }
                            issues.append(issue)
                            if report_issues:
                                self._print(
                                    f"STAR ROI ISSUE: sequence {seq.id} "
                                    f"numPredefinedStarRois='{num_predefined}' or "
                                    f"MaxNumStarRois='{max_num}' cannot be parsed as integers"
                                )

        return issues

    def validate_roll_consistency(
        self,
        calendar: ScienceCalendar,
        report_issues: bool = True,
        tolerance_deg: float = 0.001,
    ) -> List[Dict[str, Any]]:
        """Validate roll-angle consistency per target within each visit.

        Returns
        -------
        list of dict
            Issue dicts with ``message``, ``suggested_roll``, and
            per-sequence ``roll_map``.
        """
        issues = []

        for visit in calendar.visits:
            target_sequences: Dict[str, List[ObservationSequence]] = {}
            for seq in visit.sequences:
                if seq.target not in target_sequences:
                    target_sequences[seq.target] = []
                target_sequences[seq.target].append(seq)

            for target, sequences in target_sequences.items():
                if len(sequences) < 2:
                    continue

                roll_values = []
                seq_ids = []
                roll_map: Dict[str, float] = {}
                for seq in sequences:
                    if seq.roll is not None:
                        roll_values.append(seq.roll)
                        seq_ids.append(seq.id)
                        roll_map[seq.id] = seq.roll

                if len(roll_values) < 2:
                    continue

                sorted_rolls = sorted(roll_values)
                gaps = [
                    sorted_rolls[i + 1] - sorted_rolls[i]
                    for i in range(len(sorted_rolls) - 1)
                ]
                gaps.append(360.0 - (sorted_rolls[-1] - sorted_rolls[0]))
                max_diff = 360.0 - max(gaps)

                if max_diff > tolerance_deg:
                    suggested = float(np.median(roll_values))
                    msg = (
                        f"Visit {visit.id}, target {target}: "
                        f"roll spread {max_diff:.3f}° across "
                        f"{len(seq_ids)} sequences. "
                        f"Values: "
                        f"{[f'{r:.2f}' for r in roll_values]}. "
                        f"Suggest setting all to "
                        f"{suggested:.2f}°."
                    )
                    issues.append(
                        {
                            "visit_id": visit.id,
                            "target": target,
                            "sequence_ids": seq_ids,
                            "roll_values": roll_values,
                            "roll_map": roll_map,
                            "max_difference_deg": max_diff,
                            "suggested_roll": suggested,
                            "suggested_fix": (
                                f"Set roll to {suggested:.2f}° "
                                f"for all {target} sequences "
                                f"in visit {visit.id}"
                            ),
                            "message": msg,
                        }
                    )
                    if report_issues:
                        self._print(msg)

        return issues

    def _print_issue_details(
        self, category: str, item: Dict[str, Any]
    ) -> None:
        """Print structured requirement-vs-actual detail for one issue."""
        indent = "      "

        if category == "visibility":
            details = item.get("constraint_details", {})
            if details:
                for body, info in details.items():
                    status = "PASS" if info["passes"] else "FAIL"
                    side = info.get("side", "")
                    side_label = f" [{side}]" if side else ""
                    self._print(
                        f"{indent}{body:<12} {status}  "
                        f"required: >= {info['required_deg']:.1f}°"
                        f"{side_label}  "
                        f"actual: {info['actual_deg']:.1f}°"
                    )
            frac = item.get("visibility_fraction")
            nv = item.get("non_visible_minutes")
            tot = item.get("total_minutes")
            if frac is not None:
                self._print(
                    f"{indent}{'visibility':<12}       "
                    f"required: 100%  "
                    f"actual: {frac:.1%}  "
                    f"({nv}/{tot} min non-visible)"
                )

        elif category == "short_sequences":
            dur = item.get("duration_minutes")
            req = item.get("minimum_required_minutes")
            if dur is not None and req is not None:
                self._print(
                    f"{indent}duration     "
                    f"required: >= {req:.0f} min  "
                    f"actual: {dur:.1f} min  "
                    f"(short by {req - dur:.1f} min)"
                )

        elif category == "large_gaps":
            gap = item.get("gap_duration_minutes")
            if gap is not None:
                self._print(
                    f"{indent}gap          "
                    f"required: <= 2.0 min  "
                    f"actual: {gap:.1f} min  "
                    f"(over by {gap - 2.0:.1f} min)"
                )

        elif category == "overlaps":
            ov = item.get("overlap_duration_minutes")
            if ov is not None:
                self._print(
                    f"{indent}overlap      "
                    f"required: 0.0 min  "
                    f"actual: {ov:.1f} min"
                )

        elif category == "payload_exposure":
            seq_dur = item.get("sequence_duration_seconds")
            eff_dur = item.get("effective_duration_seconds")
            oh = item.get("overhead_seconds")
            if seq_dur is not None:
                self._print(
                    f"{indent}sequence     "
                    f"{seq_dur:.0f}s total  "
                    f"- {oh:.0f}s overhead  "
                    f"= {eff_dur:.0f}s effective"
                )
            if "exposure_seconds" in item:
                exp = item["exposure_seconds"]
                self._print(
                    f"{indent}single exp   "
                    f"required: <= {eff_dur:.0f}s  "
                    f"actual: {exp:.3f}s"
                )
            if "total_exposure_seconds" in item:
                tot = item["total_exposure_seconds"]
                max_f = item.get("suggested_max_frames", "?")
                self._print(
                    f"{indent}total exp    "
                    f"required: <= {eff_dur:.0f}s  "
                    f"actual: {tot:.1f}s  "
                    f"(max frames: {max_f})"
                )
            if "coadd_exposure_seconds" in item:
                coadd = item["coadd_exposure_seconds"]
                self._print(
                    f"{indent}coadd exp    "
                    f"required: <= {eff_dur:.0f}s  "
                    f"actual: {coadd:.1f}s"
                )
            if "value_seconds" in item:
                val = item["value_seconds"]
                field = item.get("field", "?")
                self._print(
                    f"{indent}{field}  "
                    f"required: <= {eff_dur:.0f}s  "
                    f"actual: {val:.3f}s"
                )

        elif category == "roll_consistency":
            spread = item.get("max_difference_deg")
            suggested = item.get("suggested_roll")
            if spread is not None:
                self._print(
                    f"{indent}roll spread  "
                    f"required: <= 0.001°  "
                    f"actual: {spread:.3f}°  "
                    f"(suggest: {suggested:.2f}°)"
                )

        elif category == "target_name":
            tgt = item.get("target", "")
            if tgt:
                self._print(
                    f"{indent}target name  "
                    f"required: no spaces  "
                    f"actual: '{tgt}'"
                )

    def print_validation_summary(
        self, calendar: ScienceCalendar
    ) -> Dict[str, Any]:
        """Run all validators and print a unified actionable report.

        Returns
        -------
        dict
            ``{"status": "VALID"|"INVALID", "counts": {...},
            "details": {...}}`` where *details* maps each category
            to the raw issue list.
        """
        results: Dict[str, Any] = {}
        counts: Dict[str, int] = {}

        # --- target names ---
        target_issues = self.validate_target_names(
            calendar, report_issues=False
        )
        if target_issues:
            results["target_name"] = target_issues
            counts["target_name"] = len(target_issues)

        # --- visibility ---
        vis_issues = self.validate_visibility(calendar, report_issues=False)
        if vis_issues:
            results["visibility"] = vis_issues
            counts["visibility"] = len(vis_issues)

        # --- payload exposures ---
        payload_issues = self.validate_payload_exposures(
            calendar, report_issues=False
        )
        if payload_issues:
            results["payload_exposure"] = payload_issues
            counts["payload_exposure"] = len(payload_issues)

        # --- overlaps ---
        overlap_issues = self.validate_no_overlaps_astropy(
            calendar, report_issues=False
        )
        if overlap_issues:
            results["overlap"] = overlap_issues
            counts["overlap"] = len(overlap_issues)

        # --- sequence timing ---
        timing_result = self.validate_sequence_timing(
            calendar, report_issues=False
        )
        timing_total = timing_result["timing_summary"]["total_issues"]
        if timing_total > 0:
            results["sequence_timing"] = timing_result
            counts["sequence_timing"] = timing_total

        # --- roll consistency ---
        roll_issues = self.validate_roll_consistency(
            calendar, report_issues=False
        )
        if roll_issues:
            results["roll_consistency"] = roll_issues
            counts["roll_consistency"] = len(roll_issues)

        total = sum(counts.values())
        status = "VALID" if total == 0 else "INVALID"

        # ── Print ──
        self._print(
            f"\n{'=' * 60}\n"
            f"  VALIDATION SUMMARY: {status} "
            f"({total} issues)\n"
            f"{'=' * 60}"
        )

        if total == 0:
            self._print("  All checks passed.\n")
            return {
                "status": status,
                "counts": counts,
                "details": results,
            }

        for cat, cnt in counts.items():
            self._print(f"\n  [{cat.upper()}] — {cnt} issue(s)")
            items = results[cat]

            # Sequence timing has a nested structure
            if cat == "sequence_timing":
                for sub_key in [
                    "overlaps",
                    "short_sequences",
                    "large_gaps",
                ]:
                    for item in items.get(sub_key, []):
                        msg = item.get("message", "")
                        if msg:
                            self._print(f"    • {msg}")
                        self._print_issue_details(sub_key, item)
                continue

            # All other categories are plain lists
            if isinstance(items, list):
                for item in items:
                    msg = item.get("message", "")
                    if msg:
                        self._print(f"    • {msg}")
                    self._print_issue_details(cat, item)

        self._print(f"\n{'=' * 60}\n")
        return {
            "status": status,
            "counts": counts,
            "details": results,
        }

    def print_timing_summary(self, calendar: ScienceCalendar) -> None:
        """Print a quick timing summary."""
        issues = self.validate_sequence_timing(calendar, report_issues=False)
        summary = issues["timing_summary"]

        if summary["total_issues"] == 0:
            self._print("✓ All sequence timing validation checks passed")
        else:
            self._print(f"✗ Found {summary['total_issues']} timing issues:")
            if summary["overlaps_found"]:
                self._print(f"  - {summary['overlaps_found']} overlaps")
            if summary["short_sequences_found"]:
                self._print(
                    f"  - {summary['short_sequences_found']} sequences too short"
                )
            if summary["large_gaps_found"]:
                self._print(f"  - {summary['large_gaps_found']} large gaps")


def _find_false_blocks(vis_bool, time_grid, return_index=False):
    """Return a list of contiguous (start, stop) times for False regions."""
    if len(vis_bool) == 0:
        return []

    blocks = []
    idx = []
    in_block = False
    block_start_idx = None

    for i, v in enumerate(vis_bool):
        if not v and not in_block:
            # Start of a False block
            block_start_idx = i
            in_block = True
        elif v and in_block:
            # End of a False block
            t_start = time_grid[block_start_idx]
            t_stop = time_grid[
                i
            ]  # or time_grid[i-1] + 1*u.min if you want to extend
            blocks.append((t_start, t_stop))
            idx.append((block_start_idx, i))
            in_block = False

    # Handle case where array ends in a False block
    if in_block and block_start_idx is not None:
        t_start = time_grid[block_start_idx]
        # Option 1: Use last time point
        t_stop = time_grid[-1]
        # Option 2: Extend past end (if this is your intended behavior)
        # t_stop = time_grid[-1] + 1 * u.min

        blocks.append((t_start, t_stop))
        idx.append((block_start_idx, len(vis_bool)))  # More consistent than -1

    if return_index:
        return blocks, idx
    else:
        return blocks
