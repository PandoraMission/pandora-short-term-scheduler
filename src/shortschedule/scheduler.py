"""Schedule processing utilities.

This module implements the ScheduleProcessor which is responsible for
adjusting a `ScienceCalendar` when updated spacecraft ephemerides (TLEs)
are provided. The processor performs these high-level steps:

- extract a time window to process
- compute minute-by-minute visibility using `pandoravisibility.Visibility`
- trim each observation back to the visible time it actually has, then
  grow it into the surrounding idle time while the target stays visible
- hold every boundary within `max_movement_minutes` of its long-term time
  and guarantee the delivered calendar contains no overlaps
- update payload integration parameters (VIS/NIR) to fit the new timing
- assemble a report of what the passes changed

Idle time between observations is expected: the scheduler adjusts each
observation around the time the long-term calendar chose for it rather
than sliding observations to close gaps.
"""

# Standard library
import copy
import logging
import re
import uuid
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime, timedelta

from .models import (  # BAD_NAME_SYMBOLS re-exported for existing imports
    BAD_NAME_SYMBOLS,  # noqa: F401
    ObservationSequence,
    ScienceCalendar,
    Visit,
    clean_target_name,
)
from .nirda import NirdaData
from .overhead import OverheadTiming
from .roll import get_best_roll_per_visit
from .transit import (
    load_ephemerides,
    planet_key,
    transit_core,
    transit_epoch,
)
from .visda import VisdaData

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - zoneinfo is stdlib on 3.9+
    ZoneInfo = None
from pathlib import Path
from typing import Any

# Third-party
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
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


class _ErrorCountingHandler(logging.Handler):
    """Counts ERROR records emitted during a processing run.

    The delivered calendar is marked invalid on the strength of this
    count. Warnings do not contribute: the gap tolerances mean a healthy
    calendar legitimately contains keepout violations, and reporting them
    is not the same as the run having failed.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1


class ScheduleProcessor:
    """Main class for processing and adjusting science calendars with updated TLE.

    Public methods
    --------------
    - process_calendar(calendar, window_start=None, window_duration_days=21, verbose=False)
        Process a calendar and return an updated ScienceCalendar.
    - get_gap_report()
        Return a structured report summarizing visibility gaps and actions taken.

    The class expects `Visibility(tle1, tle2)` to offer
    `get_visibility(coord, times, roll=...)` returning the `pandoravisibility`
    v2.0.0 dict, whose `visible` entry is a boolean array of the same length
    as `times`, and to run the roll search when given `optimize_roll=True`.
    """

    @staticmethod
    def _to_deg(
        value: float | None,
    ) -> u.Quantity | None:
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
        override_nirda_parameters: dict[int, dict[str, Any]] | None = None,
        override_visda_parameters: dict[int, dict[str, Any]] | None = None,
        override_payload_parameters: (
            dict[Any, dict[str, dict[str, Any]]] | None
        ) = None,
        max_file_size_uncompressed: u.Quantity = 830.0 * 1000 * 1000 * u.byte,
        max_file_size_compressed: u.Quantity = 255.0 * 1000 * 1000 * u.byte,
        update_nirda_reset1_for_vitl: bool = True,
        vitl_settling_time: u.Quantity = 60.0 * u.s,
        convert_single_roi_to_predefined: bool = True,
        fix_bad_data: bool = True,
        moon_min: float | None = None,
        sun_min: float | None = None,
        earthlimb_min: float | None = None,
        earthlimb_day_min: float | None = None,
        earthlimb_night_min: float | None = None,
        occultation_earthlimb_min: float | None = None,
        occultation_overrides: list[str] | None = None,
        mars_min: float | None = None,
        jupiter_min: float | None = None,
        st_sun_min: float | None = None,
        st_moon_min: float | None = None,
        st_earthlimb_min: float | None = None,
        st1_earthlimb_min: float | None = None,
        st2_earthlimb_min: float | None = None,
        roll_step: float = 1.0,
        min_power_frac: float = 0.7,
        max_movement_minutes: int = 45,
        grow_by_priority: bool = True,
        drop_priority0: bool = False,
        refine_edges_to_second: bool = True,
        earthlimb_gap_tolerance: int = 0,
        earthlimb_gap_tolerance_start_buffer: int = 7.5,
        st_gap_tolerance: int = 0,
        st_gap_tolerance_start_buffer: int = 7.5,
        use_dynamic_earthlimb: bool | None = None,
        use_trace_sweep: bool | None = None,
        trace_contamination_tol: float | None = None,
        nirda_roi_x: int | None = None,
        nirda_roi_y: int | None = None,
        drop_observations_based_on_trace: bool = False,
        target_list_dir: str | None = None,
        transit_middle_frac: float = 0.75,
        transit_duration_constraint_min: float = 15.0,
        add_high_pri_observations: bool = False,
        high_priority_adds_earthlimb_min: float | None = None,
        high_priority_adds_min_duration: float = 10.0,
        high_priority_adds_max_duration: float = 90.0,
        high_priority_adds_min_observations: int = 5,
        high_priority_adds_transit_buffer_min: float = 40.0,
        high_priority_adds_exclude: list[str] | None = None,
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
            Every keepout defaults to ``None`` deferring to ``pandoravisibility``
            as the single source of truth for keepout defaults, so they are
            deliberately not restated here.
        earthlimb_day_min : float, optional
            Earth-limb keepout angle (degrees) on the **day** side of the
            terminator.  When ``None`` (default), ``earthlimb_min`` is used
            for both day and night sides (``Visibility`` default behaviour).
        earthlimb_night_min : float, optional
            Earth-limb keepout angle (degrees) on the **night** side of the
            terminator.  When ``None`` (default), ``earthlimb_min`` is used
            for both day and night sides (``Visibility`` default behaviour).
        occultation_earthlimb_min : float, optional
            Earth-limb keepout angle (degrees) for every target that is not
            its visit's main target, so those observations can be held
            further off the Earth and dissipate more heat. The main target
            always flies the standard keepouts, whatever its priority.
            Each visit's main target is its priority 1/2 target; failing
            that, its non-Gaia target with the most minutes. A visit with
            two different priority 1/2 targets, or only Gaia targets, has
            no main target, so all its targets fly this angle.
            It is a flat angle: the day/night split and the dynamic Earth
            illumination wedge do not apply. Every other keepout, including
            all the star-tracker ones, is unchanged. When ``None``
            (default), every target flies the standard keepouts.
        occultation_overrides : list of str, optional
            Target names that fly the standard keepouts even when they are
            not their visit's main target, so ``occultation_earthlimb_min``
            never applies to them. A name matches as written or with its
            invalid symbols replaced (``+`` and space -> ``_``). Only the
            keepout changes: ``grow_by_priority`` still ranks an override
            by its priority, behind every main target. A name found in no
            visit of the long-term calendar is a warning. Default ``None``.
        st_sun_min, st_moon_min, st_earthlimb_min, st1_earthlimb_min,
        st2_earthlimb_min : float, optional
            Additional constraints for star trackers.
        roll_step : float, optional
            Roll-angle sweep resolution in degrees (default 1.0).
        min_power_frac : float, optional
            Minimum acceptable solar-panel power fraction (0-1).
            Roll candidates below this are rejected (default 0.7).
        max_movement_minutes : int, optional
            Furthest either boundary of an observation may end up from where
            the long-term calendar put it, in minutes (default 45). The
            short-term scheduler adjusts for a stale TLE, so an observation
            is expected to shift by a few minutes; a shift of an orbit or
            more means there is either a problem in the visibility calculation
            or that the long-term scheduler should be adjusted. In either
            case it requires further SOC investigation and should not be
            allowed by the short-term scheduler. Set to 0 to disable clamping.
        grow_by_priority : bool, optional
            Grow by rank: priority 2, then 1, then 0, and at equal priority
            a visit's main target before its other targets (so within a
            visit the main target grows first). Let an observation take
            minutes from an adjacent lower-ranked one, or an equal-ranked
            one in a later visit, as long as that one keeps the minimum
            duration and stays within ``max_movement_minutes`` (default
            True). False grows in start-time order with every neighbor a
            hard bound.
        drop_priority0 : bool, optional
            With ``grow_by_priority``, let an observation take as much of
            an adjacent lower-ranked priority-0 one as its own visibility,
            gap tolerances and movement limit allow, past that priority 0's
            floor. A priority 0 left unable to fly (under
            ``min_sequence_duration`` once its opening is cleaned, past its
            movement limit, or eaten whole) is removed and logged as a
            warning. Merging also drops priority 0s that are all that
            separates two otherwise mergeable priority-1/2 observations
            (default False).
        refine_edges_to_second : bool, optional
            Every other pass judges visibility once a minute, so a target
            can leave the keepouts anywhere inside an observation's last
            minute, or still be visible for part of the minute past a
            boundary. With this on, a final pass rechecks the minute either
            side of each start and stop at 1 s, at the roll the observation
            flies, and moves the boundary to the second: in, where the
            target leaves visibility inside the observation's first or last
            minute, and out, into free time the target is visible for,
            within the neighbors and ``max_movement_minutes``. A cut under
            ``min_sequence_duration`` is logged as a warning, and with
            ``drop_priority0`` a priority 0 cut that short is dropped
            (default True).
        earthlimb_gap_tolerance : int, optional
            Maximum number of contiguous minutes of earth-limb
            visibility violations to tolerate within a sequence
            (default 0).  Short dips are kept; longer gaps trigger
            trimming.  A dip this short also does not stop an observation
            growing through it into the visible time beyond.
        earthlimb_gap_tolerance_start_buffer : int, optional
            Minutes at the beginning of every observation that must be
            clear of the boresight Earth-limb keepout, measured from its
            start time (default 12).  ``earthlimb_gap_tolerance`` lets a
            dip be tolerated mid-observation, but an observation that
            opens with the boresight in the Earth limb may not acquire VITL,
            so its not worth starting, therefore no tolerance is applied
            inside this buffer. Set to 0 to disable.
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
        use_dynamic_earthlimb : bool, optional
            If True, uses the dynamic DPC boresight Earth limb, the wedge
            shaped keepout based on the Earth illumination angle.  Like
            every keepout above it defaults to ``None``, deferring to
            ``pandoravisibility``, which, currently, defaults to
            switching the wedge on. Passing ``False`` selects the
            flat or day/night limb instead.
        use_trace_sweep : bool, optional
            If True, the roll search rejects rolls where neighbor stars'
            NIRDA traces contaminate the target's spectrum by more than
            ``trace_contamination_tol``, taking the best ranked roll that
            passes. A target that some roll could see but no such roll
            passes is handled by ``drop_observations_based_on_trace``.
            Needs the gaiaoffline database. ``None`` (default) defers to
            ``pandoravisibility``, which currently switches it on.
        trace_contamination_tol : float, optional
            Largest acceptable trace contamination: the worst extraction
            row's neighbor flux over the target's, within the roll
            uncertainty. ``None`` (default) defers to ``pandoravisibility``
            (currently 1e-3).
        nirda_roi_x, nirda_roi_y : int, optional
            NIRDA ROI size in pixels (across and along the dispersion) that
            clips which neighbor light the trace check counts. ``None``
            (default) defers to ``pandoravisibility`` (currently 80 x 256).
        drop_observations_based_on_trace : bool, optional
            What happens to a target in a visit that some roll could see
            but where every such roll fails the trace check. True drops
            its observations from that visit (and the visit if that empties
            it), logged as a warning, before trimming and growth so
            neighbors can use the time. False (default) keeps them at the
            best ranked roll, trace failure and all, logged as an error.
        target_list_dir : str or Path, optional
            A PandoraTargetList checkout (or its ``target_definition_files``
            directory) whose transit ephemerides switch on the transit
            constraint for priority-2 observations; a planet defined in
            more than one category takes its most precise ephemeris (see
            ``transit.load_ephemerides``). ``None`` (default) leaves it off.
        transit_middle_frac : float, optional
            Share of a transit's total duration, centered on mid-transit,
            that forms its core (default 0.75).
        transit_duration_constraint_min : float, optional
            Visible minutes of a transit's core its priority-2 observations
            must keep between them, at the rolls they fly (default 15). A
            pass that would shrink one of them below this, or further
            below it for a transit the long-term calendar already left
            short, is refused; overlap repair still applies but is logged
            as an error. 0 turns the constraint off.
        add_high_pri_observations : bool, optional
            After the final merge, fill idle time with extra observations
            of the window's priority 1/2 targets (default False). Each
            addition joins an existing visit, so the calendar stays in time
            order: the visit it falls in, or for a gap between two visits,
            the one most of the gap's additions sit closer to. An addition
            flies its target's roll in that visit if the visit already has
            the target, otherwise the roll of the target's visit with the
            most minutes of it, and the payload of its priority 1/2
            observation there. Additions cover only idle minutes where the
            target is visible at that roll every minute; where targets
            compete, one in its padded
            transit window wins, then the longest stretch. An addition is
            priority 1 when it overlaps the planet's T1-T4 widened by
            ``high_priority_adds_transit_buffer_min`` on each side, else
            priority 0 (always 0 without ``target_list_dir``). The transit
            core constraint does not apply to additions.
        high_priority_adds_earthlimb_min : float, optional
            Flat Earth-limb keepout (degrees) for the additions, with every
            other keepout standard. ``None`` (default) uses the standard
            keepouts.
        high_priority_adds_min_duration : float, optional
            Shortest addition in minutes (default 10).
        high_priority_adds_max_duration : float, optional
            Longest addition in minutes (default 90); a longer free,
            visible stretch is split.
        high_priority_adds_min_observations : int, optional
            A target with fewer additions in the window gets none, and its
            time is offered to the other targets (default 5).
        high_priority_adds_transit_buffer_min : float, optional
            Minutes added before T1 and after T4 for the priority-1 window
            (default 40).
        high_priority_adds_exclude : list of str, optional
            Target names never given additions. A name matches as written
            or with its invalid symbols replaced (``+`` and space -> ``_``);
            one that is not a priority 1/2 target in the window is a
            warning. Default ``None``.
        """
        # Validate TLE format
        if not isinstance(tle_line1, str):
            raise ValueError("Invalid TLE line 1 format")
        if not isinstance(tle_line2, str):
            raise ValueError("Invalid TLE line 2 format")
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2

        _kw: dict[str, Any] = dict(
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
            earthlimb_day_min=self._to_deg(earthlimb_day_min),
            earthlimb_night_min=self._to_deg(earthlimb_night_min),
            use_dynamic_earthlimb=use_dynamic_earthlimb,
            use_trace_sweep=use_trace_sweep,
            trace_contamination_tol=trace_contamination_tol,
            nirda_roi_x=nirda_roi_x,
            nirda_roi_y=nirda_roi_y,
        )
        # Strip None entries so Visibility uses its own class-level defaults
        # for any constraint the caller left unset.
        _kw = {k: v for k, v in _kw.items() if v is not None}
        self.visibility = Visibility(tle_line1, tle_line2, **_kw)

        # Targets other than their visit's main target may be held further
        # off the Earth limb so the spacecraft can dissipate more heat.
        # Every other keepout, the star trackers included, is unchanged
        def _flat_earthlimb_model(angle):
            flat_kw = dict(_kw)
            # Explicit None, not a pop: dropping the key lets Visibility
            # fall back to its own day/night defaults, which since v1.3.0
            # are real angles rather than None. Those would then override
            # the flat angle the caller asked for.
            flat_kw["earthlimb_day_min"] = None
            flat_kw["earthlimb_night_min"] = None
            flat_kw["use_dynamic_earthlimb"] = False
            flat_kw["earthlimb_min"] = self._to_deg(angle)
            return Visibility(tle_line1, tle_line2, **flat_kw)

        self.occultation_earthlimb_min = occultation_earthlimb_min
        self.occultation_visibility = (
            None
            if occultation_earthlimb_min is None
            else _flat_earthlimb_model(occultation_earthlimb_min)
        )

        # High-priority additions in free time, with their own flat limb.
        self.add_high_pri_observations = add_high_pri_observations
        self.high_priority_adds_earthlimb_min = (
            high_priority_adds_earthlimb_min
        )
        self.high_priority_adds_visibility = (
            self.visibility
            if high_priority_adds_earthlimb_min is None
            else _flat_earthlimb_model(high_priority_adds_earthlimb_min)
        )
        self.high_priority_adds_min_duration = high_priority_adds_min_duration
        self.high_priority_adds_max_duration = high_priority_adds_max_duration
        self.high_priority_adds_min_observations = (
            high_priority_adds_min_observations
        )
        self.high_priority_adds_transit_buffer_min = (
            high_priority_adds_transit_buffer_min
        )
        if isinstance(high_priority_adds_exclude, str):
            high_priority_adds_exclude = [high_priority_adds_exclude]
        self.high_priority_adds_exclude = (
            None
            if high_priority_adds_exclude is None
            else [str(name).strip() for name in high_priority_adds_exclude]
        )
        # Targets that keep the standard keepouts though not main. Held
        # cleaned, and every target is cleaned before it is compared, so
        # either spelling matches before or after _normalize_target_names.
        if isinstance(occultation_overrides, str):
            occultation_overrides = [occultation_overrides]
        self.occultation_overrides = (
            None
            if occultation_overrides is None
            else [str(name).strip() for name in occultation_overrides]
        )
        self._occultation_override_names = frozenset(
            self._clean_name(name) for name in self.occultation_overrides or ()
        )

        self.min_sequence_duration = TimeDelta(8 * 60 * u.s)

        # Furthest either boundary may drift from its long-term time.
        self.max_movement_minutes = max_movement_minutes
        self.grow_by_priority = grow_by_priority
        self.drop_priority0 = drop_priority0
        self.refine_edges_to_second = refine_edges_to_second

        # Gap tolerance: maximum contiguous non-visible minutes to allow
        self.earthlimb_gap_tolerance = earthlimb_gap_tolerance
        self.earthlimb_gap_tolerance_start_buffer = (
            earthlimb_gap_tolerance_start_buffer
        )
        self.st_gap_tolerance = st_gap_tolerance
        self.st_gap_tolerance_start_buffer = st_gap_tolerance_start_buffer

        # Roll sweep configuration
        self.roll_step = roll_step
        self.min_power_frac = min_power_frac
        self.drop_observations_based_on_trace = (
            drop_observations_based_on_trace
        )

        # Priority-2 transit constraint; planet key to ephemeris, empty
        # (constraint off) without a PandoraTargetList.
        self.transit_middle_frac = transit_middle_frac
        self.transit_duration_constraint_min = transit_duration_constraint_min
        self.transit_ephemerides = (
            {}
            if target_list_dir is None
            else load_ephemerides(target_list_dir)
        )

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
        self._override_nirda_parameters: dict[int, Any] = (
            override_nirda_parameters or {}
        )
        self._override_visda_parameters: dict[int, Any] = (
            override_visda_parameters or {}
        )
        # General per-priority XML-tag payload overrides (cleaner config.json
        # format). Normalized to int priority keys.
        self._override_payload_parameters: dict[int, Any] = (
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
        self.convert_single_roi_to_predefined = (
            convert_single_roi_to_predefined
        )

        # Bad-data fixes: when enabled, target name fields have invalid
        # symbols (see BAD_NAME_SYMBOLS) replaced, and all other fields are
        # scanned for NaN-like values (reported as warnings).
        self.fix_bad_data = fix_bad_data

        # Before/after tracking of what the processing passes did. Built by
        # the same routine that resets it per run, so the two cannot drift.
        self._initialize_gap_report()

    def process_calendar(
        self,
        calendar: ScienceCalendar,
        window_start: Any | None = None,
        window_duration_days: int = 21,
        merge_similar_observations: bool = True,
        log_path: Any | None = None,
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
            Number of days to include in the processing window. The window
            is a hard boundary: an observation the long-term calendar runs
            across either edge is cut at it, growth stops at it (with a
            warning when the target would have stayed visible past it), and
            nothing in the returned calendar reaches outside it.
        merge_similar_observations : bool, optional
            When True, adjacent observation sequences in the same visit that
            share the same target and pointing are merged into a single
            longer sequence. This happens twice: on the long-term times
            before any trimming, so a dark observation merged into a
            visible one across Free Time has its dark time trimmed, and on
            the final times at the end.
            Defaults to True.
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

        # Pick each visit's main target from the long-term calendar, before
        # the window cuts any of its minutes.
        main_targets = self._mark_main_targets(calendar)

        # Extract windowed calendar FIRST. Every observation in it is a copy,
        # so nothing below changes the caller's long-term calendar.
        windowed_calendar = self._extract_time_window(
            calendar, window_start, window_duration_days, verbose
        )

        # Flag the main target on the copies; the flag rides on every later
        # copy. Set before names are cleaned, while they still match.
        for visit in windowed_calendar.visits:
            for seq in visit.sequences:
                seq.is_main = seq.target in main_targets.get(visit.id, ())

        # Normalize target names up front, so every pass and log line sees
        # the name the delivered calendar will carry.
        if getattr(self, "fix_bad_data", False):
            self._normalize_target_names(windowed_calendar, verbose)

        # Capture windowed calendar statistics (not original full calendar)
        self._analyze_original_calendar(
            windowed_calendar
        )  # Use windowed version

        # Analyze original visibility gaps in the windowed calendar
        self._analyze_original_visibility(windowed_calendar, verbose)

        # Process sequences. With merging on, same-target observations are
        # first merged on their long-term times, before the trims.
        processed_calendar = self._process_all_sequences(
            windowed_calendar,
            verbose,
            merge_first=merge_similar_observations,
        )

        # Optionally merge back-to-back same-target sequences within each
        # visit. This second merge runs *after* all gap-filling, trimming,
        # and other duration/timing adjustments so it operates on the final
        # scheduled boundaries.
        if merge_similar_observations:
            processed_calendar = self._merge_similar_observations(
                processed_calendar, verbose
            )

        # Extra observations of the week's priority 1/2 targets in the idle
        # time left once everything above has settled.
        if getattr(self, "add_high_pri_observations", False):
            processed_calendar = self._add_high_priority_observations(
                processed_calendar
            )

        # Payload overrides, then NIRDA SC_Integrations and VISDA
        # NumTotalFramesRequested from each observation's duration. Done
        # once, here, after the last change to any start or stop (the final
        # merge), so the counts, the data volumes and the .diag report all
        # match what is delivered.
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

        # The validators are run for the report, not for the verdict. Under
        # the gap tolerances a healthy calendar legitimately contains
        # keepout violations, and merging can absorb one deliberately.
        validation_counts: dict[str, int] = {}

        target_issues = self.validate_target_names(
            processed_calendar, report_issues=False
        )
        if target_issues:
            validation_counts["target_name"] = len(target_issues)

        vis_issues = self.validate_visibility(
            processed_calendar, report_issues=False
        )
        if vis_issues:
            validation_counts["visibility"] = len(vis_issues)

        payload_issues = self.validate_payload_exposures(
            processed_calendar, report_issues=False
        )
        if payload_issues:
            validation_counts["payload_exposure"] = len(payload_issues)

        overlap_issues = self.validate_no_overlaps_astropy(
            processed_calendar, report_issues=False
        )
        if overlap_issues:
            validation_counts["overlap"] = len(overlap_issues)

        timing_result = self.validate_sequence_timing(
            processed_calendar, report_issues=False
        )
        timing_total = timing_result["timing_summary"]["total_issues"]
        if timing_total > 0:
            validation_counts["sequence_timing"] = timing_total

        roll_issues = self.validate_roll_consistency(
            processed_calendar, report_issues=False
        )
        if roll_issues:
            validation_counts["roll_consistency"] = len(roll_issues)

        error_count = self.run_error_count
        calendar_status = "INVALID" if error_count else "VALID"

        # Print compact validation summary
        if validation_counts:
            self._print(
                f"\n--- Validation: {calendar_status} "
                f"({sum(validation_counts.values())} issues, "
                f"{error_count} error(s)) ---"
            )
            for cat, cnt in validation_counts.items():
                self._print(f"  {cat}: {cnt}")
            self._print(
                "Run print_validation_summary(calendar) "
                "for actionable details.\n"
            )
        else:
            self._print(
                f"\n--- Validation: {calendar_status} "
                f"(0 issues, {error_count} error(s)) ---\n"
            )

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
                "scheduler_settings": self._settings_for_header(),
            }
        )

        # Attach updated metadata to the processed calendar
        processed_calendar.metadata = new_metadata

        return processed_calendar

    def _mark_main_targets(
        self, calendar: ScienceCalendar
    ) -> dict[str, list[str]]:
        """Choose each visit's main targets.

        A visit's main target is its priority 1/2 target. With none, it is
        the target with the most minutes in the visit, skipping Gaia
        targets (``G`` followed by digits) with a warning when one of them
        has the most. A visit with two different priority 1/2 targets has
        no main target, which is an error; so does one with only Gaia
        targets, with a warning. A priority 1/2 target named in
        ``occultation_overrides`` is a main target as well and does not
        count toward that error, so a visit can have more than one. Main
        targets always fly the standard keepouts; every other target flies
        ``occultation_earthlimb_min`` when it is set.

        Parameters
        ----------
        calendar : ScienceCalendar
            The long-term calendar, before any time is changed. It is not
            modified; ``process_calendar`` sets ``is_main`` on its own
            copies of the observations.

        Returns
        -------
        dict
            Visit ID to the list of its main targets' names, empty when it
            has none. Also kept as ``self.main_targets``.
        """
        main_targets: dict[str, list[str]] = {}
        for visit in calendar.visits:
            minutes: dict[str, float] = {}
            science: list[str] = []
            for seq in visit.sequences:
                minutes[seq.target] = (
                    minutes.get(seq.target, 0.0) + seq.duration.sec / 60.0
                )
                if seq.priority in (1, 2) and seq.target not in science:
                    science.append(seq.target)

            # A priority 1/2 override is main whatever else is in the visit.
            overridden = [t for t in science if self._overrides_occultation(t)]
            science = [t for t in science if t not in overridden]
            main = None
            if len(science) > 1:
                self._print(
                    f"ERROR: Visit {visit.id}: {len(science)} different "
                    f"priority 1/2 targets ({', '.join(science)}); the visit "
                    "has no main target"
                    + (
                        f" besides the override(s) {', '.join(overridden)}."
                        if overridden
                        else "."
                    )
                )
            elif science:
                main = science[0]
            elif overridden:
                pass  # the overrides are the visit's main targets
            else:
                # sorted() is stable, so a tie goes to the first listed.
                ranked = sorted(minutes, key=lambda t: -minutes[t])
                main = next((t for t in ranked if not self._is_gaia(t)), None)
                if ranked and self._is_gaia(ranked[0]):
                    self._print(
                        f"WARNING: Visit {visit.id}: Gaia target "
                        f"{ranked[0]} has the most minutes "
                        f"({minutes[ranked[0]]:.0f}) and would be the main "
                        "target, but Gaia targets never are; "
                        + (
                            f"{main} ({minutes[main]:.0f} min) is instead."
                            if main
                            else "the visit has no main target."
                        )
                    )
            mains = ([main] if main is not None else []) + overridden
            if mains:
                self._print(
                    f"Visit {visit.id}: main target(s) {', '.join(mains)}"
                )
            main_targets[visit.id] = mains

        # An override naming no target is most likely a typo.
        cleaned = {
            self._clean_name(seq.target)
            for visit in calendar.visits
            for seq in visit.sequences
        }
        for name in getattr(self, "occultation_overrides", None) or ():
            if self._clean_name(name) not in cleaned:
                self._print(
                    f"WARNING: occultation_overrides names {name}, which is "
                    "not a target in the long-term calendar."
                )

        self.main_targets = main_targets
        return main_targets

    @staticmethod
    def _is_gaia(target: str) -> bool:
        """Whether ``target`` is a Gaia source name, ``G`` then digits."""
        return re.fullmatch(r"G\d{6,}", str(target).strip()) is not None

    def _extract_time_window(
        self,
        calendar: ScienceCalendar,
        window_start: Any | None,
        window_duration_days: int,
        verbose: bool,
    ) -> ScienceCalendar:
        """Extract the observations inside the schedule window.

        The window, ``[window_start, window_start + window_duration_days)``,
        is a hard boundary: the delivered calendar never reaches outside
        it. Every observation that overlaps the window is kept, and one the
        long-term calendar runs across either edge is cut at that edge with
        a warning, so the week before and the week after each fly their own
        part of it. A cut that leaves less than ``min_sequence_duration`` is
        a warning too; with ``drop_priority0`` a priority 0 cut that short
        is dropped instead, while any other is kept short. The input
        calendar is not modified.

        Parameters
        ----------
        calendar : ScienceCalendar
            The long-term calendar.
        window_start : str or astropy.time.Time
            Start of the window (ISO string or Time, UTC).
        window_duration_days : float
            Length of the window in days.
        verbose : bool
            Unused; kept for the call signature.

        Returns
        -------
        ScienceCalendar
            A new calendar holding only the window's observations.
        """
        if isinstance(window_start, str):
            window_start = Time(window_start, format="isot", scale="utc")

        window_end = window_start + TimeDelta(
            window_duration_days, format="jd"
        )

        self.window_start = window_start
        self.window_end = window_end

        self._print(f"Extracting window: {window_start} to {window_end}")

        clipped = dropped = 0
        windowed_visits = []
        for visit in calendar.visits:
            # complain if there are empty visits
            if not visit.sequences:
                self._print(
                    f"Warning: Empty sequence list for visit {visit.id}"
                )
            windowed_sequences = []
            for seq in visit.sequences:
                # In seconds with a millisecond tolerance: a window end
                # built from fractional days can sit a few nanoseconds off
                # an observation that starts or stops exactly on it.
                if not (
                    (window_end - seq.start_time).sec > 1e-3
                    and (seq.stop_time - window_start).sec > 1e-3
                ):
                    continue
                cut_start = (window_start - seq.start_time).sec > 1e-3
                cut_stop = (seq.stop_time - window_end).sec > 1e-3
                # Always a copy: later passes change times, names and flags
                # in place, and the caller's calendar must keep its own.
                inside = seq.copy()
                if not (cut_start or cut_stop):
                    windowed_sequences.append(inside)
                    continue

                if cut_start:
                    inside.start_time = window_start
                if cut_stop:
                    inside.stop_time = window_end
                edges = " and ".join(
                    edge
                    for edge, cut in (
                        (
                            "starts before the window start "
                            f"{window_start.isot}",
                            cut_start,
                        ),
                        (
                            f"stops after the window end {window_end.isot}",
                            cut_stop,
                        ),
                    )
                    if cut
                )
                message = (
                    f"{self._seq_prefix(visit.id, seq)} | WINDOW: the "
                    f"long-term calendar {edges}; cut to "
                    f"{inside.start_time.isot} - {inside.stop_time.isot}"
                )
                if self._below_minimum_duration(inside.duration):
                    message += (
                        f", leaving {inside.duration.sec / 60:.1f} min, "
                        "under "
                        f"{self.min_sequence_duration.sec / 60:.0f} min"
                    )
                    if int(seq.priority or 0) == 0 and getattr(
                        self, "drop_priority0", False
                    ):
                        self._print(
                            f"WARNING: {message}. Dropped (drop_priority0)."
                        )
                        dropped += 1
                        continue
                self._print(f"WARNING: {message}.")
                windowed_sequences.append(inside)
                clipped += 1

            if windowed_sequences:
                windowed_visits.append(
                    Visit(id=visit.id, sequences=windowed_sequences)
                )

        if getattr(self, "gap_report", None):
            summary = self.gap_report["processing_summary"]
            summary["window_clipped"] = clipped
            summary["priority_0_dropped"] = (
                summary.get("priority_0_dropped", 0) + dropped
            )

        return ScienceCalendar(
            metadata=calendar.metadata, visits=windowed_visits
        )

    # ------------------------------------------------------------------
    # High-priority additions in free time
    # ------------------------------------------------------------------
    def _add_high_priority_observations(
        self, calendar: ScienceCalendar
    ) -> ScienceCalendar:
        """Fill idle time with extra observations of priority 1/2 targets.

        Runs after the final merge, so nothing already scheduled moves, and
        before the payload update, so the additions get their own
        integration and frame counts. Each target with a priority 1/2
        observation in the window is a candidate, unless named in
        ``high_priority_adds_exclude``. It flies the roll of its
        visit with the most minutes of it (the earlier visit on a tie) and
        the payload of its first priority 1/2 observation there, judged
        under ``high_priority_adds_visibility``, except in a visit that
        already has the target, where it flies that visit's roll for it.

        A slot is a run of whole idle minutes in the window where the
        target is visible at both ends of every minute, at the roll it
        would fly in the visit before and the visit after (whichever it
        joins), at least
        ``high_priority_adds_min_duration`` and at most
        ``high_priority_adds_max_duration`` long. Requiring every minute
        means the start buffers hold without a separate check. Slots are
        taken one at a time: one overlapping its padded transit window
        first, then the longest, then the target with the fewest added
        minutes so far, then the earliest; every other slot is then
        clipped to what is still idle. A target left with fewer than
        ``high_priority_adds_min_observations`` additions is dropped with a
        warning and the selection is redone without it, so its time can go
        to the others.

        Each addition joins an existing visit, so the calendar stays in
        time order: inside a visit, that visit; in a gap between two
        visits, every addition in the gap joins the same one, the visit
        before or after, whichever more of them sit closer to (the visit
        before on a tie). An addition is priority 1 when it overlaps the
        target's T1-T4 widened by ``high_priority_adds_transit_buffer_min``
        on each side, else priority 0, and is flagged ``is_added``; it is
        ``is_main`` when its target is a main target of the visit it
        joins. With
        ``refine_edges_to_second`` the additions then get the same 1 s edge
        pass as everything else (``added_only``), without a movement limit.

        Parameters
        ----------
        calendar : ScienceCalendar
            The processed calendar; additions are inserted into its visits
            in place, each visit kept in time order.

        Returns
        -------
        ScienceCalendar
            The same calendar.
        """
        window_start = self.window_start
        n_minutes = int(
            np.floor((self.window_end - window_start).to_value(u.min) + 1e-6)
        )
        # Every minute edge, so minute i is [times[i], times[i + 1]).
        times = window_start + np.arange(n_minutes + 1) * u.min
        min_length = int(np.ceil(self.high_priority_adds_min_duration - 1e-9))
        max_length = int(np.floor(self.high_priority_adds_max_duration + 1e-9))

        # The observations already scheduled, in time order, with the
        # minutes each touches (sub-minute edges included).
        originals = sorted(
            (
                (visit, seq)
                for visit in calendar.visits
                for seq in visit.sequences
            ),
            key=lambda pair: pair[1].start_time,
        )
        spans = np.array(
            [
                (
                    int(
                        np.floor(
                            (seq.start_time - window_start).to_value(u.min)
                            + 1e-6
                        )
                    ),
                    int(
                        np.ceil(
                            (seq.stop_time - window_start).to_value(u.min)
                            - 1e-6
                        )
                    ),
                )
                for _, seq in originals
            ],
            dtype=int,
        ).reshape(-1, 2)
        occupied = np.zeros(n_minutes, dtype=bool)
        for first, last in spans:
            occupied[max(first, 0) : max(min(last, n_minutes), 0)] = True

        # For every minute, the observation before it and the one after it
        # (indices into ``originals``, -1 for none). An addition joins one
        # of those two visits, so it must be visible at the roll its target
        # flies in either.
        minute_index = np.arange(n_minutes)
        before = (
            np.searchsorted(
                np.maximum.accumulate(spans[:, 1]) if len(spans) else [],
                minute_index,
                side="right",
            )
            - 1
        )
        after = np.searchsorted(spans[:, 0], minute_index + 1, side="left")
        after[after >= len(originals)] = -1
        position = {id(visit): k for k, visit in enumerate(calendar.visits)}
        visit_before = np.array(
            [-1 if k < 0 else position[id(originals[k][0])] for k in before],
            dtype=int,
        )
        visit_after = np.array(
            [-1 if k < 0 else position[id(originals[k][0])] for k in after],
            dtype=int,
        )

        def _roll_in(target, visit, template):
            # One roll per target per visit: the one it already flies there.
            for seq in visit.sequences:
                if seq.target == target:
                    return seq.roll
            return template.roll

        # Each candidate's source visit: the one with the most minutes of
        # it among those where it is priority 1/2, the earlier on a tie.
        # Target -> ((seconds, -visit index), template observation, visit).
        sources: dict[str, tuple[Any, ObservationSequence, str]] = {}
        for order, visit in enumerate(calendar.visits):
            by_target: dict[str, list[ObservationSequence]] = {}
            for seq in visit.sequences:
                by_target.setdefault(seq.target, []).append(seq)
            for target, sequences in by_target.items():
                science = sorted(
                    (s for s in sequences if int(s.priority or 0) in (1, 2)),
                    key=lambda s: s.start_time,
                )
                if not science:
                    continue
                rank = (sum(s.duration.sec for s in sequences), -order)
                if target not in sources or rank > sources[target][0]:
                    sources[target] = (rank, science[0], visit.id)

        # Targets the caller asked to leave out; a name matching none of
        # the candidates is most likely a typo.
        exclude = {
            self._clean_name(name)
            for name in getattr(self, "high_priority_adds_exclude", None) or ()
        }
        for name in getattr(self, "high_priority_adds_exclude", None) or ():
            if not any(
                self._clean_name(t) == self._clean_name(name) for t in sources
            ):
                self._print(
                    f"WARNING: HIGH-PRI ADD: high_priority_adds_exclude names "
                    f"{name}, which is not a priority 1/2 target in the "
                    "window."
                )
        for target in [t for t in sources if self._clean_name(t) in exclude]:
            self._print(
                f"HIGH-PRI ADD: {target} excluded "
                "(high_priority_adds_exclude)."
            )
            del sources[target]

        if not getattr(self, "transit_ephemerides", None) and sources:
            self._print(
                "WARNING: HIGH-PRI ADD: no target_list_dir, so no transit is "
                "known and every addition is priority 0."
            )
        model = self.high_priority_adds_visibility
        usable: dict[str, np.ndarray] = {}
        in_window: dict[str, np.ndarray] = {}
        for target, (_, template, _) in sources.items():
            coord = SkyCoord(
                template.ra, template.dec, unit="deg", frame="icrs"
            )
            at_roll: dict[Any, np.ndarray] = {}
            usable[target] = np.ones(n_minutes, dtype=bool)
            for k, visit in enumerate(calendar.visits):
                need = (visit_before == k) | (visit_after == k)
                if not need.any():
                    continue
                roll = _roll_in(target, visit, template)
                if roll not in at_roll:
                    visible = np.asarray(
                        model.get_visibility(
                            coord,
                            times,
                            roll=None if roll is None else roll * u.deg,
                        )["visible"],
                        dtype=bool,
                    )
                    at_roll[roll] = visible[:-1] & visible[1:]
                usable[target] &= ~need | at_roll[roll]

            # Minutes overlapping T1 - buffer .. T4 + buffer of any transit.
            in_window[target] = np.zeros(n_minutes, dtype=bool)
            ephemerides = getattr(self, "transit_ephemerides", None) or {}
            ephemeris = ephemerides.get(planet_key(target))
            if ephemeris is None:
                if ephemerides:
                    self._print(
                        f"WARNING: HIGH-PRI ADD: {target} has no transit "
                        "ephemeris; its additions are priority 0."
                    )
                continue
            buffer = self.high_priority_adds_transit_buffer_min * u.min
            for epoch in range(
                transit_epoch(ephemeris, times[0]) - 1,
                transit_epoch(ephemeris, times[-1]) + 2,
            ):
                first_contact, last_contact = transit_core(
                    ephemeris, epoch, 1.0
                )
                low = (first_contact - buffer - window_start).to_value(u.min)
                high = (last_contact + buffer - window_start).to_value(u.min)
                in_window[target] |= (minute_index + 1 > low) & (
                    minute_index < high
                )

        def _runs(mask):
            edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
            return zip(edges[::2], edges[1::2])

        excluded = set()
        while True:
            free = ~occupied
            placed: dict[str, list[tuple[int, int, bool]]] = {
                target: [] for target in sources if target not in excluded
            }
            while True:
                best = None
                for target in placed:
                    for start, stop in _runs(free & usable[target]):
                        if stop - start < min_length:
                            continue
                        length = min(stop - start, max_length)
                        # Where a run is longer than an addition may be,
                        # place the addition on the transit window if any.
                        transit = np.flatnonzero(in_window[target][start:stop])
                        if len(transit):
                            start = min(start + transit[0], stop - length)
                        has_transit = bool(
                            in_window[target][start : start + length].any()
                        )
                        # Equal lengths go to the target with the fewest
                        # added minutes so far, so time is spread.
                        given = sum(b - a for a, b, _ in placed[target])
                        key = (has_transit, length, -given, -start)
                        if best is None or key > best[0]:
                            best = (key, target, start, start + length)
                if best is None:
                    break
                (has_transit, _, _, _), target, start, stop = best
                free[start:stop] = False
                placed[target].append((start, stop, has_transit))

            minimum = self.high_priority_adds_min_observations
            thin = [t for t, adds in placed.items() if 0 < len(adds) < minimum]
            if not thin:
                break
            for target in thin:
                self._print(
                    f"WARNING: HIGH-PRI ADD: {target} would get only "
                    f"{len(placed[target])} addition(s), under "
                    f"high_priority_adds_min_observations={minimum}; not "
                    "added, and its time is offered to the other targets."
                )
            excluded.update(thin)

        # Each addition joins a visit already in the calendar. All the
        # additions in one idle gap join the same visit, so visits stay in
        # time order: the visit before the gap or the one after it,
        # whichever more of them sit closer to (the one before on a tie).
        gaps: dict[tuple[int, int], list[tuple[str, int, int, bool]]] = {}
        for target, adds in placed.items():
            for start, stop, has_transit in adds:
                gaps.setdefault(
                    (int(before[start]), int(after[stop - 1])), []
                ).append((target, start, stop, has_transit))
        number = 0
        for (prev_k, next_k), adds in sorted(
            gaps.items(), key=lambda item: min(add[1] for add in item[1])
        ):
            if next_k < 0:
                host = originals[prev_k][0]
            elif prev_k < 0:
                host = originals[next_k][0]
            else:
                closer_before = sum(
                    start - spans[prev_k, 1] <= spans[next_k, 0] - stop
                    for _, start, stop, _ in adds
                )
                host = originals[
                    prev_k if 2 * closer_before >= len(adds) else next_k
                ][0]
            for target, start, stop, has_transit in sorted(
                adds, key=lambda add: add[1]
            ):
                _, template, source_visit = sources[target]
                number += 1
                seq = template.copy()
                # Unique until _renumber_ids numbers the visit in order.
                seq.id = f"A{number:03d}"
                seq.start_time = times[start]
                seq.stop_time = times[stop]
                seq.priority = 1 if has_transit else 0
                seq.roll = _roll_in(target, host, template)
                seq.is_main = any(
                    s.is_main for s in host.sequences if s.target == target
                )
                seq.is_added = True
                host.sequences.append(seq)
                self._print(
                    f"{self._seq_prefix(host.id, seq)} | HIGH-PRI ADD: "
                    f"{stop - start} min of idle time, priority "
                    f"{seq.priority}"
                    + (" (padded transit window)" if has_transit else "")
                    + f", joined visit {host.id}"
                    + (
                        ""
                        if seq.roll is None
                        else f" at roll {seq.roll:.1f} deg"
                    )
                    + f", payload from visit {source_visit}."
                )
            host.sequences.sort(key=lambda seq: seq.start_time)

        # The same 1 s edge pass as everything else, on the additions only.
        # They have no long-term time, so no movement limit.
        if getattr(self, "refine_edges_to_second", False) and any(
            placed.values()
        ):
            self._refine_edges_to_second(calendar, {}, added_only=True)

        added = [
            (visit.id, seq)
            for visit in calendar.visits
            for seq in visit.sequences
            if getattr(seq, "is_added", False)
        ]
        added_visits = len({visit_id for visit_id, _ in added})
        added_minutes = round(
            sum(seq.duration.sec for _, seq in added) / 60.0, 1
        )

        if getattr(self, "gap_report", None):
            summary = self.gap_report["processing_summary"]
            summary["high_pri_added_visits"] = added_visits
            summary["high_pri_added_observations"] = len(added)
            summary["high_pri_added_minutes"] = added_minutes
        self._print(
            f"High-priority additions: {len(added)} observation(s), "
            f"{added_minutes} min, in {added_visits} existing visit(s)."
        )
        return calendar

    # Tolerances used when deciding whether two sequences can be merged.
    _MERGE_ADJACENCY_TOL_SEC = 1.0  # max stop-to-start gap (seconds)
    _MERGE_POINTING_TOL_DEG = 1e-6  # max RA/Dec difference (degrees)
    _MERGE_ROLL_TOL_DEG = 1e-6  # max roll difference (degrees)

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
        visit_id_map: dict[str, str] = {}
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

        self._print(f"Renumbered IDs: {changed} identifier(s) updated.")

        return calendar

    def _merge_similar_observations(
        self,
        calendar: ScienceCalendar,
        verbose: bool = False,
        before_trims: bool = False,
    ) -> ScienceCalendar:
        """Merge same-target sequences within each visit.

        This runs twice in ``process_calendar``: once on the long-term
        times, right after the roll sweep and before any trimming
        (``before_trims=True``), and once at the end on the final times.

        Two consecutive sequences (in start-time order) are merged when
        all of the following hold:

        1. they belong to the same visit,
        2. they observe the same target at the same pointing (RA/Dec),
        3. they fly the same roll, since one observation has one attitude,
        4. nothing else is scheduled between them, checked across the whole
           calendar because visits can interleave in time, and
        5. they are contiguous, *or* the gap between them is a keepout
           violation short enough to ride out under the configured gap
           tolerance.

        Point 5 is what stops a brief star-tracker dropout splitting a
        target in two: the same dropout occurring a minute later, inside an
        observation, is simply tolerated, so an observation boundary
        happening to land on it should not change the outcome. A merged
        observation therefore can contain minutes that fail a keepout, and
        ``validate_visibility`` will report them.

        With ``drop_priority0``, point 4 is relaxed for priority-0 filler:
        when only priority-0 observations sit between two priority-1/2
        observations that meet every other point, those priority 0s are
        dropped (and logged as a warning) and the pair is merged.

        Before the trims (``before_trims``), nothing is dropped, and point 5
        is judged on the whole dark stretch at the join of the merged span,
        not just the dark minutes in the gap (see
        :meth:`_join_is_bridgeable`): the trims that follow would otherwise
        find a dark stretch longer than the tolerance in the middle of the
        merged observation and keep only one side of it. A dark stretch that
        runs to either end of the merged span is allowed, since the trims
        cut it off as a dark head or tail. This is what joins a visible
        observation to a dark one of the same target across Free Time, so
        the dark time is trimmed rather than flown.

        The merged sequence keeps the first sequence's identity and extends
        its ``stop_time`` to the second sequence's ``stop_time``. It takes
        the higher of the two priorities, and the payload parameters of the
        sequence that had it (the first's on a tie). Merging is applied
        transitively, so a run of three or more eligible sequences
        collapses into one.

        Parameters
        ----------
        calendar : ScienceCalendar
            Calendar to merge in place-safe fashion (a new calendar with
            new visits/sequences is returned; the input is not mutated).
        verbose : bool, optional
            If True, print a line for each merge performed.
        before_trims : bool, optional
            True for the merge on the long-term times, ahead of the trims.
            Defaults to False, the merge on the final times.

        Returns
        -------
        ScienceCalendar
            Calendar with eligible sequences merged.
        """
        merged_count = 0
        new_visits: list[Visit] = []

        # With drop_priority0, priority-0 observations that are all that
        # separates two otherwise mergeable higher-priority observations are
        # dropped first, so the merge below sees an empty gap and joins the
        # pair. Keyed by (visit_id, seq_id). Never before the trims.
        dropped = set()
        if getattr(self, "drop_priority0", False) and not before_trims:
            all_ordered = self._ordered_sequences(calendar)
            kept = [
                index
                for index, (_, seq) in enumerate(all_ordered)
                if int(seq.priority or 0) > 0
            ]
            for i, j in zip(kept, kept[1:]):
                (first_visit, first), (second_visit, second) = (
                    all_ordered[i],
                    all_ordered[j],
                )
                between = all_ordered[i + 1 : j]
                # No occupied list: the priority 0s between are the ones
                # being judged, and nothing else is there.
                if (
                    not between
                    or first_visit != second_visit
                    or not self._can_merge(first, second)
                ):
                    continue
                for visit_id, seq in between:
                    dropped.add((visit_id, seq.id))
                    self._print(
                        f"WARNING: {self._seq_prefix(visit_id, seq)} | "
                        f"MERGE: dropped priority-0 {seq.target}, the only "
                        f"observation between {first.target} sequences "
                        f"{first.id} and {second.id}, whose gap is within "
                        "tolerance (drop_priority0)."
                    )

        # Every observation's span, so a pair separated by a gap is only
        # joined when nothing else is scheduled inside that gap. Taken
        # across the whole calendar, not just the visit, because visits can
        # interleave in time.
        occupied = [
            (seq.start_time, seq.stop_time)
            for visit_id, seq in self._ordered_sequences(calendar)
            if (visit_id, seq.id) not in dropped
        ]

        for visit in self._progress(
            calendar.visits,
            desc="Merging same-target observations",
            total=len(calendar.visits),
        ):
            # Process sequences in chronological order so "right after each
            # other" is well defined regardless of input ordering.
            ordered = sorted(
                (
                    seq
                    for seq in visit.sequences
                    if (visit.id, seq.id) not in dropped
                ),
                key=lambda s: s.start_time,
            )

            merged_sequences: list[ObservationSequence] = []
            for seq in ordered:
                if merged_sequences and self._can_merge(
                    merged_sequences[-1],
                    seq,
                    occupied,
                    judge_join=before_trims,
                ):
                    # Extend the previous (kept) sequence over this one.
                    previous = merged_sequences[-1]
                    # The merged observation flies at the higher priority,
                    # with that piece's payload so the two agree.
                    raised = int(seq.priority or 0) > int(
                        previous.priority or 0
                    )
                    self._print(
                        f"{self._seq_prefix(visit.id, previous)} | MERGE"
                        f"{' (before trims)' if before_trims else ''}: "
                        f"absorbing sequence {seq.id} "
                        f"({self._seq_prefix(visit.id, seq)}); stop "
                        f"{previous.stop_time_str} -> {seq.stop_time_str}"
                        + (
                            f"; priority {previous.priority} -> "
                            f"{seq.priority}, with its payload"
                            if raised
                            else ""
                        )
                    )
                    previous.stop_time = seq.stop_time
                    if raised:
                        previous.priority = seq.priority
                        previous.payload_params = seq.copy().payload_params
                    merged_count += 1
                else:
                    # Copy so the returned calendar never aliases the input.
                    merged_sequences.append(seq.copy())

            # A visit left empty by the drops above goes with them.
            if merged_sequences or not visit.sequences:
                new_visits.append(
                    Visit(id=visit.id, sequences=merged_sequences)
                )

        if dropped and getattr(self, "gap_report", None):
            self.gap_report["processing_summary"]["priority_0_dropped"] += len(
                dropped
            )
        if getattr(self, "gap_report", None):
            key = "merged_before_trims" if before_trims else "merged_at_end"
            self.gap_report["processing_summary"][key] = merged_count
        self._print(
            f"Merged {merged_count} similar observation sequence(s) "
            f"{'before trimming ' if before_trims else ''}"
            f"across {len(calendar.visits)} visit(s), dropping "
            f"{len(dropped)} priority-0 observation(s) between them."
        )

        return ScienceCalendar(
            metadata=calendar.metadata,
            visits=new_visits,
            visibility=calendar.visibility,
        )

    def _can_merge(
        self,
        first: ObservationSequence,
        second: ObservationSequence,
        occupied: list[tuple[Time, Time]] | None = None,
        judge_join: bool = False,
    ) -> bool:
        """Return True if ``second`` can be merged into ``first``.

        See :meth:`_merge_similar_observations` for the merge criteria.
        ``judge_join`` applies the before-the-trims rule for point 5
        (:meth:`_join_is_bridgeable`), to contiguous pairs as well.
        """
        # Same target (case-insensitive, whitespace-insensitive).
        if (first.target or "").strip().lower() != (
            second.target or ""
        ).strip().lower():
            return False

        # Same pointing.
        if (
            abs(first.ra - second.ra) > self._MERGE_POINTING_TOL_DEG
            or abs(first.dec - second.dec) > self._MERGE_POINTING_TOL_DEG
        ):
            return False

        # Same roll: one observation flies one attitude, so two different
        # rolls cannot become one observation. Compared as an angle, since
        # -180 and +180 are the same attitude.
        if (first.roll is None) != (second.roll is None):
            return False
        if first.roll is not None:
            separation = abs(
                ((first.roll - second.roll + 180.0) % 360.0) - 180.0
            )
            if separation > self._MERGE_ROLL_TOL_DEG:
                return False

        gap_sec = (second.start_time - first.stop_time).sec
        if gap_sec < -self._MERGE_ADJACENCY_TOL_SEC:
            return False  # overlapping rather than adjacent
        if gap_sec <= self._MERGE_ADJACENCY_TOL_SEC:
            if judge_join:
                return self._join_is_bridgeable(first, second)
            return True

        # A real gap between them. It can still be absorbed when nothing
        # else is scheduled inside it and the keepout violation that opened
        # it is short enough to ride out mid-observation anyway
        if occupied is not None and any(
            start < second.start_time - self._MERGE_ADJACENCY_TOL_SEC * u.s
            and stop > first.stop_time + self._MERGE_ADJACENCY_TOL_SEC * u.s
            for start, stop in occupied
        ):
            return False

        if judge_join:
            return self._join_is_bridgeable(first, second)
        return self._gap_is_bridgeable(first, second)

    def _join_is_bridgeable(
        self,
        first: ObservationSequence,
        second: ObservationSequence,
    ) -> bool:
        """Whether merging leaves the trims a dark stretch they can handle.

        Judged over the merged span ``[first.start_time,
        second.stop_time)`` once a minute, at the roll and under the
        keepouts the observation flies. Each dark stretch that touches the
        join (the first's last minute, the gap, the second's first minute),
        taken to its full length inside the merged span, must either be
        within the gap tolerance or run to an end of the merged span. One
        that runs to an end must leave at least ``min_sequence_duration``
        from the first to the last visible minute on the other side, so the
        trims cut it off rather than refuse.

        Parameters
        ----------
        first, second : ObservationSequence
            The earlier and later observation, same target and roll.

        Returns
        -------
        bool
            True when the pair may be merged.
        """
        n = int(np.rint((second.stop_time - first.start_time).sec / 60.0))
        if n <= 0:
            return True
        coord = SkyCoord(first.ra, first.dec, frame="icrs", unit="deg")
        times = first.start_time + np.arange(n) * u.min
        visible = np.atleast_1d(
            self._visibility_for_sequence(first, coord, times)
        ).astype(bool)
        join_lo = max(
            int(np.rint((first.stop_time - first.start_time).sec / 60.0)) - 1,
            0,
        )
        join_hi = min(
            int(np.rint((second.start_time - first.start_time).sec / 60.0)),
            n - 1,
        )
        min_minutes = int(np.rint(self.min_sequence_duration.sec / 60.0))
        model = self._visibility_for(first)

        def visible_span(part: np.ndarray) -> int:
            # Minutes from the first to the last visible minute.
            lit = np.flatnonzero(part)
            return int(lit[-1] - lit[0] + 1) if lit.size else 0

        for run_start, run_end in self._find_nonvisible_gaps(visible):
            if run_end <= join_lo or run_start > join_hi:
                continue  # does not touch the join
            if self._is_gap_tolerable(
                coord,
                times,
                run_start,
                run_end - run_start,
                roll=first.roll,
                visibility=model,
            ):
                continue
            if run_start == 0 and (
                visible_span(visible[run_end:]) >= min_minutes
            ):
                continue  # a dark head the trims cut off
            if run_end == n and (
                visible_span(visible[:run_start]) >= min_minutes
            ):
                continue  # a dark tail the trims cut off
            return False
        return True

    def _gap_is_bridgeable(
        self,
        first: ObservationSequence,
        second: ObservationSequence,
    ) -> bool:
        """Whether the gap between two observations is short enough to absorb.

        The gap is not necessarily dark throughout: growth stops at the
        first minute it cannot use, so a visible minute can be left stranded
        against the neighbor. The tolerance is therefore judged on the dark
        minutes the merge would actually swallow, classified at the first of
        them, rather than on the whole gap measured from its first minute.
        """
        minutes = int(
            np.rint((second.start_time - first.stop_time).sec / 60.0)
        )
        if minutes <= 0:
            return True
        if self.earthlimb_gap_tolerance == 0 and self.st_gap_tolerance == 0:
            # No violation is tolerable, so no gap can be bridged and there
            # is no point asking the visibility model about it.
            return False

        target_coord = SkyCoord(first.ra, first.dec, frame="icrs", unit="deg")
        times = first.stop_time + np.arange(minutes) * u.min
        roll = first.roll

        model = self._visibility_for(first)
        visible = np.atleast_1d(
            np.asarray(
                model.get_visibility(
                    target_coord,
                    times,
                    **({} if roll is None else {"roll": roll * u.deg}),
                )["visible"]
            )
        )
        dark = np.flatnonzero(~visible)
        if dark.size == 0:
            return True  # nothing to ride out

        return self._is_gap_tolerable(
            target_coord,
            times,
            int(dark[0]),
            int(dark.size),
            roll=roll,
            visibility=model,
        )

    def _process_all_sequences(
        self,
        calendar: ScienceCalendar,
        verbose: bool = False,
        merge_first: bool = False,
    ) -> ScienceCalendar:
        """Iterate through sequences and build minute-resolution visibility.

        This internal routine constructs a synchronized time grid for the
        windowed calendar, queries visibility for each sequence target and
        accumulates a boolean minute-array (`all_minutes_bool`) describing
        which minutes are visible. It then calls the visibility-fixing
        steps. Payload counts are left to ``process_calendar``, which
        recomputes them after the final merge.

        Parameters
        ----------
        calendar : ScienceCalendar
            Windowed calendar to operate on.
        verbose : bool, optional
            If True, print progress messages.
        merge_first : bool, optional
            Merge same-target observations on the long-term times, after
            the roll sweep and before any trim, so a dark observation joined
            to a visible one has its dark time trimmed. The merged span
            becomes the observation's long-term time for the movement limit
            and the timing log. Defaults to False.

        Returns
        -------
        ScienceCalendar
            Calendar with adjusted sequences; payload parameters not yet
            updated for the new durations.
        """

        working_calendar = deepcopy(calendar)

        # Snapshot each sequence's original timing so any shrink/elongate
        # performed by the gap-fill / trim passes below can be logged.
        original_timing = {
            (visit.id, seq.id): (seq.start_time, seq.stop_time)
            for visit in working_calendar.visits
            for seq in visit.sequences
        }
        # Why each boundary moved, filled in by the passes below and read
        # back by _log_timing_changes.
        self._timing_notes = {}

        # One roll per target per visit, chosen by pandoravisibility and
        # written onto the sequences, so every pass below judges an
        # observation at the roll it will fly. A target whose every
        # observable roll fails the NIRDA trace check is reported, and with
        # drop_observations_based_on_trace dropped from the visit here,
        # before trimming and growth, so neighbors can use the time. A
        # fixed roll is not trace checked, so later passes see a kept one
        # as visible.
        trace_dropped = 0
        for visit in self._progress(
            working_calendar.visits,
            desc="Roll sweep",
            total=len(working_calendar.visits),
        ):
            visit_rolls = get_best_roll_per_visit(
                visit,
                self.visibility,
                roll_step=self.roll_step,
                min_power_frac=self.min_power_frac,
                occultation_visibility=getattr(
                    self, "occultation_visibility", None
                ),
                occultation_overrides=getattr(
                    self, "_occultation_override_names", frozenset()
                ),
                growth_margin_minutes=self.max_movement_minutes,
            )
            blocked = set()
            for target, result in visit_rolls.items():
                contamination = result.get("trace_contamination", np.nan)
                self._print(
                    f"  Visit {visit.id} / {target}: roll "
                    f"{result['roll_deg']:.1f} deg, "
                    f"{result['n_scheduled_visible']} of "
                    f"{int(result['scheduled'].sum())} scheduled minutes "
                    "visible"
                    + (
                        f", trace contamination {contamination:.1e}"
                        if np.isfinite(contamination)
                        else ""
                    )
                )
                if result["trace_blocked"]:
                    failure = (
                        f"Visit {visit.id} / {target}: every roll that sees "
                        "it fails the NIRDA trace check (tolerance "
                        f"{self.visibility.trace_contamination_tol:g}); "
                        f"best ranked roll {result['roll_deg']:.1f} deg "
                        f"has contamination {contamination:.1e}, the "
                        "cleanest roll "
                        f"{result['trace_min_roll_deg']:.1f} deg has "
                        f"{result['trace_min_contamination']:.1e}"
                    )
                    if getattr(
                        self, "drop_observations_based_on_trace", False
                    ):
                        blocked.add(target)
                        self._print(f"WARNING: {failure}. Dropped.")
                    else:
                        self._print(
                            f"ERROR: {failure}. Kept, flying "
                            f"{result['roll_deg']:.1f} deg "
                            "(drop_observations_based_on_trace is off)."
                        )
                elif result["n_scheduled_visible"] == 0:
                    self._print(
                        f"ERROR: Visit {visit.id} / {target}: no roll "
                        "makes any scheduled minute visible; flying "
                        f"{result['roll_deg']:.1f} deg (best for the star "
                        "trackers alone, then for solar power)"
                    )
            if blocked:
                kept = [s for s in visit.sequences if s.target not in blocked]
                trace_dropped += len(visit.sequences) - len(kept)
                visit.sequences = kept
        if trace_dropped:
            # Only visits the drop emptied can be empty here.
            working_calendar.visits = [
                visit for visit in working_calendar.visits if visit.sequences
            ]
        if getattr(self, "gap_report", None):
            self.gap_report["processing_summary"][
                "trace_dropped"
            ] = trace_dropped

        # Each priority-2 transit's coverage as the long-term calendar
        # placed it, at the rolls now held, for the report at the end.
        transits_before = self._transit_coverages(working_calendar)
        if getattr(self, "transit_ephemerides", None):
            for target in sorted(
                {
                    seq.target
                    for visit in working_calendar.visits
                    for seq in visit.sequences
                    if seq.priority == 2
                    and planet_key(seq.target) not in self.transit_ephemerides
                }
            ):
                self._print(
                    f"WARNING: {target}: no transit ephemeris in the "
                    "PandoraTargetList, so its priority-2 observations are "
                    "not held to the transit constraint."
                )

        # Observations keep the times the long-term calendar gave them and
        # are adjusted in place. Idle time between them is expected under
        # the current conops, so no attempt is made to close it by sliding
        # observations earlier.

        # Merge on the long-term times, before anything is trimmed. A dark
        # observation of a target can sit across Free Time from a visible
        # one; neither trim can touch an observation with nothing visible,
        # so merged here its dark time becomes a tail (or head) the trims
        # below cut off. Nothing has moved yet, so each surviving
        # observation's span is its long-term time from here on.
        if merge_first:
            working_calendar = self._merge_similar_observations(
                working_calendar, verbose, before_trims=True
            )
            original_timing = {
                (visit.id, seq.id): (seq.start_time, seq.stop_time)
                for visit in working_calendar.visits
                for seq in visit.sequences
            }

        # Trim each observation back to the visible time it actually has:
        # first trailing dark minutes, then any dark stretch in the middle
        # (e.g. the target dipping below the Earth-limb keepout). Trimming
        # a dark head falls out of the same pass.
        working_calendar = self._trim_non_visible_tails(working_calendar)
        working_calendar = self._trim_to_longest_visible_block(
            working_calendar
        )

        # Trimming only ever shrinks, so grow each observation back out
        # into the idle time around it wherever the target is visible.
        working_calendar = self._grow_into_free_time(
            working_calendar, original_timing
        )

        # Require the opening minutes of each observation to be clean:
        # star trackers settled and the boresight clear of the Earth. Runs
        # after growth so it judges the final start.
        working_calendar = self._enforce_start_buffers(working_calendar)

        # Nothing may end up further than max_movement_minutes from where
        # the long-term calendar put it.
        working_calendar = self._clamp_movement(
            working_calendar, original_timing
        )

        # Overlaps can never be flown, so this has the last say on how
        # observations share time.
        working_calendar = self._repair_overlaps(working_calendar)

        # Everything above works in whole minutes; settle each boundary to
        # the second. Only moves within free time and the target's own
        # visibility, so it cannot create an overlap.
        if getattr(self, "refine_edges_to_second", False):
            working_calendar = self._refine_edges_to_second(
                working_calendar, original_timing
            )

        # A transit can end up short only if the long-term calendar left it
        # short or overlap repair cut it; either needs a look.
        needed = getattr(self, "transit_duration_constraint_min", 0)
        for transit, (target, core_start, covered) in self._transit_coverages(
            working_calendar
        ).items():
            if covered >= needed:
                continue
            before = transits_before.get(transit, (None, None, 0))[2]
            self._print(
                f"ERROR: {target} transit at {core_start.isot[:16]} (epoch "
                f"{transit[1]}): its priority-2 observations cover {covered} "
                f"visible min of the middle "
                f"{self.transit_middle_frac:.0%} of the transit, under the "
                f"{needed:g} min required ({before} min as the long-term "
                "calendar placed them)."
            )

        # Report any timing changes (shrink/elongate) made above. Payload
        # counts are recomputed by process_calendar, after the final merge.
        self._log_timing_changes(working_calendar, original_timing)

        return working_calendar

    def _log_timing_changes(
        self, calendar: ScienceCalendar, original_timing: dict[Any, Any]
    ) -> None:
        """Log per-sequence shrink/elongate vs the snapshot in
        original_timing (keyed by ``(visit_id, sequence_id)``).

        This is the one pass that sees every observation's final timing
        against its long-term timing, so it also records the modification
        tallies in ``gap_report``.
        """
        modifications = self.gap_report["sequence_modifications"]
        for key in modifications:
            modifications[key] = []

        for visit in calendar.visits:
            for seq in visit.sequences:
                orig = original_timing.get((visit.id, seq.id))
                if orig is None:
                    continue
                old_start, old_stop = orig
                d_start = (seq.start_time - old_start).sec
                d_stop = (seq.stop_time - old_stop).sec
                prefix = self._seq_prefix(visit.id, seq)
                if abs(d_start) < 1.0 and abs(d_stop) < 1.0:
                    modifications["unchanged_sequences"].append(prefix)
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
                lengthened = new_dur > old_dur
                verb = "ELONGATED" if lengthened else "SHRANK"
                modifications[
                    (
                        "extended_sequences"
                        if lengthened
                        else "shortened_sequences"
                    )
                ].append(prefix)
                notes = getattr(self, "_timing_notes", {}).get(
                    (visit.id, seq.id), []
                )
                self._print(
                    f"{prefix} | {verb}: "
                    + ", ".join(parts)
                    + f" (duration {old_dur:.1f} -> {new_dur:.1f} min)"
                    + (": " + "; ".join(notes) if notes else "")
                )

        summary = self.gap_report["processing_summary"]
        summary["sequences_lengthened"] = len(
            modifications["extended_sequences"]
        )
        summary["sequences_shortened"] = len(
            modifications["shortened_sequences"]
        )
        summary["sequences_modified"] = (
            summary["sequences_lengthened"] + summary["sequences_shortened"]
        )

    def _note_timing(self, visit_id: Any, seq_id: Any, note: str) -> None:
        """Record why a pass moved a boundary, for ``_log_timing_changes``."""
        notes = self.__dict__.setdefault("_timing_notes", {})
        notes.setdefault((visit_id, seq_id), []).append(note)

    def _get_synchronized_time_grid(
        self, calendar: ScienceCalendar
    ) -> tuple[int, Time | None, Time | None, Any]:
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

    def _visibility_for(self, seq: ObservationSequence) -> Any:
        """The visibility model an observation is judged by.

        Everything goes through here rather than reading ``self.visibility``
        directly, so a pass cannot trim an observation against one set of
        keepouts while another pass grows it against a different set.

        Parameters
        ----------
        seq : ObservationSequence
            The observation; its ``is_main`` flag decides the model.

        Returns
        -------
        pandoravisibility.Visibility
            For a high-priority addition (``is_added``), the additions
            model. Otherwise the occultation model when one is configured
            and this is neither its visit's main target nor named in
            ``occultation_overrides``, else the nominal model.
        """
        if getattr(seq, "is_added", False):
            return (
                getattr(self, "high_priority_adds_visibility", None)
                or self.visibility
            )
        occultation = getattr(self, "occultation_visibility", None)
        if (
            occultation is not None
            and not getattr(seq, "is_main", False)
            and not self._overrides_occultation(seq.target)
        ):
            return occultation
        return self.visibility

    def _overrides_occultation(self, target: str) -> bool:
        """Whether ``target`` is named in ``occultation_overrides``.

        Parameters
        ----------
        target : str
            Target name, as read or after name cleaning.

        Returns
        -------
        bool
            True when it flies the standard keepouts though not main.
        """
        names = getattr(self, "_occultation_override_names", frozenset())
        return self._clean_name(str(target)) in names

    def _below_minimum_duration(self, duration: TimeDelta) -> bool:
        """Checks if ``duration`` shorter than ``min_sequence_duration``

        This function handles floating point number errors during
        subtraction between two `Time` objects.

        Parameters
        ----------
        duration : astropy.time.TimeDelta
            Span to test, typically ``stop - start``.

        Returns
        -------
        bool
            True when the span is too short to deliver.
        """
        return round(duration.sec) < round(self.min_sequence_duration.sec)

    def _trim_non_visible_tails(
        self, calendar: ScienceCalendar
    ) -> ScienceCalendar:
        """Trim non-visible tails from sequences.

        For each sequence whose last minute(s) are not visible, shrink
        ``stop_time`` to the last visible minute + 1.  Then attempt to
        extend the *next* sequence backward to absorb the freed time
        (only where that target is visible).

        Non-visible starts are handled by
        ``_trim_to_longest_visible_block``, which strips leading dark
        minutes off the span it selects.
        """
        working_cal = deepcopy(calendar)

        # Collect all sequences globally, sorted by start_time
        all_sequences: list[tuple[str, ObservationSequence]] = []
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

            model = self._visibility_for(seq)
            vis_arr = self._visibility_for_sequence(seq, target_coord, times)

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
                roll=seq.roll,
                visibility=model,
            ):
                continue  # tolerable tail — leave it

            new_stop = seq.start_time + (last_visible_idx + 1) * u.min

            if self._below_minimum_duration(new_stop - seq.start_time):
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

                    next_vis_arr = self._visibility_for_sequence(
                        next_seq, next_coord, gap_times
                    )

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
                roll=seq.roll,
                is_main=seq.is_main,
            )
            working_cal.replace_sequence(visit_id, seq.id, trimmed)
            self._note_timing(
                visit_id, seq.id, f"stop: {tail_length} min dark tail trimmed"
            )

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

            next_vis_arr = self._visibility_for_sequence(
                next_seq, next_coord, gap_times
            )

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
                roll=next_seq.roll,
                is_main=next_seq.is_main,
            )
            working_cal.replace_sequence(
                next_visit_id, next_seq.id, extended_next
            )
            self._note_timing(
                next_visit_id,
                next_seq.id,
                f"start: grew {gap_minutes - first_contiguous} min into time "
                f"freed by {seq.target}'s trimmed tail",
            )
            # Update local list so subsequent iterations see
            # the modified next sequence.
            all_sequences[idx + 1] = (next_visit_id, extended_next)

        return working_cal

    def _star_tracker_failed(
        self,
        target_coord: SkyCoord,
        time: Time,
        roll: float | None,
        visibility: Any = None,
    ) -> bool:
        """Whether the star-tracker keepout fails at ``time`` for this roll.

        ``get_star_tracker_breakdown`` is used rather than the
        ``star_tracker`` entry of ``get_all_constraints`` so the verdict is
        evaluated at the roll the observation will actually fly, which is
        the roll the sweep chose to keep the trackers clear.  Since
        pandoravisibility v1.3.0 ``get_all_constraints`` also takes a
        ``roll``, so the two could be folded into one call, but the
        duck-typed visibility objects the scheduler accepts split the
        tracker verdict out this way and nothing is gained by merging.

        A failure to evaluate the trackers is reported and answered "not a
        tracker failure", which leaves the caller treating the gap as
        intolerable and trimming it away. Guessing the other way would keep
        dark minutes in the schedule on the strength of a verdict we never
        actually got.
        """
        try:
            breakdown = (
                visibility or self.visibility
            ).get_star_tracker_breakdown(
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
        roll: float | None = None,
        visibility: Any = None,
    ) -> bool:
        """Check whether a non-visible gap is short enough to tolerate.

        Uses ``get_all_constraints`` at the first non-visible minute to
        identify which boresight constraint(s) failed, checks the star
        tracker separately at *roll* (see :meth:`_star_tracker_failed`),
        then compares the gap length against the matching tolerance
        (``earthlimb_gap_tolerance`` or ``st_gap_tolerance``). Only the
        boresight verdicts are read from the first call; the tracker one
        depends on roll and is taken from the second.

        If both tolerances are zero (the default), every gap is
        intolerable and this returns False immediately.
        """
        el_tol = self.earthlimb_gap_tolerance
        st_tol = self.st_gap_tolerance

        if el_tol == 0 and st_tol == 0:
            return False

        model = visibility or self.visibility
        try:
            constraints = model.get_all_constraints(
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
            target_coord, times[gap_start], roll, visibility=model
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
        """Trim sequences to their longest acceptable visible span.

        Runs after ``_trim_non_visible_tails``. A sequence can be dark at
        its head, in the middle (e.g. the target dipping below the
        Earth-limb keepout during an orbit), or both; the span selected
        here has its leading and trailing dark minutes stripped, so this
        is what trims a dark head.

        Short gaps are tolerated when their duration does not exceed the
        configured tolerances (``earthlimb_gap_tolerance`` and
        ``st_gap_tolerance``).  Gaps exceeding those limits cause the
        sequence to be trimmed to its longest acceptable span — the
        longest contiguous window that contains only tolerable gaps.

        After trimming, the method attempts to extend neighboring
        sequences to reclaim the freed time (where those neighbors
        are visible).
        """
        working_cal = deepcopy(calendar)

        # Collect all sequences globally, sorted by start_time.
        all_sequences: list[tuple[str, ObservationSequence]] = []
        for visit in working_cal.visits:
            for seq in visit.sequences:
                all_sequences.append((visit.id, seq))
        all_sequences.sort(key=lambda x: x[1].start_time)

        for idx, (visit_id, seq) in self._progress(
            list(enumerate(all_sequences)),
            desc="Trimming to longest visible block",
            total=len(all_sequences),
        ):
            analysis = self._analyze_mid_sequence_visibility(seq)
            if analysis is None:
                continue

            target_coord, times, vis_arr = analysis
            gaps = self._find_nonvisible_gaps(vis_arr)
            if not gaps:
                continue

            gap_tolerable = [
                self._is_gap_tolerable(
                    target_coord,
                    times,
                    gap_start,
                    gap_end - gap_start,
                    roll=seq.roll,
                    visibility=self._visibility_for(seq),
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
            if trimmed is None or not self._transit_allows(
                working_cal,
                visit_id,
                seq,
                trimmed.start_time,
                trimmed.stop_time,
                "keeping only the longest visible block",
            ):
                continue

            working_cal.replace_sequence(visit_id, seq.id, trimmed)
            all_sequences[idx] = (visit_id, trimmed)
            dropped = [
                f"{n} min at the {end}"
                for n, end in (
                    (best_window[0], "head"),
                    (len(vis_arr) - best_window[1], "tail"),
                )
                if n
            ]
            self._note_timing(
                visit_id,
                seq.id,
                "kept the longest visible block, dropping dark "
                + " and ".join(dropped),
            )

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

    def _ordered_sequences(
        self, calendar: ScienceCalendar
    ) -> list[tuple[str, ObservationSequence]]:
        """Return ``(visit_id, sequence)`` pairs in start-time order."""
        pairs = [
            (visit.id, seq)
            for visit in calendar.visits
            for seq in visit.sequences
        ]
        pairs.sort(key=lambda pair: pair[1].start_time)
        return pairs

    def _grow_into_free_time(
        self,
        calendar: ScienceCalendar,
        original_timing: dict[Any, Any],
    ) -> ScienceCalendar:
        """Extend observations into adjacent idle time while still visible.

        This grows each observation outward into the idle time around it for
        as long as the target stays visible at its scheduled roll, bounded by:
        - the neighboring observations, so growth can never create an
          overlap, and
        - ``max_movement_minutes`` either side of the long-term start and
          stop, so an observation grows in place instead of drifting, and
        - the schedule window, a hard boundary. Growth it stops while the
          target is still visible is reported as a warning.

        With ``grow_by_priority`` the walk runs priority 2, then 1, then 0,
        the main target first at equal priority and the earlier visit first
        at equal rank (see ``_growth_rank``), and a lower-ranked neighbor,
        or an equal-ranked one in a later visit, is a soft bound: the grower
        may take its minutes while the neighbor keeps
        ``min_sequence_duration`` and the moved boundary stays within
        ``max_movement_minutes`` of the neighbor's long-term time. Growth
        that visibility allowed but such a floor refused goes to the error
        log. With ``drop_priority0`` as well, a lower-ranked priority-0
        neighbor has no floor: the grower may grow past it, and a priority 0 taken
        past its floor is removed from the calendar. Off, the walk is
        start-time order and every neighbor is a hard bound. Either way a
        boundary that has already moved is what the next observation sees.
        Times are modified in place.
        """
        ordered = self._ordered_sequences(calendar)
        # Long-term visit order, for the earlier-visit tie-break.
        self._visit_order = {
            visit.id: index for index, visit in enumerate(calendar.visits)
        }
        walk = list(range(len(ordered)))
        if getattr(self, "grow_by_priority", False):
            # Highest rank first, then the earlier visit; reverse sorting is
            # stable, so start-time order holds within a visit and rank.
            walk.sort(
                key=lambda i: (
                    self._growth_rank(ordered[i][1]),
                    -self._visit_order.get(ordered[i][0], 0),
                ),
                reverse=True,
            )
        gained_starts = gained_stops = taken = 0
        # (visit_id, seq_id) of priority-0 observations eaten by growth.
        self._dropped_by_growth = set()
        # Boundaries already warned about as held at the window edge.
        self._window_reported = set()

        for index in self._progress(
            walk, desc="Growing into idle time", total=len(walk)
        ):
            visit_id, seq = ordered[index]
            if original_timing.get((visit_id, seq.id)) is None:
                continue
            if (visit_id, seq.id) in self._dropped_by_growth:
                continue
            for direction in (-1, 1):
                # A neighbor eaten whole leaves room beyond it, so that side
                # is grown again until no further neighbor is dropped.
                while True:
                    n_dropped = len(self._dropped_by_growth)
                    gained, took = self._grow_one_side(
                        ordered, index, direction, original_timing
                    )
                    if direction < 0:
                        gained_starts += gained
                    else:
                        gained_stops += gained
                    taken += took
                    if len(self._dropped_by_growth) == n_dropped:
                        break

        if self._dropped_by_growth:
            for visit in calendar.visits:
                visit.sequences = [
                    seq
                    for seq in visit.sequences
                    if (visit.id, seq.id) not in self._dropped_by_growth
                ]
            calendar.visits = [
                visit for visit in calendar.visits if visit.sequences
            ]

        summary = self.gap_report["processing_summary"]
        summary["minutes_grown_at_starts"] = gained_starts
        summary["minutes_grown_at_stops"] = gained_stops
        summary["minutes_taken_from_lower_priority"] = taken
        # Added to: window extraction may already have dropped some.
        summary["priority_0_dropped"] = summary.get(
            "priority_0_dropped", 0
        ) + len(self._dropped_by_growth)
        self._print(
            f"Grew observations into idle time:{gained_starts} min added "
            f"at starts, {gained_stops} min added at stops, {taken} min of "
            "that taken from lower-ranked neighbors, "
            f"{len(self._dropped_by_growth)} priority-0 observation(s) "
            "dropped."
        )
        return calendar

    @staticmethod
    def _growth_rank(seq: ObservationSequence) -> tuple[int, int]:
        """Where ``seq`` stands in growth under ``grow_by_priority``.

        The higher priority ranks higher; at equal priority a visit's main
        target outranks any other target. Within a visit this is the same
        as main target first, since a main target has its visit's highest
        priority. Across visits priority decides, so a priority-0 main
        target ranks below a priority-1 target of another visit. Higher
        ranks grow first and may take minutes from a lower-ranked neighbor;
        between equal ranks in different visits, the earlier visit may take
        from the later one (see ``_grow_one_side``).

        Parameters
        ----------
        seq : ObservationSequence
            The observation.

        Returns
        -------
        tuple of int
            ``(priority, 1 if main target else 0)``; compare as tuples.
        """
        is_main = bool(getattr(seq, "is_main", False))
        return int(seq.priority or 0), int(is_main)

    def _grow_one_side(
        self,
        ordered: list[tuple[str, ObservationSequence]],
        index: int,
        direction: int,
        original_timing: dict[Any, Any],
    ) -> tuple[int, int]:
        """Grow one boundary of ``ordered[index]``.

        ``direction`` -1 grows the start earlier, +1 the stop later; see
        ``_grow_into_free_time`` for the rules. Returns the minutes gained
        and how many of those came out of a lower-ranked neighbor.
        """
        limit = getattr(self, "max_movement_minutes", 0) or 0
        visit_id, seq = ordered[index]
        original = original_timing[(visit_id, seq.id)]
        if direction < 0:
            edge, bound = seq.start_time, original[0] - limit * u.min
        else:
            edge, bound = seq.stop_time, original[1] + limit * u.min
        # The schedule window is a hard boundary. Room the movement limit
        # would give past it is only looked at, to report it.
        window_edge = getattr(
            self, "window_start" if direction < 0 else "window_end", None
        )
        past_window = None
        if (
            window_edge is not None
            and direction * (bound - window_edge).sec > 0
        ):
            past_window, bound = bound, window_edge
        own_bound = bound

        # The neighbor on this side, and how far into it the grower may
        # reach: not at all unless growth is by priority and the neighbor
        # ranks lower, and then only while the neighbor keeps its minimum
        # duration once the start-buffer pass has cleaned its opening, and
        # its moved boundary stays within the movement limit. With
        # drop_priority0 a lower-ranked priority-0 neighbor may be taken
        # past that floor, which drops it.
        neighbor = None
        takeable = may_drop = False
        dropped = self.__dict__.setdefault("_dropped_by_growth", set())
        neighbor_index = index + direction
        # A priority 0 already eaten in this pass is no longer a neighbor.
        while (
            0 <= neighbor_index < len(ordered)
            and (ordered[neighbor_index][0], ordered[neighbor_index][1].id)
            in dropped
        ):
            neighbor_index += direction
        if 0 <= neighbor_index < len(ordered):
            neighbor_visit, neighbor = ordered[neighbor_index]
            neighbor_original = original_timing.get(
                (neighbor_visit, neighbor.id)
            )
            # Lower ranked, or equal ranked in a later visit.
            visit_order = getattr(self, "_visit_order", {})
            takeable = (
                bool(getattr(self, "grow_by_priority", False))
                and neighbor_original is not None
                and (
                    self._growth_rank(neighbor),
                    -visit_order.get(neighbor_visit, 0),
                )
                < (self._growth_rank(seq), -visit_order.get(visit_id, 0))
            )
            reach = (
                neighbor.stop_time if direction < 0 else neighbor.start_time
            )
            if takeable:
                n_neighbor = int(np.rint(neighbor.duration.sec / 60.0))
                min_minutes = int(
                    np.rint(self.min_sequence_duration.sec / 60.0)
                )
                requirements = self._start_buffer_requirements(
                    neighbor,
                    neighbor.start_time + np.arange(n_neighbor) * u.min,
                    SkyCoord(
                        neighbor.ra, neighbor.dec, frame="icrs", unit="deg"
                    ),
                )
            if takeable and direction < 0:
                # Its stop may come back to where its cleaned opening plus
                # the minimum duration ends.
                opening = self._first_clean_start(requirements, n_neighbor)
                floor = max(
                    neighbor.start_time
                    + ((opening or 0) + min_minutes) * u.min,
                    neighbor_original[1] - limit * u.min,
                )
                reach = min(reach, floor)
            elif takeable:
                # Its start may go forward as far as the buffer pass, once
                # it has cleaned the new opening, can still leave it the
                # minimum.
                latest = neighbor_original[0] + limit * u.min
                take = min(
                    n_neighbor - min_minutes,
                    int(np.rint((latest - neighbor.start_time).sec) // 60),
                )
                while take > 0:
                    landing = self._first_clean_start(
                        [
                            (name, buf, ok[take:])
                            for name, buf, ok in requirements
                        ],
                        n_neighbor - take,
                    )
                    if (
                        landing is not None
                        and n_neighbor - take - landing >= min_minutes
                    ):
                        break
                    take -= 1
                reach = max(reach, neighbor.start_time + max(take, 0) * u.min)
            # With drop_priority0 the floor above only decides whether a
            # priority-0 neighbor survives; the grower may reach its far edge.
            floor_reach = reach
            may_drop = (
                takeable
                and bool(getattr(self, "drop_priority0", False))
                and int(neighbor.priority or 0) == 0
            )
            if may_drop:
                reach = (
                    neighbor.start_time
                    if direction < 0
                    else neighbor.stop_time
                )
            bound = max(bound, reach) if direction < 0 else min(bound, reach)
        # The window, not a neighbor, is what stops this side.
        held_by_window = (
            past_window is not None and abs((bound - window_edge).sec) < 1e-3
        )

        # Whole seconds before the floor division: a bound built from the
        # neighbor's long-term time can come out a nanosecond short of the
        # minute and would otherwise lose a whole minute of room.
        room = int(np.rint((direction * (bound - edge)).sec) // 60)
        if room <= 0:
            if held_by_window:
                self._warn_growth_past_window(
                    visit_id, seq, direction, window_edge, past_window
                )
            return 0, 0
        if direction < 0:
            times = edge - np.arange(1, room + 1) * u.min
        else:
            times = edge + np.arange(room) * u.min
        target_coord = SkyCoord(seq.ra, seq.dec, frame="icrs", unit="deg")
        model = self._visibility_for(seq)
        gained = self._growable_minutes(
            self._visibility_for_sequence(seq, target_coord, times),
            times,
            target_coord,
            seq.roll,
            visibility=model,
        )
        if held_by_window and gained == room:
            self._warn_growth_past_window(
                visit_id, seq, direction, window_edge, past_window
            )
        if gained <= 0:
            return 0, 0
        new_edge = edge + direction * gained * u.min
        if direction < 0:
            seq.start_time = new_edge
        else:
            seq.stop_time = new_edge
        boundary = "start" if direction < 0 else "stop"
        if neighbor is None:
            self._note_timing(
                visit_id,
                seq.id,
                f"{boundary}: grew {gained} min into idle time",
            )
            return gained, 0

        prefix = self._seq_prefix(visit_id, seq)
        taken = 0
        if direction < 0 and new_edge < neighbor.stop_time:
            taken = int(np.rint((neighbor.stop_time - new_edge).sec / 60.0))
        elif direction > 0 and new_edge > neighbor.start_time:
            taken = int(np.rint((new_edge - neighbor.start_time).sec / 60.0))
        # Whole seconds, so landing exactly on the floor is not a drop.
        if may_drop and np.rint(direction * (new_edge - floor_reach).sec) > 0:
            # Past its floor the priority 0 cannot fly what is left of it.
            dropped.add((neighbor_visit, neighbor.id))
            self._print(
                f"WARNING: {prefix} | GROWTH: took {taken} min from "
                f"priority-0 {neighbor.target}, leaving it unable to fly; "
                f"dropped {self._seq_prefix(neighbor_visit, neighbor)} "
                "(drop_priority0)."
            )
            self._note_timing(
                visit_id,
                seq.id,
                f"{boundary}: grew {gained} min, {taken} of them taken from "
                f"priority-0 {neighbor.target}, which was dropped",
            )
            return gained, taken
        if direction < 0 and taken:
            neighbor.stop_time = new_edge
        elif taken:
            neighbor.start_time = new_edge
        if taken:
            self._print(
                f"{prefix} | GROWTH: took {taken} min from lower-ranked "
                f"{neighbor.target} (priority {neighbor.priority})."
            )
            self._note_timing(
                visit_id,
                seq.id,
                f"{boundary}: grew {gained} min, {taken} of them taken from "
                f"lower-ranked {neighbor.target}",
            )
            self._note_timing(
                neighbor_visit,
                neighbor.id,
                f"{'stop' if direction < 0 else 'start'}: gave {taken} min "
                f"to higher-ranked {seq.target}",
            )
        else:
            self._note_timing(
                visit_id,
                seq.id,
                f"{boundary}: grew {gained} min into idle time",
            )

        # Growth that reached a bound set by a lower-ranked neighbor's
        # floor, with visibility allowing more, is reported. A bound set by
        # the grower's own movement limit, or by an equal or higher
        # neighbor, is the rule working as intended.
        if not (takeable and gained == room and bound != own_bound):
            return gained, taken
        extra_room = int(np.rint((direction * (own_bound - bound)).sec) // 60)
        if extra_room <= 0:
            return gained, taken
        if direction < 0:
            extra_times = bound - np.arange(1, extra_room + 1) * u.min
        else:
            extra_times = bound + np.arange(extra_room) * u.min
        extra = self._growable_minutes(
            self._visibility_for_sequence(seq, target_coord, extra_times),
            extra_times,
            target_coord,
            seq.roll,
            visibility=model,
        )
        if extra > 0:
            side = "before its start" if direction < 0 else "after its stop"
            self._print(
                f"ERROR: {prefix} | GROWTH BLOCKED: visibility allows "
                f"{extra} more min {side}, but taking them would leave "
                f"lower-ranked {neighbor.target} (priority "
                f"{neighbor.priority}) under "
                f"{self.min_sequence_duration.sec / 60:.0f} min once its "
                f"opening is cleaned, or past its {limit} min movement "
                "limit."
            )
        return gained, taken

    def _warn_growth_past_window(
        self,
        visit_id: Any,
        seq: ObservationSequence,
        direction: int,
        window_edge: Time,
        movement_bound: Time,
    ) -> None:
        """Warn that visibility would grow an observation past the window.

        Growth has already stopped at the window; this only reports the
        minutes, up to the movement limit, that the target stays visible
        beyond it. Reported once per boundary per run.

        Parameters
        ----------
        visit_id : str
            The observation's visit.
        seq : ObservationSequence
            The observation, its boundary sitting on the window edge.
        direction : int
            -1 for the start (window start), +1 for the stop (window end).
        window_edge : astropy.time.Time
            The window edge on this side.
        movement_bound : astropy.time.Time
            How far the movement limit alone would have let it grow.
        """
        reported = self.__dict__.setdefault("_window_reported", set())
        key = (visit_id, seq.id, direction)
        if key in reported:
            return
        extra_room = int(
            np.rint((direction * (movement_bound - window_edge)).sec) // 60
        )
        if extra_room <= 0:
            return
        if direction < 0:
            times = window_edge - np.arange(1, extra_room + 1) * u.min
        else:
            times = window_edge + np.arange(extra_room) * u.min
        coord = SkyCoord(seq.ra, seq.dec, frame="icrs", unit="deg")
        extra = self._growable_minutes(
            self._visibility_for_sequence(seq, coord, times),
            times,
            coord,
            seq.roll,
            visibility=self._visibility_for(seq),
        )
        if extra <= 0:
            return
        reported.add(key)
        side, name = (
            ("before its start", "window start")
            if direction < 0
            else ("after its stop", "window end")
        )
        self._print(
            f"WARNING: {self._seq_prefix(visit_id, seq)} | WINDOW: "
            f"visibility allows {extra} more min {side}, past the {name} "
            f"{window_edge.isot}; held to the window."
        )

    def _growable_minutes(
        self,
        visible: np.ndarray,
        times: Any,
        target_coord: SkyCoord,
        roll: float | None,
        visibility: Any = None,
    ) -> int:
        """How many of ``times`` an observation may absorb, walking outward.

        Visible minutes are taken. A section of non-visibility is stepped
        over when it is short enough to tolerate (the same judgement
        ``_trim_to_longest_visible_block`` applies to dark stretches already
        inside an observation), so that a dip the tolerances accept does not
        block growth through it while being kept when it happens to fall
        inside. An intolerable run stops the walk.

        The count returned always ends on a visible minute, so growth never
        leaves a dark tail hanging off the end of an observation.
        """
        taken = 0
        index = 0
        while index < len(visible):
            if visible[index]:
                index += 1
                taken = index
                continue

            run_end = index
            while run_end < len(visible) and not visible[run_end]:
                run_end += 1
            if run_end >= len(visible):
                # Dark all the way to the bound; nothing worth reaching.
                break
            if not self._is_gap_tolerable(
                target_coord,
                times,
                index,
                run_end - index,
                roll=roll,
                visibility=visibility,
            ):
                break
            index = run_end

        return taken

    def _clamp_movement(
        self,
        calendar: ScienceCalendar,
        original_timing: dict[Any, Any],
    ) -> ScienceCalendar:
        """Hold every boundary within ``max_movement_minutes`` of its plan.

        The short-term scheduler exists to absorb a stale TLE, so an
        observation is expected to shift by a few minutes. Visibility can
        nonetheless argue for moving one much further.

        This is, by default, not allowed as a change this big should be
        rectified in the long-term scheduler not the short-term.
        Therefore, the boundary is clamped to the limit and the observation
        is reported.

        Times are modified in place. A clamp that would leave the
        observation shorter than ``min_sequence_duration`` is not applied.
        """
        limit = getattr(self, "max_movement_minutes", 0) or 0
        if limit <= 0:
            return calendar
        clamped = 0

        for visit in calendar.visits:
            for seq in visit.sequences:
                original = original_timing.get((visit.id, seq.id))
                if original is None:
                    continue

                drift_start = (seq.start_time - original[0]).sec / 60.0
                drift_stop = (seq.stop_time - original[1]).sec / 60.0
                new_start, new_stop = seq.start_time, seq.stop_time
                prefix = self._seq_prefix(visit.id, seq)

                # The tolerance keeps a boundary sitting exactly on the
                # limit from tripping on floating-point time arithmetic.
                if abs(drift_start) > limit + 1e-6:
                    new_start = (
                        original[0] + np.sign(drift_start) * limit * u.min
                    )
                    self._print(
                        f"ERROR: {prefix} | MOVED TOO FAR: start wanted to "
                        f"shift {drift_start:+.0f} min (limit {limit}); "
                        f"clamped to {np.sign(drift_start) * limit:+.0f} "
                        f"min. Visibility at the clamped time needs review."
                    )
                if abs(drift_stop) > limit + 1e-6:
                    new_stop = (
                        original[1] + np.sign(drift_stop) * limit * u.min
                    )
                    self._print(
                        f"ERROR: {prefix} | MOVED TOO FAR: stop wanted to "
                        f"shift {drift_stop:+.0f} min (limit {limit}); "
                        f"clamped to {np.sign(drift_stop) * limit:+.0f} "
                        f"min. Visibility at the clamped time needs review."
                    )

                if new_start == seq.start_time and new_stop == seq.stop_time:
                    continue
                if self._below_minimum_duration(new_stop - new_start):
                    self._print(
                        f"ERROR: {prefix} | MOVED TOO FAR: clamping would "
                        f"leave under "
                        f"{self.min_sequence_duration.sec / 60:.0f} min. "
                        f"Left unclamped for manual review."
                    )
                    continue
                if not self._transit_allows(
                    calendar,
                    visit.id,
                    seq,
                    new_start,
                    new_stop,
                    "clamping to the movement limit",
                ):
                    continue
                for boundary, moved in (
                    ("start", new_start != seq.start_time),
                    ("stop", new_stop != seq.stop_time),
                ):
                    if moved:
                        self._note_timing(
                            visit.id,
                            seq.id,
                            f"{boundary}: clamped to the {limit} min "
                            "movement limit",
                        )
                seq.start_time, seq.stop_time = new_start, new_stop
                clamped += 1

        self.gap_report["processing_summary"]["boundaries_clamped"] = clamped
        return calendar

    def _repair_overlaps(self, calendar: ScienceCalendar) -> ScienceCalendar:
        """Guarantee the delivered calendar contains no overlaps.

        The passes above are written so they cannot produce one, so anything
        found here is a bug in this module. It is repaired rather than
        delivered, because an overlapping calendar cannot be flown: the
        earlier observation's stop is pulled back to the later one's start.
        Every repair is reported, and so is any overlap still standing
        afterwards, since that one needs fixing by hand.

        Times are modified in place.
        """
        ordered = self._ordered_sequences(calendar)
        repaired = 0

        for (visit_id, earlier), (_, later) in zip(ordered, ordered[1:]):
            overlap_sec = (earlier.stop_time - later.start_time).sec
            if overlap_sec <= 1.0:
                continue

            prefix = self._seq_prefix(visit_id, earlier)
            if self._below_minimum_duration(
                later.start_time - earlier.start_time
            ):
                self._print(
                    f"ERROR: {prefix} | OVERLAP: overlaps {later.target} by "
                    f"{overlap_sec / 60:.1f} min and cannot be truncated "
                    f"without dropping under "
                    f"{self.min_sequence_duration.sec / 60:.0f} min. "
                    f"Needs a manual fix."
                )
                continue

            self._print(
                f"ERROR: {prefix} | OVERLAP: overlapped {later.target} by "
                f"{overlap_sec / 60:.1f} min; stop truncated to "
                f"{later.start_time.isot}. This is a scheduler bug, please "
                f"report it."
            )
            self._transit_allows(
                calendar,
                visit_id,
                earlier,
                earlier.start_time,
                later.start_time,
                "truncating to end an overlap",
                refuse=False,
            )
            earlier.stop_time = later.start_time
            repaired += 1
            self._note_timing(
                visit_id,
                earlier.id,
                f"stop: truncated to end an overlap with {later.target}",
            )

        if repaired:
            residual = self.validate_no_overlaps_astropy(
                calendar, report_issues=False
            )
            if residual:
                self._print(
                    f"ERROR: {len(residual)} overlap(s) still present after "
                    f"repairing {repaired}; these need a manual fix."
                )

        self.gap_report["processing_summary"]["overlaps_repaired"] = repaired
        return calendar

    def _refine_edges_to_second(
        self,
        calendar: ScienceCalendar,
        original_timing: dict[Any, Any],
        added_only: bool = False,
    ) -> ScienceCalendar:
        """Move each start and stop to the second visibility puts it at.

        Every other pass samples visibility once a minute, so a target can
        leave the keepouts anywhere inside an observation's first or last
        minute without a sample seeing it, and a boundary stopped at a dark
        minute sample may have had most of the minute before that sample
        visible. Each boundary is rechecked at 1 s over the minute either
        side of it, at the roll the observation flies:

        - In: a stop whose last minute opens visible is pulled back to the
          first dark second in that minute, and a start whose first minute
          opens visible is pushed past the last dark second in it.
        - Out: a boundary with no dark second inside that minute moves
          outward over the seconds the target stays visible, up to a
          minute, never past a neighbor, the schedule window, or
          ``max_movement_minutes`` from its long-term time. A visible
          second clears both start buffers,
          so a start moved earlier this way still opens cleanly.

        A boundary whose own minute sample is dark was left there by an
        earlier pass on purpose and is not touched. Pulling a boundary in
        is applied even when it costs a transit core a minute (an error),
        since those seconds cannot be observed. A cut that takes an
        observation under ``min_sequence_duration`` is a warning; with
        ``drop_priority0`` a priority 0 cut that short is dropped instead
        (and a visit left empty with it), while any other is kept short.
        Times are modified in place.

        Parameters
        ----------
        calendar : ScienceCalendar
            Calendar after overlap repair.
        original_timing : dict
            Long-term ``(start, stop)`` per ``(visit_id, sequence_id)``,
            for the movement limit. An observation without an entry has no
            movement limit.
        added_only : bool, optional
            Refine only high-priority additions (``is_added``); every other
            observation still bounds its neighbors but is not moved, as it
            was refined already. Default False.

        Returns
        -------
        ScienceCalendar
            The same calendar, boundaries refined.
        """
        limit = getattr(self, "max_movement_minutes", 0) or 0
        # The schedule window is a hard boundary for moving out.
        window_start = getattr(self, "window_start", None)
        window_end = getattr(self, "window_end", None)
        ordered = self._ordered_sequences(calendar)
        offsets = np.arange(-60, 60)
        seconds_in = seconds_out = moved_in = moved_out = 0
        # (visit_id, seq_id) of priority-0 observations a cut took under
        # the minimum, with drop_priority0 on.
        dropped = set()

        def room(seconds: float) -> int:
            # Whole seconds of free time, never negative or over a minute.
            return int(min(60, max(0, np.floor(seconds + 1e-6))))

        for index, (visit_id, seq) in enumerate(
            self._progress(
                ordered,
                desc="Refining edges to the second",
                total=len(ordered),
            )
        ):
            if added_only and not getattr(seq, "is_added", False):
                continue
            start, stop = seq.start_time, seq.stop_time
            coord = SkyCoord(seq.ra, seq.dec, frame="icrs", unit="deg")
            # One query: the minute either side of the start, then of the
            # stop. Slices are [start-60 s, start), [start, start+60 s),
            # [stop-60 s, stop), [stop, stop+60 s).
            deltas = np.concatenate([offsets, (stop - start).sec + offsets])
            vis = self._visibility_for_sequence(
                seq, coord, start + deltas * u.s
            )
            outside_start, inside_start = vis[:60], vis[60:120]
            inside_stop, outside_stop = vis[120:180], vis[180:]
            original = original_timing.get((visit_id, seq.id))

            new_start = start
            if inside_start[0]:
                if not inside_start.all():
                    later = int(np.flatnonzero(~inside_start)[-1]) + 1
                    new_start = start + later * u.s
                else:
                    free = 60
                    previous = index - 1
                    while (
                        previous >= 0
                        and (ordered[previous][0], ordered[previous][1].id)
                        in dropped
                    ):
                        previous -= 1
                    if previous >= 0:
                        free = room(
                            (start - ordered[previous][1].stop_time).sec
                        )
                    if original is not None and limit > 0:
                        free = min(
                            free,
                            room((start - (original[0] - limit * u.min)).sec),
                        )
                    if window_start is not None:
                        free = min(free, room((start - window_start).sec))
                    dark = np.flatnonzero(~outside_start)
                    visible = 60 - (int(dark[-1]) + 1) if dark.size else 60
                    new_start = start - min(visible, free) * u.s

            new_stop = stop
            if inside_stop[0]:
                if not inside_stop.all():
                    first_dark = int(np.flatnonzero(~inside_stop)[0])
                    new_stop = stop + (first_dark - 60) * u.s
                else:
                    free = 60
                    if index + 1 < len(ordered):
                        free = room(
                            (ordered[index + 1][1].start_time - stop).sec
                        )
                    if original is not None and limit > 0:
                        free = min(
                            free,
                            room(((original[1] + limit * u.min) - stop).sec),
                        )
                    if window_end is not None:
                        free = min(free, room((window_end - stop).sec))
                    dark = np.flatnonzero(~outside_stop)
                    visible = int(dark[0]) if dark.size else 60
                    new_stop = stop + min(visible, free) * u.s

            d_start = int(np.rint((new_start - start).sec))
            d_stop = int(np.rint((new_stop - stop).sec))
            if d_start == 0 and d_stop == 0:
                continue

            prefix = self._seq_prefix(visit_id, seq)
            if d_start > 0 or d_stop < 0:
                if self._below_minimum_duration(new_stop - new_start):
                    short = (
                        f"{prefix} | EDGES: the target is not visible at 1 s "
                        "resolution inside its first or last minute; cutting "
                        f"that leaves {(new_stop - new_start).sec / 60:.1f} "
                        "min, under "
                        f"{self.min_sequence_duration.sec / 60:.0f} min"
                    )
                    if int(seq.priority or 0) == 0 and getattr(
                        self, "drop_priority0", False
                    ):
                        dropped.add((visit_id, seq.id))
                        self._print(
                            f"WARNING: {short}. Dropped (drop_priority0)."
                        )
                        continue
                    self._print(f"WARNING: {short}.")
                self._transit_allows(
                    calendar,
                    visit_id,
                    seq,
                    new_start,
                    new_stop,
                    "cutting seconds the target is not visible for",
                    refuse=False,
                )

            for boundary, change, pulled_in in (
                ("start", d_start, d_start > 0),
                ("stop", d_stop, d_stop < 0),
            ):
                if change == 0:
                    continue
                if pulled_in:
                    seconds_in += abs(change)
                    moved_in += 1
                    why = "the target is not visible past there (1 s check)"
                else:
                    seconds_out += abs(change)
                    moved_out += 1
                    why = "the target is visible that far (1 s check)"
                self._note_timing(
                    visit_id,
                    seq.id,
                    f"{boundary}: {abs(change)} s "
                    f"{'later' if change > 0 else 'earlier'}, {why}",
                )
            seq.start_time, seq.stop_time = new_start, new_stop

        if dropped:
            for visit in calendar.visits:
                visit.sequences = [
                    seq
                    for seq in visit.sequences
                    if (visit.id, seq.id) not in dropped
                ]
            calendar.visits = [
                visit for visit in calendar.visits if visit.sequences
            ]

        # Added to, as the pass also runs on the high-priority additions.
        summary = self.gap_report["processing_summary"]
        summary["edge_seconds_cut"] = (
            summary.get("edge_seconds_cut", 0) + seconds_in
        )
        summary["edge_seconds_grown"] = (
            summary.get("edge_seconds_grown", 0) + seconds_out
        )
        summary["priority_0_dropped"] = summary.get(
            "priority_0_dropped", 0
        ) + len(dropped)
        self._print(
            f"Refined edges to the second"
            f"{' (high-priority additions)' if added_only else ''}: "
            f"{moved_in} boundaries pulled in "
            f"({seconds_in} s), {moved_out} pushed out ({seconds_out} s), "
            f"{len(dropped)} priority-0 observation(s) dropped."
        )
        return calendar

    def _enforce_start_buffers(
        self, calendar: ScienceCalendar
    ) -> ScienceCalendar:
        """Require clean pointing over each observation's opening minutes.

        The gap tolerances let a keepout violation be ridden out in the
        middle of an observation, but not at the start: the spacecraft
        cannot acquire good pointing without the star trackers, and an
        observation that opens with the boresight in the Earth is not worth
        starting. So the opening minutes, measured from the observation's
        start time rather than from when science begins after the
        pre-observation overhead, must be clean with no tolerance applied:

        - ``st_gap_tolerance_start_buffer`` minutes of star-tracker
          visibility, evaluated at the roll the observation will fly, and
        - ``earthlimb_gap_tolerance_start_buffer`` minutes clear of the
          boresight Earth-limb keepout, whichever limb model is configured.

        Both are enforced together, because clearing one can push the start
        into a violation of the other. Sequences that open dirty have their
        ``start_time`` moved forward (in place) to the earliest minute that
        clears both. When no minute in the sequence clears them, or trimming
        there would leave the sequence shorter than
        ``min_sequence_duration``, the sequence is left alone and the
        problem is written to the error log.

        This runs last among the passes that move a start time, so it has
        the final say. It only ever moves a start later, so it cannot
        create an overlap.
        """
        for visit in self._progress(
            calendar.visits,
            desc="Checking start buffers",
            total=len(calendar.visits),
        ):
            for seq in visit.sequences:
                n_mins = int(np.rint(seq.duration.sec / 60.0))
                if n_mins <= 0:
                    continue

                target_coord = SkyCoord(
                    seq.ra, seq.dec, frame="icrs", unit="deg"
                )
                times = seq.start_time + np.arange(n_mins) * u.min
                requirements = self._start_buffer_requirements(
                    seq, times, target_coord
                )
                if not requirements:
                    continue

                offset = self._first_clean_start(requirements, n_mins)
                if offset == 0:
                    continue

                prefix = self._seq_prefix(visit.id, seq)
                reasons = ", ".join(name for name, _, _ in requirements)
                if offset is None:
                    self._print(
                        f"ERROR: {prefix} | START BUFFER: no stretch of this "
                        f"observation opens cleanly ({reasons}); pointing "
                        f"will be unreliable. Left unchanged."
                    )
                    continue

                new_start = seq.start_time + offset * u.min
                if self._below_minimum_duration(seq.stop_time - new_start):
                    self._print(
                        f"ERROR: {prefix} | START BUFFER: does not open "
                        f"cleanly until {offset} min in ({reasons}), and "
                        f"trimming there would leave under "
                        f"{self.min_sequence_duration.sec / 60:.0f} min. "
                        f"Left unchanged."
                    )
                    continue
                if not self._transit_allows(
                    calendar,
                    visit.id,
                    seq,
                    new_start,
                    seq.stop_time,
                    f"moving the start {offset} min later to open cleanly "
                    f"({reasons})",
                ):
                    continue

                self._print(
                    f"{prefix} | START BUFFER: start moved later by "
                    f"{offset} min so the observation opens cleanly "
                    f"({reasons})."
                )
                self._note_timing(
                    visit.id,
                    seq.id,
                    f"start: moved {offset} min later to open cleanly "
                    f"({reasons})",
                )
                seq.start_time = new_start

        return calendar

    def _start_buffer_requirements(
        self,
        seq: ObservationSequence,
        times: Any,
        target_coord: SkyCoord,
    ) -> list[tuple[str, int, np.ndarray]]:
        """The per-minute checks ``_enforce_start_buffers`` holds ``seq`` to.

        Each is ``(reason, buffer_minutes, ok_per_minute)`` over ``times``,
        judged under the model the sequence's priority flies at its roll.
        Empty when no buffer is configured or a check cannot be asked of
        the model, in which case the opening is not judged at all.
        """
        st_buffer = int(getattr(self, "st_gap_tolerance_start_buffer", 0) or 0)
        earthlimb_buffer = int(
            getattr(self, "earthlimb_gap_tolerance_start_buffer", 0) or 0
        )
        if not getattr(self.visibility, "_st_constraint_active", False):
            st_buffer = 0
        model = self._visibility_for(seq)
        roll = None if seq.roll is None else seq.roll * u.deg

        requirements = []
        if st_buffer > 0:
            try:
                breakdown = model.get_star_tracker_breakdown(
                    target_coord, times, roll=roll
                )
            except Exception:
                breakdown = None
            if breakdown is not None:
                requirements.append(
                    (
                        "star trackers are not settled",
                        st_buffer,
                        np.atleast_1d(
                            np.asarray(breakdown["passed"]["combined"])
                        ),
                    )
                )
        if earthlimb_buffer > 0:
            try:
                clear = model.get_constraint(target_coord, "earthlimb", times)
            except Exception:
                clear = None
            if clear is not None:
                requirements.append(
                    (
                        "the boresight is inside the Earth limb",
                        earthlimb_buffer,
                        np.atleast_1d(np.asarray(clear)),
                    )
                )
        return requirements

    @staticmethod
    def _first_clean_start(
        requirements: list[tuple[str, int, np.ndarray]],
        n_mins: int,
    ) -> int | None:
        """Earliest offset at which every requirement holds for its buffer.

        Each requirement is ``(name, buffer_minutes, ok_per_minute)``. The
        offset is walked forward past any violation falling inside that
        requirement's window and rechecked against all of them, because
        satisfying one can drag the start into a violation of another. An
        observation shorter than a buffer simply has to be clean all the
        way to its stop, which the slicing handles on its own.

        Returns the offset, or ``None`` when no offset clears everything.
        """
        offset = 0
        while offset < n_mins:
            advanced = False
            for _, buffer_minutes, ok in requirements:
                window = ok[offset : offset + buffer_minutes]
                if window.size and not window.all():
                    # Jump past the last violation in this window; anything
                    # earlier would still leave it inside the buffer.
                    offset += int(np.flatnonzero(~window)[-1]) + 1
                    advanced = True
            if not advanced:
                return offset
        return None

    def _analyze_mid_sequence_visibility(
        self,
        seq: ObservationSequence,
    ) -> tuple[SkyCoord, Any, np.ndarray] | None:
        """Return per-minute visibility for one sequence, if useful."""
        n_mins = int(np.rint(seq.duration.sec / 60.0))
        if n_mins <= 0:
            return None

        target_coord = SkyCoord(seq.ra, seq.dec, frame="icrs", unit="deg")
        deltas = np.arange(n_mins) * u.min
        times = seq.start_time + deltas
        vis_arr = self._visibility_for_sequence(seq, target_coord, times)
        if np.all(vis_arr):
            return None
        return target_coord, times, vis_arr

    def _visibility_for_sequence(
        self,
        seq: ObservationSequence,
        target_coord: SkyCoord,
        times: Any,
    ) -> np.ndarray:
        """Per-minute visibility of ``seq`` at the roll it will fly.

        Judged under the model its priority flies. A sequence with no roll
        yet is asked at the model's own attitude.
        """
        model = self._visibility_for(seq)
        roll = None if seq.roll is None else seq.roll * u.deg
        return np.asarray(
            model.get_visibility(target_coord, times, roll=roll)["visible"]
        )

    def _transit_of(
        self, seq: ObservationSequence
    ) -> tuple[str, int, Time, Time] | None:
        """The transit a priority-2 observation is placed around.

        Returns ``(planet key, epoch, core_start, core_stop)`` for the
        transit nearest the observation's midpoint, or ``None`` when the
        transit constraint is off, the observation is not priority 2, or
        its planet has no ephemeris.
        """
        ephemerides = getattr(self, "transit_ephemerides", None)
        if (
            not ephemerides
            or not getattr(self, "transit_duration_constraint_min", 0)
            or seq.priority != 2
        ):
            return None
        key = planet_key(seq.target)
        if key not in ephemerides:
            return None
        epoch = transit_epoch(
            ephemerides[key], seq.start_time + seq.duration / 2
        )
        # The barycentric correction is the slow part; a transit's core
        # never changes, so it is worked out once.
        cores = self.__dict__.setdefault("_transit_cores", {})
        if (key, epoch) not in cores:
            cores[(key, epoch)] = transit_core(
                ephemerides[key], epoch, self.transit_middle_frac
            )
        return (key, epoch, *cores[(key, epoch)])

    def _transit_coverage(
        self,
        calendar: ScienceCalendar,
        transit: tuple[str, int, Time, Time],
        replace: tuple[ObservationSequence, Time, Time] | None = None,
    ) -> int:
        """Visible minutes of a transit's core its observations cover.

        Every priority-2 observation placed around ``transit`` counts, each
        judged at its own roll under the model its priority flies.
        ``replace`` gives one of them new start and stop times, to judge a
        change before it is made.
        """
        key, epoch, core_start, core_stop = transit
        n_minutes = int(np.floor((core_stop - core_start).to_value(u.min)))
        minutes = core_start + np.arange(n_minutes) * u.min
        covered = 0
        for visit in calendar.visits:
            for seq in visit.sequences:
                found = self._transit_of(seq)
                if found is None or found[:2] != (key, epoch):
                    continue
                start, stop = seq.start_time, seq.stop_time
                if replace is not None and seq is replace[0]:
                    start, stop = replace[1], replace[2]
                inside = (minutes >= start) & (minutes < stop)
                if not inside.any():
                    continue
                coord = SkyCoord(seq.ra, seq.dec, frame="icrs", unit="deg")
                covered += int(
                    self._visibility_for_sequence(
                        seq, coord, minutes[inside]
                    ).sum()
                )
        return covered

    def _transit_allows(
        self,
        calendar: ScienceCalendar,
        visit_id: Any,
        seq: ObservationSequence,
        new_start: Time,
        new_stop: Time,
        change: str,
        refuse: bool = True,
    ) -> bool:
        """Whether ``seq`` may move to ``new_start``-``new_stop``.

        A priority-2 observation's transit must keep
        ``transit_duration_constraint_min`` visible minutes of its core
        covered, and a transit the long-term calendar left short of that
        may not lose any. A change breaking this is refused and logged as
        a warning. With ``refuse`` False (overlap repair, which has to
        win) the break is logged as an error and the change allowed.
        """
        transit = self._transit_of(seq)
        if transit is None:
            return True
        needed = self.transit_duration_constraint_min
        before = self._transit_coverage(calendar, transit)
        after = self._transit_coverage(
            calendar, transit, (seq, new_start, new_stop)
        )
        if after >= needed or after >= before:
            return True

        message = (
            f"{self._seq_prefix(visit_id, seq)} | TRANSIT: {change} would "
            f"leave {after} visible min of the {seq.target} transit core "
            f"(epoch {transit[1]}, {transit[2].isot[:16]} to "
            f"{transit[3].isot[11:16]}) covered, down from {before}; "
            f"needs {needed:g}."
        )
        if not refuse:
            self._print(f"ERROR: {message} Applied anyway.")
            return True
        self._print(f"WARNING: {message} Refused.")
        summary = (getattr(self, "gap_report", None) or {}).get(
            "processing_summary"
        )
        if summary is not None:
            summary["transit_changes_refused"] = (
                summary.get("transit_changes_refused", 0) + 1
            )
        return False

    def _transit_coverages(
        self, calendar: ScienceCalendar
    ) -> dict[tuple[str, int], tuple[str, Time, int]]:
        """Every transit with priority-2 observations, and its coverage.

        Returns ``{(planet key, epoch): (target, core_start, covered
        minutes)}``; empty when the transit constraint is off.
        """
        coverages = {}
        for visit in calendar.visits:
            for seq in visit.sequences:
                transit = self._transit_of(seq)
                if transit is not None and transit[:2] not in coverages:
                    coverages[transit[:2]] = (
                        seq.target,
                        transit[2],
                        self._transit_coverage(calendar, transit),
                    )
        return coverages

    def _find_nonvisible_gaps(
        self,
        vis_arr: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Find contiguous non-visible runs as half-open index ranges."""
        gaps: list[tuple[int, int]] = []
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
        gaps: list[tuple[int, int]],
        gap_tolerable: list[bool],
    ) -> tuple[int, int] | None:
        """Return best [start, end) span separated by intolerable gaps."""
        segment_bounds: list[tuple[int, int]] = []
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
    ) -> ObservationSequence | None:
        """Create trimmed sequence if valid and changed from input."""
        new_start = seq.start_time + best_start * u.min
        new_stop = seq.start_time + best_end * u.min

        if self._below_minimum_duration(new_stop - new_start):
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
            roll=seq.roll,
            is_main=seq.is_main,
        )

    def _extend_previous_after_mid_trim(
        self,
        working_cal: ScienceCalendar,
        all_sequences: list[tuple[str, ObservationSequence]],
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
            roll=prev_seq.roll,
            is_main=prev_seq.is_main,
        )
        working_cal.replace_sequence(prev_visit_id, prev_seq.id, extended_prev)
        all_sequences[idx - 1] = (prev_visit_id, extended_prev)
        self._note_timing(
            prev_visit_id,
            prev_seq.id,
            f"stop: grew {extend_end} min into time freed by a neighbor's "
            "trim",
        )

    def _extend_next_after_mid_trim(
        self,
        working_cal: ScienceCalendar,
        all_sequences: list[tuple[str, ObservationSequence]],
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
            next_seq, next_coord, gap_times
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
            roll=next_seq.roll,
            is_main=next_seq.is_main,
        )
        working_cal.replace_sequence(next_visit_id, next_seq.id, extended_next)
        all_sequences[idx + 1] = (next_visit_id, extended_next)
        self._note_timing(
            next_visit_id,
            next_seq.id,
            f"start: grew {last_idx + 1 - first_contiguous} min into time "
            "freed by a neighbor's trim",
        )

    def get_minute_by_minute_assignments(
        self, calendar: ScienceCalendar
    ) -> dict[str, Any]:
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
        extra_kwargs: dict[str, Any] | None = None,
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
        kwargs: dict[str, Any] = dict(extra_kwargs or {})
        # Share the run logger so the data class's own warnings (zero frame
        # time, oversize, VITL fallback, ...) land in the same log.
        kwargs.setdefault("logger", getattr(self, "logger", None))
        writeback: dict[str, str] = {}
        missing: list[str] = []

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
    def _normalize_priority_keys(
        raw: dict[Any, Any] | None,
    ) -> dict[int, Any]:
        """Coerce override priority keys to ints (accepts 'Priority_0', '0')."""
        out: dict[int, Any] = {}
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
        mapping: dict[str, Any],
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

    # Kept as a method; the rule lives in models so roll.py can share it.
    _clean_name = staticmethod(clean_target_name)

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
        has_dec1 = (
            dec_parent is not None and dec_parent.find("Dec1") is not None
        )
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
        overhead: OverheadTiming | None = None,
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

        visda, info = self._build_payload_data(
            sequence,
            override_fields,
            VisdaData,
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
        if not success:
            self._print(
                f"Warning: {prefix} | Failed to update "
                f"NumTotalFramesRequested"
            )
        elif str(old_frames) != str(int(frames)):
            self._print(
                f"{prefix} | VISDA NumTotalFramesRequested "
                f"'{old_frames}' -> '{int(frames)}'"
            )
        return sequence

    def _update_NIRDA_integrations(
        self,
        sequence: ObservationSequence,
        duration: TimeDelta,
        overhead: OverheadTiming | None = None,
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
        if not success:
            self._print(
                f"Warning: {prefix} | Failed to update SC_Integrations"
            )
        elif str(old_integrations) != str(int(integrations)):
            self._print(
                f"{prefix} | NIRDA SC_Integrations "
                f"'{old_integrations}' -> '{int(integrations)}'"
            )

        return sequence

    def validate_visibility(
        self, calendar: ScienceCalendar, report_issues: bool = True
    ) -> list[dict[str, Any]]:
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
            for seq in visit.sequences:
                n_mins = int(np.rint(seq.duration.sec / 60.0))
                target_coord = SkyCoord(
                    seq.ra, seq.dec, frame="icrs", unit="deg"
                )
                deltas = np.arange(n_mins) * u.min
                times = seq.start_time + deltas

                model = self._visibility_for(seq)
                vis = self._visibility_for_sequence(seq, target_coord, times)

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
                    roll_used = seq.roll
                    constraint_details = {}
                    # The star tracker keepouts depend on roll, so every
                    # verdict below is asked for the roll this observation
                    # will actually fly. Left unset they would describe the
                    # model's own attitude, and could contradict the
                    # visibility result they are meant to explain.
                    roll_quantity = (
                        None if roll_used is None else roll_used * u.deg
                    )
                    try:
                        fail_time = times[non_vis_indices[0]]
                        constraint_failures = model.get_all_constraints(
                            target_coord,
                            fail_time,
                            roll=roll_quantity,
                        )
                        # Capture actual separations and limits
                        try:
                            seps = model.get_separations(
                                target_coord, fail_time
                            )
                            vis_obj = model
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
                                        # Read the geometry from the model
                                        # rather than rebuilding it here.
                                        pre = vis_obj._precompute(fail_time)
                                        zenith_u = pre["zenith_unit"]
                                        la_rad = pre["limb_angle_rad"]
                                        sun_u = pre["body_units"]["sun"]
                                        tgt_u = vis_obj._target_unit(
                                            target_coord, fail_time
                                        )
                                        eff_deg = float(
                                            vis_obj._effective_earthlimb_min_deg(
                                                tgt_u,
                                                zenith_u,
                                                sun_u,
                                                limb_angle_rad=la_rad,
                                            )
                                        )
                                        # Say which branch produced it.
                                        if getattr(
                                            vis_obj,
                                            "use_dynamic_earthlimb",
                                            False,
                                        ):
                                            illum = float(
                                                vis_obj._daynight_illumination_angle(
                                                    tgt_u,
                                                    zenith_u,
                                                    sun_u,
                                                    limb_angle_rad=la_rad,
                                                )
                                            )
                                            side = f"illum {illum:.1f} deg"
                                        else:
                                            side = (
                                                "day"
                                                if bool(
                                                    vis_obj._daynight_is_sunlit(
                                                        tgt_u,
                                                        zenith_u,
                                                        sun_u,
                                                        limb_angle_rad=la_rad,
                                                    )
                                                )
                                                else "night"
                                            )
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
                            # Star tracker details.
                            if getattr(
                                vis_obj, "_st_constraint_active", False
                            ):
                                try:
                                    breakdown = (
                                        vis_obj.get_star_tracker_breakdown(
                                            target_coord,
                                            fail_time,
                                            roll=roll_quantity,
                                        )
                                    )
                                    for row, passes in breakdown[
                                        "passed"
                                    ].items():
                                        if row in ("ST1", "ST2", "combined"):
                                            continue
                                        tracker, name = row.split(" ", 1)
                                        label = f"{tracker.lower()}_{name}"
                                        constraint_details[label] = {
                                            "passes": bool(passes),
                                            "required_deg": float(
                                                breakdown["limits"][row]
                                                .to(u.deg)
                                                .value
                                            ),
                                            "actual_deg": float(
                                                breakdown["separations"][row]
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
        day_seconds: dict[str, float] = defaultdict(float)
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
        nirda: NirdaData | None,
        sc_integrations: int,
        visda: VisdaData | None,
        num_total_frames: int,
    ) -> list[str]:
        """Return the FITS product filenames packaged in a sequence's ``.bin``.

        Names follow the Pandora flight-software conventions (InfImg from
        ``MACIEMain.cpp``, VisSci from ``PCOCameraMain.cpp``) built from the
        observation's payload parameters. The per-detector capture-start
        timestamps are the sequence start plus the detector pre-overhead.
        """
        files: list[str] = []
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
        output_path: Any | None = None,
        pass_data_volume_mb: float | None = None,
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

        def new_bucket() -> dict[str, Any]:
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

        daily: dict[str, dict[str, Any]] = {}

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
                        sc_integrations * nirda.other_integration_saved_frames
                    )
                    nir_unc = (
                        (sc_integrations * nirda.integration_data)
                        .to(u.byte)
                        .value
                    )
                    bucket["nir_frames"] += int(nir_frames)
                    bucket["nir_data_unc"] += nir_unc
                    bucket["nir_data"] += nir_unc * nirda.compression_ratio

                # VISDA frames (coadds) + data from the VisdaData model.
                visda, _ = self._build_payload_data(
                    seq,
                    (),
                    VisdaData,
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
        base = (
            Path(output_path)
            if output_path is not None
            else getattr(calendar, "source_path", None)
        )
        if base is not None:
            diag_path = base.with_suffix(".diag")
            diag_path.write_text(text, encoding="utf-8")
            self._print(f"Wrote diagnostics to {diag_path}")

        return text

    @staticmethod
    def _diag_observing_and_gaps(
        timelines: list[tuple],
    ) -> tuple[float, float]:
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
            return (
                "No observations were available for diagnostic generation.\n"
            )

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

        lines: list[str] = []

        # ── Week summary ───────────────────────────────────────────
        sum_total = summary["nir_data"] + summary["vis_data"]
        sum_total_unc = summary["nir_data_unc"] + summary["vis_data_unc"]
        sum_span = summary["observing"] + summary["gaps"]
        lines.append(f"Calendar Summary {sorted_days[0]} : {sorted_days[-1]}")
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
    ) -> list[dict[str, Any]]:
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
    def _settings_for_header(self) -> dict[str, str]:
        """The keepouts and tolerances this run actually applied.

        Read back off the ``Visibility`` instances rather than off the
        constructor arguments, so a keepout the caller left unset is
        reported as the value ``pandoravisibility`` supplied rather than
        as blank.

        Returns
        -------
        dict
            Header attribute name to formatted value. Angles are in
            degrees and tolerances in minutes.
        """

        def angle(model, name):
            value = getattr(model, name, None) if model is not None else None
            if value is None:
                return None
            return f"{float(getattr(value, 'value', value)):g}"

        nominal = self.visibility
        settings = {
            "Sun_Min_Deg": angle(nominal, "sun_min"),
            "Moon_Min_Deg": angle(nominal, "moon_min"),
            "Earthlimb_Min_Deg": angle(nominal, "earthlimb_min"),
            "Earthlimb_Day_Min_Deg": angle(nominal, "earthlimb_day_min"),
            "Earthlimb_Night_Min_Deg": angle(nominal, "earthlimb_night_min"),
            "ST_Sun_Min_Deg": angle(nominal, "st_sun_min"),
            "ST_Moon_Min_Deg": angle(nominal, "st_moon_min"),
            "ST_Earthlimb_Min_Deg": angle(nominal, "st_earthlimb_min"),
            "Use_Dynamic_Earthlimb": str(
                bool(getattr(nominal, "use_dynamic_earthlimb", False))
            ),
            "Occultation_Earthlimb_Min_Deg": angle(
                getattr(self, "occultation_visibility", None), "earthlimb_min"
            ),
            "Occultation_Overrides": (
                ",".join(self.occultation_overrides)
                if getattr(self, "occultation_overrides", None)
                else None
            ),
            "Earthlimb_Gap_Tolerance_Min": str(self.earthlimb_gap_tolerance),
            "Earthlimb_Gap_Tolerance_Start_Buffer_Min": str(
                self.earthlimb_gap_tolerance_start_buffer
            ),
            "ST_Gap_Tolerance_Min": str(self.st_gap_tolerance),
            "ST_Gap_Tolerance_Start_Buffer_Min": str(
                self.st_gap_tolerance_start_buffer
            ),
            "Max_Movement_Min": str(self.max_movement_minutes),
            "Grow_By_Priority": str(self.grow_by_priority),
            "Drop_Priority0": str(getattr(self, "drop_priority0", False)),
            "Refine_Edges_To_Second": str(
                getattr(self, "refine_edges_to_second", False)
            ),
            "Roll_Step_Deg": f"{float(self.roll_step):g}",
            "Min_Power_Frac": f"{float(self.min_power_frac):g}",
            "Use_Trace_Sweep": str(
                bool(getattr(nominal, "use_trace_sweep", False))
            ),
        }
        if getattr(nominal, "use_trace_sweep", False):
            settings["Trace_Contamination_Tol"] = (
                f"{float(nominal.trace_contamination_tol):g}"
            )
            settings["Trace_NIRDA_ROI"] = (
                f"{nominal.nirda_roi_x}x{nominal.nirda_roi_y}"
            )
            settings["Drop_Observations_Based_On_Trace"] = str(
                bool(getattr(self, "drop_observations_based_on_trace", False))
            )
        transit_on = bool(
            getattr(self, "transit_ephemerides", None)
            and getattr(self, "transit_duration_constraint_min", 0)
        )
        settings["Transit_Constraint"] = str(transit_on)
        if transit_on:
            settings["Transit_Middle_Frac"] = f"{self.transit_middle_frac:g}"
            settings["Transit_Duration_Constraint_Min"] = (
                f"{self.transit_duration_constraint_min:g}"
            )
        adds_on = bool(getattr(self, "add_high_pri_observations", False))
        settings["Add_High_Pri_Observations"] = str(adds_on)
        if adds_on:
            if self.high_priority_adds_earthlimb_min is not None:
                settings["High_Priority_Adds_Earthlimb_Min_Deg"] = (
                    f"{float(self.high_priority_adds_earthlimb_min):g}"
                )
            settings["High_Priority_Adds_Min_Duration_Min"] = (
                f"{self.high_priority_adds_min_duration:g}"
            )
            settings["High_Priority_Adds_Max_Duration_Min"] = (
                f"{self.high_priority_adds_max_duration:g}"
            )
            settings["High_Priority_Adds_Min_Observations"] = str(
                self.high_priority_adds_min_observations
            )
            settings["High_Priority_Adds_Transit_Buffer_Min"] = (
                f"{self.high_priority_adds_transit_buffer_min:g}"
            )
            if getattr(self, "high_priority_adds_exclude", None):
                settings["High_Priority_Adds_Exclude"] = ",".join(
                    self.high_priority_adds_exclude
                )
        # A keepout with no value is left out rather than written empty:
        # Occultation_Earthlimb_Min_Deg absent means it was not in use.
        return {k: v for k, v in settings.items() if v is not None}

    @property
    def run_error_count(self) -> int:
        """Errors logged during the most recent :meth:`process_calendar`.

        Zero for a processor that has not run, or one whose messages went
        to the builtin ``print`` because no run logger was configured.
        """
        counter = getattr(self, "_error_counter", None)
        return 0 if counter is None else counter.count

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
        log_path: Any | None = None,
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

        # Counts errors so the delivered calendar can be marked invalid based
        # on if/what went wrong during the run. Reset once per run.
        self._error_counter = _ErrorCountingHandler()
        logger.addHandler(self._error_counter)

        # Resolve the log file base path.
        base = (
            Path(log_path)
            if log_path is not None
            else getattr(calendar, "source_path", None)
        )
        if base is not None:
            log_file = base.with_suffix(".log")
            errors_file = base.with_suffix(".errors.log")

            # Write a header recording the run start time in UTC and US
            # Eastern, so individual entries need not carry timestamps. The
            # FileHandler below then appends to this file.
            now_utc = datetime.now(UTC)
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

    def _progress(self, iterable, desc: str, total: int | None = None):
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
                "sequences_modified": 0,
                "sequences_lengthened": 0,
                "sequences_shortened": 0,
                "minutes_grown_at_starts": 0,
                "minutes_grown_at_stops": 0,
                "minutes_taken_from_lower_priority": 0,
                "priority_0_dropped": 0,
                "trace_dropped": 0,
                "transit_changes_refused": 0,
                "boundaries_clamped": 0,
                "overlaps_repaired": 0,
                "window_clipped": 0,
                "merged_before_trims": 0,
                "merged_at_end": 0,
                "edge_seconds_cut": 0,
                "edge_seconds_grown": 0,
                "high_pri_added_visits": 0,
                "high_pri_added_observations": 0,
                "high_pri_added_minutes": 0,
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

    def get_gap_report(self) -> dict[str, Any]:
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
        self._print(
            f"  Sequences Modified: "
            f"{summary.get('sequences_modified', 0)} "
            f"({summary.get('sequences_lengthened', 0)} grown, "
            f"{summary.get('sequences_shortened', 0)} trimmed)"
        )
        self._print(
            f"  Grown Into Idle: "
            f"{summary.get('minutes_grown_at_starts', 0)} min at starts, "
            f"{summary.get('minutes_grown_at_stops', 0)} min at stops"
        )
        self._print(
            f"  Boundaries Clamped: "
            f"{summary.get('boundaries_clamped', 0)} "
            f"(over the {getattr(self, 'max_movement_minutes', 0)} min "
            f"movement limit)"
        )
        self._print(
            f"  Overlaps Repaired: {summary.get('overlaps_repaired', 0)}"
        )
        self._print(
            f"  Cut At The Window Edges: {summary.get('window_clipped', 0)}"
        )
        self._print(
            f"  Edges Refined To The Second: "
            f"{summary.get('edge_seconds_cut', 0)} s cut where the target "
            f"is not visible, {summary.get('edge_seconds_grown', 0)} s grown"
        )

    def debug_sequence_visibility(
        self,
        calendar: ScienceCalendar,
        sequence_id: str,
        target_name: str | None = None,
    ) -> dict[str, Any]:
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

        vis = self._visibility_for(target_seq).get_visibility(
            target_coord, times
        )["visible"]

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
    ) -> list[dict[str, Any]]:
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
    ) -> dict[str, Any]:
        """Comprehensive timing validation.

        Checks for overlaps, short sequences, observations reaching outside
        the schedule window (once ``process_calendar`` has set one), and
        large gaps. Each sub-issue includes a ``message`` with actionable
        detail.

        Returns
        -------
        dict
            Keys: ``overlaps``, ``short_sequences``, ``outside_window``,
            ``large_gaps``, ``timing_summary``.
        """
        issues: dict[str, Any] = {
            "overlaps": [],
            "short_sequences": [],
            "outside_window": [],
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
            if self._below_minimum_duration(seq_info["sequence"].duration):
                seq = seq_info["sequence"]
                message = (
                    f"Seq {seq.id} ({seq.target}) in visit "
                    f"{seq_info['visit_id']}: duration "
                    f"{dur_min:.1f} min < minimum "
                    f"{min_dur_min:.0f} min. "
                    f"Extend stop_time to at least "
                    f"{(seq.start_time + min_duration).isot}."
                )
                # Observations that are too short need to be flagged as an
                # error. They will not be deleted by the scheduler, but
                # likely need to be deleted before being passed to the MOC.
                self._print(f"ERROR: {message}")
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

        # Nothing may reach outside the schedule window. Extraction and
        # growth both hold to it, so a hit here is a scheduler bug.
        window_start = getattr(self, "window_start", None)
        window_end = getattr(self, "window_end", None)
        if window_start is not None and window_end is not None:
            for seq_info in all_sequences:
                seq = seq_info["sequence"]
                before = (window_start - seq.start_time).sec
                after = (seq.stop_time - window_end).sec
                if before <= 1e-3 and after <= 1e-3:
                    continue
                message = (
                    f"Seq {seq.id} ({seq.target}) in visit "
                    f"{seq_info['visit_id']}: "
                    f"{seq.start_time.isot} - {seq.stop_time.isot} reaches "
                    f"outside the window {window_start.isot} - "
                    f"{window_end.isot}."
                )
                self._print(f"ERROR: {message}")
                issues["outside_window"].append(
                    {
                        "sequence_id": seq.id,
                        "target": seq.target,
                        "visit_id": seq_info["visit_id"],
                        "start_time": seq.start_time,
                        "stop_time": seq.stop_time,
                        "message": message,
                    }
                )

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
            "outside_window_found": len(issues["outside_window"]),
            "large_gaps_found": len(issues["large_gaps"]),
            "total_issues": len(issues["overlaps"])
            + len(issues["short_sequences"])
            + len(issues["outside_window"])
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
            self._print(
                f"Total timing issues found: " f"{summary['total_issues']}"
            )
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

            if issues["outside_window"]:
                self._print(
                    f"OUTSIDE THE WINDOW "
                    f"({len(issues['outside_window'])} found):"
                )
                for i, out in enumerate(issues["outside_window"]):
                    self._print(f"  {i+1}. {out['message']}")
            else:
                self._print("\u2713 OUTSIDE THE WINDOW: None found")

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
    ) -> list[dict[str, Any]]:
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
    ) -> list[dict[str, Any]]:
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
    ) -> list[dict[str, Any]]:
        """Validate roll-angle consistency per target within each visit.

        Returns
        -------
        list of dict
            Issue dicts with ``message``, ``suggested_roll``, and
            per-sequence ``roll_map``.
        """
        issues = []

        for visit in calendar.visits:
            target_sequences: dict[str, list[ObservationSequence]] = {}
            for seq in visit.sequences:
                if seq.target not in target_sequences:
                    target_sequences[seq.target] = []
                target_sequences[seq.target].append(seq)

            for target, sequences in target_sequences.items():
                if len(sequences) < 2:
                    continue

                roll_values = []
                seq_ids = []
                roll_map: dict[str, float] = {}
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
        self, category: str, item: dict[str, Any]
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
    ) -> dict[str, Any]:
        """Run all validators and print a unified actionable report.

        Returns
        -------
        dict
            ``{"status": "VALID"|"INVALID", "counts": {...},
            "details": {...}}`` where *details* maps each category
            to the raw issue list.
        """
        results: dict[str, Any] = {}
        counts: dict[str, int] = {}

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
            if summary["outside_window_found"]:
                self._print(
                    f"  - {summary['outside_window_found']} sequences "
                    "outside the window"
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
