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
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, SkyCoord
from astropy.coordinates import get_body
from astropy.time import Time, TimeDelta
from pandoravisibility import Visibility

from .models import ObservationSequence, ScienceCalendar, Visit
from .roll import apply_rolls_to_calendar, find_best_rolls_for_visit


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
        vda_pre_sequence_overhead: u.Quantity = 260 * u.s,
        vda_post_sequence_overhead: u.Quantity = 120 * u.s,
        nirda_pre_sequence_overhead: u.Quantity = 258 * u.s,
        nirda_post_sequence_overhead: u.Quantity = 120 * u.s,
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
        twilight_margin: Optional[float] = None,
        roll_step: float = 2.0,
        min_power_frac: float = 0.7,
    ) -> None:
        """
        Initialize the scheduler with TLE and parameters.

        Parameters:
        -----------
        tle_line1, tle_line2 : str
            TLE lines for satellite
        vda_pre_sequence_overhead : Quantity, optional
            VDA pre-sequence overhead (default 260 s).
        vda_post_sequence_overhead : Quantity, optional
            VDA post-sequence overhead (default 120 s).
        nirda_pre_sequence_overhead : Quantity, optional
            NIRDA pre-sequence overhead (default 258 s).
        nirda_post_sequence_overhead : Quantity, optional
            NIRDA post-sequence overhead (default 120 s).
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
        twilight_margin : float, optional
            Degrees past the geometric terminator to classify as
            "sunlit" for day/night Earth-limb keepout.  0 (default)
            gives a sharp terminator; 18 is analogous to astronomical
            twilight.  Only has an effect when ``earthlimb_day_min``
            or ``earthlimb_night_min`` is set.
        roll_step : float, optional
            Roll-angle sweep resolution in degrees (default 2.0).
        min_power_frac : float, optional
            Minimum acceptable solar-panel power fraction (0-1).
            Roll candidates below this are rejected (default 0.7).
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
        )
        # Only forward day/night earthlimb keepouts when explicitly set so that
        # Visibility falls back to earthlimb_min for whichever side is None.
        if earthlimb_day_min is not None:
            _kw["earthlimb_day_min"] = self._to_deg(earthlimb_day_min)
        if earthlimb_night_min is not None:
            _kw["earthlimb_night_min"] = self._to_deg(earthlimb_night_min)
        if twilight_margin is not None:
            _kw["twilight_margin"] = self._to_deg(twilight_margin)
        # Strip None entries so Visibility uses its own class-level defaults
        # for any constraint the caller left unset.
        _kw = {k: v for k, v in _kw.items() if v is not None}
        self.visibility = Visibility(tle_line1, tle_line2, **_kw)

        self.min_sequence_duration = TimeDelta(8 * 60 * u.s)
        self.max_sequence_duration = TimeDelta(90 * 60 * u.s)

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

        # Payload overhead budgets — validate that each value carries time
        # units so that downstream .to(u.s) / .to(u.us) calls succeed.
        _overhead_params = {
            "vda_pre_sequence_overhead": vda_pre_sequence_overhead,
            "vda_post_sequence_overhead": vda_post_sequence_overhead,
            "nirda_pre_sequence_overhead": nirda_pre_sequence_overhead,
            "nirda_post_sequence_overhead": nirda_post_sequence_overhead,
        }
        for _name, _val in _overhead_params.items():
            if isinstance(_val, TimeDelta):
                pass
            elif isinstance(_val, u.Quantity):
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
        self.vda_pre_sequence_overhead = vda_pre_sequence_overhead
        self.vda_post_sequence_overhead = vda_post_sequence_overhead
        self.nirda_pre_sequence_overhead = nirda_pre_sequence_overhead
        self.nirda_post_sequence_overhead = nirda_post_sequence_overhead

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
        verbose : bool, optional
            Print diagnostics when True.

        Returns
        -------
        ScienceCalendar
            Processed calendar with updated sequences and metadata.
        """

        # Clear previous gap report
        self._initialize_gap_report()

        if verbose:
            print("Processing calendar with TLE:")
            print(f"  Line 1: {self.tle_line1}")
            print(f"  Line 2: {self.tle_line2}")

        # Extract windowed calendar FIRST
        windowed_calendar = self._extract_time_window(
            calendar, window_start, window_duration_days, verbose
        )

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
            print(
                f"\n--- Validation: {calendar_status} "
                f"({sum(validation_counts.values())} issues) ---"
            )
            for cat, cnt in validation_counts.items():
                print(f"  {cat}: {cnt}")
            print(
                "Run print_validation_summary(calendar) "
                "for actionable details.\n"
            )
        else:
            print(f"\n--- Validation: {calendar_status} " f"(0 issues) ---\n")

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

        if verbose:
            print(f"Extracting window: {window_start} to {window_end}")

        # Find sequences within window
        windowed_visits = []
        for visit in calendar.visits:
            # complain if there are empty visits
            if not visit.sequences:
                print(f"Warning: Empty sequence list for visit {visit.id}")
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

        # ── Pre-compute best roll per target per visit ──────────
        # Only run the sweep when star-tracker constraints are active;
        # boresight-only constraints are roll-independent.
        if self._roll_sweep_enabled:
            for visit in working_calendar.visits:
                visit_rolls = find_best_rolls_for_visit(
                    self.visibility,
                    visit,
                    roll_step=self.roll_step,
                    min_power_frac=self.min_power_frac,
                )
                self._computed_target_rolls[visit.id] = visit_rolls
                if verbose:
                    for tgt, r in visit_rolls.items():
                        print(f"  Visit {visit.id} / {tgt}: best roll = {r}")

        # Use initial time grid for processing
        total_minutes, start_time, end_time, time_grid = (
            self._get_synchronized_time_grid(working_calendar)
        )
        all_minutes_bool = np.zeros(total_minutes, dtype=bool)

        i = 0
        last_stop = deepcopy(start_time)

        for visit in working_calendar.visits:
            visit_rolls = self._computed_target_rolls.get(visit.id, {})

            for j, seq in enumerate(visit.sequences):

                # Compute gap since last sequence stop
                gap_length = int(
                    np.rint((seq.start_time - last_stop).sec / 60.0)
                )
                if gap_length > 0:
                    if verbose:
                        print(
                            f"Filling {gap_length} min gap before sequence {seq.id}"
                        )

                    seq = self._fill_gaps(seq, gap_length)
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

        # Fill remaining time after last sequence
        if i < total_minutes:
            all_minutes_bool[i:] = False
            if verbose:
                print(f"Filled trailing {total_minutes - i} minutes as False")

        self.all_minutes_bool = (
            all_minutes_bool  # this is only necessary for testing.
        )

        working_calendar = self._fix_visibility(
            working_calendar, all_minutes_bool
        )

        # Trim non-visible heads that _fix_visibility left behind
        # (e.g. when the previous target is also not visible during
        # the gap, or shrinking the current seq would violate the
        # minimum duration).
        working_calendar = self._trim_non_visible_heads(working_calendar)

        # Trim non-visible tails that _fix_visibility cannot handle
        # (it only shrinks starts forward; tails need stop_time shrunk).
        working_calendar = self._trim_non_visible_tails(working_calendar)

        # Remove sequences that are entirely non-visible (e.g. a target
        # that violates sun keepout for the entire segment).  Neighbours
        # absorb the freed time to maintain schedule contiguity.
        working_calendar = self._drop_fully_non_visible_sequences(
            working_calendar, verbose
        )

        # Absorb sequences shorter than the minimum duration into
        # their immediate neighbours so that all remaining sequences
        # are long enough for the payload overhead budget.
        working_calendar = self._merge_short_sequences(working_calendar)

        # last thing is to update all the payload parameters
        working_calendar = self._update_payload_parameters(working_calendar)

        return working_calendar

    def _fill_gaps(
        self,
        sequence: ObservationSequence,
        gap_length: int,
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

        For each sequence whose first minute(s) are not visible, move
        ``start_time`` forward to the first visible minute.  Then
        attempt to extend the *previous* sequence forward to absorb the
        freed time (only where that target is visible).

        This complements ``_fix_visibility`` which may leave non-visible
        heads when the previous target is also not visible during the
        gap or shrinking the current sequence would violate the minimum
        duration.
        """
        working_cal = deepcopy(calendar)

        all_sequences: List[Tuple[str, ObservationSequence]] = []
        for visit in working_cal.visits:
            for seq in visit.sequences:
                all_sequences.append((visit.id, seq))
        all_sequences.sort(key=lambda x: x[1].start_time)

        for idx, (visit_id, seq) in enumerate(all_sequences):
            n_mins = int(np.rint(seq.duration.sec / 60.0))
            if n_mins <= 0:
                continue

            target_coord = SkyCoord(
                seq.ra, seq.dec, frame="icrs", unit="deg"
            )
            deltas = np.arange(n_mins) * u.min
            times = seq.start_time + deltas

            target_roll = self._computed_target_rolls.get(
                visit_id, {}
            ).get(seq.target)
            if self._roll_sweep_enabled and target_roll is not None:
                vis = self.visibility.get_visibility(
                    target_coord, times, roll=target_roll * u.deg
                )
            else:
                vis = self.visibility.get_visibility(
                    target_coord, times
                )

            vis_arr = np.asarray(vis)

            # Nothing to do if first minute is visible
            if len(vis_arr) == 0 or vis_arr[0]:
                continue

            visible_indices = np.where(vis_arr)[0]
            if len(visible_indices) == 0:
                continue  # entirely non-visible — handled elsewhere

            first_visible_idx = visible_indices[0]
            new_start = seq.start_time + first_visible_idx * u.min

            if (seq.stop_time - new_start) < self.min_sequence_duration:
                continue  # trimming would make sequence too short

            # Determine how far the previous sequence can extend forward
            # BEFORE committing the trim, to avoid creating gaps.
            actual_trim_start = new_start

            if idx > 0:
                prev_visit_id, prev_seq = all_sequences[idx - 1]
                gap_minutes = int(
                    np.rint(
                        (new_start - prev_seq.stop_time).sec / 60.0
                    )
                )
                if gap_minutes > 0:
                    prev_coord = SkyCoord(
                        prev_seq.ra,
                        prev_seq.dec,
                        frame="icrs",
                        unit="deg",
                    )
                    gap_deltas = np.arange(gap_minutes) * u.min
                    gap_times = prev_seq.stop_time + gap_deltas

                    prev_roll = self._computed_target_rolls.get(
                        prev_visit_id, {}
                    ).get(prev_seq.target)
                    if (
                        self._roll_sweep_enabled
                        and prev_roll is not None
                    ):
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

                    # Walk forward from prev_seq.stop_time to find
                    # the last contiguous visible minute.
                    if prev_vis_arr[0]:
                        last_contiguous = 0
                        while (
                            last_contiguous + 1 < len(prev_vis_arr)
                            and prev_vis_arr[last_contiguous + 1]
                        ):
                            last_contiguous += 1

                        new_prev_stop = gap_times[last_contiguous] + 1 * u.min

                        # Only trim current seq to where the prev
                        # can actually extend.
                        actual_trim_start = new_prev_stop

                        new_dur = new_prev_stop - prev_seq.start_time
                        if new_dur <= self.max_sequence_duration:
                            extended_prev = ObservationSequence(
                                id=prev_seq.id,
                                target=prev_seq.target,
                                priority=prev_seq.priority,
                                start_time=prev_seq.start_time,
                                stop_time=new_prev_stop,
                                ra=prev_seq.ra,
                                dec=prev_seq.dec,
                                payload_params=deepcopy(
                                    prev_seq.payload_params
                                ),
                            )
                            working_cal.replace_sequence(
                                prev_visit_id,
                                prev_seq.id,
                                extended_prev,
                            )
                            all_sequences[idx - 1] = (
                                prev_visit_id,
                                extended_prev,
                            )
                        else:
                            # Previous at max — cannot absorb gap.
                            # Don't trim to avoid creating a gap.
                            continue
                    else:
                        # Previous target not visible at boundary —
                        # extend it forward blindly to maintain
                        # contiguity (mirrors _fill_gaps behaviour).
                        actual_trim_start = new_start
                        new_prev_stop = new_start
                        new_dur = new_prev_stop - prev_seq.start_time
                        if new_dur <= self.max_sequence_duration:
                            extended_prev = ObservationSequence(
                                id=prev_seq.id,
                                target=prev_seq.target,
                                priority=prev_seq.priority,
                                start_time=prev_seq.start_time,
                                stop_time=new_prev_stop,
                                ra=prev_seq.ra,
                                dec=prev_seq.dec,
                                payload_params=deepcopy(
                                    prev_seq.payload_params
                                ),
                            )
                            working_cal.replace_sequence(
                                prev_visit_id,
                                prev_seq.id,
                                extended_prev,
                            )
                            all_sequences[idx - 1] = (
                                prev_visit_id,
                                extended_prev,
                            )
                        else:
                            # Previous at max — cannot absorb.
                            continue
                # gap_minutes <= 0: prev seq already abuts, safe to trim

            if (
                seq.stop_time - actual_trim_start
            ) < self.min_sequence_duration:
                continue

            trimmed = ObservationSequence(
                id=seq.id,
                target=seq.target,
                priority=seq.priority,
                start_time=actual_trim_start,
                stop_time=seq.stop_time,
                ra=seq.ra,
                dec=seq.dec,
                payload_params=deepcopy(seq.payload_params),
            )
            working_cal.replace_sequence(visit_id, seq.id, trimmed)
            all_sequences[idx] = (visit_id, trimmed)

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

        for idx, (visit_id, seq) in enumerate(all_sequences):
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

            last_visible_idx = visible_indices[-1]
            new_stop = seq.start_time + (last_visible_idx + 1) * u.min

            if (new_stop - seq.start_time) < self.min_sequence_duration:
                continue  # trimming would make sequence too short

            # Determine how far the next sequence can extend backward
            # BEFORE committing the trim, to avoid creating gaps.
            actual_trim_stop = new_stop  # default: trim to last visible + 1

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

                    # Walk backward from the original next start to
                    # find the earliest contiguous visible minute.
                    last_gap_idx = len(next_vis_arr) - 1
                    if next_vis_arr[last_gap_idx]:
                        first_contiguous = last_gap_idx
                        while (
                            first_contiguous > 0
                            and next_vis_arr[first_contiguous - 1]
                        ):
                            first_contiguous -= 1

                        new_next_start = gap_times[first_contiguous]

                        # Only trim the current sequence to where the
                        # next sequence can actually extend — this
                        # keeps contiguity even when the next target
                        # cannot absorb the entire freed region.
                        actual_trim_stop = new_next_start

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
                        # Update local list so subsequent iterations
                        # see the modified next sequence.
                        all_sequences[idx + 1] = (
                            next_visit_id,
                            extended_next,
                        )
                    else:
                        # Next target not visible at boundary — extend
                        # it backward blindly to maintain contiguity.
                        # This mirrors _fill_gaps: the non-visible time
                        # now belongs to the next target and may be
                        # cleaned up by subsequent processing passes.
                        actual_trim_stop = new_stop
                        extended_next = ObservationSequence(
                            id=next_seq.id,
                            target=next_seq.target,
                            priority=next_seq.priority,
                            start_time=new_stop,
                            stop_time=next_seq.stop_time,
                            ra=next_seq.ra,
                            dec=next_seq.dec,
                            payload_params=deepcopy(next_seq.payload_params),
                        )
                        working_cal.replace_sequence(
                            next_visit_id, next_seq.id, extended_next
                        )
                        all_sequences[idx + 1] = (
                            next_visit_id,
                            extended_next,
                        )
                # gap_minutes <= 0: next seq already abuts, safe to trim
            # else: no next sequence — tail trim is safe (no
            # inter-sequence gap to worry about).

            # Only apply the trim if the sequence remains long enough.
            if (
                actual_trim_stop - seq.start_time
            ) < self.min_sequence_duration:
                continue

            trimmed = ObservationSequence(
                id=seq.id,
                target=seq.target,
                priority=seq.priority,
                start_time=seq.start_time,
                stop_time=actual_trim_stop,
                ra=seq.ra,
                dec=seq.dec,
                payload_params=deepcopy(seq.payload_params),
            )
            working_cal.replace_sequence(visit_id, seq.id, trimmed)

        return working_cal

    def _drop_fully_non_visible_sequences(
        self,
        calendar: ScienceCalendar,
        verbose: bool = False,
    ) -> ScienceCalendar:
        """Remove sequences where **zero** minutes are visible.

        A fully non-visible sequence typically results from an upstream
        scheduling error (e.g. an occultation target that violates sun
        keepout for the entire segment).  Keeping it would guarantee a
        keepout-constraint violation in the output calendar.

        For each removed sequence the method extends a neighbour
        (previous preferred, else next) to maintain schedule contiguity.
        If neither neighbour can absorb the time without exceeding
        ``max_sequence_duration``, the sequence is left in place and a
        warning is printed so the operator can investigate.
        """
        working_cal = deepcopy(calendar)

        all_seqs: List[Tuple[str, ObservationSequence]] = []
        for visit in working_cal.visits:
            for seq in visit.sequences:
                all_seqs.append((visit.id, seq))
        all_seqs.sort(key=lambda x: x[1].start_time)

        to_remove: List[Tuple[str, str]] = []

        for idx, (vid, seq) in enumerate(all_seqs):
            n_mins = int(np.rint(seq.duration.sec / 60.0))
            if n_mins <= 0:
                continue

            target_coord = SkyCoord(seq.ra, seq.dec, frame="icrs", unit="deg")
            deltas = np.arange(n_mins) * u.min
            times = seq.start_time + deltas

            target_roll = self._computed_target_rolls.get(vid, {}).get(
                seq.target
            )
            if self._roll_sweep_enabled and target_roll is not None:
                vis = self.visibility.get_visibility(
                    target_coord, times, roll=target_roll * u.deg
                )
            else:
                vis = self.visibility.get_visibility(target_coord, times)

            if np.any(vis):
                continue  # at least partially visible — keep it

            # Entirely non-visible.
            if verbose:
                print(
                    f"Dropping fully non-visible sequence "
                    f"{seq.id} ({seq.target}) in visit {vid}"
                )
            print(
                f"WARNING: removing sequence {seq.id} "
                f"({seq.target}, {seq.start_time.isot}–"
                f"{seq.stop_time.isot}): 0% visible"
            )

            absorbed = False

            # Try extending the previous sequence forward
            if idx > 0:
                prev_vid, prev_seq = all_seqs[idx - 1]
                new_stop = seq.stop_time
                new_dur = new_stop - prev_seq.start_time
                if new_dur <= self.max_sequence_duration:
                    extended = ObservationSequence(
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
                        prev_vid, prev_seq.id, extended
                    )
                    all_seqs[idx - 1] = (prev_vid, extended)
                    absorbed = True

            # Fall back: pull the next sequence backward
            if not absorbed and idx + 1 < len(all_seqs):
                next_vid, next_seq = all_seqs[idx + 1]
                new_start = seq.start_time
                new_dur = next_seq.stop_time - new_start
                if new_dur <= self.max_sequence_duration:
                    extended = ObservationSequence(
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
                        next_vid, next_seq.id, extended
                    )
                    all_seqs[idx + 1] = (next_vid, extended)
                    absorbed = True

            if absorbed:
                to_remove.append((vid, seq.id))
            else:
                print(
                    f"WARNING: could not absorb non-visible sequence "
                    f"{seq.id} ({seq.target}) — neighbours at "
                    f"max duration"
                )

        for vid, sid in to_remove:
            for visit in working_cal.visits:
                if visit.id == vid:
                    visit.sequences = [
                        s for s in visit.sequences if s.id != sid
                    ]
                    break

        working_cal.visits = [v for v in working_cal.visits if v.sequences]

        return working_cal

    def _merge_short_sequences(
        self, calendar: ScienceCalendar
    ) -> ScienceCalendar:
        """Absorb sequences shorter than *min_sequence_duration* into
        their immediate neighbours.

        For each short sequence the method first tries to extend the
        **previous** sequence's ``stop_time`` forward.  If there is no
        previous sequence (e.g. the short sequence is the first in the
        calendar), the **next** sequence's ``start_time`` is pulled
        backward instead.  The short sequence is then removed from its
        visit.

        This guarantees that:
        * no sequence remains shorter than the payload overhead budget,
        * schedule contiguity is preserved (no new gaps are introduced),
        * no sequence is extended beyond **max_sequence_duration**.
        """
        working_cal = deepcopy(calendar)

        # Collect all (visit_id, seq) pairs sorted by start_time
        all_seqs: List[Tuple[str, ObservationSequence]] = []
        for visit in working_cal.visits:
            for seq in visit.sequences:
                all_seqs.append((visit.id, seq))
        all_seqs.sort(key=lambda x: x[1].start_time)

        to_remove: List[Tuple[str, str]] = []  # (visit_id, seq_id)

        for idx, (vid, seq) in enumerate(all_seqs):
            if seq.duration >= self.min_sequence_duration:
                continue

            absorbed = False

            # Try extending the previous sequence forward
            if idx > 0:
                prev_vid, prev_seq = all_seqs[idx - 1]
                new_stop = seq.stop_time
                new_dur = new_stop - prev_seq.start_time
                if new_dur <= self.max_sequence_duration:
                    extended = ObservationSequence(
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
                        prev_vid, prev_seq.id, extended
                    )
                    all_seqs[idx - 1] = (prev_vid, extended)
                    absorbed = True

            # Fall back: pull the next sequence backward
            if not absorbed and idx + 1 < len(all_seqs):
                next_vid, next_seq = all_seqs[idx + 1]
                new_start = seq.start_time
                new_dur = next_seq.stop_time - new_start
                if new_dur <= self.max_sequence_duration:
                    extended = ObservationSequence(
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
                        next_vid, next_seq.id, extended
                    )
                    all_seqs[idx + 1] = (next_vid, extended)
                    absorbed = True

            if absorbed:
                to_remove.append((vid, seq.id))

        # Remove absorbed sequences from their visits
        for vid, sid in to_remove:
            for visit in working_cal.visits:
                if visit.id == vid:
                    visit.sequences = [
                        s for s in visit.sequences if s.id != sid
                    ]
                    break

        # Drop visits that have become empty
        working_cal.visits = [v for v in working_cal.visits if v.sequences]

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
        for gap_start_idx, gap_end_idx in false_idx:
            # Get times for this gap
            gap_times = []
            for x in range(0, gap_end_idx - gap_start_idx):
                if gap_start_idx + x < len(assignments):
                    gap_times.append(assignments[gap_start_idx + x]["time"])

            if not gap_times:
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
            extended = False
            new_stop_time = None
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
                        extended = True
                        gaps_filled += 1

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

            # Shrink the current sequence only when the previous
            # sequence was actually extended — this keeps extend and
            # shrink atomic and avoids creating time gaps.  The new
            # start is set to where the extension ends (not the end
            # of the entire false block) so contiguity is preserved
            # even when the previous target is only partially visible.
            if (
                extended
                and shrink_feasible
                and seq_to_shrink is not None
                and new_stop_time is not None
            ):
                shrink_start = new_stop_time
                remaining = seq_to_shrink.stop_time - shrink_start
                if remaining >= self.min_sequence_duration:
                    shrunk_seq = ObservationSequence(
                        id=seq_to_shrink.id,
                        target=seq_to_shrink.target,
                        priority=seq_to_shrink.priority,
                        start_time=shrink_start,
                        stop_time=seq_to_shrink.stop_time,
                        ra=seq_to_shrink.ra,
                        dec=seq_to_shrink.dec,
                        payload_params=deepcopy(seq_to_shrink.payload_params),
                    )
                    working_cal.replace_sequence(
                        current_visit_id,
                        current_sequence_id,
                        shrunk_seq,
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
        time_tolerance = 1.0 * u.s

        # Generate time grid
        times = []
        assignments = []

        for minute_idx in range(total_minutes):
            current_time = start_time + minute_idx * u.min
            times.append(current_time)

            assignment = {
                "time": current_time.isot,
                "minute_index": minute_idx,
                "sequence_id": None,
                "target": None,
                "visit_id": None,
                "ra": None,
                "dec": None,
                "priority": None,
                "status": "unassigned",
            }

            # Find the sequence that owns this minute
            assigned_sequence = None
            visit_id = None

            for visit in calendar.visits:
                for seq in visit.sequences:
                    start_diff = current_time - seq.start_time
                    stop_diff = current_time - seq.stop_time

                    starts_at_or_after = start_diff >= -time_tolerance
                    ends_before = stop_diff < -time_tolerance
                    starts_exactly = abs(start_diff) <= time_tolerance

                    if (starts_at_or_after and ends_before) or starts_exactly:
                        assigned_sequence = seq
                        visit_id = visit.id
                        break

                if assigned_sequence:
                    break

            if assigned_sequence:
                assignment.update(
                    {
                        "sequence_id": assigned_sequence.id,
                        "target": assigned_sequence.target,
                        "visit_id": visit_id,
                        "ra": assigned_sequence.ra,
                        "dec": assigned_sequence.dec,
                        "priority": assigned_sequence.priority,
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
                new_sequence = self._update_payload_parameters_sequence(seq)
                calendar.replace_sequence(visit_id, sequence_id, new_sequence)

        return calendar

    def _update_payload_parameters_sequence(
        self, sequence: ObservationSequence
    ) -> ObservationSequence:
        # Pass sequence.duration (TimeDelta) so both helpers receive the
        # correct type and the overhead subtraction uses a consistent unit.
        duration = sequence.duration

        sequence = self._update_VDA_integrations(
            sequence,
            duration,
            pre_sequence_overhead=self.vda_pre_sequence_overhead,
            post_sequence_overhead=self.vda_post_sequence_overhead,
        )
        sequence = self._update_NIRDA_integrations(
            sequence,
            duration,
            pre_sequence_overhead=self.nirda_pre_sequence_overhead,
            post_sequence_overhead=self.nirda_post_sequence_overhead,
        )

        return sequence

    def _update_VDA_integrations(
        self,
        sequence: ObservationSequence,
        duration: TimeDelta,
        pre_sequence_overhead: TimeDelta = 260 * u.s,
        post_sequence_overhead: TimeDelta = 120 * u.s,
    ) -> ObservationSequence:

        # Include VDA overheads at the start and end of the sequence using
        # pre_sequence_overhead and post_sequence_overhead.

        # Get parameters
        exposure_time_str = sequence.get_payload_parameter(
            "AcquireVisCamScienceData", "ExposureTime_us"
        )
        frames_per_coadd_str = sequence.get_payload_parameter(
            "AcquireVisCamScienceData", "FramesPerCoadd"
        )

        if not exposure_time_str or not frames_per_coadd_str:
            print(
                f"Warning: Missing VDA parameters for sequence {sequence.id} {sequence.start_time}"
            )
            return sequence

        try:
            # Convert to proper types - exposure time is always integer microseconds
            exposure_time = int(exposure_time_str) * u.us
            frames_per_coadd = int(frames_per_coadd_str)

            # Convert duration to microseconds for calculation
            duration_us = (
                duration.to(u.us)
                - pre_sequence_overhead.to(u.us)
                - post_sequence_overhead.to(u.us)
            )

            # Guard: if overhead exceeds duration, no frames are possible
            if duration_us.value <= 0:
                sequence.set_payload_parameter(
                    "AcquireVisCamScienceData",
                    "NumTotalFramesRequested",
                    "0",
                )
                return sequence

            # Calculate maximum complete coadds that fit in duration
            effective_exposure_time = exposure_time * frames_per_coadd
            max_coadds = int(np.floor(duration_us / effective_exposure_time))

            # Calculate total frames (must be multiple of FramesPerCoadd)
            num_total_frames = max_coadds * frames_per_coadd

            # Ensure at least one coadd if duration allows
            if (
                num_total_frames == 0
                and duration_us >= effective_exposure_time
            ):
                num_total_frames = frames_per_coadd

            # Set the parameter (convert to string)
            success = sequence.set_payload_parameter(
                "AcquireVisCamScienceData",
                "NumTotalFramesRequested",
                str(num_total_frames),
            )

            if not success:
                print(
                    f"Warning: Failed to update NumTotalFramesRequested for sequence {sequence.id} {sequence.start_time}"
                )
            return sequence

        except (ValueError, TypeError, AttributeError) as e:
            print(
                f"Error updating VDA parameters for sequence {sequence.id} {sequence.start_time}: {e}"
            )
            return sequence

    def _update_NIRDA_integrations(
        self,
        sequence: ObservationSequence,
        duration: TimeDelta,
        pre_sequence_overhead: TimeDelta = 258 * u.s,
        post_sequence_overhead: TimeDelta = 120 * u.s,
    ) -> ObservationSequence:

        seq_identifier = (
            f"{sequence.id} ({sequence.target} @ "
            f"{sequence.start_time.datetime.strftime('%m/%d %H:%M')})"
        )

        # Collect raw string values and guard against missing parameters
        # *before* any int() conversion so that a sequence without NIRDA
        # payload (e.g. a VDA-only sequence) returns cleanly instead of
        # raising TypeError.
        required_params = {
            name: sequence.get_payload_parameter("AcquireInfCamImages", name)
            for name in (
                "ROI_SizeX",
                "ROI_SizeY",
                "SC_Resets1",
                "SC_Resets2",
                "SC_DropFrames1",
                "SC_DropFrames2",
                "SC_DropFrames3",
                "SC_ReadFrames",
                "SC_Groups",
            )
        }

        missing_params = [
            name for name, value in required_params.items() if value is None
        ]

        if missing_params:
            print(
                f"Warning: Missing NIRDA parameters for sequence "
                f"{seq_identifier}"
            )
            print(f"Missing parameters: {', '.join(missing_params)}")
            return sequence

        # All parameters are present; convert to int.
        ROI_SizeX = int(required_params["ROI_SizeX"])
        ROI_SizeY = int(required_params["ROI_SizeY"])
        SC_Resets1 = int(required_params["SC_Resets1"])
        SC_Resets2 = int(required_params["SC_Resets2"])
        SC_DropFrames1 = int(required_params["SC_DropFrames1"])
        SC_DropFrames2 = int(required_params["SC_DropFrames2"])
        SC_DropFrames3 = int(required_params["SC_DropFrames3"])
        SC_ReadFrames = int(required_params["SC_ReadFrames"])
        SC_Groups = int(required_params["SC_Groups"])

        # per the payload users guide
        frame_time = (ROI_SizeX + 12) * (ROI_SizeY + 2) * 1e-5 * u.s
        NumIntegrations = 1
        NumFramesTotal = (
            SC_Resets1
            + (NumIntegrations - 1) * SC_Resets2
            + NumIntegrations
            * (
                SC_DropFrames1
                + (SC_Groups - 1) * (SC_ReadFrames + SC_DropFrames2)
                + SC_ReadFrames
                + SC_DropFrames3
            )
        )

        integration_time = NumFramesTotal * frame_time

        # Use the duration argument so callers can override the window
        # (e.g. _update_payload_parameters_sequence passes sequence.duration).
        duration_sequence_seconds = duration.to(u.s)

        effective_duration_s = (
            duration_sequence_seconds
            - pre_sequence_overhead.to(u.s)
            - post_sequence_overhead.to(u.s)
        )
        # Guard: if overhead exceeds duration, no integrations are possible
        if effective_duration_s.value <= 0:
            sequence.set_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations", "0"
            )
            return sequence

        SC_Integrations = int(
            np.floor(effective_duration_s / integration_time.to(u.s))
        )
        success = sequence.set_payload_parameter(
            "AcquireInfCamImages", "SC_Integrations", str(SC_Integrations)
        )
        if not success:
            print(
                f"Warning: Failed to update SC_Integrations for sequence {sequence.id} {sequence.start_time}"
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
                        print(message)

        return issues

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
                        print(
                            f"Target name issue: '{seq.target}' contains spaces (sequence {seq.id}, visit {visit.id})"
                        )

        return issues

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

                if verbose:
                    print(
                        f"Original gap: {gap_duration:.1f} min between {current_seq.id} and {next_seq.id}"
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

        print("\n" + "=" * 60)
        print("VISIBILITY GAP ANALYSIS SUMMARY")
        print("=" * 60)

        print("\nORIGINAL CALENDAR:")
        print(
            f"  Total Sequences: {report['original_calendar_stats']['total_sequences']}"
        )
        print(
            f"  Total Duration: {report['original_calendar_stats']['total_duration_hours']:.1f} hours"
        )
        print(
            f"  Duty Cycle: {report['original_calendar_stats']['duty_cycle_percent']:.1f}%"
        )

        print("\nPROCESSED CALENDAR:")
        print(
            f"  Total Sequences: {report['processed_calendar_stats']['total_sequences']}"
        )
        print(
            f"  Total Duration: {report['processed_calendar_stats']['total_duration_hours']:.1f} hours"
        )
        print(
            f"  Duty Cycle: {report['processed_calendar_stats']['duty_cycle_percent']:.1f}%"
        )

        print("\nIMPROVEMENTS:")
        print(
            f"  Duration Gained: {summary.get('duration_improvement_hours', 0):.1f} hours"
        )
        print(
            f"  Duty Cycle Improved: {summary.get('duty_cycle_improvement_percent', 0):.1f}%"
        )
        print(f"  Sequences Modified: {summary.get('sequences_modified', 0)}")

        if "gaps_filled" in summary:
            print(
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
            print(f"Sequence {sequence_id} not found")
            return

        print(f"\n{'='*60}")
        print(f"DEBUGGING SEQUENCE {sequence_id}: {target_seq.target}")
        print(f"{'='*60}")
        print(f"Visit ID: {target_visit_id}")
        print(f"Start Time: {target_seq.start_time}")
        print(f"Stop Time: {target_seq.stop_time}")
        print(f"Duration: {target_seq.duration.sec/60:.1f} minutes")
        print(f"Target: {target_seq.target}")
        print(f"RA/Dec: {target_seq.ra:.3f}, {target_seq.dec:.3f}")

        # Check visibility minute by minute
        n_mins = int(np.rint(target_seq.duration.sec / 60.0))
        target_coord = SkyCoord(
            target_seq.ra, target_seq.dec, frame="icrs", unit="deg"
        )
        deltas = np.arange(n_mins) * u.min
        times = target_seq.start_time + deltas

        vis = self.visibility.get_visibility(target_coord, times)

        print("\nMinute-by-minute visibility:")
        for i, (time, visible) in enumerate(zip(times, vis)):
            status = "✓ VISIBLE" if visible else "✗ NOT VISIBLE"
            print(f"  Minute {i+1}: {time.isot} - {status}")

        print("\nVisibility Summary:")
        print(f"  Total minutes: {len(vis)}")
        print(f"  Visible minutes: {np.sum(vis)}")
        print(f"  Visibility fraction: {np.sum(vis)/len(vis):.3f}")

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
                    print(message)

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
            print("\n" + "=" * 60)
            print("SEQUENCE TIMING VALIDATION REPORT")
            print("=" * 60)

            summary = issues["timing_summary"]
            print(
                f"Total sequences analyzed: " f"{summary['total_sequences']}"
            )
            print(f"Total timing issues found: " f"{summary['total_issues']}")
            print()

            if issues["overlaps"]:
                print(f"OVERLAPS ({len(issues['overlaps'])} found):")
                for i, ov in enumerate(issues["overlaps"]):
                    print(f"  {i+1}. {ov['message']}")
            else:
                print("\u2713 OVERLAPS: None found")

            print()

            if issues["short_sequences"]:
                print(
                    f"SHORT SEQUENCES "
                    f"({len(issues['short_sequences'])} found, "
                    f"< {min_dur_min:.0f} min):"
                )
                for i, sh in enumerate(issues["short_sequences"]):
                    print(f"  {i+1}. {sh['message']}")
            else:
                print("\u2713 SHORT SEQUENCES: None found")

            print()

            if issues["large_gaps"]:
                print(
                    f"LARGE GAPS ({len(issues['large_gaps'])} "
                    f"found, > 2 min):"
                )
                for i, gap in enumerate(issues["large_gaps"][:5]):
                    print(f"  {i+1}. {gap['message']}")
                if len(issues["large_gaps"]) > 5:
                    print(
                        f"     ... and "
                        f"{len(issues['large_gaps']) - 5} more"
                    )
            else:
                print("\u2713 LARGE GAPS: None found")

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

        # Compute effective overhead budget (max of VDA/NIRDA)
        pre_oh_sec = max(
            getattr(self, "vda_pre_sequence_overhead", 0 * u.s).to(u.s).value,
            getattr(self, "nirda_pre_sequence_overhead", 0 * u.s)
            .to(u.s)
            .value,
        )
        post_oh_sec = max(
            getattr(self, "vda_post_sequence_overhead", 0 * u.s).to(u.s).value,
            getattr(self, "nirda_post_sequence_overhead", 0 * u.s)
            .to(u.s)
            .value,
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

                        # Sequence too short for overhead
                        if effective_sec <= 0:
                            msg = (
                                f"Seq {seq.id} ({seq.target}, "
                                f"visit {visit.id}): sequence "
                                f"too short for payload "
                                f"overhead "
                                f"({seq_dur_sec:.0f}s < "
                                f"{total_oh_sec:.0f}s "
                                f"overhead). No science "
                                f"exposures possible."
                            )
                            issues.append(
                                {
                                    "visit_id": visit.id,
                                    "sequence_id": seq.id,
                                    "target": seq.target,
                                    "problem": (
                                        "sequence_shorter_than" "_overhead"
                                    ),
                                    "sequence_duration_seconds": (seq_dur_sec),
                                    "effective_duration_seconds": (
                                        effective_sec
                                    ),
                                    "overhead_seconds": total_oh_sec,
                                    "suggested_fix": (
                                        "Extend sequence to at "
                                        "least "
                                        f"{int(total_oh_sec)}s "
                                        "or merge with "
                                        "adjacent sequence"
                                    ),
                                    "message": msg,
                                }
                            )
                            if report_issues:
                                print(msg)
                            # Skip frame-level checks — they
                            # are meaningless for this sequence
                            continue

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
                                print(msg)

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
                                        print(msg)
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
                                        print(msg)
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
                                print(msg)

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
                                    print(
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
                                print(
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
                                    print(
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
                                print(
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
                                    print(
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
                                print(
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
                        print(msg)

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
                    print(
                        f"{indent}{body:<12} {status}  "
                        f"required: >= {info['required_deg']:.1f}°"
                        f"{side_label}  "
                        f"actual: {info['actual_deg']:.1f}°"
                    )
            frac = item.get("visibility_fraction")
            nv = item.get("non_visible_minutes")
            tot = item.get("total_minutes")
            if frac is not None:
                print(
                    f"{indent}{'visibility':<12}       "
                    f"required: 100%  "
                    f"actual: {frac:.1%}  "
                    f"({nv}/{tot} min non-visible)"
                )

        elif category == "short_sequences":
            dur = item.get("duration_minutes")
            req = item.get("minimum_required_minutes")
            if dur is not None and req is not None:
                print(
                    f"{indent}duration     "
                    f"required: >= {req:.0f} min  "
                    f"actual: {dur:.1f} min  "
                    f"(short by {req - dur:.1f} min)"
                )

        elif category == "large_gaps":
            gap = item.get("gap_duration_minutes")
            if gap is not None:
                print(
                    f"{indent}gap          "
                    f"required: <= 2.0 min  "
                    f"actual: {gap:.1f} min  "
                    f"(over by {gap - 2.0:.1f} min)"
                )

        elif category == "overlaps":
            ov = item.get("overlap_duration_minutes")
            if ov is not None:
                print(
                    f"{indent}overlap      "
                    f"required: 0.0 min  "
                    f"actual: {ov:.1f} min"
                )

        elif category == "payload_exposure":
            seq_dur = item.get("sequence_duration_seconds")
            eff_dur = item.get("effective_duration_seconds")
            oh = item.get("overhead_seconds")
            if seq_dur is not None:
                print(
                    f"{indent}sequence     "
                    f"{seq_dur:.0f}s total  "
                    f"- {oh:.0f}s overhead  "
                    f"= {eff_dur:.0f}s effective"
                )
            if "exposure_seconds" in item:
                exp = item["exposure_seconds"]
                print(
                    f"{indent}single exp   "
                    f"required: <= {eff_dur:.0f}s  "
                    f"actual: {exp:.3f}s"
                )
            if "total_exposure_seconds" in item:
                tot = item["total_exposure_seconds"]
                max_f = item.get("suggested_max_frames", "?")
                print(
                    f"{indent}total exp    "
                    f"required: <= {eff_dur:.0f}s  "
                    f"actual: {tot:.1f}s  "
                    f"(max frames: {max_f})"
                )
            if "coadd_exposure_seconds" in item:
                coadd = item["coadd_exposure_seconds"]
                print(
                    f"{indent}coadd exp    "
                    f"required: <= {eff_dur:.0f}s  "
                    f"actual: {coadd:.1f}s"
                )
            if "value_seconds" in item:
                val = item["value_seconds"]
                field = item.get("field", "?")
                print(
                    f"{indent}{field}  "
                    f"required: <= {eff_dur:.0f}s  "
                    f"actual: {val:.3f}s"
                )

        elif category == "roll_consistency":
            spread = item.get("max_difference_deg")
            suggested = item.get("suggested_roll")
            if spread is not None:
                print(
                    f"{indent}roll spread  "
                    f"required: <= 0.001°  "
                    f"actual: {spread:.3f}°  "
                    f"(suggest: {suggested:.2f}°)"
                )

        elif category == "target_name":
            tgt = item.get("target", "")
            if tgt:
                print(
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
        print(
            f"\n{'=' * 60}\n"
            f"  VALIDATION SUMMARY: {status} "
            f"({total} issues)\n"
            f"{'=' * 60}"
        )

        if total == 0:
            print("  All checks passed.\n")
            return {
                "status": status,
                "counts": counts,
                "details": results,
            }

        for cat, cnt in counts.items():
            print(f"\n  [{cat.upper()}] — {cnt} issue(s)")
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
                            print(f"    • {msg}")
                        self._print_issue_details(sub_key, item)
                continue

            # All other categories are plain lists
            if isinstance(items, list):
                for item in items:
                    msg = item.get("message", "")
                    if msg:
                        print(f"    • {msg}")
                    self._print_issue_details(cat, item)

        print(f"\n{'=' * 60}\n")
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
            print("✓ All sequence timing validation checks passed")
        else:
            print(f"✗ Found {summary['total_issues']} timing issues:")
            if summary["overlaps_found"]:
                print(f"  - {summary['overlaps_found']} overlaps")
            if summary["short_sequences_found"]:
                print(
                    f"  - {summary['short_sequences_found']} sequences too short"
                )
            if summary["large_gaps_found"]:
                print(f"  - {summary['large_gaps_found']} large gaps")


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
