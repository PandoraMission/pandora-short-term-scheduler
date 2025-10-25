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

The implementation focuses on correctness and traceability: it stores a
`gap_report` with before/after statistics and keeps intermediate
assignments for testing and visualization.
"""

# Standard library
import copy
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from pandoravisibility import Visibility

from .models import ObservationSequence, ScienceCalendar, Visit


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

    def __init__(self, tle_line1: str, tle_line2: str) -> None:
        """
        Initialize the scheduler with TLE and parameters.

        Parameters:
        -----------
        tle_line1, tle_line2 : str
            TLE lines for satellite
        """
        # Validate TLE format
        if not isinstance(tle_line1, str):
            raise ValueError("Invalid TLE line 1 format")
        if not isinstance(tle_line2, str):
            raise ValueError("Invalid TLE line 2 format")
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2

        self.visibility = Visibility(tle_line1, tle_line2)

        self.min_sequence_duration = TimeDelta(2 * 60 * u.s)
        self.max_sequence_duration = TimeDelta(90 * 60 * u.s)

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

        # Analyze processed calendar
        self._analyze_processed_calendar(processed_calendar)

        # Generate comprehensive report
        self._finalize_gap_report()

        calendar_status = "VALID"
        if (
            len(
                self.validate_visibility(
                    processed_calendar, report_issues=False
                )
            )
            > 0
        ):
            print("Warning: Some visibility gaps remain unfilled.")
            calendar_status = "INVALID"
        if (
            len(
                self.validate_payload_exposures(
                    processed_calendar, report_issues=False
                )
            )
            > 0
        ):
            print("Warning: Some payload exposures are invalid.")
            calendar_status = "INVALID"
        if (
            len(
                self.validate_no_overlaps_astropy(
                    processed_calendar, report_issues=False
                )
            )
            > 0
        ):
            print("Warning: Some sequence timings are invalid.")
            calendar_status = "INVALID"
        if (
            self.validate_sequence_timing(
                processed_calendar, report_issues=False
            )["timing_summary"]["total_issues"]
            > 0
        ):
            print("Warning: Some sequence timings are invalid.")
            calendar_status = "INVALID"

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

        # Attach updated metadata to the processed calendar so writers
        # and callers can access TLEs, processing timestamp and the
        # comprehensive gap_report. This was intentionally assigned
        # here to ensure downstream XML output contains the processing
        # audit information.
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

        # Use initial time grid for processing
        total_minutes, start_time, end_time, time_grid = (
            self._get_synchronized_time_grid(working_calendar)
        )
        all_minutes_bool = np.zeros(total_minutes, dtype=bool)

        i = 0
        last_stop = deepcopy(start_time)

        for visit in working_calendar.visits:

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

        # last thing is to update all the payload parameters
        working_calendar = self._update_payload_parameters(working_calendar)

        return working_calendar

    def _fill_gaps(
        self, sequence: ObservationSequence, gap_length: int
    ) -> ObservationSequence:
        """
        Extend the start of a sequence backward in time to fill a gap.

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
            target_coord = SkyCoord(ra, dec, frame="icrs", unit="deg")

            # Check visibility of previous target during gap
            vis = self.visibility.get_visibility(target_coord, Time(gap_times))

            # Extend previous sequence if target is visible
            if np.any(vis):
                visible_times = np.array(gap_times)[vis]
                if len(visible_times) > 0:
                    last_visible_time = visible_times[-1]
                    prev_assignment = assignments[prev_idx]
                    visit_id = prev_assignment["visit_id"]
                    sequence_id = prev_assignment["sequence_id"]

                    # Get and extend the previous sequence
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

            # Shrink the current sequence to start after the gap
            if gap_start_idx < len(assignments):
                current_assignment = assignments[gap_start_idx]
                current_visit_id = current_assignment["visit_id"]
                current_sequence_id = current_assignment["sequence_id"]

                seq_to_shrink = working_cal.get_sequence(
                    current_visit_id, current_sequence_id
                )
                if seq_to_shrink:
                    # Calculate new start time (after the visibility gap)
                    gap_end_time = Time(gap_times[-1]) + 1 * u.minute

                    # Only shrink if there's still meaningful duration left
                    remaining_duration = seq_to_shrink.stop_time - gap_end_time
                    min_duration = self.min_sequence_duration

                    if remaining_duration >= min_duration:
                        shrunk_seq = ObservationSequence(
                            id=seq_to_shrink.id,
                            target=seq_to_shrink.target,
                            priority=seq_to_shrink.priority,
                            start_time=gap_end_time,
                            stop_time=seq_to_shrink.stop_time,
                            ra=seq_to_shrink.ra,
                            dec=seq_to_shrink.dec,
                            payload_params=deepcopy(
                                seq_to_shrink.payload_params
                            ),
                        )

                        working_cal.replace_sequence(
                            current_visit_id, current_sequence_id, shrunk_seq
                        )
                    else:
                        print("WARNING MINIMUM SEQUENCE DURATION ISSUE WITH")
                        print(
                            f"{seq_to_shrink.target}: {gap_end_time}, {seq_to_shrink.stop_time}"
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
        duration = sequence.duration.to(u.us)

        sequence = self._update_VDA_integrations(sequence, duration)
        sequence = self._update_NIRDA_integrations(sequence, duration)

        return sequence

    def _update_VDA_integrations(
        self, sequence: ObservationSequence, duration: TimeDelta
    ) -> ObservationSequence:

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
            duration_us = duration.to(u.us)

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
        self, sequence: ObservationSequence, duration: TimeDelta
    ) -> ObservationSequence:

        # Get parameters
        ROI_SizeX = int(
            sequence.get_payload_parameter("AcquireInfCamImages", "ROI_SizeX")
        )
        ROI_SizeY = int(
            sequence.get_payload_parameter("AcquireInfCamImages", "ROI_SizeY")
        )
        SC_Resets1 = int(
            sequence.get_payload_parameter("AcquireInfCamImages", "SC_Resets1")
        )
        SC_Resets2 = int(
            sequence.get_payload_parameter("AcquireInfCamImages", "SC_Resets2")
        )
        SC_DropFrames1 = int(
            sequence.get_payload_parameter(
                "AcquireInfCamImages", "SC_DropFrames1"
            )
        )
        SC_DropFrames2 = int(
            sequence.get_payload_parameter(
                "AcquireInfCamImages", "SC_DropFrames2"
            )
        )
        SC_DropFrames3 = int(
            sequence.get_payload_parameter(
                "AcquireInfCamImages", "SC_DropFrames3"
            )
        )
        SC_ReadFrames = int(
            sequence.get_payload_parameter(
                "AcquireInfCamImages", "SC_ReadFrames"
            )
        )
        SC_Groups = int(
            sequence.get_payload_parameter("AcquireInfCamImages", "SC_Groups")
        )

        seq_identifier = f"{sequence.id} ({sequence.target} @ {sequence.start_time.datetime.strftime('%m/%d %H:%M')})"

        required_params = {
            "SC_Groups": sequence.get_payload_parameter(
                "AcquireInfCamImages", "SC_Groups"
            ),
            "SC_ReadFrames": sequence.get_payload_parameter(
                "AcquireInfCamImages", "SC_ReadFrames"
            ),
            "SC_DropFrames1": sequence.get_payload_parameter(
                "AcquireInfCamImages", "SC_DropFrames1"
            ),
            "SC_DropFrames2": sequence.get_payload_parameter(
                "AcquireInfCamImages", "SC_DropFrames2"
            ),
            "SC_DropFrames3": sequence.get_payload_parameter(
                "AcquireInfCamImages", "SC_DropFrames3"
            ),
            "SC_Resets1": sequence.get_payload_parameter(
                "AcquireInfCamImages", "SC_Resets1"
            ),
            "SC_Resets2": sequence.get_payload_parameter(
                "AcquireInfCamImages", "SC_Resets2"
            ),
            "ROI_SizeX": sequence.get_payload_parameter(
                "AcquireInfCamImages", "ROI_SizeX"
            ),
            "ROI_SizeY": sequence.get_payload_parameter(
                "AcquireInfCamImages", "ROI_SizeY"
            ),
        }

        # Check for missing parameters
        missing_params = [
            name for name, value in required_params.items() if value is None
        ]

        if missing_params:
            print(
                f"Warning: Missing NIRDA parameters for sequence {seq_identifier}"
            )
            print(f"Missing parameters: {', '.join(missing_params)}")
            return sequence

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

        SC_Integrations = int(
            np.floor(sequence.duration.to(u.s) / integration_time)
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
        """Validate that all sequences have good visibility."""
        issues = []

        for visit in calendar.visits:
            for seq in visit.sequences:
                n_mins = int(np.rint(seq.duration.sec / 60.0))
                target_coord = SkyCoord(
                    seq.ra, seq.dec, frame="icrs", unit="deg"
                )
                deltas = np.arange(n_mins) * u.min
                times = seq.start_time + deltas

                vis = self.visibility.get_visibility(target_coord, times)
                if not np.all(vis):
                    issue = {
                        "sequence_id": seq.id,
                        "target": seq.target,
                        "start_time": seq.start_time,
                        "stop_time": seq.stop_time,
                        "visibility_fraction": np.sum(vis) / len(vis),
                    }
                    issues.append(issue)

                    if report_issues:
                        print(
                            f"Visibility issue: {seq.target} {seq.start_time} {seq.stop_time} {seq.id}"
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
        """
        Use Astropy's time comparison with proper tolerance.
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
            seq1 = all_sequences[i]["sequence"]
            seq2 = all_sequences[i + 1]["sequence"]

            # Check if seq1 ends significantly after seq2 starts
            if seq1.stop_time > (seq2.start_time + tolerance):
                overlap_duration = (
                    (seq1.stop_time - seq2.start_time).to(u.min).value
                )

                overlap_issue = {
                    "sequence1_id": seq1.id,
                    "sequence1_target": seq1.target,
                    "sequence1_start": seq1.start_time,
                    "sequence1_stop": seq1.stop_time,
                    "sequence2_id": seq2.id,
                    "sequence2_target": seq2.target,
                    "sequence2_start": seq2.start_time,
                    "sequence2_stop": seq2.stop_time,
                    "overlap_duration_minutes": overlap_duration,
                }
                overlaps.append(overlap_issue)

                if report_issues:
                    print("True overlap detected:")
                    print(
                        f"  Sequence {seq1.id} ({seq1.target}) ends at {seq1.stop_time}"
                    )
                    print(
                        f"  Sequence {seq2.id} ({seq2.target}) starts at {seq2.start_time}"
                    )
                    print(
                        f"  Overlap duration: {overlap_duration:.2f} minutes"
                    )
                    print()

        return overlaps

    def validate_sequence_timing(
        self, calendar: ScienceCalendar, report_issues: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive timing validation including overlaps, gaps, and minimum durations.

        Returns
        -------
        dict
            Dictionary with different types of timing issues
        """
        issues = {
            "overlaps": [],
            "short_sequences": [],
            "large_gaps": [],
            "timing_summary": {},
        }

        # Check for overlaps
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
        for seq_info in all_sequences:
            if seq_info["sequence"].duration < min_duration:
                short_issue = {
                    "sequence_id": seq_info["sequence"].id,
                    "target": seq_info["sequence"].target,
                    "visit_id": seq_info["visit_id"],
                    "start_time": seq_info["start_time"],
                    "stop_time": seq_info["stop_time"],
                    "duration_minutes": seq_info["duration_minutes"],
                    "minimum_required": min_duration,
                }
                issues["short_sequences"].append(short_issue)

        # Check for large gaps between sequences
        max_acceptable_gap = 2.0 * u.minute  # 2 minutes
        for i in range(len(all_sequences) - 1):
            seq1 = all_sequences[i]
            seq2 = all_sequences[i + 1]

            gap_duration = seq2["start_time"] - seq1["stop_time"]

            if gap_duration > max_acceptable_gap:
                gap_issue = {
                    "after_sequence": seq1["sequence"].id,
                    "after_target": seq1["sequence"].target,
                    "before_sequence": seq2["sequence"].id,
                    "before_target": seq2["sequence"].target,
                    "gap_start": seq1["stop_time"],
                    "gap_end": seq2["start_time"],
                    "gap_duration_minutes": gap_duration,
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
            print(f"Total sequences analyzed: {summary['total_sequences']}")
            print(f"Total timing issues found: {summary['total_issues']}")
            print()

            # Report overlaps
            if issues["overlaps"]:
                print(f"OVERLAPS ({len(issues['overlaps'])} found):")
                for i, overlap in enumerate(issues["overlaps"]):
                    print(
                        f"  {i+1}. Sequences {overlap['sequence1_id']} and {overlap['sequence2_id']}"
                    )
                    print(
                        f"     Overlap: {overlap['overlap_start']} to {overlap['overlap_end']} "
                        f"({overlap['overlap_duration_minutes']:.1f} min)"
                    )
            else:
                print("✓ OVERLAPS: None found")

            print()

            # Report short sequences
            if issues["short_sequences"]:
                print(
                    f"SHORT SEQUENCES ({len(issues['short_sequences'])} found, < {min_duration} min):"
                )
                for i, short in enumerate(issues["short_sequences"]):
                    print(
                        f"  {i+1}. Sequence {short['sequence_id']} ({short['target']}): "
                        f"{short['duration_minutes']:.1f} min"
                    )
            else:
                print("✓ SHORT SEQUENCES: None found")

            print()

            # Report large gaps
            if issues["large_gaps"]:
                print(
                    f"LARGE GAPS ({len(issues['large_gaps'])} found, > {max_acceptable_gap} min):"
                )
                for i, gap in enumerate(
                    issues["large_gaps"][:5]
                ):  # Show first 5
                    print(
                        f"  {i+1}. After sequence {gap['after_sequence']}: "
                        f"{gap['gap_duration_minutes']:.1f} min gap"
                    )
                if len(issues["large_gaps"]) > 5:
                    print(f"     ... and {len(issues['large_gaps']) - 5} more")
            else:
                print("✓ LARGE GAPS: None found")

        return issues

    def validate_payload_exposures(
        self, calendar: ScienceCalendar, report_issues: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Validate that payload exposure times (single exposure and total requested exposure)
        do not exceed the enclosing sequence duration.

        This currently checks the common VIS camera payload category
        `AcquireVisCamScienceData` for keys:
          - `ExposureTime_us` (microseconds per frame)
          - `NumTotalFramesRequested` (total frames)
          - `FramesPerCoadd` (used when NumTotalFramesRequested is not present)

        It also heuristically scans flattened payload parameters for any key containing
        the string `exposure` and checks single-exposure values against the sequence
        duration (assumes microseconds when the key ends with `_us`).

        Returns
        -------
        list
            A list of issue dicts found. Empty list if none.
        """
        issues = []

        for visit in calendar.visits:
            for seq in visit.sequences:
                seq_duration_sec = seq.duration.sec

                # 1) Check AcquireVisCamScienceData (VDA) - common VIS camera
                exposure_us = seq.get_payload_parameter(
                    "AcquireVisCamScienceData", "ExposureTime_us"
                )
                num_frames = seq.get_payload_parameter(
                    "AcquireVisCamScienceData", "NumTotalFramesRequested"
                )
                frames_per_coadd = seq.get_payload_parameter(
                    "AcquireVisCamScienceData", "FramesPerCoadd"
                )

                if exposure_us is not None:
                    try:
                        exposure_us_val = float(exposure_us)
                    except (ValueError, TypeError):
                        exposure_us_val = None

                    if exposure_us_val is not None:
                        single_exp_sec = exposure_us_val / 1e6
                        if single_exp_sec > seq_duration_sec:
                            issue = {
                                "visit_id": visit.id,
                                "sequence_id": seq.id,
                                "target": seq.target,
                                "problem": "single_exposure_longer_than_sequence",
                                "exposure_seconds": single_exp_sec,
                                "sequence_duration_seconds": seq_duration_sec,
                            }
                            issues.append(issue)
                            if report_issues:
                                print(
                                    f"PAYLOAD ISSUE: sequence {seq.id} single exposure ({single_exp_sec:.3f}s) "
                                    f"> sequence duration ({seq_duration_sec:.3f}s)"
                                )

                        # If total frames provided, check total exposure
                        if num_frames is not None:
                            try:
                                total_frames = int(num_frames)
                                total_exp_sec = (
                                    exposure_us_val * total_frames
                                ) / 1e6
                                if total_exp_sec > seq_duration_sec:
                                    issue = {
                                        "visit_id": visit.id,
                                        "sequence_id": seq.id,
                                        "target": seq.target,
                                        "problem": "total_exposure_longer_than_sequence",
                                        "total_exposure_seconds": total_exp_sec,
                                        "sequence_duration_seconds": seq_duration_sec,
                                    }
                                    issues.append(issue)
                                    if report_issues:
                                        print(
                                            f"PAYLOAD ISSUE: sequence {seq.id} total exposure ({total_exp_sec:.1f}s) "
                                            f"> sequence duration ({seq_duration_sec:.1f}s)"
                                        )
                            except (ValueError, TypeError):
                                # ignore parse errors; nothing to validate
                                pass

                        # If NumTotalFramesRequested not present but FramesPerCoadd is,
                        # check at least the coadd exposure vs duration (heuristic)
                        if num_frames is None and frames_per_coadd is not None:
                            try:
                                fpc = int(frames_per_coadd)
                                total_exp_sec = (exposure_us_val * fpc) / 1e6
                                if total_exp_sec > seq_duration_sec:
                                    issue = {
                                        "visit_id": visit.id,
                                        "sequence_id": seq.id,
                                        "target": seq.target,
                                        "problem": "coadd_exposure_longer_than_sequence",
                                        "coadd_exposure_seconds": total_exp_sec,
                                        "sequence_duration_seconds": seq_duration_sec,
                                    }
                                    issues.append(issue)
                                    if report_issues:
                                        print(
                                            f"PAYLOAD ISSUE: sequence {seq.id} coadd exposure ({total_exp_sec:.1f}s) "
                                            f"> sequence duration ({seq_duration_sec:.1f}s)"
                                        )
                            except (ValueError, TypeError):
                                pass

                # 2) Heuristic scan: any flattened payload key containing 'exposure'
                flat = seq.get_flat_payload_parameters()
                for key, val in flat.items():
                    if "exposure" in key.lower() and val is not None:
                        # skip keys we already handled
                        if key.startswith("AcquireVisCamScienceData"):
                            continue
                        try:
                            v = float(val)
                        except (ValueError, TypeError):
                            continue

                        # If key ends with _us assume microseconds, else seconds
                        if key.lower().endswith("_us"):
                            val_sec = v / 1e6
                        else:
                            val_sec = v

                        if val_sec > seq_duration_sec:
                            issue = {
                                "visit_id": visit.id,
                                "sequence_id": seq.id,
                                "target": seq.target,
                                "problem": "payload_exposure_field_longer_than_sequence",
                                "field": key,
                                "value_seconds": val_sec,
                                "sequence_duration_seconds": seq_duration_sec,
                            }
                            issues.append(issue)
                            if report_issues:
                                print(
                                    f"PAYLOAD ISSUE: sequence {seq.id} field {key} ({val_sec:.3f}s) "
                                    f"> sequence duration ({seq_duration_sec:.3f}s)"
                                )

        return issues

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
