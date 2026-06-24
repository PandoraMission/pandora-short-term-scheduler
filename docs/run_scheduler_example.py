"""Example: run the short-term scheduler against a long-term science calendar.

This script takes a long-term science calendar (exported as XML), processes it
into a short-term schedule for a given observing window, runs the built-in
validation checks, writes the resulting calendar back out as XML, and produces
a few diagnostic plots.

Usage
-----
Update the three blocks marked ``UPDATE ME`` below (calendar path, observing
window, and spacecraft TLE), then run::

    python run_scheduler_example.py

The TLE, window, and calendar path are the only inputs that normally change
between runs; the scheduler configuration is mission-driven and is documented
inline where it is constructed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from astropy.time import Time

from shortschedule import ScheduleProcessor, XMLWriter, parse_science_calendar
from shortschedule.visualizer import ScheduleVisualizer


# ---------------------------------------------------------------------------
# Inputs — UPDATE ME between runs
# ---------------------------------------------------------------------------
xml_file_path = Path("path", "to", "Pandora_science_calendar.xml")

# Base name (without extension) used to label the output plots, e.g.
# "Pandora_science_calendar".
calendar_name = xml_file_path.stem

# Observing window: the short-term schedule is built starting at ``window_start``
# and spanning ``window_duration_days``.
window_start = Time("2026-06-22T00:00:00Z")
window_duration_days = 7

# Spacecraft two-line element set (TLE) describing the orbit used for
# visibility, power, and gap calculations. Refresh this for each new run.
new_tle1 = "1 67395U 80229J   26166.70761574  .00000000  00000-0  37770-3 0    09"
new_tle2 = "2 67395  97.8045 165.3849 0006889 124.8823 324.4856 14.88069544    01"

# ---------------------------------------------------------------------------
# Load the long-term calendar
# ---------------------------------------------------------------------------
print("\n\nParsing long-term calendar...")
original_calendar = parse_science_calendar(str(xml_file_path), verbose=True)

# ---------------------------------------------------------------------------
# Configure the scheduler
# ---------------------------------------------------------------------------
# These constraints are mission-driven and rarely change run-to-run. Gap
# tolerances are in minutes; the ``*_min`` keel-angle constraints are in
# degrees. See ScheduleProcessor for the full parameter documentation.
print("\n\nBuilding scheduler...")
scheduler = ScheduleProcessor(
    new_tle1,
    new_tle2,
    earthlimb_gap_tolerance=6,    # max allowed Earth-limb gap (minutes)
    st_gap_tolerance=12,          # max allowed star-tracker gap (minutes); raised per Tom 2026-06-01
    earthlimb_day_min=44,         # min Earth-limb angle, daytime (deg)
    earthlimb_night_min=13,       # min Earth-limb angle, nighttime (deg)
    sun_min=91,                   # min payload Sun keel angle (deg)
    moon_min=20,                  # min payload Moon keel angle (deg)
    st_sun_min=50,                # min star-tracker Sun angle (deg)
    st_moon_min=20,               # min star-tracker Moon angle (deg)
    st_earthlimb_min=30,          # min star-tracker Earth-limb angle (deg)
    roll_step=1.0,                # roll search step size (deg)
    min_power_frac=0.68,          # min acceptable orbit-average power fraction
    force_gap_fill=False,         # if True, aggressively fill scheduling gaps
    convert_single_roi_to_predefined=True,  # single auto-detect ROI -> predefined ROI at target RA/Dec
    # ----------------------------------------------------------------------
    # Per-priority payload overrides, by literal XML tag (CalendarCleaner
    # config.json format): {priority: {section: {xml_tag: value}}}. These are
    # forced onto every observation of the given priority (creating any
    # missing tag) before integration counts are recomputed, so ROI / coadd /
    # reset changes flow through. Priority keys may be ints or "Priority_N".
    override_payload_parameters={
        "Priority_0": {
            "Observational_Parameters": {
                "Boresight": {
                    "PRI_CMD_DIR": 9,
                },
            },
            "AcquireInfCamImages": {
                "ROI_StartX": 1737,
                "ROI_StartY": 962,
                "ROI_SizeX": 80,
                "ROI_SizeY": 250,
                "SC_Resets2": 1,
            },
            "AcquireVisCamScienceData": {
                "FramesPerCoadd": 50,
                "MaxNumStarRois": 1,
                "StarRoiDimension": 50,
            },
        },
        "Priority_1": {
            "Observational_Parameters": {
                "Boresight": {
                    "PRI_CMD_DIR": 9,
                },
            },
            "AcquireInfCamImages": {
                "ROI_StartX": 1737,
                "ROI_StartY": 962,
                "ROI_SizeX": 80,
                "ROI_SizeY": 250,
                "RiceX": 5,
                "RiceY": 28,
                "SC_Resets2": 1,
            },
            "AcquireVisCamScienceData": {
                "StarRoiDimension": 50,
                "FramesPerCoadd": 5,
            },
        },
        "Priority_2": {
            "Observational_Parameters": {
                "Boresight": {
                    "PRI_CMD_DIR": 9,
                },
            },
            "AcquireInfCamImages": {
                "ROI_StartX": 1737,
                "ROI_StartY": 962,
                "ROI_SizeX": 80,
                "ROI_SizeY": 250,
                "SC_Resets2": 1,
            },
            "AcquireVisCamScienceData": {
                "StarRoiDimension": 50,
                "FramesPerCoadd": 5,
            },
        },
    },
    # Field-name overrides (NirdaData/VisdaData fields; None -> class default)
    # are also available and recompute integrations; the XML-tag form above
    # is preferred when mirroring the cleaner config:
    # override_nirda_parameters={0: {"drop_frames_1": 2}},
    # override_visda_parameters={0: {"frames_per_coadd": 5}},
)

# ---------------------------------------------------------------------------
# Process the calendar into a short-term schedule
# ---------------------------------------------------------------------------
print("\n\nProcessing long-term into short-term calendar...")
processed_calendar = scheduler.process_calendar(
    original_calendar,
    window_start=window_start,
    window_duration_days=window_duration_days,
    merge_similar_observations=True
)

# ---------------------------------------------------------------------------
# Validate the result
# ---------------------------------------------------------------------------
# process_calendar already runs these checks internally; they are repeated
# here explicitly so the output can be inspected.
print("\n\nValidating visibility...")
issues = scheduler.validate_visibility(processed_calendar)
print("visibility issues:", issues)

print("\n\nValidating exposures...")
exposure_issues = scheduler.validate_payload_exposures(processed_calendar, report_issues=True)
print("exposure issues:", exposure_issues)

# Quick overlap check (analogous to validate_visibility).
print("\n\nValidating overlaps...")
overlap_issues = scheduler.validate_no_overlaps_astropy(processed_calendar)
print(f"Found {len(overlap_issues)} overlaps")

# Comprehensive sequence-timing validation.
print("\n\nValidating timing...")
all_timing_issues = scheduler.validate_sequence_timing(processed_calendar)

# Human-readable summaries.
print("\n\nPrinting diagnostics...")
scheduler.print_timing_summary(processed_calendar)
scheduler.print_gap_summary()
scheduler.print_validation_summary(processed_calendar)
processed_calendar.get_summary_stats()

# Per-day ".diag" report (week summary + per-day data volumes, priority
# counts, unique targets, observing/gap minutes and percentages, and a file
# manifest). Written next to the input calendar as "<calendar>.diag" when
# output_path is omitted; pass pass_data_volume_mb to report required passes.
print("\n\nGenerating .diag report...")
scheduler.generate_diagnostics(
    processed_calendar,
    output_path=xml_file_path,
    pass_data_volume_mb=281.3,
)


# ---------------------------------------------------------------------------
# Write the scheduled calendar back out as XML
# ---------------------------------------------------------------------------
print("\n\nWriting calendar...")
XMLWriter().write_calendar(processed_calendar, mission_phase="COM")


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------
print("\n\nCreating visualizations...")
visualizer = ScheduleVisualizer(scheduler)

# Gantt timeline coloured by observation priority; saved using the calendar name.
priority_fig = visualizer.plot_gantt_timeline_by_priority(
    processed_calendar,
    figsize=(12, 16),
    show_sequence_labels=False,
    title="Schedule by Priority",
)
priority_fig.savefig(f"{calendar_name}_priority.png", dpi=300)

# Gantt timeline overlaid with visibility windows; saved using the calendar name.
vis_fig = visualizer.plot_gantt_with_visibility(
    processed_calendar,
    figsize=(14, 18),
    show_sequence_labels=False,
    title="Schedule — Visibility Check",
)
vis_fig.savefig(f"{calendar_name}_visibility.png", dpi=300)

# Simple timeline view including visit boundaries.
fig, ax = visualizer.plot_timeline(processed_calendar, show_visits=True)

# Display all open figures.
plt.show()
