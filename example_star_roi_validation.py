#!/usr/bin/env python3
"""
Example script demonstrating the MaxNumStarRois validation and fix.

This script shows how to:
1. Check for MaxNumStarRois/numPredefinedStarRois mismatches
2. Automatically fix them when writing calendars
"""

from shortschedule.parser import parse_science_calendar
from shortschedule.scheduler import ScheduleProcessor
from shortschedule.writer import XMLWriter


def main():
    # Parse a science calendar
    calendar_path = "./src/shortschedule/data/Pandora_science_calendar_20251018_tsb-futz.xml"
    print(f"Parsing calendar: {calendar_path}")
    cal = parse_science_calendar(calendar_path)
    print(
        f"Loaded {len(cal.visits)} visits with {cal.total_sequences} sequences"
    )

    # Create a scheduler (requires TLE lines, but we just need it for validation)
    # These are example TLE lines - in production, use current TLEs
    tle1 = (
        "1 25544U 98067A   21001.00000000  .00002182  00000-0  41420-4 0  9990"
    )
    tle2 = (
        "2 25544  51.6461 339.8014 0002571  34.5857 120.4689 15.48908950266831"
    )
    sched = ScheduleProcessor(tle1, tle2)

    # Validate star ROI consistency
    print("\nValidating MaxNumStarRois == numPredefinedStarRois...")
    issues = sched.validate_star_roi_consistency(cal, report_issues=False)

    if issues:
        print(f"⚠️  Found {len(issues)} sequences with mismatched values")
        print("\nExample issues:")
        for issue in issues[:3]:
            print(
                f"  - Sequence {issue['sequence_id']}: "
                f"MaxNumStarRois={issue['MaxNumStarRois']}, "
                f"numPredefinedStarRois={issue['numPredefinedStarRois']}"
            )

        # Write the calendar - this will automatically fix the issues
        print("\nWriting calendar with automatic fix...")
        writer = XMLWriter()
        output_path = "fixed_calendar.xml"
        writer.write_calendar(cal, output_path=output_path)
        print(f"✓ Written corrected calendar to: {output_path}")

        # Verify the fix
        cal2 = parse_science_calendar(output_path)
        issues2 = sched.validate_star_roi_consistency(
            cal2, report_issues=False
        )
        print(f"\n✓ After fix: {len(issues2)} issues remaining")

        if len(issues2) == 0:
            print("✅ SUCCESS: All issues automatically fixed!")
    else:
        print(
            "✅ No issues found - all sequences already have matching values"
        )


if __name__ == "__main__":
    main()
