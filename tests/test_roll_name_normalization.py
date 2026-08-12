"""Regression test: target-name normalization must not orphan swept rolls.

The roll sweep keys its results by target name. The ``fix_bad_data`` feature
rewrites invalid name symbols (``+`` and space -> ``_``). If that rename runs
*after* the sweep, a renamed target's swept roll is lost and the scheduler
falls back to the continuous sun-derived roll. ``process_calendar`` therefore
normalizes target names up front (before the sweep). These tests pin that:

- a ``+``/space target keeps a *quantized* (swept) roll, not a continuous
  sun-derived fallback;
- the target name is normalized in the output.
"""

# Standard library
from pathlib import Path

# Third-party
import numpy as np

# First-party/Local
import shortschedule
from shortschedule.parser import parse_science_calendar
from shortschedule.scheduler import ScheduleProcessor


class DummyVisibilityAllTrue:
    def __init__(self, l1, l2, **kwargs):
        pass

    def get_visibility(self, coord, times, roll=None):
        try:
            length = len(times)
        except Exception:
            return np.array([True], dtype=bool)
        return np.ones(length, dtype=bool)


def _load_sample():
    sample = (
        Path(shortschedule.__file__).parent
        / "data"
        / "Pandora_science_calendar_20251018_tsb-futz.xml"
    )
    return parse_science_calendar(sample)


def test_plus_space_target_keeps_swept_roll(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "shortschedule.scheduler.Visibility", DummyVisibilityAllTrue
    )

    cal = _load_sample()
    # Inject a target name carrying both a '+' and a space.
    bad_name = "BD+99 9999"
    first = cal.visits[0].sequences[0]
    for seq in cal.visits[0].sequences:
        if seq.target == first.target:
            seq.target = bad_name
    window_start = cal.visits[0].sequences[0].start_time.isot

    # Star-tracker constraints enable the roll sweep; fix_bad_data is on by
    # default. roll_step=1.0 => swept rolls are whole degrees.
    sched = ScheduleProcessor(
        "LINE1",
        "LINE2",
        st_sun_min=50,
        st_moon_min=20,
        st_earthlimb_min=30,
        roll_step=1.0,
        min_power_frac=0.0,
    )
    processed = sched.process_calendar(
        cal,
        window_start=window_start,
        window_duration_days=1,
        log_path=tmp_path / "run",
        verbose=False,
    )

    # The renamed target should appear normalized (no '+' or space).
    normalized = "BD_99_9999"
    matches = [
        seq
        for visit in processed.visits
        for seq in visit.sequences
        if seq.target == normalized
    ]
    assert matches, "normalized target name not found in output"
    assert not any(
        seq.target == bad_name
        for visit in processed.visits
        for seq in visit.sequences
    )

    # Its roll must be a swept (integer) value, not the continuous
    # sun-derived fallback that a post-rename orphaning would produce.
    for seq in matches:
        assert seq.roll is not None
        assert abs(seq.roll - round(seq.roll)) < 1e-9, (
            f"target {seq.target} fell back to a continuous roll "
            f"{seq.roll!r} instead of keeping its swept (integer) roll"
        )
