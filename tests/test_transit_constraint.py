"""The priority-2 transit constraint.

A transit's priority-2 observations must keep ``transit_duration_constraint_min``
visible minutes of its core (the middle ``transit_middle_frac`` of T14)
between them; a transit already short of that may lose none. Tested: the
ephemeris reading and core timing, the guard itself, and that a pass
(start buffers) is held back by it.
"""

# Standard library
import json
import shutil
import subprocess

# Third-party
import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

# First-party/Local
from shortschedule.models import ObservationSequence, ScienceCalendar, Visit
from shortschedule.scheduler import ScheduleProcessor
from shortschedule.transit import (
    load_ephemerides,
    planet_key,
    transit_core,
    transit_epoch,
)
from tests.doubles import answers_visibility

EPHEMERIS = {
    "planet_name": "TEST-1b",
    "period_days": 3.0,
    "epoch_bjd_tdb": Time("2026-07-16T12:00:00", scale="tdb").jd,
    "duration_hr": 2.0,
    "ra_deg": 10.0,
    "dec_deg": 20.0,
}
# Core of transit 0: the middle 90 of its 120 min
CORE_START, CORE_STOP = transit_core(EPHEMERIS, 0, 0.75)


class _AlwaysVisible:
    def __init__(self, *args, **kwargs):
        pass

    @answers_visibility
    def get_visibility(self, coord, times, roll=None):
        return np.ones(len(np.atleast_1d(times.jd)), dtype=bool)


def _seq(sid, start_min, stop_min, priority=2, target="TEST-1b"):
    """An observation from ``start_min`` to ``stop_min`` past the core start."""
    return ObservationSequence(
        id=sid,
        target=target,
        priority=priority,
        start_time=CORE_START + start_min * u.min,
        stop_time=CORE_START + stop_min * u.min,
        ra=10.0,
        dec=20.0,
        payload_params={},
        roll=0.0,
    )


@pytest.fixture
def processor(monkeypatch):
    monkeypatch.setattr(
        "shortschedule.scheduler.Visibility", _AlwaysVisible
    )
    scheduler = ScheduleProcessor("LINE1", "LINE2")
    scheduler.transit_ephemerides = {planet_key("TEST-1b"): EPHEMERIS}
    return scheduler


def _calendar(*sequences):
    return ScienceCalendar(metadata={}, visits=[Visit("v1", list(sequences))])


class TestEphemerides:
    def test_complete_definitions_are_read(self, tmp_path):
        tree = tmp_path / "target_definition_files" / "exoplanet"
        tree.mkdir(parents=True)
        full = {
            "Planet Name": "HAT-P-11 b",
            "Period (days)": 4.88,
            "Transit Epoch (BJD_TDB)": 2454957.8,
            "Transit Duration (hrs)": 2.36,
            "RA": 297.7,
            "DEC": 48.1,
        }
        (tree / "HAT-P-11b_target_definition.json").write_text(
            json.dumps(full)
        )
        partial = dict(full, **{"Planet Name": "X-1b"})
        del partial["Transit Duration (hrs)"]
        (tree / "X-1b_target_definition.json").write_text(
            json.dumps(partial)
        )

        ephemerides = load_ephemerides(tmp_path)
        assert set(ephemerides) == {planet_key("HAT-P-11b")}
        assert ephemerides["hatp11b"]["period_days"] == 4.88

    def test_a_missing_tree_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_ephemerides(tmp_path)

    def test_one_tree_only_and_the_most_precise_wins(self, tmp_path):
        """A clone nested in the checkout (another branch) is not read;
        within the tree, precision beats path order."""
        definition = {
            "Planet Name": "TOI-270 d",
            "Period (days)": 11.38,
            "Transit Epoch (BJD_TDB)": 2458389.5,
            "Transit Duration (hrs)": 2.1,
            "RA": 68.4,
            "DEC": -51.9,
            "Period Uncertainty (days)": 1e-4,
            "Transit Epoch Uncertainty (days)": 1e-3,
        }
        loose = tmp_path / "target_definition_files" / "exoplanet"
        tight = tmp_path / "target_definition_files" / "time-critical"
        only_nested = tmp_path / "Nested" / "target_definition_files" / "x"
        for folder in (loose, tight, only_nested):
            folder.mkdir(parents=True)
        (loose / "TOI-270d_target_definition.json").write_text(
            json.dumps(definition)
        )
        (tight / "TOI-270d_target_definition.json").write_text(
            json.dumps(dict(definition, **{"Period (days)": 11.379}))
            .replace("0.0001", "1e-06")
        )
        (only_nested / "HD_20329b_target_definition.json").write_text(
            json.dumps(dict(definition, **{"Planet Name": "HD_20329b"}))
        )

        ephemerides = load_ephemerides(
            tmp_path, reference_time=Time("2026-11-05")
        )
        assert set(ephemerides) == {"toi270d"}
        assert ephemerides["toi270d"]["period_days"] == 11.379
        assert "time-critical" in ephemerides["toi270d"]["definition_file"]

    @pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
    def test_only_files_the_branch_tracks_are_read(self, tmp_path):
        tree = tmp_path / "target_definition_files" / "exoplanet"
        tree.mkdir(parents=True)
        definition = {
            "Planet Name": "A-1b",
            "Period (days)": 1.5,
            "Transit Epoch (BJD_TDB)": 2459000.0,
            "Transit Duration (hrs)": 1.0,
            "RA": 1.0,
            "DEC": 2.0,
        }
        for name in ("A-1b", "B-1b"):
            (tree / f"{name}_target_definition.json").write_text(
                json.dumps(dict(definition, **{"Planet Name": name}))
            )
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run(git + ["init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(
            git
            + ["add", "target_definition_files/exoplanet/A-1b_target_definition.json"],
            cwd=tmp_path,
            check=True,
        )
        subprocess.run(
            git + ["commit", "-q", "-m", "a"], cwd=tmp_path, check=True
        )

        assert set(load_ephemerides(tmp_path)) == {"a1b"}

    def test_core_is_the_middle_of_the_transit(self):
        mid = CORE_START + (CORE_STOP - CORE_START) / 2
        # Barycentric correction: at most 8.3 min of light travel
        naive = Time(EPHEMERIS["epoch_bjd_tdb"], format="jd", scale="tdb")
        assert abs((mid - naive).to_value(u.min)) < 8.4
        assert (CORE_STOP - CORE_START).to_value(u.min) == pytest.approx(90)
        assert transit_epoch(EPHEMERIS, mid + 1.4 * u.day) == 0
        assert transit_epoch(EPHEMERIS, mid + 1.6 * u.day) == 1


class TestGuard:
    def test_a_shrink_keeping_enough_is_allowed(self, processor):
        seq = _seq("s1", 0, 40)
        cal = _calendar(seq)
        assert processor._transit_allows(
            cal, "v1", seq, seq.start_time + 20 * u.min, seq.stop_time, "test"
        )

    def test_a_shrink_under_the_minimum_is_refused(self, processor):
        seq = _seq("s1", 0, 40)
        cal = _calendar(seq)
        assert not processor._transit_allows(
            cal, "v1", seq, seq.start_time + 30 * u.min, seq.stop_time, "test"
        )
        summary = processor.gap_report["processing_summary"]
        assert summary["transit_changes_refused"] == 1

    def test_coverage_is_summed_over_the_transit(self, processor):
        """Another observation of the same transit carries the rest."""
        seq = _seq("s1", 0, 40)
        cal = _calendar(seq, _seq("s2", 60, 72))
        assert processor._transit_allows(
            cal, "v1", seq, seq.start_time + 37 * u.min, seq.stop_time, "test"
        )

    def test_a_short_transit_may_lose_nothing(self, processor):
        seq = _seq("s1", -30, 10)  # 10 min of the core
        cal = _calendar(seq)
        assert processor._transit_allows(
            cal, "v1", seq, seq.start_time + 20 * u.min, seq.stop_time, "test"
        )
        assert not processor._transit_allows(
            cal, "v1", seq, seq.start_time, seq.stop_time - 5 * u.min, "test"
        )

    def test_overlap_repair_is_logged_not_refused(self, processor):
        seq = _seq("s1", 0, 40)
        cal = _calendar(seq)
        assert processor._transit_allows(
            cal,
            "v1",
            seq,
            seq.start_time,
            seq.start_time + 5 * u.min,
            "test",
            refuse=False,
        )

    @pytest.mark.parametrize("priority", [0, 1])
    def test_other_priorities_are_free(self, processor, priority):
        seq = _seq("s1", 0, 40, priority=priority)
        cal = _calendar(seq)
        assert processor._transit_allows(
            cal, "v1", seq, seq.start_time + 39 * u.min, seq.stop_time, "test"
        )

    def test_off_without_a_target_list(self, processor):
        processor.transit_ephemerides = {}
        seq = _seq("s1", 0, 40)
        cal = _calendar(seq)
        assert processor._transit_allows(
            cal, "v1", seq, seq.start_time + 39 * u.min, seq.stop_time, "test"
        )


def test_start_buffer_is_held_back(processor):
    """A start buffer wanting 30 min of a 40 min core observation."""
    seq = _seq("s1", 0, 40)
    cal = _calendar(seq)
    ok = np.arange(40) >= 30
    processor._start_buffer_requirements = lambda s, times, coord: [
        ("star trackers are not settled", 5, ok[: len(times)])
    ]
    processor._enforce_start_buffers(cal)
    assert cal.visits[0].sequences[0].start_time == CORE_START
