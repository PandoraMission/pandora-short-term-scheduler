"""Tests for keeping the target inside its keepouts while the spacecraft is on it.

The MOC slews before an observation's start (the NIRDA start) and leaves
after its stop (payload control off), so the spacecraft is settled on the
target 50 s before the start and stays on it 64 s after the stop. The
pointing pad moves a boundary in, to the second, until the target is
visible over those stretches too.
"""

# Third-party
import numpy as np
from astropy import units as u

# First-party/Local
from shortschedule.nirda import NirdaData
from shortschedule.overhead import OverheadTiming
from shortschedule.scheduler import ScheduleProcessor
from shortschedule.visda import VisdaData
from tests.test_edge_refinement import (
    _at,
    _calendar,
    _off_by,
    _SecondsVis,
    _seq,
)


def _processor(monkeypatch, *spans, **kwargs):
    monkeypatch.setattr(
        "shortschedule.scheduler.Visibility",
        lambda *a, **kw: _SecondsVis(*spans),
    )
    kwargs.setdefault("st_gap_tolerance_start_buffer", 0)
    kwargs.setdefault("earthlimb_gap_tolerance_start_buffer", 0)
    proc = ScheduleProcessor("L1", "L2", **kwargs)
    proc._initialize_gap_report()
    proc.printed = []
    proc._print = lambda message, *a, **kw: proc.printed.append(message)
    return proc


def _only(calendar):
    (visit,) = calendar.visits
    (seq,) = visit.sequences
    return seq


# ---------------------------------------------------------------------------
# The pass on its own
# ---------------------------------------------------------------------------
def test_defaults_are_the_moc_times(monkeypatch):
    proc = _processor(monkeypatch, (0, 10_000))
    assert proc.overhead.settled_before_start_time == 50 * u.s
    assert proc.overhead.on_target_after_stop_time == 64 * u.s


def test_a_start_on_a_rising_edge_moves_50_s_later(monkeypatch):
    """Visible from minute 10: settled 50 s before the start needs 10:50."""
    proc = _processor(monkeypatch, (600, 3600))
    cal = _calendar(_seq(1, _at(10), _at(40)))

    proc._pad_edges_for_pointing(cal)

    seq = _only(cal)
    assert _off_by(seq.start_time, _at(10, 50)) < 0.01
    assert _off_by(seq.stop_time, _at(40)) < 0.01


def test_a_stop_on_a_setting_edge_moves_64_s_earlier(monkeypatch):
    """Visible until (not at) minute 30: on target 64 s after the stop needs
    the stop at 28:56, the last visible second being 29:59."""
    proc = _processor(monkeypatch, (0, 1800))
    cal = _calendar(_seq(1, _at(5), _at(30)))

    proc._pad_edges_for_pointing(cal)

    seq = _only(cal)
    assert _off_by(seq.start_time, _at(5)) < 0.01
    assert _off_by(seq.stop_time, _at(28, 56)) < 0.01


def test_boundaries_inside_visibility_are_not_touched(monkeypatch):
    proc = _processor(monkeypatch, (0, 10_000))
    cal = _calendar(_seq(1, _at(10), _at(40)), _seq(2, _at(40), _at(70)))

    proc._pad_edges_for_pointing(cal)

    first, second = cal.visits[0].sequences
    assert _off_by(first.start_time, _at(10)) < 0.01
    assert _off_by(first.stop_time, _at(40)) < 0.01
    assert _off_by(second.start_time, _at(40)) < 0.01
    assert _off_by(second.stop_time, _at(70)) < 0.01
    summary = proc.gap_report["processing_summary"]
    assert summary["pointing_pad_boundaries_moved"] == 0


def test_a_partly_clear_pad_moves_only_as_far_as_needed(monkeypatch):
    """Visible from 20 s before minute 10: 30 s more clears the 50 s pad."""
    proc = _processor(monkeypatch, (580, 3600))
    cal = _calendar(_seq(1, _at(10), _at(40)))

    proc._pad_edges_for_pointing(cal)

    assert _off_by(_only(cal).start_time, _at(10, 30)) < 0.01


def test_pads_set_by_the_caller(monkeypatch):
    proc = _processor(
        monkeypatch,
        (600, 1800),
        pointing_pad_before_start=30 * u.s,
        pointing_pad_after_stop=10 * u.s,
    )
    cal = _calendar(_seq(1, _at(10), _at(30)))

    proc._pad_edges_for_pointing(cal)

    seq = _only(cal)
    assert _off_by(seq.start_time, _at(10, 30)) < 0.01
    assert _off_by(seq.stop_time, _at(29, 50)) < 0.01


def test_zero_pads_turn_it_off(monkeypatch):
    proc = _processor(
        monkeypatch,
        (600, 1800),
        pointing_pad_before_start=0 * u.s,
        pointing_pad_after_stop=0 * u.s,
    )
    cal = _calendar(_seq(1, _at(10), _at(30)))

    proc._pad_edges_for_pointing(cal)

    seq = _only(cal)
    assert _off_by(seq.start_time, _at(10)) < 0.01
    assert _off_by(seq.stop_time, _at(30)) < 0.01


def test_a_boundary_that_never_clears_is_left_with_a_warning(monkeypatch):
    """Dark for 1 s every 40 s: no 51 s clear stretch anywhere."""
    spans = [(t + 1, t + 40) for t in range(0, 4000, 40)]
    proc = _processor(monkeypatch, *spans, pointing_pad_after_stop=0 * u.s)
    cal = _calendar(_seq(1, _at(10, 1), _at(40)))

    proc._pad_edges_for_pointing(cal)

    assert _off_by(_only(cal).start_time, _at(10, 1)) < 0.01
    assert any("POINTING PAD" in m and "left at" in m for m in proc.printed)


def test_a_boundary_inside_a_tolerated_gap_is_not_touched(monkeypatch):
    """Dark from 38:00 on: the stop at 40:00 ends inside a gap the
    tolerances let it ride through, so it is no visibility edge; the start
    at 10:00, inside a gap that ends at 10:30, likewise."""
    proc = _processor(monkeypatch, (630, 38 * 60))
    cal = _calendar(_seq(1, _at(10), _at(40)))

    proc._pad_edges_for_pointing(cal)

    seq = _only(cal)
    assert _off_by(seq.start_time, _at(10)) < 0.01
    assert _off_by(seq.stop_time, _at(40)) < 0.01
    assert not any("POINTING PAD" in m for m in proc.printed)
    assert any("2 inside a tolerated gap" in m for m in proc.printed)


def test_a_priority_0_cut_under_the_minimum_is_dropped(monkeypatch):
    proc = _processor(monkeypatch, (600, 1080), drop_priority0=True)
    seq = _seq(1, _at(10), _at(18))
    seq.priority = 0
    cal = _calendar(seq)

    proc._pad_edges_for_pointing(cal)

    assert cal.visits == []
    assert proc.gap_report["processing_summary"]["priority_0_dropped"] == 1


def test_a_priority_1_cut_under_the_minimum_is_kept_with_a_warning(
    monkeypatch,
):
    proc = _processor(monkeypatch, (600, 1080), drop_priority0=True)
    cal = _calendar(_seq(1, _at(10), _at(18)))

    proc._pad_edges_for_pointing(cal)

    seq = _only(cal)
    assert _off_by(seq.start_time, _at(10, 50)) < 0.01
    assert _off_by(seq.stop_time, _at(16, 56)) < 0.01
    assert any("POINTING PAD" in m and "under" in m for m in proc.printed)


def test_the_change_is_noted_and_counted(monkeypatch):
    proc = _processor(monkeypatch, (600, 1800))
    cal = _calendar(_seq(1, _at(10), _at(30)))

    proc._pad_edges_for_pointing(cal)

    summary = proc.gap_report["processing_summary"]
    assert summary["pointing_pad_boundaries_moved"] == 2
    assert summary["pointing_pad_seconds_cut"] == 50 + 64


# ---------------------------------------------------------------------------
# In the pipeline, and in the header
# ---------------------------------------------------------------------------
def test_the_pipeline_pads_after_refining(monkeypatch):
    """Visible until 30:25: refined to 30:25, then padded to 29:21."""
    proc = _processor(monkeypatch, (0, 30 * 60 + 25))
    cal = _calendar(_seq(1, _at(10), _at(40)))

    result = proc._process_all_sequences(cal)

    assert _off_by(_only(result).stop_time, _at(29, 21)) < 0.01


def test_the_header_reports_the_pads(monkeypatch):
    proc = _processor(monkeypatch, (0, 10))
    settings = proc._settings_for_header()
    assert settings["Pointing_Pad_Before_Start_Sec"] == "50"
    assert settings["Pointing_Pad_After_Stop_Sec"] == "64"


# ---------------------------------------------------------------------------
# Detectors end by payload control off (the calendar stop)
# ---------------------------------------------------------------------------
def test_nirda_integrations_fill_the_window_and_end_by_the_stop():
    nirda = NirdaData()
    duration = 1800 * u.s
    n, _, _ = nirda.solve_integrations(duration, OverheadTiming())
    # The solver fits the whole run, dropped integrations included, then
    # commands that many less the drops.
    n += nirda.dropped_integrations
    used = (
        nirda.first_integration_time + (n - 1) * nirda.other_integration_time
    )
    assert used <= duration
    # One more would not fit: nothing is held back for overheads.
    assert used + nirda.other_integration_time > duration


def test_visda_frames_fill_the_window_after_its_2_s_and_end_by_the_stop():
    visda = VisdaData()
    duration = 1800 * u.s
    frames, _, _ = visda.solve_integrations(duration, OverheadTiming())
    used = frames * visda.single_frame_time
    assert 2 * u.s + used <= duration
    # One more coadd would not fit.
    coadd = max(1, visda.frames_per_coadd) * visda.single_frame_time
    assert 2 * u.s + used + coadd > duration
    assert np.isclose(
        OverheadTiming().visda_pre_overhead_time.to_value(u.s), 2
    )
