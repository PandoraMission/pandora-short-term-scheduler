"""The NIRDA trace check of the roll search (pandoravisibility v2.1.0).

The check itself is upstream's. What is tested here is the plumbing: the
kwargs reach both visibility models, a trace rejection is told apart from
a target no roll can see, and a trace-rejected target leaves its visit.
"""

# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

# First-party/Local
import shortschedule
from shortschedule.models import ObservationSequence, Visit
from shortschedule.parser import parse_science_calendar
from shortschedule.roll import get_best_roll_per_visit
from shortschedule.scheduler import ScheduleProcessor

TLE1 = "1 67395U 80229J   26223.79570602  .00000000  00000-0  37770-3 0    02"
TLE2 = "2 67395  97.8072 221.5977 0003967 320.1766 134.7478 14.88176196    03"
T0 = Time("2026-07-16T00:00:00", scale="utc")


class _TraceVisibility:
    """Visible everywhere; the roll search rejects ``blocked`` RAs on trace.

    A rejected target comes back from the search with nothing visible and
    ``trace_ok`` False, as upstream reports it. ``seen_at_fixed_roll``
    is what a fixed-roll query of it then answers.
    """

    trace_contamination_tol = 1e-3

    def __init__(self, *args, blocked=(), seen_at_fixed_roll=True, **kwargs):
        self.blocked = set(blocked)
        self.seen_at_fixed_roll = seen_at_fixed_roll

    def get_visibility(
        self,
        coord,
        times,
        roll=None,
        *,
        optimize_roll=False,
        roll_step=None,
        min_power_frac=None,
        weights=None,
    ):
        n_times = len(np.atleast_1d(times.jd))
        rejected = round(float(coord.ra.deg), 6) in self.blocked
        if optimize_roll:
            visible = np.full(n_times, not rejected)
        else:
            visible = np.full(n_times, self.seen_at_fixed_roll or not rejected)
        return {
            "roll_deg": np.full(n_times, 30.0),
            "visible": visible,
            "boresight_visible": visible,
            "n_visible": int(visible.sum()),
            "n_st_pass": visible.astype(int),
            "solar_power_frac": np.where(visible, 1.0, np.nan),
            "trace_contamination": np.full(
                n_times, 5e-3 if rejected else 1e-4
            ),
            "trace_ok": np.full(n_times, not rejected),
            "trace_min_contamination": np.full(n_times, 2e-3),
            "trace_min_roll_deg": np.full(n_times, -150.0),
        }


def _visit(ra=10.0):
    start = T0
    return Visit(
        "v1",
        [
            ObservationSequence(
                id="s1",
                target="A",
                priority=1,
                start_time=start,
                stop_time=start + 10 * u.min,
                ra=ra,
                dec=20.0,
                payload_params={},
            )
        ],
    )


class TestRollResult:
    def test_trace_fields_are_reduced_to_the_held_roll(self):
        result = get_best_roll_per_visit(_visit(), _TraceVisibility())["A"]
        assert result["roll_deg"] == 30.0
        assert result["trace_contamination"] == 1e-4
        assert result["trace_ok"] is True
        assert result["trace_min_roll_deg"] == -150.0
        assert result["trace_blocked"] is False

    def test_a_trace_rejection_is_blocked(self):
        vis = _TraceVisibility(blocked={10.0})
        result = get_best_roll_per_visit(_visit(), vis)["A"]
        assert result["n_scheduled_visible"] == 0
        assert result["trace_blocked"] is True

    def test_a_target_no_roll_sees_is_not_blocked(self):
        """Dark at the held roll even without the trace check."""
        vis = _TraceVisibility(blocked={10.0}, seen_at_fixed_roll=False)
        result = get_best_roll_per_visit(_visit(), vis)["A"]
        assert result["trace_blocked"] is False


class TestKwargs:
    def test_unset_defers_to_pandoravisibility(self):
        scheduler = ScheduleProcessor(TLE1, TLE2)
        vis = scheduler.visibility
        assert vis.use_trace_sweep is True
        assert vis.trace_contamination_tol == pytest.approx(1e-3)
        assert (vis.nirda_roi_x, vis.nirda_roi_y) == (80, 256)

    def test_reach_both_models(self):
        scheduler = ScheduleProcessor(
            TLE1,
            TLE2,
            priority_0_earthlimb_min=69,
            use_trace_sweep=False,
            trace_contamination_tol=3e-3,
            nirda_roi_x=60,
            nirda_roi_y=200,
        )
        for vis in (scheduler.visibility, scheduler.priority_0_visibility):
            assert vis.use_trace_sweep is False
            assert vis.trace_contamination_tol == pytest.approx(3e-3)
            assert (vis.nirda_roi_x, vis.nirda_roi_y) == (60, 200)


@pytest.mark.parametrize("drop", [True, False])
def test_a_trace_rejected_target_is_dropped_or_reported(
    monkeypatch, tmp_path, drop
):
    """Dropped with a warning when asked, otherwise kept with an error."""
    sample = (
        Path(shortschedule.__file__).parent
        / "data"
        / "Pandora_science_calendar_20251018_tsb-futz.xml"
    )
    calendar = parse_science_calendar(sample)
    window_start = calendar.visits[0].sequences[0].start_time
    window_end = window_start + 1 * u.day
    rejected = calendar.visits[0].sequences[0]
    key = round(float(rejected.ra), 6)
    in_window = [
        seq
        for visit in calendar.visits
        for seq in visit.sequences
        if window_start <= seq.start_time < window_end
    ]
    n_rejected = sum(round(float(s.ra), 6) == key for s in in_window)
    assert 0 < n_rejected < len(in_window)

    monkeypatch.setattr(
        "shortschedule.scheduler.Visibility",
        lambda *a, **k: _TraceVisibility(blocked={key}),
    )
    scheduler = ScheduleProcessor(
        "LINE1", "LINE2", drop_observations_based_on_trace=drop
    )
    processed = scheduler.process_calendar(
        calendar,
        window_start=window_start.isot,
        window_duration_days=1,
        log_path=tmp_path / "run",
        verbose=False,
    )

    remaining = [s for v in processed.visits for s in v.sequences]
    n_kept = sum(round(float(s.ra), 6) == key for s in remaining)
    summary = scheduler.get_gap_report()["processing_summary"]
    errors_log = "".join(
        path.read_text() for path in tmp_path.glob("*errors*")
    )
    trace_lines = [
        line for line in errors_log.splitlines() if "NIRDA trace" in line
    ]
    assert trace_lines
    if drop:
        assert n_kept == 0
        assert all(visit.sequences for visit in processed.visits)
        assert summary["trace_dropped"] == n_rejected
        assert all("WARNING" in line and "Dropped" in line for line in trace_lines)
    else:
        assert n_kept > 0
        assert summary["trace_dropped"] == 0
        assert all("ERROR" in line and "Kept" in line for line in trace_lines)
