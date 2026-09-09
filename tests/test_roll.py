"""Tests for get_best_roll_per_visit.

The search is pandoravisibility's; what is tested here is the visit rule
built around it: one call per target per visit, which minutes are scored
and how much each counts, which keepout model applies, and that the roll
lands on every sequence of the target.
"""

# Third-party
import numpy as np
from astropy import units as u
from astropy.time import Time

# First-party/Local
from shortschedule.models import ObservationSequence, Visit
from shortschedule.roll import get_best_roll_per_visit

T0 = Time("2026-07-16T00:00:00", scale="utc")


def _make_seq(sid, target, start_min, duration_min, ra=10.0, dec=20.0,
              priority=1):
    start = T0 + start_min * u.min
    return ObservationSequence(
        id=sid,
        target=target,
        priority=priority,
        start_time=start,
        stop_time=start + duration_min * u.min,
        ra=ra,
        dec=dec,
        payload_params={},
    )


def _minutes(times):
    return np.rint((times - T0).to_value(u.min)).astype(int)


class _RecordingVisibility:
    """A get_visibility that records the roll searches asked of it.

    Answers ``roll_deg`` (a number, or a function of the call count) and
    marks visible whatever ``visible`` says of the times, everything by
    default.
    """

    def __init__(self, roll_deg=30.0, visible=None):
        self.calls = []
        self.roll_deg = roll_deg
        self.visible = visible

    def get_visibility(self, coord, times, roll=None, *, optimize_roll=False,
                       roll_step=None, min_power_frac=None, weights=None):
        assert optimize_roll and roll is None, "the visit rule searches"
        self.calls.append(dict(coord=coord, times=times, roll_step=roll_step,
                               min_power_frac=min_power_frac,
                               weights=weights))
        visible = (np.ones(len(times), dtype=bool) if self.visible is None
                   else self.visible(times))
        roll = (self.roll_deg(len(self.calls)) if callable(self.roll_deg)
                else self.roll_deg)
        return {
            # Echoed per timestep, as pandoravisibility v2.0.0 does.
            "roll_deg": np.full(len(times), float(roll)),
            "n_visible": int(visible.sum()),
            "visible": visible,
            "boresight_visible": visible,
            "n_st_pass": visible.astype(int),
            "solar_power_frac": np.where(visible, 1.0, np.nan),
        }


def test_same_target_in_a_visit_shares_one_roll():
    vis = _RecordingVisibility(roll_deg=30.0)
    visit = Visit("v1", [_make_seq("s1", "A", 0, 10),
                         _make_seq("s2", "A", 100, 10)])
    result = get_best_roll_per_visit(visit, vis)
    assert len(vis.calls) == 1
    assert set(result) == {"A"}
    assert [seq.roll for seq in visit.sequences] == [30.0, 30.0]


def test_targets_in_a_visit_are_solved_independently():
    vis = _RecordingVisibility(roll_deg=lambda n: 10.0 * n)
    visit = Visit("v1", [_make_seq("s1", "A", 0, 10, ra=10.0),
                         _make_seq("s2", "B", 20, 10, ra=50.0),
                         _make_seq("s3", "A", 40, 10, ra=10.0)])
    get_best_roll_per_visit(visit, vis)
    assert len(vis.calls) == 2
    assert [call["coord"].ra.deg for call in vis.calls] == [10.0, 50.0]
    rolls = {seq.target: seq.roll for seq in visit.sequences}
    assert rolls == {"A": 10.0, "B": 20.0}


def test_same_target_in_another_visit_gets_its_own_roll():
    vis = _RecordingVisibility(roll_deg=lambda n: 10.0 * n)
    first = Visit("v1", [_make_seq("s1", "A", 0, 10)])
    second = Visit("v2", [_make_seq("s2", "A", 500, 10)])
    get_best_roll_per_visit(first, vis)
    get_best_roll_per_visit(second, vis)
    assert first.sequences[0].roll != second.sequences[0].roll


def test_scored_minutes_are_scheduled_plus_margin():
    vis = _RecordingVisibility()
    visit = Visit("v1", [_make_seq("s1", "A", 0, 10),
                         _make_seq("s2", "A", 60, 10)])
    result = get_best_roll_per_visit(visit, vis, growth_margin_minutes=5)
    call = vis.calls[0]
    expected = np.concatenate([np.arange(-5, 15), np.arange(55, 75)])
    np.testing.assert_array_equal(_minutes(call["times"]), expected)
    scheduled = result["A"]["scheduled"]
    np.testing.assert_array_equal(
        _minutes(call["times"])[scheduled],
        np.concatenate([np.arange(0, 10), np.arange(60, 70)]),
    )
    # 20 margin minutes, so a scheduled minute outweighs all of them.
    np.testing.assert_array_equal(call["weights"],
                                  np.where(scheduled, 21, 1))


def test_without_margin_every_minute_weighs_one():
    vis = _RecordingVisibility()
    visit = Visit("v1", [_make_seq("s1", "A", 0, 10)])
    get_best_roll_per_visit(visit, vis)
    call = vis.calls[0]
    np.testing.assert_array_equal(_minutes(call["times"]), np.arange(10))
    np.testing.assert_array_equal(call["weights"], np.ones(10, dtype=int))


def test_roll_step_and_floor_are_forwarded():
    vis = _RecordingVisibility()
    visit = Visit("v1", [_make_seq("s1", "A", 0, 10)])
    get_best_roll_per_visit(visit, vis, roll_step=2.0, min_power_frac=0.68)
    call = vis.calls[0]
    assert call["roll_step"] == 2.0 * u.deg
    assert call["min_power_frac"] == 0.68


def test_n_scheduled_visible_counts_scheduled_minutes_only():
    def only_margin(times):
        minutes = _minutes(times)
        return (minutes < 0) | (minutes >= 10)

    vis = _RecordingVisibility(visible=only_margin)
    visit = Visit("v1", [_make_seq("s1", "A", 0, 10)])
    result = get_best_roll_per_visit(visit, vis, growth_margin_minutes=5)
    assert result["A"]["n_visible"] == 10
    assert result["A"]["n_scheduled_visible"] == 0
    assert visit.sequences[0].roll == 30.0


def test_priority_0_model_only_when_every_observation_is_priority_0():
    nominal, strict = _RecordingVisibility(), _RecordingVisibility()
    all_zero = Visit("v1", [_make_seq("s1", "A", 0, 10, priority=0),
                            _make_seq("s2", "A", 60, 10, priority=0)])
    get_best_roll_per_visit(all_zero, nominal, priority_0_visibility=strict)
    assert (len(nominal.calls), len(strict.calls)) == (0, 1)

    nominal, strict = _RecordingVisibility(), _RecordingVisibility()
    mixed = Visit("v1", [_make_seq("s1", "A", 0, 10, priority=0),
                         _make_seq("s2", "A", 60, 10, priority=1)])
    get_best_roll_per_visit(mixed, nominal, priority_0_visibility=strict)
    assert (len(nominal.calls), len(strict.calls)) == (1, 0)
