"""Tests for add_high_pri_observations.

After the final merge, the window's priority 1/2 targets get extra
observations in idle time where they are visible. Each joins an existing
visit, so the calendar stays in time order, and flies its target's roll in
that visit (or the roll of its visit with the most minutes of it).
Additions are priority 0, or 1 when they overlap a transit widened by a
buffer. A target with too few additions gets none, and its time goes to
the others.
"""

# Standard library
import xml.etree.ElementTree as ET

# Third-party
import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

# First-party/Local
from shortschedule.models import ObservationSequence, ScienceCalendar, Visit
from shortschedule.scheduler import ScheduleProcessor
from tests.doubles import answers_visibility
from tests.test_occultation_keepout import TLE1, TLE2

T0 = Time("2026-01-01T00:00:00", scale="utc")
WINDOW_MIN = 300


class _ScriptedVis:
    """Visible where ``visible_by_ra[ra](minute)`` says, minute from T0."""

    def __init__(self, visible_by_ra):
        self.visible_by_ra = visible_by_ra
        self.asked = []

    @answers_visibility
    def get_visibility(self, coord, times, roll):
        ra = int(round(coord.ra.deg))
        self.asked.append(
            (ra, None if roll is None else float(roll.to_value(u.deg)))
        )
        minutes = np.rint((times - T0).to_value(u.min)).astype(int)
        return self.visible_by_ra[ra](minutes)


def _seq(sid, target, priority, start_min, stop_min, ra, roll):
    section = ET.Element("AcquireVisCamScienceData")
    ET.SubElement(section, "TargetID").text = target
    return ObservationSequence(
        id=sid,
        target=target,
        priority=priority,
        start_time=T0 + start_min * u.min,
        stop_time=T0 + stop_min * u.min,
        ra=ra,
        dec=-20.0,
        payload_params={"AcquireVisCamScienceData": section},
        roll=roll,
    )


def _span(lo, hi):
    """Visible at every sample minute from ``lo`` to ``hi`` inclusive."""
    return lambda m: (m >= lo) & (m <= hi)


def _processor(visible_by_ra, **kwargs):
    # The 1 s edge pass and the pointing pad are off here so the slot rules
    # can be read in whole minutes; test_additions_get_the_1_s_edge_pass and
    # test_pointing_pad.py cover them.
    settings = dict(
        add_high_pri_observations=True,
        high_priority_adds_min_observations=1,
        refine_edges_to_second=False,
        pointing_pad_before_start=0 * u.s,
        pointing_pad_after_stop=0 * u.s,
    )
    settings.update(kwargs)
    proc = ScheduleProcessor(TLE1, TLE2, **settings)
    proc.printed = []
    proc._print = lambda message, *a, **kw: proc.printed.append(message)
    proc._initialize_gap_report()
    proc.window_start = T0
    proc.window_end = T0 + WINDOW_MIN * u.min
    proc.high_priority_adds_visibility = _ScriptedVis(visible_by_ra)
    return proc


def _added(calendar):
    """Visit ID -> the additions it received, in time order."""
    added = {}
    for visit in calendar.visits:
        seqs = [s for s in visit.sequences if getattr(s, "is_added", False)]
        if seqs:
            added[visit.id] = seqs
    return added


def _spans(sequences):
    return [
        (
            int(np.rint((s.start_time - T0).to_value(u.min))),
            int(np.rint((s.stop_time - T0).to_value(u.min))),
        )
        for s in sequences
    ]


def _assert_sequential(calendar):
    """Each visit in time order, and no two visits interleave."""
    for visit in calendar.visits:
        starts = [s.start_time for s in visit.sequences]
        assert starts == sorted(starts), visit.id
    for first, second in zip(calendar.visits, calendar.visits[1:]):
        assert max(s.stop_time for s in first.sequences) <= min(
            s.start_time for s in second.sequences
        ), (first.id, second.id)


def test_off_by_default():
    assert ScheduleProcessor(TLE1, TLE2).add_high_pri_observations is False


def test_idle_visible_time_is_filled_into_the_existing_visits():
    """A, priority 1/2 in two visits, gets the idle time around everything
    else, in pieces of at most 90 min. The gap between the visits joins
    0001 (a tie, so the visit before) at A's roll there; the time after
    0002 joins 0002 at A's roll there. No visit is created. The payload
    comes from the visit holding more of A (0002). B, only priority 0, is
    not a candidate."""
    proc = _processor({10: _span(0, WINDOW_MIN), 50: _span(0, WINDOW_MIN)})
    calendar = ScienceCalendar(
        metadata={},
        visits=[
            Visit(
                id="0001",
                sequences=[
                    _seq("001", "A", 1, 0, 20, ra=10.0, roll=30.0),
                    _seq("002", "B", 0, 20, 40, ra=50.0, roll=5.0),
                ],
            ),
            Visit(
                id="0002",
                sequences=[_seq("001", "A", 2, 100, 130, ra=10.0, roll=50.0)],
            ),
        ],
    )
    calendar.visits[0].sequences[0].is_main = True
    calendar.visits[1].sequences[0].is_main = True
    template = calendar.visits[1].sequences[0]

    proc._add_high_priority_observations(calendar)

    assert [visit.id for visit in calendar.visits] == ["0001", "0002"]
    added = _added(calendar)
    assert {vid: _spans(seqs) for vid, seqs in added.items()} == {
        "0001": [(40, 100)],
        "0002": [(130, 220), (220, 300)],
    }
    assert [s.roll for s in added["0001"]] == [30.0]
    assert [s.roll for s in added["0002"]] == [50.0, 50.0]
    for seq in added["0001"] + added["0002"]:
        assert seq.target == "A"
        assert seq.priority == 0
        assert seq.is_added and seq.is_main
        # A copy of the template's payload, not the same element.
        assert (
            seq.payload_params["AcquireVisCamScienceData"]
            is not template.payload_params["AcquireVisCamScienceData"]
        )
    assert proc.high_priority_adds_visibility.asked == [
        (10, 30.0),
        (10, 50.0),
    ]
    _assert_sequential(calendar)
    summary = proc.gap_report["processing_summary"]
    assert summary["high_pri_added_visits"] == 2
    assert summary["high_pri_added_observations"] == 3
    assert summary["high_pri_added_minutes"] == 230


def _gap_calendar():
    """A in 0001 at roll 0; B, and A as priority 0 at roll 45, in 0002."""
    calendar = ScienceCalendar(
        metadata={},
        visits=[
            Visit(
                id="0001", sequences=[_seq("001", "A", 1, 0, 20, 10.0, 0.0)]
            ),
            Visit(
                id="0002",
                sequences=[
                    _seq("001", "B", 1, 200, 220, 70.0, 0.0),
                    _seq("002", "A", 0, 220, 230, 10.0, 45.0),
                ],
            ),
        ],
    )
    calendar.visits[0].sequences[0].is_main = True
    calendar.visits[1].sequences[0].is_main = True
    return calendar


def test_a_gap_on_a_tie_joins_the_visit_before():
    """30-60 is closer to 0001, 150-180 closer to 0002: one each, so the
    whole gap joins 0001, at A's roll and main flag there."""
    proc = _processor(
        {
            10: lambda m: _span(30, 60)(m) | _span(150, 180)(m),
            70: _span(-9, -9),
        }
    )
    calendar = _gap_calendar()

    proc._add_high_priority_observations(calendar)

    added = _added(calendar)
    assert list(added) == ["0001"]
    assert _spans(added["0001"]) == [(30, 60), (150, 180)]
    assert all(s.roll == 0.0 and s.is_main for s in added["0001"])
    _assert_sequential(calendar)


def test_a_gap_joins_the_visit_most_of_its_additions_are_closer_to():
    """A third addition at 185-195, closer to 0002, tips the gap: all three
    join 0002 at A's roll there (45), not main there. Every gap minute had
    to be visible at both rolls, since either visit could take it."""
    proc = _processor(
        {
            10: lambda m: _span(30, 60)(m)
            | _span(150, 180)(m)
            | _span(185, 195)(m),
            70: _span(-9, -9),
        }
    )
    calendar = _gap_calendar()

    proc._add_high_priority_observations(calendar)

    added = _added(calendar)
    assert list(added) == ["0002"]
    assert _spans(added["0002"]) == [(30, 60), (150, 180), (185, 195)]
    assert all(s.roll == 45.0 and not s.is_main for s in added["0002"])
    assert {roll for ra, roll in proc.high_priority_adds_visibility.asked} == {
        0.0,
        45.0,
    }
    # 0002 now starts at 30, after 0001 ends at 20.
    _assert_sequential(calendar)
    assert [s.target for s in calendar.visits[1].sequences] == [
        "A",
        "A",
        "A",
        "B",
        "A",
    ]


@pytest.mark.parametrize("dark_roll", [0.0, 45.0])
def test_a_gap_minute_must_be_visible_at_both_neighbors_rolls(dark_roll):
    """Where A is visible at only one of its two rolls (0 in 0001, 45 in
    0002), a gap minute is unusable: the gap might join either visit."""

    class _ByRoll(_ScriptedVis):
        @answers_visibility
        def get_visibility(self, coord, times, roll):
            minutes = np.rint((times - T0).to_value(u.min)).astype(int)
            if int(round(coord.ra.deg)) != 10 or (
                roll is not None and float(roll.to_value(u.deg)) == dark_roll
            ):
                return np.zeros(minutes.shape, dtype=bool)
            return _span(30, 60)(minutes)

    proc = _processor({})
    proc.high_priority_adds_visibility = _ByRoll({})
    calendar = _gap_calendar()

    proc._add_high_priority_observations(calendar)

    assert _added(calendar) == {}


def test_before_the_first_observation_joins_the_first_visit():
    proc = _processor({10: _span(0, 40)})
    calendar = ScienceCalendar(
        metadata={},
        visits=[
            Visit(
                id="0001", sequences=[_seq("001", "A", 1, 100, 120, 10.0, 0.0)]
            )
        ],
    )

    proc._add_high_priority_observations(calendar)

    assert _spans(_added(calendar)["0001"]) == [(0, 40)]
    assert calendar.visits[0].sequences[0].is_added


def test_a_slot_needs_the_minimum_and_both_ends_of_every_minute():
    """Visible samples 40-48 make an 8 min slot (too short); samples
    150-170 make 150-170, but 150-169 alone would only make 150-169."""
    visible = {10: lambda m: _span(40, 48)(m) | _span(150, 170)(m)}
    proc = _processor(visible)
    calendar = ScienceCalendar(
        metadata={},
        visits=[
            Visit(id="0001", sequences=[_seq("001", "A", 1, 0, 20, 10.0, 0.0)])
        ],
    )
    proc._add_high_priority_observations(calendar)
    assert _spans(_added(calendar)["0001"]) == [(150, 170)]

    proc = _processor({10: _span(150, 169)})
    calendar = ScienceCalendar(
        metadata={},
        visits=[
            Visit(id="0001", sequences=[_seq("001", "A", 1, 0, 20, 10.0, 0.0)])
        ],
    )
    proc._add_high_priority_observations(calendar)
    assert _spans(_added(calendar)["0001"]) == [(150, 169)]


def test_never_overlaps_an_observation_ending_mid_minute():
    proc = _processor({10: _span(0, WINDOW_MIN)})
    observation = _seq("001", "A", 1, 0, 20, 10.0, 0.0)
    observation.stop_time = T0 + 20.5 * u.min
    calendar = ScienceCalendar(
        metadata={}, visits=[Visit(id="0001", sequences=[observation])]
    )
    proc._add_high_priority_observations(calendar)
    assert _spans(_added(calendar)["0001"])[0][0] == 21


class _SecondsVis:
    """Visible where ``visible(second)`` says, second from T0."""

    def __init__(self, visible):
        self.visible = visible

    @answers_visibility
    def get_visibility(self, coord, times, roll):
        return self.visible(np.rint((times - T0).to_value(u.s)))


def test_additions_get_the_1_s_edge_pass():
    """A is visible from 20 s before minute 150 to 26 s after minute 170,
    and through its own observation's stop at minute 20 plus 30 s. The
    addition at 150-170 is pushed out to those seconds; the original
    observation, refined already, is left alone."""
    lo, hi = 150 * 60 - 20, 170 * 60 + 26
    proc = _processor({}, refine_edges_to_second=True)
    proc.high_priority_adds_visibility = _SecondsVis(
        lambda s: ((s >= lo) & (s < hi)) | (s < 20 * 60 + 30)
    )
    proc.visibility = proc.high_priority_adds_visibility
    original = _seq("001", "A", 1, 0, 20, 10.0, 0.0)
    calendar = ScienceCalendar(
        metadata={}, visits=[Visit(id="0001", sequences=[original])]
    )
    # What the main edge pass already counted is kept and added to.
    proc.gap_report["processing_summary"]["edge_seconds_grown"] = 100

    proc._add_high_priority_observations(calendar)

    (addition,) = _added(calendar)["0001"]
    assert (addition.start_time - T0).sec == pytest.approx(150 * 60 - 20)
    assert (addition.stop_time - T0).sec == pytest.approx(170 * 60 + 26)
    assert (original.stop_time - T0).sec == pytest.approx(20 * 60)
    summary = proc.gap_report["processing_summary"]
    assert summary["edge_seconds_grown"] == 146
    assert summary["high_pri_added_minutes"] == pytest.approx(20.8)


def test_a_thin_target_gives_its_time_to_the_others():
    """A wins 150-181 first but gets nothing else, under the minimum of 2:
    it is dropped with a warning, and C takes that time on the redo."""
    visible = {
        10: _span(150, 181),
        70: lambda m: _span(150, 180)(m)
        | _span(250, 262)(m)
        | _span(270, 282)(m),
    }
    proc = _processor(visible, high_priority_adds_min_observations=2)
    calendar = ScienceCalendar(
        metadata={},
        visits=[
            Visit(
                id="0001", sequences=[_seq("001", "A", 1, 0, 20, 10.0, 0.0)]
            ),
            Visit(
                id="0002", sequences=[_seq("001", "C", 1, 20, 40, 70.0, 0.0)]
            ),
        ],
    )

    proc._add_high_priority_observations(calendar)

    added = _added(calendar)
    assert list(added) == ["0002"]
    assert {s.target for s in added["0002"]} == {"C"}
    assert _spans(added["0002"]) == [(150, 180), (250, 262), (270, 282)]
    warnings = [m for m in proc.printed if m.startswith("WARNING")]
    assert any("A would get only 1" in m for m in warnings)


def _two_targets(name="NGC 1433 SN"):
    return ScienceCalendar(
        metadata={},
        visits=[
            Visit(
                id="0001", sequences=[_seq("001", "A", 1, 0, 20, 10.0, 0.0)]
            ),
            Visit(
                id="0002",
                sequences=[_seq("001", name, 2, 20, 40, 70.0, 0.0)],
            ),
        ],
    )


@pytest.mark.parametrize(
    "given, target",
    [("NGC_1433_SN", "NGC 1433 SN"), ("NGC 1433 SN", "NGC_1433_SN")],
)
def test_an_excluded_target_gets_no_additions(given, target):
    """Its time goes to the other target. Either spelling matches."""
    visible = {10: _span(0, WINDOW_MIN), 70: _span(0, WINDOW_MIN)}
    proc = _processor(visible, high_priority_adds_exclude=[given])
    calendar = _two_targets(target)

    proc._add_high_priority_observations(calendar)

    added = _added(calendar)
    assert list(added) == ["0002"]
    assert {s.target for s in added["0002"]} == {"A"}
    assert _spans(added["0002"]) == [(40, 130), (130, 220), (220, 300)]
    assert [ra for ra, _ in proc.high_priority_adds_visibility.asked] == [10]
    assert any(
        f"{target} excluded" in m and not m.startswith("WARNING")
        for m in proc.printed
    )


def test_an_exclude_naming_no_candidate_is_a_warning():
    visible = {10: _span(0, WINDOW_MIN), 70: _span(0, WINDOW_MIN)}
    proc = _processor(visible, high_priority_adds_exclude="Titan")
    calendar = _two_targets()

    proc._add_high_priority_observations(calendar)

    assert proc.high_priority_adds_exclude == ["Titan"]
    assert {s.target for s in _added(calendar)["0002"]} == {
        "A",
        "NGC 1433 SN",
    }
    warnings = [m for m in proc.printed if m.startswith("WARNING")]
    assert len(warnings) == 2  # Titan, and the no-target_list_dir warning
    assert any("Titan" in m for m in warnings)


def test_a_padded_transit_goes_first_and_is_priority_1(monkeypatch):
    """T1-T4 at 200-210 min, padded 40 min, covers minutes 160-249. The
    transit pieces are placed first, ahead of longer stretches and of C,
    and only they are priority 1."""
    monkeypatch.setattr(
        "shortschedule.scheduler.transit_epoch", lambda eph, near: 0
    )
    monkeypatch.setattr(
        "shortschedule.scheduler.transit_core",
        lambda eph, epoch, frac: (T0 + 200 * u.min, T0 + 210 * u.min),
    )
    proc = _processor(
        {10: _span(0, WINDOW_MIN), 70: _span(0, WINDOW_MIN)},
        high_priority_adds_max_duration=30,
    )
    proc.transit_ephemerides = {"a": {}}
    calendar = ScienceCalendar(
        metadata={},
        visits=[
            Visit(
                id="0001", sequences=[_seq("001", "A", 1, 0, 20, 10.0, 0.0)]
            ),
            Visit(
                id="0002", sequences=[_seq("001", "C", 1, 20, 40, 70.0, 0.0)]
            ),
        ],
    )

    proc._add_high_priority_observations(calendar)

    added = [s for seqs in _added(calendar).values() for s in seqs]
    by_target = {
        target: [s for s in added if s.target == target]
        for target in ("A", "C")
    }
    priorities = {
        span: seq.priority
        for span, seq in zip(_spans(by_target["A"]), by_target["A"])
    }
    assert {span for span, p in priorities.items() if p == 1} == {
        (160, 190),
        (190, 220),
        (220, 250),
    }
    assert all(seq.priority == 0 for seq in by_target["C"])
    every = sorted(_spans(added))
    assert all(a[1] <= b[0] for a, b in zip(every, every[1:]))
    _assert_sequential(calendar)
    assert any(
        m.startswith("WARNING") and "C has no transit ephemeris" in m
        for m in proc.printed
    )


def test_without_ephemerides_everything_is_priority_0_with_a_warning():
    proc = _processor({10: _span(0, WINDOW_MIN)})
    calendar = ScienceCalendar(
        metadata={},
        visits=[
            Visit(id="0001", sequences=[_seq("001", "A", 1, 0, 20, 10.0, 0.0)])
        ],
    )
    proc._add_high_priority_observations(calendar)
    assert all(s.priority == 0 for s in _added(calendar)["0001"])
    assert any("no target_list_dir" in m for m in proc.printed)


def test_additions_are_judged_by_their_own_model():
    proc = ScheduleProcessor(
        TLE1,
        TLE2,
        occultation_earthlimb_min=69,
        high_priority_adds_earthlimb_min=50,
    )
    seq = _seq("001", "A", 0, 0, 20, 10.0, 0.0)
    assert proc._visibility_for(seq) is proc.occultation_visibility
    seq.is_added = True
    assert proc._visibility_for(seq) is proc.high_priority_adds_visibility
    assert proc.high_priority_adds_visibility.earthlimb_min.to_value(
        u.deg
    ) == pytest.approx(50)
    assert seq.copy().is_added is True


def test_unset_limb_uses_the_standard_model():
    proc = ScheduleProcessor(TLE1, TLE2)
    assert proc.high_priority_adds_visibility is proc.visibility


def test_header():
    off = ScheduleProcessor(TLE1, TLE2)._settings_for_header()
    assert off["Add_High_Pri_Observations"] == "False"
    assert "High_Priority_Adds_Min_Duration_Min" not in off

    on = ScheduleProcessor(
        TLE1,
        TLE2,
        add_high_pri_observations=True,
        high_priority_adds_earthlimb_min=55,
    )._settings_for_header()
    assert on["Add_High_Pri_Observations"] == "True"
    assert on["High_Priority_Adds_Earthlimb_Min_Deg"] == "55"
    assert on["High_Priority_Adds_Min_Duration_Min"] == "10"
    assert on["High_Priority_Adds_Max_Duration_Min"] == "90"
    assert on["High_Priority_Adds_Min_Observations"] == "5"
    assert on["High_Priority_Adds_Transit_Buffer_Min"] == "40"
    assert "High_Priority_Adds_Exclude" not in on

    excluded = ScheduleProcessor(
        TLE1,
        TLE2,
        add_high_pri_observations=True,
        high_priority_adds_exclude=["Titan", "NGC_1433_SN"],
    )._settings_for_header()
    assert excluded["High_Priority_Adds_Exclude"] == "Titan,NGC_1433_SN"
