"""Keepouts passed to ScheduleProcessor must be what Visibility applies.

These tests guard against silent drift between this package and
``pandoravisibility``: a keepout default restated here, or a constraint
forwarded on one code path but not another, changes the science quietly.
They deliberately use the real ``Visibility`` rather than a mock, because
a mock cannot catch a divergence from the library's own defaults.
"""

# Third-party
import pytest
from pandoravisibility import Visibility

# First-party/Local
from shortschedule.scheduler import ScheduleProcessor

# A real TLE is needed: Visibility parses it with sgp4 at construction.
TLE1 = "1 67395U 80229J   26196.69732639  .00000000  00000-0  37770-3 0    00"
TLE2 = "2 67395  97.8056 194.9117 0006480  50.2285  39.6294 14.88117629    09"

# Every keepout ScheduleProcessor forwards, as (argument name, degrees).
KEEPOUTS = [
    "moon_min",
    "sun_min",
    "earthlimb_min",
    "earthlimb_day_min",
    "earthlimb_night_min",
    "mars_min",
    "jupiter_min",
    "st_sun_min",
    "st_moon_min",
    "st_earthlimb_min",
    "st1_earthlimb_min",
    "st2_earthlimb_min",
]


@pytest.mark.parametrize("keepout", KEEPOUTS)
def test_unset_keepout_matches_the_library_default(keepout):
    """An unset keepout leaves pandoravisibility's own default in place.

    ScheduleProcessor must hold no opinion of its own here. Restating a
    default lets the two drift: ``moon_min`` once defaulted to 20 deg here
    against the library's 25 deg, quietly loosening the keepout for every
    caller that did not pass one.
    """
    library = Visibility(TLE1, TLE2)
    scheduler = ScheduleProcessor(TLE1, TLE2).visibility

    assert getattr(scheduler, keepout) == getattr(library, keepout)


@pytest.mark.parametrize("keepout", KEEPOUTS)
def test_explicit_keepout_is_applied_verbatim(keepout):
    """A keepout passed to __init__ reaches Visibility unchanged."""
    scheduler = ScheduleProcessor(TLE1, TLE2, **{keepout: 37.5}).visibility

    applied = getattr(scheduler, keepout)
    assert applied is not None
    assert applied.to_value("deg") == pytest.approx(37.5)


@pytest.mark.parametrize(
    "star_tracker_limits",
    [
        {},
        {"st_sun_min": 0.0},
        {"st_sun_min": 50.0},
        {"st_moon_min": 20.0},
        {"st_earthlimb_min": 30.0},
        {"st1_earthlimb_min": 30.0},
        {"st2_earthlimb_min": 30.0},
        {"st_sun_min": 0.0, "st_moon_min": 0.0, "st_earthlimb_min": 0.0},
        {"st_sun_min": 50.0, "st_moon_min": 20.0, "st_earthlimb_min": 30.0},
    ],
)
def test_roll_sweep_gate_agrees_with_the_library(star_tracker_limits):
    """The roll sweep runs exactly when tracker constraints are active.

    ``_roll_sweep_enabled`` is derived from the constructor arguments so
    that a duck-typed visibility object still works, which means it can
    drift from what the library actually applies. This pins the two
    together across the configurations the scheduler can produce. A limit
    of zero disables that keepout, so passing one is not a reason to sweep.
    """
    processor = ScheduleProcessor(TLE1, TLE2, **star_tracker_limits)

    assert processor._roll_sweep_enabled == bool(
        processor.visibility._st_constraint_active
    )
