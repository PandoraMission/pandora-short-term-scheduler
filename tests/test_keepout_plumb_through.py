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


def test_unset_dynamic_earthlimb_matches_the_library_default():
    """``use_dynamic_earthlimb`` defers to the library like every keepout.

    It is a bool, so it survives the "drop the Nones" filter that leaves
    the rest to ``pandoravisibility``. Defaulting it to ``False`` here
    pinned the wedge off for every caller that never mentioned it, right
    past the library turning it on.
    """
    library = Visibility(TLE1, TLE2)
    scheduler = ScheduleProcessor(TLE1, TLE2).visibility

    assert scheduler.use_dynamic_earthlimb == library.use_dynamic_earthlimb


@pytest.mark.parametrize("wedge", [True, False])
def test_explicit_dynamic_earthlimb_is_applied_verbatim(wedge):
    """Passing it reaches Visibility unchanged, either way."""
    scheduler = ScheduleProcessor(
        TLE1, TLE2, use_dynamic_earthlimb=wedge
    ).visibility

    assert scheduler.use_dynamic_earthlimb is wedge


def test_priority_0_limb_is_not_overridden_by_library_day_night():
    """The priority-0 limb stays the flat angle the caller asked for.

    The day/night pair has to be sent as an explicit ``None``. Dropping
    the keys instead lets Visibility fall back to its own defaults, which
    are real angles since v1.3.0, and those would then decide the
    threshold rather than ``priority_0_earthlimb_min``.
    """
    processor = ScheduleProcessor(TLE1, TLE2, priority_0_earthlimb_min=54)
    strict = processor.priority_0_visibility

    assert strict.use_dynamic_earthlimb is False
    assert strict.earthlimb_day_min is None
    assert strict.earthlimb_night_min is None
    assert strict.earthlimb_min.to_value("deg") == pytest.approx(54.0)
