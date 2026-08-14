"""Unit tests for the OverheadTiming class in shortschedule.overhead.

Tests cover:
- Default construction and that all four overhead times are populated
- Default values match the modelled MOC command sequence
- VISDA pre-overhead is larger than NIRDA pre-overhead (extra acquire offset)
- NIRDA and VISDA post-overheads are equal
- Units are seconds
- Explicit values override the derived defaults
- Partial overrides leave the other fields at their defaults
"""

# Third-party
from astropy import units as u

# First-party/Local
from shortschedule.overhead import OverheadTiming

# ---------------------------------------------------------------------------
# Expected defaults, computed straight from the modelled command sequence in
# OverheadTiming._update_derived so the maths lives in one place.
#
# Pre-observation:
#   GOTO_TARGET                          +2.0
#   MACRO_EXECUTE 65                     +20.0
#   PAYLOAD_READ                         +186.0
#   SADA_MODE 1 INDEX STOP               +50.0
#   -> NIRDA pre-overhead snapshot       = 258.0
#   PAYLOAD_ACQUIRE_INF_CAM_IMAGES       +2.0
#   -> VISDA pre-overhead snapshot       = 260.0
#
# Post-observation:
#   PAYLOAD_HALT_IMAGING_OR_COMMAND_SEQ  +2.0
#   SADA_MODE 1 INDEX AUTO_TRACK_QEST    +40.0
#   GOTO_TARGET Idle                     +18.0
#   PAYLOAD_READ close file              +42.0
#   -> NIRDA/VISDA post-overhead         = 102.0
# ---------------------------------------------------------------------------
_NIRDA_PRE = 258.0
_VISDA_PRE = 260.0
_POST = 102.0


# ---------------------------------------------------------------------------
# Default construction
# ---------------------------------------------------------------------------
class TestOverheadTimingDefaults:
    """OverheadTiming() with no arguments should populate all four times."""

    def test_default_construction_succeeds(self):
        """OverheadTiming with no arguments should not raise."""
        oh = OverheadTiming()
        assert oh is not None

    def test_all_fields_populated(self):
        """All four overhead times should be filled in (not None)."""
        oh = OverheadTiming()
        assert oh.nirda_pre_overhead_time is not None
        assert oh.visda_pre_overhead_time is not None
        assert oh.nirda_post_overhead_time is not None
        assert oh.visda_post_overhead_time is not None

    def test_nirda_pre_overhead_value(self):
        """NIRDA pre-overhead should match the modelled command sequence."""
        oh = OverheadTiming()
        assert (
            abs(oh.nirda_pre_overhead_time.to(u.s).value - _NIRDA_PRE) < 1e-12
        )

    def test_visda_pre_overhead_value(self):
        """VISDA pre-overhead should match the modelled command sequence."""
        oh = OverheadTiming()
        assert (
            abs(oh.visda_pre_overhead_time.to(u.s).value - _VISDA_PRE) < 1e-12
        )

    def test_post_overhead_value(self):
        """Post-overheads should match the modelled command sequence."""
        oh = OverheadTiming()
        assert abs(oh.nirda_post_overhead_time.to(u.s).value - _POST) < 1e-12
        assert abs(oh.visda_post_overhead_time.to(u.s).value - _POST) < 1e-12

    def test_units_are_seconds(self):
        """Every derived overhead should be convertible to seconds."""
        oh = OverheadTiming()
        for q in (
            oh.nirda_pre_overhead_time,
            oh.visda_pre_overhead_time,
            oh.nirda_post_overhead_time,
            oh.visda_post_overhead_time,
        ):
            _ = q.to(u.s)  # should not raise


# ---------------------------------------------------------------------------
# Relationships between the default overheads
# ---------------------------------------------------------------------------
class TestOverheadRelationships:
    """The derived overheads should hold the expected relative ordering."""

    def test_visda_pre_exceeds_nirda_pre(self):
        """VISDA pre-overhead carries an extra acquire offset over NIRDA."""
        oh = OverheadTiming()
        assert oh.visda_pre_overhead_time > oh.nirda_pre_overhead_time

    def test_post_overheads_equal(self):
        """NIRDA and VISDA share the same post-overhead."""
        oh = OverheadTiming()
        assert oh.nirda_post_overhead_time == oh.visda_post_overhead_time

    def test_all_overheads_positive(self):
        """All derived overheads should be strictly positive."""
        oh = OverheadTiming()
        assert oh.nirda_pre_overhead_time.to(u.s).value > 0
        assert oh.visda_pre_overhead_time.to(u.s).value > 0
        assert oh.nirda_post_overhead_time.to(u.s).value > 0
        assert oh.visda_post_overhead_time.to(u.s).value > 0


# ---------------------------------------------------------------------------
# Explicit overrides
# ---------------------------------------------------------------------------
class TestOverheadOverrides:
    """Explicit Quantity values should override the derived defaults."""

    def test_all_overrides_respected(self):
        """Supplying all four values should leave them untouched."""
        oh = OverheadTiming(
            nirda_pre_overhead_time=1.0 * u.s,
            visda_pre_overhead_time=2.0 * u.s,
            nirda_post_overhead_time=3.0 * u.s,
            visda_post_overhead_time=4.0 * u.s,
        )
        assert oh.nirda_pre_overhead_time.to(u.s).value == 1.0
        assert oh.visda_pre_overhead_time.to(u.s).value == 2.0
        assert oh.nirda_post_overhead_time.to(u.s).value == 3.0
        assert oh.visda_post_overhead_time.to(u.s).value == 4.0

    def test_partial_override_fills_remaining_defaults(self):
        """An overridden field stays; unset fields fall back to defaults."""
        oh = OverheadTiming(nirda_pre_overhead_time=5.0 * u.s)
        assert oh.nirda_pre_overhead_time.to(u.s).value == 5.0
        # The others should still be derived.
        assert (
            abs(oh.visda_pre_overhead_time.to(u.s).value - _VISDA_PRE) < 1e-12
        )
        assert abs(oh.nirda_post_overhead_time.to(u.s).value - _POST) < 1e-12
        assert abs(oh.visda_post_overhead_time.to(u.s).value - _POST) < 1e-12

    def test_zero_override_is_kept(self):
        """A zero override should be honoured, not replaced by the default."""
        oh = OverheadTiming(nirda_pre_overhead_time=0.0 * u.s)
        assert oh.nirda_pre_overhead_time.to(u.s).value == 0.0
