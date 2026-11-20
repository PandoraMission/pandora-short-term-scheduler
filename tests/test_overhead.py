"""Unit tests for the OverheadTiming class in shortschedule.overhead.

Tests cover:
- Default construction and that all six times are populated
- Default values match the modelled MOC command sequence: calendar Start is
  the NIRDA start, VISDA follows 2 s later, both end by payload control off
  (the calendar Stop), and the spacecraft is settled on the target 50 s
  before Start and stays on it until 64 s after Stop
- The module constants hold the MOC timeline in order
- Units are seconds
- Explicit values override the derived defaults
- Partial overrides leave the other fields at their defaults
"""

# Third-party
from astropy import units as u

# First-party/Local
from shortschedule import overhead as oh_module
from shortschedule.overhead import OverheadTiming

# ---------------------------------------------------------------------------
# Expected defaults. Written out rather than read from the module, so a
# changed constant fails here and gets looked at.
#
# Pre-observation (seconds from the calendar Start):
#   GOTO_TARGET                          -258
#   MACRO_EXECUTE 65                     -256
#   PAYLOAD_READ                         -236
#   SADA_MODE 1 INDEX STOP               -50   -> settled 50 s before Start
#   PAYLOAD_ACQUIRE_INF_CAM_IMAGES       0     -> NIRDA pre-overhead 0 s
#   PAYLOAD_ACQUIRE_VIS_CAM_SCIENCE_DATA +2    -> VISDA pre-overhead 2 s
#   PAYLOAD_CONTROL ENABLE               +16
#
# Post-observation (seconds after the calendar Stop):
#   PAYLOAD_CONTROL DISABLE              0     -> NIRDA/VISDA post 0 s
#   MACRO_EXECUTE 65                     +2
#   PAYLOAD_HALT_IMAGING_OR_COMMAND_SEQ  +22
#   SADA_MODE 1 INDEX AUTO_TRACK_QEST    +24
#   GOTO_TARGET Idle                     +64   -> on target 64 s after Stop
#   PAYLOAD_READ (empty)                 +82
#   at idle                              +124
# ---------------------------------------------------------------------------
_NIRDA_PRE = 0.0
_VISDA_PRE = 2.0
_POST = 0.0
_SETTLED_BEFORE = 50.0
_ON_TARGET_AFTER = 64.0


def _sec(q):
    return q.to(u.s).value


# ---------------------------------------------------------------------------
# Default construction
# ---------------------------------------------------------------------------
class TestOverheadTimingDefaults:
    """OverheadTiming() with no arguments should populate every time."""

    def test_default_construction_succeeds(self):
        """OverheadTiming with no arguments should not raise."""
        oh = OverheadTiming()
        assert oh is not None

    def test_all_fields_populated(self):
        """Every time should be filled in (not None)."""
        oh = OverheadTiming()
        assert oh.nirda_pre_overhead_time is not None
        assert oh.visda_pre_overhead_time is not None
        assert oh.nirda_post_overhead_time is not None
        assert oh.visda_post_overhead_time is not None
        assert oh.settled_before_start_time is not None
        assert oh.on_target_after_stop_time is not None

    def test_nirda_starts_at_the_calendar_start(self):
        """The MOC starts NIRDA at the calendar start."""
        assert (
            abs(_sec(OverheadTiming().nirda_pre_overhead_time) - _NIRDA_PRE)
            < 1e-12
        )

    def test_visda_starts_2_s_after(self):
        """The VISDA acquisition command follows the NIRDA one by 2 s."""
        assert (
            abs(_sec(OverheadTiming().visda_pre_overhead_time) - _VISDA_PRE)
            < 1e-12
        )

    def test_both_detectors_end_by_payload_control_off(self):
        """Integrations and frames end by the calendar stop."""
        oh = OverheadTiming()
        assert abs(_sec(oh.nirda_post_overhead_time) - _POST) < 1e-12
        assert abs(_sec(oh.visda_post_overhead_time) - _POST) < 1e-12

    def test_on_target_times(self):
        """Settled 50 s before the start, on target 64 s after the stop."""
        oh = OverheadTiming()
        assert (
            abs(_sec(oh.settled_before_start_time) - _SETTLED_BEFORE) < 1e-12
        )
        assert (
            abs(_sec(oh.on_target_after_stop_time) - _ON_TARGET_AFTER) < 1e-12
        )

    def test_units_are_seconds(self):
        """Every derived time should be convertible to seconds."""
        oh = OverheadTiming()
        for q in (
            oh.nirda_pre_overhead_time,
            oh.visda_pre_overhead_time,
            oh.nirda_post_overhead_time,
            oh.visda_post_overhead_time,
            oh.settled_before_start_time,
            oh.on_target_after_stop_time,
        ):
            _ = q.to(u.s)  # should not raise


# ---------------------------------------------------------------------------
# Relationships, and the module constants
# ---------------------------------------------------------------------------
class TestOverheadRelationships:
    """The derived times should hold the expected relative ordering."""

    def test_visda_pre_exceeds_nirda_pre(self):
        """VISDA starts after NIRDA."""
        oh = OverheadTiming()
        assert oh.visda_pre_overhead_time > oh.nirda_pre_overhead_time

    def test_post_overheads_equal(self):
        """NIRDA and VISDA both end by payload control off."""
        oh = OverheadTiming()
        assert oh.nirda_post_overhead_time == oh.visda_post_overhead_time

    def test_no_overhead_is_negative(self):
        """No detector starts before the calendar start or runs past its stop."""
        oh = OverheadTiming()
        for q in (
            oh.nirda_pre_overhead_time,
            oh.visda_pre_overhead_time,
            oh.nirda_post_overhead_time,
            oh.visda_post_overhead_time,
        ):
            assert _sec(q) >= 0

    def test_pre_sequence_is_in_order(self):
        m = oh_module
        assert (
            m.MOC_SLEW_TO_TARGET_S
            < m.MOC_MACRO65_PRE_S
            < m.MOC_OPEN_SCIENCE_FILE_S
            < m.MOC_SETTLED_S
            < m.MOC_NIRDA_START_S
            < m.MOC_VISDA_START_S
            < m.MOC_PAYLOAD_CONTROL_ON_S
        )
        # Calendar Start is the NIRDA start.
        assert m.MOC_NIRDA_START_S == 0.0

    def test_post_sequence_is_in_order(self):
        m = oh_module
        assert (
            m.MOC_PAYLOAD_CONTROL_OFF_S
            < m.MOC_MACRO65_POST_S
            < m.MOC_HALT_IMAGING_S
            < m.MOC_SADA_TRACK_S
            < m.MOC_SLEW_TO_IDLE_S
            < m.MOC_CLOSE_SCIENCE_FILE_S
            < m.MOC_AT_IDLE_S
        )
        # Calendar Stop is payload control off, and the detectors end there.
        assert m.MOC_PAYLOAD_CONTROL_OFF_S == 0.0
        assert m.DETECTORS_END_S == m.MOC_PAYLOAD_CONTROL_OFF_S

    def test_the_slew_is_258_s_before_the_start(self):
        assert oh_module.MOC_SLEW_TO_TARGET_S == -258.0

    def test_other_overhead_constants(self):
        assert oh_module.VITL_SETTLING_TIME == 60.0 * u.s
        assert oh_module.ADDITIONAL_OVERHEAD_TIME == 0.0 * u.s


# ---------------------------------------------------------------------------
# Explicit overrides
# ---------------------------------------------------------------------------
class TestOverheadOverrides:
    """Explicit Quantity values should override the derived defaults."""

    def test_all_overrides_respected(self):
        """Supplying every value should leave them untouched."""
        oh = OverheadTiming(
            nirda_pre_overhead_time=1.0 * u.s,
            visda_pre_overhead_time=2.0 * u.s,
            nirda_post_overhead_time=3.0 * u.s,
            visda_post_overhead_time=4.0 * u.s,
            settled_before_start_time=5.0 * u.s,
            on_target_after_stop_time=6.0 * u.s,
        )
        assert _sec(oh.nirda_pre_overhead_time) == 1.0
        assert _sec(oh.visda_pre_overhead_time) == 2.0
        assert _sec(oh.nirda_post_overhead_time) == 3.0
        assert _sec(oh.visda_post_overhead_time) == 4.0
        assert _sec(oh.settled_before_start_time) == 5.0
        assert _sec(oh.on_target_after_stop_time) == 6.0

    def test_partial_override_fills_remaining_defaults(self):
        """An overridden field stays; unset fields fall back to defaults."""
        oh = OverheadTiming(nirda_pre_overhead_time=5.0 * u.s)
        assert _sec(oh.nirda_pre_overhead_time) == 5.0
        # The others should still be derived.
        assert abs(_sec(oh.visda_pre_overhead_time) - _VISDA_PRE) < 1e-12
        assert abs(_sec(oh.nirda_post_overhead_time) - _POST) < 1e-12
        assert abs(_sec(oh.visda_post_overhead_time) - _POST) < 1e-12
        assert (
            abs(_sec(oh.settled_before_start_time) - _SETTLED_BEFORE) < 1e-12
        )
        assert (
            abs(_sec(oh.on_target_after_stop_time) - _ON_TARGET_AFTER) < 1e-12
        )

    def test_zero_override_is_kept(self):
        """A zero override should be honoured, not replaced by the default."""
        oh = OverheadTiming(
            visda_pre_overhead_time=0.0 * u.s,
            settled_before_start_time=0.0 * u.s,
        )
        assert _sec(oh.visda_pre_overhead_time) == 0.0
        assert _sec(oh.settled_before_start_time) == 0.0
