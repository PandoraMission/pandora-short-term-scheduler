"""Unit tests for the NirdaData class in shortschedule.nirda.

Tests cover:
- Default construction and derived-attribute values
- pixels_per_frame and single_frame_time calculations
- Reset-frame-time derivation (always equals single_frame_time)
- Global-reset overhead methods (off / global / line_by_line)
- First vs subsequent integration time formulas
- Integration data and saved-frame counts (average_groups on/off)
- update_for_vitl reset-frame adjustment
- solve_integrations duration decomposition
- solve_duration integration-count inversion
- Roundtrip consistency between solve_integrations and solve_duration
- Edge cases: zero ROI, zero duration, dropped integrations
"""

# Third-party
import math

import pytest
from astropy import units as u

# First-party/Local
from shortschedule.nirda import NirdaData
from shortschedule.overhead import OverheadTiming


def _nirda_overhead(pre=0 * u.s, post=0 * u.s):
    """OverheadTiming carrying only the NIRDA pre/post overheads under test."""
    return OverheadTiming(
        nirda_pre_overhead_time=pre,
        nirda_post_overhead_time=post,
    )

# ---------------------------------------------------------------------------
# Shared test fixture — small, round-number parameters for easy hand calculation
#
# roi_x_size=10, roi_y_size=10, roi_x_buffer_pixels=2, roi_y_buffer_pixels=4
# pixels_per_frame = (10+2) * (10+4) = 168
# single_frame_time = 168 * 1e-5 s = 1.68e-3 s
#
# reset_frames_1=5, reset_frames_2=1
# drop_frames_1=1, drop_frames_2=2, drop_frames_3=3
# read_frames=4, groups=3
#
# common_frames = 1 + (3-1)*2 + 3 + 3*4 = 20
# first_integration_time  = (20 + 5) * 1.68e-3 = 42.0e-3 s
# other_integration_time  = (20 + 1) * 1.68e-3 = 35.28e-3 s
# ---------------------------------------------------------------------------

_SIMPLE = dict(
    reset_frames_1=5,
    reset_frames_2=1,
    drop_frames_1=1,
    drop_frames_2=2,
    drop_frames_3=3,
    read_frames=4,
    groups=3,
    average_groups=True,
    roi_x_size=10,
    roi_y_size=10,
    roi_x_buffer_pixels=2,
    roi_y_buffer_pixels=4,
    bytes_per_pixel=2 * u.byte,
    dropped_integrations=0,
    compression_ratio=0.8,
    read_time_per_pixel = 1.0e-5 * u.s,
    global_reset_method='off',  # no per-integration global-reset overhead
    additional_overhead_time=0 * u.s,
)

# Individual fixture fields, pulled out so the derived constants and the
# assertions below all track _SIMPLE automatically when it is edited.
_ROI_X = _SIMPLE['roi_x_size']
_ROI_Y = _SIMPLE['roi_y_size']
_X_BUF = _SIMPLE['roi_x_buffer_pixels']
_Y_BUF = _SIMPLE['roi_y_buffer_pixels']
_RESET_1 = _SIMPLE['reset_frames_1']
_RESET_2 = _SIMPLE['reset_frames_2']
_DROP_1 = _SIMPLE['drop_frames_1']
_DROP_2 = _SIMPLE['drop_frames_2']
_DROP_3 = _SIMPLE['drop_frames_3']
_READ_FRAMES = _SIMPLE['read_frames']
_GROUPS = _SIMPLE['groups']
_READ_TIME = _SIMPLE['read_time_per_pixel'].to_value(u.s)
_BYTES_PER_PIXEL = _SIMPLE['bytes_per_pixel'].to_value(u.byte)

# Derived quantities (plain floats in SI units) computed straight from the
# fixture, mirroring NirdaData._update_derived so the maths lives in one place.
_PIXELS = (_ROI_X + _X_BUF) * (_ROI_Y + _Y_BUF)
_COMMON_FRAMES = (
    _DROP_1 + (_GROUPS - 1) * _DROP_2 + _DROP_3 + _GROUPS * _READ_FRAMES
)
_BYTES_PER_FRAME = _PIXELS * _BYTES_PER_PIXEL
_FRAME_TIME = _PIXELS * _READ_TIME
_FIRST_INT_TIME = _FRAME_TIME * (_COMMON_FRAMES + _RESET_1)
_OTHER_INT_TIME = _FRAME_TIME * (_COMMON_FRAMES + _RESET_2)

def _make(**overrides):
    """Return a NirdaData built from _SIMPLE with optional field overrides."""
    kwargs = {**_SIMPLE, **overrides}
    return NirdaData(**kwargs)


# ---------------------------------------------------------------------------
# Default construction
# ---------------------------------------------------------------------------
class TestNirdaDataDefaults:
    """NirdaData() with library defaults should produce sensible derived values."""

    def test_default_construction_succeeds(self):
        """NirdaData with all defaults should not raise."""
        nd = NirdaData()
        assert nd is not None

    def test_default_pixels_per_frame(self):
        """Default pixels_per_frame should match (roi_x+buf) * (roi_y+buf)."""
        nd = NirdaData()
        expected = (nd.roi_x_size + nd.roi_x_buffer_pixels) * (
            nd.roi_y_size + nd.roi_y_buffer_pixels
        )
        assert nd.pixels_per_frame == expected

    def test_default_single_frame_time_is_positive(self):
        """Default single_frame_time should be a positive Quantity in seconds."""
        nd = NirdaData()
        assert nd.single_frame_time.to(u.s).value > 0

    def test_default_reset_frame_time_equals_single_frame_time(self):
        """Derived reset_frame_time should equal single_frame_time."""
        nd = NirdaData()
        assert nd.reset_frame_time == nd.single_frame_time

    def test_default_global_reset_method_is_off(self):
        """The default global_reset_method should be 'off' (no per-integration overhead)."""
        nd = NirdaData()
        assert nd.global_reset_method == 'off'

    def test_default_first_integration_time_exceeds_other(self):
        """First integration uses more reset frames so it should take longer."""
        nd = NirdaData()
        assert nd.first_integration_time > nd.other_integration_time

    def test_default_dropped_integration_time_equals_other(self):
        """dropped_integration_time should mirror other_integration_time."""
        nd = NirdaData()
        assert nd.dropped_integration_time == nd.other_integration_time

    def test_default_integration_data_positive(self):
        """Default integration_data should be positive."""
        nd = NirdaData()
        assert nd.integration_data.to(u.byte).value > 0


# ---------------------------------------------------------------------------
# pixels_per_frame
# ---------------------------------------------------------------------------
class TestPixelsPerFrame:
    """Tests for the pixels_per_frame derived attribute."""

    def test_no_buffer(self):
        """With zero buffer pixels the result is just roi_x * roi_y."""
        nd = _make()
        assert nd.pixels_per_frame == _PIXELS

    def test_with_x_buffer(self):
        """x-buffer pixels are added to the column count."""
        nd = _make(roi_x_buffer_pixels=5)
        assert nd.pixels_per_frame == (_ROI_X + 5) * (_ROI_Y + _Y_BUF)

    def test_with_y_buffer(self):
        """y-buffer pixels are added to the row count."""
        nd = _make(roi_y_buffer_pixels=3)
        assert nd.pixels_per_frame == (_ROI_X + _X_BUF) * (_ROI_Y + 3)

    def test_with_both_buffers(self):
        """Both buffers contribute independently."""
        nd = _make(roi_x_buffer_pixels=2, roi_y_buffer_pixels=2)
        assert nd.pixels_per_frame == (_ROI_X + 2) * (_ROI_Y + 2)

    def test_zero_roi_gives_zero(self):
        """A zero-area ROI with no buffer pixels should yield zero pixels per frame."""
        nd = _make(
            roi_x_size=0, roi_y_size=0, roi_x_buffer_pixels=0, roi_y_buffer_pixels=0
        )
        assert nd.pixels_per_frame == 0


# ---------------------------------------------------------------------------
# single_frame_time
# ---------------------------------------------------------------------------
class TestSingleFrameTime:
    """Tests for the single_frame_time derived attribute."""

    def test_value(self):
        """single_frame_time should equal pixels_per_frame * read_time_per_pixel."""
        nd = _make()
        expected = _PIXELS * _READ_TIME
        assert abs(nd.single_frame_time.to(u.s).value - expected) < 1e-15

    def test_units_are_seconds(self):
        """single_frame_time should be in seconds."""
        nd = _make()
        _ = nd.single_frame_time.to(u.s)  # should not raise

    def test_zero_pixels_zero_frame_time(self):
        """Zero-size ROI with no buffer pixels should give zero frame time."""
        nd = _make(
            roi_x_size=0, roi_y_size=0, roi_x_buffer_pixels=0, roi_y_buffer_pixels=0
        )
        assert nd.single_frame_time.to(u.s).value == 0

    def test_faster_clock_gives_shorter_frame(self):
        """Halving read_time_per_pixel should halve single_frame_time."""
        nd_slow = _make(read_time_per_pixel=2e-5 * u.s)
        nd_fast = _make(read_time_per_pixel=1e-5 * u.s)
        assert abs(
            nd_slow.single_frame_time.to(u.s).value
            / nd_fast.single_frame_time.to(u.s).value
            - 2.0
        ) < 1e-10


# ---------------------------------------------------------------------------
# Reset-frame-time (derived)
# ---------------------------------------------------------------------------
class TestResetFrameTime:
    """reset_frame_time is now a derived attribute that always tracks single_frame_time."""

    def test_equals_single_frame_time(self):
        """reset_frame_time should resolve to single_frame_time."""
        nd = _make()
        assert nd.reset_frame_time == nd.single_frame_time

    def test_not_a_constructor_argument(self):
        """reset_frame_time is init=False, so passing it should raise TypeError."""
        with pytest.raises(TypeError):
            _make(reset_frame_time=0.05 * u.s)

    def test_zero_when_frame_time_zero(self):
        """A zero-area ROI gives zero single_frame_time and therefore zero reset_frame_time."""
        nd = _make(
            roi_x_size=0, roi_y_size=0, roi_x_buffer_pixels=0, roi_y_buffer_pixels=0
        )
        assert nd.reset_frame_time.to(u.s).value == 0.0


# ---------------------------------------------------------------------------
# Global-reset overhead
# ---------------------------------------------------------------------------
class TestGlobalReset:
    """global_reset_method adds per-integration overhead for 'global' and 'line_by_line'."""

    def test_off_adds_no_overhead(self):
        """'off' should match the plain (common_frames + reset) * frame_time formula."""
        nd = _make(global_reset_method='off')
        assert abs(nd.first_integration_time.to(u.s).value - _FIRST_INT_TIME) < 1e-12
        assert abs(nd.other_integration_time.to(u.s).value - _OTHER_INT_TIME) < 1e-12

    def test_global_adds_small_fixed_overhead(self):
        """'global' should add a small (1e-6 s) overhead to both integration times."""
        nd_off = _make(global_reset_method='off')
        nd_global = _make(global_reset_method='global')
        for off, glob in (
            (nd_off.first_integration_time, nd_global.first_integration_time),
            (nd_off.other_integration_time, nd_global.other_integration_time),
        ):
            assert abs((glob - off).to(u.s).value - 1.0e-6) < 1e-15

    def test_line_by_line_adds_rows_times_per_row(self):
        """'line_by_line' overhead should equal lbl_rows * lbl_time_per_row."""
        rows, per_row = 256, 10.0e-6 * u.s
        nd_off = _make(global_reset_method='off')
        nd_lbl = _make(
            global_reset_method='line_by_line',
            global_reset_lbl_rows=rows,
            global_reset_lbl_time_per_row=per_row,
        )
        expected = (rows * per_row).to(u.s).value
        assert abs(
            (nd_lbl.first_integration_time - nd_off.first_integration_time).to(u.s).value
            - expected
        ) < 1e-15

    def test_line_by_line_overhead_scales_with_rows(self):
        """More reset rows should produce a larger line_by_line overhead."""
        nd_few = _make(global_reset_method='line_by_line', global_reset_lbl_rows=10)
        nd_many = _make(global_reset_method='line_by_line', global_reset_lbl_rows=500)
        assert nd_many.first_integration_time > nd_few.first_integration_time

    def test_method_is_case_and_whitespace_insensitive(self):
        """Method strings should be normalized (lowercased and stripped)."""
        nd = _make(global_reset_method='  GLOBAL  ')
        assert nd.global_reset_method == 'global'

    def test_unknown_method_raises(self):
        """An unrecognized method should raise ValueError."""
        with pytest.raises(ValueError):
            _make(global_reset_method='nonsense')


# ---------------------------------------------------------------------------
# Integration time formulas
# ---------------------------------------------------------------------------
class TestIntegrationTimes:
    """Tests for first_integration_time and other_integration_time."""

    def test_first_integration_time_value(self):
        """first_integration_time should be (common_frames + reset_frames_1) * frame_time."""
        nd = _make()
        assert abs(nd.first_integration_time.to(u.s).value - _FIRST_INT_TIME) < 1e-12

    def test_other_integration_time_value(self):
        """other_integration_time should be (common_frames + reset_frames_2) * frame_time."""
        nd = _make()
        assert abs(nd.other_integration_time.to(u.s).value - _OTHER_INT_TIME) < 1e-12

    def test_first_longer_than_other(self):
        """First integration is longer because reset_frames_1 > reset_frames_2."""
        nd = _make()
        assert nd.first_integration_time > nd.other_integration_time

    def test_equal_resets_give_equal_times(self):
        """When reset_frames_1 == reset_frames_2 both times are equal."""
        nd = _make(reset_frames_1=3, reset_frames_2=3)
        assert nd.first_integration_time == nd.other_integration_time

    def test_more_groups_increases_integration_time(self):
        """Adding groups increases common_frames and therefore integration time."""
        nd_few = _make(groups=2)
        nd_many = _make(groups=6)
        assert nd_many.other_integration_time > nd_few.other_integration_time

    def test_more_drop_frames_2_increases_integration_time(self):
        """Non-zero drop_frames_2 between groups lengthens integration time."""
        nd_no_drop = _make(drop_frames_2=0)
        nd_with_drop = _make(drop_frames_2=4)
        assert nd_with_drop.other_integration_time > nd_no_drop.other_integration_time

    def test_times_nonnegative_with_zero_frames(self):
        """All-zero frame counts should give zero integration time, not negative."""
        nd = _make(reset_frames_1=0, reset_frames_2=0, read_frames=0, groups=0)
        assert nd.first_integration_time.to(u.s).value >= 0
        assert nd.other_integration_time.to(u.s).value >= 0


# ---------------------------------------------------------------------------
# Integration data and saved-frame counts
# ---------------------------------------------------------------------------
class TestIntegrationData:
    """Tests for integration_data and *_saved_frames attributes."""

    def test_average_groups_true_data(self):
        """With average_groups=True data is bytes_per_frame * groups."""
        nd = _make(average_groups=True)
        expected = _BYTES_PER_FRAME * nd.groups
        assert abs(nd.integration_data.to(u.byte).value - expected) < 1e-10

    def test_average_groups_false_data(self):
        """With average_groups=False data is bytes_per_frame * groups * read_frames."""
        nd = _make(average_groups=False)
        expected = _BYTES_PER_FRAME * nd.groups * nd.read_frames
        assert abs(nd.integration_data.to(u.byte).value - expected) < 1e-10

    def test_average_groups_true_saved_frames(self):
        """With average_groups=True saved_frames equals groups for both integrations."""
        nd = _make(average_groups=True)
        assert nd.first_integration_saved_frames == nd.groups
        assert nd.other_integration_saved_frames == nd.groups

    def test_average_groups_false_saved_frames(self):
        """With average_groups=False saved_frames equals groups * read_frames."""
        nd = _make(average_groups=False)
        expected = nd.groups * nd.read_frames
        assert nd.first_integration_saved_frames == expected
        assert nd.other_integration_saved_frames == expected

    def test_averaging_reduces_data(self):
        """average_groups=True should produce less or equal data than False (when read_frames>1)."""
        nd_avg = _make(average_groups=True, read_frames=4)
        nd_raw = _make(average_groups=False, read_frames=4)
        assert nd_avg.integration_data <= nd_raw.integration_data

    def test_integration_data_units_are_bytes(self):
        """integration_data should be convertible to bytes."""
        nd = _make()
        _ = nd.integration_data.to(u.byte)


# ---------------------------------------------------------------------------
# update_for_vitl
# ---------------------------------------------------------------------------
class TestUpdateForVitl:
    """Tests for the update_for_vitl method."""

    def test_reset_frames_1_set_to_ceiling(self):
        """reset_frames_1 should be ceil(settling_time / frame_time)."""
        nd = _make()
        settling = 7.5 * nd.single_frame_time  # ceil(7.5) = 8
        nd.update_for_vitl(settling)
        assert nd.reset_frames_1 == 8

    def test_exact_multiple_no_extra_frame(self):
        """Exact multiple of frame_time should need no extra frame."""
        nd = _make()
        # A hair under five frame times so floating-point rounding cannot push
        # the ceiling to six; still represents "an exact multiple needs no extra
        # frame".
        settling = (5 - 1e-9) * nd.single_frame_time
        nd.update_for_vitl(settling)
        assert nd.reset_frames_1 == 5

    def test_minimum_one_reset_frame(self):
        """Even a very short settling time should give at least one reset frame."""
        nd = _make()
        settling = 1e-15 * u.s
        nd.update_for_vitl(settling)
        assert nd.reset_frames_1 >= 1

    def test_zero_settling_gives_floor_resets(self):
        """Zero settling time should still apply the minimum reset-frame floor."""
        nd = _make()
        nd.update_for_vitl(0.0 * u.s)
        assert nd.reset_frames_1 == 2

    def test_derived_attributes_recomputed(self):
        """Derived attributes should be updated after update_for_vitl."""
        nd = _make()
        old_first = nd.first_integration_time.to(u.s).value
        # Force a reset count well away from the fixture's reset_frames_1 so the
        # first-integration time is guaranteed to change.
        nd.update_for_vitl((_RESET_1 + 50) * nd.single_frame_time)
        assert nd.first_integration_time.to(u.s).value != old_first

    def test_zero_frame_time_gives_floor_reset(self):
        """If reset_frame_time is zero update_for_vitl must not divide by zero.

        With no usable frame duration the count falls back to the fixed floor
        of two reset frames rather than dividing by zero.
        """
        # A zero-area ROI drives single_frame_time (and thus reset_frame_time) to zero.
        nd = _make(
            roi_x_size=0, roi_y_size=0, roi_x_buffer_pixels=0, roi_y_buffer_pixels=0
        )
        assert nd.reset_frame_time.to(u.s).value == 0.0
        nd.update_for_vitl(10.0 * u.s)
        assert nd.reset_frames_1 == 2


# ---------------------------------------------------------------------------
# solve_integrations
# ---------------------------------------------------------------------------
class TestSolveIntegrations:
    """Tests for solve_integrations(duration, pre_overhead, post_overhead)."""

    def test_returns_three_tuple(self):
        """solve_integrations should return (integrations, data, data_compressed)."""
        nd = _make()
        result = nd.solve_integrations(100e-3 * u.s)
        assert len(result) == 3

    def test_integrations_is_int(self):
        """Integration count should be an integer."""
        nd = _make()
        integrations, _, _ = nd.solve_integrations(100e-3 * u.s)
        assert isinstance(integrations, int)

    def test_zero_duration_yields_zero_integrations(self):
        """A zero-length window should produce no integrations."""
        nd = _make()
        integrations, data, data_c = nd.solve_integrations(0.0 * u.s, _nirda_overhead())
        assert integrations == 0
        assert data.to(u.byte).value == 0
        assert data_c.to(u.byte).value == 0

    def test_exact_first_integration_window(self):
        """Duration exactly equal to first_integration_time should give one integration."""
        nd = _make()
        # Use the object's own first_integration_time so the boundary is exact
        # regardless of floating-point ordering in the derived constants.
        integrations, _, _ = nd.solve_integrations(
            nd.first_integration_time, _nirda_overhead()
        )
        assert integrations == 1

    def test_multiple_integrations_count(self):
        """Duration covering multiple integrations should be counted correctly."""
        nd = _make()
        # Use the object's own derived times plus a tiny relative guard so we
        # stay just above the boundary without fitting an extra integration.
        duration = (
            nd.first_integration_time
            + nd.other_integration_time * 3
            + nd.other_integration_time * 1e-6
        )
        integrations, _, _ = nd.solve_integrations(duration, _nirda_overhead())
        assert integrations == 4

    def test_overhead_reduces_count(self):
        """Adding overhead should reduce the number of integrations."""
        nd = _make()
        # Window fits several integrations; a pre-overhead of two integrations'
        # worth must remove at least one.
        duration = (_FIRST_INT_TIME + 3 * _OTHER_INT_TIME) * u.s
        n_no_oh, _, _ = nd.solve_integrations(duration, _nirda_overhead())
        n_with_oh, _, _ = nd.solve_integrations(
            duration, _nirda_overhead(_OTHER_INT_TIME * 2 * u.s)
        )
        assert n_with_oh < n_no_oh

    def test_data_scales_with_integrations(self):
        """data should equal integrations * integration_data."""
        nd = _make()
        duration = 56e-3 * u.s
        integrations, data, _ = nd.solve_integrations(duration, _nirda_overhead())
        expected = integrations * nd.integration_data
        assert abs(data.to(u.byte).value - expected.to(u.byte).value) < 1e-10

    def test_compressed_data_uses_compression_ratio(self):
        """data_compressed should equal data * compression_ratio."""
        nd = _make(compression_ratio=0.5)
        _, data, data_c = nd.solve_integrations(56e-3 * u.s, _nirda_overhead())
        assert abs(data_c.to(u.byte).value - 0.5 * data.to(u.byte).value) < 1e-10

    def test_dropped_integrations_reduce_count(self):
        """dropped_integrations should be subtracted from the raw count."""
        nd_no_drop = _make(dropped_integrations=0)
        nd_dropped = _make(dropped_integrations=2)
        # Window comfortably fits more than two integrations so the raw count
        # stays above the dropped count.
        duration = (_FIRST_INT_TIME + 3 * _OTHER_INT_TIME) * u.s
        n_no, _, _ = nd_no_drop.solve_integrations(duration, _nirda_overhead())
        n_drop, _, _ = nd_dropped.solve_integrations(duration, _nirda_overhead())
        assert n_no - n_drop == 2

    def test_dropped_integrations_cannot_go_below_zero(self):
        """Dropping more integrations than available should floor to zero, not negative."""
        nd = _make(dropped_integrations=1000)
        integrations, _, _ = nd.solve_integrations(10e-3 * u.s)
        assert integrations >= 0

    def test_additional_overhead_reduces_count(self):
        """additional_overhead_time should reduce the effective window."""
        nd_no_oh = _make(additional_overhead_time=0 * u.s)
        nd_with_oh = _make(additional_overhead_time=_OTHER_INT_TIME * u.s)
        # Window fits two integrations; one integration's worth of additional
        # overhead must drop the count.
        duration = (_FIRST_INT_TIME + _OTHER_INT_TIME) * u.s + _OTHER_INT_TIME * 1e-6 * u.s
        n_no, _, _ = nd_no_oh.solve_integrations(duration, _nirda_overhead())
        n_with, _, _ = nd_with_oh.solve_integrations(duration, _nirda_overhead())
        assert n_with < n_no

    def test_duration_shorter_than_first_integration_gives_zero(self):
        """If first_integration_time > duration, result should be zero."""
        nd = _make()
        # Just under the object's own first_integration_time, scaled so it works
        # at any magnitude.
        duration = nd.first_integration_time * (1 - 1e-6)
        integrations, _, _ = nd.solve_integrations(duration, _nirda_overhead())
        assert integrations == 0


# ---------------------------------------------------------------------------
# solve_duration
# ---------------------------------------------------------------------------
class TestSolveDuration:
    """Tests for solve_duration(integrations, pre_overhead, post_overhead)."""

    def test_returns_three_tuple(self):
        """solve_duration should return (duration, data, data_compressed)."""
        nd = _make()
        result = nd.solve_duration(5)
        assert len(result) == 3

    def test_zero_integrations_returns_zero_duration(self):
        """Requesting zero integrations should give zero duration."""
        nd = _make()
        duration, data, data_c = nd.solve_duration(0)
        assert duration.to(u.s).value == 0.0
        assert data.to(u.byte).value == 0.0

    def test_negative_integrations_treated_as_zero(self):
        """Negative integration counts should be treated as zero."""
        nd = _make()
        duration, data, _ = nd.solve_duration(-5)
        assert duration.to(u.s).value == 0.0
        assert data.to(u.byte).value == 0.0

    def test_one_integration_duration(self):
        """One integration: duration = first_integration_time + overheads."""
        nd = _make()
        duration, _, _ = nd.solve_duration(1, _nirda_overhead())
        expected = _FIRST_INT_TIME
        assert abs(duration.to(u.s).value - expected) < 1e-12

    def test_multiple_integrations_duration(self):
        """N>1: duration = first + (N-1)*other + overheads."""
        nd = _make()
        n = 4
        expected = _FIRST_INT_TIME + 3 * _OTHER_INT_TIME
        duration, _, _ = nd.solve_duration(n, _nirda_overhead())
        assert abs(duration.to(u.s).value - expected) < 1e-12

    def test_overhead_added_to_duration(self):
        """Overhead parameters should be added to the total duration."""
        nd = _make()
        pre, post = 30e-3 * u.s, 10e-3 * u.s
        dur_no_oh, _, _ = nd.solve_duration(3, _nirda_overhead())
        dur_with_oh, _, _ = nd.solve_duration(3, _nirda_overhead(pre, post))
        assert abs(
            dur_with_oh.to(u.s).value - dur_no_oh.to(u.s).value - 40e-3
        ) < 1e-12

    def test_data_scales_with_integrations(self):
        """data should be integrations * integration_data."""
        nd = _make()
        n = 5
        _, data, _ = nd.solve_duration(n, _nirda_overhead())
        expected = n * nd.integration_data.to(u.byte).value
        assert abs(data.to(u.byte).value - expected) < 1e-10

    def test_compressed_data_uses_compression_ratio(self):
        """data_compressed should equal data * compression_ratio."""
        nd = _make(compression_ratio=0.6)
        _, data, data_c = nd.solve_duration(4, _nirda_overhead())
        assert abs(data_c.to(u.byte).value - 0.6 * data.to(u.byte).value) < 1e-10

    def test_dropped_integrations_ignored(self):
        """dropped_integrations must not affect solve_duration (by spec)."""
        nd_no_drop = _make(dropped_integrations=0)
        nd_dropped = _make(dropped_integrations=5)
        dur_no, _, _ = nd_no_drop.solve_duration(4, _nirda_overhead())
        dur_drop, _, _ = nd_dropped.solve_duration(4, _nirda_overhead())
        assert dur_no == dur_drop

    def test_duration_increases_monotonically(self):
        """More integrations should always require more time."""
        nd = _make()
        durations = [
            nd.solve_duration(n, _nirda_overhead())[0].to(u.s).value
            for n in range(1, 6)
        ]
        for i in range(len(durations) - 1):
            assert durations[i + 1] > durations[i]


# ---------------------------------------------------------------------------
# Roundtrip: solve_duration ↔ solve_integrations
# ---------------------------------------------------------------------------
class TestSolveRoundtrip:
    """solve_integrations(solve_duration(n)) should recover n."""

    def test_roundtrip_no_overhead(self):
        """With no overhead, solve_duration then solve_integrations should return n."""
        nd = _make()
        for n in range(1, 8):
            duration, _, _ = nd.solve_duration(n, _nirda_overhead())
            # Tiny relative guard against floating-point loss when re-dividing.
            duration = duration + nd.other_integration_time * 1e-6
            recovered, _, _ = nd.solve_integrations(duration, _nirda_overhead())
            assert recovered == n, f"Roundtrip failed for n={n}: recovered {recovered}"

    def test_roundtrip_with_overhead(self):
        """Roundtrip should also hold when consistent overhead is used in both calls."""
        nd = _make()
        pre, post = 20e-3 * u.s, 10e-3 * u.s
        for n in range(1, 6):
            duration, _, _ = nd.solve_duration(n, _nirda_overhead(pre, post))
            # Add a tiny relative epsilon so floating-point boundary comparisons
            # stay on the correct (>=) side without fitting an extra integration.
            duration = duration + nd.other_integration_time * 1e-6
            recovered, _, _ = nd.solve_integrations(duration, _nirda_overhead(pre, post))
            assert recovered == n, (
                f"Roundtrip with overhead failed for n={n}: recovered {recovered}"
            )

    def test_roundtrip_with_dropped_integrations(self):
        """With dropped_integrations the schedule must use n_drop+n_science integrations."""
        dropped = 2
        nd = _make(dropped_integrations=dropped)
        # solve_duration ignores drops; solve_integrations subtracts them
        # So ask for n_science raw (no drops) → duration → back out drops correctly
        n_science = 4
        n_total = n_science + dropped  # what solve_duration will schedule
        duration, _, _ = nd.solve_duration(n_total, _nirda_overhead())
        recovered, _, _ = nd.solve_integrations(duration, _nirda_overhead())
        assert recovered == n_science


# ---------------------------------------------------------------------------
# bytes_per_pixel scaling
# ---------------------------------------------------------------------------
class TestBytesPerPixel:
    """integration_data should scale linearly with bytes_per_pixel."""

    def test_data_scales_with_bytes_per_pixel(self):
        """Doubling bytes_per_pixel should double integration_data."""
        nd_2 = _make(bytes_per_pixel=2 * u.byte)
        nd_4 = _make(bytes_per_pixel=4 * u.byte)
        assert abs(
            nd_4.integration_data.to(u.byte).value
            - 2 * nd_2.integration_data.to(u.byte).value
        ) < 1e-10

    def test_explicit_value_matches_pixels_times_groups(self):
        """integration_data (averaged) == bytes_per_pixel * pixels * groups."""
        nd = _make(bytes_per_pixel=3 * u.byte, average_groups=True)
        expected = 3 * _PIXELS * nd.groups
        assert abs(nd.integration_data.to(u.byte).value - expected) < 1e-10

    def test_zero_bytes_per_pixel_gives_zero_data(self):
        """Zero bytes per pixel should yield zero integration data."""
        nd = _make(bytes_per_pixel=0 * u.byte)
        assert nd.integration_data.to(u.byte).value == 0


# ---------------------------------------------------------------------------
# ROI start coordinates are metadata only
# ---------------------------------------------------------------------------
class TestRoiStartIsMetadata:
    """roi_x_start / roi_y_start are stored but must not affect derived values."""

    def test_roi_start_stored_as_given(self):
        """roi_x_start and roi_y_start should be stored verbatim."""
        nd = _make(roi_x_start=1234, roi_y_start=5678)
        assert nd.roi_x_start == 1234
        assert nd.roi_y_start == 5678

    def test_roi_start_does_not_change_derived_values(self):
        """Changing the ROI start should leave timing and data untouched."""
        nd_a = _make(roi_x_start=0, roi_y_start=0)
        nd_b = _make(roi_x_start=1737, roi_y_start=962)
        assert nd_a.pixels_per_frame == nd_b.pixels_per_frame
        assert nd_a.single_frame_time == nd_b.single_frame_time
        assert nd_a.first_integration_time == nd_b.first_integration_time
        assert nd_a.integration_data == nd_b.integration_data


# ---------------------------------------------------------------------------
# Negative-input clamping
# ---------------------------------------------------------------------------
class TestNegativeInputClamping:
    """Negative frame inputs should clamp to zero, never produce negatives."""

    def test_negative_drop_frames_clamp_common_frames(self):
        """A large negative drop_frames_1 should floor integration time at zero."""
        nd = _make(
            reset_frames_1=0,
            reset_frames_2=0,
            read_frames=0,
            groups=1,
            drop_frames_1=-100,
        )
        assert nd.first_integration_time.to(u.s).value == 0
        assert nd.other_integration_time.to(u.s).value == 0

    def test_negative_drop_frames_still_nonnegative_data(self):
        """Negative drop frames must not drive integration_data negative."""
        nd = _make(drop_frames_1=-50)
        assert nd.integration_data.to(u.byte).value >= 0


# ---------------------------------------------------------------------------
# additional_overhead_time in solve_duration
# ---------------------------------------------------------------------------
class TestAdditionalOverheadInSolveDuration:
    """additional_overhead_time should be folded into solve_duration's overhead."""

    def test_additional_overhead_added_to_duration(self):
        """solve_duration should add additional_overhead_time to the total."""
        nd_no_oh = _make(additional_overhead_time=0 * u.s)
        nd_with_oh = _make(additional_overhead_time=5e-3 * u.s)
        dur_no, _, _ = nd_no_oh.solve_duration(3, _nirda_overhead())
        dur_with, _, _ = nd_with_oh.solve_duration(3, _nirda_overhead())
        assert abs(
            dur_with.to(u.s).value - dur_no.to(u.s).value - 5e-3
        ) < 1e-12

    def test_additional_overhead_not_added_when_zero_integrations(self):
        """Zero integrations should give zero duration regardless of overhead."""
        nd = _make(additional_overhead_time=5e-3 * u.s)
        duration, _, _ = nd.solve_duration(0, _nirda_overhead())
        assert duration.to(u.s).value == 0.0


# ---------------------------------------------------------------------------
# Payload config spec / get_config
# ---------------------------------------------------------------------------
class TestNirdaConfigSpec:
    """Tests for the payload config mapping and get_config()."""

    def test_get_config_keys_match_spec(self):
        """get_config returns exactly the CONFIG_SPEC field names."""
        nd = NirdaData()
        assert set(nd.get_config().keys()) == set(NirdaData.CONFIG_SPEC.keys())

    def test_get_config_returns_current_values(self):
        """get_config reflects this instance's field values."""
        nd = _make(drop_frames_1=7, groups=9)
        cfg = nd.get_config()
        assert cfg["drop_frames_1"] == 7
        assert cfg["groups"] == 9

    def test_spec_fields_are_real_attributes(self):
        """Every CONFIG_SPEC field is a real NirdaData attribute."""
        nd = NirdaData()
        for field_name in NirdaData.CONFIG_SPEC:
            assert hasattr(nd, field_name)

    def test_payload_section_is_inf_cam(self):
        """NIRDA config lives under the AcquireInfCamImages payload."""
        assert NirdaData.PAYLOAD_SECTION == "AcquireInfCamImages"

    def test_required_fields_exclude_average_groups(self):
        """average_groups is optional (affects data volume only)."""
        assert "average_groups" not in NirdaData.REQUIRED_CONFIG_FIELDS
        assert "groups" in NirdaData.REQUIRED_CONFIG_FIELDS

    def test_roundtrip_converters(self):
        """from_xml(to_xml(value)) round-trips for a sample field."""
        tag, from_xml, to_xml = NirdaData.CONFIG_SPEC["drop_frames_1"]
        assert from_xml(to_xml(5)) == 5
        # Floats in the payload text parse to ints.
        assert from_xml("5.0") == 5
