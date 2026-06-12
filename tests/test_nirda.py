"""Unit tests for the NirdaData class in shortschedule.nirda.

Tests cover:
- Default construction and derived-attribute values
- pixels_per_frame and single_frame_time calculations
- Reset-frame-time sentinel resolution
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

# ---------------------------------------------------------------------------
# Shared test fixture — small, round-number parameters for easy hand calculation
#
# roi_x_size=10, roi_y_size=10, roi_x_buffer_pixels=0, roi_y_buffer_pixels=0
# pixels_per_frame = 10 * 10 = 100
# single_frame_time = 100 * 1e-5 s = 1e-3 s
#
# reset_frames_1=5, reset_frames_2=1
# drop_frames_1=0, drop_frames_2=0, drop_frames_3=0
# read_frames=4, groups=3
#
# common_frames = 0 + (3-1)*0 + 0 + 3*4 = 12
# first_integration_time  = (12 + 5) * 1e-3 = 17e-3 s
# other_integration_time  = (12 + 1) * 1e-3 = 13e-3 s
# ---------------------------------------------------------------------------

_SIMPLE = dict(
    reset_frames_1=5,
    reset_frames_2=1,
    drop_frames_1=0,
    drop_frames_2=0,
    drop_frames_3=0,
    read_frames=4,
    groups=3,
    average_groups=True,
    roi_x_size=10,
    roi_y_size=10,
    roi_x_buffer_pixels=0,
    roi_y_buffer_pixels=0,
    bytes_per_pixel=2 * u.byte,
    dropped_integrations=0,
    compression_ratio=0.8,
    reset_frame_time=-1.0 * u.s,  # sentinel → becomes single_frame_time
    additional_overhead_time=0 * u.s,
)

_FRAME_TIME = 1e-3  # seconds, derived from _SIMPLE
_FIRST_INT_TIME = 17e-3  # seconds
_OTHER_INT_TIME = 13e-3  # seconds
_PIXELS = 100
_BYTES_PER_FRAME = 200  # bytes  (2 bytes/pixel * 100 pixels)


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

    def test_default_reset_frame_time_sentinel_resolved(self):
        """Default reset_frame_time sentinel (-1 s) should be replaced with single_frame_time."""
        nd = NirdaData()
        assert nd.reset_frame_time == nd.single_frame_time

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
        assert nd.pixels_per_frame == (10 + 5) * 10

    def test_with_y_buffer(self):
        """y-buffer pixels are added to the row count."""
        nd = _make(roi_y_buffer_pixels=3)
        assert nd.pixels_per_frame == 10 * (10 + 3)

    def test_with_both_buffers(self):
        """Both buffers contribute independently."""
        nd = _make(roi_x_buffer_pixels=2, roi_y_buffer_pixels=2)
        assert nd.pixels_per_frame == 12 * 12

    def test_zero_roi_gives_zero(self):
        """A zero-area ROI should yield zero pixels per frame."""
        nd = _make(roi_x_size=0, roi_y_size=0)
        assert nd.pixels_per_frame == 0


# ---------------------------------------------------------------------------
# single_frame_time
# ---------------------------------------------------------------------------
class TestSingleFrameTime:
    """Tests for the single_frame_time derived attribute."""

    def test_value(self):
        """single_frame_time should equal pixels_per_frame * read_time_per_pixel."""
        nd = _make()
        expected = _PIXELS * 1.0e-5
        assert abs(nd.single_frame_time.to(u.s).value - expected) < 1e-15

    def test_units_are_seconds(self):
        """single_frame_time should be in seconds."""
        nd = _make()
        _ = nd.single_frame_time.to(u.s)  # should not raise

    def test_zero_pixels_zero_frame_time(self):
        """Zero-size ROI should give zero frame time."""
        nd = _make(roi_x_size=0, roi_y_size=0)
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
# Reset-frame-time sentinel
# ---------------------------------------------------------------------------
class TestResetFrameTimeSentinel:
    """Negative reset_frame_time should be replaced; positive should be kept."""

    def test_sentinel_becomes_single_frame_time(self):
        """Sentinel value (-1 s) should resolve to single_frame_time."""
        nd = _make(reset_frame_time=-1.0 * u.s)
        assert nd.reset_frame_time == nd.single_frame_time

    def test_explicit_positive_value_kept(self):
        """A positive reset_frame_time should be used as-is."""
        custom = 0.05 * u.s
        nd = _make(reset_frame_time=custom)
        assert nd.reset_frame_time == custom

    def test_explicit_zero_value_kept(self):
        """An explicit zero reset_frame_time should remain zero."""
        nd = _make(reset_frame_time=0.0 * u.s)
        assert nd.reset_frame_time.to(u.s).value == 0.0

    def test_sentinel_affects_integration_time(self):
        """Changing reset_frame_time should change first_integration_time."""
        nd_default = _make()
        nd_custom = _make(reset_frame_time=0.1 * u.s)
        assert nd_custom.first_integration_time != nd_default.first_integration_time


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
        settling = 7.5e-3 * u.s  # frame_time=1e-3 → ceil(7.5)=8
        nd.update_for_vitl(settling)
        assert nd.reset_frames_1 == 8

    def test_exact_multiple_no_extra_frame(self):
        """Exact multiple of frame_time should need no extra frame."""
        nd = _make()
        settling = 5e-3 * u.s  # exactly 5 frame times
        nd.update_for_vitl(settling)
        assert nd.reset_frames_1 == 5

    def test_minimum_one_reset_frame(self):
        """Even a very short settling time should give at least one reset frame."""
        nd = _make()
        settling = 1e-15 * u.s
        nd.update_for_vitl(settling)
        assert nd.reset_frames_1 >= 1

    def test_zero_settling_gives_one_frame(self):
        """Zero settling time should still require one reset frame."""
        nd = _make()
        nd.update_for_vitl(0.0 * u.s)
        assert nd.reset_frames_1 == 1

    def test_derived_attributes_recomputed(self):
        """Derived attributes should be updated after update_for_vitl."""
        nd = _make()
        old_first = nd.first_integration_time.to(u.s).value
        nd.update_for_vitl(100e-3 * u.s)  # much longer settling → more resets
        assert nd.first_integration_time.to(u.s).value != old_first

    def test_zero_frame_time_gives_one_reset(self):
        """If reset_frame_time is zero update_for_vitl must not divide by zero."""
        nd = _make(reset_frame_time=0.0 * u.s, roi_x_size=0, roi_y_size=0)
        nd.update_for_vitl(10.0 * u.s)
        assert nd.reset_frames_1 == 1


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
        integrations, data, data_c = nd.solve_integrations(0.0 * u.s, 0 * u.s, 0 * u.s)
        assert integrations == 0
        assert data.to(u.byte).value == 0
        assert data_c.to(u.byte).value == 0

    def test_exact_first_integration_window(self):
        """Duration exactly equal to first_integration_time should give one integration."""
        nd = _make()
        integrations, _, _ = nd.solve_integrations(_FIRST_INT_TIME * u.s, 0 * u.s, 0 * u.s)
        assert integrations == 1

    def test_multiple_integrations_count(self):
        """Duration covering multiple integrations should be counted correctly."""
        nd = _make()
        # Use object's derived times + 1 ns guard to stay above the exact boundary
        # without crossing into the next integration window (other_int ≈ 13 ms).
        duration = nd.first_integration_time + nd.other_integration_time * 3 + 1e-9 * u.s
        integrations, _, _ = nd.solve_integrations(duration, 0 * u.s, 0 * u.s)
        assert integrations == 4

    def test_overhead_reduces_count(self):
        """Adding overhead should reduce the number of integrations."""
        nd = _make()
        duration = 100e-3 * u.s
        n_no_oh, _, _ = nd.solve_integrations(duration, 0 * u.s, 0 * u.s)
        n_with_oh, _, _ = nd.solve_integrations(duration, 30e-3 * u.s, 0 * u.s)
        assert n_with_oh < n_no_oh

    def test_data_scales_with_integrations(self):
        """data should equal integrations * integration_data."""
        nd = _make()
        duration = 56e-3 * u.s
        integrations, data, _ = nd.solve_integrations(duration, 0 * u.s, 0 * u.s)
        expected = integrations * nd.integration_data
        assert abs(data.to(u.byte).value - expected.to(u.byte).value) < 1e-10

    def test_compressed_data_uses_compression_ratio(self):
        """data_compressed should equal data * compression_ratio."""
        nd = _make(compression_ratio=0.5)
        _, data, data_c = nd.solve_integrations(56e-3 * u.s, 0 * u.s, 0 * u.s)
        assert abs(data_c.to(u.byte).value - 0.5 * data.to(u.byte).value) < 1e-10

    def test_dropped_integrations_reduce_count(self):
        """dropped_integrations should be subtracted from the raw count."""
        nd_no_drop = _make(dropped_integrations=0)
        nd_dropped = _make(dropped_integrations=2)
        duration = 100e-3 * u.s
        n_no, _, _ = nd_no_drop.solve_integrations(duration, 0 * u.s, 0 * u.s)
        n_drop, _, _ = nd_dropped.solve_integrations(duration, 0 * u.s, 0 * u.s)
        assert n_no - n_drop == 2

    def test_dropped_integrations_cannot_go_below_zero(self):
        """Dropping more integrations than available should floor to zero, not negative."""
        nd = _make(dropped_integrations=1000)
        integrations, _, _ = nd.solve_integrations(10e-3 * u.s)
        assert integrations >= 0

    def test_additional_overhead_reduces_count(self):
        """additional_overhead_time should reduce the effective window."""
        nd_no_oh = _make(additional_overhead_time=0 * u.s)
        nd_with_oh = _make(additional_overhead_time=20e-3 * u.s)
        duration = 60e-3 * u.s
        n_no, _, _ = nd_no_oh.solve_integrations(duration, 0 * u.s, 0 * u.s)
        n_with, _, _ = nd_with_oh.solve_integrations(duration, 0 * u.s, 0 * u.s)
        assert n_with < n_no

    def test_duration_shorter_than_first_integration_gives_zero(self):
        """If first_integration_time > duration, result should be zero."""
        nd = _make()
        duration = (_FIRST_INT_TIME - 1e-6) * u.s
        integrations, _, _ = nd.solve_integrations(duration, 0 * u.s, 0 * u.s)
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
        duration, _, _ = nd.solve_duration(1, 0 * u.s, 0 * u.s)
        expected = _FIRST_INT_TIME
        assert abs(duration.to(u.s).value - expected) < 1e-12

    def test_multiple_integrations_duration(self):
        """N>1: duration = first + (N-1)*other + overheads."""
        nd = _make()
        n = 4
        expected = _FIRST_INT_TIME + 3 * _OTHER_INT_TIME
        duration, _, _ = nd.solve_duration(n, 0 * u.s, 0 * u.s)
        assert abs(duration.to(u.s).value - expected) < 1e-12

    def test_overhead_added_to_duration(self):
        """Overhead parameters should be added to the total duration."""
        nd = _make()
        pre, post = 30e-3 * u.s, 10e-3 * u.s
        dur_no_oh, _, _ = nd.solve_duration(3, 0 * u.s, 0 * u.s)
        dur_with_oh, _, _ = nd.solve_duration(3, pre, post)
        assert abs(
            dur_with_oh.to(u.s).value - dur_no_oh.to(u.s).value - 40e-3
        ) < 1e-12

    def test_data_scales_with_integrations(self):
        """data should be integrations * integration_data."""
        nd = _make()
        n = 5
        _, data, _ = nd.solve_duration(n, 0 * u.s, 0 * u.s)
        expected = n * nd.integration_data.to(u.byte).value
        assert abs(data.to(u.byte).value - expected) < 1e-10

    def test_compressed_data_uses_compression_ratio(self):
        """data_compressed should equal data * compression_ratio."""
        nd = _make(compression_ratio=0.6)
        _, data, data_c = nd.solve_duration(4, 0 * u.s, 0 * u.s)
        assert abs(data_c.to(u.byte).value - 0.6 * data.to(u.byte).value) < 1e-10

    def test_dropped_integrations_ignored(self):
        """dropped_integrations must not affect solve_duration (by spec)."""
        nd_no_drop = _make(dropped_integrations=0)
        nd_dropped = _make(dropped_integrations=5)
        dur_no, _, _ = nd_no_drop.solve_duration(4, 0 * u.s, 0 * u.s)
        dur_drop, _, _ = nd_dropped.solve_duration(4, 0 * u.s, 0 * u.s)
        assert dur_no == dur_drop

    def test_duration_increases_monotonically(self):
        """More integrations should always require more time."""
        nd = _make()
        durations = [
            nd.solve_duration(n, 0 * u.s, 0 * u.s)[0].to(u.s).value
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
            duration, _, _ = nd.solve_duration(n, 0 * u.s, 0 * u.s)
            recovered, _, _ = nd.solve_integrations(duration, 0 * u.s, 0 * u.s)
            assert recovered == n, f"Roundtrip failed for n={n}: recovered {recovered}"

    def test_roundtrip_with_overhead(self):
        """Roundtrip should also hold when consistent overhead is used in both calls."""
        nd = _make()
        pre, post = 20e-3 * u.s, 10e-3 * u.s
        for n in range(1, 6):
            duration, _, _ = nd.solve_duration(n, pre, post)
            # Add a sub-nanosecond epsilon so floating-point boundary comparisons
            # stay on the correct (>=) side without fitting an extra integration.
            duration = duration + 1e-9 * u.s
            recovered, _, _ = nd.solve_integrations(duration, pre, post)
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
        duration, _, _ = nd.solve_duration(n_total, 0 * u.s, 0 * u.s)
        recovered, _, _ = nd.solve_integrations(duration, 0 * u.s, 0 * u.s)
        assert recovered == n_science
