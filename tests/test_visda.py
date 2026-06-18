"""Unit tests for the VisdaData class in shortschedule.data.visda.

Tests cover:
- Default construction and derived-attribute values
- pixels_per_frame, frame_bytes and coadd_bytes calculations
- single_frame_time (exposure plus read time)
- Science vs imaging byte-per-pixel selection (is_vissci)
- dropped_integration_time
- solve_integrations frame counting and coadd flooring
- solve_duration frame-count inversion
- Roundtrip consistency between solve_integrations and solve_duration
- Edge cases: zero ROI, zero duration, zero frame time
"""

# Third-party
import pytest
from astropy import units as u

# First-party/Local
from shortschedule.data.visda import VisdaData

# ---------------------------------------------------------------------------
# Shared test fixture — small, round-number parameters for easy hand calculation
#
# roi_dimension=4, num_rois=2
# pixels_per_frame = 4**2 * 2 = 32
#
# is_vissci=True, vissci_bytes_per_pixel=4 byte
# frame_bytes = 32 * 4 = 128 byte
# coadd_bytes = 128 * 5 = 640 byte
#
# exposure_time_s=0.1 s, read_time_per_frame_s=0 s
# single_frame_time = 0.1 s
# ---------------------------------------------------------------------------

_SIMPLE = dict(
    frames_per_coadd=5,
    roi_dimension=4,
    num_rois=2,
    exposure_time_s=0.1 * u.s,
    is_vissci=True,
    compression_ratio=0.5,
    vissci_bytes_per_pixel=4 * u.byte,
    visimg_bytes_per_pixel=2 * u.byte,
    read_time_per_frame_s=0.0 * u.s,
    additional_overhead_time=0 * u.s,
    dropped_frames=0,
)

# Individual fixture fields, pulled out so the derived constants and the
# assertions below all track _SIMPLE automatically when it is edited.
_ROI_DIM = _SIMPLE['roi_dimension']
_NUM_ROIS = _SIMPLE['num_rois']
_COADD = _SIMPLE['frames_per_coadd']
_VISSCI_BPP = _SIMPLE['vissci_bytes_per_pixel'].to_value(u.byte)
_VISIMG_BPP = _SIMPLE['visimg_bytes_per_pixel'].to_value(u.byte)
_EXPOSURE = _SIMPLE['exposure_time_s'].to_value(u.s)
_READ_TIME = _SIMPLE['read_time_per_frame_s'].to_value(u.s)
_COMP = _SIMPLE['compression_ratio']

# Derived quantities (plain floats in SI units) computed straight from the
# fixture, mirroring VisdaData._update_derived so the maths lives in one place.
_PIXELS = _ROI_DIM**2 * _NUM_ROIS
_FRAME_BYTES = _PIXELS * _VISSCI_BPP
_COADD_BYTES = _FRAME_BYTES * _COADD
_SFT = _EXPOSURE + _READ_TIME


def _make(**overrides):
    """Return a VisdaData built from _SIMPLE with optional field overrides."""
    kwargs = {**_SIMPLE, **overrides}
    return VisdaData(**kwargs)


# ---------------------------------------------------------------------------
# Default construction
# ---------------------------------------------------------------------------
class TestVisdaDataDefaults:
    """VisdaData() with library defaults should produce sensible derived values."""

    def test_default_construction_succeeds(self):
        """VisdaData with all defaults should not raise."""
        vd = VisdaData()
        assert vd is not None

    def test_default_pixels_per_frame(self):
        """Default pixels_per_frame should match roi_dimension**2 * num_rois."""
        vd = VisdaData()
        assert vd.pixels_per_frame == vd.roi_dimension**2 * vd.num_rois

    def test_default_frame_bytes_positive(self):
        """Default frame_bytes should be a positive Quantity in bytes."""
        vd = VisdaData()
        assert vd.frame_bytes.to(u.byte).value > 0

    def test_default_single_frame_time_positive(self):
        """Default single_frame_time should be a positive Quantity in seconds."""
        vd = VisdaData()
        assert vd.single_frame_time.to(u.s).value > 0

    def test_default_coadd_bytes_is_frame_bytes_times_coadds(self):
        """coadd_bytes should equal frame_bytes * frames_per_coadd."""
        vd = VisdaData()
        expected = vd.frame_bytes * vd.frames_per_coadd
        assert vd.coadd_bytes == expected


# ---------------------------------------------------------------------------
# pixels_per_frame
# ---------------------------------------------------------------------------
class TestPixelsPerFrame:
    """Tests for the pixels_per_frame derived attribute."""

    def test_value(self):
        """pixels_per_frame should equal roi_dimension**2 * num_rois."""
        vd = _make()
        assert vd.pixels_per_frame == _PIXELS

    def test_scales_with_num_rois(self):
        """Doubling num_rois should double pixels_per_frame."""
        vd_1 = _make(num_rois=1)
        vd_2 = _make(num_rois=2)
        assert vd_2.pixels_per_frame == 2 * vd_1.pixels_per_frame

    def test_scales_with_dimension_squared(self):
        """pixels_per_frame scales with the square of roi_dimension."""
        vd_2 = _make(roi_dimension=2)
        vd_4 = _make(roi_dimension=4)
        assert vd_4.pixels_per_frame == 4 * vd_2.pixels_per_frame

    def test_zero_dimension_gives_zero(self):
        """A zero ROI dimension should yield zero pixels per frame."""
        vd = _make(roi_dimension=0)
        assert vd.pixels_per_frame == 0


# ---------------------------------------------------------------------------
# frame_bytes / coadd_bytes
# ---------------------------------------------------------------------------
class TestFrameBytes:
    """Tests for frame_bytes and coadd_bytes derived attributes."""

    def test_vissci_frame_bytes(self):
        """Science mode uses vissci_bytes_per_pixel."""
        vd = _make(is_vissci=True)
        assert abs(vd.frame_bytes.to(u.byte).value - _FRAME_BYTES) < 1e-10

    def test_visimg_frame_bytes(self):
        """Imaging mode uses visimg_bytes_per_pixel."""
        vd = _make(is_vissci=False)
        expected = _PIXELS * _VISIMG_BPP
        assert abs(vd.frame_bytes.to(u.byte).value - expected) < 1e-10

    def test_coadd_bytes_value(self):
        """coadd_bytes should equal frame_bytes * frames_per_coadd."""
        vd = _make()
        assert abs(vd.coadd_bytes.to(u.byte).value - _COADD_BYTES) < 1e-10

    def test_frame_bytes_scales_with_bytes_per_pixel(self):
        """Doubling vissci_bytes_per_pixel should double frame_bytes."""
        vd_4 = _make(vissci_bytes_per_pixel=4 * u.byte)
        vd_8 = _make(vissci_bytes_per_pixel=8 * u.byte)
        assert abs(
            vd_8.frame_bytes.to(u.byte).value - 2 * vd_4.frame_bytes.to(u.byte).value
        ) < 1e-10

    def test_zero_roi_gives_zero_frame_bytes(self):
        """A zero ROI dimension should give zero frame bytes."""
        vd = _make(roi_dimension=0)
        assert vd.frame_bytes.to(u.byte).value == 0


# ---------------------------------------------------------------------------
# single_frame_time
# ---------------------------------------------------------------------------
class TestSingleFrameTime:
    """Tests for the single_frame_time derived attribute."""

    def test_value(self):
        """single_frame_time should equal exposure_time_s + read_time_per_frame_s."""
        vd = _make()
        assert abs(vd.single_frame_time.to(u.s).value - _SFT) < 1e-15

    def test_read_time_is_added(self):
        """A non-zero read time should lengthen the frame time."""
        vd_no_read = _make(read_time_per_frame_s=0.0 * u.s)
        vd_read = _make(read_time_per_frame_s=0.05 * u.s)
        assert abs(
            vd_read.single_frame_time.to(u.s).value
            - vd_no_read.single_frame_time.to(u.s).value
            - 0.05
        ) < 1e-12

    def test_units_are_seconds(self):
        """single_frame_time should be convertible to seconds."""
        vd = _make()
        _ = vd.single_frame_time.to(u.s)  # should not raise


# ---------------------------------------------------------------------------
# dropped_integration_time
# ---------------------------------------------------------------------------
class TestDroppedIntegrationTime:
    """Tests for dropped_integration_time."""

    def test_zero_dropped_frames(self):
        """No dropped frames should give zero dropped_integration_time."""
        vd = _make(dropped_frames=0)
        assert vd.dropped_integration_time.to(u.s).value == 0

    def test_dropped_integration_time_value(self):
        """dropped_integration_time should equal dropped_frames * single_frame_time."""
        vd = _make(dropped_frames=3)
        assert abs(vd.dropped_integration_time.to(u.s).value - 3 * _SFT) < 1e-12


# ---------------------------------------------------------------------------
# solve_integrations
# ---------------------------------------------------------------------------
class TestSolveIntegrations:
    """Tests for solve_integrations(duration, pre_overhead, post_overhead)."""

    def test_returns_three_tuple(self):
        """solve_integrations should return (frames, data, data_compressed)."""
        vd = _make()
        result = vd.solve_integrations(10.0 * u.s)
        assert len(result) == 3

    def test_frames_is_int(self):
        """Frame count should be an integer."""
        vd = _make()
        frames, _, _ = vd.solve_integrations(10.0 * u.s)
        assert isinstance(frames, int)

    def test_zero_duration_yields_zero_frames(self):
        """A zero-length window should produce no frames."""
        vd = _make()
        frames, data, data_c = vd.solve_integrations(0.0 * u.s, 0 * u.s, 0 * u.s)
        assert frames == 0
        assert data.to(u.byte).value == 0
        assert data_c.to(u.byte).value == 0

    def test_frames_floored_to_coadd_multiple(self):
        """Frame count should be floored to a whole number of coadds."""
        vd = _make()
        # Seven frames' worth of time, plus a tiny guard, must floor to 5.
        duration = vd.single_frame_time * 7 * (1 + 1e-9)
        frames, _, _ = vd.solve_integrations(duration, 0 * u.s, 0 * u.s)
        assert frames == 5

    def test_exact_coadd_window(self):
        """Time for exactly one coadd should give frames_per_coadd frames."""
        vd = _make()
        duration = vd.single_frame_time * _COADD * (1 + 1e-9)
        frames, _, _ = vd.solve_integrations(duration, 0 * u.s, 0 * u.s)
        assert frames == _COADD

    def test_frames_always_multiple_of_coadd(self):
        """Whatever the window, frames is a multiple of frames_per_coadd."""
        vd = _make()
        for n in range(1, 40):
            duration = vd.single_frame_time * n * (1 + 1e-9)
            frames, _, _ = vd.solve_integrations(duration, 0 * u.s, 0 * u.s)
            assert frames % _COADD == 0

    def test_overhead_reduces_frames(self):
        """Adding overhead should reduce the number of frames."""
        vd = _make()
        duration = vd.single_frame_time * 20
        n_no_oh, _, _ = vd.solve_integrations(duration, 0 * u.s, 0 * u.s)
        n_with_oh, _, _ = vd.solve_integrations(
            duration, vd.single_frame_time * 10, 0 * u.s
        )
        assert n_with_oh < n_no_oh

    def test_additional_overhead_reduces_frames(self):
        """additional_overhead_time should reduce the effective window."""
        vd_no_oh = _make(additional_overhead_time=0 * u.s)
        vd_with_oh = _make(additional_overhead_time=_SFT * 10 * u.s)
        duration = _make().single_frame_time * 20
        n_no, _, _ = vd_no_oh.solve_integrations(duration, 0 * u.s, 0 * u.s)
        n_with, _, _ = vd_with_oh.solve_integrations(duration, 0 * u.s, 0 * u.s)
        assert n_with < n_no

    def test_data_scales_with_frames(self):
        """data should equal frames * frame_bytes."""
        vd = _make()
        frames, data, _ = vd.solve_integrations(
            vd.single_frame_time * 20, 0 * u.s, 0 * u.s
        )
        expected = frames * vd.frame_bytes
        assert abs(data.to(u.byte).value - expected.to(u.byte).value) < 1e-6

    def test_compressed_data_uses_compression_ratio(self):
        """data_compressed should equal data * compression_ratio."""
        vd = _make(compression_ratio=0.5)
        _, data, data_c = vd.solve_integrations(
            vd.single_frame_time * 20, 0 * u.s, 0 * u.s
        )
        assert abs(data_c.to(u.byte).value - 0.5 * data.to(u.byte).value) < 1e-6

    def test_duration_shorter_than_one_coadd_gives_zero(self):
        """If less than one coadd fits, frame count should be zero."""
        vd = _make()
        duration = vd.single_frame_time * (_COADD - 1)
        frames, _, _ = vd.solve_integrations(duration, 0 * u.s, 0 * u.s)
        assert frames == 0

    def test_zero_frame_time_gives_zero_frames(self):
        """Zero single_frame_time must not divide by zero; it yields zero frames."""
        vd = _make(exposure_time_s=0.0 * u.s, read_time_per_frame_s=0.0 * u.s)
        assert vd.single_frame_time.to(u.s).value == 0
        frames, _, _ = vd.solve_integrations(100.0 * u.s, 0 * u.s, 0 * u.s)
        assert frames == 0

    def test_zero_frames_per_coadd_does_not_raise(self):
        """frames_per_coadd of zero must not trigger a modulo-by-zero error."""
        vd = _make(frames_per_coadd=0)
        frames, _, _ = vd.solve_integrations(
            vd.single_frame_time * 7, 0 * u.s, 0 * u.s
        )
        assert frames >= 0


# ---------------------------------------------------------------------------
# solve_duration
# ---------------------------------------------------------------------------
class TestSolveDuration:
    """Tests for solve_duration(integrations, pre_overhead, post_overhead)."""

    def test_returns_three_tuple(self):
        """solve_duration should return (duration, data, data_compressed)."""
        vd = _make()
        result = vd.solve_duration(5)
        assert len(result) == 3

    def test_zero_integrations_returns_zero_duration(self):
        """Requesting zero frames should give zero duration and zero data."""
        vd = _make()
        duration, data, data_c = vd.solve_duration(0)
        assert duration.to(u.s).value == 0.0
        assert data.to(u.byte).value == 0.0
        assert data_c.to(u.byte).value == 0.0

    def test_negative_integrations_treated_as_zero(self):
        """Negative frame counts should be treated as zero."""
        vd = _make()
        duration, data, _ = vd.solve_duration(-5)
        assert duration.to(u.s).value == 0.0
        assert data.to(u.byte).value == 0.0

    def test_duration_value(self):
        """duration = integrations * single_frame_time + overheads."""
        vd = _make()
        duration, _, _ = vd.solve_duration(10, 0 * u.s, 0 * u.s)
        assert abs(duration.to(u.s).value - 10 * _SFT) < 1e-12

    def test_overhead_added_to_duration(self):
        """Overhead parameters should be added to the total duration."""
        vd = _make()
        pre, post = 30.0 * u.s, 10.0 * u.s
        dur_no_oh, _, _ = vd.solve_duration(3, 0 * u.s, 0 * u.s)
        dur_with_oh, _, _ = vd.solve_duration(3, pre, post)
        assert abs(
            dur_with_oh.to(u.s).value - dur_no_oh.to(u.s).value - 40.0
        ) < 1e-12

    def test_data_is_pure_bytes(self):
        """data should be a plain byte Quantity, not byte*second."""
        vd = _make()
        _, data, _ = vd.solve_duration(10, 0 * u.s, 0 * u.s)
        # Convertible to bytes without a leftover time dimension.
        assert data.unit.is_equivalent(u.byte)

    def test_data_scales_with_integrations(self):
        """data should equal integrations * frame_bytes."""
        vd = _make()
        n = 7
        _, data, _ = vd.solve_duration(n, 0 * u.s, 0 * u.s)
        expected = n * vd.frame_bytes.to(u.byte).value
        assert abs(data.to(u.byte).value - expected) < 1e-6

    def test_compressed_data_uses_compression_ratio(self):
        """data_compressed should equal data * compression_ratio."""
        vd = _make(compression_ratio=0.25)
        _, data, data_c = vd.solve_duration(8, 0 * u.s, 0 * u.s)
        assert abs(data_c.to(u.byte).value - 0.25 * data.to(u.byte).value) < 1e-6

    def test_additional_overhead_added_to_duration(self):
        """solve_duration should add additional_overhead_time to the total."""
        vd_no_oh = _make(additional_overhead_time=0 * u.s)
        vd_with_oh = _make(additional_overhead_time=5.0 * u.s)
        dur_no, _, _ = vd_no_oh.solve_duration(3, 0 * u.s, 0 * u.s)
        dur_with, _, _ = vd_with_oh.solve_duration(3, 0 * u.s, 0 * u.s)
        assert abs(dur_with.to(u.s).value - dur_no.to(u.s).value - 5.0) < 1e-12

    def test_additional_overhead_not_added_when_zero_integrations(self):
        """Zero frames should give zero duration regardless of overhead."""
        vd = _make(additional_overhead_time=5.0 * u.s)
        duration, _, _ = vd.solve_duration(0, 0 * u.s, 0 * u.s)
        assert duration.to(u.s).value == 0.0

    def test_duration_increases_monotonically(self):
        """More frames should always require more time."""
        vd = _make()
        durations = [
            vd.solve_duration(n, 0 * u.s, 0 * u.s)[0].to(u.s).value
            for n in range(1, 6)
        ]
        for i in range(len(durations) - 1):
            assert durations[i + 1] > durations[i]


# ---------------------------------------------------------------------------
# Roundtrip: solve_duration ↔ solve_integrations
# ---------------------------------------------------------------------------
class TestSolveRoundtrip:
    """solve_integrations(solve_duration(n)) should recover n for coadd multiples."""

    def test_roundtrip_no_overhead(self):
        """With no overhead, a coadd-multiple frame count should round-trip."""
        vd = _make()
        for n in (_COADD, 2 * _COADD, 3 * _COADD):
            duration, _, _ = vd.solve_duration(n, 0 * u.s, 0 * u.s)
            # Tiny relative guard against floating-point loss when re-dividing.
            duration = duration + vd.single_frame_time * 1e-6
            recovered, _, _ = vd.solve_integrations(duration, 0 * u.s, 0 * u.s)
            assert recovered == n, f"Roundtrip failed for n={n}: recovered {recovered}"

    def test_roundtrip_with_overhead(self):
        """Roundtrip should hold when consistent overhead is used in both calls."""
        vd = _make()
        pre, post = 20.0 * u.s, 10.0 * u.s
        for n in (_COADD, 2 * _COADD):
            duration, _, _ = vd.solve_duration(n, pre, post)
            duration = duration + vd.single_frame_time * 1e-6
            recovered, _, _ = vd.solve_integrations(duration, pre, post)
            assert recovered == n, (
                f"Roundtrip with overhead failed for n={n}: recovered {recovered}"
            )


# ---------------------------------------------------------------------------
# Science vs imaging mode
# ---------------------------------------------------------------------------
class TestVissciVsVisimg:
    """is_vissci selects the bytes-per-pixel and therefore the data volume."""

    def test_science_uses_more_bytes_than_imaging(self):
        """With vissci_bpp > visimg_bpp, science mode produces more data."""
        vd_sci = _make(is_vissci=True)
        vd_img = _make(is_vissci=False)
        assert vd_sci.frame_bytes > vd_img.frame_bytes

    def test_mode_does_not_change_timing(self):
        """Switching mode must not change frame timing."""
        vd_sci = _make(is_vissci=True)
        vd_img = _make(is_vissci=False)
        assert vd_sci.single_frame_time == vd_img.single_frame_time
        assert vd_sci.pixels_per_frame == vd_img.pixels_per_frame
