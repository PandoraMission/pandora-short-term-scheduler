"""VISDA detector data helper functions.

This module implements ``VisdaData``, a dataclass that calculates VISDA
frame times and data volumes for a single observation. The calculations
require:

- per-observation details such as the ROI dimension, number of ROIs, and
  frames per coadd.
- detector configuration such as exposure time, read time per frame, and
  bytes per pixel.

Notes
-----
VISDA reads out one or more square ROIs and coadds a fixed number of
frames on-board. Each frame takes ``exposure_time_s``
to acquire, and ``frames_per_coadd`` consecutive frames are combined into a
single coadd before downlink.
"""

from __future__ import annotations

from warnings import warn
import math
from dataclasses import dataclass, field

from astropy import units as u
from astropy.units import Quantity

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .overhead import OverheadTiming


# Converters between payload XML text and VisdaData config values. Payload
# values are sometimes written as floats (e.g. "5.0"), so ints are parsed
# via float first. Exposure time is stored on the payload in microseconds.
def _xml_to_int(value: str) -> int:
    return int(float(value))


def _int_to_xml(value) -> str:
    return str(int(value))


def _xml_to_exposure(value: str) -> Quantity:
    return int(float(value)) * u.us


def _exposure_to_xml(value: Quantity) -> str:
    return str(int(value.to(u.us).value))


@dataclass
class VisdaData:
    """VISDA timing and data-volume parameters for a single observation.

    All timing and data results are computed automatically in
    ``__post_init__`` via ``_update_derived``.  Derived attributes
    (``init=False``) should be treated as read-only after construction.

    Parameters
    ----------
    frames_per_coadd : int
        Number of consecutive frames combined into a single coadd on-board.
    roi_dimension : int
        Side length of each square ROI in pixels.
    num_rois : int
        Number of ROIs read out per frame.
    exposure_time_s : Quantity[second]
        Per-frame exposure time.
    is_vissci : bool
        Whether the observation is in science mode (``True``) or imaging
        mode (``False``). Selects which ``*_bytes_per_pixel`` applies.
    compression_ratio : float
        Effective on-board compression factor applied to raw data.
        This is an empirical parameter.
    vissci_bytes_per_pixel : Quantity[byte]
        Raw storage cost per pixel in science mode.
    visimg_bytes_per_pixel : Quantity[byte]
        Raw storage cost per pixel in imaging mode.
    additional_overhead_time : Quantity[second]
        Extra fixed overhead beyond ``pre_overhead_time`` and
        ``post_overhead_time`` subtracted before scheduling frames.
    dropped_frames : int
        Number of frames dropped as a buffer.

    Attributes
    ----------
    pixels_per_frame : int
        Total pixels read out per frame across all ROIs.
    frame_bytes : Quantity[byte]
        Raw data volume of a single frame before compression.
    coadd_bytes : Quantity[byte]
        Raw data volume of one coadd (``frame_bytes * frames_per_coadd``).
    single_frame_time : Quantity[second]
        Wall-clock duration of one frame (exposure plus read time).
    dropped_integration_time : Quantity[second]
        Wall-clock duration of the dropped-frame buffer.
    """

    # Payload (PAN-SCICAL XML) integration. These are plain class attributes
    # (no annotations) so the dataclass does not treat them as fields.
    #
    # PAYLOAD_SECTION : the payload element these config fields live under.
    # CONFIG_SPEC     : maps a config field -> (xml_tag, from_xml, to_xml),
    #                   describing how to read/write each field from/to the
    #                   payload XML.
    # REQUIRED_CONFIG_FIELDS : fields needed to compute
    #                   NumTotalFramesRequested (the ROI fields only affect
    #                   data volume, so they are optional).
    PAYLOAD_SECTION = "AcquireVisCamScienceData"
    CONFIG_SPEC = {
        "exposure_time_s": (
            "ExposureTime_us",
            _xml_to_exposure,
            _exposure_to_xml,
        ),
        "frames_per_coadd": ("FramesPerCoadd", _xml_to_int, _int_to_xml),
        "roi_dimension": ("StarRoiDimension", _xml_to_int, _int_to_xml),
        "num_rois": ("MaxNumStarRois", _xml_to_int, _int_to_xml),
    }
    REQUIRED_CONFIG_FIELDS = frozenset(("exposure_time_s", "frames_per_coadd"))

    # VISDA observation configurations
    frames_per_coadd: int = 5
    roi_dimension: int = 50
    num_rois: int = 9
    exposure_time_s: Quantity = 0.2 * u.s
    is_vissci: bool = True

    # VISDA detector configuration
    compression_ratio: float = 0.25
    vissci_bytes_per_pixel: Quantity = 4 * u.byte
    visimg_bytes_per_pixel: Quantity = 2 * u.byte
    additional_overhead_time: Quantity = 0 * u.s
    dropped_frames: int = 1  # TODO: As a buffer we drop one VISDA frame

    # Derived attributes: computed in _update_derived, not constructor arguments
    pixels_per_frame: int = field(init=False)
    frame_bytes: Quantity = field(init=False)
    coadd_bytes: Quantity = field(init=False)
    single_frame_time: Quantity = field(init=False)
    dropped_integration_time: Quantity = field(init=False)

    # Optional logger. When provided (e.g. by ScheduleProcessor) warnings are
    # sent to it so they share the run log; otherwise ``warnings.warn`` is
    # used. Excluded from repr/equality so it never affects data comparisons.
    logger: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self._update_derived()

    def _warn(self, message: str) -> None:
        """Emit a warning via the attached logger, or ``warnings.warn``."""
        if self.logger is not None:
            self.logger.warning(message)
        else:
            warn(message)

    def _update_derived(self):
        """Recompute all derived timing and data-volume attributes."""
        self.pixels_per_frame = max(0, self.roi_dimension**2 * self.num_rois)

        if self.is_vissci:
            bytes_per_pixel = self.vissci_bytes_per_pixel
        else:
            bytes_per_pixel = self.visimg_bytes_per_pixel

        self.frame_bytes = (
            max(0, (self.pixels_per_frame * bytes_per_pixel).to(u.byte).value)
            * u.byte
        )
        self.coadd_bytes = self.frame_bytes * self.frames_per_coadd
        if self.coadd_bytes == 0.0 * u.s:
            self._warn("VISDA: Data size of one coadd was found to be 0.")

        self.single_frame_time = (
            max(
                0,
                (self.exposure_time_s.to(u.s)).value,
            )
            * u.s
        )
        if self.single_frame_time == 0.0 * u.s:
            self._warn("VISDA: Single frame time found to be 0.")

        self.dropped_integration_time = (
            self.dropped_frames * self.single_frame_time
        )

    def solve_integrations(
        self, duration: Quantity, overhead: OverheadTiming = None
    ):
        """Compute the number of frames that fit within a duration.

        Parameters
        ----------
        duration : Quantity[second]
            Total available observation time.
        overhead : OverheadTiming, default=None
            Overhead timings for nirda and visda.
            If None, then use default overheads.

        Returns
        -------
        frames : int
            Number of frames that fit in the available time, floored to a
            whole number of coadds.
        data : Quantity[byte]
            Total raw data volume *downlinked* for those frames. On-board
            coadding combines ``frames_per_coadd`` consecutive frames into a
            single saved frame, so the saved volume is the number of coadds
            times ``frame_bytes`` (not the raw pre-coadd frame count).
        data_compressed : Quantity[byte]
            Total compressed data volume.
        """

        if overhead is None:
            # Use default overheads
            from .overhead import OverheadTiming

            overhead = OverheadTiming()

        buffered_time = (
            duration
            - overhead.visda_pre_overhead_time
            - overhead.visda_post_overhead_time
            - self.additional_overhead_time
        )
        # Snap to the nearest microsecond before flooring. astropy ``Time``
        # subtraction (e.g. computing a merged sequence's duration) carries
        # sub-microsecond floating-point noise that varies with the absolute
        # epoch, so an exact integer frame count can land at, say,
        # 11389.9999998 and floor down a whole frame, which the
        # coadd-multiple flooring below then amplifies into a lost coadd.
        # Microsecond precision is far finer than one exposure, so this snap
        # is physically lossless.
        buffered_time = max(0.0, round(buffered_time.to(u.s).value, 6)) * u.s

        # Find total number of frames that can fit into our observation time.
        if self.single_frame_time.to(u.s).value <= 0:
            frames = 0
        else:
            frames = max(
                0,
                math.floor(
                    (buffered_time / self.single_frame_time)
                    .to(u.dimensionless_unscaled)
                    .value
                ),
            )

        # Number of frames needs to be a multiple of coadds.
        if self.frames_per_coadd > 0:
            frames = max(0, frames - (frames % self.frames_per_coadd))

        if self.frames_per_coadd > 0:
            coadds = frames // self.frames_per_coadd
        else:
            coadds = frames
        data = coadds * self.frame_bytes
        return (frames, data, data * self.compression_ratio)

    def solve_duration(
        self, integrations: int, overhead: OverheadTiming = None
    ):
        """Compute the total duration required to acquire a given number of frames.

        Parameters
        ----------
        integrations : int
            Desired number of frames.
        overhead : OverheadTiming, default=None
            Overhead timings for nirda and visda.
            If None, then use default overheads.

        Returns
        -------
        duration : Quantity[second]
            Total wall-clock time needed, including all overheads.
        data : Quantity[byte]
            Total raw data volume *downlinked* for those frames (net of
            on-board coadding; see :meth:`solve_integrations`).
        data_compressed : Quantity[byte]
            Total compressed data volume.
        """

        if overhead is None:
            # Use default overheads
            from .overhead import OverheadTiming

            overhead = OverheadTiming()

        integrations = max(0, integrations)
        overhead_time = (
            overhead.visda_pre_overhead_time
            + overhead.visda_post_overhead_time
            + self.additional_overhead_time
        )
        if self.frames_per_coadd > 0:
            coadds = integrations // self.frames_per_coadd
        else:
            coadds = integrations
        data = coadds * self.frame_bytes

        if integrations == 0:
            duration = 0.0 * u.s
        else:
            duration = integrations * self.single_frame_time + overhead_time

        return (duration, data, data * self.compression_ratio)

    def get_config(self) -> dict:
        """Return the current payload-config field values.

        Returns
        -------
        dict
            Maps each :data:`CONFIG_SPEC` field name to this instance's
            current value (e.g. ``{'frames_per_coadd': 5,
            'exposure_time_s': <Quantity 0.2 s>, ...}``). Useful for reading
            the default detector configuration when applying per-field
            overrides.
        """
        return {field: getattr(self, field) for field in self.CONFIG_SPEC}
