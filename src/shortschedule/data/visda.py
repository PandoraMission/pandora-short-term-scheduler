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
frames on-board. Each frame takes ``exposure_time_s + read_time_per_frame_s``
to acquire, and ``frames_per_coadd`` consecutive frames are combined into a
single coadd before downlink.
"""

import math
from dataclasses import dataclass, field

from astropy import units as u
from astropy.units import Quantity


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
    read_time_per_frame_s : Quantity[second]
        Detector read time per frame. Effectively negligible compared to
        the exposure time.
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
    read_time_per_frame_s: Quantity = 1.0e-6 * u.s  # This is not correct but this parameter is effectively 0 compared to the exposure time.
    additional_overhead_time: Quantity = 0 * u.s
    dropped_frames: int = 0

    # Derived attributes: computed in _update_derived, not constructor arguments
    pixels_per_frame: int = field(init=False)
    frame_bytes: Quantity = field(init=False)
    coadd_bytes: Quantity = field(init=False)
    single_frame_time: Quantity = field(init=False)
    dropped_integration_time: Quantity = field(init=False)

    def __post_init__(self):
        self._update_derived()

    def _update_derived(self):
        """Recompute all derived timing and data-volume attributes."""
        self.pixels_per_frame = max(0, self.roi_dimension**2 * self.num_rois)

        if self.is_vissci:
            bytes_per_pixel = self.vissci_bytes_per_pixel
        else:
            bytes_per_pixel = self.visimg_bytes_per_pixel

        self.frame_bytes = max(
            0, (self.pixels_per_frame * bytes_per_pixel).to(u.byte).value
        ) * u.byte
        self.coadd_bytes = self.frame_bytes * self.frames_per_coadd

        self.single_frame_time = max(
            0,
            (self.exposure_time_s.to(u.s) + self.read_time_per_frame_s.to(u.s)).value,
        ) * u.s
        self.dropped_integration_time = self.dropped_frames * self.single_frame_time

    def solve_integrations(
        self,
        duration: Quantity,
        pre_overhead_time: Quantity = 258.0 * u.s,
        post_overhead_time: Quantity = 102.0 * u.s,
    ):
        """Compute the number of frames that fit within a duration.

        Parameters
        ----------
        duration : Quantity[second]
            Total available observation time.
        pre_overhead_time : Quantity[second], optional
            Fixed overhead before science frames begin.
        post_overhead_time : Quantity[second], optional
            Fixed overhead after science frames end.

        Returns
        -------
        frames : int
            Number of frames that fit in the available time, floored to a
            whole number of coadds.
        data : Quantity[byte]
            Total raw data volume for those frames.
        data_compressed : Quantity[byte]
            Total compressed data volume.
        """
        buffered_time = (
            duration.to(u.s)
            - pre_overhead_time.to(u.s)
            - post_overhead_time.to(u.s)
            - self.additional_overhead_time.to(u.s)
        )
        buffered_time = max(0, buffered_time.value) * u.s

        # Find total number of frames that can fit into our observation time.
        if self.single_frame_time.to(u.s).value <= 0:
            frames = 0
        else:
            frames = max(
                0,
                math.floor(
                    (buffered_time / self.single_frame_time).to(u.dimensionless_unscaled).value
                ),
            )

        # Number of frames needs to be a multiple of coadds.
        if self.frames_per_coadd > 0:
            frames = max(0, frames - (frames % self.frames_per_coadd))

        data = frames * self.frame_bytes
        return (frames, data, data * self.compression_ratio)

    def solve_duration(
        self,
        integrations: int,
        pre_overhead_time: Quantity = 258.0 * u.s,
        post_overhead_time: Quantity = 102.0 * u.s,
    ):
        """Compute the total duration required to acquire a given number of frames.

        Parameters
        ----------
        integrations : int
            Desired number of frames.
        pre_overhead_time : Quantity[second], optional
            Fixed overhead before science frames begin.
        post_overhead_time : Quantity[second], optional
            Fixed overhead after science frames end.

        Returns
        -------
        duration : Quantity[second]
            Total wall-clock time needed, including all overheads.
        data : Quantity[byte]
            Total raw data volume for those frames.
        data_compressed : Quantity[byte]
            Total compressed data volume.
        """
        integrations = max(0, integrations)
        overhead_time = pre_overhead_time + post_overhead_time + self.additional_overhead_time
        data = integrations * self.frame_bytes

        if integrations == 0:
            duration = 0.0 * u.s
        else:
            duration = integrations * self.single_frame_time + overhead_time

        return (duration, data, data * self.compression_ratio)
