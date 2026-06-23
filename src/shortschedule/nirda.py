"""NIRDA detector data helper functions.

This module implements ``NirdaData``, a dataclass that calculates NIRDA
frame times and data volumes for a single observation. The calculations
require:

- per-observation details such as read frames, drop frames, and resets.
- detector configuration such as ROI size, read time per pixel, and bytes
  per pixel.

Notes
-----
NIRDA reads out a 2-D ROI up a ramp. Each integration consists of::

    reset frames → drop_frames_1 → groups x (read_frames + drop_frames_2)
                                 → drop_frames_3

Two integration lengths are tracked: the *first* integration (which uses
``reset_frames_1``) and all *subsequent* integrations (which use the
shorter ``reset_frames_2``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from astropy import units as u
from astropy.units import Quantity

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .overhead import OverheadTiming

@dataclass
class NirdaData:
    """NIRDA timing and data-volume parameters for a single observation.

    All timing and data results are computed automatically in
    ``__post_init__`` via ``_update_derived``.  Derived attributes
    (``init=False``) should be treated as read-only after construction.

    Parameters
    ----------
    reset_frames_1 : int
        Reset frames at the start of the *first* integration.
    reset_frames_2 : int
        Reset frames at the start of all *subsequent* integrations.
    drop_frames_1 : int
        Drop frames between the initial reset and the first group.
    drop_frames_2 : int
        Drop frames between consecutive groups.
    drop_frames_3 : int
        Drop frames after the last group.
    read_frames : int
        Non-destructive read frames per group.
    groups : int
        Number of groups per integration.
    average_groups : bool
        Whether groups are averaged on-board before downlink. Affects
        ``integration_data`` but not timing.
    roi_x_size : int
        Width of the science ROI in pixels, excluding buffer columns.
    roi_y_size : int
        Height of the science ROI in pixels, excluding buffer rows.
    roi_x_start : int
        Starting column of the ROI on the full detector (pixels).
    roi_y_start : int
        Starting row of the ROI on the full detector (pixels).
    roi_x_buffer_pixels : int
        Extra columns read per row for detector reference pixels.
    roi_y_buffer_pixels : int
        Extra rows read per frame for detector reference pixels.
    read_time_per_pixel : Quantity[second]
        Detector clock period per pixel during readout.
    bytes_per_pixel : Quantity[byte]
        Raw storage cost per pixel before compression.
    dropped_integrations : int
        Number of integrations that are dropped as a buffer.
    compression_ratio : float
        Effective on-board compression factor applied to raw data.
        This is an empirical parameter.
    global_reset_method : str
        Per-integration global-reset overhead model. 
        - ``'off'`` (no outside roi reset; no overhead)
        - ``'global'`` (outside roi are quickly resetl; a small fixed overhead)
        - ``'line_by_line'`` (``global_reset_lbl_rows`` rows at ``global_reset_lbl_time_per_row`` each).
    global_reset_lbl_rows : int
        Number of reset rows used by the ``'line_by_line'`` method.
    global_reset_lbl_time_per_row : Quantity[second]
        Reset time per row used by the ``'line_by_line'`` method.
    additional_overhead_time : Quantity[second]
        Extra fixed overhead beyond ``pre_overhead_time`` and
        ``post_overhead_time`` subtracted before scheduling integrations.

    Attributes
    ----------
    pixels_per_frame : int
        Total pixels clocked out per frame, including buffer pixels.
    single_frame_time : Quantity[second]
        Wall-clock duration of one detector frame.
    reset_frame_time : Quantity[second]
        Reset-frame duration. Derived as equal to ``single_frame_time``.
    first_integration_saved_frames : int
        Frames written to memory for the first integration.
    other_integration_saved_frames : int
        Frames written to memory for each subsequent integration.
    first_integration_time : Quantity[second]
        Wall-clock duration of the first integration.
    other_integration_time : Quantity[second]
        Wall-clock duration of each subsequent integration.
    dropped_integration_time : Quantity[second]
        Wall-clock duration of a dropped (non-science) integration.
    integration_data : Quantity[byte]
        Data volume per science integration after on-board averaging.
    """

    # NIRDA observation configurations
    reset_frames_1: int = 50
    reset_frames_2: int = 1
    drop_frames_1: int = 1
    drop_frames_2: int = 16
    drop_frames_3: int = 0
    read_frames: int = 4
    groups: int = 6
    average_groups: bool = True
    roi_x_size: int = 80
    roi_y_size: int = 250
    roi_x_start: int = 1737
    roi_y_start: int = 962

    # NIRDA detector configurations
    roi_x_buffer_pixels: int = 12
    roi_y_buffer_pixels: int = 2
    read_time_per_pixel: Quantity = 1.0e-5 * u.s
    bytes_per_pixel: Quantity = 2 * u.byte
    dropped_integrations: int = 0
    compression_ratio: float = 0.8
    global_reset_method: str = 'off'
    global_reset_lbl_rows: int = 256
    global_reset_lbl_time_per_row: Quantity = 10.0e-6 * u.s
    additional_overhead_time: Quantity = 0 * u.s

    # Derived attributes: computed in _update_derived, not constructor arguments
    pixels_per_frame: int = field(init=False)
    single_frame_time: Quantity = field(init=False)
    first_integration_saved_frames: int = field(init=False)
    other_integration_saved_frames: int = field(init=False)
    first_integration_time: Quantity = field(init=False)
    other_integration_time: Quantity = field(init=False)
    dropped_integration_time: Quantity = field(init=False)
    integration_data: Quantity = field(init=False)
    reset_frame_time: Quantity = field(init=False)

    def __post_init__(self):
        self._update_derived()

    def _update_derived(self):
        """Recompute all derived timing and data-volume attributes."""
        common_frames = max(
            0,
            self.drop_frames_1
            + (self.groups - 1) * self.drop_frames_2
            + self.drop_frames_3
            + self.groups * self.read_frames
        )

        self.pixels_per_frame = max(
            0,
            (self.roi_x_size + self.roi_x_buffer_pixels)
            * (self.roi_y_size + self.roi_y_buffer_pixels),
        )

        self.single_frame_time = max(
            0, (self.pixels_per_frame * self.read_time_per_pixel).to(u.s).value
        ) * u.s

        # Assume reset frames take the same as regular frames.
        self.reset_frame_time = self.single_frame_time

        # Determine common integration time (part that is the same for 1st and subsequent integrations)
        # There is additional per-integration overhead if a global reset is used.
        self.global_reset_method = self.global_reset_method.lower().strip()
        global_reset_overhead = 0.0 * u.s
        if self.global_reset_method not in ('off', 'global', 'line_by_line'):
            raise ValueError(f"NIRDA: Unknown global reset method requested: {self.global_reset_method}")
        elif self.global_reset_method == 'global':
            # Global reset requires a "negligible compared to frame time" overhead.
            # This value is not real but just something small.
            global_reset_overhead = 1.0e-6 * u.s
        elif self.global_reset_method == 'line_by_line':
            # Line by line method requires 10us per line.
            global_reset_overhead = self.global_reset_lbl_time_per_row * self.global_reset_lbl_rows

        # Common integration time in seconds (part shared by 1st and subsequent
        # integrations). Kept as a plain float here and converted to a Quantity
        # below, matching the float-then-``* u.s`` style used elsewhere.
        common_integration_time = max(
            0,
            self.single_frame_time.to(u.s).value * common_frames
            + global_reset_overhead.to(u.s).value
        )

        self.first_integration_time = max(
            0,
            common_integration_time
            + self.reset_frame_time.to(u.s).value * self.reset_frames_1,
        ) * u.s

        self.other_integration_time = max(
            0,
            common_integration_time
            + self.reset_frame_time.to(u.s).value * self.reset_frames_2,
        ) * u.s

        # If we drop any integrations then the duration of those drops will always be equal to the "other"
        # integration time.
        self.dropped_integration_time = self.other_integration_time

        bytes_per_frame = max(
            0, (self.bytes_per_pixel * self.pixels_per_frame).to(u.byte).value
        ) * u.byte

        if self.average_groups:
            self.first_integration_saved_frames = self.groups
            self.other_integration_saved_frames = self.groups
            self.integration_data = bytes_per_frame * self.groups
        else:
            self.first_integration_saved_frames = self.groups * self.read_frames
            self.other_integration_saved_frames = self.groups * self.read_frames
            self.integration_data = bytes_per_frame * self.groups * self.read_frames

        self.integration_data = max(
            0, self.integration_data.to(u.byte).value
        ) * u.byte

    def update_for_vitl(self, vitl_settling_time: Quantity):
        """Adjust ``reset_frames_1`` to cover a VITL settling time and recompute.

        Sets ``reset_frames_1`` to the minimum number of reset frames whose
        total duration is at least ``vitl_settling_time``, then calls
        ``_update_derived`` to propagate the change.

        Parameters
        ----------
        vitl_settling_time : Quantity[second]
            Minimum time required for VITL detector settling.
        """
        # reset_frame_time is derived (equal to single_frame_time) during
        # __post_init__, so it holds the actual frame duration here.
        
        # TODO: Open Question! Do we add the global reset time to this?

        if self.reset_frame_time.to(u.s).value == 0.0:
            # Have at least 2 reset1
            self.reset_frames_1 = 2
        else:
            self.reset_frames_1 = max(
                1,
                math.ceil((vitl_settling_time / self.reset_frame_time).decompose().value),
            )

        # TODO: Open Question! Do we add the global reset time to this?

        self._update_derived()

    def solve_integrations(
        self,
        duration: Quantity,
        overhead: OverheadTiming = None
    ):
        """Compute the number of integrations that fit within a duration.

        Parameters
        ----------
        duration : Quantity[second]
            Total available observation time.
        overhead : OverheadTiming, default=None
            Overhead timings for nirda and visda.
            If None, then use default overheads.

        Returns
        -------
        integrations : int
            Number of science integrations that fit in the available time.
        data : Quantity[byte]
            Total data volume for those integrations.
        data_compressed : Quantity[byte]
            Total compressed data volume.
        """

        if overhead is None:
            # Use default overheads
            from .overhead import OverheadTiming
            overhead = OverheadTiming()

        buffered_time = (
            duration
            - overhead.nirda_pre_overhead_time
            - overhead.nirda_post_overhead_time
            - self.additional_overhead_time
        )
        buffered_time = max(0, buffered_time.value) * u.s

        integrations = 0
        if self.first_integration_time <= buffered_time:
            integrations += 1
            buffered_time -= self.first_integration_time

            if self.other_integration_time <= buffered_time:
                other_integrations = math.floor(
                    (buffered_time / self.other_integration_time).value
                )
                integrations += other_integrations

        if self.dropped_integrations > 0:
            integrations = max(0, integrations - self.dropped_integrations)

        data = integrations * self.integration_data
        return (integrations, data, data * self.compression_ratio)

    def solve_duration(
        self,
        integrations: int,
        overhead: OverheadTiming = None
    ):
        """Compute the total duration required to acquire a given number of integrations.

        Parameters
        ----------
        integrations : int
            Desired number of science integrations.
        overhead : OverheadTiming, default=None
            Overhead timings for nirda and visda.
            If None, then use default overheads.

        Returns
        -------
        duration : Quantity[second]
            Total wall-clock time needed, including all overheads.
        data : Quantity[byte]
            Total data volume for those integrations.
        data_compressed : Quantity[byte]
            Total compressed data volume.

        Notes
        -----
        ``dropped_integrations`` is intentionally ignored here: when the
        caller requests a specific integration count, that count is taken
        as-is rather than changed to compensate for drops.
        """
        integrations = max(0, integrations)
        
        if overhead is None:
            # Use default overheads
            from .overhead import OverheadTiming
            overhead = OverheadTiming()
        
        overhead_time = overhead.nirda_pre_overhead_time + overhead.nirda_post_overhead_time + self.additional_overhead_time
        data = integrations * self.integration_data

        if integrations == 0:
            duration = 0.0 * u.s
        elif integrations == 1:
            duration = self.first_integration_time + overhead_time
        else:
            duration = (
                self.first_integration_time
                + self.other_integration_time * (integrations - 1)
                + overhead_time
            )

        return (duration, data, data * self.compression_ratio)
