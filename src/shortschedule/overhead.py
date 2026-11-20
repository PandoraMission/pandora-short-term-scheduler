"""Observation overhead timing helper.

This module implements ``OverheadTiming``, a dataclass that holds the
fixed pre- and post-observation overhead times applied to NIRDA and VISDA
observations. These overheads model the command sequences the MOC
generates for each observation based on SOC-provided calendars.

Notes
-----
The MOC places its commands around each calendar observation so that the
calendar ``Start`` is the NIRDA start (``PAYLOAD_ACQUIRE_INF_CAM_IMAGES``)
and the calendar ``Stop`` is when payload control of the attitude is turned
off (``PAYLOAD_CONTROL ... DISABLE``)::

    Start - 258 s   slew to the target
    Start -  50 s   settled on the target (SADA stopped)
    Start           NIRDA starts
    Start +   2 s   VISDA starts
    Start +  16 s   payload control on
         ...        observation
    Stop            payload control off; NIRDA and VISDA must be done
    Stop  +  22 s   halt imaging
    Stop  +  64 s   slew to idle
    Stop  + 124 s   at idle

The slew-in and the wind-down sit outside the calendar window, so the
detector overheads inside it are small: VISDA starts 2 s after NIRDA, and
both end at ``Stop``.

The spacecraft is settled on the target from 50 s before ``Start`` until it
slews away 64 s after ``Stop``; ``settled_before_start_time`` and
``on_target_after_stop_time`` carry those two times so the scheduler can
require the target to be inside its keepouts for them too.

Any value left as ``None`` at construction is filled in with the default
derived from the modelled command sequence; supplying an explicit
``Quantity`` overrides the corresponding default.

Every overhead constant the package uses lives in this module: the MOC
command times below, the VITL settling time, and the detectors' default
extra overhead.
"""

from dataclasses import dataclass

from astropy import units as u
from astropy.units import Quantity

# ---------------------------------------------------------------------------
# MOC command sequence around each calendar observation. This is the main
# place to change when the MOC changes what it commands. Pre-observation
# times are seconds relative to the calendar Start (negative = before), and
# post-observation times are seconds after the calendar Stop.
# ---------------------------------------------------------------------------
# Pre-observation
MOC_SLEW_TO_TARGET_S = -258.0  # GOTO_TARGET
MOC_MACRO65_PRE_S = -256.0  # MACRO_EXECUTE 65
# TODO: We generally say macro65 should run 25 seconds before a read but moc is only doing 20 here.
MOC_OPEN_SCIENCE_FILE_S = -236.0  # PAYLOAD_READ
MOC_SETTLED_S = -50.0  # SADA_MODE 1 INDEX STOP: settled on the target
MOC_NIRDA_START_S = 0.0  # PAYLOAD_ACQUIRE_INF_CAM_IMAGES = calendar Start
MOC_VISDA_START_S = 2.0  # PAYLOAD_ACQUIRE_VIS_CAM_SCIENCE_DATA
MOC_PAYLOAD_CONTROL_ON_S = 16.0  # PAYLOAD_CONTROL ENABLE
# Post-observation
MOC_PAYLOAD_CONTROL_OFF_S = 0.0  # PAYLOAD_CONTROL DISABLE = calendar Stop
MOC_MACRO65_POST_S = 2.0  # MACRO_EXECUTE 65
MOC_HALT_IMAGING_S = 22.0  # PAYLOAD_HALT_IMAGING_OR_COMMAND_SEQUENCE
MOC_SADA_TRACK_S = 24.0  # SADA_MODE 1 INDEX AUTO_TRACK_QEST
MOC_SLEW_TO_IDLE_S = 64.0  # GOTO_TARGET idle: leaves the target
MOC_CLOSE_SCIENCE_FILE_S = 82.0  # PAYLOAD_READ (empty)
MOC_AT_IDLE_S = 124.0  # at idle

# Detectors must be done by payload control off (the calendar Stop).
DETECTORS_END_S = MOC_PAYLOAD_CONTROL_OFF_S

# ---------------------------------------------------------------------------
# Other overheads
# ---------------------------------------------------------------------------
# Minimum VITL settling time NIRDA's first reset (reset_frames_1) must cover.
VITL_SETTLING_TIME = 60.0 * u.s
# Default extra fixed overhead a NirdaData/VisdaData subtracts on top of the
# pre/post overheads before fitting integrations or frames.
ADDITIONAL_OVERHEAD_TIME = 0.0 * u.s


@dataclass
class OverheadTiming:
    """Overhead times for NIRDA and VISDA, and the on-target time outside them.

    Any field left as ``None`` is populated in ``__post_init__`` via
    ``_update_derived`` with the default value derived from the modelled
    MOC command sequence. Supplying an explicit ``Quantity`` for a field
    overrides that default and leaves it untouched.

    Parameters
    ----------
    nirda_pre_overhead_time : Quantity[second] or None, default=None
        Time from the calendar start to the NIRDA start. If ``None``, 0 s:
        the MOC starts NIRDA at the calendar start.
    visda_pre_overhead_time : Quantity[second] or None, default=None
        Time from the calendar start to the VISDA start. If ``None``, 2 s:
        the VISDA acquisition command follows the NIRDA one.
    nirda_post_overhead_time : Quantity[second] or None, default=None
        Time between the last NIRDA integration and the calendar stop. If
        ``None``, 0 s: integrations end by payload control off, which is
        the calendar stop.
    visda_post_overhead_time : Quantity[second] or None, default=None
        Time between the last VISDA frame and the calendar stop. If
        ``None``, 0 s, as for NIRDA.
    settled_before_start_time : Quantity[second] or None, default=None
        How long the spacecraft is settled on the target before the
        calendar start. If ``None``, 50 s (SADA stop to the NIRDA start).
    on_target_after_stop_time : Quantity[second] or None, default=None
        How long the spacecraft stays on the target after the calendar
        stop. If ``None``, 64 s (payload control off to the slew to idle).
    """

    nirda_pre_overhead_time: Quantity | None = None
    visda_pre_overhead_time: Quantity | None = None
    nirda_post_overhead_time: Quantity | None = None
    visda_post_overhead_time: Quantity | None = None
    settled_before_start_time: Quantity | None = None
    on_target_after_stop_time: Quantity | None = None

    def __post_init__(self):
        self._update_derived()

    def _update_derived(self):
        """Fill in any unset times from the modelled MOC sequence."""
        # Pre: calendar Start to each detector's start, and how long the
        # spacecraft is settled before Start.
        if self.nirda_pre_overhead_time is None:
            self.nirda_pre_overhead_time = MOC_NIRDA_START_S * u.s  # 0 s
        if self.visda_pre_overhead_time is None:
            self.visda_pre_overhead_time = MOC_VISDA_START_S * u.s  # 2 s
        if self.settled_before_start_time is None:
            self.settled_before_start_time = -MOC_SETTLED_S * u.s  # 50 s

        # Post: each detector ends by payload control off (the calendar
        # Stop), and the spacecraft stays on the target until the slew to
        # idle.
        if self.nirda_post_overhead_time is None:
            self.nirda_post_overhead_time = DETECTORS_END_S * u.s  # 0 s
        if self.visda_post_overhead_time is None:
            self.visda_post_overhead_time = DETECTORS_END_S * u.s  # 0 s
        if self.on_target_after_stop_time is None:
            self.on_target_after_stop_time = MOC_SLEW_TO_IDLE_S * u.s  # 64 s
