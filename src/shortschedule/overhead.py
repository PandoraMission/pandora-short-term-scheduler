"""Observation overhead timing helper.

This module implements ``OverheadTiming``, a dataclass that holds the
fixed pre- and post-observation overhead times applied to NIRDA and VISDA
observations. These overheads model the command sequences the MOC
generates for each observation based on SOC-provided calendars.

Notes
-----
The overheads bracket every observation::

    pre-overhead (slew, macro, read, SADA, acquire) → observation → post-overhead (halt, SADA, slew to idle, close file)

The pre-overhead differs between NIRDA and VISDA because the VISDA science
acquisition command is issued slightly after the NIRDA imaging command, so
VISDA accrues a small additional offset. The post-overhead is identical for
both detectors.

Any value left as ``None`` at construction is filled in with the default
derived from the modelled command sequence; supplying an explicit
``Quantity`` overrides the corresponding default.
"""

from dataclasses import dataclass

from astropy import units as u
from astropy.units import Quantity


@dataclass
class OverheadTiming:
    """Pre- and post-observation overhead times for NIRDA and VISDA.

    Any field left as ``None`` is populated in ``__post_init__`` via
    ``_update_derived`` with the default value derived from the modelled
    MOC command sequence. Supplying an explicit ``Quantity`` for a field
    overrides that default and leaves it untouched.

    Parameters
    ----------
    nirda_pre_overhead_time : Quantity[second] or None, default=None
        Overhead before a NIRDA observation begins. If ``None``, derived
        from the pre-observation command sequence up to the NIRDA
        infrared-camera image acquisition.
    visda_pre_overhead_time : Quantity[second] or None, default=None
        Overhead before a VISDA observation begins. If ``None``, derived
        from the pre-observation command sequence up to the VISDA science
        data acquisition (slightly longer than the NIRDA pre-overhead).
    nirda_post_overhead_time : Quantity[second] or None, default=None
        Overhead after a NIRDA observation ends. If ``None``, derived from
        the post-observation command sequence (halt, SADA, slew to idle,
        close file).
    visda_post_overhead_time : Quantity[second] or None, default=None
        Overhead after a VISDA observation ends. If ``None``, derived from
        the same post-observation command sequence as NIRDA.
    """

    nirda_pre_overhead_time: Quantity | None = None
    visda_pre_overhead_time: Quantity | None = None
    nirda_post_overhead_time: Quantity | None = None
    visda_post_overhead_time: Quantity | None = None

    def __post_init__(self):
        self._update_derived()

    def _update_derived(self):
        """Fill in any unset overhead times from the modelled MOC sequence."""

        # This is the main place to change what happens during sequences
        # These are the commands that the MOC generates for each observation
        # based on SOC-provided calendars.

        time_delta = 0.0
        # COMMANDS:
        # GOTO_TARGET 
        time_delta += 2.0
        # MACRO_EXECUTE 65 
        time_delta += 20.0  # TODO: We generally say macro65 should run 25 seconds before a read but moc is only doing 20 here.
        # PAYLOAD_READ
        time_delta += 186.0
        # SADA_MODE 1 INDEX STOP
        time_delta += 50.0
        # PAYLOAD_ACQUIRE_INF_CAM_IMAGES 
        if self.nirda_pre_overhead_time is None:
            self.nirda_pre_overhead_time = time_delta * u.s
        time_delta += 2.0
        # PAYLOAD_ACQUIRE_VIS_CAM_SCIENCE_DATA
        if self.visda_pre_overhead_time is None:
            self.visda_pre_overhead_time = time_delta * u.s
        
        # ... 
        # Observation
        # ...

        # Post-Observation COMMANDS:
        time_delta = 0.0
        # PAYLOAD_HALT_IMAGING_OR_COMMAND_SEQUENCE
        time_delta += 2.0
        # SADA_MODE with SADA_NUM 1, INDEX AUTO_TRACK_QEST
        time_delta += 40.0
        # GOTO_TARGET Idle
        time_delta += 18.0
        # PAYLOAD_READ close file
        time_delta += 42.0 # Delta between end of sequence and when MOC considers the slew to idle to be complete.
        # End of Observation
        if self.nirda_post_overhead_time is None:
            self.nirda_post_overhead_time = time_delta * u.s
        if self.visda_post_overhead_time is None:
            self.visda_post_overhead_time = time_delta * u.s







