

# Class to determine overhead times for visda and nirda observations

import math
from dataclasses import dataclass, field

from astropy import units as u
from astropy.units import Quantity


@dataclass
class OverheadTiming:

    nirda_pre_overhead_time: Quantity = field(init=False)
    visda_pre_overhead_time: Quantity = field(init=False)
    nirda_post_overhead_time: Quantity = field(init=False)
    visda_post_overhead_time: Quantity = field(init=False)

    def __post_init__(self):
        self._update_derived()

    def _update_derived(self):

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
        self.nirda_pre_overhead_time = time_delta * u.s
        time_delta += 2.0
        # PAYLOAD_ACQUIRE_VIS_CAM_SCIENCE_DATA
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
        self.nirda_post_overhead_time = time_delta
        self.visda_post_overhead_time = time_delta







