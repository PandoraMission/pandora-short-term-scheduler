# Standard library
import copy
import xml.etree.ElementTree as ET

# Third-party
import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta

# First-party/Local
from shortschedule.models import ObservationSequence
from shortschedule.scheduler import ScheduleProcessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sched():
    """Return a ScheduleProcessor instance without triggering __init__."""
    return ScheduleProcessor.__new__(ScheduleProcessor)


def _make_vda_seq(duration_sec, exposure_us, frames_per_coadd):
    """Minimal sequence with AcquireVisCamScienceData payload."""
    start = Time("2026-06-15T12:00:00", scale="utc")
    root = ET.Element("AcquireVisCamScienceData")
    ET.SubElement(root, "ExposureTime_us").text = str(exposure_us)
    ET.SubElement(root, "FramesPerCoadd").text = str(frames_per_coadd)
    ET.SubElement(root, "NumTotalFramesRequested").text = "0"
    return ObservationSequence(
        id="vda_seq",
        target="T",
        priority=1,
        start_time=start,
        stop_time=start + TimeDelta(duration_sec, format="sec"),
        ra=0.0,
        dec=0.0,
        payload_params={"AcquireVisCamScienceData": root},
    )


def _make_nirda_seq(
    duration_sec,
    roi_x=100,
    roi_y=256,
    sc_resets1=1,
    sc_resets2=1,
    sc_drop1=0,
    sc_drop2=0,
    sc_drop3=0,
    sc_read=5,
    sc_groups=3,
):
    """Minimal sequence with AcquireInfCamImages payload."""
    start = Time("2026-06-15T12:00:00", scale="utc")
    root = ET.Element("AcquireInfCamImages")
    ET.SubElement(root, "ROI_SizeX").text = str(roi_x)
    ET.SubElement(root, "ROI_SizeY").text = str(roi_y)
    ET.SubElement(root, "SC_Resets1").text = str(sc_resets1)
    ET.SubElement(root, "SC_Resets2").text = str(sc_resets2)
    ET.SubElement(root, "SC_DropFrames1").text = str(sc_drop1)
    ET.SubElement(root, "SC_DropFrames2").text = str(sc_drop2)
    ET.SubElement(root, "SC_DropFrames3").text = str(sc_drop3)
    ET.SubElement(root, "SC_ReadFrames").text = str(sc_read)
    ET.SubElement(root, "SC_Groups").text = str(sc_groups)
    ET.SubElement(root, "SC_Integrations").text = "0"
    return ObservationSequence(
        id="nirda_seq",
        target="T",
        priority=1,
        start_time=start,
        stop_time=start + TimeDelta(duration_sec, format="sec"),
        ra=0.0,
        dec=0.0,
        payload_params={"AcquireInfCamImages": root},
    )


# ---------------------------------------------------------------------------
# VDA (_update_VDA_integrations)
# ---------------------------------------------------------------------------


class TestUpdateVDAIntegrations:

    def test_default_overhead_reduces_frames_vs_no_overhead(self):
        """Default 260+60 s overhead should yield fewer frames than zero overhead."""
        seq = _make_vda_seq(
            duration_sec=1800,
            exposure_us=1_000_000,  # 1 s per frame
            frames_per_coadd=1,
        )
        sched = _sched()
        duration = seq.duration.to(u.us)

        seq_no = sched._update_VDA_integrations(
            copy.deepcopy(seq),
            duration,
            pre_sequence_overhead=0 * u.s,
            post_sequence_overhead=0 * u.s,
        )
        seq_with = sched._update_VDA_integrations(
            copy.deepcopy(seq),
            duration,
            pre_sequence_overhead=260 * u.s,
            post_sequence_overhead=60 * u.s,
        )

        frames_no = int(
            seq_no.get_payload_parameter(
                "AcquireVisCamScienceData", "NumTotalFramesRequested"
            )
        )
        frames_with = int(
            seq_with.get_payload_parameter(
                "AcquireVisCamScienceData", "NumTotalFramesRequested"
            )
        )
        assert frames_with < frames_no

    def test_exact_frame_count_with_no_overhead(self):
        """With zero overhead, all available time goes to frames."""
        # 1800 s duration, 1 s per frame, 1 frame per coadd → 1800 frames
        seq = _make_vda_seq(
            duration_sec=1800,
            exposure_us=1_000_000,
            frames_per_coadd=1,
        )
        sched = _sched()
        duration = seq.duration.to(u.us)
        seq_out = sched._update_VDA_integrations(
            seq,
            duration,
            pre_sequence_overhead=0 * u.s,
            post_sequence_overhead=0 * u.s,
        )
        frames = int(
            seq_out.get_payload_parameter(
                "AcquireVisCamScienceData", "NumTotalFramesRequested"
            )
        )
        assert frames == 1800

    def test_exact_frame_count_with_default_overhead(self):
        """1800 s - 320 s overhead = 1480 s effective → 1480 frames at 1 s/frame."""
        seq = _make_vda_seq(
            duration_sec=1800,
            exposure_us=1_000_000,
            frames_per_coadd=1,
        )
        sched = _sched()
        duration = seq.duration.to(u.us)
        seq_out = sched._update_VDA_integrations(
            seq,
            duration,
            pre_sequence_overhead=260 * u.s,
            post_sequence_overhead=60 * u.s,
        )
        frames = int(
            seq_out.get_payload_parameter(
                "AcquireVisCamScienceData", "NumTotalFramesRequested"
            )
        )
        assert frames == 1480

    def test_frames_multiple_of_frames_per_coadd(self):
        """NumTotalFramesRequested must always be a multiple of FramesPerCoadd."""
        seq = _make_vda_seq(
            duration_sec=1800,
            exposure_us=500_000,  # 0.5 s per frame
            frames_per_coadd=5,
        )
        sched = _sched()
        duration = seq.duration.to(u.us)
        seq_out = sched._update_VDA_integrations(
            seq,
            duration,
            pre_sequence_overhead=260 * u.s,
            post_sequence_overhead=60 * u.s,
        )
        frames = int(
            seq_out.get_payload_parameter(
                "AcquireVisCamScienceData", "NumTotalFramesRequested"
            )
        )
        assert frames % 5 == 0

    def test_sequence_shorter_than_overhead_yields_zero_frames(self):
        """Sequence shorter than 320 s overhead budget → 0 frames."""
        seq = _make_vda_seq(
            duration_sec=200,  # < 260+60 = 320 s
            exposure_us=1_000_000,
            frames_per_coadd=1,
        )
        sched = _sched()
        duration = seq.duration.to(u.us)
        seq_out = sched._update_VDA_integrations(
            seq,
            duration,
            pre_sequence_overhead=260 * u.s,
            post_sequence_overhead=60 * u.s,
        )
        frames = int(
            seq_out.get_payload_parameter(
                "AcquireVisCamScienceData", "NumTotalFramesRequested"
            )
        )
        assert frames == 0

    def test_custom_overhead_values_applied_correctly(self):
        """Custom overhead values should be used instead of defaults."""
        seq = _make_vda_seq(
            duration_sec=1800,
            exposure_us=1_000_000,
            frames_per_coadd=1,
        )
        sched = _sched()
        duration = seq.duration.to(u.us)

        # 100 s start + 50 s end = 150 s total → 1650 available
        seq_out = sched._update_VDA_integrations(
            seq,
            duration,
            pre_sequence_overhead=100 * u.s,
            post_sequence_overhead=50 * u.s,
        )
        frames = int(
            seq_out.get_payload_parameter(
                "AcquireVisCamScienceData", "NumTotalFramesRequested"
            )
        )
        assert frames == 1650


# ---------------------------------------------------------------------------
# NIRDA (_update_NIRDA_integrations)
# ---------------------------------------------------------------------------

# Reference parameter set
# ---
# ROI_SizeX=100, ROI_SizeY=256
# frame_time = (100+12)*(256+2)*1e-5 = 112*258*1e-5 = 0.28896 s
# SC_Resets1=1, SC_Resets2=1
# SC_DropFrames1=0, SC_DropFrames2=0, SC_DropFrames3=0
# SC_ReadFrames=5, SC_Groups=3
#
# NumIntegrations (hardcoded) = 1
# NumFramesTotal = 1 + 0 + 1*(0 + 2*(5+0) + 5 + 0) = 16
# integration_time = 16 * 0.28896 = 4.62336 s
#
# 1800 s, no overhead:  floor(1800 / 4.62336) = 389
# 1800 s, 318 s overhead: floor(1482 / 4.62336) = 320

_NIRDA_KWARGS = dict(
    roi_x=100,
    roi_y=256,
    sc_resets1=1,
    sc_resets2=1,
    sc_drop1=0,
    sc_drop2=0,
    sc_drop3=0,
    sc_read=5,
    sc_groups=3,
)


class TestUpdateNIRDAIntegrations:

    def test_default_overhead_reduces_integrations_vs_no_overhead(self):
        """Default 258+60 s overhead should yield fewer integrations than zero."""
        seq = _make_nirda_seq(duration_sec=1800, **_NIRDA_KWARGS)
        sched = _sched()
        duration = seq.duration.to(u.us)

        seq_no = sched._update_NIRDA_integrations(
            copy.deepcopy(seq),
            duration,
            pre_sequence_overhead=0 * u.s,
            post_sequence_overhead=0 * u.s,
        )
        seq_with = sched._update_NIRDA_integrations(
            copy.deepcopy(seq),
            duration,
            pre_sequence_overhead=258 * u.s,
            post_sequence_overhead=60 * u.s,
        )

        integ_no = int(
            seq_no.get_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations"
            )
        )
        integ_with = int(
            seq_with.get_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations"
            )
        )
        assert integ_with < integ_no

    def test_exact_integration_count_no_overhead(self):
        """With zero overhead, verifies exact SC_Integrations count."""
        seq = _make_nirda_seq(duration_sec=1800, **_NIRDA_KWARGS)
        sched = _sched()
        duration = seq.duration.to(u.us)
        seq_out = sched._update_NIRDA_integrations(
            seq,
            duration,
            pre_sequence_overhead=0 * u.s,
            post_sequence_overhead=0 * u.s,
        )
        integ = int(
            seq_out.get_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations"
            )
        )
        # integration_time = 16 * 0.28896 = 4.62336 s → floor(1800/4.62336) = 389
        frame_time = (100 + 12) * (256 + 2) * 1e-5
        num_frames_total = 1 + 0 + 1 * (0 + 2 * (5 + 0) + 5 + 0)
        integration_time = num_frames_total * frame_time
        expected = int(np.floor(1800 / integration_time))
        assert integ == expected

    def test_exact_integration_count_with_default_overhead(self):
        """1800 s - 318 s overhead → verified SC_Integrations count."""
        seq = _make_nirda_seq(duration_sec=1800, **_NIRDA_KWARGS)
        sched = _sched()
        duration = seq.duration.to(u.us)
        seq_out = sched._update_NIRDA_integrations(
            seq,
            duration,
            pre_sequence_overhead=258 * u.s,
            post_sequence_overhead=60 * u.s,
        )
        integ = int(
            seq_out.get_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations"
            )
        )
        frame_time = (100 + 12) * (256 + 2) * 1e-5
        num_frames_total = 1 + 0 + 1 * (0 + 2 * (5 + 0) + 5 + 0)
        integration_time = num_frames_total * frame_time
        effective_duration = 1800 - 258 - 60  # seconds
        expected = int(np.floor(effective_duration / integration_time))
        assert integ == expected

    def test_sequence_shorter_than_overhead_yields_zero_integrations(self):
        """Sequence shorter than 318 s overhead budget → 0 integrations."""
        seq = _make_nirda_seq(duration_sec=200, **_NIRDA_KWARGS)
        sched = _sched()
        duration = seq.duration.to(u.us)
        seq_out = sched._update_NIRDA_integrations(
            seq,
            duration,
            pre_sequence_overhead=258 * u.s,
            post_sequence_overhead=60 * u.s,
        )
        integ = int(
            seq_out.get_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations"
            )
        )
        assert integ == 0

    def test_custom_overhead_values_applied_correctly(self):
        """Custom overhead values should produce a deterministic count."""
        seq = _make_nirda_seq(duration_sec=1800, **_NIRDA_KWARGS)
        sched = _sched()
        duration = seq.duration.to(u.us)

        start_oh = 100
        end_oh = 50
        seq_out = sched._update_NIRDA_integrations(
            seq,
            duration,
            pre_sequence_overhead=start_oh * u.s,
            post_sequence_overhead=end_oh * u.s,
        )
        integ = int(
            seq_out.get_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations"
            )
        )
        frame_time = (100 + 12) * (256 + 2) * 1e-5
        num_frames_total = 1 + 0 + 1 * (0 + 2 * (5 + 0) + 5 + 0)
        integration_time = num_frames_total * frame_time
        effective = 1800 - start_oh - end_oh
        expected = int(np.floor(effective / integration_time))
        assert integ == expected
