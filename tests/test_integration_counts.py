# Standard library
import copy
import unittest.mock as mock
import warnings
import xml.etree.ElementTree as ET

# Third-party
import pytest
from astropy import units as u
from astropy.time import Time, TimeDelta

# First-party/Local
from shortschedule.models import (
    ObservationSequence,
    ScienceCalendar,
    Visit,
)
from shortschedule.nirda import NirdaData
from shortschedule.overhead import OverheadTiming
from shortschedule.scheduler import ScheduleProcessor
from shortschedule.visda import VisdaData

# ---------------------------------------------------------------------------
# Helpers
#
# Sequences are built from the NirdaData/VisdaData classes: a data object is
# constructed with the desired configuration, then its payload XML is emitted
# from the class's CONFIG_SPEC. This keeps the tests in lock-step with the
# detector models the scheduler uses.
# ---------------------------------------------------------------------------


def _sched():
    """Return a ScheduleProcessor instance without triggering __init__."""
    return ScheduleProcessor.__new__(ScheduleProcessor)


def _payload_from_config(data_obj, extra_tags=None):
    """Build a payload XML element for *data_obj* from its CONFIG_SPEC.

    Each config field is written to its mapped XML tag using the class's
    own ``to_xml`` converter, so the payload always matches the data model.
    *extra_tags* adds any output-only tags (e.g. the integration count the
    scheduler will overwrite).
    """
    root = ET.Element(data_obj.PAYLOAD_SECTION)
    config = data_obj.get_config()
    for field, (tag, _from_xml, to_xml) in data_obj.CONFIG_SPEC.items():
        ET.SubElement(root, tag).text = to_xml(config[field])
    for tag, text in (extra_tags or {}).items():
        ET.SubElement(root, tag).text = text
    return root


def _visda_from_kwargs(exposure_us, frames_per_coadd):
    """Build a VisdaData for the test's VDA parameters.

    """
    return VisdaData(
        exposure_time_s=exposure_us * u.us,
        frames_per_coadd=frames_per_coadd,
    )


def _make_vda_seq(duration_sec, exposure_us, frames_per_coadd):
    """Sequence whose AcquireVisCamScienceData payload is built from VisdaData."""
    start = Time("2026-06-15T12:00:00", scale="utc")
    vd = _visda_from_kwargs(exposure_us, frames_per_coadd)
    root = _payload_from_config(vd, {"NumTotalFramesRequested": "0"})
    return ObservationSequence(
        id="vda_seq",
        target="T",
        priority=1,
        start_time=start,
        stop_time=start + TimeDelta(duration_sec, format="sec"),
        ra=0.0,
        dec=0.0,
        payload_params={vd.PAYLOAD_SECTION: root},
    )


# Maps the legacy keyword names used by these tests to NirdaData fields.
_NIRDA_KW_TO_FIELD = {
    "roi_x": "roi_x_size",
    "roi_y": "roi_y_size",
    "sc_resets1": "reset_frames_1",
    "sc_resets2": "reset_frames_2",
    "sc_drop1": "drop_frames_1",
    "sc_drop2": "drop_frames_2",
    "sc_drop3": "drop_frames_3",
    "sc_read": "read_frames",
    "sc_groups": "groups",
}


def _nirda_from_kwargs(
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
    """Build a NirdaData from the test's NIRDA parameters."""
    kwargs = dict(
        roi_x=roi_x,
        roi_y=roi_y,
        sc_resets1=sc_resets1,
        sc_resets2=sc_resets2,
        sc_drop1=sc_drop1,
        sc_drop2=sc_drop2,
        sc_drop3=sc_drop3,
        sc_read=sc_read,
        sc_groups=sc_groups,
    )
    fields = {_NIRDA_KW_TO_FIELD[k]: v for k, v in kwargs.items()}
    return NirdaData(**fields)


def _make_nirda_seq(duration_sec, **kwargs):
    """Sequence whose AcquireInfCamImages payload is built from NirdaData."""
    start = Time("2026-06-15T12:00:00", scale="utc")
    nd = _nirda_from_kwargs(**kwargs)
    root = _payload_from_config(nd, {"SC_Integrations": "0"})
    return ObservationSequence(
        id="nirda_seq",
        target="T",
        priority=1,
        start_time=start,
        stop_time=start + TimeDelta(duration_sec, format="sec"),
        ra=0.0,
        dec=0.0,
        payload_params={nd.PAYLOAD_SECTION: root},
    )


def _nirda_overhead(pre_sec=0.0, post_sec=0.0):
    """OverheadTiming with the given NIRDA pre/post overheads (seconds)."""
    return OverheadTiming(
        nirda_pre_overhead_time=pre_sec * u.s,
        nirda_post_overhead_time=post_sec * u.s,
    )


def _visda_overhead(pre_sec=0.0, post_sec=0.0):
    """OverheadTiming with the given VISDA pre/post overheads (seconds)."""
    return OverheadTiming(
        visda_pre_overhead_time=pre_sec * u.s,
        visda_post_overhead_time=post_sec * u.s,
    )


def _overhead(pre=0 * u.s, post=0 * u.s):
    """OverheadTiming applying the same pre/post to both detectors.

    Lets a test pass one overhead to either the VDA or NIRDA update without
    caring which detector reads which field.
    """
    return OverheadTiming(
        visda_pre_overhead_time=pre,
        visda_post_overhead_time=post,
        nirda_pre_overhead_time=pre,
        nirda_post_overhead_time=post,
    )


def _expected_vda_frames(
    duration_sec,
    *,
    exposure_us,
    frames_per_coadd,
    pre_sequence_overhead_sec=0,
    post_sequence_overhead_sec=0,
):
    """Expected NumTotalFramesRequested, computed via VisdaData."""
    vd = _visda_from_kwargs(exposure_us, frames_per_coadd)
    frames, _, _ = vd.solve_integrations(
        duration_sec * u.s,
        _visda_overhead(pre_sequence_overhead_sec, post_sequence_overhead_sec),
    )
    return int(frames)


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
        duration = seq.duration

        seq_no = sched._update_VDA_integrations(
            copy.deepcopy(seq),
            duration,
            overhead=_overhead(),
        )
        seq_with = sched._update_VDA_integrations(
            copy.deepcopy(seq),
            duration,
            overhead=_overhead(260 * u.s, 60 * u.s),
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
        duration = seq.duration
        seq_out = sched._update_VDA_integrations(
            seq,
            duration,
            overhead=_overhead(),
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
        duration = seq.duration
        seq_out = sched._update_VDA_integrations(
            seq,
            duration,
            overhead=_overhead(260 * u.s, 60 * u.s),
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
        duration = seq.duration
        seq_out = sched._update_VDA_integrations(
            seq,
            duration,
            overhead=_overhead(260 * u.s, 60 * u.s),
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
        duration = seq.duration
        seq_out = sched._update_VDA_integrations(
            seq,
            duration,
            overhead=_overhead(260 * u.s, 60 * u.s),
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
        duration = seq.duration

        # 100 s start + 50 s end = 150 s total → 1650 available
        seq_out = sched._update_VDA_integrations(
            seq,
            duration,
            overhead=_overhead(100 * u.s, 50 * u.s),
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
# SC_Resets1=2, SC_Resets2=1
# SC_DropFrames1=0, SC_DropFrames2=0, SC_DropFrames3=0
# SC_ReadFrames=5, SC_Groups=3
#
# NumFramesBase = 0 + 2*(5+0) + 5 + 0 = 15
# first_integration_time = (15 + 2) * 0.28896 = 4.91232 s
# other_integration_time = (15 + 1) * 0.28896 = 4.62336 s

_NIRDA_KWARGS = dict(
    roi_x=100,
    roi_y=256,
    sc_resets1=2,
    sc_resets2=1,
    sc_drop1=0,
    sc_drop2=0,
    sc_drop3=0,
    sc_read=5,
    sc_groups=3,
)


def _expected_nirda_integrations(
    duration_sec,
    *,
    pre_sequence_overhead_sec=0,
    post_sequence_overhead_sec=0,
    **kwargs,
):
    """Expected SC_Integrations, computed via NirdaData."""
    nd = _nirda_from_kwargs(**kwargs)
    integrations, _, _ = nd.solve_integrations(
        duration_sec * u.s,
        _nirda_overhead(pre_sequence_overhead_sec, post_sequence_overhead_sec),
    )
    return int(integrations)


class TestUpdateNIRDAIntegrations:

    def test_default_overhead_reduces_integrations_vs_no_overhead(self):
        """Default 258+60 s overhead should yield fewer integrations than zero."""
        seq = _make_nirda_seq(duration_sec=1800, **_NIRDA_KWARGS)
        sched = _sched()
        duration = seq.duration

        seq_no = sched._update_NIRDA_integrations(
            copy.deepcopy(seq),
            duration,
            overhead=_overhead(),
        )
        seq_with = sched._update_NIRDA_integrations(
            copy.deepcopy(seq),
            duration,
            overhead=_overhead(258 * u.s, 60 * u.s),
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
        duration = seq.duration
        seq_out = sched._update_NIRDA_integrations(
            seq,
            duration,
            overhead=_overhead(),
        )
        integ = int(
            seq_out.get_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations"
            )
        )
        expected = _expected_nirda_integrations(1800, **_NIRDA_KWARGS)
        assert integ == expected

    def test_exact_integration_count_with_default_overhead(self):
        """1800 s - 318 s overhead → verified SC_Integrations count."""
        seq = _make_nirda_seq(duration_sec=1800, **_NIRDA_KWARGS)
        sched = _sched()
        duration = seq.duration
        seq_out = sched._update_NIRDA_integrations(
            seq,
            duration,
            overhead=_overhead(258 * u.s, 60 * u.s),
        )
        integ = int(
            seq_out.get_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations"
            )
        )
        expected = _expected_nirda_integrations(
            1800,
            pre_sequence_overhead_sec=258,
            post_sequence_overhead_sec=60,
            **_NIRDA_KWARGS,
        )
        assert integ == expected

    def test_sequence_shorter_than_overhead_yields_zero_integrations(self):
        """Sequence shorter than 318 s overhead budget → 0 integrations."""
        seq = _make_nirda_seq(duration_sec=200, **_NIRDA_KWARGS)
        sched = _sched()
        duration = seq.duration
        seq_out = sched._update_NIRDA_integrations(
            seq,
            duration,
            overhead=_overhead(258 * u.s, 60 * u.s),
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
        duration = seq.duration

        start_oh = 100
        end_oh = 50
        seq_out = sched._update_NIRDA_integrations(
            seq,
            duration,
            overhead=_overhead(start_oh * u.s, end_oh * u.s),
        )
        integ = int(
            seq_out.get_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations"
            )
        )
        expected = _expected_nirda_integrations(
            1800,
            pre_sequence_overhead_sec=start_oh,
            post_sequence_overhead_sec=end_oh,
            **_NIRDA_KWARGS,
        )
        assert integ == expected

    def test_second_integration_fits_at_boundary(self):
        """Extending a window by one ``other`` integration adds exactly one.

        NirdaData.solve_integrations fits the second integration as soon as
        the remaining time reaches the "other" integration time. The check is
        framed as a delta so it stays valid regardless of any fixed offset
        (e.g. dropped_integrations) applied to the reported count.
        """
        nd = _nirda_from_kwargs(**_NIRDA_KWARGS)
        first_integration_time = nd.first_integration_time.to(u.s).value
        other_integration_time = nd.other_integration_time.to(u.s).value
        sched = _sched()

        def _count(duration_sec):
            seq = _make_nirda_seq(duration_sec=duration_sec, **_NIRDA_KWARGS)
            out = sched._update_NIRDA_integrations(
                seq, seq.duration, overhead=_overhead()
            )
            return int(
                out.get_payload_parameter(
                    "AcquireInfCamImages", "SC_Integrations"
                )
            )

        # Small positive margins keep both cases off the floating-point edge.
        one = _count(first_integration_time + 1e-3)
        two = _count(first_integration_time + other_integration_time + 1e-3)
        assert two == one + 1


# ---------------------------------------------------------------------------
# _update_payload_parameters_sequence (orchestrating wrapper)
# ---------------------------------------------------------------------------


def _sched_with_overhead(
    vda_pre=0 * u.s,
    vda_post=0 * u.s,
    nirda_pre=0 * u.s,
    nirda_post=0 * u.s,
):
    """ScheduleProcessor instance whose only state is its OverheadTiming."""
    sched = ScheduleProcessor.__new__(ScheduleProcessor)
    sched.overhead = OverheadTiming(
        visda_pre_overhead_time=vda_pre,
        visda_post_overhead_time=vda_post,
        nirda_pre_overhead_time=nirda_pre,
        nirda_post_overhead_time=nirda_post,
    )
    return sched


class TestUpdatePayloadParametersSequence:
    """Tests for the _update_payload_parameters_sequence orchestrator.

    These tests exercise the full VDA/NIRDA path through the wrapper,
    verifying that the *instance* overhead values are applied correctly.
    """

    # ---- VDA path --------------------------------------------------------

    def test_vda_overhead_reduces_frames(self):
        """Instance overhead should reduce NumTotalFramesRequested vs zero overhead.

        Verifies requirement (1): overhead reduces the computed frames.
        """
        seq_zero = _make_vda_seq(
            duration_sec=1800,
            exposure_us=1_000_000,
            frames_per_coadd=1,
        )
        seq_with = copy.deepcopy(seq_zero)

        sched_zero = _sched_with_overhead()
        sched_with = _sched_with_overhead(vda_pre=260 * u.s, vda_post=60 * u.s)

        seq_zero = sched_zero._update_payload_parameters_sequence(seq_zero)
        seq_with = sched_with._update_payload_parameters_sequence(seq_with)

        frames_zero = int(
            seq_zero.get_payload_parameter(
                "AcquireVisCamScienceData", "NumTotalFramesRequested"
            )
        )
        frames_with = int(
            seq_with.get_payload_parameter(
                "AcquireVisCamScienceData", "NumTotalFramesRequested"
            )
        )
        assert frames_with < frames_zero

    def test_vda_exact_frame_count_via_wrapper(self):
        """1800 s - 320 s overhead = 1480 frames at 1 s/frame (via wrapper)."""
        seq = _make_vda_seq(
            duration_sec=1800,
            exposure_us=1_000_000,
            frames_per_coadd=1,
        )
        sched = _sched_with_overhead(vda_pre=260 * u.s, vda_post=60 * u.s)
        seq_out = sched._update_payload_parameters_sequence(seq)
        frames = int(
            seq_out.get_payload_parameter(
                "AcquireVisCamScienceData", "NumTotalFramesRequested"
            )
        )
        assert frames == 1480

    def test_vda_overhead_exceeds_duration_yields_zero_frames(self):
        """Sequence shorter than overhead budget → 0 frames (via wrapper).

        Verifies requirement (2): integrations/frames become 0 when
        overhead >= sequence duration.
        """
        seq = _make_vda_seq(
            duration_sec=200,  # < 260+60=320 s overhead
            exposure_us=1_000_000,
            frames_per_coadd=1,
        )
        sched = _sched_with_overhead(vda_pre=260 * u.s, vda_post=60 * u.s)
        seq_out = sched._update_payload_parameters_sequence(seq)
        frames = int(
            seq_out.get_payload_parameter(
                "AcquireVisCamScienceData", "NumTotalFramesRequested"
            )
        )
        assert frames == 0

    def test_vda_overhead_equals_duration_yields_zero_frames(self):
        """Sequence duration exactly equal to overhead → 0 frames."""
        seq = _make_vda_seq(
            duration_sec=320,  # exactly 260+60 s
            exposure_us=1_000_000,
            frames_per_coadd=1,
        )
        sched = _sched_with_overhead(vda_pre=260 * u.s, vda_post=60 * u.s)
        seq_out = sched._update_payload_parameters_sequence(seq)
        frames = int(
            seq_out.get_payload_parameter(
                "AcquireVisCamScienceData", "NumTotalFramesRequested"
            )
        )
        assert frames == 0

    # ---- NIRDA path ------------------------------------------------------

    def test_nirda_overhead_reduces_integrations(self):
        """Instance NIRDA overhead should reduce SC_Integrations vs zero overhead.

        Verifies requirement (1) for the NIRDA path.
        """
        seq_zero = _make_nirda_seq(duration_sec=1800, **_NIRDA_KWARGS)
        seq_with = copy.deepcopy(seq_zero)

        sched_zero = _sched_with_overhead()
        sched_with = _sched_with_overhead(
            nirda_pre=258 * u.s, nirda_post=60 * u.s
        )

        seq_zero = sched_zero._update_payload_parameters_sequence(seq_zero)
        seq_with = sched_with._update_payload_parameters_sequence(seq_with)

        integ_zero = int(
            seq_zero.get_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations"
            )
        )
        integ_with = int(
            seq_with.get_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations"
            )
        )
        assert integ_with < integ_zero

    def test_nirda_exact_integration_count_via_wrapper(self):
        """1800 s - 318 s overhead: deterministic SC_Integrations (via wrapper)."""
        seq = _make_nirda_seq(duration_sec=1800, **_NIRDA_KWARGS)
        sched = _sched_with_overhead(nirda_pre=258 * u.s, nirda_post=60 * u.s)
        seq_out = sched._update_payload_parameters_sequence(seq)
        integ = int(
            seq_out.get_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations"
            )
        )
        expected = _expected_nirda_integrations(
            1800,
            pre_sequence_overhead_sec=258,
            post_sequence_overhead_sec=60,
            **_NIRDA_KWARGS,
        )
        assert integ == expected

    def test_nirda_second_integration_fits_at_boundary_via_wrapper(self):
        """Wrapper adds exactly one integration for one more ``other`` window."""
        nd = _nirda_from_kwargs(**_NIRDA_KWARGS)
        first_integration_time = nd.first_integration_time.to(u.s).value
        other_integration_time = nd.other_integration_time.to(u.s).value
        sched = _sched_with_overhead()

        def _count(duration_sec):
            seq = _make_nirda_seq(duration_sec=duration_sec, **_NIRDA_KWARGS)
            out = sched._update_payload_parameters_sequence(seq)
            return int(
                out.get_payload_parameter(
                    "AcquireInfCamImages", "SC_Integrations"
                )
            )

        one = _count(first_integration_time + 1e-3)
        two = _count(first_integration_time + other_integration_time + 1e-3)
        assert two == one + 1

    def test_nirda_overhead_exceeds_duration_yields_zero_integrations(self):
        """Sequence shorter than NIRDA overhead budget → 0 integrations (via wrapper).

        Verifies requirement (2) for the NIRDA path.
        """
        seq = _make_nirda_seq(
            duration_sec=200,  # < 258+60=318 s overhead
            **_NIRDA_KWARGS,
        )
        sched = _sched_with_overhead(nirda_pre=258 * u.s, nirda_post=60 * u.s)
        seq_out = sched._update_payload_parameters_sequence(seq)
        integ = int(
            seq_out.get_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations"
            )
        )
        assert integ == 0

    def test_nirda_overhead_equals_duration_yields_zero_integrations(self):
        """Sequence duration exactly equal to NIRDA overhead → 0 integrations."""
        seq = _make_nirda_seq(
            duration_sec=318,  # exactly 258+60 s
            **_NIRDA_KWARGS,
        )
        sched = _sched_with_overhead(nirda_pre=258 * u.s, nirda_post=60 * u.s)
        seq_out = sched._update_payload_parameters_sequence(seq)
        integ = int(
            seq_out.get_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations"
            )
        )
        assert integ == 0


# ---------------------------------------------------------------------------
# Per-priority parameter overrides
# ---------------------------------------------------------------------------


class TestNirdaParameterOverride:
    """override_fields replaces observation values with NirdaData defaults."""

    def test_overridden_fields_written_back_as_defaults(self):
        """Listed fields are replaced by class defaults and written to XML."""
        defaults = NirdaData()
        # Build a sequence whose drop frames differ from the class defaults.
        seq = _make_nirda_seq(
            duration_sec=1800,
            sc_drop1=5,
            sc_drop3=7,
            **{
                k: v
                for k, v in _NIRDA_KWARGS.items()
                if k not in ("sc_drop1", "sc_drop3")
            },
        )
        sched = _sched()
        seq_out = sched._update_NIRDA_integrations(
            seq,
            seq.duration,
            overhead=_overhead(),
            override_fields=["drop_frames_1", "drop_frames_3"],
        )

        drop1 = seq_out.get_payload_parameter(
            "AcquireInfCamImages", "SC_DropFrames1"
        )
        drop3 = seq_out.get_payload_parameter(
            "AcquireInfCamImages", "SC_DropFrames3"
        )
        assert int(drop1) == defaults.drop_frames_1
        assert int(drop3) == defaults.drop_frames_3

    def test_non_overridden_fields_untouched(self):
        """Fields not listed keep the observation's original values."""
        seq = _make_nirda_seq(
            duration_sec=1800,
            sc_drop2=9,
            **{k: v for k, v in _NIRDA_KWARGS.items() if k != "sc_drop2"},
        )
        sched = _sched()
        seq_out = sched._update_NIRDA_integrations(
            seq,
            seq.duration,
            overhead=_overhead(),
            override_fields=["drop_frames_1"],
        )
        drop2 = seq_out.get_payload_parameter(
            "AcquireInfCamImages", "SC_DropFrames2"
        )
        assert int(drop2) == 9

    def test_override_changes_integration_count(self):
        """Overriding reset frames changes the computed SC_Integrations."""
        # Large SC_Resets1 in the observation lengthens the first integration;
        # overriding reset_frames_1 with the (smaller) default should let more
        # integrations fit.
        seq_a = _make_nirda_seq(
            duration_sec=1800,
            **{**_NIRDA_KWARGS, "sc_resets1": 500},
        )
        seq_b = seq_a.copy()
        sched = _sched()

        out_no = sched._update_NIRDA_integrations(
            seq_a,
            seq_a.duration,
            overhead=_overhead(),
        )
        out_override = sched._update_NIRDA_integrations(
            seq_b,
            seq_b.duration,
            overhead=_overhead(),
            override_fields=["reset_frames_1"],
        )
        integ_no = int(
            out_no.get_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations"
            )
        )
        integ_override = int(
            out_override.get_payload_parameter(
                "AcquireInfCamImages", "SC_Integrations"
            )
        )
        # Default reset_frames_1 (50) < 500, so the first integration is
        # shorter and the count should not decrease.
        assert integ_override >= integ_no


class TestVisdaParameterOverride:
    """override_fields replaces observation values with VisdaData defaults."""

    def test_overridden_field_written_back_as_default(self):
        defaults = VisdaData()
        seq = _make_vda_seq(
            duration_sec=1800,
            exposure_us=500_000,
            frames_per_coadd=3,
        )
        sched = _sched()
        seq_out = sched._update_VDA_integrations(
            seq,
            seq.duration,
            overhead=_overhead(),
            override_fields=["frames_per_coadd"],
        )
        fpc = seq_out.get_payload_parameter(
            "AcquireVisCamScienceData", "FramesPerCoadd"
        )
        assert int(fpc) == defaults.frames_per_coadd
        # Exposure (not overridden) is unchanged.
        exposure = seq_out.get_payload_parameter(
            "AcquireVisCamScienceData", "ExposureTime_us"
        )
        assert int(exposure) == 500_000

    def test_exposure_override_written_in_microseconds(self):
        defaults = VisdaData()
        seq = _make_vda_seq(
            duration_sec=1800,
            exposure_us=500_000,
            frames_per_coadd=1,
        )
        sched = _sched()
        seq_out = sched._update_VDA_integrations(
            seq,
            seq.duration,
            overhead=_overhead(),
            override_fields=["exposure_time_s"],
        )
        exposure = seq_out.get_payload_parameter(
            "AcquireVisCamScienceData", "ExposureTime_us"
        )
        assert int(exposure) == int(defaults.exposure_time_s.to(u.us).value)


class TestParameterOverrideValues:
    """Dict-form overrides force explicit values (None -> class default)."""

    def test_nirda_explicit_value_written(self):
        defaults = NirdaData()
        seq = _make_nirda_seq(
            duration_sec=1800,
            sc_drop1=5,
            sc_drop3=7,
            **{
                k: v
                for k, v in _NIRDA_KWARGS.items()
                if k not in ("sc_drop1", "sc_drop3")
            },
        )
        sched = _sched()
        out = sched._update_NIRDA_integrations(
            seq,
            seq.duration,
            overhead=_overhead(),
            # drop_frames_1 -> explicit 2; drop_frames_3 -> class default.
            override_fields={"drop_frames_1": 2, "drop_frames_3": None},
        )
        drop1 = int(
            out.get_payload_parameter("AcquireInfCamImages", "SC_DropFrames1")
        )
        drop3 = int(
            out.get_payload_parameter("AcquireInfCamImages", "SC_DropFrames3")
        )
        assert drop1 == 2
        assert drop3 == defaults.drop_frames_3

    def test_nirda_explicit_reset_changes_count(self):
        """An explicit reset_frames_1 value drives SC_Integrations."""
        seq_default = _make_nirda_seq(
            duration_sec=1800, **{**_NIRDA_KWARGS, "sc_resets1": 1}
        )
        seq_big = seq_default.copy()
        sched = _sched()
        n_small = int(
            sched._update_NIRDA_integrations(
                seq_default,
                seq_default.duration,
                overhead=_overhead(),
                override_fields={"reset_frames_1": 1},
            ).get_payload_parameter("AcquireInfCamImages", "SC_Integrations")
        )
        n_big = int(
            sched._update_NIRDA_integrations(
                seq_big,
                seq_big.duration,
                overhead=_overhead(),
                override_fields={"reset_frames_1": 5000},
            ).get_payload_parameter("AcquireInfCamImages", "SC_Integrations")
        )
        # A much larger first-integration reset count fits fewer integrations.
        assert n_big <= n_small

    def test_visda_explicit_value_written(self):
        seq = _make_vda_seq(
            duration_sec=1800, exposure_us=500_000, frames_per_coadd=3
        )
        sched = _sched()
        out = sched._update_VDA_integrations(
            seq,
            seq.duration,
            overhead=_overhead(),
            override_fields={"frames_per_coadd": 7},
        )
        fpc = int(
            out.get_payload_parameter(
                "AcquireVisCamScienceData", "FramesPerCoadd"
            )
        )
        assert fpc == 7

    def test_none_value_uses_default(self):
        defaults = VisdaData()
        seq = _make_vda_seq(
            duration_sec=1800, exposure_us=500_000, frames_per_coadd=3
        )
        sched = _sched()
        out = sched._update_VDA_integrations(
            seq,
            seq.duration,
            overhead=_overhead(),
            override_fields={"frames_per_coadd": None},
        )
        fpc = int(
            out.get_payload_parameter(
                "AcquireVisCamScienceData", "FramesPerCoadd"
            )
        )
        assert fpc == defaults.frames_per_coadd


class TestOverrideRoutingByPriority:
    """The wrapper applies overrides only to matching priorities."""

    def _make_nirda_seq_priority(self, priority):
        seq = _make_nirda_seq(
            duration_sec=1800,
            sc_drop1=5,
            **{k: v for k, v in _NIRDA_KWARGS.items() if k != "sc_drop1"},
        )
        seq.priority = priority
        return seq

    def test_matching_priority_is_overridden(self):
        defaults = NirdaData()
        sched = _sched_with_overhead()
        sched._override_nirda_parameters = {0: ["drop_frames_1"]}
        sched._override_visda_parameters = {}

        seq = self._make_nirda_seq_priority(0)
        out = sched._update_payload_parameters_sequence(seq)
        drop1 = out.get_payload_parameter(
            "AcquireInfCamImages", "SC_DropFrames1"
        )
        assert int(drop1) == defaults.drop_frames_1

    def test_non_matching_priority_not_overridden(self):
        sched = _sched_with_overhead()
        sched._override_nirda_parameters = {0: ["drop_frames_1"]}
        sched._override_visda_parameters = {}

        seq = self._make_nirda_seq_priority(2)
        out = sched._update_payload_parameters_sequence(seq)
        drop1 = out.get_payload_parameter(
            "AcquireInfCamImages", "SC_DropFrames1"
        )
        # Priority 2 not in override map -> observation value (5) preserved.
        assert int(drop1) == 5

    def test_overrides_accepted_and_stored_via_constructor(self):
        """override_*_parameters are constructor args stored on the instance."""
        with mock.patch("shortschedule.scheduler.Visibility"):
            proc = ScheduleProcessor(
                "L1",
                "L2",
                override_nirda_parameters={0: ["drop_frames_1"]},
                override_visda_parameters={1: ["frames_per_coadd"]},
            )
        assert proc._override_nirda_parameters == {0: ["drop_frames_1"]}
        assert proc._override_visda_parameters == {1: ["frames_per_coadd"]}

    def test_override_defaults_to_empty_when_unset(self):
        """Omitting the override args yields empty override maps."""
        with mock.patch("shortschedule.scheduler.Visibility"):
            proc = ScheduleProcessor("L1", "L2")
        assert proc._override_nirda_parameters == {}
        assert proc._override_visda_parameters == {}


# ---------------------------------------------------------------------------
# Data-volume limit warnings
# ---------------------------------------------------------------------------


def _sched_with_limits(max_uncompressed, max_compressed):
    """Bare ScheduleProcessor carrying only the data-volume limits."""
    sched = ScheduleProcessor.__new__(ScheduleProcessor)
    sched.max_file_size_uncompressed = max_uncompressed
    sched.max_file_size_compressed = max_compressed
    return sched


class TestVitlReset1:
    """update_nirda_reset1_for_vitl adjusts SC_Resets1 before solving."""

    _KW = dict(
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

    def _sched(self, enabled, vitl=60.0 * u.s):
        sched = ScheduleProcessor.__new__(ScheduleProcessor)
        sched.update_nirda_reset1_for_vitl = enabled
        sched.vitl_settling_time = vitl
        return sched

    def test_reset1_updated_when_enabled(self):
        """SC_Resets1 is set to cover the VITL settling time."""
        seq = _make_nirda_seq(duration_sec=2000, **self._KW)
        sched = self._sched(enabled=True, vitl=60.0 * u.s)
        out = sched._update_NIRDA_integrations(
            seq, seq.duration, overhead=_overhead()
        )
        reset1 = int(
            out.get_payload_parameter("AcquireInfCamImages", "SC_Resets1")
        )
        # Expected value comes from NirdaData.update_for_vitl itself.
        expected = _nirda_from_kwargs(**self._KW)
        expected.update_for_vitl(60.0 * u.s)
        assert reset1 == expected.reset_frames_1
        assert reset1 > 1  # the seq started at sc_resets1=1

    def test_reset1_untouched_when_disabled(self):
        """With the flag off, SC_Resets1 keeps the observation's value."""
        seq = _make_nirda_seq(duration_sec=2000, **self._KW)
        sched = self._sched(enabled=False)
        out = sched._update_NIRDA_integrations(
            seq, seq.duration, overhead=_overhead()
        )
        reset1 = int(
            out.get_payload_parameter("AcquireInfCamImages", "SC_Resets1")
        )
        assert reset1 == 1

    def test_longer_settling_needs_more_resets(self):
        """A longer VITL settling time requires at least as many resets."""
        seq_short = _make_nirda_seq(duration_sec=2000, **self._KW)
        seq_long = _make_nirda_seq(duration_sec=2000, **self._KW)
        r_short = int(
            self._sched(True, 30.0 * u.s)
            ._update_NIRDA_integrations(
                seq_short, seq_short.duration, overhead=_overhead()
            )
            .get_payload_parameter("AcquireInfCamImages", "SC_Resets1")
        )
        r_long = int(
            self._sched(True, 120.0 * u.s)
            ._update_NIRDA_integrations(
                seq_long, seq_long.duration, overhead=_overhead()
            )
            .get_payload_parameter("AcquireInfCamImages", "SC_Resets1")
        )
        assert r_long >= r_short

    def test_defaults_stored_via_constructor(self):
        """Constructor enables VITL reset adjustment with a 60 s default."""
        with mock.patch("shortschedule.scheduler.Visibility"):
            proc = ScheduleProcessor("L1", "L2")
        assert proc.update_nirda_reset1_for_vitl is True
        assert proc.vitl_settling_time == 60.0 * u.s


class TestDataSizeWarnings:
    """The scheduler warns when computed data exceeds the size limits."""

    def test_vda_oversize_raises_warning(self):
        """A VISDA sequence over the uncompressed limit warns."""
        # 1800 s at 0.1 s/frame -> 18000 frames; default ROI gives a large
        # data volume that exceeds 830 MB uncompressed.
        seq = _make_vda_seq(
            duration_sec=1800,
            exposure_us=100_000,
            frames_per_coadd=1,
        )
        sched = _sched_with_limits(
            830.0 * 1000 * 1000 * u.byte,
            255.0 * 1000 * 1000 * u.byte,
        )
        with pytest.warns(UserWarning, match="exceeds"):
            sched._update_VDA_integrations(
                seq,
                seq.duration,
                overhead=_overhead(),
            )

    def test_vda_within_limits_no_warning(self):
        """A small VISDA sequence under the limits must not warn."""
        seq = _make_vda_seq(
            duration_sec=60,
            exposure_us=1_000_000,
            frames_per_coadd=1,
        )
        sched = _sched_with_limits(
            830.0 * 1000 * 1000 * u.byte,
            255.0 * 1000 * 1000 * u.byte,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sched._update_VDA_integrations(
                seq,
                seq.duration,
                overhead=_overhead(),
            )

    def test_nirda_oversize_raises_warning(self):
        """A NIRDA sequence over the uncompressed limit warns."""
        # Many groups + single read frame + long window pack many
        # integrations' worth of saved frames into the window, inflating the
        # data volume past the limit.
        seq = _make_nirda_seq(
            duration_sec=5000,
            roi_x=200,
            roi_y=200,
            sc_resets1=1,
            sc_resets2=1,
            sc_drop1=0,
            sc_drop2=0,
            sc_drop3=0,
            sc_read=1,
            sc_groups=20,
        )
        sched = _sched_with_limits(
            830.0 * 1000 * 1000 * u.byte,
            255.0 * 1000 * 1000 * u.byte,
        )
        with pytest.warns(UserWarning, match="exceeds"):
            sched._update_NIRDA_integrations(
                seq,
                seq.duration,
                overhead=_overhead(),
            )

    def test_no_limits_attribute_no_warning(self):
        """A bare processor without limits set must not warn (or crash)."""
        seq = _make_vda_seq(
            duration_sec=1800,
            exposure_us=100_000,
            frames_per_coadd=1,
        )
        sched = ScheduleProcessor.__new__(ScheduleProcessor)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sched._update_VDA_integrations(
                seq,
                seq.duration,
                overhead=_overhead(),
            )

    def test_limits_stored_with_defaults(self):
        """Constructor stores the default data-volume limits."""
        with mock.patch("shortschedule.scheduler.Visibility"):
            proc = ScheduleProcessor("L1", "L2")
        assert proc.max_file_size_uncompressed.to(u.byte).value == (
            830.0 * 1000 * 1000
        )
        assert proc.max_file_size_compressed.to(u.byte).value == (
            255.0 * 1000 * 1000
        )

    def test_limits_overridable(self):
        """Constructor accepts custom data-volume limits."""
        with mock.patch("shortschedule.scheduler.Visibility"):
            proc = ScheduleProcessor(
                "L1",
                "L2",
                max_file_size_uncompressed=1.0 * u.byte,
                max_file_size_compressed=2.0 * u.byte,
            )
        assert proc.max_file_size_uncompressed == 1.0 * u.byte
        assert proc.max_file_size_compressed == 2.0 * u.byte


# ---------------------------------------------------------------------------
# ScheduleProcessor.__init__ overhead validation
# ---------------------------------------------------------------------------


class TestScheduleProcessorOverheadValidation:
    """Tests for type/unit validation of overhead parameters in __init__."""

    def test_default_overhead_quantities_accepted(self):
        """Default overhead values (Quantity with time units) are accepted."""
        with mock.patch("shortschedule.scheduler.Visibility"):
            proc = ScheduleProcessor("L1", "L2")
        assert proc.overhead.visda_pre_overhead_time == 260 * u.s

    def test_timedelta_overhead_accepted(self):
        """TimeDelta overhead values must also be accepted without error."""
        with mock.patch("shortschedule.scheduler.Visibility"):
            proc = ScheduleProcessor(
                "L1",
                "L2",
                vda_pre_sequence_overhead=TimeDelta(300 * u.s),
            )
        assert proc.overhead.visda_pre_overhead_time == TimeDelta(300 * u.s)

    def test_wrong_units_raises_value_error(self):
        """A Quantity with non-time units must raise ValueError."""
        with mock.patch("shortschedule.scheduler.Visibility"):
            with pytest.raises(ValueError, match="time units"):
                ScheduleProcessor(
                    "L1",
                    "L2",
                    vda_pre_sequence_overhead=260 * u.meter,
                )

    def test_plain_number_raises_type_error(self):
        """A plain number (no units) must raise TypeError."""
        with mock.patch("shortschedule.scheduler.Visibility"):
            with pytest.raises(
                TypeError, match="astropy Quantity or TimeDelta"
            ):
                ScheduleProcessor(
                    "L1",
                    "L2",
                    nirda_pre_sequence_overhead=258,  # bare int, no units
                )


# ---------------------------------------------------------------------------
# Sequential ID renumbering
# ---------------------------------------------------------------------------


def _seq_with_id(seq_id):
    """Minimal ObservationSequence carrying just an ID."""
    start = Time("2026-06-15T12:00:00", scale="utc")
    return ObservationSequence(
        id=seq_id,
        target="T",
        priority=1,
        start_time=start,
        stop_time=start + TimeDelta(60, format="sec"),
        ra=0.0,
        dec=0.0,
        payload_params={},
    )


class TestRenumberIds:
    """_renumber_ids makes visit and observation IDs contiguous."""

    def _cal(self):
        return ScienceCalendar(
            metadata={},
            visits=[
                Visit(
                    id="0007",
                    sequences=[_seq_with_id("050"), _seq_with_id("099")],
                ),
                Visit(id="0003", sequences=[_seq_with_id("012")]),
                Visit(
                    id="ABC",
                    sequences=[
                        _seq_with_id("x"),
                        _seq_with_id("y"),
                        _seq_with_id("z"),
                    ],
                ),
            ],
        )

    def test_visit_ids_renumbered_from_one(self):
        """Visit IDs become 0001, 0002, ... in order."""
        sched = ScheduleProcessor.__new__(ScheduleProcessor)
        cal = self._cal()
        sched._renumber_ids(cal)
        assert [v.id for v in cal.visits] == ["0001", "0002", "0003"]

    def test_observation_ids_renumbered_per_visit(self):
        """Each visit's observation IDs restart at 001 and increment."""
        sched = ScheduleProcessor.__new__(ScheduleProcessor)
        cal = self._cal()
        sched._renumber_ids(cal)
        assert [s.id for s in cal.visits[0].sequences] == ["001", "002"]
        assert [s.id for s in cal.visits[1].sequences] == ["001"]
        assert [s.id for s in cal.visits[2].sequences] == ["001", "002", "003"]

    def test_already_sequential_is_noop(self):
        """Correctly numbered IDs are left unchanged."""
        sched = ScheduleProcessor.__new__(ScheduleProcessor)
        cal = ScienceCalendar(
            metadata={},
            visits=[
                Visit(id="0001", sequences=[_seq_with_id("001")]),
                Visit(
                    id="0002",
                    sequences=[_seq_with_id("001"), _seq_with_id("002")],
                ),
            ],
        )
        sched._renumber_ids(cal)
        assert [v.id for v in cal.visits] == ["0001", "0002"]
        assert [s.id for s in cal.visits[1].sequences] == ["001", "002"]

    def test_widths_are_zero_padded(self):
        """Visit IDs use 4 digits and observation IDs use 3 digits."""
        sched = ScheduleProcessor.__new__(ScheduleProcessor)
        cal = self._cal()
        sched._renumber_ids(cal)
        assert all(len(v.id) == 4 for v in cal.visits)
        assert all(len(s.id) == 3 for v in cal.visits for s in v.sequences)

    def test_roll_cache_follows_renumbered_visit_ids(self):
        """The precomputed roll cache is re-keyed onto the new visit IDs.

        validate_visibility() and the visibility Gantt plot both run after
        renumbering and look rolls up by the *new* visit ID.  A cache left
        on the old IDs silently hands back a neighbouring visit's roll,
        which produces bogus keepout violations.
        """
        sched = ScheduleProcessor.__new__(ScheduleProcessor)
        cal = ScienceCalendar(
            metadata={},
            visits=[
                Visit(id="0002", sequences=[_seq_with_id("001")]),
                Visit(id="0003", sequences=[_seq_with_id("001")]),
                Visit(id="0004", sequences=[_seq_with_id("001")]),
            ],
        )
        sched._computed_target_rolls = {
            "0002": {"KELT-9b": 14.0},
            "0003": {"KELT-9b": 16.0},
            "0004": {"KELT-9b": 10.0},
        }

        sched._renumber_ids(cal)

        assert [v.id for v in cal.visits] == ["0001", "0002", "0003"]
        # Each visit still resolves to the roll that was computed for it.
        assert sched._computed_target_rolls == {
            "0001": {"KELT-9b": 14.0},
            "0002": {"KELT-9b": 16.0},
            "0003": {"KELT-9b": 10.0},
        }

    def test_renumber_without_roll_cache_is_safe(self):
        """Renumbering works when no rolls were ever precomputed."""
        sched = ScheduleProcessor.__new__(ScheduleProcessor)
        cal = self._cal()
        sched._renumber_ids(cal)  # no _computed_target_rolls attribute at all
        assert [v.id for v in cal.visits] == ["0001", "0002", "0003"]
