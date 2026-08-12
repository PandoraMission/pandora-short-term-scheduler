"""Tests for general per-priority XML-tag payload overrides.

Covers ScheduleProcessor(override_payload_parameters=...):
- Existing payload tags are overwritten by priority
- Missing tags are created
- Priority routing (int and 'Priority_N' string keys)
- Free-time observations are skipped
- Overrides applied before integration recompute (size/coadd/reset flow
  through to SC_Integrations / NumTotalFramesRequested)
"""

# Standard library
import unittest.mock as mock
import xml.etree.ElementTree as ET

# Third-party
from astropy import units as u
from astropy.time import Time, TimeDelta

# First-party/Local
from shortschedule.models import ObservationSequence
from shortschedule.nirda import NirdaData
from shortschedule.scheduler import ScheduleProcessor
from shortschedule.visda import VisdaData


def _nir_payload(**tags):
    root = ET.Element(NirdaData.PAYLOAD_SECTION)
    cfg = NirdaData().get_config()
    for field, (tag, _from, to_xml) in NirdaData.CONFIG_SPEC.items():
        ET.SubElement(root, tag).text = to_xml(cfg[field])
    ET.SubElement(root, "SC_Integrations").text = "0"
    for tag, value in tags.items():
        existing = root.find(tag)
        if existing is None:
            existing = ET.SubElement(root, tag)
        existing.text = str(value)
    return root


def _vis_payload(**tags):
    root = ET.Element(VisdaData.PAYLOAD_SECTION)
    cfg = VisdaData().get_config()
    for field, (tag, _from, to_xml) in VisdaData.CONFIG_SPEC.items():
        ET.SubElement(root, tag).text = to_xml(cfg[field])
    ET.SubElement(root, "NumTotalFramesRequested").text = "0"
    for tag, value in tags.items():
        existing = root.find(tag)
        if existing is None:
            existing = ET.SubElement(root, tag)
        existing.text = str(value)
    return root


def _seq(priority=0, target="TargetA", dur_min=60):
    start = Time("2026-03-01T00:00:00", scale="utc")
    return ObservationSequence(
        id="001",
        target=target,
        priority=priority,
        start_time=start,
        stop_time=start + TimeDelta(dur_min * 60, format="sec"),
        ra=10.0,
        dec=20.0,
        payload_params={
            NirdaData.PAYLOAD_SECTION: _nir_payload(ROI_SizeX=999),
            VisdaData.PAYLOAD_SECTION: _vis_payload(FramesPerCoadd=3),
        },
    )


def _bare_sched(overrides):
    sched = ScheduleProcessor.__new__(ScheduleProcessor)
    sched._override_payload_parameters = (
        ScheduleProcessor._normalize_priority_keys(overrides)
    )
    return sched


class TestApplyPayloadOverrides:
    def test_existing_tag_overwritten(self):
        seq = _seq(priority=0)
        sched = _bare_sched(
            {0: {"AcquireInfCamImages": {"ROI_SizeX": 80}}}
        )
        sched._apply_payload_overrides(seq)
        assert (
            seq.get_payload_parameter("AcquireInfCamImages", "ROI_SizeX")
            == "80"
        )

    def test_missing_tag_created(self):
        seq = _seq(priority=0)
        # ROI_StartX is not present in the default NIR payload.
        assert (
            seq.get_payload_parameter("AcquireInfCamImages", "ROI_StartX")
            is None
        )
        sched = _bare_sched(
            {0: {"AcquireInfCamImages": {"ROI_StartX": 1737}}}
        )
        sched._apply_payload_overrides(seq)
        assert (
            seq.get_payload_parameter("AcquireInfCamImages", "ROI_StartX")
            == "1737"
        )

    def test_priority_string_key(self):
        seq = _seq(priority=1)
        sched = _bare_sched(
            {"Priority_1": {"AcquireVisCamScienceData": {"FramesPerCoadd": 5}}}
        )
        sched._apply_payload_overrides(seq)
        assert (
            seq.get_payload_parameter(
                "AcquireVisCamScienceData", "FramesPerCoadd"
            )
            == "5"
        )

    def test_non_matching_priority_untouched(self):
        seq = _seq(priority=2)
        sched = _bare_sched(
            {0: {"AcquireInfCamImages": {"ROI_SizeX": 80}}}
        )
        sched._apply_payload_overrides(seq)
        assert (
            seq.get_payload_parameter("AcquireInfCamImages", "ROI_SizeX")
            == "999"
        )

    def test_free_time_skipped(self):
        seq = _seq(priority=0, target="Free Time")
        sched = _bare_sched(
            {0: {"AcquireInfCamImages": {"ROI_SizeX": 80}}}
        )
        sched._apply_payload_overrides(seq)
        assert (
            seq.get_payload_parameter("AcquireInfCamImages", "ROI_SizeX")
            == "999"
        )


class TestObservationalParameterOverride:
    """Nested Observational_Parameters overrides reach the written XML."""

    def test_boresight_override_stored_nested(self):
        seq = _seq(priority=0)
        sched = _bare_sched(
            {0: {"Observational_Parameters": {"Boresight": {
                "PRI_CMD_DIR": 9}}}}
        )
        sched._apply_payload_overrides(seq)
        obs = seq.payload_params["Observational_Parameters"]
        bore = obs.find("Boresight")
        assert bore is not None
        assert bore.find("PRI_CMD_DIR").text == "9"

    def test_override_written_to_xml(self, tmp_path):
        """A non-default PRI_CMD_DIR override appears in the output XML."""
        from shortschedule.models import ScienceCalendar, Visit
        from shortschedule.writer import XMLWriter

        seq = _seq(priority=0)
        sched = _bare_sched(
            {0: {"Observational_Parameters": {"Boresight": {
                "PRI_CMD_DIR": 7}}}}
        )
        sched._apply_payload_overrides(seq)
        cal = ScienceCalendar(metadata={}, visits=[Visit("0001", [seq])])
        out = tmp_path / "obs.xml"
        XMLWriter().write_calendar(cal, str(out))

        root = ET.parse(str(out)).getroot()

        def local(elem, name):
            return next(
                (c for c in elem if c.tag.endswith(name)), None
            )

        visit = local(root, "Visit")
        obs_seq = local(visit, "Observation_Sequence")
        obs_params = local(obs_seq, "Observational_Parameters")
        boresight = local(obs_params, "Boresight")
        pri = local(boresight, "PRI_CMD_DIR")
        assert pri is not None and pri.text == "7"
        # The override block must NOT leak into Payload_Parameters.
        payload = local(obs_seq, "Payload_Parameters")
        assert local(payload, "Observational_Parameters") is None


class TestSingleRoiConversion:
    """Convert single-ROI auto-detect VIS sections to predefined-ROI."""

    def _sched(self):
        sched = ScheduleProcessor.__new__(ScheduleProcessor)
        sched.convert_single_roi_to_predefined = True
        return sched

    def _seq_single_auto(self, **vis_extra):
        seq = _seq(priority=0)
        vis = seq.payload_params[VisdaData.PAYLOAD_SECTION]
        for tag in ("MaxNumStarRois", "StarRoiDetMethod"):
            existing = vis.find(tag)
            if existing is None:
                existing = ET.SubElement(vis, tag)
        vis.find("MaxNumStarRois").text = "1"
        vis.find("StarRoiDetMethod").text = "2"
        for tag, value in vis_extra.items():
            elem = vis.find(tag)
            if elem is None:
                elem = ET.SubElement(vis, tag)
            elem.text = str(value)
        return seq

    def test_converts_to_predefined(self):
        seq = self._seq_single_auto()
        assert self._sched()._convert_single_roi_to_predefined(seq) is True
        vis = seq.payload_params[VisdaData.PAYLOAD_SECTION]
        assert vis.find("StarRoiDetMethod").text == "1"
        assert vis.find("numPredefinedStarRois").text == "1"
        # RA/Dec come from the sequence's own coordinates (10.0/20.0).
        assert vis.find("PredefinedStarRoiRa").find("RA1").text == "10"
        assert vis.find("PredefinedStarRoiDec").find("Dec1").text == "20"

    def test_prefers_target_radec_verbatim(self):
        seq = self._seq_single_auto(TargetRA="123.456", TargetDEC="-7.89")
        assert self._sched()._convert_single_roi_to_predefined(seq) is True
        vis = seq.payload_params[VisdaData.PAYLOAD_SECTION]
        assert vis.find("PredefinedStarRoiRa").find("RA1").text == "123.456"
        assert vis.find("PredefinedStarRoiDec").find("Dec1").text == "-7.89"

    def test_idempotent_when_already_predefined(self):
        seq = self._seq_single_auto()
        sched = self._sched()
        assert sched._convert_single_roi_to_predefined(seq) is True
        # A second pass finds RA1/Dec1 already present and StarRoiDetMethod 1.
        assert sched._convert_single_roi_to_predefined(seq) is False

    def test_skips_non_single_roi(self):
        seq = self._seq_single_auto()
        seq.payload_params[VisdaData.PAYLOAD_SECTION].find(
            "MaxNumStarRois"
        ).text = "9"
        assert self._sched()._convert_single_roi_to_predefined(seq) is False

    def test_skips_non_autodetect_method(self):
        seq = self._seq_single_auto()
        seq.payload_params[VisdaData.PAYLOAD_SECTION].find(
            "StarRoiDetMethod"
        ).text = "1"
        assert self._sched()._convert_single_roi_to_predefined(seq) is False

    def test_skips_free_time(self):
        seq = self._seq_single_auto()
        seq.target = "Free Time"
        assert self._sched()._convert_single_roi_to_predefined(seq) is False


class TestFixBadData:
    """Replace invalid name symbols and flag NaN-like field values."""

    def _sched(self):
        sched = ScheduleProcessor.__new__(ScheduleProcessor)
        sched.fix_bad_data = True
        sched.logger = None
        return sched

    def test_replaces_plus_in_target_attribute(self):
        seq = _seq(target="WASP+12b")
        self._sched()._fix_bad_data(seq)
        assert seq.target == "WASP_12b"

    def test_replaces_plus_in_target_payload_tags(self):
        seq = _seq()
        vis = seq.payload_params[VisdaData.PAYLOAD_SECTION]
        ET.SubElement(vis, "TargetID").text = "TOI+674b"
        self._sched()._fix_bad_data(seq)
        assert vis.find("TargetID").text == "TOI_674b"

    def test_nan_in_numeric_field_warns(self, capsys):
        seq = _seq()
        seq.payload_params[VisdaData.PAYLOAD_SECTION].find(
            "FramesPerCoadd"
        ).text = "nan"
        self._sched()._fix_bad_data(seq)
        out = capsys.readouterr().out
        assert "NaN-like value" in out and "FramesPerCoadd" in out

    def test_nan_in_name_field_not_flagged(self, capsys):
        seq = _seq()
        vis = seq.payload_params[VisdaData.PAYLOAD_SECTION]
        ET.SubElement(vis, "TargetID").text = "nan"
        self._sched()._fix_bad_data(seq)
        assert "NaN-like value" not in capsys.readouterr().out

    def test_free_time_nan_skipped(self, capsys):
        seq = _seq(target="Free Time")
        seq.payload_params[VisdaData.PAYLOAD_SECTION].find(
            "FramesPerCoadd"
        ).text = "nan"
        self._sched()._fix_bad_data(seq)
        assert "NaN-like value" not in capsys.readouterr().out


class TestNormalizePriorityKeys:
    def test_int_and_string_keys(self):
        norm = ScheduleProcessor._normalize_priority_keys(
            {0: {"a": 1}, "Priority_1": {"b": 2}, "2": {"c": 3}}
        )
        assert set(norm.keys()) == {0, 1, 2}

    def test_empty(self):
        assert ScheduleProcessor._normalize_priority_keys(None) == {}


class TestPayloadOverrideThroughConstructor:
    def test_constructor_stores_normalized(self):
        with mock.patch("shortschedule.scheduler.Visibility"):
            proc = ScheduleProcessor(
                "L1",
                "L2",
                override_payload_parameters={
                    "Priority_0": {"AcquireInfCamImages": {"ROI_SizeX": 80}}
                },
            )
        assert 0 in proc._override_payload_parameters

    def _nir_seq_big_roi(self):
        """A NIR sequence with a huge ROI (slow frames -> few integrations)."""
        start = Time("2026-03-01T00:00:00", scale="utc")
        return ObservationSequence(
            id="001",
            target="T",
            priority=0,
            start_time=start,
            stop_time=start + TimeDelta(3600, format="sec"),
            ra=1.0,
            dec=2.0,
            payload_params={
                NirdaData.PAYLOAD_SECTION: _nir_payload(
                    ROI_SizeX=2000, ROI_SizeY=2000
                ),
                VisdaData.PAYLOAD_SECTION: _vis_payload(FramesPerCoadd=1),
            },
        )

    def _run(self, overrides):
        sched = ScheduleProcessor.__new__(ScheduleProcessor)
        sched._override_payload_parameters = (
            ScheduleProcessor._normalize_priority_keys(overrides or {})
        )
        sched._override_nirda_parameters = {}
        sched._override_visda_parameters = {}
        sched.overhead = None  # _update_* falls back to OverheadTiming()
        out = sched._update_payload_parameters_sequence(
            self._nir_seq_big_roi(), visit_id="1"
        )
        return (
            out.get_payload_parameter("AcquireInfCamImages", "ROI_SizeX"),
            int(
                out.get_payload_parameter(
                    "AcquireInfCamImages", "SC_Integrations"
                )
            ),
            int(
                out.get_payload_parameter(
                    "AcquireVisCamScienceData", "NumTotalFramesRequested"
                )
            ),
        )

    def test_override_flows_into_integration_recompute(self):
        """ROI/coadd overrides change the recomputed timing and data sizes."""
        roi_no, sc_no, _ = self._run(None)
        roi_ov, sc_ov, ntf_ov = self._run(
            {
                0: {
                    "AcquireInfCamImages": {"ROI_SizeX": 80, "ROI_SizeY": 250},
                    "AcquireVisCamScienceData": {"FramesPerCoadd": 50},
                }
            }
        )
        # The override changed the ROI used by the data class...
        assert roi_no == "2000"
        assert roi_ov == "80"
        # ...which makes each NIR frame faster, fitting more integrations.
        assert sc_ov > sc_no
        # ...and the VIS frame count is floored to the new coadd multiple.
        assert ntf_ov % 50 == 0
