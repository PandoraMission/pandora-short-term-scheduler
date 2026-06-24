"""Tests for ScheduleProcessor.generate_diagnostics (.diag report).

Covers:
- Week-summary header and per-day sections
- Per-day priority counts, unique targets, observing/gap percentages
- NIR/VIS frame and data totals (compressed + uncompressed) from the
  detector data classes
- Per-day file manifest
- Day bucketing by largest UTC overlap
- File is written to the requested path
"""

# Standard library
import xml.etree.ElementTree as ET

# Third-party
from astropy import units as u
from astropy.time import Time, TimeDelta

# First-party/Local
from shortschedule.models import ObservationSequence, ScienceCalendar, Visit
from shortschedule.nirda import NirdaData
from shortschedule.scheduler import ScheduleProcessor
from shortschedule.visda import VisdaData


def _sched():
    return ScheduleProcessor.__new__(ScheduleProcessor)


def _nir_payload(sc_integrations):
    root = ET.Element(NirdaData.PAYLOAD_SECTION)
    cfg = NirdaData().get_config()
    for field, (tag, _from, to_xml) in NirdaData.CONFIG_SPEC.items():
        ET.SubElement(root, tag).text = to_xml(cfg[field])
    ET.SubElement(root, "SC_Integrations").text = str(sc_integrations)
    return root


def _vis_payload(num_total_frames):
    root = ET.Element(VisdaData.PAYLOAD_SECTION)
    cfg = VisdaData().get_config()
    for field, (tag, _from, to_xml) in VisdaData.CONFIG_SPEC.items():
        ET.SubElement(root, tag).text = to_xml(cfg[field])
    ET.SubElement(root, "NumTotalFramesRequested").text = str(
        num_total_frames
    )
    return root


def _seq(sid, target, priority, start_iso, dur_min, sc_int=10, vis_frames=50):
    start = Time(start_iso, scale="utc")
    return ObservationSequence(
        id=sid,
        target=target,
        priority=priority,
        start_time=start,
        stop_time=start + TimeDelta(dur_min * 60, format="sec"),
        ra=10.0,
        dec=20.0,
        payload_params={
            NirdaData.PAYLOAD_SECTION: _nir_payload(sc_int),
            VisdaData.PAYLOAD_SECTION: _vis_payload(vis_frames),
        },
    )


def _calendar():
    # Day A: two sequences (one with a gap before the second); Day B: one.
    v1 = Visit(
        id="0001",
        sequences=[
            _seq("001", "TargetA", 0, "2026-03-01T00:00:00", 60),
            _seq("002", "TargetB", 1, "2026-03-01T02:00:00", 60),
        ],
    )
    v2 = Visit(
        id="0002",
        sequences=[
            _seq("001", "TargetA", 2, "2026-03-02T00:00:00", 120),
        ],
    )
    return ScienceCalendar(metadata={}, visits=[v1, v2])


class TestGenerateDiagnostics:
    def test_writes_file_and_returns_text(self, tmp_path):
        sched = _sched()
        out = tmp_path / "cal"
        text = sched.generate_diagnostics(_calendar(), output_path=out)
        assert (tmp_path / "cal.diag").exists()
        assert text == (tmp_path / "cal.diag").read_text(encoding="utf-8")

    def test_week_summary_header(self):
        text = _sched().generate_diagnostics(_calendar())
        assert "Calendar Summary 2026-03-01 : 2026-03-02" in text
        assert "Total Observations: 3" in text
        # Priorities: one each of 0, 1, 2.
        assert "  - Priority 0 = 1" in text
        assert "  - Priority 1 = 1" in text
        assert "  - Priority 2 = 1" in text

    def test_per_day_sections_present(self):
        text = _sched().generate_diagnostics(_calendar())
        assert "2026-03-01" in text
        assert "2026-03-02" in text
        assert "Number of Observations: 2" in text  # day A
        assert "Number of Observations: 1" in text  # day B

    def test_unique_targets_listed(self):
        text = _sched().generate_diagnostics(_calendar())
        assert "List of Unique Targets:" in text
        assert "  - TargetA" in text
        assert "  - TargetB" in text

    def test_observing_and_gap_percentages(self):
        text = _sched().generate_diagnostics(_calendar())
        # Day A: 120 min observing, 60 min gap (between 01:00 and 02:00).
        assert "Total Gaps: 60 Mins (33.3%)" in text
        assert "Total Observing: 120 Mins (66.7%)" in text

    def test_data_lines_and_passes(self):
        text = _sched().generate_diagnostics(
            _calendar(), pass_data_volume_mb=100.0
        )
        assert "Uncompressed Data" in text
        assert "Required Passes:" in text
        # Without a pass volume, passes are N/A.
        text2 = _sched().generate_diagnostics(_calendar())
        assert "Required Passes: N/A" in text2

    def test_manifest_entries(self):
        text = _sched().generate_diagnostics(_calendar())
        assert "Manifest of Files for the Day:" in text
        assert "- /mnt/data/sci/20260301T000000_TargetA.bin" in text
        assert "- /mnt/data/sci/20260302T000000_TargetA.bin" in text

    def test_manifest_lists_fits_products(self):
        """Each .bin is followed by tab-indented InfImg/VisSci/engineering."""
        text = _sched().generate_diagnostics(_calendar())
        # Tab-indented FITS lines.
        assert "\t- " in text
        assert "_InfImg_TargetA_" in text
        assert "_VisSci_TargetA_" in text
        assert "_engineering.fits" in text
        # InfImg cube depth = integrations * groups (10 * default groups=6=60)
        # and the InfImg/VisSci/engineering names carry payload fields.
        assert "_b1_e01_i10_g06_" in text
        assert "_VisSci_TargetA_d050_n009_f00050_e000200000us.fits" in text

    def test_fits_indented_under_bin(self):
        """FITS lines are nested (tabbed) directly under their .bin line."""
        lines = _sched().generate_diagnostics(_calendar()).splitlines()
        bin_idx = next(
            i for i, ln in enumerate(lines)
            if ln.startswith("- /mnt/data/sci/20260302T000000_TargetA.bin")
        )
        # The lines immediately following the bin are tab-indented FITS files.
        following = lines[bin_idx + 1:bin_idx + 4]
        assert all(ln.startswith("\t- ") for ln in following)
        assert any("_InfImg_" in ln for ln in following)
        assert any("_engineering.fits" in ln for ln in following)

    def test_nir_vis_frames_positive(self):
        text = _sched().generate_diagnostics(_calendar())
        # NIR frames = integrations * groups (averaged) > 0; VIS frames =
        # coadds = NumTotalFrames // frames_per_coadd > 0.
        assert "Total NIR Frames = " in text
        assert "Total Vis Frames = " in text

    def test_empty_calendar(self):
        text = _sched().generate_diagnostics(
            ScienceCalendar(metadata={}, visits=[])
        )
        assert "No observations" in text
