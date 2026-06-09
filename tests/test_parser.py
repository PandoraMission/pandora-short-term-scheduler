from shortschedule.parser import parse_science_calendar


def test_parse_science_calendar_skips_free_time_targets(tmp_path):
    xml_path = tmp_path / "calendar.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<ScienceCalendar xmlns="/pandora/calendar/">
  <Meta Valid_From="2026-05-01T00:00:00" Expires="2026-05-02T00:00:00" />
  <Visit>
    <ID>visit-1</ID>
    <Observation_Sequence>
      <ID>seq-free</ID>
      <Observational_Parameters>
        <Target> Free Time </Target>
        <Priority>0</Priority>
        <Timing>
          <Start>2026-05-01T00:00:00.000</Start>
          <Stop>2026-05-01T00:10:00.000</Stop>
        </Timing>
        <Boresight>
          <RA>0.0</RA>
          <DEC>0.0</DEC>
        </Boresight>
      </Observational_Parameters>
    </Observation_Sequence>
    <Observation_Sequence>
      <ID>seq-target</ID>
      <Observational_Parameters>
        <Target>WASP-39b</Target>
        <Priority>1</Priority>
        <Timing>
          <Start>2026-05-01T00:10:00.000</Start>
          <Stop>2026-05-01T00:20:00.000</Stop>
        </Timing>
        <Boresight>
          <RA>10.0</RA>
          <DEC>20.0</DEC>
        </Boresight>
      </Observational_Parameters>
    </Observation_Sequence>
  </Visit>
  <Visit>
    <ID>visit-2</ID>
    <Observation_Sequence>
      <ID>seq-free-only</ID>
      <Observational_Parameters>
        <Target>Free Time</Target>
        <Priority>0</Priority>
        <Timing>
          <Start>2026-05-01T01:00:00.000</Start>
          <Stop>2026-05-01T01:10:00.000</Stop>
        </Timing>
        <Boresight>
          <RA>0.0</RA>
          <DEC>0.0</DEC>
        </Boresight>
      </Observational_Parameters>
    </Observation_Sequence>
  </Visit>
</ScienceCalendar>
""",
        encoding="utf-8",
    )

    calendar = parse_science_calendar(str(xml_path))

    assert len(calendar.visits) == 1
    assert calendar.visits[0].id == "visit-1"
    assert len(calendar.visits[0].sequences) == 1
    assert calendar.visits[0].sequences[0].id == "seq-target"
    assert calendar.visits[0].sequences[0].target == "WASP-39b"
