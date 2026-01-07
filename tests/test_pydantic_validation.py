"""Test Pydantic validation in data models.

These tests demonstrate the validation capabilities added by Pydantic,
ensuring data integrity at construction time rather than at runtime.
"""

# Third-party
import pytest
from astropy.time import Time
from pydantic import ValidationError

# First-party/Local
from shortschedule.models import (
    ObservationSequence,
    ScienceCalendar,
    Visit,
)


def test_observation_sequence_validates_time_objects():
    """Test that non-Time objects for start/stop times raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ObservationSequence(
            id="test_seq",
            target="test_target",
            priority=1,
            start_time="not a Time object",  # Should be Time
            stop_time=Time("2026-01-01T00:00:00", format="isot"),
            ra=180.0,
            dec=45.0,
            payload_params={},
        )
    assert "start_time" in str(exc_info.value)


def test_observation_sequence_validates_priority_non_negative():
    """Test that negative priority raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ObservationSequence(
            id="test_seq",
            target="test_target",
            priority=-1,  # Invalid: negative
            start_time=Time("2026-01-01T00:00:00", format="isot"),
            stop_time=Time("2026-01-01T01:00:00", format="isot"),
            ra=180.0,
            dec=45.0,
            payload_params={},
        )
    assert "priority" in str(exc_info.value)


def test_observation_sequence_validates_ra_range():
    """Test that RA outside [0, 360) raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ObservationSequence(
            id="test_seq",
            target="test_target",
            priority=1,
            start_time=Time("2026-01-01T00:00:00", format="isot"),
            stop_time=Time("2026-01-01T01:00:00", format="isot"),
            ra=361.0,  # Invalid: > 360
            dec=45.0,
            payload_params={},
        )
    assert "ra" in str(exc_info.value)


def test_observation_sequence_validates_dec_range():
    """Test that Dec outside [-90, 90] raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ObservationSequence(
            id="test_seq",
            target="test_target",
            priority=1,
            start_time=Time("2026-01-01T00:00:00", format="isot"),
            stop_time=Time("2026-01-01T01:00:00", format="isot"),
            ra=180.0,
            dec=91.0,  # Invalid: > 90
            payload_params={},
        )
    assert "dec" in str(exc_info.value)


def test_observation_sequence_validates_roll_range():
    """Test that roll outside [0, 360) raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ObservationSequence(
            id="test_seq",
            target="test_target",
            priority=1,
            start_time=Time("2026-01-01T00:00:00", format="isot"),
            stop_time=Time("2026-01-01T01:00:00", format="isot"),
            ra=180.0,
            dec=45.0,
            payload_params={},
            roll=360.5,  # Invalid: >= 360
        )
    assert "roll" in str(exc_info.value)


def test_observation_sequence_accepts_valid_data():
    """Test that valid data creates an ObservationSequence successfully."""
    seq = ObservationSequence(
        id="test_seq",
        target="test_target",
        priority=1,
        start_time=Time("2026-01-01T00:00:00", format="isot"),
        stop_time=Time("2026-01-01T01:00:00", format="isot"),
        ra=180.0,
        dec=45.0,
        payload_params={},
        roll=90.0,
    )
    assert seq.id == "test_seq"
    assert seq.target == "test_target"
    assert seq.priority == 1
    assert seq.ra == 180.0
    assert seq.dec == 45.0
    assert seq.roll == 90.0


def test_visit_requires_id_and_sequences():
    """Test that Visit requires id and sequences fields."""
    seq = ObservationSequence(
        id="test_seq",
        target="test_target",
        priority=1,
        start_time=Time("2026-01-01T00:00:00", format="isot"),
        stop_time=Time("2026-01-01T01:00:00", format="isot"),
        ra=180.0,
        dec=45.0,
        payload_params={},
    )
    visit = Visit(id="visit_1", sequences=[seq])
    assert visit.id == "visit_1"
    assert len(visit.sequences) == 1


def test_science_calendar_ensures_metadata_is_dict():
    """Test that ScienceCalendar converts None metadata to empty dict."""
    seq = ObservationSequence(
        id="test_seq",
        target="test_target",
        priority=1,
        start_time=Time("2026-01-01T00:00:00", format="isot"),
        stop_time=Time("2026-01-01T01:00:00", format="isot"),
        ra=180.0,
        dec=45.0,
        payload_params={},
    )
    visit = Visit(id="visit_1", sequences=[seq])
    cal = ScienceCalendar(metadata=None, visits=[visit])
    assert cal.metadata == {}


def test_science_calendar_preserves_metadata():
    """Test that ScienceCalendar preserves provided metadata."""
    seq = ObservationSequence(
        id="test_seq",
        target="test_target",
        priority=1,
        start_time=Time("2026-01-01T00:00:00", format="isot"),
        stop_time=Time("2026-01-01T01:00:00", format="isot"),
        ra=180.0,
        dec=45.0,
        payload_params={},
    )
    visit = Visit(id="visit_1", sequences=[seq])
    metadata = {"key": "value"}
    cal = ScienceCalendar(metadata=metadata, visits=[visit])
    assert cal.metadata == {"key": "value"}


def test_observation_sequence_iso_string_conversion():
    """Test that ISO string times are automatically converted to Time objects."""
    seq = ObservationSequence(
        id="test_seq",
        target="test_target",
        priority=1,
        start_time="2026-01-01T00:00:00",  # ISO string
        stop_time="2026-01-01T01:00:00",  # ISO string
        ra=180.0,
        dec=45.0,
        payload_params={},
    )
    assert isinstance(seq.start_time, Time)
    assert isinstance(seq.stop_time, Time)
    assert seq.start_time.isot == "2026-01-01T00:00:00.000"
