"""Unit tests for the roll module.

Tests cover:
- Basic roll angle calculation
- Vector conversion utilities
- Visit-level roll calculation
- Calendar-level roll application
- Edge cases and error handling
"""

# Standard library
from unittest.mock import MagicMock

# Third-party
import numpy as np
import pytest
from astropy.time import Time

# First-party/Local
from shortschedule.roll import (
    apply_rolls_to_calendar,
    apply_rolls_to_visit,
    calculate_roll,
    calculate_visit_rolls,
    normalize,
    radec_to_vector,
    vector_to_radec,
)


class TestRadecToVector:
    """Tests for radec_to_vector conversion."""

    def test_ra0_dec0(self):
        """RA=0, Dec=0 should point along +X axis."""
        vec = radec_to_vector(0, 0)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(vec, expected)

    def test_ra90_dec0(self):
        """RA=90, Dec=0 should point along +Y axis."""
        vec = radec_to_vector(90, 0)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(vec, expected)

    def test_ra0_dec90(self):
        """RA=0, Dec=90 should point along +Z axis (north pole)."""
        vec = radec_to_vector(0, 90)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(vec, expected)

    def test_ra0_dec_minus90(self):
        """RA=0, Dec=-90 should point along -Z axis (south pole)."""
        vec = radec_to_vector(0, -90)
        expected = np.array([0.0, 0.0, -1.0])
        np.testing.assert_array_almost_equal(vec, expected)

    def test_unit_vector(self):
        """Output should always be a unit vector."""
        for ra in [0, 45, 90, 180, 270]:
            for dec in [-60, -30, 0, 30, 60]:
                vec = radec_to_vector(ra, dec)
                assert np.isclose(np.linalg.norm(vec), 1.0)

    def test_roundtrip_radec_vector_radec(self):
        """Converting RA/Dec to vector and back should preserve values."""
        test_cases = [
            (0.0, 0.0),
            (90.0, 0.0),
            (180.0, 0.0),
            (270.0, 0.0),
            (45.0, 45.0),
            (123.456, -30.0),
            (359.9, 89.0),
            (0.0, -89.0),
        ]
        for ra, dec in test_cases:
            vec = radec_to_vector(ra, dec)
            ra_out, dec_out = vector_to_radec(vec)
            np.testing.assert_almost_equal(
                ra_out,
                ra,
                decimal=10,
                err_msg=f"RA roundtrip failed for ({ra}, {dec})",
            )
            np.testing.assert_almost_equal(
                dec_out,
                dec,
                decimal=10,
                err_msg=f"Dec roundtrip failed for ({ra}, {dec})",
            )


class TestNormalize:
    """Tests for the normalize helper function."""

    def test_normalize_simple(self):
        """Normalize a simple vector."""
        vec = np.array([3.0, 4.0, 0.0])
        result = normalize(vec)
        expected = np.array([0.6, 0.8, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_unit_vector(self):
        """Normalizing a unit vector returns the same vector."""
        vec = np.array([1.0, 0.0, 0.0])
        result = normalize(vec)
        np.testing.assert_array_almost_equal(result, vec)

    def test_normalize_zero_vector_raises(self):
        """Normalizing a zero vector should raise ValueError."""
        vec = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="Zero-length vector"):
            normalize(vec)


class TestCalculateRoll:
    """Tests for the main roll calculation function."""

    def test_roll_returns_float(self):
        """Roll calculation should return a float."""
        obs_time = Time("2026-06-15T12:00:00")
        roll = calculate_roll(ra=180.0, dec=30.0, obs_time=obs_time)
        assert isinstance(roll, float)

    def test_roll_in_valid_range(self):
        """Roll angle should be in [0, 360) range."""
        obs_time = Time("2026-06-15T12:00:00")
        for ra in [0, 90, 180, 270]:
            for dec in [-45, 0, 45]:
                roll = calculate_roll(ra=ra, dec=dec, obs_time=obs_time)
                assert (
                    0.0 <= roll < 360.0
                ), f"Roll {roll} out of range for RA={ra}, Dec={dec}"

    def test_roll_consistency_same_target(self):
        """Same target at same time should give same roll."""
        obs_time = Time("2026-06-15T12:00:00")
        roll1 = calculate_roll(ra=120.0, dec=25.0, obs_time=obs_time)
        roll2 = calculate_roll(ra=120.0, dec=25.0, obs_time=obs_time)
        assert roll1 == roll2

    def test_roll_changes_with_time(self):
        """Roll should change as Sun position changes over time."""
        time1 = Time("2026-06-15T12:00:00")
        time2 = Time("2026-09-15T12:00:00")  # 3 months later

        roll1 = calculate_roll(ra=180.0, dec=0.0, obs_time=time1)
        roll2 = calculate_roll(ra=180.0, dec=0.0, obs_time=time2)

        # Sun has moved ~90 degrees in 3 months, so rolls should differ
        assert roll1 != roll2

    def test_roll_handles_high_declination(self):
        """Roll calculation should handle high declination targets."""
        obs_time = Time("2026-06-15T12:00:00")
        # High declination but not exactly at pole
        roll = calculate_roll(ra=0.0, dec=85.0, obs_time=obs_time)
        assert 0.0 <= roll < 360.0


class TestCalculateVisitRolls:
    """Tests for visit-level roll calculations."""

    def _make_mock_sequence(self, target, ra, dec, start_time):
        """Create a mock observation sequence."""
        seq = MagicMock()
        seq.target = target
        seq.ra = ra
        seq.dec = dec
        seq.start_time = start_time
        return seq

    def _make_mock_visit(self, sequences):
        """Create a mock visit with given sequences."""
        visit = MagicMock()
        visit.sequences = sequences
        return visit

    def test_single_target_visit(self):
        """Visit with single target should return one roll value."""
        start_time = Time("2026-06-15T12:00:00")
        seq = self._make_mock_sequence("HD_123456", 120.0, 30.0, start_time)
        visit = self._make_mock_visit([seq])

        rolls = calculate_visit_rolls(visit)

        assert len(rolls) == 1
        assert "HD_123456" in rolls
        assert isinstance(rolls["HD_123456"], float)

    def test_multiple_targets_visit(self):
        """Visit with multiple targets should return roll for each."""
        start_time = Time("2026-06-15T12:00:00")
        seq1 = self._make_mock_sequence("Target_A", 100.0, 20.0, start_time)
        seq2 = self._make_mock_sequence("Target_B", 200.0, -10.0, start_time)
        visit = self._make_mock_visit([seq1, seq2])

        rolls = calculate_visit_rolls(visit)

        assert len(rolls) == 2
        assert "Target_A" in rolls
        assert "Target_B" in rolls
        # Different targets should generally have different rolls
        assert rolls["Target_A"] != rolls["Target_B"]

    def test_same_target_multiple_sequences(self):
        """Multiple sequences of same target should share one roll."""
        start_time = Time("2026-06-15T12:00:00")
        seq1 = self._make_mock_sequence("HD_123456", 120.0, 30.0, start_time)
        seq2 = self._make_mock_sequence(
            "HD_123456", 120.0, 30.0, Time("2026-06-15T14:00:00")
        )
        visit = self._make_mock_visit([seq1, seq2])

        rolls = calculate_visit_rolls(visit)

        assert len(rolls) == 1
        assert "HD_123456" in rolls

    def test_reference_time_override(self):
        """Providing reference_time should use that for all calculations."""
        start_time1 = Time("2026-06-15T12:00:00")
        start_time2 = Time("2026-06-15T18:00:00")
        reference = Time("2026-01-01T00:00:00")

        seq1 = self._make_mock_sequence("Target_A", 100.0, 20.0, start_time1)
        seq2 = self._make_mock_sequence("Target_B", 200.0, -10.0, start_time2)
        visit = self._make_mock_visit([seq1, seq2])

        rolls_with_ref = calculate_visit_rolls(visit, reference_time=reference)
        rolls_without_ref = calculate_visit_rolls(visit)

        # Rolls should differ when using different reference times
        assert rolls_with_ref["Target_A"] != rolls_without_ref["Target_A"]


class TestApplyRollsToVisit:
    """Tests for applying rolls to visit sequences."""

    def _make_mock_sequence(self, target, ra, dec, start_time):
        """Create a mock observation sequence."""
        seq = MagicMock()
        seq.target = target
        seq.ra = ra
        seq.dec = dec
        seq.start_time = start_time
        seq.roll = None  # Will be set by apply_rolls_to_visit
        return seq

    def _make_mock_visit(self, sequences):
        """Create a mock visit with given sequences."""
        visit = MagicMock()
        visit.sequences = sequences
        return visit

    def test_apply_rolls_sets_roll_attribute(self):
        """Applying rolls should set roll attribute on sequences."""
        start_time = Time("2026-06-15T12:00:00")
        seq = self._make_mock_sequence("HD_123456", 120.0, 30.0, start_time)
        visit = self._make_mock_visit([seq])

        apply_rolls_to_visit(visit)

        assert seq.roll is not None
        assert isinstance(seq.roll, float)

    def test_apply_precomputed_rolls(self):
        """Providing target_rolls dict should use those values."""
        start_time = Time("2026-06-15T12:00:00")
        seq = self._make_mock_sequence("HD_123456", 120.0, 30.0, start_time)
        visit = self._make_mock_visit([seq])

        precomputed = {"HD_123456": 42.5}
        apply_rolls_to_visit(visit, target_rolls=precomputed)

        assert seq.roll == 42.5

    def test_same_target_gets_same_roll(self):
        """All sequences of same target should get the same roll."""
        start_time = Time("2026-06-15T12:00:00")
        seq1 = self._make_mock_sequence("HD_123456", 120.0, 30.0, start_time)
        seq2 = self._make_mock_sequence(
            "HD_123456", 120.0, 30.0, Time("2026-06-15T14:00:00")
        )
        visit = self._make_mock_visit([seq1, seq2])

        apply_rolls_to_visit(visit)

        assert seq1.roll == seq2.roll


class TestApplyRollsToCalendar:
    """Tests for calendar-level roll application."""

    def _make_mock_sequence(self, target, ra, dec, start_time):
        """Create a mock observation sequence."""
        seq = MagicMock()
        seq.target = target
        seq.ra = ra
        seq.dec = dec
        seq.start_time = start_time
        seq.roll = None
        return seq

    def _make_mock_visit(self, visit_id, sequences):
        """Create a mock visit with given sequences."""
        visit = MagicMock()
        visit.id = visit_id
        visit.sequences = sequences
        return visit

    def _make_mock_calendar(self, visits):
        """Create a mock calendar with given visits."""
        calendar = MagicMock()
        calendar.visits = visits
        return calendar

    def test_apply_rolls_to_all_visits(self):
        """Rolls should be applied to all visits in calendar."""
        start_time = Time("2026-06-15T12:00:00")

        seq1 = self._make_mock_sequence("Target_A", 100.0, 20.0, start_time)
        seq2 = self._make_mock_sequence("Target_B", 200.0, -10.0, start_time)

        visit1 = self._make_mock_visit("001", [seq1])
        visit2 = self._make_mock_visit("002", [seq2])

        calendar = self._make_mock_calendar([visit1, visit2])

        apply_rolls_to_calendar(calendar)

        assert seq1.roll is not None
        assert seq2.roll is not None

    def test_verbose_mode(self, capsys):
        """Verbose mode should print progress information."""
        start_time = Time("2026-06-15T12:00:00")
        seq = self._make_mock_sequence("Target_A", 100.0, 20.0, start_time)
        visit = self._make_mock_visit("001", [seq])
        calendar = self._make_mock_calendar([visit])

        apply_rolls_to_calendar(calendar, verbose=True)

        captured = capsys.readouterr()
        assert "visit 001" in captured.out
        assert "Target_A" in captured.out


class TestRollPhysicalReasonableness:
    """Tests to verify roll calculations are physically reasonable."""

    def test_roll_reasonable_pointing_north_pole(self):
        """Roll should be well-defined when pointing near celestial north pole."""
        obs_time = Time("2026-06-01T12:00:00", scale="utc")
        # Point near north celestial pole (Dec = 89.9)
        roll = calculate_roll(ra=0.0, dec=89.9, obs_time=obs_time)
        # Roll should still be a valid number in [0, 360)
        assert isinstance(roll, float)
        assert 0.0 <= roll < 360.0
        assert not np.isnan(roll)
        assert not np.isinf(roll)

    def test_roll_changes_with_target_position(self):
        """Different target positions should generally give different rolls."""
        obs_time = Time("2026-06-15T12:00:00")

        # Targets at different positions
        roll_at_0_0 = calculate_roll(ra=0.0, dec=0.0, obs_time=obs_time)
        roll_at_180_0 = calculate_roll(ra=180.0, dec=0.0, obs_time=obs_time)

        # Opposite sides of sky should have different rolls
        assert roll_at_0_0 != roll_at_180_0

    def test_roll_symmetry(self):
        """Roll calculation should be deterministic."""
        obs_time = Time("2026-06-15T12:00:00")

        # Calculate same roll multiple times
        rolls = [
            calculate_roll(ra=150.0, dec=25.0, obs_time=obs_time)
            for _ in range(5)
        ]

        # All should be identical
        assert len(set(rolls)) == 1

    def test_anti_sun_pointing(self):
        """Roll should be valid for anti-Sun pointing (Pandora's operational case)."""
        from shortschedule.roll import _spacecraft_roll_from_radec

        # Exact anti-Sun: target 180° from Sun
        target_ra, target_dec = 270.0, 0.0
        sun_ra, sun_dec = 90.0, 0.0

        roll, (xB, yB, zB) = _spacecraft_roll_from_radec(
            target_ra, target_dec, sun_ra, sun_dec
        )

        # Roll should be a valid number
        assert isinstance(roll, float)
        assert not np.isnan(roll)
        assert not np.isinf(roll)

        # Body frame should be orthonormal
        assert np.isclose(np.linalg.norm(xB), 1.0)
        assert np.isclose(np.linalg.norm(yB), 1.0)
        assert np.isclose(np.linalg.norm(zB), 1.0)
        assert np.isclose(np.dot(xB, yB), 0.0, atol=1e-10)
        assert np.isclose(np.dot(yB, zB), 0.0, atol=1e-10)
        assert np.isclose(np.dot(xB, zB), 0.0, atol=1e-10)

        # Should be right-handed: xB × yB = zB
        cross = np.cross(xB, yB)
        np.testing.assert_array_almost_equal(cross, zB, decimal=10)
