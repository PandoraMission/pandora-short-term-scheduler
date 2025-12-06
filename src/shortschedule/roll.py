"""Spacecraft roll angle calculations for Pandora observations.

This module provides utilities to compute the spacecraft roll angle
required for each observation sequence. The roll angle depends on:
- Target coordinates (RA, Dec)
- Sun position at the time of observation

Roll must remain consistent for all
observations of the same target within a single visit.

Notes
-----
Roll angles are expressed in degrees. The calculation ensures the
solar panels maintain proper sun exposure.
"""

# Standard library
from typing import Dict, Optional

# Third-party
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time


def calculate_roll(
    ra: float,
    dec: float,
    obs_time: Time,
) -> float:
    """Calculate the spacecraft roll angle for a given target and time.

    The roll angle is computed based on the position of the target
    relative to the Sun at the time of observation. This ensures
    proper solar panel orientation.

    Parameters
    ----------
    ra : float
        Right ascension of the target in degrees.
    dec : float
        Declination of the target in degrees.
    obs_time : Time
        The time of observation (used to determine Sun position).

    Returns
    -------
    float
        The spacecraft roll angle in degrees, in the range [0, 360).
    """

    # Get target coordinates
    target_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    # Get Sun position at observation time (astropy >=5 returns GCRS)
    sun_coord = get_sun(obs_time)
    # Convert to ICRS if needed
    if not hasattr(sun_coord, "ra") or not hasattr(sun_coord, "dec"):
        sun_coord = sun_coord.transform_to("icrs")

    # Calculate roll angle
    roll, _ = _spacecraft_roll_from_radec(
        target_coord.ra.deg,
        target_coord.dec.deg,
        sun_coord.ra.deg,
        sun_coord.dec.deg,
    )
    roll = roll % 360.0  # Normalize to [0, 360)
    return roll


def _spacecraft_roll_from_radec(target_ra, target_dec, sun_ra, sun_dec):
    """
    Compute spacecraft roll about boresight defined by:
        zB = target direction
        yB = sun × target (or fallback for anti-Solar pointing)
        xB = yB × zB
    All in ECI.

    For when the target is nearly anti-Solar, we use celestial north projected onto the plane
    perpendicular to the boresight as the reference for yB.
    """
    # Convert to vectors
    t = radec_to_vector(target_ra, target_dec)  # boresight
    s = radec_to_vector(sun_ra, sun_dec)  # Sun direction

    # boresight body axes, we define the others later because of edge cases
    zB = normalize(t)

    # Celestial north in ECI (equatorial frame) is +Z_ECI
    north = np.array([0.0, 0.0, 1.0])
    north_proj = north - np.dot(north, zB) * zB

    # catch edge case where boresight is very close to celestial north
    # in practice this should never happen because of Sun constraints
    if np.linalg.norm(north_proj) < 1.0e-8:
        # boresight ≈ celestial north: pick arbitrary east-like vector
        east = np.array([1.0, 0.0, 0.0])
        north_proj = east - np.dot(east, zB) * zB

    # reference axes for roll angle calculation
    x_ref = normalize(north_proj)
    y_ref = normalize(np.cross(zB, x_ref))

    # Define spacecraft xB,yB using the Sun direction
    sun_cross = np.cross(s, zB)
    if np.linalg.norm(sun_cross) > 1e-6:
        yB = normalize(sun_cross)
        xB = normalize(np.cross(yB, zB))
    else:
        # we need to catch edge cases where Sun is very close to anti-boresight
        xB = x_ref
        yB = y_ref

    # Compute roll angle x_ref -> xB around zB
    cos_r = np.dot(x_ref, xB)
    sin_r = np.dot(y_ref, xB)
    roll = np.rad2deg(np.arctan2(sin_r, cos_r))  # degrees

    return roll, (xB, yB, zB)


def radec_to_vector(ra_deg, dec_deg):
    """Convert RA/Dec (deg) to an Earth-Centered Inertial (ECI) coordinate frame unit vector."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y, z])


def vector_to_radec(v):
    """Convert an ECI coordinate frame unit vector to RA/Dec (deg)."""
    x, y, z = v
    r = np.sqrt(x**2 + y**2 + z**2)
    dec = np.arcsin(z / r)
    ra = np.arctan2(y, x)
    ra_deg = np.rad2deg(ra) % 360.0
    dec_deg = np.rad2deg(dec)
    return ra_deg, dec_deg


def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n


def calculate_visit_rolls(
    visit,
    reference_time: Optional[Time] = None,
) -> Dict[str, float]:
    """Calculate roll angles for all unique targets in a visit.

    This function ensures that the same roll angle is used for all
    observation sequences of the same target within a visit, as
    required by spacecraft pointing constraints.

    Parameters
    ----------
    visit : Visit
        The visit containing observation sequences.
    reference_time : Time, optional
        Reference time to use for roll calculation. If None, uses the
        start time of the first sequence for each target.

    Returns
    -------
    dict
        Mapping of target name to roll angle in degrees.
        Example: {"WASP-18": 45.2, "K2-18": 15.1}

    Notes
    -----

    """
    target_rolls: Dict[str, float] = {}

    # Group sequences by target
    target_sequences: Dict[str, list] = {}
    for seq in visit.sequences:
        if seq.target not in target_sequences:
            target_sequences[seq.target] = []
        target_sequences[seq.target].append(seq)

    # Calculate roll for each unique target
    for target, sequences in target_sequences.items():
        # Sort sequences by start time to use the earliest one
        sequences_sorted = sorted(sequences, key=lambda s: s.start_time)

        # Use provided reference time or earliest sequence start time
        if reference_time is not None:
            calc_time = reference_time
        else:
            calc_time = sequences_sorted[0].start_time

        # Get coordinates from first sequence (should be same for all)
        ra = sequences_sorted[0].ra
        dec = sequences_sorted[0].dec

        # Calculate roll
        roll = calculate_roll(ra, dec, calc_time)
        target_rolls[target] = roll

    return target_rolls


def apply_rolls_to_visit(
    visit,
    target_rolls: Optional[Dict[str, float]] = None,
) -> None:
    """Apply roll angles to all observation sequences in a visit.

    This function modifies the visit's sequences in place, setting
    the roll attribute on each ObservationSequence.

    Parameters
    ----------
    visit : Visit
        The visit to update. Modified in place.
    target_rolls : dict, optional
        Pre-calculated mapping of target name to roll angle.
        If None, rolls will be calculated using calculate_visit_rolls().

    Returns
    -------
    None
        The visit is modified in place.
    """
    if target_rolls is None:
        target_rolls = calculate_visit_rolls(visit)

    for seq in visit.sequences:
        if seq.target in target_rolls:
            seq.roll = target_rolls[seq.target]
        else:
            # Target not found - calculate individually
            seq.roll = calculate_roll(seq.ra, seq.dec, seq.start_time)


def apply_rolls_to_calendar(
    calendar,
    verbose: bool = False,
) -> None:
    """Apply roll angles to all visits in a science calendar.

    This is a convenience function that applies roll angles to all
    visits in a calendar, ensuring consistency within each visit.

    Parameters
    ----------
    calendar : ScienceCalendar
        The calendar to update. Modified in place.
    verbose : bool, optional
        If True, print progress information.

    Returns
    -------
    None
        The calendar is modified in place.
    """
    for visit in calendar.visits:
        if verbose:
            print(f"Calculating rolls for visit {visit.id}")

        target_rolls = calculate_visit_rolls(visit)
        apply_rolls_to_visit(visit, target_rolls)

        if verbose:
            for target, roll in target_rolls.items():
                print(f"  {target}: {roll:.2f} deg")
