"""shortschedule.parser

Utilities to parse PAN-SCICAL science calendar XML files into the
shortschedule data model objects (ScienceCalendar, Visit,
ObservationSequence).

Primary entry point:
- parse_science_calendar(xml_path, verbose=False)

The parser preserves payload XML elements as ElementTree elements so the
payload structure can be modified programmatically and written back to XML
without losing attributes or nested elements.
"""

# Standard library
import xml.etree.ElementTree as ET

# Third-party
import numpy as np
from astropy.time import Time

from .models import ObservationSequence, ScienceCalendar, Visit


def parse_science_calendar(xml_path, verbose=False):
    """
    Parse a PAN-SCICAL Science Calendar XML file into a `ScienceCalendar` object.

    Parameters
    ----------
    xml_path : str
        Path to the PAN-SCICAL XML file.
    verbose : bool, optional
        If True, print progress messages.

    Returns
    -------
    ScienceCalendar
        In-memory representation of the calendar with visits and observation
        sequences. Payload parameter XML fragments are preserved as
        ElementTree elements under each sequence.

    Notes
    -----
    The parser tolerates missing optional sections and returns an empty
    `ScienceCalendar` when no visits are found. It uses the namespace
    '/pandora/calendar/' expected in PAN-SCICAL files.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Define the namespace
    namespace = {"pandora": "/pandora/calendar/"}

    if verbose:
        print(f"Parsing XML file: {xml_path}")
        print(f"Root tag: {root.tag}")

    # Parse metadata using namespace
    meta = root.find("pandora:Meta", namespace)

    if meta is not None:
        metadata = {
            "valid_from": meta.get("Valid_From"),
            "expires": meta.get("Expires"),
            "calendar_weights": meta.get("Calendar_Weights"),
            "ephemeris": meta.get("Ephemeris"),
            "keepout_angles": meta.get("Keepout_Angles"),
            "observation_sequence_duration": meta.get(
                "Observation_Sequence_Duration_hrs"
            ),
            "removed_sequences_shorter_than": meta.get(
                "Removed_Sequences_Shorter_Than_min"
            ),
            "created": meta.get("Created"),
            "delivery_id": meta.get("Delivery_Id"),
        }
        if verbose:
            print(
                f"Calendar valid from {metadata['valid_from']} to {metadata['expires']}"
            )
    else:
        if verbose:
            print("Warning: No Meta element found")
        metadata = {}

    # Parse visits using namespace
    visits = []
    visit_elements = root.findall("pandora:Visit", namespace)

    if verbose:
        print(f"Found {len(visit_elements)} visits")

    for visit_elem in visit_elements:
        visit_id_elem = visit_elem.find("pandora:ID", namespace)
        visit_id = (
            visit_id_elem.text if visit_id_elem is not None else "unknown"
        )

        # Parse observation sequences
        sequences = []
        seq_elements = visit_elem.findall(
            "pandora:Observation_Sequence", namespace
        )

        for seq_elem in seq_elements:
            try:
                sequence = _parse_observation_sequence(seq_elem, namespace)
                if sequence:
                    sequences.append(sequence)
            except Exception as e:
                if verbose:
                    print(
                        f"    Error parsing sequence in visit {visit_id}: {e}"
                    )
                continue

        if sequences:
            visit = Visit(id=visit_id, sequences=sequences)
            visits.append(visit)

    if verbose:
        total_sequences = np.sum([len(visit.sequences) for visit in visits])
        print(
            f"Successfully parsed {len(visits)} visits with {total_sequences} total sequences"
        )

    return ScienceCalendar(metadata=metadata, visits=visits)


def _parse_observation_sequence(seq_elem, namespace):
    """Parse a single `Observation_Sequence` XML element into an
    `ObservationSequence` object.

    Parameters
    ----------
    seq_elem : xml.etree.ElementTree.Element
        The XML element representing the observation sequence.
    namespace : dict
        Namespace mapping used for element lookups.

    Returns
    -------
    ObservationSequence or None
        The parsed sequence or ``None`` if required fields are missing.
    """
    seq_id_elem = seq_elem.find("pandora:ID", namespace)
    seq_id = seq_id_elem.text if seq_id_elem is not None else "unknown"

    obs_params_elem = seq_elem.find(
        "pandora:Observational_Parameters", namespace
    )
    if obs_params_elem is None:
        return None

    target_elem = obs_params_elem.find("pandora:Target", namespace)
    target = target_elem.text if target_elem is not None else "unknown"

    priority_elem = obs_params_elem.find("pandora:Priority", namespace)
    priority = int(priority_elem.text) if priority_elem is not None else 0

    # --- Parse timing ---
    timing_elem = obs_params_elem.find("pandora:Timing", namespace)
    if timing_elem is None:
        return None

    start_elem = timing_elem.find("pandora:Start", namespace)
    stop_elem = timing_elem.find("pandora:Stop", namespace)
    if start_elem is None or stop_elem is None:
        return None

    # Use Astropy Time directly
    start_time = Time(start_elem.text.strip(), format="isot", scale="utc")
    stop_time = Time(stop_elem.text.strip(), format="isot", scale="utc")

    # --- Boresight ---
    boresight_elem = obs_params_elem.find("pandora:Boresight", namespace)
    if boresight_elem is not None:
        ra_elem = boresight_elem.find("pandora:RA", namespace)
        dec_elem = boresight_elem.find("pandora:DEC", namespace)
        ra = float(ra_elem.text) if ra_elem is not None else 0.0
        dec = float(dec_elem.text) if dec_elem is not None else 0.0
    else:
        ra, dec = 0.0, 0.0

    # --- Payload parameters ---
    payload_params_elem = seq_elem.find(
        "pandora:Payload_Parameters", namespace
    )
    payload_params = {}

    if payload_params_elem is not None:
        for child in payload_params_elem:
            tag_name = (
                child.tag.split("}")[-1] if "}" in child.tag else child.tag
            )
            clean_element = _create_clean_element_copy(child)
            payload_params[tag_name] = clean_element

    return ObservationSequence(
        id=seq_id,
        target=target,
        priority=priority,
        start_time=start_time,
        stop_time=stop_time,
        ra=ra,
        dec=dec,
        payload_params=payload_params,
    )


def _create_clean_element_copy(element):
    """Create a clean copy of element without namespace prefixes."""
    tag_name = (
        element.tag.split("}")[-1] if "}" in element.tag else element.tag
    )
    new_element = ET.Element(tag_name)

    if element.text and element.text.strip():
        new_element.text = element.text

    for attr_name, attr_value in element.attrib.items():
        clean_attr_name = (
            attr_name.split("}")[-1] if "}" in attr_name else attr_name
        )
        new_element.set(clean_attr_name, attr_value)

    for child in element:
        clean_child = _create_clean_element_copy(child)
        new_element.append(clean_child)

    return new_element


def _get_element_text(element, child_path, namespace=None, default=None):
    """Safely get text content from an element's child."""
    if element is None:
        return default

    if namespace:
        child = element.find(f"pandora:{child_path}", namespace)
    else:
        child = element.find(child_path)

    if child is not None and child.text:
        return child.text
    return default


def parse_xml_element(xml_string):
    """Parse an XML string into the library's element format."""
    root = ET.fromstring(xml_string)
    return _create_clean_element_copy(root)


def write_science_calendar(calendar, output_path):
    """Write science calendar to XML file."""
    from .writer import XMLWriter

    writer = XMLWriter()
    return writer.write_calendar(calendar, output_path)
