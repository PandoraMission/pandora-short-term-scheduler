"""XML writer utilities for shortschedule.

This module contains `XMLWriter`, a small helper to serialize
`ScienceCalendar` objects back into a PAN-SCICAL compliant XML file.
The writer preserves payload XML elements and copies them into the
`Payload_Parameters` section of each `Observation_Sequence`.
"""

# Standard library
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta


class XMLWriter:
    """Class for writing processed science calendars back to XML format.

    The writer provides `write_calendar(calendar, output_path=None, ...)` which
    either writes to the provided `output_path` or generates a filename using
    the PAN naming convention.
    """

    def __init__(self):
        self.namespace = "/pandora/calendar/"

    def write_calendar(
        self,
        calendar,
        output_path=None,
        mission_phase="TST",
        revision=1,
        verbose=False,
    ):
        """
        Write science calendar to XML file with proper naming convention.

        Parameters:
        -----------
        calendar : ScienceCalendar
            Calendar to write
        output_path : str, optional
            Full output path. If None, generates filename automatically
        mission_phase : str
            Mission phase code: 'TST', 'COM', or 'OPS' (default: 'TST')
        revision : int
            Revision number (default: 1)
        verbose : bool
            Print writing details

        Returns:
        --------
        str
            Path to written file
        """
        if output_path is None:
            output_path = self._generate_filename(
                calendar, mission_phase, revision
            )

        if verbose:
            print(f"Writing calendar to: {output_path}")

        # Create root element with namespace
        root = ET.Element("ScienceCalendar")
        root.set("xmlns", self.namespace)

        # Add metadata
        self._add_metadata(root, calendar.metadata)

        # Add visits and sequences
        for visit in calendar.visits:
            self._add_visit(root, visit)

        # Write to file with proper formatting
        self._write_formatted_xml(root, output_path)

        if verbose:
            total_sequences = sum(
                len(visit.sequences) for visit in calendar.visits
            )  # FIXED LINE
            print(
                f"Written {len(calendar.visits)} visits with {total_sequences} sequences"
            )

        return output_path

    def _generate_filename(self, calendar, mission_phase, revision):
        """Generate filename following PAN-SCICAL naming convention."""
        now = datetime.now()

        # Extract dates from metadata
        valid_from = calendar.metadata.get("valid_from", "")
        expires = calendar.metadata.get("expires", "")

        # Parse dates or use defaults
        try:
            if valid_from:
                # Handle different date formats
                if "T" in valid_from:
                    vf_date = datetime.fromisoformat(
                        valid_from.replace("Z", "+00:00")
                    )
                else:
                    vf_date = datetime.strptime(
                        valid_from, "%Y-%m-%d %H:%M:%S"
                    )
                vf_str = vf_date.strftime("%Y%m%d")
            else:
                vf_str = now.strftime("%Y%m%d")
        except Exception:
            vf_str = now.strftime("%Y%m%d")

        try:
            if expires:
                # Handle different date formats
                if "T" in expires:
                    ex_date = datetime.fromisoformat(
                        expires.replace("Z", "+00:00")
                    )
                else:
                    ex_date = datetime.strptime(expires, "%Y-%m-%d %H:%M:%S")
                ex_str = ex_date.strftime("%Y%m%d")
            else:
                ex_str = (now + timedelta(days=21)).strftime("%Y%m%d")
        except Exception:
            ex_str = (now + timedelta(days=21)).strftime("%Y%m%d")

        # Generate filename
        gen_date = now.strftime("%Y%m%d")
        filename = f"PAN-SCICAL-{mission_phase}-{gen_date}-VF-{vf_str}-EX-{ex_str}-R{revision:03d}.xml"

        return filename

    def _add_metadata(self, root, metadata):
        """Add metadata element to root."""
        meta = ET.SubElement(root, "Meta")

        # Standard metadata fields mapping
        meta_mapping = {
            "valid_from": "Valid_From",
            "expires": "Expires",
            "created": "Created",
            "delivery_id": "Delivery_Id",
            "total_visits": "Total_Visits",
            "total_sequences": "Total_Sequences",
            "calendar_status": "Calendar_Status",
        }

        for internal_name, xml_attr in meta_mapping.items():
            if internal_name in metadata and metadata[internal_name]:
                meta.set(xml_attr, str(metadata[internal_name]))

        # Add TLE information if present
        if "tle_line1" in metadata:
            meta.set("TLE_Line1", metadata["tle_line1"])
        if "tle_line2" in metadata:
            meta.set("TLE_Line2", metadata["tle_line2"])

    def _add_visit(self, root, visit):
        """Add visit element with all observation sequences."""
        visit_elem = ET.SubElement(root, "Visit")

        # Add visit ID
        id_elem = ET.SubElement(visit_elem, "ID")
        id_elem.text = str(visit.id)

        # Add all observation sequences
        for sequence in visit.sequences:
            self._add_observation_sequence(visit_elem, sequence)

    def _add_observation_sequence(self, visit_elem, sequence):
        """Add observation sequence element."""
        seq_elem = ET.SubElement(visit_elem, "Observation_Sequence")

        # Sequence ID
        id_elem = ET.SubElement(seq_elem, "ID")
        id_elem.text = str(sequence.id)

        # Observational Parameters
        obs_params = ET.SubElement(seq_elem, "Observational_Parameters")

        # Target
        target_elem = ET.SubElement(obs_params, "Target")
        target_elem.text = sequence.target

        # Priority
        priority_elem = ET.SubElement(obs_params, "Priority")
        priority_elem.text = str(sequence.priority)

        # Timing
        timing_elem = ET.SubElement(obs_params, "Timing")
        start_elem = ET.SubElement(timing_elem, "Start")
        start_elem.text = sequence.start_time_str  # Using the new property
        stop_elem = ET.SubElement(timing_elem, "Stop")
        stop_elem.text = sequence.stop_time_str  # Using the new property

        # Boresight
        boresight_elem = ET.SubElement(obs_params, "Boresight")
        ra_elem = ET.SubElement(boresight_elem, "RA")
        ra_elem.text = str(sequence.ra)
        dec_elem = ET.SubElement(boresight_elem, "DEC")
        dec_elem.text = str(sequence.dec)

        # Payload Parameters - copy the XML elements directly
        self._add_payload_parameters(seq_elem, sequence.payload_params)

    def _add_payload_parameters(self, seq_elem, payload_params):
        """Add payload parameters section by copying XML elements."""
        payload_elem = ET.SubElement(seq_elem, "Payload_Parameters")

        # Copy each payload parameter XML element directly
        for param_name, xml_element in payload_params.items():
            if xml_element is not None:
                # Create a deep copy of the XML element
                copied_element = self._deep_copy_xml_element(xml_element)
                payload_elem.append(copied_element)

    def _deep_copy_xml_element(self, element):
        """Create a deep copy of an XML element."""
        # Create new element with same tag
        new_elem = ET.Element(element.tag, element.attrib)

        # Copy text content
        if element.text:
            new_elem.text = element.text
        if element.tail:
            new_elem.tail = element.tail

        # Recursively copy all children
        for child in element:
            new_elem.append(self._deep_copy_xml_element(child))

        return new_elem

    def _write_formatted_xml(self, root, output_path):
        """Write XML with proper formatting."""
        # Create the tree
        tree = ET.ElementTree(root)

        # Write with XML declaration
        with open(output_path, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)

        # Read back and reformat for pretty printing
        self._pretty_print_xml(output_path)

    def _pretty_print_xml(self, file_path):
        """Add proper indentation to XML file."""
        try:
            # Standard library
            import xml.dom.minidom

            # Parse and pretty print
            dom = xml.dom.minidom.parse(file_path)
            pretty_xml = dom.toprettyxml(indent="\t", encoding="utf-8")

            # Remove extra blank lines and fix formatting
            lines = pretty_xml.decode("utf-8").split("\n")
            filtered_lines = []

            for line in lines:
                # Skip empty lines but keep lines with just whitespace/tabs that have content structure
                if line.strip() or (
                    not filtered_lines
                ):  # Keep first line (XML declaration)
                    filtered_lines.append(line)

            # Remove any trailing empty lines
            while filtered_lines and not filtered_lines[-1].strip():
                filtered_lines.pop()

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(filtered_lines))
                f.write("\n")  # Ensure file ends with newline

        except Exception as e:
            # If pretty printing fails, file is still valid XML
            print(f"Warning: Could not pretty print XML: {e}")


def write_science_calendar(calendar, output_path, **kwargs):
    """
    Convenience function to write a science calendar.

    Parameters:
    -----------
    calendar : ScienceCalendar
        Calendar to write
    output_path : str
        Output file path
    **kwargs
        Additional arguments passed to XMLWriter.write_calendar()

    Returns:
    --------
    str
        Path to written file
    """
    writer = XMLWriter()
    return writer.write_calendar(calendar, output_path, **kwargs)
