"""
Utility functions for identifying and filtering lanes in SUMO simulations.

This module provides universal functions to distinguish between "real" lanes
(entry/exit lanes that are part of the traffic network) and SUMO overhead
lanes (internal junction lanes, etc.). It can dynamically discover lanes
from SUMO configuration files.
"""

from pathlib import Path
from xml.etree import ElementTree as ET


def _extract_edge_ids(root: ET.Element) -> set[str]:
    """Extract edge IDs from network XML root, excluding junction edges."""
    edge_ids = set()
    for edge in root.findall(".//edge"):
        edge_id = edge.get("id")
        if edge_id and not edge_id.startswith(":"):
            edge_ids.add(edge_id)
    return edge_ids


def _is_valid_main_lane(lane_id: str, edge_ids: set[str]) -> bool:
    """Check if lane ID belongs to a main edge."""
    if lane_id.startswith(":"):
        return False

    lane_parts = lane_id.rsplit("_", 1)
    if len(lane_parts) == 2 and lane_parts[1].isdigit():
        return lane_parts[0] in edge_ids

    return lane_id in edge_ids


def discover_main_lanes_from_network(intersection_dir: Path) -> set[str]:
    """
    Dynamically discover main entry/exit lanes from SUMO network files.

    This function parses the network.net.xml file to find all lanes that belong
    to edges defined in edges.edg.xml. It excludes junction lanes
    (starting with ':').

    Args:
        intersection_dir: Path to intersection directory
                          (e.g., sim/intersections/simple4)

    Returns:
        Set of lane IDs that are considered "main" lanes (entry/exit lanes)

    Raises:
        FileNotFoundError: If network.net.xml is not found
        ValueError: If unable to parse network files
    """
    net_file = intersection_dir / "network.net.xml"

    if not net_file.exists():
        raise FileNotFoundError(
            f"Network file not found: {net_file}\n"
            "Run netconvert first to generate network.net.xml"
        )

    try:
        tree = ET.parse(net_file)
        root = tree.getroot()

        edge_ids = _extract_edge_ids(root)
        main_lanes: set[str] = set()
        for lane in root.findall(".//lane"):
            lane_id = lane.get("id")
            if lane_id and _is_valid_main_lane(lane_id, edge_ids):
                main_lanes.add(lane_id)

        return main_lanes

    except ET.ParseError as e:
        raise ValueError(f"Failed to parse network file {net_file}: {e}")
    except Exception as e:
        raise ValueError(f"Error discovering lanes from network: {e}")


def discover_main_lanes_from_edges(intersection_dir: Path) -> set[str]:
    """
    Discover main lanes by parsing edges.edg.xml and inferring lane IDs.

    This is a fallback method if network.net.xml is not available.
    It assumes each edge has at least one lane with index 0.

    Args:
        intersection_dir: Path to intersection directory

    Returns:
        Set of lane IDs (format: {edge_id}_0 for each edge)
    """
    edges_file = intersection_dir / "edges.edg.xml"

    if not edges_file.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_file}")

    try:
        tree = ET.parse(edges_file)
        root = tree.getroot()

        # Get all edge IDs (excluding junction edges that start with ':')
        edge_ids = set()
        for edge in root.findall(".//edge"):
            edge_id = edge.get("id")
            if edge_id and not edge_id.startswith(":"):
                edge_ids.add(edge_id)

        # Generate lane IDs: typically {edge_id}_0 for the first lane
        # SUMO may create multiple lanes per edge, but we start with _0
        main_lanes = {f"{edge_id}_0" for edge_id in edge_ids}

        return main_lanes

    except ET.ParseError as e:
        raise ValueError(f"Failed to parse edges file {edges_file}: {e}")


def get_main_lanes_for_intersection(
    intersection_name: str, intersection_base_dir: Path | None = None
) -> set[str]:
    """
    Get the main lanes for a given intersection.

    This function automatically discovers lanes from SUMO network files,
    making it work with any intersection without hardcoding.

    Args:
        intersection_name: Name of the intersection (e.g., 'simple4')
        intersection_base_dir: Base directory for intersections
                                (default: sim/intersections)

    Returns:
        Set of lane IDs that are considered "real" lanes

    Raises:
        FileNotFoundError: If intersection directory or network files not found
        ValueError: If unable to discover lanes
    """
    if intersection_base_dir is None:
        # Assume we're in the project root or sim directory
        current_file = Path(__file__)
        # Navigate from sim/sumo/lane_utils.py to sim/intersections
        intersection_base_dir = current_file.parent.parent / "intersections"

    intersection_dir = intersection_base_dir / intersection_name

    if not intersection_dir.exists():
        raise FileNotFoundError(
            f"Intersection directory not found: {intersection_dir}\n"
            f"Expected intersection name: {intersection_name}"
        )

    # Try to discover from network.net.xml first (most accurate)
    try:
        return discover_main_lanes_from_network(intersection_dir)
    except FileNotFoundError:
        # Fall back to edges.edg.xml if network.net.xml doesn't exist
        try:
            return discover_main_lanes_from_edges(intersection_dir)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Neither network.net.xml nor edges.edg.xml found in "
                f"{intersection_dir}\n"
                "Please generate network files first "
                "(run netconvert or generate_sumocfg)"
            )


def is_real_lane(
    lane_id: str,
    intersection_name: str | None = None,
    intersection_base_dir: Path | None = None,
) -> bool:
    """
    Universal function to identify if a lane is a "real" lane or SUMO overhead.

    Rules:
    - Junction lanes (start with ':') are always SUMO overhead
    - If intersection_name is provided, check against discovered main lanes
    - Otherwise, use pattern matching (conservative approach)

    Args:
        lane_id: Lane ID to check
        intersection_name: Optional intersection name for specific filtering
        intersection_base_dir: Base directory for intersections

    Returns:
        True if lane is a "real" lane, False if it's SUMO overhead
    """
    # Junction lanes are always SUMO overhead
    if lane_id.startswith(":"):
        return False

    # If intersection name provided, check against discovered main lanes
    if intersection_name:
        try:
            main_lanes = get_main_lanes_for_intersection(
                intersection_name, intersection_base_dir
            )
            return lane_id in main_lanes
        except (FileNotFoundError, ValueError):
            # Fall back to pattern matching if discovery fails
            pass

    # Pattern matching: entry/exit lanes starting with 'e' are typically real
    # But we want to be conservative - only accept known patterns
    if lane_id.startswith("e"):
        # Accept patterns like: eN_0, eS_0, eW_0, eE_0, eN_out_0, etc.
        # Reject internal lanes that might have complex patterns
        parts = lane_id.split("_")
        if len(parts) <= 3:  # eN_0 or eN_out_0 format
            return True

    # Default: reject unknown patterns
    return False


def filter_lanes_by_intersection(
    lane_ids: list[str],
    intersection_name: str,
    intersection_base_dir: Path | None = None,
    keep_main_only: bool = True,
) -> list[str]:
    """
    Filter a list of lane IDs to keep only main lanes for a given intersection.

    Args:
        lane_ids: List of lane IDs to filter
        intersection_name: Name of the intersection
        intersection_base_dir: Base directory for intersections
        keep_main_only: If True, keep only main lanes; if False, keep all
                        non-junction lanes

    Returns:
        Filtered list of lane IDs
    """
    if keep_main_only:
        try:
            main_lanes = get_main_lanes_for_intersection(
                intersection_name, intersection_base_dir
            )
            return [lid for lid in lane_ids if lid in main_lanes]
        except (FileNotFoundError, ValueError):
            # Fall back to excluding junction lanes only
            return [lid for lid in lane_ids if not lid.startswith(":")]
    else:
        # Keep all non-junction lanes
        return [lid for lid in lane_ids if not lid.startswith(":")]
