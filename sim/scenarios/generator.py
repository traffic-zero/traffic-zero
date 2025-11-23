"""
XML generation from scenario configurations.

Generates SUMO XML files (routes.rou.xml and tls.add.xml) from Scenario objects.
"""

from pathlib import Path
from xml.etree import ElementTree as ET

from .scenario import Scenario


def _add_vehicle_type_to_xml(root: ET.Element, vtype) -> None:
    """Add vehicle type element to XML root."""
    vtype_elem = ET.SubElement(root, "vType")
    vtype_elem.set("id", vtype.id)
    vtype_elem.set("accel", str(vtype.accel))
    vtype_elem.set("decel", str(vtype.decel))
    vtype_elem.set("sigma", str(vtype.sigma))
    vtype_elem.set("length", str(vtype.length))
    vtype_elem.set("maxSpeed", str(vtype.max_speed))

    if vtype.color is not None:
        vtype_elem.set("color", vtype.color)
    if vtype.min_gap is not None:
        vtype_elem.set("minGap", str(vtype.min_gap))
    if vtype.tau is not None:
        vtype_elem.set("tau", str(vtype.tau))
    if vtype.speed_factor is not None:
        vtype_elem.set("speedFactor", str(vtype.speed_factor))
    if vtype.speed_dev is not None:
        vtype_elem.set("speedDev", str(vtype.speed_dev))


def _add_vehicle_to_xml(root: ET.Element, vehicle) -> None:
    """Add vehicle element to XML root."""
    veh_elem = ET.SubElement(root, "vehicle")
    veh_elem.set("id", vehicle.id)
    veh_elem.set("type", vehicle.type)
    veh_elem.set("route", vehicle.route)
    veh_elem.set("depart", str(vehicle.depart))

    if vehicle.depart_lane is not None:
        veh_elem.set("departLane", vehicle.depart_lane)
    if vehicle.depart_pos is not None:
        veh_elem.set("departPos", str(vehicle.depart_pos))
    if vehicle.depart_speed is not None:
        veh_elem.set("departSpeed", str(vehicle.depart_speed))
    if vehicle.arrival_lane is not None:
        veh_elem.set("arrivalLane", vehicle.arrival_lane)
    if vehicle.arrival_pos is not None:
        veh_elem.set("arrivalPos", str(vehicle.arrival_pos))
    if vehicle.arrival_speed is not None:
        veh_elem.set("arrivalSpeed", str(vehicle.arrival_speed))


def _generate_unique_flow_id(
    flow, used_flow_ids: set[str], flow_counter: int
) -> tuple[str, int]:
    """Generate unique flow ID, updating counter and used set."""
    if flow.id is not None and flow.id not in used_flow_ids:
        flow_id = flow.id
        used_flow_ids.add(flow_id)
        return flow_id, flow_counter

    # Auto-generate unique ID
    while True:
        flow_id = f"flow_{flow.route}_{flow_counter}"
        if flow_id not in used_flow_ids:
            break
        flow_counter += 1
    used_flow_ids.add(flow_id)
    return flow_id, flow_counter + 1


def _add_flow_to_xml(
    root: ET.Element, flow, used_flow_ids: set[str], flow_counter: int
) -> int:
    """Add flow element to XML root, return updated flow_counter."""
    flow_elem = ET.SubElement(root, "flow")

    flow_id, flow_counter = _generate_unique_flow_id(
        flow, used_flow_ids, flow_counter
    )
    flow_elem.set("id", flow_id)
    flow_elem.set("route", flow.route)
    flow_elem.set("type", flow.type)
    flow_elem.set("begin", str(flow.begin))
    flow_elem.set("end", str(flow.end))

    # Flow rate specification
    if flow.vehs_per_hour is not None:
        flow_elem.set("vehsPerHour", str(flow.vehs_per_hour))
    if flow.period is not None:
        flow_elem.set("period", str(flow.period))
    if flow.probability is not None:
        flow_elem.set("probability", str(flow.probability))
    if flow.number is not None:
        flow_elem.set("number", str(flow.number))
    if flow.depart_lane is not None:
        flow_elem.set("departLane", flow.depart_lane)
    if flow.depart_pos is not None:
        flow_elem.set("departPos", str(flow.depart_pos))
    if flow.depart_speed is not None:
        flow_elem.set("departSpeed", str(flow.depart_speed))
    if flow.arrival_lane is not None:
        flow_elem.set("arrivalLane", flow.arrival_lane)
    if flow.arrival_pos is not None:
        flow_elem.set("arrivalPos", str(flow.arrival_pos))
    if flow.arrival_speed is not None:
        flow_elem.set("arrivalSpeed", str(flow.arrival_speed))

    return flow_counter


def generate_routes_xml(scenario: Scenario, output_path: Path) -> None:
    """
    Generate routes.rou.xml file from scenario.

    Args:
        scenario: Scenario object containing traffic configuration
        output_path: Path where routes.rou.xml should be written
    """
    root = ET.Element("routes")

    # Vehicle types
    for vtype in scenario.traffic.vehicle_types:
        _add_vehicle_type_to_xml(root, vtype)

    # Routes
    for route in scenario.traffic.routes:
        route_elem = ET.SubElement(root, "route")
        route_elem.set("id", route.id)
        route_elem.set("edges", " ".join(route.edges))

    # Vehicles (programmatic)
    if scenario.traffic.vehicles:
        for vehicle in scenario.traffic.vehicles:
            _add_vehicle_to_xml(root, vehicle)

    # Flows (statistical)
    if scenario.traffic.flows:
        used_flow_ids = set()
        flow_counter = 0
        for flow in scenario.traffic.flows:
            flow_counter = _add_flow_to_xml(
                root, flow, used_flow_ids, flow_counter
            )

    # Write to file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def generate_tls_xml(scenario: Scenario, output_path: Path) -> None:
    """
    Generate tls.add.xml file from scenario.

    Args:
        scenario: Scenario object containing traffic light configuration
        output_path: Path where tls.add.xml should be written
    """
    root = ET.Element("additional")

    for program in scenario.traffic_lights.programs:
        tls_elem = ET.SubElement(root, "tlLogic")
        tls_elem.set("id", program.id)
        tls_elem.set("type", program.type)
        tls_elem.set("programID", program.program_id)
        tls_elem.set("offset", str(program.offset))

        for phase in program.phases:
            phase_elem = ET.SubElement(tls_elem, "phase")
            phase_elem.set("duration", str(phase.duration))
            phase_elem.set("state", phase.state)

            if phase.min_dur is not None:
                phase_elem.set("minDur", str(phase.min_dur))
            if phase.max_dur is not None:
                phase_elem.set("maxDur", str(phase.max_dur))
            if phase.name is not None:
                phase_elem.set("name", phase.name)

    # Write to file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
