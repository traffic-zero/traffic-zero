"""
XML generation from scenario configurations.

Generates SUMO XML files (routes.rou.xml and tls.add.xml) from Scenario objects.
"""

from pathlib import Path
from xml.etree import ElementTree as ET

from .scenario import Scenario


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
        vtype_elem = ET.SubElement(root, "vType")
        vtype_elem.set("id", vtype.id)
        vtype_elem.set("accel", str(vtype.accel))
        vtype_elem.set("decel", str(vtype.decel))
        vtype_elem.set("sigma", str(vtype.sigma))
        vtype_elem.set("length", str(vtype.length))
        vtype_elem.set("maxSpeed", str(vtype.maxSpeed))

        if vtype.color is not None:
            vtype_elem.set("color", vtype.color)
        if vtype.minGap is not None:
            vtype_elem.set("minGap", str(vtype.minGap))
        if vtype.tau is not None:
            vtype_elem.set("tau", str(vtype.tau))
        if vtype.speedFactor is not None:
            vtype_elem.set("speedFactor", str(vtype.speedFactor))
        if vtype.speedDev is not None:
            vtype_elem.set("speedDev", str(vtype.speedDev))

    # Routes
    for route in scenario.traffic.routes:
        route_elem = ET.SubElement(root, "route")
        route_elem.set("id", route.id)
        route_elem.set("edges", " ".join(route.edges))

    # Vehicles (programmatic)
    if scenario.traffic.vehicles:
        for vehicle in scenario.traffic.vehicles:
            veh_elem = ET.SubElement(root, "vehicle")
            veh_elem.set("id", vehicle.id)
            veh_elem.set("type", vehicle.type)
            veh_elem.set("route", vehicle.route)
            veh_elem.set("depart", str(vehicle.depart))

            if vehicle.departLane is not None:
                veh_elem.set("departLane", vehicle.departLane)
            if vehicle.departPos is not None:
                veh_elem.set("departPos", str(vehicle.departPos))
            if vehicle.departSpeed is not None:
                veh_elem.set("departSpeed", str(vehicle.departSpeed))
            if vehicle.arrivalLane is not None:
                veh_elem.set("arrivalLane", vehicle.arrivalLane)
            if vehicle.arrivalPos is not None:
                veh_elem.set("arrivalPos", str(vehicle.arrivalPos))
            if vehicle.arrivalSpeed is not None:
                veh_elem.set("arrivalSpeed", str(vehicle.arrivalSpeed))

    # Flows (statistical)
    if scenario.traffic.flows:
        # Track all used flow IDs to ensure uniqueness
        used_flow_ids = set()
        flow_counter = 0
        
        for flow in scenario.traffic.flows:
            flow_elem = ET.SubElement(root, "flow")
            
            # SUMO requires unique IDs for flows, especially when multiple flows
            # use the same route. Generate ID if not provided or if provided ID conflicts.
            if flow.id is not None and flow.id not in used_flow_ids:
                # Use provided ID if it's unique
                flow_id = flow.id
                flow_elem.set("id", flow_id)
                used_flow_ids.add(flow_id)
            else:
                # Auto-generate unique ID: flow_<route>_<counter>
                # Keep generating until we find a unique ID
                while True:
                    flow_id = f"flow_{flow.route}_{flow_counter}"
                    if flow_id not in used_flow_ids:
                        break
                    flow_counter += 1
                flow_elem.set("id", flow_id)
                used_flow_ids.add(flow_id)
                flow_counter += 1
            
            flow_elem.set("route", flow.route)
            flow_elem.set("type", flow.type)
            flow_elem.set("begin", str(flow.begin))
            flow_elem.set("end", str(flow.end))

            # Flow rate specification (mutually exclusive in SUMO, but we allow)
            if flow.vehsPerHour is not None:
                flow_elem.set("vehsPerHour", str(flow.vehsPerHour))
            if flow.period is not None:
                flow_elem.set("period", str(flow.period))
            if flow.probability is not None:
                flow_elem.set("probability", str(flow.probability))
            if flow.number is not None:
                flow_elem.set("number", str(flow.number))
            if flow.departLane is not None:
                flow_elem.set("departLane", flow.departLane)
            if flow.departPos is not None:
                flow_elem.set("departPos", str(flow.departPos))
            if flow.departSpeed is not None:
                flow_elem.set("departSpeed", str(flow.departSpeed))
            if flow.arrivalLane is not None:
                flow_elem.set("arrivalLane", flow.arrivalLane)
            if flow.arrivalPos is not None:
                flow_elem.set("arrivalPos", str(flow.arrivalPos))
            if flow.arrivalSpeed is not None:
                flow_elem.set("arrivalSpeed", str(flow.arrivalSpeed))

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

            if phase.minDur is not None:
                phase_elem.set("minDur", str(phase.minDur))
            if phase.maxDur is not None:
                phase_elem.set("maxDur", str(phase.maxDur))
            if phase.name is not None:
                phase_elem.set("name", phase.name)

    # Write to file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

