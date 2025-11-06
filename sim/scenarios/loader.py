"""
YAML loader and saver for scenario configurations with JSON schema validation.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

import jsonschema
import yaml

from .scenario import (
    Scenario,
    TrafficConfig,
    TrafficLightConfig,
    VehicleType,
    Route,
    Vehicle,
    Flow,
    Phase,
    TrafficLightProgram,
)


def _get_schema_path() -> Path:
    """Get the path to the JSON schema file."""
    # Schema is in experiments/scenarios/schema.json
    # This file is in sim/scenarios/, so go up to project root
    project_root = Path(__file__).parent.parent.parent
    return project_root / "experiments" / "scenarios" / "schema.json"


def _get_experiments_dir() -> Path:
    """Get the path to the experiments directory."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "experiments"


def validate_scenario(data: Dict[str, Any], schema_path: Optional[Path] = None) -> None:
    """
    Validate scenario data against JSON schema.

    Args:
        data: Dictionary containing scenario data
        schema_path: Optional path to schema file. If None, uses default.

    Raises:
        jsonschema.ValidationError: If validation fails
        FileNotFoundError: If schema file not found
    """
    if schema_path is None:
        schema_path = _get_schema_path()

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema file not found: {schema_path}. "
            "Please ensure experiments/scenarios/schema.json exists."
        )

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    jsonschema.validate(instance=data, schema=schema)


def _dict_to_vehicle_type(data: Dict[str, Any]) -> VehicleType:
    """Convert dictionary to VehicleType dataclass."""
    return VehicleType(
        id=data["id"],
        accel=data.get("accel", 2.6),
        decel=data.get("decel", 4.5),
        sigma=data.get("sigma", 0.5),
        length=data.get("length", 5.0),
        maxSpeed=data.get("maxSpeed", 13.9),
        color=data.get("color"),
        minGap=data.get("minGap"),
        tau=data.get("tau"),
        speedFactor=data.get("speedFactor"),
        speedDev=data.get("speedDev"),
    )


def _dict_to_route(data: Dict[str, Any]) -> Route:
    """Convert dictionary to Route dataclass."""
    return Route(id=data["id"], edges=data["edges"])


def _dict_to_vehicle(data: Dict[str, Any]) -> Vehicle:
    """Convert dictionary to Vehicle dataclass."""
    return Vehicle(
        id=data["id"],
        type=data["type"],
        route=data["route"],
        depart=data["depart"],
        departLane=data.get("departLane"),
        departPos=data.get("departPos"),
        departSpeed=data.get("departSpeed"),
        arrivalLane=data.get("arrivalLane"),
        arrivalPos=data.get("arrivalPos"),
        arrivalSpeed=data.get("arrivalSpeed"),
    )


def _dict_to_flow(data: Dict[str, Any]) -> Flow:
    """Convert dictionary to Flow dataclass."""
    return Flow(
        route=data["route"],
        type=data["type"],
        begin=data.get("begin", 0.0),
        end=data.get("end", 3600.0),
        vehsPerHour=data.get("vehsPerHour"),
        period=data.get("period"),
        probability=data.get("probability"),
        number=data.get("number"),
        departLane=data.get("departLane"),
        departPos=data.get("departPos"),
        departSpeed=data.get("departSpeed"),
        arrivalLane=data.get("arrivalLane"),
        arrivalPos=data.get("arrivalPos"),
        arrivalSpeed=data.get("arrivalSpeed"),
        id=data.get("id"),
    )


def _dict_to_phase(data: Dict[str, Any]) -> Phase:
    """Convert dictionary to Phase dataclass."""
    return Phase(
        duration=data["duration"],
        state=data["state"],
        minDur=data.get("minDur"),
        maxDur=data.get("maxDur"),
        name=data.get("name"),
    )


def _dict_to_tl_program(data: Dict[str, Any]) -> TrafficLightProgram:
    """Convert dictionary to TrafficLightProgram dataclass."""
    return TrafficLightProgram(
        id=data["id"],
        program_id=data["program_id"],
        offset=data.get("offset", 0.0),
        type=data.get("type", "static"),
        phases=[_dict_to_phase(p) for p in data["phases"]],
    )


def _dict_to_scenario(data: Dict[str, Any]) -> Scenario:
    """Convert dictionary to Scenario dataclass."""
    traffic_data = data["traffic"]
    traffic = TrafficConfig(
        vehicle_types=[
            _dict_to_vehicle_type(vt) for vt in traffic_data["vehicle_types"]
        ],
        routes=[_dict_to_route(r) for r in traffic_data["routes"]],
        vehicles=(
            [_dict_to_vehicle(v) for v in traffic_data["vehicles"]]
            if traffic_data.get("vehicles")
            else None
        ),
        flows=(
            [_dict_to_flow(f) for f in traffic_data["flows"]]
            if traffic_data.get("flows")
            else None
        ),
    )

    tls_data = data["traffic_lights"]
    traffic_lights = TrafficLightConfig(
        programs=[_dict_to_tl_program(p) for p in tls_data["programs"]]
    )

    return Scenario(
        intersection=data["intersection"],
        traffic=traffic,
        traffic_lights=traffic_lights,
    )


def load_scenario_from_path(path: Path, validate: bool = True) -> Scenario:
    """
    Load scenario from a YAML file path.

    Args:
        path: Path to YAML file
        validate: Whether to validate against JSON schema

    Returns:
        Scenario object

    Raises:
        FileNotFoundError: If file not found
        yaml.YAMLError: If YAML parsing fails
        jsonschema.ValidationError: If validation fails
    """
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty or invalid YAML file: {path}")

    if validate:
        validate_scenario(data)

    return _dict_to_scenario(data)


def load_scenario(
    intersection_name: str, experiment_name: str, validate: bool = True
) -> Scenario:
    """
    Load scenario from experiments directory.

    Args:
        intersection_name: Name of intersection (e.g., "simple4")
        experiment_name: Name of experiment (e.g., "light_traffic")
        validate: Whether to validate against JSON schema

    Returns:
        Scenario object

    Raises:
        FileNotFoundError: If file not found
        yaml.YAMLError: If YAML parsing fails
        jsonschema.ValidationError: If validation fails
    """
    experiments_dir = _get_experiments_dir()
    scenario_path = experiments_dir / intersection_name / f"{experiment_name}.yaml"

    return load_scenario_from_path(scenario_path, validate=validate)


def _scenario_to_dict(scenario: Scenario) -> Dict[str, Any]:
    """Convert Scenario dataclass to dictionary."""
    traffic_dict = {
        "vehicle_types": [
            {
                "id": vt.id,
                "accel": vt.accel,
                "decel": vt.decel,
                "sigma": vt.sigma,
                "length": vt.length,
                "maxSpeed": vt.maxSpeed,
                **(
                    {"color": vt.color}
                    if vt.color is not None
                    else {}
                ),
                **(
                    {"minGap": vt.minGap}
                    if vt.minGap is not None
                    else {}
                ),
                **(
                    {"tau": vt.tau}
                    if vt.tau is not None
                    else {}
                ),
                **(
                    {"speedFactor": vt.speedFactor}
                    if vt.speedFactor is not None
                    else {}
                ),
                **(
                    {"speedDev": vt.speedDev}
                    if vt.speedDev is not None
                    else {}
                ),
            }
            for vt in scenario.traffic.vehicle_types
        ],
        "routes": [
            {"id": r.id, "edges": r.edges} for r in scenario.traffic.routes
        ],
    }

    if scenario.traffic.vehicles:
        traffic_dict["vehicles"] = [
            {
                "id": v.id,
                "type": v.type,
                "route": v.route,
                "depart": v.depart,
                **(
                    {"departLane": v.departLane}
                    if v.departLane is not None
                    else {}
                ),
                **(
                    {"departPos": v.departPos}
                    if v.departPos is not None
                    else {}
                ),
                **(
                    {"departSpeed": v.departSpeed}
                    if v.departSpeed is not None
                    else {}
                ),
                **(
                    {"arrivalLane": v.arrivalLane}
                    if v.arrivalLane is not None
                    else {}
                ),
                **(
                    {"arrivalPos": v.arrivalPos}
                    if v.arrivalPos is not None
                    else {}
                ),
                **(
                    {"arrivalSpeed": v.arrivalSpeed}
                    if v.arrivalSpeed is not None
                    else {}
                ),
            }
            for v in scenario.traffic.vehicles
        ]

    if scenario.traffic.flows:
        traffic_dict["flows"] = [
            {
                "route": f.route,
                "type": f.type,
                "begin": f.begin,
                "end": f.end,
                **(
                    {"vehsPerHour": f.vehsPerHour}
                    if f.vehsPerHour is not None
                    else {}
                ),
                **(
                    {"period": f.period}
                    if f.period is not None
                    else {}
                ),
                **(
                    {"probability": f.probability}
                    if f.probability is not None
                    else {}
                ),
                **(
                    {"number": f.number}
                    if f.number is not None
                    else {}
                ),
                **(
                    {"departLane": f.departLane}
                    if f.departLane is not None
                    else {}
                ),
                **(
                    {"departPos": f.departPos}
                    if f.departPos is not None
                    else {}
                ),
                **(
                    {"departSpeed": f.departSpeed}
                    if f.departSpeed is not None
                    else {}
                ),
                **(
                    {"arrivalLane": f.arrivalLane}
                    if f.arrivalLane is not None
                    else {}
                ),
                **(
                    {"arrivalPos": f.arrivalPos}
                    if f.arrivalPos is not None
                    else {}
                ),
                **(
                    {"arrivalSpeed": f.arrivalSpeed}
                    if f.arrivalSpeed is not None
                    else {}
                ),
                **({"id": f.id} if f.id is not None else {}),
            }
            for f in scenario.traffic.flows
        ]

    tls_dict = {
        "programs": [
            {
                "id": p.id,
                "program_id": p.program_id,
                "offset": p.offset,
                "type": p.type,
                "phases": [
                    {
                        "duration": ph.duration,
                        "state": ph.state,
                        **(
                            {"minDur": ph.minDur}
                            if ph.minDur is not None
                            else {}
                        ),
                        **(
                            {"maxDur": ph.maxDur}
                            if ph.maxDur is not None
                            else {}
                        ),
                        **({"name": ph.name} if ph.name is not None else {}),
                    }
                    for ph in p.phases
                ],
            }
            for p in scenario.traffic_lights.programs
        ]
    }

    return {
        "intersection": scenario.intersection,
        "traffic": traffic_dict,
        "traffic_lights": tls_dict,
    }


def save_scenario_to_path(
    scenario: Scenario, path: Path, validate: bool = True
) -> None:
    """
    Save scenario to a YAML file path.

    Args:
        scenario: Scenario object to save
        path: Path to save YAML file
        validate: Whether to validate before saving

    Raises:
        jsonschema.ValidationError: If validation fails
    """
    data = _scenario_to_dict(scenario)

    if validate:
        validate_scenario(data)

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def save_scenario(
    scenario: Scenario,
    intersection_name: str,
    experiment_name: str,
    validate: bool = True,
) -> None:
    """
    Save scenario to experiments directory.

    Args:
        scenario: Scenario object to save
        intersection_name: Name of intersection (e.g., "simple4")
        experiment_name: Name of experiment (e.g., "light_traffic")
        validate: Whether to validate before saving

    Raises:
        jsonschema.ValidationError: If validation fails
    """
    experiments_dir = _get_experiments_dir()
    scenario_path = experiments_dir / intersection_name / f"{experiment_name}.yaml"

    save_scenario_to_path(scenario, scenario_path, validate=validate)


def list_experiments(intersection_name: str) -> List[str]:
    """
    List available experiments for an intersection.

    Args:
        intersection_name: Name of intersection (e.g., "simple4")

    Returns:
        List of experiment names (without .yaml extension)

    Raises:
        FileNotFoundError: If intersection directory doesn't exist
    """
    experiments_dir = _get_experiments_dir()
    intersection_dir = experiments_dir / intersection_name

    if not intersection_dir.exists():
        raise FileNotFoundError(
            f"Intersection directory not found: {intersection_dir}"
        )

    experiments = []
    for file in intersection_dir.glob("*.yaml"):
        experiments.append(file.stem)

    return sorted(experiments)

