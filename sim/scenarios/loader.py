"""
YAML loader and saver for scenario configurations with JSON schema validation.
"""

import json
from pathlib import Path
from typing import Any

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


def validate_scenario(
    data: dict[str, Any], schema_path: Path | None = None
) -> None:
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


def _get_field(
    data: dict[str, Any], snake_key: str, camel_key: str, default: Any = None
) -> Any:
    """Get field from dict supporting both snake_case and camelCase."""
    return data.get(snake_key, data.get(camel_key, default))


def _dict_to_vehicle_type(data: dict[str, Any]) -> VehicleType:
    """Convert dictionary to VehicleType dataclass."""
    return VehicleType(
        id=data["id"],
        accel=data.get("accel", 2.6),
        decel=data.get("decel", 4.5),
        sigma=data.get("sigma", 0.5),
        length=data.get("length", 5.0),
        max_speed=_get_field(data, "max_speed", "maxSpeed", 13.9),
        color=data.get("color"),
        min_gap=_get_field(data, "min_gap", "minGap"),
        tau=data.get("tau"),
        speed_factor=_get_field(data, "speed_factor", "speedFactor"),
        speed_dev=_get_field(data, "speed_dev", "speedDev"),
    )


def _dict_to_route(data: dict[str, Any]) -> Route:
    """Convert dictionary to Route dataclass."""
    return Route(id=data["id"], edges=data["edges"])


def _dict_to_vehicle(data: dict[str, Any]) -> Vehicle:
    """Convert dictionary to Vehicle dataclass."""
    return Vehicle(
        id=data["id"],
        type=data["type"],
        route=data["route"],
        depart=data["depart"],
        depart_lane=_get_field(data, "depart_lane", "departLane"),
        depart_pos=_get_field(data, "depart_pos", "departPos"),
        depart_speed=_get_field(data, "depart_speed", "departSpeed"),
        arrival_lane=_get_field(data, "arrival_lane", "arrivalLane"),
        arrival_pos=_get_field(data, "arrival_pos", "arrivalPos"),
        arrival_speed=_get_field(data, "arrival_speed", "arrivalSpeed"),
    )


def _dict_to_flow(data: dict[str, Any]) -> Flow:
    """Convert dictionary to Flow dataclass."""
    return Flow(
        route=data["route"],
        type=data["type"],
        begin=data.get("begin", 0.0),
        end=data.get("end", 3600.0),
        vehs_per_hour=_get_field(data, "vehs_per_hour", "vehsPerHour"),
        period=data.get("period"),
        probability=data.get("probability"),
        number=data.get("number"),
        depart_lane=_get_field(data, "depart_lane", "departLane"),
        depart_pos=_get_field(data, "depart_pos", "departPos"),
        depart_speed=_get_field(data, "depart_speed", "departSpeed"),
        arrival_lane=_get_field(data, "arrival_lane", "arrivalLane"),
        arrival_pos=_get_field(data, "arrival_pos", "arrivalPos"),
        arrival_speed=_get_field(data, "arrival_speed", "arrivalSpeed"),
        id=data.get("id"),
    )


def _dict_to_phase(data: dict[str, Any]) -> Phase:
    """Convert dictionary to Phase dataclass."""
    return Phase(
        duration=data["duration"],
        state=data["state"],
        min_dur=_get_field(data, "min_dur", "minDur"),
        max_dur=_get_field(data, "max_dur", "maxDur"),
        name=data.get("name"),
    )


def _dict_to_tl_program(data: dict[str, Any]) -> TrafficLightProgram:
    """Convert dictionary to TrafficLightProgram dataclass."""
    return TrafficLightProgram(
        id=data["id"],
        program_id=data["program_id"],
        offset=data.get("offset", 0.0),
        type=data.get("type", "static"),
        phases=[_dict_to_phase(p) for p in data["phases"]],
    )


def _dict_to_scenario(data: dict[str, Any]) -> Scenario:
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
    scenario_path = (
        experiments_dir / intersection_name / f"{experiment_name}.yaml"
    )

    return load_scenario_from_path(scenario_path, validate=validate)


def _vehicle_type_to_dict(vt) -> dict[str, Any]:
    """Convert VehicleType to dictionary."""
    result = {
        "id": vt.id,
        "accel": vt.accel,
        "decel": vt.decel,
        "sigma": vt.sigma,
        "length": vt.length,
        "maxSpeed": vt.max_speed,
    }
    if vt.color is not None:
        result["color"] = vt.color
    if vt.min_gap is not None:
        result["minGap"] = vt.min_gap
    if vt.tau is not None:
        result["tau"] = vt.tau
    if vt.speed_factor is not None:
        result["speedFactor"] = vt.speed_factor
    if vt.speed_dev is not None:
        result["speedDev"] = vt.speed_dev
    return result


def _vehicle_to_dict(v) -> dict[str, Any]:
    """Convert Vehicle to dictionary."""
    result = {
        "id": v.id,
        "type": v.type,
        "route": v.route,
        "depart": v.depart,
    }
    if v.depart_lane is not None:
        result["departLane"] = v.depart_lane
    if v.depart_pos is not None:
        result["departPos"] = v.depart_pos
    if v.depart_speed is not None:
        result["departSpeed"] = v.depart_speed
    if v.arrival_lane is not None:
        result["arrivalLane"] = v.arrival_lane
    if v.arrival_pos is not None:
        result["arrivalPos"] = v.arrival_pos
    if v.arrival_speed is not None:
        result["arrivalSpeed"] = v.arrival_speed
    return result


def _flow_to_dict(f) -> dict[str, Any]:
    """Convert Flow to dictionary."""
    result = {
        "route": f.route,
        "type": f.type,
        "begin": f.begin,
        "end": f.end,
    }
    if f.vehs_per_hour is not None:
        result["vehsPerHour"] = f.vehs_per_hour
    if f.period is not None:
        result["period"] = f.period
    if f.probability is not None:
        result["probability"] = f.probability
    if f.number is not None:
        result["number"] = f.number
    if f.depart_lane is not None:
        result["departLane"] = f.depart_lane
    if f.depart_pos is not None:
        result["departPos"] = f.depart_pos
    if f.depart_speed is not None:
        result["departSpeed"] = f.depart_speed
    if f.arrival_lane is not None:
        result["arrivalLane"] = f.arrival_lane
    if f.arrival_pos is not None:
        result["arrivalPos"] = f.arrival_pos
    if f.arrival_speed is not None:
        result["arrivalSpeed"] = f.arrival_speed
    if f.id is not None:
        result["id"] = f.id
    return result


def _phase_to_dict(ph) -> dict[str, Any]:
    """Convert Phase to dictionary."""
    result = {
        "duration": ph.duration,
        "state": ph.state,
    }
    if ph.min_dur is not None:
        result["minDur"] = ph.min_dur
    if ph.max_dur is not None:
        result["maxDur"] = ph.max_dur
    if ph.name is not None:
        result["name"] = ph.name
    return result


def _scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    """Convert Scenario dataclass to dictionary."""
    traffic_dict = {
        "vehicle_types": [
            _vehicle_type_to_dict(vt) for vt in scenario.traffic.vehicle_types
        ],
        "routes": [
            {"id": r.id, "edges": r.edges} for r in scenario.traffic.routes
        ],
    }

    if scenario.traffic.vehicles:
        traffic_dict["vehicles"] = [
            _vehicle_to_dict(v) for v in scenario.traffic.vehicles
        ]

    if scenario.traffic.flows:
        traffic_dict["flows"] = [
            _flow_to_dict(f) for f in scenario.traffic.flows
        ]

    tls_dict = {
        "programs": [
            {
                "id": p.id,
                "program_id": p.program_id,
                "offset": p.offset,
                "type": p.type,
                "phases": [_phase_to_dict(ph) for ph in p.phases],
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
    scenario_path = (
        experiments_dir / intersection_name / f"{experiment_name}.yaml"
    )

    save_scenario_to_path(scenario, scenario_path, validate=validate)


def list_experiments(intersection_name: str) -> list[str]:
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
