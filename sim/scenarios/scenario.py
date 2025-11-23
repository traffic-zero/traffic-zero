"""
Core dataclasses for scenario definition.

Defines the structure for traffic scenarios including vehicle types,
routes, vehicles, flows, and traffic light programs.
"""

from dataclasses import dataclass, field


@dataclass
class VehicleType:
    """Vehicle type definition (vType in SUMO)."""

    id: str
    accel: float = 2.6
    decel: float = 4.5
    sigma: float = 0.5
    length: float = 5.0
    max_speed: float = 13.9
    color: str | None = None
    # Additional optional attributes
    min_gap: float | None = None
    tau: float | None = None
    speed_factor: float | None = None
    speed_dev: float | None = None


@dataclass
class Route:
    """Route definition connecting edges."""

    id: str
    edges: list[str]


@dataclass
class Vehicle:
    """Individual vehicle definition (programmatic spawning)."""

    id: str
    type: str
    route: str
    depart: float
    depart_lane: str | None = None
    depart_pos: float | None = None
    depart_speed: float | None = None
    arrival_lane: str | None = None
    arrival_pos: float | None = None
    arrival_speed: float | None = None


@dataclass
class Flow:
    """Flow definition for statistical vehicle spawning."""

    route: str
    type: str
    begin: float = 0.0
    end: float = 3600.0
    # Flow rate options (mutually exclusive in SUMO, but we allow any)
    vehs_per_hour: float | None = None
    period: float | None = None
    probability: float | None = None
    number: int | None = None
    # Additional optional attributes
    depart_lane: str | None = None
    depart_pos: float | None = None
    depart_speed: float | None = None
    arrival_lane: str | None = None
    arrival_pos: float | None = None
    arrival_speed: float | None = None
    id: str | None = None


@dataclass
class Phase:
    """Traffic light phase definition."""

    duration: float
    state: str
    min_dur: float | None = None
    max_dur: float | None = None
    name: str | None = None


@dataclass
class TrafficLightProgram:
    """Traffic light program definition."""

    id: str
    program_id: str
    offset: float = 0.0
    phases: list[Phase] = field(default_factory=list)
    type: str = "static"  # static or actuated


@dataclass
class TrafficConfig:
    """Traffic configuration (vehicles, routes, flows)."""

    vehicle_types: list[VehicleType] = field(default_factory=list)
    routes: list[Route] = field(default_factory=list)
    vehicles: list[Vehicle] | None = None
    flows: list[Flow] | None = None


@dataclass
class TrafficLightConfig:
    """Traffic light configuration."""

    programs: list[TrafficLightProgram] = field(default_factory=list)


@dataclass
class Scenario:
    """Complete scenario definition."""

    intersection: str
    traffic: TrafficConfig
    traffic_lights: TrafficLightConfig
