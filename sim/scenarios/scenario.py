"""
Core dataclasses for scenario definition.

Defines the structure for traffic scenarios including vehicle types,
routes, vehicles, flows, and traffic light programs.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class VehicleType:
    """Vehicle type definition (vType in SUMO)."""
    id: str
    accel: float = 2.6
    decel: float = 4.5
    sigma: float = 0.5
    length: float = 5.0
    maxSpeed: float = 13.9
    color: Optional[str] = None
    # Additional optional attributes
    minGap: Optional[float] = None
    tau: Optional[float] = None
    speedFactor: Optional[float] = None
    speedDev: Optional[float] = None


@dataclass
class Route:
    """Route definition connecting edges."""
    id: str
    edges: List[str]


@dataclass
class Vehicle:
    """Individual vehicle definition (programmatic spawning)."""
    id: str
    type: str
    route: str
    depart: float
    departLane: Optional[str] = None
    departPos: Optional[float] = None
    departSpeed: Optional[float] = None
    arrivalLane: Optional[str] = None
    arrivalPos: Optional[float] = None
    arrivalSpeed: Optional[float] = None


@dataclass
class Flow:
    """Flow definition for statistical vehicle spawning."""
    route: str
    type: str
    begin: float = 0.0
    end: float = 3600.0
    # Flow rate options (mutually exclusive in SUMO, but we allow any)
    vehsPerHour: Optional[float] = None
    period: Optional[float] = None
    probability: Optional[float] = None
    number: Optional[int] = None
    # Additional optional attributes
    departLane: Optional[str] = None
    departPos: Optional[float] = None
    departSpeed: Optional[float] = None
    arrivalLane: Optional[str] = None
    arrivalPos: Optional[float] = None
    arrivalSpeed: Optional[float] = None
    id: Optional[str] = None


@dataclass
class Phase:
    """Traffic light phase definition."""
    duration: float
    state: str
    minDur: Optional[float] = None
    maxDur: Optional[float] = None
    name: Optional[str] = None


@dataclass
class TrafficLightProgram:
    """Traffic light program definition."""
    id: str
    program_id: str
    offset: float = 0.0
    phases: List[Phase] = field(default_factory=list)
    type: str = "static"  # static or actuated


@dataclass
class TrafficConfig:
    """Traffic configuration (vehicles, routes, flows)."""
    vehicle_types: List[VehicleType] = field(default_factory=list)
    routes: List[Route] = field(default_factory=list)
    vehicles: Optional[List[Vehicle]] = None
    flows: Optional[List[Flow]] = None


@dataclass
class TrafficLightConfig:
    """Traffic light configuration."""
    programs: List[TrafficLightProgram] = field(default_factory=list)


@dataclass
class Scenario:
    """Complete scenario definition."""
    intersection: str
    traffic: TrafficConfig
    traffic_lights: TrafficLightConfig

