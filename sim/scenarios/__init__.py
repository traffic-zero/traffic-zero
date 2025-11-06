"""
Scenario generation module for SUMO traffic simulations.

Provides tools for creating, loading, and generating SUMO scenarios
with configurable traffic patterns and traffic light programs.
"""

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
from .loader import (
    load_scenario,
    load_scenario_from_path,
    save_scenario,
    save_scenario_to_path,
    validate_scenario,
    list_experiments,
)
from .generator import (
    generate_routes_xml,
    generate_tls_xml,
)

__all__ = [
    # Dataclasses
    'Scenario',
    'TrafficConfig',
    'TrafficLightConfig',
    'VehicleType',
    'Route',
    'Vehicle',
    'Flow',
    'Phase',
    'TrafficLightProgram',
    # Loader functions
    'load_scenario',
    'load_scenario_from_path',
    'save_scenario',
    'save_scenario_to_path',
    'validate_scenario',
    'list_experiments',
    # Generator functions
    'generate_routes_xml',
    'generate_tls_xml',
]

