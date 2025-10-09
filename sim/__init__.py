"""
Simulations module.

Provides tools for running traffic simulations in SUMO and CARLA.
"""

# SUMO imports (always available)
from .sumo import run_interactive, run_automated, generate_sumocfg

# CARLA imports (lazy loaded to make CARLA optional)
def run_carla(*args, **kwargs):
    """
    Run a SUMO scenario in CARLA simulator for co-simulation.
    
    This function bridges SUMO traffic simulation with CARLA's 3D environment,
    allowing vehicles to be rendered in CARLA while being controlled by SUMO.
    This enables realistic 3D visualization and interaction with traffic scenarios.
    
    Features:
        - Real-time synchronization between SUMO and CARLA
        - 3D vehicle rendering in CARLA's environment
        - Traffic light visualization and control
        - Vehicle spawning and routing from SUMO
    
    Args:
        simulation_name (str): Name of simulation to run (e.g., "simple4")
        duration (int, optional): Simulation duration in seconds. Default: 120
        *args, **kwargs: Additional arguments passed to CARLA runner
    
    Example:
        >>> run_carla("simple4", duration=300)
        >>> Starting CARLA-SUMO co-simulation for simple4
        >>> Simulation will run for 300 seconds
    
    Note:
        Requires CARLA installation and proper environment setup.
        Make sure CARLA is running before calling this function.
    """
    from .carla import run_in_carla as _run_in_carla
    return _run_in_carla(*args, **kwargs)

def list_simulations():
    """List all available SUMO simulations."""
    from .carla import list_simulations as _list_simulations
    return _list_simulations()

__all__ = [
    'run_interactive',
    'run_automated', 
    'generate_sumocfg',
    'run_carla',
    'list_simulations'
]
