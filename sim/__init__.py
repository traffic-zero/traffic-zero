"""
Simulations module.

Provides tools for running traffic simulations in SUMO and CARLA.
"""

# SUMO imports (always available)
from .sumo import run_intersection, generate_sumocfg

# CARLA imports (lazy loaded to make CARLA optional)
def run_in_carla(*args, **kwargs):
    """Run a SUMO scenario in CARLA simulator (requires CARLA installation)."""
    from .carla import run_in_carla as _run_in_carla
    return _run_in_carla(*args, **kwargs)

def list_simulations():
    """List all available SUMO simulations."""
    from .carla import list_simulations as _list_simulations
    return _list_simulations()

__all__ = [
    'run_intersection',
    'generate_sumocfg',
    'run_in_carla',
    'list_simulations'
]
