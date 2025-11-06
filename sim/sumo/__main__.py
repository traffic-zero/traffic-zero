"""
Command-line interface for SUMO simulations.

Usage:
    python -m sim.sumo simple4
    python -m sim.sumo.generate_config simple4
"""

if __name__ == '__main__':
    import sys
    from .runner import run_interactive
    
    if len(sys.argv) < 2:
        print("Usage: python -m sim.sumo <simulation_name>")
        print("Example: python -m sim.sumo simple4")
        sys.exit(1)
    
    simulation_name = sys.argv[1]
    run_interactive(simulation_name)

