"""
Simplified CARLA runner for SUMO scenarios.

This module provides an easy-to-use interface for running SUMO scenarios
in CARLA simulator using the official co-simulation bridge.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_in_carla(
    simulation_name: str,
    duration: int = 300,
    use_sumo_network: bool = True
):
    """
    Run a SUMO scenario in CARLA simulator.
    
    Args:
        simulation_name: Name of the simulation (e.g., 'simple4')
        duration: Simulation duration in seconds (default: 300s = 5 minutes)
        use_sumo_network: Use SUMO network as CARLA map (default: True)
    
    Example:
        >>> from sim import run_in_carla
        >>> run_in_carla('simple4', duration=120)
    """
    
    # Build paths (relative to project root)
    base_dir = Path(__file__).parent.parent / "intersections" / simulation_name
    sumo_cfg = base_dir / f"{simulation_name}.sumocfg"
    
    # Auto-generate .sumocfg if it doesn't exist
    if not sumo_cfg.exists():
        print(f"[INFO] Configuration file not found: {sumo_cfg}")
        print(f"[INFO] Auto-generating SUMO configuration file...")
        
        # Import and generate the config
        from ..sumo.generate_config import generate_sumocfg
        success = generate_sumocfg(simulation_name)
        
        if not success:
            print("[ERROR] Failed to generate SUMO configuration file")
            print(f"[ERROR] Please check that the simulation '{simulation_name}' exists")
            return
        
        print("[SUCCESS] Configuration file generated successfully")
    
    # Check CARLA installation
    carla_root = os.environ.get('CARLA_ROOT')
    if not carla_root:
        print("[ERROR] CARLA_ROOT environment variable not set!")
        print("\nPlease:")
        print("1. Install CARLA (see CARLA_SETUP.md)")
        print("2. Set CARLA_ROOT environment variable:")
        print("   Windows: set CARLA_ROOT=C:\\path\\to\\CARLA")
        print("   Linux/Mac: export CARLA_ROOT=/path/to/CARLA")
        return
    
    # Check if CARLA server is running
    print("\nChecking CARLA server...")
    print("If CARLA is not running, start it with:")
    print(f"  cd {carla_root}")
    print("  ./CarlaUE4.exe (Windows) or ./CarlaUE4.sh (Linux)\n")
    
    # Import and run co-simulation
    try:
        from sim.carla.bridge import CarlaSumoSync
        
        print(f"Starting co-simulation for '{simulation_name}'...")
        print(f"Duration: {duration}s")
        print(f"Use SUMO network: {use_sumo_network}")
        print("-" * 60)
        
        cosim = CarlaSumoSync(
            sumo_cfg_file=str(sumo_cfg),
            step_length=0.05,
            tls_manager='sumo',
            auto_camera=False,  # Let user control camera freely
            use_sumo_network=use_sumo_network
        )
        
        cosim.run_cosimulation(duration=duration)
        
    except ImportError as e:
        print(f"[ERROR] Error importing CARLA components: {e}")
        print("\nPlease check:")
        print("1. CARLA is installed correctly")
        print("2. CARLA Python API is in PYTHONPATH")
        print("3. See CARLA_SETUP.md for detailed instructions")
    except Exception as e:
        print(f"[ERROR] Error during co-simulation: {e}")
        raise


def list_simulations():
    """List all available SUMO simulations."""
    sim_dir = Path(__file__).parent.parent / "intersections"
    
    if not sim_dir.exists():
        print("No simulations directory found.")
        return []
    
    simulations = [d.name for d in sim_dir.iterdir() if d.is_dir()]
    
    print("\nAvailable simulations:")
    for i, sim in enumerate(simulations, 1):
        print(f"  {i}. {sim}")
    
    return simulations


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run SUMO scenarios in CARLA'
    )
    parser.add_argument(
        'simulation',
        nargs='?',
        help='Simulation name (e.g., simple4)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=300,
        help='Simulation duration in seconds'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available simulations'
    )
    parser.add_argument(
        '--no-sumo-network',
        action='store_true',
        help='Use CARLA town map instead of SUMO network'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_simulations()
    elif args.simulation:
        run_in_carla(
            args.simulation,
            args.duration,
            use_sumo_network=not args.no_sumo_network
        )
    else:
        parser.print_help()

