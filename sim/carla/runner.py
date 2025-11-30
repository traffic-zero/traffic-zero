"""
Simplified CARLA runner for SUMO scenarios.

This module provides an easy-to-use interface for running SUMO scenarios
in CARLA simulator using the official co-simulation bridge.
"""

import os
import sys
import time
import subprocess
import platform
import socket
from pathlib import Path


def _is_carla_running(host: str = "localhost", port: int = 2000, verbose: bool = False) -> bool:
    """Check if CARLA server is running by attempting to connect."""
    try:
        # Try to connect to CARLA's port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)  # Increased timeout
        result = sock.connect_ex((host, port))
        sock.close()
        
        if verbose:
            print(f"  [DEBUG] Socket connection to {host}:{port} = {result}")
        
        if result == 0:
            # Port is open, but verify it's actually CARLA
            try:
                # Add CARLA to Python path if CARLA_ROOT is set
                carla_root = os.environ.get("CARLA_ROOT")
                if carla_root:
                    carla_egg_path = os.path.join(carla_root, "PythonAPI", "carla")
                    if os.path.exists(carla_egg_path) and carla_egg_path not in sys.path:
                        sys.path.append(carla_egg_path)
                    
                    # Also try to find and add the .egg file for specific Python version
                    dist_path = os.path.join(carla_egg_path, "dist")
                    if os.path.exists(dist_path):
                        py_version = f"py{sys.version_info.major}{sys.version_info.minor}"
                        for egg_file in os.listdir(dist_path):
                            if egg_file.endswith(".egg") and py_version in egg_file:
                                egg_full_path = os.path.join(dist_path, egg_file)
                                if egg_full_path not in sys.path:
                                    sys.path.append(egg_full_path)
                                break
                
                import carla
                client = carla.Client(host, port)
                client.set_timeout(5.0)  # Increased timeout for get_world()
                world = client.get_world()
                if verbose:
                    print(f"  [DEBUG] Successfully connected to CARLA world: {world.get_map().name}")
                return True
            except ImportError as e:
                if verbose:
                    print(f"  [DEBUG] Failed to import carla: {e}")
                return False
            except RuntimeError as e:
                error_msg = str(e)
                if verbose:
                    print(f"  [DEBUG] CARLA connection error: {error_msg}")
                # Check if it's a timeout - might mean CARLA is starting but not ready
                if "time-out" in error_msg.lower() or "timeout" in error_msg.lower():
                    if verbose:
                        print("  [DEBUG] CARLA port is open but server not ready yet")
                    return False
                return False
            except Exception as e:
                if verbose:
                    print(f"  [DEBUG] Unexpected error checking CARLA: {e}")
                return False
        else:
            if verbose:
                print(f"  [DEBUG] Port {port} is not open")
            return False
    except Exception as e:
        if verbose:
            print(f"  [DEBUG] Socket check failed: {e}")
        return False


def _launch_carla(carla_root: str) -> bool:
    """Launch CARLA server automatically."""
    system = platform.system()
    carla_exe = None
    
    if system == "Windows":
        carla_exe = os.path.join(carla_root, "CarlaUE4.exe")
        if not os.path.exists(carla_exe):
            # Try alternative location
            carla_exe = os.path.join(carla_root, "CarlaUE4", "Binaries", "Win64", "CarlaUE4.exe")
    elif system == "Linux":
        carla_exe = os.path.join(carla_root, "CarlaUE4.sh")
    else:
        print(f"[ERROR] Unsupported platform: {system}")
        print("Please start CARLA manually:")
        print(f"  cd {carla_root}")
        if system == "Windows":
            print("  .\\CarlaUE4.exe")
        else:
            print("  ./CarlaUE4.sh")
        return False
    
    if not os.path.exists(carla_exe):
        print(f"[ERROR] CARLA executable not found at: {carla_exe}")
        print("Please check your CARLA_ROOT path and start CARLA manually:")
        print(f"  cd {carla_root}")
        if system == "Windows":
            print("  .\\CarlaUE4.exe")
        else:
            print("  ./CarlaUE4.sh")
        return False
    
    # Double-check CARLA isn't already running (might have started between checks)
    if _is_carla_running():
        print("✓ CARLA server is already running - will use existing instance")
        return True
    
    print(f"Launching CARLA from: {carla_exe}")
    print("This may take 30-60 seconds to start...")
    
    try:
        # Launch CARLA in the background
        if system == "Windows":
            # Launch CARLA with window visible
            subprocess.Popen(
                [
                    carla_exe,
                    "-d3d11",
                    "-carla-server",
                    "-benchmark",
                    "-fps=30",
                    "-windowed",
                    "-ResX=1280",
                    "-ResY=720",
                ],
                cwd=carla_root,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            # Linux/Mac
            subprocess.Popen(
                [carla_exe, "-carla-server", "-benchmark", "-fps=30"],
                cwd=carla_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        
        # Wait for CARLA to start (check every 2 seconds, up to 60 seconds)
        print("Waiting for CARLA to start...", end="", flush=True)
        max_wait = 60
        waited = 0
        while waited < max_wait:
            if _is_carla_running():
                print("\n✓ CARLA server is ready!")
                return True
            time.sleep(2)
            waited += 2
            print(".", end="", flush=True)
        
        print(f"\n[WARNING] CARLA did not start within {max_wait} seconds")
        print("It may still be starting. The connection will be retried...")
        return False
        
    except Exception as e:
        print(f"\n[ERROR] Failed to launch CARLA: {e}")
        print("Please start CARLA manually:")
        print(f"  cd {carla_root}")
        if system == "Windows":
            print("  .\\CarlaUE4.exe")
        else:
            print("  ./CarlaUE4.sh")
        return False


def run_in_carla(
    simulation_name: str,
    experiment_name: str | None = None,
    duration: int = 300,
    use_sumo_network: bool = True,
):
    """
    Run a SUMO scenario in CARLA simulator.

    Args:
        simulation_name: Name of the simulation (e.g., 'simple4')
        experiment_name: Name of experiment scenario to use (optional).
                         If provided, generates routes and tls from scenario.
        duration: Simulation duration in seconds (default: 300s = 5 minutes)
        use_sumo_network: Use SUMO network as CARLA map (default: True)

    Example:
        >>> from sim import run_in_carla
        >>> run_in_carla('simple4', experiment_name='light_traffic', duration=120)
    """

    # Build paths (relative to project root)
    base_dir = Path(__file__).parent.parent / "intersections" / simulation_name
    sumo_cfg = base_dir / f"{simulation_name}.sumocfg"

    # Import and generate the config
    from ..sumo.generate_config import generate_sumocfg

    # Generate config if it doesn't exist or if experiment is specified
    if not sumo_cfg.exists() or experiment_name is not None:
        if experiment_name is not None:
            print(
                f"[INFO] Generating configuration with experiment: "
                f"{experiment_name}"
            )
        else:
            print(f"[INFO] Configuration file not found: {sumo_cfg}")
            print("[INFO] Auto-generating SUMO configuration file...")

        success = generate_sumocfg(simulation_name, experiment_name)

        if not success:
            print("[ERROR] Failed to generate SUMO configuration file")
            print(
                f"[ERROR] Please check that the simulation "
                f"'{simulation_name}' exists"
            )
            return

        print("[SUCCESS] Configuration file generated successfully")

    # Check CARLA installation
    carla_root = os.environ.get("CARLA_ROOT")
    if not carla_root:
        print("[ERROR] CARLA_ROOT environment variable not set!")
        print("\nPlease:")
        print("1. Install CARLA (see CARLA_SETUP.md)")
        print("2. Set CARLA_ROOT environment variable:")
        print("   Windows: set CARLA_ROOT=C:\\path\\to\\CARLA")
        print("   Linux/Mac: export CARLA_ROOT=/path/to/CARLA")
        return

    # Check if CARLA server is running and launch if needed
    print("\nChecking CARLA server...")
    is_running = _is_carla_running(verbose=False)  # Set to True for debugging
    if is_running:
        print("✓ CARLA server is already running - will use existing instance")
    else:
        print("CARLA server is not running. Attempting to launch...")
        _launch_carla(carla_root)

    # Import and run co-simulation
    try:
        from sim.carla.bridge import CarlaSumoSync

        print(f"Starting co-simulation for '{simulation_name}'...")
        if experiment_name:
            print(f"Experiment: {experiment_name}")
        print(f"Duration: {duration}s")
        print(f"Use SUMO network: {use_sumo_network}")
        print("-" * 60)

        cosim = CarlaSumoSync(
            sumo_cfg_file=str(sumo_cfg),
            step_length=0.05,
            tls_manager="sumo",
            auto_camera=False,  # Let user control camera freely
            use_sumo_network=use_sumo_network,
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SUMO scenarios in CARLA")
    parser.add_argument(
        "simulation", nargs="?", help="Simulation name (e.g., simple4)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Simulation duration in seconds",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available simulations"
    )
    parser.add_argument(
        "--no-sumo-network",
        action="store_true",
        help="Use CARLA town map instead of SUMO network",
    )

    args = parser.parse_args()

    if args.list:
        list_simulations()
    elif args.simulation:
        run_in_carla(
            args.simulation,
            args.duration,
            use_sumo_network=not args.no_sumo_network,
        )
    else:
        parser.print_help()
