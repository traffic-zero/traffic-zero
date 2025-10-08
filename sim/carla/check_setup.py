"""
CARLA Setup Verification Script

Run this script to check if your CARLA installation is properly configured.
"""

import os
import sys
from pathlib import Path


def check_environment_variables():
    """Check if required environment variables are set."""
    print("\n1. Checking Environment Variables...")
    print("-" * 60)
    
    carla_root = os.environ.get('CARLA_ROOT')
    pythonpath = os.environ.get('PYTHONPATH', '')
    
    if carla_root:
        print(f"✓ CARLA_ROOT is set: {carla_root}")
        
        # Check if path exists
        if Path(carla_root).exists():
            print(f"✓ CARLA_ROOT directory exists")
        else:
            print(f"✗ CARLA_ROOT directory does not exist!")
            return False
    else:
        print("✗ CARLA_ROOT is not set!")
        print("\nSet it with:")
        print("  Windows: $env:CARLA_ROOT = 'C:\\CARLA\\CARLA_0.9.15'")
        print("  Linux:   export CARLA_ROOT='/opt/carla/CARLA_0.9.15'")
        return False
    
    if 'carla' in pythonpath.lower():
        print(f"✓ PYTHONPATH includes CARLA")
    else:
        print("⚠ PYTHONPATH might not include CARLA Python API")
        print(f"  Current PYTHONPATH: {pythonpath}")
    
    return True


def check_carla_installation():
    """Check if CARLA files exist."""
    print("\n2. Checking CARLA Installation...")
    print("-" * 60)
    
    carla_root = os.environ.get('CARLA_ROOT')
    if not carla_root:
        print("✗ Cannot check - CARLA_ROOT not set")
        return False
    
    carla_path = Path(carla_root)
    
    # Check for CarlaUE4 executable
    if sys.platform == 'win32':
        carla_exe = carla_path / 'CarlaUE4.exe'
        carla_exe_alt = carla_path / 'CarlaUE4' / 'Binaries' / 'Win64' / 'CarlaUE4.exe'
    else:
        carla_exe = carla_path / 'CarlaUE4.sh'
        carla_exe_alt = carla_path / 'CarlaUE4'
    
    if carla_exe.exists() or carla_exe_alt.exists():
        print("✓ CARLA executable found")
    else:
        print(f"✗ CARLA executable not found at {carla_exe}")
        return False
    
    # Check for Python API
    python_api = carla_path / 'PythonAPI' / 'carla'
    if python_api.exists():
        print(f"✓ Python API directory exists: {python_api}")
    else:
        print(f"✗ Python API not found at {python_api}")
        return False
    
    # Check for co-simulation scripts
    cosim_path = carla_path / 'Co-Simulation' / 'Sumo'
    if cosim_path.exists():
        print(f"✓ Co-Simulation scripts found: {cosim_path}")
    else:
        print(f"⚠ Co-Simulation scripts not found at {cosim_path}")
        print("  You may need to download them separately from CARLA repo")
    
    return True


def check_python_api():
    """Try to import CARLA Python API."""
    print("\n3. Checking CARLA Python API...")
    print("-" * 60)
    
    try:
        import carla
        print(f"✓ CARLA module imported successfully")
        print(f"  Version: {carla.__version__ if hasattr(carla, '__version__') else 'Unknown'}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import CARLA module: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure CARLA_ROOT is set correctly")
        print("2. Add CARLA Python API to PYTHONPATH:")
        print("   Windows: $env:PYTHONPATH = \"$env:CARLA_ROOT\\PythonAPI\\carla\"")
        print("   Linux:   export PYTHONPATH=\"${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}\"")
        print("3. Or install the wheel:")
        print("   pip install $CARLA_ROOT/PythonAPI/carla/dist/carla-*.whl")
        return False


def check_carla_server():
    """Check if CARLA server is running."""
    print("\n4. Checking CARLA Server...")
    print("-" * 60)
    
    try:
        import carla
        
        print("Attempting to connect to CARLA server...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        
        world = client.get_world()
        map_name = world.get_map().name
        
        print(f"✓ Connected to CARLA server!")
        print(f"  Current map: {map_name}")
        return True
        
    except ImportError:
        print("✗ Cannot check - CARLA module not available")
        return False
    except Exception as e:
        print(f"✗ Cannot connect to CARLA server: {e}")
        print("\nTo start CARLA server:")
        print("  cd $CARLA_ROOT")
        print("  ./CarlaUE4.exe (Windows) or ./CarlaUE4.sh (Linux)")
        return False


def check_sumo_config():
    """Check if SUMO configuration exists for scenarios."""
    print("\n5. Checking SUMO Configuration...")
    print("-" * 60)
    
    sim_dir = Path(__file__).parent.parent / 'intersections'
    
    if not sim_dir.exists():
        print("✗ Simulations directory not found")
        return False
    
    scenarios = [d for d in sim_dir.iterdir() if d.is_dir()]
    
    if not scenarios:
        print("⚠ No scenarios found")
        return True
    
    print(f"Found {len(scenarios)} scenario(s):")
    
    all_ok = True
    for scenario in scenarios:
        scenario_name = scenario.name
        sumocfg = scenario / f"{scenario_name}.sumocfg"
        
        if sumocfg.exists():
            print(f"  ✓ {scenario_name}: config exists")
        else:
            print(f"  ✗ {scenario_name}: config missing")
            print(f"    Generate with: python -m sim.generate_sumocfg {scenario_name}")
            all_ok = False
    
    return all_ok


def main():
    """Run all checks."""
    print("="*60)
    print("CARLA-SUMO Co-Simulation Setup Verification")
    print("="*60)
    
    checks = [
        ("Environment Variables", check_environment_variables),
        ("CARLA Installation", check_carla_installation),
        ("CARLA Python API", check_python_api),
        ("CARLA Server (optional)", check_carla_server),
        ("SUMO Configuration", check_sumo_config),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ Error during {name} check: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
    
    required_checks = [
        "Environment Variables",
        "CARLA Installation", 
        "CARLA Python API",
        "SUMO Configuration"
    ]
    
    required_passed = all(
        results.get(check, False) 
        for check in required_checks
    )
    
    if required_passed:
        print("\n✓ All required checks passed!")
        print("\nYou're ready to run co-simulations!")
        print("\nNext steps:")
        print("1. Start CARLA server: cd $CARLA_ROOT && ./CarlaUE4.exe")
        print("2. Run co-simulation: python -m sim.carla_runner simple4")
    else:
        print("\n✗ Some required checks failed.")
        print("\nPlease:")
        print("1. Review the errors above")
        print("2. Follow the instructions in sim/CARLA_SETUP.md")
        print("3. Run this script again to verify")
    
    server_running = results.get("CARLA Server (optional)", False)
    if not server_running:
        print("\nNote: CARLA server is not running (optional for this check)")
        print("      Start it when you're ready to run simulations")
    
    print()
    return 0 if required_passed else 1


if __name__ == '__main__':
    exit(main())

