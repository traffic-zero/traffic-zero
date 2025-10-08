"""
Example: How to run SUMO scenarios in CARLA

This file demonstrates different ways to use the CARLA-SUMO co-simulation.
"""

# Example 1: Simple usage - Run scenario in CARLA
# ================================================
def example_simple():
    """Run a scenario in CARLA (simplest method)."""
    from sim import run_in_carla
    
    # Run simple4 intersection for 60 seconds
    run_in_carla('simple4', duration=60)


# Example 2: List available scenarios
# ====================================
def example_list_scenarios():
    """List all available SUMO scenarios."""
    from sim import list_simulations
    
    available = list_simulations()
    print(f"\nFound {len(available)} scenarios")


# Example 3: Generate configuration before running
# =================================================
def example_with_config_generation():
    """Generate config and run."""
    from sim import generate_sumocfg, run_in_carla
    
    # Generate configuration
    generate_sumocfg('simple4')
    
    # Run in CARLA
    run_in_carla('simple4', duration=120)


# Example 4: Advanced - Direct bridge usage
# ==========================================
def example_advanced():
    """Use the bridge directly for more control."""
    from sim.carla.bridge import CarlaSumoSync
    
    # Create co-simulation with custom settings
    cosim = CarlaSumoSync(
        sumo_cfg_file='./sim/intersections/simple4/simple4.sumocfg',
        carla_host='localhost',
        carla_port=2000,
        step_length=0.05,  # 50ms steps
        tls_manager='sumo',  # SUMO controls traffic lights
        sync_vehicle_lights=True
    )
    
    # Run for 5 minutes
    cosim.run_cosimulation(duration=300)


# Example 5: Integration with your RL agent
# ==========================================
def example_with_rl_agent():
    """
    How to integrate with reinforcement learning.
    
    This shows the basic structure - you'll need to implement
    the actual RL agent logic.
    """
    import traci
    from sim.carla.bridge import CarlaSumoSync
    
    # Start co-simulation
    cosim = CarlaSumoSync(
        sumo_cfg_file='./sim/intersections/simple4/simple4.sumocfg',
        step_length=0.05
    )
    
    # Connect to TraCI for RL control
    # Note: The bridge handles TraCI connection internally,
    # so you'll access it through the sumo_network object
    
    print("RL agent can now:")
    print("1. Observe traffic state via TraCI")
    print("2. Take actions (change traffic light phases)")
    print("3. Train while seeing results in CARLA 3D view")
    
    # Run simulation
    cosim.run_cosimulation(duration=300)


# Main execution
# ==============
if __name__ == '__main__':
    print("CARLA-SUMO Co-Simulation Examples")
    print("=" * 60)
    print("\nMake sure:")
    print("1. CARLA server is running (./CarlaUE4.exe)")
    print("2. CARLA_ROOT environment variable is set")
    print("\nUncomment one of the examples below:\n")
    
    # Uncomment the example you want to run:
    
    # example_simple()
    # example_list_scenarios()
    # example_with_config_generation()
    # example_advanced()
    # example_with_rl_agent()
    
    print("\n✓ Example file ready. Edit and run!")

