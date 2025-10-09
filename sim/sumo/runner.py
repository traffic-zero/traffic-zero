import os
import subprocess
import traci
from .generate_config import generate_sumocfg

def run_interactive(simulation_name: str):
    """
    Run SUMO-GUI for interactive manual control and visual exploration.
    
    This function launches SUMO-GUI where you can manually start, pause, 
    and stop the simulation. Perfect for learning SUMO, debugging networks,
    or visually exploring traffic patterns. No programmatic control is available
    - you interact directly with the GUI.
    
    Args:
        simulation_name (str): Name of simulation from SUMO.md
                              (e.g., "simple4")
    
    Example:
        >>> run_interactive("simple4")
        >>> Starting SUMO-GUI with ./sim/intersections/simple4/simple4.sumocfg
        >>> This will run interactively - you can control the simulation manually
    
    Note:
        Make sure to generate the configuration file first using generate_sumocfg()
        if it doesn't exist.
    """
    base_dir = "./sim/intersections/" + simulation_name
    SUMO_CFG = os.path.join(base_dir, simulation_name + ".sumocfg")
    
    # Check if config file exists
    if not os.path.exists(SUMO_CFG):
        print(f"Error: Configuration file not found: {SUMO_CFG}")
        print("Run generate_sumocfg first to create the configuration file.")
        return
    
    # Launch SUMO-GUI directly
    sumo_cmd = ["sumo-gui", "-c", SUMO_CFG]
    print(f">>> Starting SUMO-GUI with {SUMO_CFG}")
    print(">>> This will run interactively - you can control the simulation manually")
    subprocess.run(sumo_cmd)


def run_automated(simulation_name: str):
    """
    Run SUMO with programmatic control for automated experiments.
    
    This function launches SUMO-GUI with TraCI control, allowing your Python
    code to programmatically control traffic lights, collect vehicle data,
    and implement custom traffic algorithms. The simulation runs for 30 minutes
    with adaptive traffic light control and provides real-time feedback.
    
    Features:
        - Automatic traffic light phase control
        - Vehicle detection and queue length monitoring  
        - Adaptive control based on traffic conditions
        - Real-time simulation statistics
    
    Args:
        simulation_name (str): Name of simulation from SUMO.md
                              (e.g., "simple4")
    
    Example:
        >>> run_automated("simple4")
        >>> Running netconvert for ./sim/intersections/simple4...
        >>> Network generated at ./sim/intersections/simple4/network.net.xml
        >>> Step 0: TLS junction phase 0
        >>> Step 100: TLS junction phase 2
    
    Note:
        Requires the simulation files (nodes.nod.xml, edges.edg.xml, routes.rou.xml)
        to exist in the intersection folder. The function will automatically
        generate the network and configuration files if needed.
    """
    base_dir = "./sim/intersections/" + simulation_name

    # Build paths
    NODES_FILE = os.path.join(base_dir, "nodes.nod.xml")
    EDGES_FILE = os.path.join(base_dir, "edges.edg.xml")
    NET_FILE   = os.path.join(base_dir, "network.net.xml")
    SUMO_CFG   = os.path.join(base_dir, simulation_name + ".sumocfg")

    # 1. Build network
    cmd = [
        "netconvert",
        f"--node-files={NODES_FILE}",
        f"--edge-files={EDGES_FILE}",
        "--tls.guess=false",
        f"--output-file={NET_FILE}"
    ]
    print(f">>> Running netconvert for {base_dir}...")
    subprocess.run(cmd, check=True)
    print(">>> Network generated at", NET_FILE)

    print(">>> Generating SUMO configuration file...")
    generate_sumocfg(simulation_name)
    print(">>> SUMO configuration file generated at", SUMO_CFG)

    # 2. Launch SUMO-GUI
    sumo_cmd = ["sumo-gui", "-c", SUMO_CFG]
    print(">>> Starting SUMO-GUI with", SUMO_CFG)
    traci.start(sumo_cmd)

    # 3. Traffic light control
    tls_ids = traci.trafficlight.getIDList()
    print("TLS detected:", tls_ids)

    if tls_ids:
        # Use first TL automatically
        tls_id = tls_ids[0]
        print("Using traffic light:", tls_id)

        # Try loading your custom program "custom"
        try:
            traci.trafficlight.setProgram(tls_id, "custom")
        except traci.TraCIException as e:
            print("⚠️ Warning: Could not set program 'custom' -", e)

        # Run simulation loop for much longer
        step = 0
        while step < 36000:  # Run for 30 minutes (36000 * 0.05s = 1800s)
            traci.simulationStep()
            phase = traci.trafficlight.getPhase(tls_id)
            
            # Print every 100 steps to avoid spam
            if step % 100 == 0:
                print(f"Step {step}: TLS {tls_id} phase {phase}")

            # Example adaptive control: check East approach
            if phase == 0:
                # Update lane id to match your network naming
                try:
                    q_len_east = traci.lane.getLastStepVehicleNumber("eE_0")
                    if q_len_east > 3:
                        print(f"Step {step}: Jam east: switching early")
                        traci.trafficlight.setPhase(tls_id, 2)
                except traci.TraCIException:
                    pass  # skip if lane id not found
            
            step += 1

    print(">>> Simulation complete.")
    traci.close()