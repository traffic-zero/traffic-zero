import os
import subprocess
import traci
from .generate_config import generate_sumocfg

def run_intersection_gui(simulation_name: str):
    """
    Run SUMO-GUI directly without TraCI control for interactive simulation.
    
    Args:
        simulation_name (str): Name of simulation from SIMULATION.md
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


def run_intersection(simulation_name: str):
    """
    Run SUMO for a given intersection project folder.

    Args:
        simulation_name (str): Name of simulation form SIMULATION.md
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