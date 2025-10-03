import os
import subprocess
import traci

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
    SUMO_CFG   = os.path.join(base_dir, ".sumocfg")

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

        # Run simulation loop
        for step in range(100):
            traci.simulationStep()
            phase = traci.trafficlight.getPhase(tls_id)
            print(f"Step {step}: TLS {tls_id} phase {phase}")

            # Example adaptive control: check East approach
            if phase == 0:
                # Update lane id to match your network naming
                try:
                    q_len_east = traci.lane.getLastStepVehicleNumber("eE_0")
                    if q_len_east > 3:
                        print("--> Jam east: switching early")
                        traci.trafficlight.setPhase(tls_id, 2)
                except traci.TraCIException:
                    pass  # skip if lane id not found

    print(">>> Simulation complete.")
    traci.close()