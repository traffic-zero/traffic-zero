import os
import subprocess
import traci
from typing import Optional, Dict, Any
from .generate_config import generate_sumocfg
from .data_collector import DataCollector
from .metrics import MetricsCalculator
from .tls_controller import TLSController
from .dataset import RealismMode

def run_interactive(simulation_name: str, experiment_name: str = None):
    """
    Run SUMO-GUI for interactive manual control and visual exploration.
    
    This function launches SUMO-GUI where you can manually start, pause, 
    and stop the simulation. Perfect for learning SUMO, debugging networks,
    or visually exploring traffic patterns. No programmatic control is available
    - you interact directly with the GUI.
    
    Args:
        simulation_name (str): Name of simulation from SUMO.md
                              (e.g., "simple4")
        experiment_name (str, optional): Name of experiment scenario to use.
                                         If provided, generates routes and tls
                                         from scenario before running.
    
    Example:
        >>> run_interactive("simple4")
        >>> Starting SUMO-GUI with ./sim/intersections/simple4/simple4.sumocfg
        >>> This will run interactively - you can control the simulation manually
        
        >>> run_interactive("simple4", "light_traffic")
        >>> Loading experiment scenario and generating routes/tls...
        >>> Starting SUMO-GUI with experiment configuration
    
    Note:
        If experiment_name is provided, routes and tls will be generated from
        the scenario before running. Otherwise, uses existing XML files.
    """
    base_dir = "./sim/intersections/" + simulation_name
    SUMO_CFG = os.path.join(base_dir, simulation_name + ".sumocfg")
    
    # Generate config if it doesn't exist or if experiment is specified
    if not os.path.exists(SUMO_CFG) or experiment_name is not None:
        if experiment_name is not None:
            print(f">>> Generating configuration with experiment: {experiment_name}")
        else:
            print(f">>> Generating configuration file...")
        generate_sumocfg(simulation_name, experiment_name)
    
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


def _run_simulation_loop(
    max_steps: int,
    tls_id: str,
    tls_controller: TLSController,
    data_collector: Optional[DataCollector],
    metrics_calculator: Optional[MetricsCalculator],
    enable_data_collection: bool,
) -> None:
    """Run the main simulation loop with traffic light control and data collection."""
    PROGRESS_REPORT_INTERVAL = 100
    METRICS_REPORT_INTERVAL = 1000
    QUEUE_LENGTH_THRESHOLD = 3
    EXAMPLE_LANE_ID = "eE_0"
    
    step = 0
    print(f">>> Starting simulation (max {max_steps} steps)...")
    
    while step < max_steps:
        traci.simulationStep()
        tls_controller.set_step(step)
        
        if enable_data_collection and data_collector:
            data_collector.collect_step(step)
        
        if step % PROGRESS_REPORT_INTERVAL == 0:
            phase = traci.trafficlight.getPhase(tls_id)
            print(f">>> Step {step}: TLS {tls_id} phase {phase}")
            
            should_report_metrics = (
                enable_data_collection
                and data_collector
                and metrics_calculator
                and step > 0
                and step % METRICS_REPORT_INTERVAL == 0
            )
            if should_report_metrics:
                dfs = data_collector.get_dataframes()
                if not dfs['vehicles'].empty:
                    metrics = metrics_calculator.calculate_metrics(
                        dfs['vehicles'],
                        dfs['lanes'],
                        dfs['edges'],
                        dfs['simulation'],
                    )
                    print(f">>>   Average waiting time: {metrics.get('average_waiting_time', 0):.2f}s")
                    print(f">>>   Throughput: {metrics.get('throughput', 0):.0f} vehicles")
        
        _apply_adaptive_control(step, tls_id, tls_controller, QUEUE_LENGTH_THRESHOLD, EXAMPLE_LANE_ID)
        step += 1


def _apply_adaptive_control(
    step: int,
    tls_id: str,
    tls_controller: TLSController,
    queue_threshold: int,
    example_lane_id: str,
) -> None:
    """Apply example adaptive traffic light control."""
    if step == 0:
        return
    
    phase = traci.trafficlight.getPhase(tls_id)
    if phase != 0:
        return
    
    # Example adaptive control: check East approach
    # NOTE: This is example code specific to the 'simple4' intersection layout.
    # The lane ID 'eE_0' is hardcoded for demonstration purposes.
    # For production use, lane IDs should be dynamically determined from the network
    # or made configurable via parameters/config files.
    try:
        q_len_east = traci.lane.getLastStepVehicleNumber(example_lane_id)
        if q_len_east > queue_threshold:
            print(f">>> Step {step}: Jam east: switching early")
            tls_controller.set_phase(tls_id, 2)
    except traci.TraCIException:
        # Lane ID not found - skip adaptive control for this step
        pass


def _collect_final_results(
    data_collector: Optional[DataCollector],
    metrics_calculator: Optional[MetricsCalculator],
    tls_controller: TLSController,
    enable_data_collection: bool,
) -> Dict[str, Any]:
    """Collect and export final simulation results."""
    result = {
        'data': {},
        'metrics': {},
        'tls_controller': tls_controller,
    }
    
    if not enable_data_collection or not data_collector:
        tls_controller.export_action_log()
        return result
    
    print(">>> Computing final metrics...")
    dfs = data_collector.get_dataframes()
    result['data'] = dfs
    
    if metrics_calculator:
        metrics = metrics_calculator.calculate_metrics(
            dfs['vehicles'],
            dfs['lanes'],
            dfs['edges'],
            dfs['simulation'],
        )
        result['metrics'] = metrics
        metrics_calculator.print_metrics(metrics)
        metrics_calculator.export_metrics(metrics)
    
    print(">>> Exporting data to CSV...")
    data_collector.export_to_csv()
    tls_controller.export_action_log()
    
    return result


def run_automated(
    simulation_name: str,
    experiment_name: Optional[str] = None,
    collect_interval: int = 1,
    output_dir: Optional[str] = None,
    enable_data_collection: bool = True,
    max_steps: int = 36000,
    gui: bool = False,
    realism_mode: Optional[RealismMode] = None,
) -> Dict[str, Any]:
    """
    Run SUMO with programmatic control for automated experiments.
    
    This function launches SUMO (with or without GUI) with TraCI control, allowing
    your Python code to programmatically control traffic lights, collect vehicle data,
    and implement custom traffic algorithms. The simulation runs with data
    collection, metrics computation, and traffic light control logging.
    
    Features:
        - Comprehensive TraCI data collection at configurable intervals
        - Evaluation metrics computation (waiting times, throughput, etc.)
        - Dynamic traffic light control with action logging
        - CSV export of all collected data and metrics
        - In-memory data storage for ML training access
        - Headless mode support (no GUI) for batch processing and servers
    
    Args:
        simulation_name (str): Name of simulation from SUMO.md (e.g., "simple4")
        experiment_name (str, optional): Name of experiment scenario to use.
                                        If provided, generates routes and tls
                                        from scenario before running.
        collect_interval (int): Collect data every N simulation steps (default: 1)
        output_dir (str, optional): Directory to save CSV files. If None, no files
                                   are written but data is still collected in-memory.
        enable_data_collection (bool): Enable data collection (default: True)
        max_steps (int): Maximum simulation steps (default: 36000 = 30 minutes)
        gui (bool): Show SUMO-GUI window (default: False, runs headless with 'sumo')
        realism_mode (RealismMode, optional): RealismMode instance for sensor noise.
                                             If None, uses RealismMode(RealismLevel.NONE)
    
    Returns:
        Dictionary containing:
            - 'data': Dict of pandas DataFrames (vehicles, traffic_lights, lanes,
                     junctions, edges, simulation)
            - 'metrics': Dict of computed metrics
            - 'tls_controller': TLSController instance for accessing action log
    
    Example:
        >>> result = run_automated("simple4", collect_interval=10, output_dir="./output")
        >>> vehicle_data = result['data']['vehicles']
        >>> metrics = result['metrics']
        >>> print(f"Average waiting time: {metrics['average_waiting_time']}")
        
        >>> run_automated("simple4", "rush_hour", collect_interval=5)
        >>> Loading experiment scenario and generating routes/tls...
        >>> Running simulation with experiment configuration
        
        >>> # Run with GUI (for visualization)
        >>> result = run_automated("simple4", gui=True, output_dir="./output")
        
        >>> # Run headless (default, no GUI) for batch processing
        >>> result = run_automated("simple4", output_dir="./output")
    
    Note:
        Requires the simulation files (nodes.nod.xml, edges.edg.xml, routes.rou.xml)
        to exist in the intersection folder. The function will automatically
        generate the network and configuration files if needed. If experiment_name
        is provided, routes and tls will be generated from the scenario.
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
    if experiment_name is not None:
        print(f">>> Using experiment scenario: {experiment_name}")
    generate_sumocfg(simulation_name, experiment_name)
    print(">>> SUMO configuration file generated at", SUMO_CFG)

    # 2. Launch SUMO (with or without GUI)
    if gui:
        sumo_cmd = ["sumo-gui", "-c", SUMO_CFG]
        print(">>> Starting SUMO-GUI with", SUMO_CFG)
    else:
        sumo_cmd = ["sumo", "-c", SUMO_CFG]
        print(">>> Starting SUMO (headless) with", SUMO_CFG)
    traci.start(sumo_cmd)

    # 3. Initialize data collection, metrics, and TLS controller
    data_collector = None
    metrics_calculator = None
    
    if enable_data_collection:
        print(f">>> Initializing data collection (interval: {collect_interval} steps)")
        # Option to exclude empty lanes and filter by lane type
        # Set exclude_empty_lanes=True to skip lanes with zero occupancy/density
        # Set lane_filter='entry_exit' to only collect entry/exit lanes (e*)
        # Set lane_filter='main_roads' to only collect main roads (eN_0, eS_0, eW_0, eE_0, etc.)
        data_collector = DataCollector(
            collect_interval=collect_interval,
            output_dir=output_dir,
            exclude_empty_lanes=False,  # Set to True to skip empty lanes
            lane_filter=None,  # Options: None, 'entry_exit', 'junction', 'main_roads'
            realism_mode=realism_mode,
        )
        metrics_calculator = MetricsCalculator(output_dir=output_dir)
    
    tls_controller = TLSController(output_dir=output_dir)
    print(">>> Traffic light controller initialized")

    # 4. Traffic light control and simulation loop
    tls_ids = traci.trafficlight.getIDList()
    print(">>> TLS detected:", tls_ids)

    if not tls_ids:
        print(">>> Simulation complete (no traffic lights found).")
        traci.close()
        return _collect_final_results(data_collector, metrics_calculator, tls_controller, enable_data_collection)

    # Use first TL automatically
    tls_id = tls_ids[0]
    print(">>> Using traffic light:", tls_id)

    # Try loading custom program "custom"
    try:
        tls_controller.set_step(0)
        tls_controller.set_program(tls_id, "custom")
    except Exception as e:
        print(f">>> Warning: Could not set program 'custom' - {e}")

    _run_simulation_loop(
        max_steps=max_steps,
        tls_id=tls_id,
        tls_controller=tls_controller,
        data_collector=data_collector,
        metrics_calculator=metrics_calculator,
        enable_data_collection=enable_data_collection,
    )

    print(">>> Simulation complete.")
    traci.close()
    
    return _collect_final_results(data_collector, metrics_calculator, tls_controller, enable_data_collection)