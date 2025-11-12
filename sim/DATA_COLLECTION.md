# Data Collection and Traffic Light Control Guide

This guide explains how to use the automated SUMO simulation with comprehensive data collection, metrics computation, and dynamic traffic light control.

## Overview

The enhanced `run_automated()` function provides:

- **Comprehensive TraCI data collection** at configurable intervals
- **Evaluation metrics computation** (waiting times, throughput, emissions, etc.)
- **Dynamic traffic light control** with action logging
- **CSV export** of all collected data and metrics
- **In-memory data storage** for ML training access

## Get Started

### Quick Test Run

To verify that data collection is working correctly, run a test simulation with data collection enabled:

```bash
python run_simulation.py --generate_dataset
```

**What this does:**

1. Runs the `light_traffic_random` scenario for 1000 simulation steps
2. Collects all TraCI data (vehicles, traffic lights, lanes, edges, simulation state)
3. Computes evaluation metrics (waiting times, throughput, emissions, etc.)
4. Exports all data to CSV files in `./data/light_traffic_random/`
5. Prints a summary of collected data and metrics

**Expected output:**

- Console output showing simulation progress (runs headless by default, no GUI window)
- Data collection summary with metrics
- CSV files created in the output directory
- **Note**: To see the GUI, add `gui=True` to the `run_automated()` call

**Output files created:**

- `vehicle_data.csv` - Vehicle positions, speeds, waiting times, emissions
- `traffic_light_data.csv` - Traffic light states and phases
- `traffic_light_actions.csv` - Log of all traffic light control actions
- `lane_data.csv` - Lane occupancy, density, queue lengths
- `junction_data.csv` - Junction positions
- `edge_data.csv` - Edge speeds, travel times, occupancy
- `simulation_data.csv` - Simulation state over time
- `metrics_summary.csv` - Final evaluation metrics

If the simulation runs successfully and you see the data collection summary, you're ready to use the data collection system!

## Basic Usage

### Simple Run with Default Settings

```python
from sim import run_automated

# Run simulation with default settings (collects data every step)
result = run_automated("simple4")
```

This will:

- Collect all TraCI data at every simulation step
- Compute metrics at the end
- Return data and metrics in a dictionary
- **Note**: No CSV files are written unless `output_dir` is specified

### Run with CSV Export

```python
from sim import run_automated

# Run simulation and save all data to CSV files
result = run_automated(
    simulation_name="simple4",
    output_dir="./simulation_output"
)
```

This creates the following CSV files in `./simulation_output/`:

- `vehicle_data.csv` - All vehicle data collected
- `traffic_light_data.csv` - Traffic light states at collection points
- `traffic_light_actions.csv` - Log of all traffic light control actions
- `lane_data.csv` - Lane metrics
- `junction_data.csv` - Junction metrics
- `edge_data.csv` - Edge metrics
- `simulation_data.csv` - Simulation state
- `metrics_summary.csv` - Final evaluation metrics

### Run with Experiment Scenario

```python
from sim import run_automated

# Run with experiment scenario (generates routes/tls from YAML)
result = run_automated(
    simulation_name="simple4",
    experiment_name="light_traffic",
    output_dir="./output/light_traffic"
)
```

## Advanced Configuration

### Configurable Data Collection Interval

To reduce data volume, collect data every N steps:

```python
from sim import run_automated

# Collect data every 10 steps (reduces data volume by 10x)
result = run_automated(
    simulation_name="simple4",
    collect_interval=10,  # Collect every 10 steps
    output_dir="./output"
)
```

**When to use intervals:**

- **Every step (interval=1)**: For detailed analysis, short simulations, or when you need precise timing
- **Every N steps (interval>1)**: For long simulations, reduced storage, or when high-frequency data isn't needed

### Custom Simulation Duration

```python
from sim import run_automated

# Run for 1000 steps instead of default 36000
result = run_automated(
    simulation_name="simple4",
    max_steps=1000,
    output_dir="./output"
)
```

### Disable Data Collection

If you only need traffic light control without data collection:

```python
from sim import run_automated

result = run_automated(
    simulation_name="simple4",
    enable_data_collection=False
)
```

## Accessing Collected Data

### In-Memory DataFrames

The function returns a dictionary with all collected data:

```python
from sim import run_automated

result = run_automated("simple4", output_dir="./output")

# Access vehicle data
vehicle_df = result['data']['vehicles']
print(f"Collected {len(vehicle_df)} vehicle data points")
print(vehicle_df.head())

# Access traffic light data
tls_df = result['data']['traffic_lights']
print(tls_df.head())

# Access lane data
lane_df = result['data']['lanes']
print(lane_df.head())

# Access metrics
metrics = result['metrics']
print(f"Average waiting time: {metrics['average_waiting_time']:.2f}s")
print(f"Max waiting time: {metrics['max_waiting_time']:.2f}s")
print(f"Throughput: {metrics['throughput']:.0f} vehicles")
```

### Available DataFrames

The `result['data']` dictionary contains:

- `vehicles`: Vehicle positions, speeds, waiting times, emissions, fuel consumption
- `traffic_lights`: Traffic light phases, programs, states
- `lanes`: Lane occupancy, density, queue lengths, speeds
- `junctions`: Junction waiting times
- `edges`: Edge speeds, travel times, occupancy
- `simulation`: Simulation state (time, vehicle counts, departed/arrived)

### Data Columns

#### Vehicle Data (`vehicles`)

- `step`, `time`: Simulation step and time
- `vehicle_id`: Vehicle identifier
- `position_x`, `position_y`: Vehicle position
- `speed`, `acceleration`: Vehicle speed and acceleration
- `angle`: Vehicle heading angle
- `waiting_time`: Time vehicle has been waiting
- `lane_id`, `lane_position`: Current lane and position in lane
- `route`: Vehicle route (comma-separated edges)
- `co2_emission`, `co_emission`, `nox_emission`: Emissions
- `fuel_consumption`: Fuel consumption

#### Traffic Light Data (`traffic_lights`)

- `step`, `time`: Simulation step and time
- `tls_id`: Traffic light identifier
- `phase`: Current phase number
- `phase_duration`: Duration of current phase
- `program`: Current program name
- `state`: Traffic light state string (e.g., "GGGrrrGGGrrr")
- `controlled_lanes`: Comma-separated list of controlled lanes

#### Lane Data (`lanes`)

- `step`, `time`: Simulation step and time
- `lane_id`: Lane identifier
- `occupancy`: Lane occupancy (0-100%)
- `density`: Vehicle density (vehicles/km)
- `vehicle_count`: Number of vehicles in lane
- `mean_speed`: Mean speed of vehicles in lane
- `waiting_time`: Total waiting time in lane
- `queue_length`: Number of halted vehicles

## Traffic Light Control

### Using the TLS Controller

The `run_automated()` function returns a `TLSController` instance that you can use to control traffic lights during simulation. However, for programmatic control, you should access it within a custom control loop.

### Basic Traffic Light Control

Here's how to modify the simulation loop to control traffic lights:

```python
from sim.sumo.runner import run_automated
from sim.sumo.tls_controller import TLSController
import traci

# Start simulation (you'll need to modify runner.py or create custom loop)
# For now, traffic lights are controlled via the TLSController returned

result = run_automated("simple4", output_dir="./output")

# Access the controller
tls_controller = result['tls_controller']

# Get action log
action_log = tls_controller.get_action_log()
print("Traffic light actions:")
print(action_log)
```

### Custom Control Loop Example

To have full control during simulation, you can create a custom control function:

```python
from sim.sumo.data_collector import DataCollector
from sim.sumo.metrics import MetricsCalculator
from sim.sumo.tls_controller import TLSController
import traci
import os
import subprocess

def run_with_custom_control(simulation_name, output_dir="./output"):
    """Run simulation with custom traffic light control."""

    # Setup SUMO (similar to run_automated)
    base_dir = f"./sim/intersections/{simulation_name}"
    SUMO_CFG = os.path.join(base_dir, f"{simulation_name}.sumocfg")

    # Start SUMO (headless by default, use ["sumo-gui", ...] for GUI)
    sumo_cmd = ["sumo", "-c", SUMO_CFG]
    traci.start(sumo_cmd)

    # Initialize controllers
    data_collector = DataCollector(collect_interval=1, output_dir=output_dir)
    metrics_calculator = MetricsCalculator(output_dir=output_dir)
    tls_controller = TLSController(output_dir=output_dir)

    # Get traffic lights
    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids:
        print("No traffic lights found!")
        traci.close()
        return

    tls_id = tls_ids[0]

    # Simulation loop
    step = 0
    max_steps = 1000

    while step < max_steps:
        traci.simulationStep()
        tls_controller.set_step(step)

        # Collect data
        data_collector.collect_step(step)

        # YOUR CUSTOM CONTROL LOGIC HERE
        # Example: Change phase based on queue length
        phase = traci.trafficlight.getPhase(tls_id)
        queue_length = traci.lane.getLastStepVehicleNumber("eE_0")

        if queue_length > 5 and phase == 0:
            # Switch to phase 2 if queue is too long
            tls_controller.set_phase(tls_id, 2)
            print(f"Step {step}: Changed phase due to queue length {queue_length}")

        step += 1

    # Finalize
    dfs = data_collector.get_dataframes()
    metrics = metrics_calculator.calculate_metrics(
        dfs['vehicles'], dfs['lanes'], dfs['edges'], dfs['simulation']
    )

    data_collector.export_to_csv()
    metrics_calculator.export_metrics(metrics)
    tls_controller.export_action_log()

    traci.close()

    return {'data': dfs, 'metrics': metrics, 'tls_controller': tls_controller}
```

### Traffic Light Control Methods

The `TLSController` provides these methods:

#### Set Phase

```python
tls_controller.set_phase(tls_id="junction", phase=2)
```

Changes the traffic light to a specific phase. The action is automatically logged.

#### Set Program

```python
tls_controller.set_program(tls_id="junction", program="custom")
```

Changes the traffic light program. Useful for switching between different timing plans.

#### Set Phase Duration

```python
tls_controller.set_phase_duration(tls_id="junction", phase=0, duration=30.0)
```

Sets the duration for a specific phase in seconds.

#### Get Current State

```python
state = tls_controller.get_current_state(tls_id="junction")
print(state)
# Output: {'phase': 0, 'phase_duration': 20.0, 'program': 'custom',
#          'state': 'GGGrrrGGGrrr', 'controlled_lanes': ['lane1', 'lane2', ...]}
```

#### Get All Traffic Light IDs

```python
tls_ids = tls_controller.get_all_tls_ids()
print(f"Found {len(tls_ids)} traffic lights: {tls_ids}")
```

### Accessing Traffic Light Action Log

All traffic light control actions are automatically logged:

```python
result = run_automated("simple4", output_dir="./output")

# Get action log as DataFrame
action_log = result['tls_controller'].get_action_log()
print(action_log)

# Action log columns:
# - step: Simulation step when action occurred
# - time: Simulation time when action occurred
# - tls_id: Traffic light identifier
# - action_type: Type of action (phase_change, program_change, duration_change)
# - old_value: Previous value
# - new_value: New value
# - controlled_lanes: Lanes controlled by this traffic light
```

The action log is also automatically exported to `traffic_light_actions.csv` if `output_dir` is specified.

## Metrics

### Available Metrics

The metrics calculator computes the following metrics:

#### Primary Metrics

- `average_waiting_time`: Average waiting time across all vehicles (seconds)
- `max_waiting_time`: Maximum waiting time of any vehicle (seconds)
- `total_waiting_time`: Sum of all waiting times (seconds)
- `average_travel_time`: Average travel time across edges (seconds)
- `throughput`: Total number of vehicles that arrived (vehicles)

#### Queue Metrics

- `max_queue_length`: Maximum queue length across all lanes (vehicles)
- `average_queue_length`: Average queue length across all lanes (vehicles)
- `total_queue_length`: Sum of all queue lengths (vehicles)

#### Speed Metrics

- `average_speed`: Average vehicle speed (m/s)
- `max_speed`: Maximum vehicle speed (m/s)
- `min_speed`: Minimum vehicle speed (m/s)
- `average_lane_speed`: Average lane speed (m/s)
- `average_edge_speed`: Average edge speed (m/s)

#### Emissions Metrics

- `total_co2_emission`: Total CO2 emissions (mg)
- `average_co2_emission`: Average CO2 emissions per vehicle (mg)
- `total_co_emission`: Total CO emissions (mg)
- `total_nox_emission`: Total NOx emissions (mg)

#### Fuel Consumption

- `total_fuel_consumption`: Total fuel consumed (ml)
- `average_fuel_consumption`: Average fuel consumption per vehicle (ml)

#### Lane Metrics

- `max_lane_occupancy`: Maximum lane occupancy (%)
- `average_lane_occupancy`: Average lane occupancy (%)
- `max_lane_density`: Maximum lane density (vehicles/km)
- `average_lane_density`: Average lane density (vehicles/km)

#### Simulation Metrics

- `simulation_duration`: Total simulation duration (seconds)
- `unique_vehicles`: Number of unique vehicles in simulation
- `average_vehicle_count`: Average number of vehicles present
- `max_vehicle_count`: Maximum number of vehicles present

### Accessing Metrics

```python
result = run_automated("simple4", output_dir="./output")

metrics = result['metrics']

# Print all metrics
for key, value in metrics.items():
    print(f"{key}: {value}")

# Access specific metrics
avg_wait = metrics['average_waiting_time']
max_wait = metrics['max_waiting_time']
throughput = metrics['throughput']

print(f"Performance: {throughput} vehicles, avg wait {avg_wait:.2f}s, max wait {max_wait:.2f}s")
```

Metrics are automatically printed at the end of simulation and exported to `metrics_summary.csv`.

## Data Analysis Examples

### Analyze Vehicle Waiting Times

```python
import pandas as pd
from sim import run_automated

result = run_automated("simple4", output_dir="./output")

vehicle_df = result['data']['vehicles']

# Filter vehicles with waiting time > 0
waiting_vehicles = vehicle_df[vehicle_df['waiting_time'] > 0]

# Group by vehicle to get total waiting time per vehicle
vehicle_wait_times = vehicle_df.groupby('vehicle_id')['waiting_time'].max()
print(f"Vehicles that waited: {len(vehicle_wait_times)}")
print(f"Max waiting time: {vehicle_wait_times.max():.2f}s")
print(f"Average waiting time: {vehicle_wait_times.mean():.2f}s")
```

### Analyze Traffic Light Phase Distribution

```python
from sim import run_automated

result = run_automated("simple4", output_dir="./output")

tls_df = result['data']['traffic_lights']

# Count phase occurrences
phase_counts = tls_df['phase'].value_counts()
print("Phase distribution:")
print(phase_counts)

# Calculate average phase duration
avg_durations = tls_df.groupby('phase')['phase_duration'].mean()
print("\nAverage phase durations:")
print(avg_durations)
```

### Analyze Lane Occupancy Over Time

```python
from sim import run_automated
import matplotlib.pyplot as plt

result = run_automated("simple4", output_dir="./output")

lane_df = result['data']['lanes']

# Plot occupancy over time for a specific lane
specific_lane = lane_df[lane_df['lane_id'] == 'eE_0']
if not specific_lane.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(specific_lane['time'], specific_lane['occupancy'])
    plt.xlabel('Time (s)')
    plt.ylabel('Occupancy (%)')
    plt.title('Lane Occupancy Over Time')
    plt.grid(True)
    plt.show()
```

### Compare Different Scenarios

```python
from sim import run_automated

scenarios = ["light_traffic", "rush_hour", "default"]
results = {}

for scenario in scenarios:
    print(f"\nRunning scenario: {scenario}")
    result = run_automated(
        "simple4",
        experiment_name=scenario,
        output_dir=f"./output/{scenario}",
        collect_interval=10  # Reduce data for comparison
    )
    results[scenario] = result['metrics']

# Compare metrics
comparison_df = pd.DataFrame(results).T
print("\nScenario Comparison:")
print(comparison_df[['average_waiting_time', 'max_waiting_time', 'throughput']])
```

## Tips and Best Practices

### 1. Data Collection Interval

- Use `collect_interval=1` for short simulations or detailed analysis
- Use `collect_interval=10` or higher for long simulations to reduce storage
- Consider your analysis needs: do you need every step or is sampling sufficient?

### 2. Output Directory Organization

```python
# Organize outputs by experiment
output_dir = f"./experiments/{experiment_name}/run_{run_number}"
result = run_automated("simple4", experiment_name="light_traffic", output_dir=output_dir)
```

### 3. Memory Management

For very long simulations with `collect_interval=1`, data can become large. Consider:

- Using larger collection intervals
- Processing data in chunks
- Exporting to CSV and clearing in-memory data periodically

### 4. Traffic Light Control

- Always use `TLSController` methods instead of direct `traci.trafficlight` calls to ensure actions are logged
- Check traffic light state before making changes
- Consider minimum phase durations to avoid rapid switching

### 5. Metrics Interpretation

- `throughput`: Number of vehicles that completed their journey
- `average_waiting_time`: Good indicator of overall system performance
- `max_waiting_time`: Identifies worst-case scenarios
- Queue lengths: Indicate congestion hotspots

## Troubleshooting

### No Data Collected

- Check that `enable_data_collection=True` (default)
- Verify SUMO simulation is running (check for TraCI connection)
- Ensure vehicles exist in simulation

### Missing CSV Files

- Verify `output_dir` is specified and writable
- Check that data collection is enabled
- Ensure simulation ran to completion

### Traffic Light Control Not Working

- Verify traffic light ID exists: `tls_controller.get_all_tls_ids()`
- Check current state: `tls_controller.get_current_state(tls_id)`
- Ensure you're using valid phase numbers for the program

### Memory Issues

- Reduce `collect_interval` to collect less frequently
- Use `output_dir` to export data and clear memory
- Process data in batches for long simulations

## Next Steps

For ML training:

1. Use collected data to create state representations
2. Use traffic light action log as action history
3. Use metrics to compute rewards
4. Combine vehicle, lane, and traffic light data for observations

Example ML data preparation:

```python
result = run_automated("simple4", output_dir="./training_data")

# Prepare state: combine lane occupancy, queue lengths, vehicle positions
lane_data = result['data']['lanes']
vehicle_data = result['data']['vehicles']
tls_data = result['data']['traffic_lights']

# Prepare actions: traffic light phase changes
actions = result['tls_controller'].get_action_log()

# Prepare rewards: based on metrics
rewards = -result['metrics']['average_waiting_time']  # Negative waiting time as reward
```

## API Reference

### `run_automated()` Parameters

- `simulation_name` (str, required): Name of simulation (e.g., "simple4")
- `experiment_name` (str, optional): Experiment scenario name
- `collect_interval` (int, default=1): Collect data every N steps
- `output_dir` (str, optional): Directory for CSV exports
- `enable_data_collection` (bool, default=True): Enable/disable data collection
- `max_steps` (int, default=36000): Maximum simulation steps
- `gui` (bool, default=False): Show SUMO-GUI window (default: headless mode, no GUI)

### Return Value

Dictionary with keys:

- `data`: Dict of pandas DataFrames (vehicles, traffic_lights, lanes, junctions, edges, simulation)
- `metrics`: Dict of computed metrics
- `tls_controller`: TLSController instance

### TLSController Methods

- `set_phase(tls_id, phase)`: Set traffic light phase
- `set_program(tls_id, program)`: Set traffic light program
- `set_phase_duration(tls_id, phase, duration)`: Set phase duration
- `get_current_state(tls_id)`: Get current traffic light state
- `get_all_tls_ids()`: Get list of all traffic light IDs
- `get_action_log()`: Get action log as DataFrame
