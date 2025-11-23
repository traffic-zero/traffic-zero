# Experiments Guide

This document explains how to create, validate, and test traffic simulation experiments using the scenario generation system.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Writing Scenario Files](#writing-scenario-files)
4. [Schema Validation](#schema-validation)
5. [Testing Experiments](#testing-experiments)
6. [Python API](#python-api)
7. [Examples](#examples)
8. [Best Practices](#best-practices)
9. [Integration with RL](#integration-with-rl)

## Overview

The experiment system allows you to define traffic scenarios using YAML configuration files. Each scenario specifies:

- **Traffic patterns**: Vehicle types, routes, and spawning behavior (programmatic or statistical)
- **Traffic light programs**: Phase definitions and timing for traffic lights

Scenarios are stored in `experiments/<intersection_name>/<experiment_name>.yaml` and can be used to generate SUMO XML files (`routes.rou.xml` and `tls.add.xml`) automatically.

### Key Benefits

- **Python-based**: Easy to programmatically create and modify scenarios
- **Validated**: JSON schema ensures correctness before simulation
- **Flexible**: Support both exact vehicle definitions and statistical flows
- **RL-ready**: Traffic light programs can be overridden at runtime for reinforcement learning

## Directory Structure

```
experiments/
├── scenarios/
│   └── schema.json          # JSON schema for validation
├── <intersection_name>/     # e.g., simple4
│   ├── <experiment_name>.yaml
│   ├── <experiment_name>.yaml
│   └── ...
└── EXPERIMENTS.md           # This file
```

### Naming Conventions

- **Intersection names**: Match the folder name in `sim/intersections/` (e.g., `simple4`)
- **Experiment names**: Descriptive names using lowercase with underscores (e.g., `light_traffic`, `rush_hour`, `baseline_1`)

## Writing Scenario Files

### Basic Structure

Every scenario YAML file must have three main sections:

```yaml
intersection: <intersection_name>
traffic:
  vehicle_types: [...]
  routes: [...]
  vehicles: [...]        # Optional: for programmatic spawning
  flows: [...]          # Optional: for statistical spawning
traffic_lights:
  programs: [...]
```

### Traffic Configuration

#### Vehicle Types

Define vehicle characteristics that will be used in the simulation:

```yaml
traffic:
  vehicle_types:
    - id: car
      accel: 2.0          # Acceleration (m/s²)
      decel: 4.5          # Deceleration (m/s²)
      sigma: 0.5          # Driver imperfection (0-1)
      length: 5           # Vehicle length (m)
      maxSpeed: 13.9      # Maximum speed (m/s)
      color: "1,0,0"      # RGB color (optional)
      # Optional attributes:
      minGap: 2.5
      tau: 1.0
      speedFactor: 1.0
      speedDev: 0.1
```

**Required fields**: `id`, `accel`, `decel`, `sigma`, `length`, `maxSpeed`

#### Routes

Define paths through the intersection network:

```yaml
traffic:
  routes:
    - id: north_south
      edges: [eN, eS_out]
    - id: east_west
      edges: [eE, eW_out]
```

**Required fields**: `id`, `edges` (list of edge IDs from the network)

**Note**: Edge IDs must match those defined in the intersection's `edges.edg.xml` file.

#### Vehicles (Programmatic Spawning)

Define exact vehicles with specific departure times:

```yaml
traffic:
  vehicles:
    - id: veh0
      type: car
      route: north_south
      depart: 0
    - id: veh1
      type: car
      route: east_west
      depart: 5.5
      departLane: "best"      # Optional
      departPos: 0            # Optional
      departSpeed: 0          # Optional
```

**Required fields**: `id`, `type`, `route`, `depart`

**Use cases**:
- Precise control over vehicle timing
- Reproducible scenarios
- Testing specific traffic patterns

#### Flows (Statistical Spawning)

Define statistical vehicle flows:

```yaml
traffic:
  flows:
    - route: north_south
      type: car
      begin: 0
      end: 3600
      vehsPerHour: 180        # Vehicles per hour
    - route: east_west
      type: car
      begin: 0
      end: 1800
      period: 20              # Alternative: seconds between vehicles
    - route: south_north
      type: car
      begin: 0
      end: 3600
      probability: 0.1        # Alternative: probability per second
    - route: west_east
      type: car
      begin: 0
      end: 3600
      number: 50              # Alternative: exact number of vehicles
```

**Required fields**: `route`, `type`, `begin`, `end`

**Flow rate options** (use one):
- `vehsPerHour`: Average vehicles per hour
- `period`: Fixed interval between vehicles (seconds)
- `probability`: Probability of spawning each second (0-1)
- `number`: Exact number of vehicles to spawn

**Use cases**:
- Realistic traffic patterns
- Variable traffic intensity
- Long-duration simulations

### Traffic Light Configuration

Define traffic light programs with phases:

```yaml
traffic_lights:
  programs:
    - id: n0                 # Traffic light ID (must match network)
      program_id: custom     # Program identifier
      offset: 0               # Start offset (seconds)
      type: static            # static or actuated
      phases:
        - duration: 20       # Phase duration (seconds)
          state: "GGGGrrrrGGGGrrrr"  # Light states
        - duration: 4
          state: "yyyyrrrryyyyrrrr"
        - duration: 20
          state: "rrrrGGGGrrrrGGGG"
        - duration: 4
          state: "rrrryyyyrrrryyyy"
```

**Phase State String**

The state string defines the light state for each lane:
- `G` = Green
- `r` = Red
- `y` = Yellow
- `g` = Green (priority, not used in standard traffic lights)

The length of the string must match the number of controlled lanes. For a 4-way intersection, typically 16 characters (4 lanes × 4 directions).

**Required fields**: `id`, `program_id`, `phases` (with `duration` and `state`)

**Optional phase fields**:
- `minDur`: Minimum duration (for actuated lights)
- `maxDur`: Maximum duration (for actuated lights)
- `name`: Phase name for reference

## Schema Validation

All scenario files are automatically validated against a JSON schema (`experiments/scenarios/schema.json`) to ensure:

- Required fields are present
- Data types are correct
- Values are within valid ranges
- Structure matches expected format

### Validation Errors

If validation fails, you'll see detailed error messages indicating:
- Missing required fields
- Invalid data types
- Out-of-range values
- Structural issues

Example error:
```
jsonschema.exceptions.ValidationError: 'vehsPerHour' is a required property
```

### Manual Validation

You can validate a scenario programmatically:

```python
from sim.scenarios import load_scenario, validate_scenario
import yaml

# Load and validate
scenario = load_scenario("simple4", "my_experiment")

# Or validate raw YAML data
with open("experiments/simple4/my_experiment.yaml") as f:
    data = yaml.safe_load(f)
validate_scenario(data)
```

## Testing Experiments

### Quick Test with Interactive Mode

The easiest way to test an experiment is using interactive mode:

```python
from sim import run_interactive

# Test with experiment
run_interactive("simple4", experiment_name="light_traffic")
```

This will:
1. Load the scenario from `experiments/simple4/light_traffic.yaml`
2. Generate `routes.rou.xml` and `tls.add.xml` in the intersection folder
3. Generate/update `simple4.sumocfg`
4. Launch SUMO-GUI for visual inspection

### Automated Testing

For programmatic testing and data collection:

```python
from sim import run_automated

# Run with experiment
run_automated("simple4", experiment_name="rush_hour")
```

This runs the simulation with TraCI control for 30 minutes, allowing you to:
- Monitor traffic metrics
- Collect performance data
- Test RL agents (future)

### Command Line Testing

You can also test scenarios directly:

```python
from sim.scenarios import load_scenario, generate_routes_xml, generate_tls_xml
from pathlib import Path

# Load scenario
scenario = load_scenario("simple4", "my_experiment")

# Generate XML files
base_dir = Path("./sim/intersections/simple4")
generate_routes_xml(scenario, base_dir / "routes.rou.xml")
generate_tls_xml(scenario, base_dir / "tls.add.xml")

# Then run SUMO normally
```

## Python API

### Loading Scenarios

```python
from sim.scenarios import load_scenario, load_scenario_from_path

# Load from experiments directory
scenario = load_scenario("simple4", "light_traffic")

# Load from explicit path
scenario = load_scenario_from_path(Path("experiments/simple4/custom.yaml"))
```

### Creating Scenarios Programmatically

```python
from sim.scenarios import (
    Scenario, TrafficConfig, TrafficLightConfig,
    VehicleType, Route, Vehicle, Flow, Phase, TrafficLightProgram
)

# Create vehicle type
car_type = VehicleType(
    id="car",
    accel=2.0,
    decel=4.5,
    length=5.0,
    maxSpeed=13.9
)

# Create routes
routes = [
    Route(id="north_south", edges=["eN", "eS_out"]),
    Route(id="east_west", edges=["eE", "eW_out"])
]

# Create flows
flows = [
    Flow(
        route="north_south",
        type="car",
        begin=0,
        end=3600,
        vehsPerHour=180
    )
]

# Create traffic config
traffic = TrafficConfig(
    vehicle_types=[car_type],
    routes=routes,
    flows=flows
)

# Create traffic light program
phases = [
    Phase(duration=20, state="GGGGrrrrGGGGrrrr"),
    Phase(duration=4, state="yyyyrrrryyyyrrrr"),
    Phase(duration=20, state="rrrrGGGGrrrrGGGG"),
    Phase(duration=4, state="rrrryyyyrrrryyyy")
]

tls_program = TrafficLightProgram(
    id="n0",
    program_id="custom",
    offset=0,
    phases=phases
)

traffic_lights = TrafficLightConfig(programs=[tls_program])

# Create scenario
scenario = Scenario(
    intersection="simple4",
    traffic=traffic,
    traffic_lights=traffic_lights
)

# Save scenario
from sim.scenarios import save_scenario
save_scenario(scenario, "simple4", "my_custom_experiment")
```

### Listing Experiments

```python
from sim.scenarios import list_experiments

# List all experiments for an intersection
experiments = list_experiments("simple4")
print(experiments)  # ['default', 'light_traffic', 'rush_hour']
```

### Generating XML

```python
from sim.scenarios import generate_routes_xml, generate_tls_xml
from pathlib import Path

scenario = load_scenario("simple4", "my_experiment")
base_dir = Path("./sim/intersections/simple4")

# Generate routes
generate_routes_xml(scenario, base_dir / "routes.rou.xml")

# Generate traffic lights
generate_tls_xml(scenario, base_dir / "tls.add.xml")
```

## Examples

### Example 1: Simple Programmatic Scenario

```yaml
intersection: simple4
traffic:
  vehicle_types:
    - id: car
      accel: 2.0
      decel: 4.5
      length: 5
      maxSpeed: 13.9
  routes:
    - id: route1
      edges: [eN, eS_out]
  vehicles:
    - id: veh0
      type: car
      route: route1
      depart: 0
traffic_lights:
  programs:
    - id: n0
      program_id: custom
      offset: 0
      phases:
        - duration: 30
          state: "GGGGrrrrGGGGrrrr"
        - duration: 5
          state: "yyyyrrrryyyyrrrr"
        - duration: 30
          state: "rrrrGGGGrrrrGGGG"
        - duration: 5
          state: "rrrryyyyrrrryyyy"
```

### Example 2: Statistical Flow Scenario

```yaml
intersection: simple4
traffic:
  vehicle_types:
    - id: car
      accel: 2.0
      decel: 4.5
      length: 5
      maxSpeed: 13.9
  routes:
    - id: north_south
      edges: [eN, eS_out]
    - id: east_west
      edges: [eE, eW_out]
  flows:
    - route: north_south
      type: car
      begin: 0
      end: 3600
      vehsPerHour: 240
    - route: east_west
      type: car
      begin: 0
      end: 3600
      vehsPerHour: 180
traffic_lights:
  programs:
    - id: n0
      program_id: custom
      offset: 0
      phases:
        - duration: 20
          state: "GGGGrrrrGGGGrrrr"
        - duration: 4
          state: "yyyyrrrryyyyrrrr"
        - duration: 20
          state: "rrrrGGGGrrrrGGGG"
        - duration: 4
          state: "rrrryyyyrrrryyyy"
```

### Example 3: Variable Traffic Intensity

```yaml
intersection: simple4
traffic:
  vehicle_types:
    - id: car
      accel: 2.0
      decel: 4.5
      length: 5
      maxSpeed: 13.9
  routes:
    - id: north_south
      edges: [eN, eS_out]
  flows:
    # Rush hour: high traffic
    - route: north_south
      type: car
      begin: 0
      end: 1800
      vehsPerHour: 360
    # Off-peak: low traffic
    - route: north_south
      type: car
      begin: 1800
      end: 3600
      vehsPerHour: 120
traffic_lights:
  programs:
    - id: n0
      program_id: custom
      offset: 0
      phases:
        - duration: 20
          state: "GGGGrrrrGGGGrrrr"
        - duration: 4
          state: "yyyyrrrryyyyrrrr"
        - duration: 20
          state: "rrrrGGGGrrrrGGGG"
        - duration: 4
          state: "rrrryyyyrrrryyyy"
```

## Best Practices

### 1. Naming Conventions

- Use descriptive experiment names: `rush_hour`, `light_traffic`, `baseline_v1`
- Avoid spaces and special characters
- Use lowercase with underscores

### 2. Vehicle Types

- Define vehicle types that match your use case
- Use realistic parameters (check SUMO documentation for defaults)
- Consider creating multiple types for variety (cars, trucks, buses)

### 3. Routes

- Verify edge IDs match your network (`sim/intersections/<name>/edges.edg.xml`)
- Use descriptive route IDs
- Ensure routes are valid paths through the network

### 4. Traffic Patterns

- **Programmatic vehicles**: Use for precise control, testing, reproducibility
- **Statistical flows**: Use for realistic traffic, long simulations, variable intensity
- Mix both if needed (vehicles for specific events, flows for background traffic)

### 5. Traffic Lights

- Match traffic light IDs to network nodes
- Ensure phase state strings match the number of controlled lanes
- Use appropriate phase durations (green: 15-30s, yellow: 3-5s)
- Test phase sequences for safety (always include yellow between green changes)

### 6. Validation

- Always validate scenarios before running long simulations
- Check for common errors:
  - Missing required fields
  - Invalid edge IDs
  - Mismatched route/vehicle type references
  - Invalid phase state strings

### 7. Version Control

- Commit scenario files to version control
- Use descriptive commit messages
- Tag important experiment versions

### 8. Documentation

- Add comments in YAML for complex scenarios
- Document experiment purpose and parameters
- Note any special considerations or assumptions

## Integration with RL

The scenario system is designed to work seamlessly with reinforcement learning:

### Static Baseline

Traffic light programs defined in scenarios serve as:
- **Baseline controllers**: Compare RL performance against
- **Initial state**: Start RL training from a known good configuration
- **Fallback**: Revert to static program if RL fails

### Dynamic Override

RL agents can override traffic light programs at runtime using TraCI:

```python
import traci

# Start simulation with scenario
run_automated("simple4", experiment_name="baseline")

# RL agent can override traffic lights
tls_id = "n0"
traci.trafficlight.setPhase(tls_id, new_phase)
traci.trafficlight.setPhaseDuration(tls_id, new_duration)

# Or set custom program
traci.trafficlight.setProgram(tls_id, "rl_controlled")
```

### Experiment Design for RL

1. **Baseline scenarios**: Create scenarios with static traffic light programs
2. **Training scenarios**: Use statistical flows for varied traffic conditions
3. **Evaluation scenarios**: Use programmatic vehicles for reproducible testing
4. **Progressive difficulty**: Start with light traffic, increase to heavy traffic

### Example RL Workflow

```python
# 1. Load baseline scenario
from sim.scenarios import load_scenario
scenario = load_scenario("simple4", "baseline")

# 2. Run simulation with RL agent
from sim import run_automated
# (RL agent would be integrated here)
run_automated("simple4", experiment_name="baseline")

# 3. Compare performance
# - Average waiting time
# - Throughput
# - Queue lengths
# - Travel times
```

## Troubleshooting

### Common Issues

1. **"Scenario file not found"**
   - Check file path: `experiments/<intersection>/<experiment>.yaml`
   - Verify intersection name matches directory

2. **"Validation error"**
   - Check JSON schema: `experiments/scenarios/schema.json`
   - Verify all required fields are present
   - Check data types match schema

3. **"Edge not found"**
   - Verify edge IDs in routes match network
   - Check `sim/intersections/<name>/edges.edg.xml`

4. **"Traffic light ID not found"**
   - Verify traffic light ID matches network node
   - Check `sim/intersections/<name>/nodes.nod.xml`

5. **"Invalid phase state"**
   - Ensure state string length matches number of lanes
   - Use only valid characters: G, r, y, g

### Getting Help

- Check existing examples in `experiments/simple4/`
- Review JSON schema for field requirements
- Validate YAML syntax with online validators
- Test with simple scenarios first, then add complexity

## Additional Resources

- [SUMO Documentation](https://sumo.dlr.de/docs/index.html)
- [TraCI API](https://sumo.dlr.de/docs/TraCI.html)
- [SUMO Vehicle Types](https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html)
- [SUMO Traffic Lights](https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html)
