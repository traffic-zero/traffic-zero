# Simulations

## Getting started

1. Finish project setup ["Getting Started"](../README.md)
2. Launch `sumo-gui` for any simulation:
    ```py
    from sim import run_intersection

    # Call the runner for a specific simulation
    run_intersection("simple4")
    ```
3. Choose the name of simulation you want to run from "List of Simulations"

## Run in CARLA 3D Environment (Optional)

Want to see your simulations in 3D? Run them in CARLA simulator!

```python
from sim import run_in_carla

# Run simple4 in CARLA's 3D environment
run_in_carla("simple4", duration=120)
```

**Setup required**: See [CARLA.md](./CARLA.md) for quick setup instructions.

## List of Simulations

1. `simple4`

![](./intersections/simple4/preview.png)

A 4-way intersection with fixed traffic lights.

Custom traffic light programs
TraCI-based control (e.g., switching between programs via traci.trafficlight.setProgram)
Basic vehicle flow and route generation

---

Coming Soon 🚧
We plan to add more scenarios, including:

- t-junction - T-shaped intersection
- complex - Grid intersection with multiple traffic sources