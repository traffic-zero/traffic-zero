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