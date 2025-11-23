#!/usr/bin/env python3
"""
Generate random traffic flows for rush hour scenarios.

This script generates a large number of random flows with varying:
- Routes (all 8 routes)
- Begin times (randomly distributed)
- End times (3600 seconds or random intervals)
- Probabilities (0.15 to 0.35 for high congestion)
"""

import random
import yaml
from pathlib import Path
from typing import Any

# Available routes
ROUTES = [
    "north_south",
    "south_north",
    "west_east",
    "east_west",
    "north_east",
    "south_west",
    "west_north",
    "east_south",
]

# Simulation parameters
SIMULATION_DURATION = 3600  # 1 hour in seconds
MIN_PROBABILITY = 0.50  # Increased for rush hour - 50% chance per second
MAX_PROBABILITY = 0.90  # Increased for rush hour - 90% chance per second
MIN_BEGIN = 0
MAX_BEGIN = 10  # Start all flows within first 10 seconds for immediate rush


def generate_random_flow(
    route: str,
    begin: float | None = None,
    end: float | None = None,
    probability: float | None = None,
    vehs_per_hour: float | None = None,
    use_vehs_per_hour: bool = True,
) -> dict[str, Any]:
    """
    Generate a single random flow.

    Args:
        route: Route ID
        begin: Start time (seconds)
        end: End time (seconds)
        probability: Probability per second (0-1)
        vehs_per_hour: Vehicles per hour (for consistent spawning)
        use_vehs_per_hour: If True, use vehsPerHour instead of probability
    """
    if begin is None:
        begin = random.uniform(MIN_BEGIN, MAX_BEGIN)
    if end is None:
        # Most flows run full duration for rush hour
        if random.random() < 0.9:
            end = SIMULATION_DURATION
        else:
            end = random.uniform(begin + 100, SIMULATION_DURATION)

    flow = {
        "route": route,
        "type": "car",
        "begin": round(begin, 2),
        "end": round(end, 2),
    }

    if use_vehs_per_hour:
        # Use vehsPerHour for more consistent vehicle spawning
        # Rush hour: 500-2000 vehicles per hour per flow (very high traffic)
        if vehs_per_hour is None:
            vehs_per_hour = random.uniform(500, 2000)
        flow["vehsPerHour"] = round(vehs_per_hour, 2)
    else:
        # Use probability (less consistent)
        if probability is None:
            probability = random.uniform(MIN_PROBABILITY, MAX_PROBABILITY)
        flow["probability"] = round(probability, 4)

    return flow


def generate_flows(
    num_flows: int = 200,
    seed: int | None = None,
    use_vehs_per_hour: bool = True,
    min_vehs_per_hour: float = 100,
    max_vehs_per_hour: float = 600,
) -> list[dict[str, Any]]:
    """
    Generate multiple random flows.

    Args:
        num_flows: Number of flows to generate
        seed: Random seed for reproducibility
        use_vehs_per_hour: If True, use vehsPerHour (more consistent)
        min_vehs_per_hour: Minimum vehicles per hour
        max_vehs_per_hour: Maximum vehicles per hour
    """
    if seed is not None:
        random.seed(seed)

    flows = []
    for _ in range(num_flows):
        route = random.choice(ROUTES)
        if use_vehs_per_hour:
            vehs_per_hour = random.uniform(min_vehs_per_hour, max_vehs_per_hour)
            flow = generate_random_flow(
                route, use_vehs_per_hour=True, vehs_per_hour=vehs_per_hour
            )
        else:
            flow = generate_random_flow(route, use_vehs_per_hour=False)
        flows.append(flow)

    # Sort flows by begin time for better readability
    flows.sort(key=lambda x: (x["begin"], x["route"]))

    return flows


def create_rush_hour_scenario(
    num_flows: int = 200,
    output_path: Path | None = None,
    seed: int | None = None,
    use_vehs_per_hour: bool = True,
    min_vehs_per_hour: float = 500,
    max_vehs_per_hour: float = 2000,
) -> tuple[dict[str, Any], Path]:
    """
    Create a complete rush hour scenario with random flows.

    Args:
        num_flows: Number of flows to generate
        output_path: Output file path
        seed: Random seed
        use_vehs_per_hour: Use vehsPerHour for consistent spawning (recommended)
        min_vehs_per_hour: Minimum vehicles per hour per flow
        max_vehs_per_hour: Maximum vehicles per hour per flow
    """
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent
            / "experiments"
            / "simple4"
            / "rush_hour_random.yaml"
        )

    flows = generate_flows(
        num_flows,
        seed,
        use_vehs_per_hour=use_vehs_per_hour,
        min_vehs_per_hour=min_vehs_per_hour,
        max_vehs_per_hour=max_vehs_per_hour,
    )

    scenario = {
        "intersection": "simple4",
        "traffic": {
            "vehicle_types": [
                {
                    "id": "car",
                    "accel": 2.0,
                    "decel": 4.5,
                    "sigma": 0.5,
                    "length": 5,
                    "maxSpeed": 13.9,
                    "color": "0,1,1",  # Cyan
                }
            ],
            "routes": [
                {"id": "north_south", "edges": ["eN", "eS_out"]},
                {"id": "south_north", "edges": ["eS", "eN_out"]},
                {"id": "west_east", "edges": ["eW", "eE_out"]},
                {"id": "east_west", "edges": ["eE", "eW_out"]},
                {"id": "north_east", "edges": ["eN", "eE_out"]},
                {"id": "south_west", "edges": ["eS", "eW_out"]},
                {"id": "west_north", "edges": ["eW", "eN_out"]},
                {"id": "east_south", "edges": ["eE", "eS_out"]},
            ],
            "flows": flows,
        },
        "traffic_lights": {
            "programs": [
                {
                    "id": "n0",
                    "program_id": "custom",
                    "offset": 0,
                    "type": "static",
                    "phases": [
                        {"duration": 20, "state": "GGGGrrrrGGGGrrrr"},
                        {"duration": 4, "state": "yyyyrrrryyyyrrrr"},
                        {"duration": 20, "state": "rrrrGGGGrrrrGGGG"},
                        {"duration": 4, "state": "rrrryyyyrrrryyyy"},
                    ],
                }
            ]
        },
    }

    return scenario, output_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate random traffic flows for rush hour scenarios"
    )
    parser.add_argument(
        "--num-flows",
        type=int,
        default=200,
        help="Number of flows to generate (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output file path "
            "(default: experiments/simple4/rush_hour_random.yaml)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use-vehs-per-hour",
        action="store_true",
        default=True,
        help=(
            "Use vehsPerHour instead of probability "
            "(more consistent, default: True)"
        ),
    )
    parser.add_argument(
        "--min-vehs-per-hour",
        type=float,
        default=500,
        help="Minimum vehicles per hour (default: 500 for rush hour)",
    )
    parser.add_argument(
        "--max-vehs-per-hour",
        type=float,
        default=2000,
        help="Maximum vehicles per hour (default: 2000 for extreme rush hour)",
    )
    parser.add_argument(
        "--max-probability",
        type=float,
        default=0.90,
        help=(
            "Maximum probability for flows "
            "(if not using vehsPerHour, default: 0.90)"
        ),
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.50,
        help=(
            "Minimum probability for flows "
            "(if not using vehsPerHour, default: 0.50)"
        ),
    )

    args = parser.parse_args()

    # Update global constants
    global MIN_PROBABILITY, MAX_PROBABILITY
    MIN_PROBABILITY = args.min_probability
    MAX_PROBABILITY = args.max_probability

    # Generate scenario
    scenario, output_path = create_rush_hour_scenario(
        num_flows=args.num_flows,
        output_path=args.output,
        seed=args.seed,
        use_vehs_per_hour=args.use_vehs_per_hour,
        min_vehs_per_hour=args.min_vehs_per_hour,
        max_vehs_per_hour=args.max_vehs_per_hour,
    )

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(scenario, f, default_flow_style=False, sort_keys=False)

    print(f"[SUCCESS] Generated {args.num_flows} random flows")
    print(f"[SUCCESS] Written to: {output_path}")
    print(
        "[INFO] Routes used: "
        + str(len({f["route"] for f in scenario["traffic"]["flows"]}))
    )

    if args.use_vehs_per_hour:
        vehs_per_hour_values = [
            f.get("vehsPerHour", 0)
            for f in scenario["traffic"]["flows"]
            if f.get("vehsPerHour")
        ]
        if vehs_per_hour_values:
            print(
                "[INFO] Vehicles per hour range: "
                + str(min(vehs_per_hour_values))
                + " - "
                + str(max(vehs_per_hour_values))
            )
            print(
                "[INFO] Average vehicles per hour: "
                + str(sum(vehs_per_hour_values) / len(vehs_per_hour_values))
            )
    else:
        prob_values = [
            f.get("probability", 0)
            for f in scenario["traffic"]["flows"]
            if f.get("probability")
        ]
        if prob_values:
            print(
                "[INFO] Probability range: "
                + str(min(prob_values))
                + " - "
                + str(max(prob_values))
            )

    print(f"[INFO] Simulation duration: {SIMULATION_DURATION}s (1 hour)\n")
    print("[TIP] This scenario will create MASSIVE congestion and overflow!")
    print(
        "[TIP] Adjust --num-flows and --max-vehs-per-hour "
        "to control traffic volume"
    )
    print(
        "[TIP] Using vehsPerHour provides more consistent vehicle spawning "
        "than probability"
    )


if __name__ == "__main__":
    main()
