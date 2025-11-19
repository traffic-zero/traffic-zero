#!/usr/bin/env python3
"""
Create an extreme rush hour scenario with maximum traffic.

This script creates a scenario where all lanes should be constantly filled
with vehicles, ensuring occupancy and density are always > 0 for main lanes.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any

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

def create_extreme_rush_hour_scenario(
    output_path: Path = None,
    vehicles_per_hour_per_route: float = 2000,
    flows_per_route: int = 50,
) -> Dict[str, Any]:
    """
    Create extreme rush hour scenario with maximum traffic.
    
    Args:
        output_path: Output file path
        vehicles_per_hour_per_route: Vehicles per hour per route (default: 2000)
        flows_per_route: Number of flows per route (default: 50)
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent / "experiments" / "simple4" / "rush_hour_random.yaml"
    
    flows = []
    
    # Create multiple flows for each route, all starting at 0
    for route in ROUTES:
        for i in range(flows_per_route):
            # Stagger begin times slightly to avoid simultaneous spawn
            begin = i * 0.1  # 0.1 second intervals
            flows.append({
                "route": route,
                "type": "car",
                "begin": begin,
                "end": 3600,
                "vehsPerHour": vehicles_per_hour_per_route,
            })
    
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
        description="Create extreme rush hour scenario with maximum traffic"
    )
    parser.add_argument(
        "--vehicles-per-hour",
        type=float,
        default=2000,
        help="Vehicles per hour per route (default: 2000)",
    )
    parser.add_argument(
        "--flows-per-route",
        type=int,
        default=50,
        help="Number of flows per route (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path",
    )
    
    args = parser.parse_args()
    
    # Generate scenario
    scenario, output_path = create_extreme_rush_hour_scenario(
        output_path=args.output,
        vehicles_per_hour_per_route=args.vehicles_per_hour,
        flows_per_route=args.flows_per_route,
    )
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(scenario, f, default_flow_style=False, sort_keys=False)
    
    total_flows = len(scenario['traffic']['flows'])
    total_routes = len(set(f['route'] for f in scenario['traffic']['flows']))
    total_vehs_per_hour = args.vehicles_per_hour * args.flows_per_route * total_routes
    
    print(f"[SUCCESS] Generated extreme rush hour scenario")
    print(f"[SUCCESS] Written to: {output_path}")
    print(f"[INFO] Total flows: {total_flows}")
    print(f"[INFO] Routes: {total_routes}")
    print(f"[INFO] Flows per route: {args.flows_per_route}")
    print(f"[INFO] Vehicles per hour per route: {args.vehicles_per_hour}")
    print(f"[INFO] Total vehicles per hour: {total_vehs_per_hour:.0f}")
    print(f"\n[TIP] This will create EXTREME congestion with vehicles constantly on all lanes!")
    print(f"[TIP] Main entry/exit lanes (eN, eS, eW, eE) should have high occupancy")
    print(f"[TIP] Junction lanes (:n0_*_0) may still show zeros when vehicles are crossing")

if __name__ == "__main__":
    main()



