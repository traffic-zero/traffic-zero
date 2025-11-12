"""
Command-line interface for CARLA-SUMO co-simulation.

Usage:
    python -m sim.carla simple4 --duration 120
"""

if __name__ == "__main__":
    import argparse
    from .runner import run_in_carla, list_simulations

    parser = argparse.ArgumentParser(description="Run SUMO scenarios in CARLA")
    parser.add_argument(
        "simulation", nargs="?", help="Simulation name (e.g., simple4)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Simulation duration in seconds",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available simulations"
    )
    parser.add_argument(
        "--no-sumo-network",
        action="store_true",
        help="Use CARLA town map instead of SUMO network",
    )

    args = parser.parse_args()

    if args.list:
        list_simulations()
    elif args.simulation:
        run_in_carla(
            args.simulation,
            args.duration,
            use_sumo_network=not args.no_sumo_network,
        )
    else:
        parser.print_help()
