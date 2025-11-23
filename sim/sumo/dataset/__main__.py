"""
Command-line interface for SUMO dataset generation with realism modes.

Usage:
    python -m sim.sumo.dataset --realism [none/low/med/high]
    [simulation_name] [experiment_name]
"""

import argparse
import sys
from pathlib import Path

# Import directly from parent module to avoid package conflict
import importlib.util

_dataset_module_path = Path(__file__).parent.parent / "dataset.py"
spec = importlib.util.spec_from_file_location(
    "sumo_dataset_module", _dataset_module_path
)
_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_dataset_module)

get_realism_level = _dataset_module.get_realism_level
RealismMode = _dataset_module.RealismMode

# Add parent directory to path to import from sim.sumo
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from sim.sumo.runner import run_automated  # noqa: E402


def main():
    """Main CLI entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate SUMO simulation datasets with realism/noise modes"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m sim.sumo.dataset --realism low simple4
  python -m sim.sumo.dataset --realism med simple4 rush_hour
  python -m sim.sumo.dataset --realism high simple4 --output ./data/output

Realism levels:
  none  - No noise, perfect sensor readings (default)
  low   - Minor jitter, occasional missing data (1% failures)
  med   - Moderate noise, some sensor failures, errors (5% failures)
  high  - High noise, frequent failures, significant errors (15% failures)
        """,
    )

    parser.add_argument(
        "--realism",
        type=str,
        default="none",
        choices=["none", "low", "med", "high"],
        help="Realism/noise level for sensor data (default: none)",
    )

    parser.add_argument(
        "simulation_name",
        type=str,
        nargs="?",
        default="simple4",
        help="Name of simulation to run (default: simple4)",
    )

    parser.add_argument(
        "experiment_name",
        type=str,
        nargs="?",
        default=None,
        help="Optional experiment scenario name",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output directory for CSV files "
            "(default: data/<experiment_name> or data/<simulation_name>)"
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=36000,
        help="Maximum simulation steps (default: 36000 = 30 minutes)",
    )

    parser.add_argument(
        "--collect-interval",
        type=int,
        default=1,
        help="Collect data every N steps (default: 1)",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show SUMO-GUI window (default: headless)",
    )

    args = parser.parse_args()

    # Parse realism level
    try:
        realism_level = get_realism_level(args.realism)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    output_dir = args.output
    if output_dir is None:
        if args.experiment_name:
            output_dir = f"data/{args.experiment_name}"
        else:
            output_dir = f"data/{args.simulation_name}"

    print(">>> Running SUMO dataset generation")
    print(f">>> Simulation: {args.simulation_name}")
    if args.experiment_name:
        print(f">>> Experiment: {args.experiment_name}")
    print(f">>> Realism level: {realism_level.value}")
    print(f">>> Output directory: {output_dir}")
    print(f">>> Max steps: {args.max_steps}")
    print()

    # Create realism mode
    realism_mode = RealismMode(realism_level)

    # Run simulation with realism mode
    try:
        run_automated(
            simulation_name=args.simulation_name,
            experiment_name=args.experiment_name,
            collect_interval=args.collect_interval,
            output_dir=output_dir,
            enable_data_collection=True,
            max_steps=args.max_steps,
            gui=args.gui,
            realism_mode=realism_mode,
        )

        print()
        print(">>> Dataset generation complete!")
        print(f">>> Output files saved to: {output_dir}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
