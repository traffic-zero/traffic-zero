from argparse import ArgumentParser

from sim import run_automated, run_interactive


def generate_dataset(experiment_name: str):
    # Basic data collection for light_traffic_random scenario
    # Collects all TraCI data and exports to CSV files
    result = run_automated(
        simulation_name="simple4",
        experiment_name=experiment_name,
        # collect_interval=10,  # Collect data every 10th step
        output_dir=f"./data/{experiment_name}",
        enable_data_collection=True,
        max_steps=1000,  # Run for 1000 steps (adjust as needed)
    )

    # Access collected data
    if result:
        print("\n=== Data Collection Summary ===")
        print(f"Vehicle data points: {len(result['data']['vehicles'])}")
        print(
            "Traffic light data points: "
            f"{len(result['data']['traffic_lights'])}"
        )
        print("Lane data points: " f"{len(result['data']['lanes'])}")

        # Access metrics
        metrics = result["metrics"]
        print(
            "\nAverage waiting time: "
            f"{metrics.get('average_waiting_time', 0):.2f}s"
        )
        print(f"Max waiting time: {metrics.get('max_waiting_time', 0):.2f}s")
        print(f"Throughput: {metrics.get('throughput', 0):.0f} vehicles")

        # Access traffic light action log
        action_log = result["tls_controller"].get_action_log()
        print(f"\nTraffic light actions: {len(action_log)}")

        print(f"\n=== All data exported to: ./data/{experiment_name}/ ===")


def main():
    parser = ArgumentParser(description="Run SUMO scenarios in CARLA")
    parser.add_argument(
        "--generate-dataset",
        type=str,
        help="Experiment name",
    )

    args = parser.parse_args()
    if args.generate_dataset:
        print("Generating dataset...")
        generate_dataset(args.generate_dataset)

    # Call the runner for a specific simulation

    # Use run_interactive for manual control
    # and visual exploration (recommended)

    # Basic usage with default XML files:
    # run_interactive("simple4")

    # Use with experiment scenario (generates routes/tls from YAML):
    # run_interactive("simple4", experiment_name="light_traffic")
    # run_interactive("simple4", experiment_name="light_traffic_random")
    run_interactive("simple4", experiment_name="rush_hour")

    # Use run_automated for programmatic control
    # and experiments (runs for 30 minutes)

    # Basic usage:
    # run_automated("simple4")

    # Use with experiment scenario:
    # run_automated("simple4", experiment_name="light_traffic")
    # run_automated("simple4", experiment_name="rush_hour")

    # Use run_carla for CARLA co-simulation
    # run_carla("simple4")


if __name__ == "__main__":
    main()
