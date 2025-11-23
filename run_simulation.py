from argparse import ArgumentParser

from sim import run_automated, run_carla, run_interactive


def generate_dataset(experiment_name: str):
    # Basic data collection for the specified experiment scenario
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "automated", "carla"],
        default="interactive",
        help="Mode to run the simulation in",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        choices=["light_traffic", "light_traffic_random", "rush_hour"],
        default="rush_hour",
        help="Experiment name",
    )

    args = parser.parse_args()
    if args.generate_dataset:
        print("Generating dataset...")
        generate_dataset(args.generate_dataset)
        return

    if args.mode == "interactive":
        run_interactive("simple4", args.experiment_name)
    elif args.mode == "automated":
        run_automated("simple4", args.experiment_name)
    elif args.mode == "carla":
        run_carla("simple4", args.experiment_name)


if __name__ == "__main__":
    main()
