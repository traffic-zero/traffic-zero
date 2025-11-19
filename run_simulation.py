from argparse import ArgumentParser

from sim import run_automated


def generate_dataset():
    # Collects all TraCI data and exports to CSV files
    result = run_automated(
        simulation_name="simple4",
        experiment_name="light_traffic_random",
        # collect_interval=10,  # Collect data every 10th step
        output_dir="./data/light_traffic_random",
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

        print("\n=== All data exported to: ./data/light_traffic_random/ ===")


def main():
    parser = ArgumentParser(description="Run SUMO scenarios in CARLA")
    parser.add_argument(
        "--generate-dataset",
        action="store_true",
        help="Trigger dataset generation",
    )

    args = parser.parse_args()
    if args.generate_dataset:
        print("Generating dataset...")
        generate_dataset()
    else:
        print("Use --generate-dataset to generate datasets")

if __name__ == "__main__":
    main()
