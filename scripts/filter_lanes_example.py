#!/usr/bin/env python3
"""
Simple example of filtering lane data to
only show lanes with non-zero occupancy/density.

This script demonstrates how to filter
lane data to focus on lanes with actual traffic.
"""

import pandas as pd
from pathlib import Path


def filter_lanes_simple(csv_path: Path, output_path: Path | None = None):
    """
    Filter lane data to only show lanes with non-zero occupancy/density.

    Args:
        csv_path: Path to input CSV file
        output_path: Path to output CSV file (optional)
    """
    # Load data
    df = pd.read_csv(csv_path)

    print(f"Original data: {len(df)} rows")
    print(f"Unique lanes: {df['lane_id'].nunique()}")

    # Filter to only entry/exit lanes (starting with 'e')
    # These are the main traffic lanes, not junction lanes
    df_filtered = df[df["lane_id"].str.startswith("e")]
    print(
        "\nAfter filtering to entry/exit lanes (e*): "
        + str(len(df_filtered))
        + " rows"
    )

    # Filter to only lanes with non-zero occupancy/density
    df_non_zero = df_filtered[
        (df_filtered["occupancy"] > 0)
        | (df_filtered["density"] > 0)
        | (df_filtered["vehicle_count"] > 0)
    ].copy()
    print(
        "After filtering to non-zero occupancy/density: "
        + str(len(df_non_zero))
        + " rows"
    )

    # Show summary
    print("\nFiltered data summary:")
    print(f"  Unique lanes: {df_non_zero['lane_id'].nunique()}")
    print(f"  Unique steps: {df_non_zero['step'].nunique()}")
    print(
        "  Occupancy range: "
        + str(df_non_zero["occupancy"].min())
        + " - "
        + str(df_non_zero["occupancy"].max())
        + "%"
    )
    print(
        "  Density range: "
        + str(df_non_zero["density"].min())
        + " - "
        + str(df_non_zero["density"].max())
        + " veh/km"
    )
    print(
        "  Vehicle count range: "
        + str(df_non_zero["vehicle_count"].min())
        + " - "
        + str(df_non_zero["vehicle_count"].max())
    )

    # Show lanes
    lanes = df_non_zero["lane_id"].unique()
    print("\nLanes with traffic: " + str(sorted(lanes)))

    # Save to file if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_non_zero.to_csv(output_path, index=False)
        print("\nFiltered data saved to: " + str(output_path))

    return df_non_zero


def filter_main_roads_only(csv_path: Path, output_path: Path | None = None):
    """
    Filter to only main roads (eN_0, eS_0, eW_0, eE_0, eN_out_0, eS_out_0,
    eW_out_0, eE_out_0).

    Args:
        csv_path: Path to input CSV file
        output_path: Path to output CSV file (optional)
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Main roads only
    main_lanes = [
        "eN_0",
        "eS_0",
        "eW_0",
        "eE_0",
        "eN_out_0",
        "eS_out_0",
        "eW_out_0",
        "eE_out_0",
    ]
    df_main = df[df["lane_id"].isin(main_lanes)]

    # Filter to non-zero occupancy/density
    df_main_non_zero = df_main[
        (df_main["occupancy"] > 0)
        | (df_main["density"] > 0)
        | (df_main["vehicle_count"] > 0)
    ]

    print("Main roads with traffic: " + str(len(df_main_non_zero)) + " rows")
    print(f"Unique lanes: {df_main_non_zero['lane_id'].nunique()}")

    # Save to file if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_main_non_zero.to_csv(output_path, index=False)
        print("Filtered data saved to: " + str(output_path))

    return df_main_non_zero


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python filter_lanes_example.py <input_csv> [output_csv]")
        print(
            "Example: python filter_lanes_example.py "
            "data/rush_hour_random/lane_data.csv"
        )
        sys.exit(1)

    input_csv = Path(sys.argv[1])
    output_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not input_csv.exists():
        print(f"Error: File not found: {input_csv}")
        sys.exit(1)

    # Filter to entry/exit lanes with non-zero occupancy/density
    print("=" * 60)
    print("FILTERING TO ENTRY/EXIT LANES WITH NON-ZERO OCCUPANCY/DENSITY")
    print("=" * 60)
    df_filtered = filter_lanes_simple(input_csv, output_csv)

    # Also filter to main roads only
    if output_csv:
        main_roads_output = output_csv.parent / (
            output_csv.stem + "_main_roads.csv"
        )
        print("\n" + "=" * 60)
        print("FILTERING TO MAIN ROADS ONLY")
        print("=" * 60)
        df_main = filter_main_roads_only(input_csv, main_roads_output)
