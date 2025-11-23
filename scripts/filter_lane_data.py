#!/usr/bin/env python3
"""
Filter lane data to only include lanes with non-zero occupancy and density.

This script filters out junction lanes (which are mostly empty) and focuses
on entry/exit lanes that show actual traffic patterns.
"""

import pandas as pd
import argparse
from pathlib import Path


def filter_lane_data(
    input_csv: Path,
    output_csv: Path | None = None,
    filter_type: str = "non_zero",
    lane_type: str | None = None,
    min_occupancy: float = 0.0,
    min_density: float = 0.0,
) -> pd.DataFrame:
    """
    Filter lane data based on various criteria.

    Args:
        input_csv: Input CSV file path
        output_csv: Output CSV file path (optional)
        filter_type: Type of filter ('non_zero', 'entry_exit', 'junction',
                                     'all')
        lane_type: Filter by lane type ('entry_exit' for e*, 'junction' for :n*)
        min_occupancy: Minimum occupancy threshold (default: 0.0)
        min_density: Minimum density threshold (default: 0.0)

    Returns:
        Filtered DataFrame
    """
    df = pd.read_csv(input_csv)

    print(f"[INFO] Original data: {len(df)} rows")
    print(f"[INFO] Unique lanes: {df['lane_id'].nunique()}")

    # Apply lane type filter
    if lane_type == "entry_exit":
        # Only entry/exit lanes (starting with 'e')
        df = df[df["lane_id"].str.startswith("e")]
        print(f"[INFO] After filtering to entry/exit lanes: {len(df)} rows")
    elif lane_type == "junction":
        # Only junction lanes (starting with ':n')
        df = df[df["lane_id"].str.startswith(":n")]
        print(f"[INFO] After filtering to junction lanes: {len(df)} rows")
    elif lane_type == "main_roads":
        # Only main entry/exit lanes (eN, eS, eW, eE, eN_out, eS_out,
        # eW_out, eE_out)
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
        df = df[df["lane_id"].isin(main_lanes)]
        print(f"[INFO] After filtering to main roads: {len(df)} rows")

    # Apply occupancy/density filters
    if filter_type == "non_zero":
        # Only rows where occupancy > 0 OR density > 0 OR vehicle_count > 0
        df = df[
            (df["occupancy"] > 0)
            | (df["density"] > 0)
            | (df["vehicle_count"] > 0)
        ]
        print(
            "[INFO] After filtering to non-zero occupancy/density: "
            + str(len(df))
            + " rows"
        )
    elif filter_type == "has_vehicles":
        # Only rows where vehicle_count > 0
        df = df[df["vehicle_count"] > 0]
        print(f"[INFO] After filtering to rows with vehicles: {len(df)} rows")
    elif filter_type == "min_threshold":
        # Only rows where occupancy >= min_occupancy AND density >= min_density
        df = df[
            (df["occupancy"] >= min_occupancy) & (df["density"] >= min_density)
        ]
        print(
            "[INFO] After filtering to min thresholds (occ>="
            + str(min_occupancy)
            + ", dens>="
            + str(min_density)
            + "): "
            + str(len(df))
            + " rows"
        )

    # Additional filters
    if min_occupancy > 0:
        df = df[df["occupancy"] >= min_occupancy]
        print(
            "[INFO] After min occupancy filter (>= "
            + str(min_occupancy)
            + "): "
            + str(len(df))
            + " rows"
        )

    if min_density > 0:
        df = df[df["density"] >= min_density]
        print(
            "[INFO] After min density filter (>= "
            + str(min_density)
            + "): "
            + str(len(df))
            + " rows"
        )

    # Save to file if output path provided
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[SUCCESS] Filtered data saved to: {output_csv}")
        print(
            "[INFO] Filtered data: "
            + str(len(df))
            + " rows ("
            + str(len(df) / len(pd.read_csv(input_csv)) * 100)
            + "% of original)"
        )

    return df


def _print_lane_type_stats(
    df_filtered: pd.DataFrame, lane_type_name: str
) -> None:
    """Print statistics for a filtered lane type DataFrame."""
    if len(df_filtered) == 0:
        return

    print("\n" + lane_type_name + " (" + str(len(df_filtered)) + " rows):")
    print(f"  Unique lanes: {df_filtered['lane_id'].nunique()}")

    with_vehicles = df_filtered[df_filtered["vehicle_count"] > 0]
    print(f"  Rows with vehicles: {len(with_vehicles)}")
    print(
        "  Percentage with vehicles: "
        + str(len(with_vehicles) / len(df_filtered) * 100)
        + "%"
    )

    with_occupancy = df_filtered[df_filtered["occupancy"] > 0]
    if len(with_occupancy) > 0:
        print(
            "  Occupancy range: "
            + str(with_occupancy["occupancy"].min())
            + " - "
            + str(with_occupancy["occupancy"].max())
            + "%"
        )
        with_density = df_filtered[df_filtered["density"] > 0]
        if len(with_density) > 0:
            print(
                "  Density range: "
                + str(with_density["density"].min())
                + " - "
                + str(with_density["density"].max())
                + " veh/km"
            )


def _print_main_road_breakdown(
    main_roads: pd.DataFrame, main_lanes: list[str]
) -> None:
    """Print detailed breakdown for main roads."""
    print("\n  Lane breakdown:")
    for lane in main_lanes:
        lane_data = main_roads[main_roads["lane_id"] == lane]
        if len(lane_data) == 0:
            continue

        with_vehicles = lane_data[lane_data["vehicle_count"] > 0]
        if len(with_vehicles) == 0:
            continue

        print(
            "    "
            + lane
            + ": "
            + str(len(with_vehicles))
            + "/"
            + str(len(lane_data))
            + " steps with vehicles ("
            + str(len(with_vehicles) / len(lane_data) * 100)
            + "%)"
        )
        print(f"      Avg occupancy: {with_vehicles['occupancy'].mean():.2f}%")
        print(
            "      Avg density: "
            + str(with_vehicles["density"].mean())
            + " veh/km"
        )


def analyze_lane_types(df: pd.DataFrame):
    """Analyze different lane types in the data."""
    print("\n" + "=" * 60)
    print("LANE TYPE ANALYSIS")
    print("=" * 60)

    junction = df[df["lane_id"].str.startswith(":n")]
    _print_lane_type_stats(junction, "Junction lanes")

    entry_exit = df[df["lane_id"].str.startswith("e")]
    _print_lane_type_stats(entry_exit, "Entry/exit lanes")

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
    main_roads = df[df["lane_id"].isin(main_lanes)]
    if len(main_roads) > 0:
        _print_lane_type_stats(main_roads, "Main roads")
        with_occupancy = main_roads[main_roads["occupancy"] > 0]
        if len(with_occupancy) > 0:
            _print_main_road_breakdown(main_roads, main_lanes)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Filter lane data to only include lanes with non-zero "
            "occupancy and density"
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV file path (optional)",
    )
    parser.add_argument(
        "--filter-type",
        choices=["non_zero", "has_vehicles", "min_threshold", "all"],
        default="non_zero",
        help="Type of filter to apply (default: non_zero)",
    )
    parser.add_argument(
        "--lane-type",
        choices=["entry_exit", "junction", "main_roads", "all"],
        default="all",
        help="Filter by lane type (default: all)",
    )
    parser.add_argument(
        "--min-occupancy",
        type=float,
        default=0.0,
        help="Minimum occupancy threshold (default: 0.0)",
    )
    parser.add_argument(
        "--min-density",
        type=float,
        default=0.0,
        help="Minimum density threshold (default: 0.0)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Show detailed analysis of lane types",
    )

    args = parser.parse_args()

    if not args.input_csv.exists():
        print(f"[ERROR] File not found: {args.input_csv}")
        return 1

    # Load and analyze original data
    if args.analyze:
        df_original = pd.read_csv(args.input_csv)
        analyze_lane_types(df_original)
        print("\n" + "=" * 60)

    # Filter data
    df_filtered = filter_lane_data(
        input_csv=args.input_csv,
        output_csv=args.output,
        filter_type=args.filter_type,
        lane_type=args.lane_type if args.lane_type != "all" else None,
        min_occupancy=args.min_occupancy,
        min_density=args.min_density,
    )

    # Show summary
    print("\n" + "=" * 60)
    print("FILTERED DATA SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(df_filtered)}")
    print(f"Unique lanes: {df_filtered['lane_id'].nunique()}")
    print(f"Unique steps: {df_filtered['step'].nunique()}")
    if len(df_filtered) > 0:
        print(
            "Occupancy range: "
            + str(df_filtered["occupancy"].min())
            + " - "
            + str(df_filtered["occupancy"].max())
            + "%"
        )
        print(
            "Density range: "
            + str(df_filtered["density"].min())
            + " - "
            + str(df_filtered["density"].max())
            + " veh/km"
        )
        print(
            "Vehicle count range: "
            + str(df_filtered["vehicle_count"].min())
            + " - "
            + str(df_filtered["vehicle_count"].max())
        )

    # Show top lanes by occupancy
    if len(df_filtered) > 0:
        print("\n" + "=" * 60)
        print("TOP 10 LANES BY MAX OCCUPANCY")
        print("=" * 60)
        lane_max_occupancy = (
            df_filtered.groupby("lane_id")["occupancy"]
            .max()
            .sort_values(ascending=False)
        )
        for lane_id, max_occ in lane_max_occupancy.head(10).items():
            lane_data = df_filtered[df_filtered["lane_id"] == lane_id]
            avg_occ = (
                lane_data[lane_data["occupancy"] > 0]["occupancy"].mean()
                if len(lane_data[lane_data["occupancy"] > 0]) > 0
                else 0
            )
            print(f"  {lane_id}: max={max_occ:.2f}%, avg={avg_occ:.2f}%")

    return 0


if __name__ == "__main__":
    exit(main())
