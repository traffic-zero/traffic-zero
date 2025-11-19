#!/usr/bin/env python3
"""
Filter lane_data.csv to keep only the main entry/exit lanes.

This script automatically discovers main lanes from SUMO network configuration files,
making it work with any intersection without hardcoding lane names.
"""

import pandas as pd
import argparse
from pathlib import Path
import sys

# Add project root to path to import from sim.sumo
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sim.sumo.lane_utils import get_main_lanes_for_intersection


def filter_lane_data_to_main_lanes(
    input_csv: Path,
    output_csv: Path = None,
    intersection_name: str = "simple4",
    backup: bool = True,
    intersection_base_dir: Path = None,
) -> pd.DataFrame:
    """
    Filter lane_data.csv to keep only main entry/exit lanes.
    
    Main lanes are automatically discovered from SUMO network configuration files,
    ensuring compatibility with any intersection.
    
    Args:
        input_csv: Input CSV file path
        output_csv: Output CSV file path (if None, overwrites input)
        intersection_name: Name of intersection (default: 'simple4')
        backup: If True, create backup of original file
        intersection_base_dir: Base directory for intersections (default: sim/intersections)
    
    Returns:
        Filtered DataFrame
    """
    # Load data
    print(f"[INFO] Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    original_rows = len(df)
    original_lanes = df['lane_id'].nunique()
    
    print(f"[INFO] Original data: {original_rows:,} rows, {original_lanes} unique lanes")
    
    # Discover main lanes from SUMO network files
    print(f"[INFO] Discovering main lanes for intersection: {intersection_name}")
    try:
        main_lanes = get_main_lanes_for_intersection(intersection_name, intersection_base_dir)
        print(f"[INFO] Discovered {len(main_lanes)} main lanes: {sorted(main_lanes)}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[ERROR] Unable to discover lanes. Make sure:")
        print(f"  1. Intersection '{intersection_name}' exists in sim/intersections/")
        print("  2. network.net.xml has been generated (run netconvert or generate_sumocfg)")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to discover lanes: {e}")
        raise
    
    # Filter to main lanes only
    df_filtered = df[df['lane_id'].isin(main_lanes)].copy()
    
    filtered_rows = len(df_filtered)
    filtered_lanes = df_filtered['lane_id'].nunique()
    
    print(f"[INFO] Filtered data: {filtered_rows:,} rows, {filtered_lanes} unique lanes")
    print(f"[INFO] Removed: {original_rows - filtered_rows:,} rows ({100*(original_rows-filtered_rows)/original_rows:.1f}%)")
    
    # Show lane breakdown
    print("\n" + "=" * 60)
    print("LANE BREAKDOWN")
    print("=" * 60)
    for lane in sorted(main_lanes):
        lane_data = df_filtered[df_filtered['lane_id'] == lane]
        if len(lane_data) > 0:
            with_vehicles = lane_data[lane_data['vehicle_count'] > 0]
            print(f"  {lane}: {len(lane_data):,} rows, "
                  f"{len(with_vehicles):,} with vehicles ({100*len(with_vehicles)/len(lane_data):.1f}%)")
    
    # Create backup if requested
    if backup and output_csv is None:
        backup_path = input_csv.parent / f"{input_csv.stem}_backup.csv"
        print(f"\n[INFO] Creating backup: {backup_path}")
        df.to_csv(backup_path, index=False)
    
    # Determine output path
    if output_csv is None:
        output_csv = input_csv
    
    # Save filtered data
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Filtered data saved to: {output_csv}")
    
    return df_filtered


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Filter lane_data.csv to keep only main entry/exit lanes"
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
        help="Output CSV file path (default: overwrites input)",
    )
    parser.add_argument(
        "--intersection",
        type=str,
        default="simple4",
        help="Intersection name (default: simple4). Main lanes are auto-discovered from SUMO network files.",
    )
    parser.add_argument(
        "--intersection-dir",
        type=Path,
        default=None,
        help="Base directory for intersections (default: sim/intersections)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of original file",
    )
    
    args = parser.parse_args()
    
    if not args.input_csv.exists():
        print(f"[ERROR] File not found: {args.input_csv}")
        return 1
    
    # Filter data
    try:
        df_filtered = filter_lane_data_to_main_lanes(
            input_csv=args.input_csv,
            output_csv=args.output,
            intersection_name=args.intersection,
            backup=not args.no_backup,
            intersection_base_dir=args.intersection_dir,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

