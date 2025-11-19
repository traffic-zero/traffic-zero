#!/usr/bin/env python3
"""Analyze lane_data.csv to understand occupancy and density values."""

import pandas as pd
import sys
from pathlib import Path

def analyze_lane_data(csv_path: Path):
    """Analyze lane data CSV file."""
    df = pd.read_csv(csv_path)
    
    print("=" * 60)
    print("LANE DATA ANALYSIS")
    print("=" * 60)
    print(f"\nTotal rows: {len(df)}")
    print(f"Unique steps: {df['step'].nunique()}")
    print(f"Unique lanes: {df['lane_id'].nunique()}")
    
    # Analyze vehicles
    rows_with_vehicles = df[df['vehicle_count'] > 0]
    print(f"\nRows WITH vehicles: {len(rows_with_vehicles)}")
    print(f"Rows WITHOUT vehicles: {len(df) - len(rows_with_vehicles)}")
    
    if len(rows_with_vehicles) > 0:
        print("\n" + "=" * 60)
        print("STATISTICS FOR ROWS WITH VEHICLES")
        print("=" * 60)
        print(f"Occupancy range: {rows_with_vehicles['occupancy'].min():.4f} - {rows_with_vehicles['occupancy'].max():.4f}")
        print(f"Density range: {rows_with_vehicles['density'].min():.4f} - {rows_with_vehicles['density'].max():.4f}")
        print(f"Vehicle count range: {rows_with_vehicles['vehicle_count'].min()} - {rows_with_vehicles['vehicle_count'].max()}")
        print(f"Mean speed range: {rows_with_vehicles['mean_speed'].min():.4f} - {rows_with_vehicles['mean_speed'].max():.4f}")
        print(f"Waiting time range: {rows_with_vehicles['waiting_time'].min():.4f} - {rows_with_vehicles['waiting_time'].max():.4f}")
        print(f"Queue length range: {rows_with_vehicles['queue_length'].min()} - {rows_with_vehicles['queue_length'].max()}")
        
        print("\n" + "=" * 60)
        print("SAMPLE ROWS WITH VEHICLES (first 20)")
        print("=" * 60)
        print(rows_with_vehicles[['step', 'time', 'lane_id', 'occupancy', 'density', 'vehicle_count', 'mean_speed']].head(20))
        
        # Check when vehicles first appear
        first_vehicle_step = rows_with_vehicles['step'].min()
        first_vehicle_time = rows_with_vehicles['time'].min()
        print(f"\nFirst vehicle appears at:")
        print(f"  Step: {first_vehicle_step}")
        print(f"  Time: {first_vehicle_time:.2f} seconds")
        
        # Check occupancy values
        print("\n" + "=" * 60)
        print("OCCUPANCY ANALYSIS")
        print("=" * 60)
        print(f"Occupancy values > 0: {len(rows_with_vehicles[rows_with_vehicles['occupancy'] > 0])}")
        print(f"Occupancy values > 1: {len(rows_with_vehicles[rows_with_vehicles['occupancy'] > 1])}")
        print(f"Occupancy values > 100: {len(rows_with_vehicles[rows_with_vehicles['occupancy'] > 100])}")
        print(f"\nMax occupancy: {rows_with_vehicles['occupancy'].max():.4f}")
        print(f"NOTE: SUMO occupancy is typically 0-100% (percentage)")
        print(f"If max is > 100, it might be in wrong units")
        print(f"If max is < 1, it might already be a fraction (0-1)")
        print(f"If max is 1-100, it's likely in percentage format")
        
        # Check lanes with highest occupancy
        print("\n" + "=" * 60)
        print("TOP 10 LANES BY MAX OCCUPANCY")
        print("=" * 60)
        lane_max_occupancy = rows_with_vehicles.groupby('lane_id')['occupancy'].max().sort_values(ascending=False)
        print(lane_max_occupancy.head(10))
        
    else:
        print("\n[WARNING] No vehicles found in the data!")
        print("This suggests:")
        print("1. Vehicles haven't spawned yet (check flow begin times)")
        print("2. Simulation ended before vehicles appeared")
        print("3. No vehicles were generated in the scenario")
    
    # Check early steps
    print("\n" + "=" * 60)
    print("EARLY STEPS ANALYSIS (steps 0-50)")
    print("=" * 60)
    early_steps = df[df['step'] <= 50]
    print(f"Rows in steps 0-50: {len(early_steps)}")
    print(f"Vehicles in steps 0-50: {len(early_steps[early_steps['vehicle_count'] > 0])}")
    if len(early_steps[early_steps['vehicle_count'] > 0]) == 0:
        print("[INFO] No vehicles in early steps - this is normal if flows start later")
        print(f"First vehicle step: {rows_with_vehicles['step'].min() if len(rows_with_vehicles) > 0 else 'N/A'}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = Path("data/light_traffic_random/lane_data.csv")
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    analyze_lane_data(csv_path)




