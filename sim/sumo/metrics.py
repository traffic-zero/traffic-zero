"""
Metrics calculation module for SUMO simulation evaluation.

Computes evaluation metrics including waiting times, travel times,
throughput, queue lengths, emissions, and other performance indicators.
"""

import pandas as pd
from typing import Dict, Optional
from pathlib import Path


class MetricsCalculator:
    """
    Calculates evaluation metrics from collected simulation data.
    
    Computes both per-step snapshots and end-of-simulation aggregates.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize metrics calculator.
        
        Args:
            output_dir: Directory to save metrics CSV (None = no file output)
        """
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_metrics(
        self,
        vehicle_df: pd.DataFrame,
        lane_df: pd.DataFrame,
        edge_df: pd.DataFrame,
        simulation_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics from collected data.
        
        Args:
            vehicle_df: DataFrame with vehicle data
            lane_df: DataFrame with lane data
            edge_df: DataFrame with edge data
            simulation_df: DataFrame with simulation state data
        
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        if not vehicle_df.empty:
            # Vehicle-based metrics
            metrics.update(self._calculate_vehicle_metrics(vehicle_df))
        
        if not lane_df.empty:
            # Lane-based metrics
            metrics.update(self._calculate_lane_metrics(lane_df))
        
        if not edge_df.empty:
            # Edge-based metrics
            metrics.update(self._calculate_edge_metrics(edge_df))
        
        if not simulation_df.empty:
            # Simulation-based metrics
            metrics.update(self._calculate_simulation_metrics(simulation_df))
        
        return metrics
    
    def _calculate_vehicle_metrics(self, vehicle_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate metrics from vehicle data."""
        metrics = {}
        
        # Average waiting time
        if 'waiting_time' in vehicle_df.columns:
            metrics['average_waiting_time'] = vehicle_df['waiting_time'].mean()
            metrics['max_waiting_time'] = vehicle_df['waiting_time'].max()
            metrics['total_waiting_time'] = vehicle_df['waiting_time'].sum()
        else:
            metrics['average_waiting_time'] = 0.0
            metrics['max_waiting_time'] = 0.0
            metrics['total_waiting_time'] = 0.0
        
        # Speed metrics
        if 'speed' in vehicle_df.columns:
            metrics['average_speed'] = vehicle_df['speed'].mean()
            metrics['max_speed'] = vehicle_df['speed'].max()
            metrics['min_speed'] = vehicle_df['speed'].min()
        else:
            metrics['average_speed'] = 0.0
            metrics['max_speed'] = 0.0
            metrics['min_speed'] = 0.0
        
        # Emissions
        if 'co2_emission' in vehicle_df.columns:
            metrics['total_co2_emission'] = vehicle_df['co2_emission'].sum()
            metrics['average_co2_emission'] = vehicle_df['co2_emission'].mean()
        else:
            metrics['total_co2_emission'] = 0.0
            metrics['average_co2_emission'] = 0.0
        
        if 'co_emission' in vehicle_df.columns:
            metrics['total_co_emission'] = vehicle_df['co_emission'].sum()
        else:
            metrics['total_co_emission'] = 0.0
        
        if 'nox_emission' in vehicle_df.columns:
            metrics['total_nox_emission'] = vehicle_df['nox_emission'].sum()
        else:
            metrics['total_nox_emission'] = 0.0
        
        # Fuel consumption
        if 'fuel_consumption' in vehicle_df.columns:
            metrics['total_fuel_consumption'] = vehicle_df['fuel_consumption'].sum()
            metrics['average_fuel_consumption'] = vehicle_df['fuel_consumption'].mean()
        else:
            metrics['total_fuel_consumption'] = 0.0
            metrics['average_fuel_consumption'] = 0.0
        
        # Number of unique vehicles
        if 'vehicle_id' in vehicle_df.columns:
            metrics['unique_vehicles'] = vehicle_df['vehicle_id'].nunique()
        else:
            metrics['unique_vehicles'] = 0
        
        return metrics
    
    def _calculate_lane_metrics(self, lane_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate metrics from lane data."""
        metrics = {}
        
        # Queue lengths
        if 'queue_length' in lane_df.columns:
            metrics['max_queue_length'] = lane_df['queue_length'].max()
            metrics['average_queue_length'] = lane_df['queue_length'].mean()
            metrics['total_queue_length'] = lane_df['queue_length'].sum()
        else:
            metrics['max_queue_length'] = 0.0
            metrics['average_queue_length'] = 0.0
            metrics['total_queue_length'] = 0.0
        
        # Lane waiting times
        if 'waiting_time' in lane_df.columns:
            metrics['max_lane_waiting_time'] = lane_df['waiting_time'].max()
            metrics['average_lane_waiting_time'] = lane_df['waiting_time'].mean()
        else:
            metrics['max_lane_waiting_time'] = 0.0
            metrics['average_lane_waiting_time'] = 0.0
        
        # Occupancy
        if 'occupancy' in lane_df.columns:
            metrics['max_lane_occupancy'] = lane_df['occupancy'].max()
            metrics['average_lane_occupancy'] = lane_df['occupancy'].mean()
        else:
            metrics['max_lane_occupancy'] = 0.0
            metrics['average_lane_occupancy'] = 0.0
        
        # Density
        if 'density' in lane_df.columns:
            metrics['max_lane_density'] = lane_df['density'].max()
            metrics['average_lane_density'] = lane_df['density'].mean()
        else:
            metrics['max_lane_density'] = 0.0
            metrics['average_lane_density'] = 0.0
        
        # Mean speeds
        if 'mean_speed' in lane_df.columns:
            metrics['average_lane_speed'] = lane_df['mean_speed'].mean()
            metrics['min_lane_speed'] = lane_df['mean_speed'].min()
        else:
            metrics['average_lane_speed'] = 0.0
            metrics['min_lane_speed'] = 0.0
        
        return metrics
    
    def _calculate_edge_metrics(self, edge_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate metrics from edge data."""
        metrics = {}
        
        # Travel time
        if 'travel_time' in edge_df.columns:
            metrics['average_travel_time'] = edge_df['travel_time'].mean()
            metrics['max_travel_time'] = edge_df['travel_time'].max()
            metrics['min_travel_time'] = edge_df['travel_time'].min()
        else:
            metrics['average_travel_time'] = 0.0
            metrics['max_travel_time'] = 0.0
            metrics['min_travel_time'] = 0.0
        
        # Edge speeds
        if 'mean_speed' in edge_df.columns:
            metrics['average_edge_speed'] = edge_df['mean_speed'].mean()
        else:
            metrics['average_edge_speed'] = 0.0
        
        # Edge occupancy
        if 'occupancy' in edge_df.columns:
            metrics['average_edge_occupancy'] = edge_df['occupancy'].mean()
            metrics['max_edge_occupancy'] = edge_df['occupancy'].max()
        else:
            metrics['average_edge_occupancy'] = 0.0
            metrics['max_edge_occupancy'] = 0.0
        
        return metrics
    
    def _calculate_simulation_metrics(
        self, simulation_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate metrics from simulation state data."""
        metrics = {}
        
        # Throughput (vehicles that arrived)
        if 'arrived_count' in simulation_df.columns:
            metrics['throughput'] = simulation_df['arrived_count'].max()
            metrics['total_departed'] = simulation_df['departed_count'].max()
        else:
            metrics['throughput'] = 0.0
            metrics['total_departed'] = 0.0
        
        # Simulation duration
        if 'time' in simulation_df.columns:
            metrics['simulation_duration'] = simulation_df['time'].max()
            metrics['simulation_start_time'] = simulation_df['time'].min()
        else:
            metrics['simulation_duration'] = 0.0
            metrics['simulation_start_time'] = 0.0
        
        # Average vehicle count
        if 'vehicle_count' in simulation_df.columns:
            metrics['average_vehicle_count'] = simulation_df['vehicle_count'].mean()
            metrics['max_vehicle_count'] = simulation_df['vehicle_count'].max()
        else:
            metrics['average_vehicle_count'] = 0.0
            metrics['max_vehicle_count'] = 0.0
        
        return metrics
    
    def export_metrics(self, metrics: Dict[str, float]):
        """
        Export metrics to CSV file.
        
        Args:
            metrics: Dictionary of metric names to values
        """
        if not self.output_dir:
            return
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame([metrics])
        
        # Export to CSV
        filepath = self.output_dir / 'metrics_summary.csv'
        metrics_df.to_csv(filepath, index=False)
        print(f">>> Exported metrics_summary.csv to {filepath}")
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metric names to values
        """
        print("\n" + "=" * 60)
        print("SIMULATION METRICS SUMMARY")
        print("=" * 60)
        
        # Primary metrics
        print("\n--- Primary Metrics ---")
        if 'average_waiting_time' in metrics:
            print(f"Average Waiting Time: {metrics['average_waiting_time']:.2f} s")
        if 'max_waiting_time' in metrics:
            print(f"Max Waiting Time: {metrics['max_waiting_time']:.2f} s")
        if 'average_travel_time' in metrics:
            print(f"Average Travel Time: {metrics['average_travel_time']:.2f} s")
        if 'throughput' in metrics:
            print(f"Throughput: {metrics['throughput']:.0f} vehicles")
        
        # Queue metrics
        print("\n--- Queue Metrics ---")
        if 'max_queue_length' in metrics:
            print(f"Max Queue Length: {metrics['max_queue_length']:.0f} vehicles")
        if 'average_queue_length' in metrics:
            print(f"Average Queue Length: {metrics['average_queue_length']:.2f} vehicles")
        
        # Speed metrics
        print("\n--- Speed Metrics ---")
        if 'average_speed' in metrics:
            print(f"Average Speed: {metrics['average_speed']:.2f} m/s")
        if 'average_lane_speed' in metrics:
            print(f"Average Lane Speed: {metrics['average_lane_speed']:.2f} m/s")
        
        # Emissions
        print("\n--- Emissions ---")
        if 'total_co2_emission' in metrics:
            print(f"Total CO2 Emission: {metrics['total_co2_emission']:.2f} mg")
        if 'total_fuel_consumption' in metrics:
            print(f"Total Fuel Consumption: {metrics['total_fuel_consumption']:.2f} ml")
        
        # Simulation info
        print("\n--- Simulation Info ---")
        if 'simulation_duration' in metrics:
            print(f"Simulation Duration: {metrics['simulation_duration']:.2f} s")
        if 'unique_vehicles' in metrics:
            print(f"Unique Vehicles: {metrics['unique_vehicles']:.0f}")
        
        print("=" * 60 + "\n")

