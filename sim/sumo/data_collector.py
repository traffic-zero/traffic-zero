"""
Data collection module for SUMO TraCI.

Collects comprehensive data from SUMO simulation including vehicles,
traffic lights, lanes, junctions, edges, and simulation state.
"""

import traci
import pandas as pd
from typing import Dict, Optional
from pathlib import Path


class DataCollector:
    """
    Collects TraCI data from SUMO simulation at configurable intervals.
    
    Stores data in pandas DataFrames (in-memory) and exports to CSV files.
    """
    
    def __init__(self, collect_interval: int = 1, output_dir: Optional[str] = None):
        """
        Initialize data collector.
        
        Args:
            collect_interval: Collect data every N simulation steps (default: 1)
            output_dir: Directory to save CSV files (None = no file output)
        """
        self.collect_interval = collect_interval
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage (in-memory DataFrames)
        self.vehicle_data = []
        self.traffic_light_data = []
        self.lane_data = []
        self.junction_data = []
        self.edge_data = []
        self.simulation_data = []
        
        self.current_step = 0
    
    def should_collect(self, step: int) -> bool:
        """Check if data should be collected at this step."""
        return step % self.collect_interval == 0
    
    def collect_step(self, step: int):
        """
        Collect all available TraCI data at current simulation step.
        
        Args:
            step: Current simulation step number
        """
        if not self.should_collect(step):
            return
        
        self.current_step = step
        current_time = traci.simulation.getTime()
        
        # Collect vehicle data
        self._collect_vehicle_data(step, current_time)
        
        # Collect traffic light data
        self._collect_traffic_light_data(step, current_time)
        
        # Collect lane data
        self._collect_lane_data(step, current_time)
        
        # Collect junction data
        self._collect_junction_data(step, current_time)
        
        # Collect edge data
        self._collect_edge_data(step, current_time)
        
        # Collect simulation data
        self._collect_simulation_data(step, current_time)
    
    def _collect_vehicle_data(self, step: int, time: float):
        """Collect data for all vehicles in simulation."""
        vehicle_ids = traci.vehicle.getIDList()
        
        for veh_id in vehicle_ids:
            try:
                pos = traci.vehicle.getPosition(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                angle = traci.vehicle.getAngle(veh_id)
                waiting_time = traci.vehicle.getWaitingTime(veh_id)
                lane_id = traci.vehicle.getLaneID(veh_id)
                lane_pos = traci.vehicle.getLanePosition(veh_id)
                
                # Try to get route
                try:
                    route = traci.vehicle.getRoute(veh_id)
                    route_str = ','.join(route) if route else None
                except traci.TraCIException:
                    route_str = None
                
                # Try to get acceleration
                try:
                    accel = traci.vehicle.getAcceleration(veh_id)
                except traci.TraCIException:
                    accel = None
                
                # Try to get emissions
                try:
                    co2 = traci.vehicle.getCO2Emission(veh_id)
                except traci.TraCIException:
                    co2 = None
                
                try:
                    co = traci.vehicle.getCOEmission(veh_id)
                except traci.TraCIException:
                    co = None
                
                try:
                    nox = traci.vehicle.getNOxEmission(veh_id)
                except traci.TraCIException:
                    nox = None
                
                try:
                    fuel = traci.vehicle.getFuelConsumption(veh_id)
                except traci.TraCIException:
                    fuel = None
                
                self.vehicle_data.append({
                    'step': step,
                    'time': time,
                    'vehicle_id': veh_id,
                    'position_x': pos[0],
                    'position_y': pos[1],
                    'speed': speed,
                    'acceleration': accel,
                    'angle': angle,
                    'waiting_time': waiting_time,
                    'lane_id': lane_id,
                    'lane_position': lane_pos,
                    'route': route_str,
                    'co2_emission': co2,
                    'co_emission': co,
                    'nox_emission': nox,
                    'fuel_consumption': fuel,
                })
            except traci.TraCIException:
                # Vehicle may have left simulation, skip
                continue
    
    def _collect_traffic_light_data(self, step: int, time: float):
        """Collect data for all traffic lights."""
        tls_ids = traci.trafficlight.getIDList()
        
        for tls_id in tls_ids:
            try:
                phase = traci.trafficlight.getPhase(tls_id)
                phase_duration = traci.trafficlight.getPhaseDuration(tls_id)
                program = traci.trafficlight.getProgram(tls_id)
                state = traci.trafficlight.getRedYellowGreenState(tls_id)
                
                # Get controlled lanes
                try:
                    controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
                    controlled_lanes_str = ','.join(controlled_lanes) if controlled_lanes else None
                except traci.TraCIException:
                    controlled_lanes_str = None
                
                self.traffic_light_data.append({
                    'step': step,
                    'time': time,
                    'tls_id': tls_id,
                    'phase': phase,
                    'phase_duration': phase_duration,
                    'program': program,
                    'state': state,
                    'controlled_lanes': controlled_lanes_str,
                })
            except traci.TraCIException:
                continue
    
    def _collect_lane_data(self, step: int, time: float):
        """Collect data for all lanes."""
        lane_ids = traci.lane.getIDList()
        
        for lane_id in lane_ids:
            try:
                occupancy = traci.lane.getLastStepOccupancy(lane_id)
                vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                waiting_time = traci.lane.getWaitingTime(lane_id)
                
                # Calculate density (vehicles per km)
                try:
                    length = traci.lane.getLength(lane_id)
                    density = (vehicle_count / length * 1000) if length > 0 else 0.0
                except traci.TraCIException:
                    density = None
                
                # Get queue length
                try:
                    queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                except traci.TraCIException:
                    queue_length = None
                
                self.lane_data.append({
                    'step': step,
                    'time': time,
                    'lane_id': lane_id,
                    'occupancy': occupancy,
                    'density': density,
                    'vehicle_count': vehicle_count,
                    'mean_speed': mean_speed,
                    'waiting_time': waiting_time,
                    'queue_length': queue_length,
                })
            except traci.TraCIException:
                continue
    
    def _collect_junction_data(self, step: int, time: float):
        """Collect data for all junctions."""
        junction_ids = traci.junction.getIDList()
        
        for junction_id in junction_ids:
            try:
                # Get position
                try:
                    pos = traci.junction.getPosition(junction_id)
                    pos_x, pos_y = pos[0], pos[1]
                except traci.TraCIException:
                    pos_x, pos_y = None, None
                
                # Note: TraCI junction domain doesn't have getWaitingTime()
                # Junction waiting time can be calculated from lane waiting times
                # if needed, but is not directly available
                
                self.junction_data.append({
                    'step': step,
                    'time': time,
                    'junction_id': junction_id,
                    'position_x': pos_x,
                    'position_y': pos_y,
                })
            except traci.TraCIException:
                continue
    
    def _collect_edge_data(self, step: int, time: float):
        """Collect data for all edges."""
        edge_ids = traci.edge.getIDList()
        
        for edge_id in edge_ids:
            try:
                mean_speed = traci.edge.getLastStepMeanSpeed(edge_id)
                vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
                occupancy = traci.edge.getLastStepOccupancy(edge_id)
                
                # Get travel time
                try:
                    travel_time = traci.edge.getTraveltime(edge_id)
                except traci.TraCIException:
                    travel_time = None
                
                self.edge_data.append({
                    'step': step,
                    'time': time,
                    'edge_id': edge_id,
                    'mean_speed': mean_speed,
                    'vehicle_count': vehicle_count,
                    'occupancy': occupancy,
                    'travel_time': travel_time,
                })
            except traci.TraCIException:
                continue
    
    def _collect_simulation_data(self, step: int, time: float):
        """Collect simulation-level data."""
        try:
            vehicle_count = traci.simulation.getMinExpectedNumber()
            departed_count = traci.simulation.getDepartedNumber()
            arrived_count = traci.simulation.getArrivedNumber()
            
            # Get number of vehicles currently in simulation
            current_vehicles = len(traci.vehicle.getIDList())
            
            self.simulation_data.append({
                'step': step,
                'time': time,
                'vehicle_count': current_vehicles,
                'expected_vehicles': vehicle_count,
                'departed_count': departed_count,
                'arrived_count': arrived_count,
            })
        except traci.TraCIException:
            pass
    
    def get_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Convert collected data to pandas DataFrames.
        
        Returns:
            Dictionary of DataFrames with keys: 'vehicles', 'traffic_lights',
            'lanes', 'junctions', 'edges', 'simulation'
        """
        dfs = {}
        
        if self.vehicle_data:
            dfs['vehicles'] = pd.DataFrame(self.vehicle_data)
        else:
            dfs['vehicles'] = pd.DataFrame()
        
        if self.traffic_light_data:
            dfs['traffic_lights'] = pd.DataFrame(self.traffic_light_data)
        else:
            dfs['traffic_lights'] = pd.DataFrame()
        
        if self.lane_data:
            dfs['lanes'] = pd.DataFrame(self.lane_data)
        else:
            dfs['lanes'] = pd.DataFrame()
        
        if self.junction_data:
            dfs['junctions'] = pd.DataFrame(self.junction_data)
        else:
            dfs['junctions'] = pd.DataFrame()
        
        if self.edge_data:
            dfs['edges'] = pd.DataFrame(self.edge_data)
        else:
            dfs['edges'] = pd.DataFrame()
        
        if self.simulation_data:
            dfs['simulation'] = pd.DataFrame(self.simulation_data)
        else:
            dfs['simulation'] = pd.DataFrame()
        
        return dfs
    
    def export_to_csv(self):
        """Export all collected data to CSV files."""
        if not self.output_dir:
            return
        
        dfs = self.get_dataframes()
        
        # Export each DataFrame to CSV
        file_mapping = {
            'vehicles': 'vehicle_data.csv',
            'traffic_lights': 'traffic_light_data.csv',
            'lanes': 'lane_data.csv',
            'junctions': 'junction_data.csv',
            'edges': 'edge_data.csv',
            'simulation': 'simulation_data.csv',
        }
        
        for key, filename in file_mapping.items():
            if key in dfs and not dfs[key].empty:
                filepath = self.output_dir / filename
                dfs[key].to_csv(filepath, index=False)
                print(f">>> Exported {filename} to {filepath}")
    
    def reset(self):
        """Reset all collected data."""
        self.vehicle_data = []
        self.traffic_light_data = []
        self.lane_data = []
        self.junction_data = []
        self.edge_data = []
        self.simulation_data = []
        self.current_step = 0

