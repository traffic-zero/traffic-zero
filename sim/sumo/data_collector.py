"""
Data collection module for SUMO TraCI.

Collects comprehensive data from SUMO simulation including vehicles,
traffic lights, lanes, junctions, edges, and simulation state.
"""

import traci
import pandas as pd
from pathlib import Path
from typing import Any
from .dataset import RealismMode, RealismLevel


def _normalize_traci_value(value: float | tuple | None) -> float:
    """Normalize traci return value to float (handles tuple returns)."""
    if value is None:
        return 0.0
    if isinstance(value, tuple):
        return float(value[0]) if len(value) > 0 else 0.0
    return float(value)


def _normalize_traci_int(value: int | tuple | None) -> int:
    """Normalize traci return value to int (handles tuple returns)."""
    if value is None:
        return 0
    if isinstance(value, tuple):
        return int(value[0]) if len(value) > 0 else 0
    return int(value)


class DataCollector:
    """
    Collects TraCI data from SUMO simulation at configurable intervals.

    Stores data in pandas DataFrames (in-memory) and exports to CSV files.
    """

    def __init__(
        self,
        collect_interval: int = 1,
        output_dir: str | None = None,
        exclude_empty_lanes: bool = False,
        lane_filter: str | None = None,
        realism_mode: RealismMode | None = None,
    ):
        """
        Initialize data collector.

        Args:
            collect_interval: Collect data every N simulation steps (default: 1)
            output_dir: Directory to save CSV files (None = no file output)
            exclude_empty_lanes: If True, skip lanes with zero
                                  occupancy/density (default: False)
            lane_filter: Filter lanes by type ('entry_exit' for e*,
                        'junction' for :n*, 'main_roads' for eN_0, eS_0,
                        eW_0, eE_0, etc., None for all)
            realism_mode: RealismMode instance for applying sensor noise
                         (default: None, no noise)
        """
        self.collect_interval = collect_interval
        self.output_dir = Path(output_dir) if output_dir else None
        self.exclude_empty_lanes = exclude_empty_lanes
        self.lane_filter = lane_filter
        self.realism_mode = realism_mode or RealismMode(RealismLevel.NONE)

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
        current_time_raw = traci.simulation.getTime()
        current_time = _normalize_traci_value(current_time_raw)

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

    def _get_vehicle_emissions(self, veh_id: str) -> dict[str, float | None]:
        """Get vehicle emission data with noise applied."""
        emissions = {}
        emission_types = [
            ("CO2", "getCO2Emission", "co2"),
            ("CO", "getCOEmission", "co"),
            ("NOx", "getNOxEmission", "nox"),
            ("Fuel", "getFuelConsumption", "fuel"),
        ]

        for name, method, key in emission_types:
            try:
                value = getattr(traci.vehicle, method)(veh_id)
                if value is not None:
                    value = self.realism_mode.apply_noise_to_value(
                        value,
                        f"vehicle_{veh_id}_{key}",
                        min_value=0.0,
                        default_value=value,
                    )
                emissions[key] = value
            except traci.TraCIException:
                emissions[key] = None

        return emissions

    def _get_vehicle_basic_data(self, veh_id: str) -> dict[str, Any] | None:
        """Get basic vehicle data with noise applied."""
        pos = traci.vehicle.getPosition(veh_id)
        noisy_pos = self.realism_mode.apply_noise_to_position(
            pos, f"vehicle_{veh_id}_pos"
        )
        if noisy_pos is None:
            return None

        pos_x, pos_y = noisy_pos
        speed = traci.vehicle.getSpeed(veh_id)
        speed = self.realism_mode.apply_noise_to_value(
            speed, f"vehicle_{veh_id}_speed", min_value=0.0, default_value=speed
        )
        if speed is None:
            return None

        angle = traci.vehicle.getAngle(veh_id)
        angle = self.realism_mode.apply_noise_to_value(
            angle,
            f"vehicle_{veh_id}_angle",
            min_value=-360.0,
            max_value=360.0,
            default_value=angle,
        )
        waiting_time = traci.vehicle.getWaitingTime(veh_id)
        waiting_time = self.realism_mode.apply_noise_to_value(
            waiting_time,
            f"vehicle_{veh_id}_waiting",
            min_value=0.0,
            default_value=waiting_time,
        )
        lane_pos = traci.vehicle.getLanePosition(veh_id)
        lane_pos = self.realism_mode.apply_noise_to_value(
            lane_pos,
            f"vehicle_{veh_id}_lane_pos",
            min_value=0.0,
            default_value=lane_pos,
        )

        return {
            "position_x": pos_x,
            "position_y": pos_y,
            "speed": speed,
            "angle": angle,
            "waiting_time": waiting_time,
            "lane_position": lane_pos,
            "lane_id": traci.vehicle.getLaneID(veh_id),
        }

    def _collect_vehicle_data(self, step: int, time: float):
        """Collect data for all vehicles in simulation."""
        vehicle_ids = traci.vehicle.getIDList()

        for veh_id in vehicle_ids:
            try:
                basic_data = self._get_vehicle_basic_data(veh_id)
                if basic_data is None:
                    continue

                # Get route
                try:
                    route = traci.vehicle.getRoute(veh_id)
                    route_str = ",".join(route) if route else None
                except traci.TraCIException:
                    route_str = None

                # Get acceleration
                try:
                    accel = traci.vehicle.getAcceleration(veh_id)
                    if accel is not None:
                        accel = self.realism_mode.apply_noise_to_value(
                            accel,
                            f"vehicle_{veh_id}_accel",
                            default_value=accel,
                        )
                except traci.TraCIException:
                    accel = None

                # Get emissions
                emissions = self._get_vehicle_emissions(veh_id)

                self.vehicle_data.append(
                    {
                        "step": step,
                        "time": time,
                        "vehicle_id": veh_id,
                        **basic_data,
                        "acceleration": accel,
                        "route": route_str,
                        "co2_emission": emissions["co2"],
                        "co_emission": emissions["co"],
                        "nox_emission": emissions["nox"],
                        "fuel_consumption": emissions["fuel"],
                    }
                )
            except traci.TraCIException:
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
                    controlled_lanes = traci.trafficlight.getControlledLanes(
                        tls_id
                    )
                    controlled_lanes_str = (
                        ",".join(controlled_lanes) if controlled_lanes else None
                    )
                except traci.TraCIException:
                    controlled_lanes_str = None

                self.traffic_light_data.append(
                    {
                        "step": step,
                        "time": time,
                        "tls_id": tls_id,
                        "phase": phase,
                        "phase_duration": phase_duration,
                        "program": program,
                        "state": state,
                        "controlled_lanes": controlled_lanes_str,
                    }
                )
            except traci.TraCIException:
                continue

    def _matches_lane_filter(self, lane_id: str) -> bool:
        """Check if lane matches the lane_filter setting."""
        if self.lane_filter == "entry_exit":
            return lane_id.startswith("e")
        if self.lane_filter == "junction":
            return lane_id.startswith(":n")
        if self.lane_filter == "main_roads":
            if not lane_id.startswith("e"):
                return False
            parts = lane_id.split("_")
            if len(parts) > 3:
                return False
            if len(parts) == 3 and parts[1] not in ["out"]:
                return False
            return True
        return True  # No filter or unknown filter

    def _is_empty_lane(
        self, vehicle_count: int, occupancy: float, density: float
    ) -> bool:
        """Check if lane is empty (no vehicles, zero occupancy, zero
        density).
        """
        density_zero = density is None or abs(density) < 1e-9
        return vehicle_count == 0 and abs(occupancy) < 1e-9 and density_zero

    def _should_collect_lane(
        self, lane_id: str, vehicle_count: int, occupancy: float, density: float
    ) -> bool:
        """
        Determine if a lane should be collected based on filters.

        Args:
            lane_id: Lane ID
            vehicle_count: Number of vehicles on lane
            occupancy: Occupancy percentage (0-100)
            density: Density (vehicles per km)

        Returns:
            True if lane should be collected, False otherwise
        """
        if not self._matches_lane_filter(lane_id):
            return False

        if self.exclude_empty_lanes and self._is_empty_lane(
            vehicle_count, occupancy, density
        ):
            return False

        return True

    def _get_lane_raw_data(self, lane_id: str) -> dict[str, Any] | None:
        """Get raw lane data from TraCI."""
        try:
            occupancy_fraction_raw = traci.lane.getLastStepOccupancy(lane_id)
            occupancy_fraction = _normalize_traci_value(occupancy_fraction_raw)
            occupancy_percentage = occupancy_fraction * 100.0

            vehicle_count_raw = traci.lane.getLastStepVehicleNumber(lane_id)
            vehicle_count = _normalize_traci_int(vehicle_count_raw)
            mean_speed_raw = traci.lane.getLastStepMeanSpeed(lane_id)
            mean_speed = _normalize_traci_value(mean_speed_raw)
            waiting_time_raw = traci.lane.getWaitingTime(lane_id)
            waiting_time = _normalize_traci_value(waiting_time_raw)

            try:
                length_raw = traci.lane.getLength(lane_id)
                length = _normalize_traci_value(length_raw)
                density = (vehicle_count / length * 1000) if length > 0 else 0.0
            except traci.TraCIException:
                density = None

            try:
                queue_length_raw = traci.lane.getLastStepHaltingNumber(lane_id)
                queue_length = _normalize_traci_int(queue_length_raw)
            except traci.TraCIException:
                queue_length = None

            return {
                "occupancy_percentage": occupancy_percentage,
                "vehicle_count": vehicle_count,
                "mean_speed": mean_speed,
                "waiting_time": waiting_time,
                "density": density,
                "queue_length": queue_length,
            }
        except traci.TraCIException:
            return None

    def _apply_lane_noise(
        self, lane_id: str, raw_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Apply noise to lane sensor readings."""
        occupancy = self.realism_mode.apply_noise_to_value(
            raw_data["occupancy_percentage"],
            f"lane_{lane_id}_occupancy",
            min_value=0.0,
            max_value=100.0,
            default_value=raw_data["occupancy_percentage"],
        )
        if occupancy is None:
            return None

        density = raw_data["density"]
        if density is not None:
            density = self.realism_mode.apply_noise_to_value(
                density,
                f"lane_{lane_id}_density",
                min_value=0.0,
                default_value=density,
            )

        vehicle_count = self.realism_mode.apply_noise_to_value(
            float(raw_data["vehicle_count"]),
            f"lane_{lane_id}_vehicle_count",
            min_value=0.0,
            default_value=float(raw_data["vehicle_count"]),
        )
        if vehicle_count is None:
            return None
        vehicle_count = max(0, int(round(vehicle_count)))

        mean_speed = self.realism_mode.apply_noise_to_value(
            raw_data["mean_speed"],
            f"lane_{lane_id}_speed",
            min_value=0.0,
            default_value=raw_data["mean_speed"],
        )
        waiting_time = self.realism_mode.apply_noise_to_value(
            raw_data["waiting_time"],
            f"lane_{lane_id}_waiting",
            min_value=0.0,
            default_value=raw_data["waiting_time"],
        )

        queue_length = raw_data["queue_length"]
        if queue_length is not None:
            queue_length = self.realism_mode.apply_noise_to_value(
                float(queue_length),
                f"lane_{lane_id}_queue",
                min_value=0.0,
                default_value=float(queue_length),
            )
            if queue_length is not None:
                queue_length = max(0, int(round(queue_length)))

        return {
            "occupancy": occupancy,
            "density": density,
            "vehicle_count": vehicle_count,
            "mean_speed": mean_speed,
            "waiting_time": waiting_time,
            "queue_length": queue_length,
        }

    def _collect_lane_data(self, step: int, time: float):
        """
        Collect data for all lanes.

        Note: SUMO's getLastStepOccupancy() returns occupancy as a fraction
        (0-1), where 0 = 0% occupied and 1 = 100% occupied. We convert this
        to percentage (0-100) for consistency with typical traffic metrics.

        Lanes can be filtered by type (entry_exit, junction, main_roads) or
        excluded if empty (exclude_empty_lanes=True).
        """
        lane_ids = traci.lane.getIDList()

        for lane_id in lane_ids:
            raw_data = self._get_lane_raw_data(lane_id)
            if raw_data is None:
                continue

            if not self._should_collect_lane(
                lane_id,
                raw_data["vehicle_count"],
                raw_data["occupancy_percentage"],
                raw_data["density"] or 0.0,
            ):
                continue

            noisy_data = self._apply_lane_noise(lane_id, raw_data)
            if noisy_data is None:
                continue

            self.lane_data.append(
                {
                    "step": step,
                    "time": time,
                    "lane_id": lane_id,
                    **noisy_data,
                }
            )

    def _collect_junction_data(self, step: int, time: float):
        """Collect data for all junctions."""
        junction_ids = traci.junction.getIDList()

        for junction_id in junction_ids:
            try:
                # Get position
                try:
                    pos = traci.junction.getPosition(junction_id)
                    noisy_pos = self.realism_mode.apply_noise_to_position(
                        pos, f"junction_{junction_id}_pos"
                    )
                    if noisy_pos is None:
                        pos_x, pos_y = None, None
                    else:
                        pos_x, pos_y = noisy_pos
                except traci.TraCIException:
                    pos_x, pos_y = None, None

                # Note: TraCI junction domain doesn't have getWaitingTime()
                # Junction waiting time can be calculated from lane waiting
                # times
                # if needed, but is not directly available

                self.junction_data.append(
                    {
                        "step": step,
                        "time": time,
                        "junction_id": junction_id,
                        "position_x": pos_x,
                        "position_y": pos_y,
                    }
                )
            except traci.TraCIException:
                continue

    def _collect_edge_data(self, step: int, time: float):
        """
        Collect data for all edges.

        Note: SUMO's getLastStepOccupancy() returns occupancy as a fraction
        (0-1), where 0 = 0% occupied and 1 = 100% occupied. We convert this
        to percentage (0-100) for consistency with typical traffic metrics.
        """
        edge_ids = traci.edge.getIDList()

        for edge_id in edge_ids:
            try:
                mean_speed_raw = traci.edge.getLastStepMeanSpeed(edge_id)
                mean_speed = _normalize_traci_value(mean_speed_raw)
                vehicle_count_raw = traci.edge.getLastStepVehicleNumber(edge_id)
                vehicle_count = _normalize_traci_int(vehicle_count_raw)

                # SUMO returns occupancy as fraction (0-1), convert to
                # percentage (0-100)
                occupancy_fraction_raw = traci.edge.getLastStepOccupancy(
                    edge_id
                )
                occupancy_fraction = _normalize_traci_value(
                    occupancy_fraction_raw
                )
                occupancy_percentage = occupancy_fraction * 100.0

                # Apply realism/noise to edge sensor readings
                mean_speed = self.realism_mode.apply_noise_to_value(
                    mean_speed,
                    f"edge_{edge_id}_speed",
                    min_value=0.0,
                    default_value=mean_speed,
                )
                if mean_speed is None:
                    continue

                vehicle_count_noisy = self.realism_mode.apply_noise_to_value(
                    float(vehicle_count),
                    f"edge_{edge_id}_vehicle_count",
                    min_value=0.0,
                    default_value=float(vehicle_count),
                )
                if vehicle_count_noisy is None:
                    continue
                vehicle_count = max(0, int(round(vehicle_count_noisy)))

                occupancy_percentage = self.realism_mode.apply_noise_to_value(
                    occupancy_percentage,
                    f"edge_{edge_id}_occupancy",
                    min_value=0.0,
                    max_value=100.0,
                    default_value=occupancy_percentage,
                )
                if occupancy_percentage is None:
                    continue

                # Get travel time
                try:
                    travel_time_raw = traci.edge.getTraveltime(edge_id)
                    travel_time = (
                        _normalize_traci_value(travel_time_raw)
                        if travel_time_raw is not None
                        else None
                    )
                    if travel_time is not None:
                        travel_time = self.realism_mode.apply_noise_to_value(
                            travel_time,
                            f"edge_{edge_id}_travel_time",
                            min_value=0.0,
                            default_value=travel_time,
                        )
                except traci.TraCIException:
                    travel_time = None

                self.edge_data.append(
                    {
                        "step": step,
                        "time": time,
                        "edge_id": edge_id,
                        "mean_speed": mean_speed,
                        "vehicle_count": vehicle_count,
                        "occupancy": occupancy_percentage,  # Now in
                        # percentage (0-100)
                        "travel_time": travel_time,
                    }
                )
            except traci.TraCIException:
                continue

    def _collect_simulation_data(self, step: int, time: float):
        """Collect simulation-level data."""
        try:
            vehicle_count_raw = traci.simulation.getMinExpectedNumber()
            vehicle_count = _normalize_traci_int(vehicle_count_raw)
            departed_count_raw = traci.simulation.getDepartedNumber()
            departed_count = _normalize_traci_int(departed_count_raw)
            arrived_count_raw = traci.simulation.getArrivedNumber()
            arrived_count = _normalize_traci_int(arrived_count_raw)

            # Get number of vehicles currently in simulation
            current_vehicles = len(traci.vehicle.getIDList())

            self.simulation_data.append(
                {
                    "step": step,
                    "time": time,
                    "vehicle_count": current_vehicles,
                    "expected_vehicles": vehicle_count,
                    "departed_count": departed_count,
                    "arrived_count": arrived_count,
                }
            )
        except traci.TraCIException:
            # Simulation may have ended or be in invalid state, ignore
            pass

    def get_dataframes(self) -> dict[str, pd.DataFrame]:
        """
        Convert collected data to pandas DataFrames.

        Returns:
            Dictionary of DataFrames with keys: 'vehicles', 'traffic_lights',
            'lanes', 'junctions', 'edges', 'simulation'
        """
        # Map from singular attribute names to plural DataFrame keys
        key_mapping = {
            "vehicle": "vehicles",
            "traffic_light": "traffic_lights",
            "lane": "lanes",
            "junction": "junctions",
            "edge": "edges",
            "simulation": "simulation",
        }

        return {
            key_mapping.get(
                name.removesuffix("_data"), name.removesuffix("_data")
            ): pd.DataFrame(getattr(self, name) or [])
            for name in vars(self)
            if name.endswith("_data")
        }

    def export_to_csv(self):
        """Export all collected data to CSV files."""
        if not self.output_dir:
            return

        dfs = self.get_dataframes()

        # Export each DataFrame to CSV
        file_mapping = {
            "vehicles": "vehicle_data.csv",
            "traffic_lights": "traffic_light_data.csv",
            "lanes": "lane_data.csv",
            "junctions": "junction_data.csv",
            "edges": "edge_data.csv",
            "simulation": "simulation_data.csv",
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
