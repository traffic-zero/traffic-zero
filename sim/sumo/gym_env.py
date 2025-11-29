"""
SUMO-Only Gymnasium Environment

Provides a Gymnasium-compatible environment for SUMO traffic simulation
without requiring CARLA. This is the lightweight alternative for fast
RL training.
"""

import time
import uuid
from typing import Any

import numpy as np
import traci
import gymnasium as gym
from gymnasium import spaces


def _normalize_traci_value(value: float | tuple | None) -> float:
    """Normalize traci return value to float (handles tuple returns)."""
    if value is None:
        return 0.0
    if isinstance(value, tuple):
        return float(value[0]) if len(value) > 0 else 0.0
    return float(value)


# Error message constant
TRACI_CONNECTION_ERROR = "TraCI connection not initialized"


class SumoGymEnv(gym.Env):
    """
    SUMO-only Gymnasium environment for traffic light control.

    This environment runs SUMO simulations without CARLA, providing
    a lightweight alternative for fast RL training. It mirrors the
    interface of CarlaSumoGymEnv but without 3D visualization.
    """

    def __init__(
        self,
        sumo_cfg_file: str,
        enable_rl_control: bool = False,
        observation_config: dict[str, Any] | None = None,
        action_config: dict[str, Any] | None = None,
        video_config: dict[str, Any] | None = None,
        device: str | None = None,
        step_length: float = 0.05,
        gui: bool = False,
        **kwargs,
    ):
        """
        Initialize SUMO-only Gymnasium environment.

        Args:
            sumo_cfg_file: Path to SUMO configuration file
            enable_rl_control: Enable RL action space for traffic light control
            observation_config: Configuration dict for observation space
            action_config: Configuration dict for action space
            video_config: Configuration dict for video recording
                    (limited in SUMO-only mode)
            device: Compute device
                    ('cuda', 'npu', 'cpu', or None for auto detection)
            step_length: Simulation step length in seconds
            gui: If True, use SUMO-GUI for visualization
            **kwargs: Additional parameters (ignored for compatibility)
        """
        super().__init__()

        self.sumo_cfg = sumo_cfg_file
        self.enable_rl_control = enable_rl_control
        self.observation_config = observation_config or {}
        self.action_config = action_config or {}
        self.video_config = video_config or {}
        self.step_length = step_length
        self.gui = gui
        self.device = self._detect_device(device)

        # Unique connection label for TraCI (allows multiple environments)
        self._connection_label = f"sumo_{uuid.uuid4().hex[:8]}"
        self._conn: Any = None  # TraCI connection object

        self._tls_ids: list[str] = []
        self._initialized = False
        self._step_count = 0
        self._start_time: float | None = None

        self._initialize_spaces()

    def _detect_device(self, device: str | None) -> str:
        """
        Detect available compute device (GPU/NPU/CPU).

        Args:
            device: Preferred device or None for auto-detect

        Returns:
            Device string ('cuda', 'npu', or 'cpu')
        """
        if device:
            return device.lower()

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        try:
            import intel_extension_for_pytorch as ipex

            if hasattr(ipex, "xpu") and ipex.xpu.is_available():
                return "npu"
        except ImportError:
            pass

        return "cpu"

    def _initialize_spaces(self) -> None:
        """Initialize action and observation spaces based on configuration."""
        if self.enable_rl_control:
            num_phases = self.action_config.get("num_phases") or 4
            num_tls = self.action_config.get("num_traffic_lights") or 1

            if num_tls == 1:
                self.action_space = spaces.Discrete(num_phases)
            else:
                self.action_space = spaces.MultiDiscrete([num_phases] * num_tls)
        else:
            self.action_space = spaces.Discrete(1)

        obs_shape = self.observation_config.get("shape", (10,))
        obs_low = self.observation_config.get("low", 0.0)
        obs_high = self.observation_config.get("high", np.inf)

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=obs_shape,
            dtype=np.float32,
        )

    def _start_sumo(self) -> None:
        """Start SUMO simulation via TraCI with unique connection label."""
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c",
            self.sumo_cfg,
            "--step-length",
            str(self.step_length),
            "--no-warnings",
            "true",
            "--no-step-log",
            "true",
            "--verbose",
            "false",
        ]
        if self.gui:
            # Start paused so user can see initial state
            sumo_cmd.extend(["--start", "--quit-on-end"])
        traci.start(sumo_cmd, label=self._connection_label)
        self._conn = traci.getConnection(self._connection_label)

    def _get_intersection_observation(self, tls_id: str) -> np.ndarray:
        """
        Extract per-intersection observation features.

        Args:
            tls_id: Traffic light ID

        Returns:
            numpy array with intersection-specific features
        """
        features = []
        assert self._conn is not None, TRACI_CONNECTION_ERROR

        try:
            phase = self._conn.trafficlight.getPhase(tls_id)
            if isinstance(phase, tuple):
                phase = phase[0] if len(phase) > 0 else 0
            features.append(float(phase))

            controlled_lanes = self._conn.trafficlight.getControlledLanes(
                tls_id
            )
            for lane_id in controlled_lanes[:4]:
                occupancy = self._conn.lane.getLastStepOccupancy(lane_id)
                if isinstance(occupancy, tuple):
                    occupancy = occupancy[0] if len(occupancy) > 0 else 0.0
                features.append(float(occupancy))

                vehicle_count = self._conn.lane.getLastStepVehicleNumber(
                    lane_id
                )
                if isinstance(vehicle_count, tuple):
                    vehicle_count = (
                        vehicle_count[0] if len(vehicle_count) > 0 else 0
                    )
                features.append(float(vehicle_count))

        except traci.TraCIException:
            pass

        return np.array(features, dtype=np.float32)

    def _get_neighbor_intersections(self, tls_id: str) -> list[str]:
        """
        Identify adjacent intersections for a given traffic light.

        Uses spatial proximity based on controlled lanes to find neighbors.

        Args:
            tls_id: Traffic light ID

        Returns:
            List of neighbor traffic light IDs
        """
        assert self._conn is not None, TRACI_CONNECTION_ERROR
        all_tls = self._conn.trafficlight.getIDList()
        neighbors = []

        try:
            controlled_lanes = self._conn.trafficlight.getControlledLanes(
                tls_id
            )
            controlled_edges = set()
            for lane_id in controlled_lanes:
                edge_id = self._conn.lane.getEdgeID(lane_id)
                controlled_edges.add(edge_id)

            for other_tls in all_tls:
                if other_tls == tls_id:
                    continue

                other_lanes = self._conn.trafficlight.getControlledLanes(
                    other_tls
                )
                other_edges = {
                    self._conn.lane.getEdgeID(lane) for lane in other_lanes
                }

                if controlled_edges.intersection(other_edges):
                    neighbors.append(other_tls)

        except traci.TraCIException:
            pass

        return neighbors

    def _get_intersection_reward(self, tls_id: str) -> float:
        """
        Calculate per-intersection reward.

        Args:
            tls_id: Traffic light ID

        Returns:
            Reward value for this intersection
        """
        reward = 0.0
        assert self._conn is not None, TRACI_CONNECTION_ERROR

        try:
            controlled_lanes = self._conn.trafficlight.getControlledLanes(
                tls_id
            )
            total_waiting = 0.0
            vehicle_count = 0

            for lane_id in controlled_lanes:
                vehicles = self._conn.lane.getLastStepVehicleIDs(lane_id)
                vehicle_count += len(vehicles)

                for veh_id in vehicles:
                    waiting = self._conn.vehicle.getWaitingTime(veh_id)
                    if isinstance(waiting, tuple):
                        waiting = waiting[0] if len(waiting) > 0 else 0.0
                    total_waiting += float(waiting)

            reward = -total_waiting * 0.01 + vehicle_count * 0.1

        except traci.TraCIException:
            pass

        return float(reward)

    def _get_num_phases_from_tls(self, tls_id: str) -> int:
        """Get number of phases for a traffic light."""
        assert self._conn is not None, TRACI_CONNECTION_ERROR
        try:
            program = self._conn.trafficlight.getAllProgramLogics(tls_id)
            if program and len(program) > 0:
                return len(program[0].phases)
        except traci.TraCIException:
            pass
        return 4

    def _get_traffic_light_phases(self, obs_features: list[float]) -> None:
        """Extract traffic light phases and append to observation features."""
        if not self._tls_ids or self._conn is None:
            return

        for tls_id in self._tls_ids[:5]:
            try:
                phase = self._conn.trafficlight.getPhase(tls_id)
                if isinstance(phase, tuple):
                    phase = phase[0] if len(phase) > 0 else 0
                obs_features.append(float(phase))
            except traci.TraCIException:
                obs_features.append(0.0)

    def _get_lane_metrics(self, obs_features: list[float]) -> None:
        """Extract lane metrics and append to observation features."""
        if not self.observation_config.get("include_lane_metrics", False):
            return
        if self._conn is None:
            return

        try:
            lane_ids = self._conn.lane.getIDList()
            for lane_id in lane_ids[:5]:
                try:
                    occupancy = self._conn.lane.getLastStepOccupancy(lane_id)
                    if isinstance(occupancy, tuple):
                        occupancy = occupancy[0] if len(occupancy) > 0 else 0.0
                    obs_features.append(float(occupancy))
                except traci.TraCIException:
                    obs_features.append(0.0)
        except traci.TraCIException:
            pass

    def _normalize_observation_size(
        self, obs_features: list[float]
    ) -> np.ndarray:
        """Normalize observation features to match observation space."""
        has_shape = hasattr(self.observation_space, "shape")
        if has_shape and self.observation_space.shape is not None:
            target_size = self.observation_space.shape[0]
        else:
            target_size = 10

        obs_features = obs_features + [0.0] * (target_size - len(obs_features))

        obs_array = np.array(obs_features, dtype=np.float32)

        if self.device != "cpu":
            obs_array = self._move_observation_to_device(obs_array)

        return obs_array

    def _move_observation_to_device(self, obs: np.ndarray) -> np.ndarray:
        """
        Move observation to compute device (GPU/NPU) if available.

        Args:
            obs: Observation array

        Returns:
            Observation array (possibly moved to device and back)
        """
        try:
            import torch

            obs_tensor = torch.from_numpy(obs)
            if self.device == "cuda":
                obs_tensor = obs_tensor.cuda()
            elif self.device == "npu":
                obs_tensor = obs_tensor.to("xpu")
            return obs_tensor.cpu().numpy()
        except ImportError:
            return obs

    def _get_observation(self) -> np.ndarray:
        """
        Extract observation from current simulation state.

        Returns:
            numpy array with observation values
        """
        obs_features = []
        assert self._conn is not None, TRACI_CONNECTION_ERROR

        sim_time = _normalize_traci_value(self._conn.simulation.getTime())
        obs_features.append(float(sim_time))
        obs_features.append(float(self._step_count))

        vehicle_ids = self._conn.vehicle.getIDList()
        num_vehicles = len(vehicle_ids)
        obs_features.append(float(num_vehicles))

        self._get_traffic_light_phases(obs_features)
        self._get_lane_metrics(obs_features)

        return self._normalize_observation_size(obs_features)

    def _calculate_waiting_time_reward(self) -> float:
        """Calculate negative reward based on total waiting time."""
        assert self._conn is not None, TRACI_CONNECTION_ERROR
        vehicle_ids = self._conn.vehicle.getIDList()
        total_waiting = 0.0
        for veh_id in vehicle_ids:
            try:
                waiting = self._conn.vehicle.getWaitingTime(veh_id)
                if isinstance(waiting, tuple):
                    waiting = waiting[0] if len(waiting) > 0 else 0.0
                total_waiting += float(waiting)
            except traci.TraCIException:
                continue
        return -total_waiting * 0.01

    def _calculate_throughput_reward(self) -> float:
        """Calculate positive reward based on vehicle throughput."""
        assert self._conn is not None, TRACI_CONNECTION_ERROR
        vehicle_ids = self._conn.vehicle.getIDList()
        return len(vehicle_ids) * 0.1

    def _calculate_reward(self) -> float:
        """
        Calculate reward based on traffic metrics.

        Returns:
            Reward value (default: 0.0)
        """
        if not self.enable_rl_control:
            return 0.0

        reward = 0.0

        if self.observation_config.get("reward_waiting_time", False):
            reward += self._calculate_waiting_time_reward()

        if self.observation_config.get("reward_throughput", False):
            reward += self._calculate_throughput_reward()

        return float(reward)

    def _apply_action(self, action: Any) -> None:
        """
        Apply traffic light control action.

        Args:
            action: Action from action space
        """
        if (
            not self.enable_rl_control
            or not self._tls_ids
            or self._conn is None
        ):
            return

        if isinstance(self.action_space, spaces.Discrete):
            if len(self._tls_ids) > 0:
                tls_id = self._tls_ids[0]
                phase = int(action)
                self._conn.trafficlight.setPhase(tls_id, phase)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            action_array = np.asarray(action)
            for i, tls_id in enumerate(self._tls_ids):
                if i < len(action_array):
                    phase = int(action_array[i])
                    self._conn.trafficlight.setPhase(tls_id, phase)

    def _update_action_space_for_multiple_tls(self) -> None:
        """Update action space when multiple traffic lights are present."""
        if not self.enable_rl_control or not self._tls_ids:
            return

        num_tls = len(self._tls_ids)
        if num_tls <= 1:
            return

        num_phases = self._get_num_phases_from_tls(self._tls_ids[0])
        self.action_space = spaces.MultiDiscrete([num_phases] * num_tls)

    def _collect_traffic_light_ids(self) -> None:
        """Collect and update traffic light IDs from SUMO."""
        assert self._conn is not None, TRACI_CONNECTION_ERROR
        self._tls_ids = list(self._conn.trafficlight.getIDList())
        self._update_action_space_for_multiple_tls()

    def reset(  # type: ignore[override]
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            Tuple of (observation, info)
        """
        if self._initialized:
            self.close()

        self._start_sumo()
        self._collect_traffic_light_ids()

        self._initialized = True
        self._step_count = 0
        self._start_time = time.time()

        observation = self._get_observation()
        info = {
            "step": 0,
            "time": _normalize_traci_value(traci.simulation.getTime()),
            "num_vehicles": len(traci.vehicle.getIDList()),
            "tls_ids": self._tls_ids,
        }

        return observation, info

    def step(  # type: ignore[override]
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics.

        Args:
            action: Action to take in the environment

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        assert (
            self._initialized
        ), "Environment not initialized. Call reset() first."
        assert self._conn is not None, TRACI_CONNECTION_ERROR

        if action is not None and self.enable_rl_control:
            self._apply_action(action)

        # Run simulation step
        self._conn.simulationStep()
        self._step_count += 1

        observation = self._get_observation()
        reward = self._calculate_reward()

        done = False
        truncated = False

        try:
            min_expected = self._conn.simulation.getMinExpectedNumber()
            if isinstance(min_expected, tuple):
                min_expected = min_expected[0] if len(min_expected) > 0 else 0
            if min_expected <= 0:
                done = True
        except traci.TraCIException:
            pass

        if self._start_time is not None:
            max_duration = self.observation_config.get("max_duration", None)
            if max_duration and (time.time() - self._start_time) > max_duration:
                truncated = True

        info = {
            "step": self._step_count,
            "time": _normalize_traci_value(self._conn.simulation.getTime()),
            "num_vehicles": len(self._conn.vehicle.getIDList()),
        }

        return observation, reward, done, truncated, info

    def render(self) -> None:
        """
        Render the environment.

        For SUMO-only mode, this is a no-op as there's no 3D visualization.
        Use SUMO-GUI for visual debugging if needed.
        """
        pass

    def close(self) -> None:
        """Close the environment and clean up resources."""
        if self._initialized and self._conn is not None:
            try:
                self._conn.close()
            except traci.TraCIException:
                pass

            self._conn = None
            self._initialized = False
            self._step_count = 0
            self._start_time = None
            self._tls_ids = []
