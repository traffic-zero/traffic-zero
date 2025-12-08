"""
SUMO-Only Gymnasium Environment

Provides a Gymnasium-compatible environment for SUMO traffic simulation
without requiring CARLA. This is the lightweight alternative for fast
RL training.

Features:
- Per-lane state observations (queue length,
    waiting count, avg speed, occupancy)
- Duration-based actions with yellow transition phases
- Exponential penalty reward function
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

# Number of metrics per lane (queue_length, waiting_count, avg_speed, occupancy)
LANE_METRICS_COUNT = 4


class SumoGymEnv(gym.Env):
    """
    SUMO-only Gymnasium environment for traffic light control.

    This environment runs SUMO simulations without CARLA, providing
    a lightweight alternative for fast RL training. It mirrors the
    interface of CarlaSumoGymEnv but without 3D visualization.

    Features:
    - Per-lane observations: queue_length, waiting_count, avg_speed, occupancy
    - Duration-based actions: (phase_id, duration_seconds)
    - Yellow transition phases before switching to new green
    - Exponential penalty reward function scaled to (-1000, 1000)
    """

    def __init__(
        self,
        sumo_cfg_file: str,
        enable_rl_control: bool = False,
        observation_config: dict[str, Any] | None = None,
        action_config: dict[str, Any] | None = None,
        video_config: dict[str, Any] | None = None,
        reward_config: dict[str, Any] | None = None,
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
            reward_config: Configuration dict for reward function
            device: Compute device ('cuda', 'npu', 'cpu', or None for auto)
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
        self.reward_config = reward_config or {}
        self.step_length = step_length
        self.gui = gui
        self.device = self._detect_device(device)

        # Action configuration
        self.num_phases = self.action_config.get("num_phases", 4)
        self.min_duration = self.action_config.get("min_duration", 10)
        self.max_duration = self.action_config.get("max_duration", 90)
        self.duration_step = self.action_config.get("duration_step", 10)
        self.yellow_seconds = self.action_config.get("yellow_seconds", 3)

        # Calculate number of duration options
        self.num_duration_steps = (
            (self.max_duration - self.min_duration) // self.duration_step
        ) + 1

        # Observation configuration
        self.max_lanes_per_tls = self.observation_config.get(
            "max_lanes_per_tls", 8
        )

        # Reward configuration
        self.wait_time_scale = self.reward_config.get("wait_time_scale", 60.0)
        self.new_waiting_scale = self.reward_config.get(
            "new_waiting_scale", 10.0
        )
        self.max_reward = self.reward_config.get(
            "max_reward", 100.0
        )  # Reduced from 1000

        # Unique connection label for TraCI (allows multiple environments)
        self._connection_label = f"sumo_{uuid.uuid4().hex[:8]}"
        self._conn: Any = None  # TraCI connection object

        self._tls_ids: list[str] = []
        self._tls_controlled_lanes: dict[str, list[str]] = {}
        self._initialized = False
        self._step_count = 0
        self._start_time: float | None = None

        # Phase transition state per TLS
        self._pending_phase: dict[str, int | None] = {}
        self._pending_duration: dict[str, float] = {}
        self._yellow_countdown: dict[str, float] = {}
        self._green_countdown: dict[str, float] = {}
        self._current_green_phase: dict[str, int] = {}

        # Reward tracking
        self._prev_waiting_count: int = 0
        self._prev_total_wait_time: float = 0.0

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
            num_tls = self.action_config.get("num_traffic_lights") or 1

            # Action space: [phase, duration_index] per traffic light
            if num_tls == 1:
                self.action_space = spaces.MultiDiscrete(
                    [self.num_phases, self.num_duration_steps]
                )
            else:
                self.action_space = spaces.MultiDiscrete(
                    [self.num_phases, self.num_duration_steps] * num_tls
                )
        else:
            self.action_space = spaces.Discrete(1)

        # Observation space: per-lane metrics
        # for each lane controlled by a TLS
        num_tls = self.action_config.get("num_traffic_lights") or 1
        obs_per_tls = self.max_lanes_per_tls * LANE_METRICS_COUNT + 3
        total_obs_dim = obs_per_tls * num_tls

        obs_low = self.observation_config.get("low", 0.0)
        obs_high = self.observation_config.get("high", 10000.0)

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(total_obs_dim,),
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

    def _duration_from_index(self, duration_idx: int) -> float:
        """
        Convert duration index to actual seconds.

        Args:
            duration_idx: Index into duration options [0, num_duration_steps-1]

        Returns:
            Duration in seconds
        """
        duration_idx = max(0, min(duration_idx, self.num_duration_steps - 1))
        return float(self.min_duration + duration_idx * self.duration_step)

    def _get_lane_state(self, lane_id: str) -> np.ndarray:
        """
        Get full metrics for a single lane.

        Args:
            lane_id: SUMO lane ID

        Returns:
            numpy array with [queue_length, waiting_count, avg_speed, occupancy]
        """
        assert self._conn is not None, TRACI_CONNECTION_ERROR

        try:
            queue_length = _normalize_traci_value(
                self._conn.lane.getLastStepHaltingNumber(lane_id)
            )
            waiting_count = _normalize_traci_value(
                self._conn.lane.getLastStepVehicleNumber(lane_id)
            )
            avg_speed = _normalize_traci_value(
                self._conn.lane.getLastStepMeanSpeed(lane_id)
            )
            occupancy = _normalize_traci_value(
                self._conn.lane.getLastStepOccupancy(lane_id)
            )
        except traci.TraCIException:
            queue_length = 0.0
            waiting_count = 0.0
            avg_speed = 0.0
            occupancy = 0.0

        return np.array(
            [queue_length, waiting_count, avg_speed, occupancy],
            dtype=np.float32,
        )

    def _get_intersection_observation(self, tls_id: str) -> np.ndarray:
        """
        Extract per-intersection observation features with per-lane metrics.

        Args:
            tls_id: Traffic light ID

        Returns:
            numpy array with intersection-specific features including:
            - Per-lane metrics:
                queue_length, waiting_count, avg_speed, occupancy
            - Current phase, time in phase, yellow active flag per TLS
        """
        features: list[float] = []
        assert self._conn is not None, TRACI_CONNECTION_ERROR

        try:
            # Get controlled lanes for this TLS
            controlled_lanes = self._tls_controlled_lanes.get(tls_id, [])

            # Collect lane metrics (up to max_lanes_per_tls)
            for i in range(self.max_lanes_per_tls):
                if i < len(controlled_lanes):
                    lane_state = self._get_lane_state(controlled_lanes[i])
                    features.extend(lane_state.tolist())
                else:
                    # Pad with zeros for missing lanes
                    features.extend([0.0] * LANE_METRICS_COUNT)

            # Add phase information
            phase = self._conn.trafficlight.getPhase(tls_id)
            if isinstance(phase, tuple):
                phase = phase[0] if len(phase) > 0 else 0
            features.append(float(phase))

            # Time in current phase (approximation based on green countdown)
            time_in_phase = self._green_countdown.get(tls_id, 0.0)
            features.append(float(time_in_phase))

            # Yellow active flag
            yellow_active = (
                1.0 if self._yellow_countdown.get(tls_id, 0.0) > 0 else 0.0
            )
            features.append(yellow_active)

        except traci.TraCIException:
            # Return zeros if error
            total_features = self.max_lanes_per_tls * LANE_METRICS_COUNT + 3
            features = [0.0] * total_features

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
        Calculate per-intersection reward based on traffic flow.

        Rewards throughput (moving vehicles) and penalizes waiting.
        Uses balanced scaling to make reward informative.

        Args:
            tls_id: Traffic light ID

        Returns:
            Reward value for this intersection in [-max_reward, max_reward]
        """
        assert self._conn is not None, TRACI_CONNECTION_ERROR

        try:
            controlled_lanes = self._tls_controlled_lanes.get(tls_id, [])
            moving_count = 0
            waiting_count = 0
            total_speed = 0.0

            for lane_id in controlled_lanes:
                try:
                    vehicles = self._conn.lane.getLastStepVehicleIDs(lane_id)

                    for veh_id in vehicles:
                        speed = _normalize_traci_value(
                            self._conn.vehicle.getSpeed(veh_id)
                        )
                        wait_time = _normalize_traci_value(
                            self._conn.vehicle.getWaitingTime(veh_id)
                        )

                        if speed > 0.5:
                            moving_count += 1
                            total_speed += speed
                        if wait_time > 0:
                            waiting_count += 1
                except traci.TraCIException:
                    continue

            # Throughput reward: bonus for moving vehicles
            avg_speed = total_speed / max(moving_count, 1)
            throughput_reward = moving_count * 0.1 + avg_speed * 0.05

            # Waiting penalty: pressure to clear queues
            waiting_penalty = waiting_count * 0.05

            # Combine with bias toward positive reward for any movement
            reward = throughput_reward - waiting_penalty

            return max(min(reward, self.max_reward), -self.max_reward)

        except traci.TraCIException:
            return 0.0

    def _get_num_phases_from_tls(self, tls_id: str) -> int:
        """Get number of phases for a traffic light."""
        assert self._conn is not None, TRACI_CONNECTION_ERROR
        try:
            program = (
                self._conn.trafficlight.getCompleteRedYellowGreenDefinition(
                    tls_id
                )
            )
            if program and len(program) > 0:
                return len(program[0].phases)
        except traci.TraCIException:
            pass
        return 4

    def _is_yellow_phase(self, tls_id: str, phase: int) -> bool:
        """
        Check if a phase is a yellow (transition) phase.

        Args:
            tls_id: Traffic light ID
            phase: Phase index to check

        Returns:
            True if phase is yellow, False otherwise
        """
        assert self._conn is not None, TRACI_CONNECTION_ERROR
        try:
            program = (
                self._conn.trafficlight.getCompleteRedYellowGreenDefinition(
                    tls_id
                )
            )
            if program and len(program) > 0:
                phases = program[0].phases
                if 0 <= phase < len(phases):
                    state = phases[phase].state
                    # Yellow phases contain 'y' in the state string
                    return "y" in state.lower()
        except traci.TraCIException:
            pass
        return False

    def _get_yellow_phase_for_transition(
        self, tls_id: str, from_phase: int, to_phase: int
    ) -> int | None:
        """
        Find the yellow phase between two green phases.

        Args:
            tls_id: Traffic light ID
            from_phase: Current green phase
            to_phase: Target green phase

        Returns:
            Yellow phase index or None if not found
        """
        assert self._conn is not None, TRACI_CONNECTION_ERROR
        try:
            program = (
                self._conn.trafficlight.getCompleteRedYellowGreenDefinition(
                    tls_id
                )
            )
            if program and len(program) > 0:
                phases = program[0].phases
                num_phases = len(phases)

                # Look for yellow phase after from_phase
                yellow_candidate = (from_phase + 1) % num_phases
                if self._is_yellow_phase(tls_id, yellow_candidate):
                    return yellow_candidate

                # If no yellow found, return None (will use current phase)
                return None
        except traci.TraCIException:
            pass
        return None

    def _is_green_phase(self, tls_id: str, phase: int) -> bool:
        """
        Check if a phase is a green (non-transition) phase.

        A green phase must contain at least one 'g' or 'G' and no 'y'.

        Args:
            tls_id: Traffic light ID
            phase: Phase index to check

        Returns:
            True if phase has green lights and no yellow
        """
        assert self._conn is not None, TRACI_CONNECTION_ERROR
        try:
            program = (
                self._conn.trafficlight.getCompleteRedYellowGreenDefinition(
                    tls_id
                )
            )
            if program and len(program) > 0:
                phases = program[0].phases
                if 0 <= phase < len(phases):
                    state = phases[phase].state.lower()
                    # Green phase has 'g' and no 'y'
                    return "g" in state and "y" not in state
        except traci.TraCIException:
            pass
        return False

    def _get_nearest_green_phase(self, tls_id: str, phase: int) -> int:
        """
        Get the nearest green phase to the given phase.

        If the given phase is green, returns it. Otherwise, finds the
        nearest green phase (preferring previous green phase).

        Args:
            tls_id: Traffic light ID
            phase: Current phase index

        Returns:
            Nearest green phase index
        """
        assert self._conn is not None, TRACI_CONNECTION_ERROR
        try:
            program = (
                self._conn.trafficlight.getCompleteRedYellowGreenDefinition(
                    tls_id
                )
            )
            if program and len(program) > 0:
                phases = program[0].phases
                num_phases = len(phases)

                # If current phase is green, return it
                if self._is_green_phase(tls_id, phase):
                    return phase

                # Find nearest green phase (prefer previous)
                for offset in range(1, num_phases):
                    prev_phase = (phase - offset) % num_phases
                    if self._is_green_phase(tls_id, prev_phase):
                        return prev_phase

                # Check next phases
                for offset in range(1, num_phases):
                    next_phase = (phase + offset) % num_phases
                    if self._is_green_phase(tls_id, next_phase):
                        return next_phase
        except traci.TraCIException:
            pass

        # Fallback: even phases are typically green in standard TLS programs
        if phase % 2 == 1 and phase > 0:
            return phase - 1
        return 0  # Default to phase 0

    def _get_observation(self) -> np.ndarray:
        """
        Extract observation from current simulation state.

        Returns:
            numpy array with per-lane metrics for all traffic lights
        """
        all_features: list[float] = []
        assert self._conn is not None, TRACI_CONNECTION_ERROR

        for tls_id in self._tls_ids:
            tls_obs = self._get_intersection_observation(tls_id)
            all_features.extend(tls_obs.tolist())

        # Ensure observation matches expected shape
        obs_array = np.array(all_features, dtype=np.float32)

        # Pad or truncate to match observation space
        if (
            hasattr(self.observation_space, "shape")
            and self.observation_space.shape
        ):
            target_size = self.observation_space.shape[0]
            if len(obs_array) < target_size:
                obs_array = np.pad(
                    obs_array,
                    (0, target_size - len(obs_array)),
                    constant_values=0.0,
                )
            elif len(obs_array) > target_size:
                obs_array = obs_array[:target_size]

        return obs_array

    def _calculate_reward(self) -> float:
        """
        Calculate reward based on traffic flow improvement.

        Uses a differential reward that measures change in waiting time,
        making the reward directly connected to the agent's actions.

        Rewards:
        - Decrease in total waiting time (vehicles started moving)
        - Vehicles passing through intersection (throughput)
        - Maintaining flow (vehicles at good speed)

        Penalizes:
        - Increase in total waiting time
        - Vehicles stopped at red when they could move

        Returns:
            Reward value in range [-max_reward, max_reward]
        """
        if not self.enable_rl_control:
            return 0.0

        assert self._conn is not None, TRACI_CONNECTION_ERROR

        try:
            vehicle_ids = self._conn.vehicle.getIDList()
            current_total_wait = 0.0
            current_waiting_count = 0
            moving_count = 0
            total_speed = 0.0

            for veh_id in vehicle_ids:
                try:
                    wait_time = _normalize_traci_value(
                        self._conn.vehicle.getWaitingTime(veh_id)
                    )
                    speed = _normalize_traci_value(
                        self._conn.vehicle.getSpeed(veh_id)
                    )

                    current_total_wait += wait_time
                    if wait_time > 0:
                        current_waiting_count += 1
                    if speed > 0.5:
                        moving_count += 1
                        total_speed += speed
                except traci.TraCIException:
                    continue

            # Differential reward: reward for reducing wait time
            wait_time_delta = self._prev_total_wait_time - current_total_wait
            # Scale: +1 reward per second of wait time reduced
            wait_improvement_reward = wait_time_delta * 0.1

            # Throughput bonus: reward vehicles that are moving
            # Higher bonus for more vehicles moving at good speed
            avg_speed = total_speed / max(moving_count, 1)
            throughput_reward = moving_count * 0.05 + avg_speed * 0.02

            # Penalty for vehicles waiting (pressure to take action)
            waiting_penalty = current_waiting_count * 0.02

            # Combine rewards
            reward = (
                wait_improvement_reward + throughput_reward - waiting_penalty
            )

            # Update tracking for next step
            self._prev_waiting_count = current_waiting_count
            self._prev_total_wait_time = current_total_wait

            # Clamp to prevent extreme values
            return max(min(reward, self.max_reward), -self.max_reward)

        except traci.TraCIException:
            return 0.0

    def _apply_action(self, action: Any) -> None:
        """
        Apply traffic light control action with yellow transition.

        Action format:
            [phase, duration_index] for single TLS
            [phase1, dur1, phase2, dur2, ...] for multiple TLS

        When action is applied:
        1. Switch to yellow phase for yellow_seconds
        2. Then switch to requested green phase for duration seconds

        Actions are ignored during yellow transitions to prevent interrupting
        the safety-critical yellow phase.

        Args:
            action: phase + duration per TLS
        """
        if (
            not self.enable_rl_control
            or not self._tls_ids
            or self._conn is None
        ):
            return

        action_array = np.asarray(action).flatten()

        for i, tls_id in enumerate(self._tls_ids):
            # Skip if currently in yellow transition - don't interrupt!
            if self._yellow_countdown.get(tls_id, 0.0) > 0:
                continue

            # Get phase and duration for this TLS
            base_idx = i * 2
            if base_idx + 1 < len(action_array):
                target_phase = int(action_array[base_idx])
                duration_idx = int(action_array[base_idx + 1])
            elif base_idx < len(action_array):
                target_phase = int(action_array[base_idx])
                duration_idx = self.num_duration_steps // 2  # Default to middle
            else:
                continue

            # Map to green phase if target is yellow
            if self._is_yellow_phase(tls_id, target_phase):
                target_phase = self._get_nearest_green_phase(
                    tls_id, target_phase
                )

            # Calculate duration
            duration = self._duration_from_index(duration_idx)

            # Get current phase
            current_phase = self._current_green_phase.get(tls_id, 0)

            # If requesting different phase, initiate yellow transition
            if target_phase != current_phase:
                # Find yellow phase for transition
                yellow_phase = self._get_yellow_phase_for_transition(
                    tls_id, current_phase, target_phase
                )

                if yellow_phase is not None:
                    # Set yellow phase and start countdown
                    self._conn.trafficlight.setPhase(tls_id, yellow_phase)
                    self._yellow_countdown[tls_id] = self.yellow_seconds
                    self._pending_phase[tls_id] = target_phase
                    self._pending_duration[tls_id] = duration
                else:
                    # No yellow phase found, directly switch to green
                    self._conn.trafficlight.setPhase(tls_id, target_phase)
                    self._current_green_phase[tls_id] = target_phase
                    self._green_countdown[tls_id] = duration
                    self._pending_phase[tls_id] = None
            else:
                # Same phase, just update duration
                self._green_countdown[tls_id] = duration

    def _update_phase_timers(self) -> None:
        """Update yellow and green phase countdowns."""
        if self._conn is None:
            return

        for tls_id in self._tls_ids:
            # Update yellow countdown
            yellow_remaining = self._yellow_countdown.get(tls_id, 0.0)
            if yellow_remaining > 0:
                yellow_remaining -= self.step_length
                self._yellow_countdown[tls_id] = yellow_remaining

                # Yellow phase complete, switch to green
                if yellow_remaining <= 0:
                    pending = self._pending_phase.get(tls_id)
                    if pending is not None:
                        try:
                            self._conn.trafficlight.setPhase(tls_id, pending)
                            self._current_green_phase[tls_id] = pending
                            pending_dur = self._pending_duration.get(tls_id)
                            self._green_countdown[tls_id] = (
                                pending_dur
                                if pending_dur is not None
                                else float(self.min_duration)
                            )
                        except traci.TraCIException:
                            pass
                        self._pending_phase[tls_id] = None
                        self._pending_duration[tls_id] = 0.0

            # Update green countdown
            green_remaining = self._green_countdown.get(tls_id, 0.0)
            if green_remaining > 0:
                self._green_countdown[tls_id] = (
                    green_remaining - self.step_length
                )

    def _update_action_space_for_multiple_tls(self) -> None:
        """Update action space when multiple traffic lights are present."""
        if not self.enable_rl_control or not self._tls_ids:
            return

        num_tls = len(self._tls_ids)

        # Update number of phases based on first TLS
        if self._tls_ids:
            self.num_phases = self._get_num_phases_from_tls(self._tls_ids[0])

        # Action space: [phase, duration] per TLS
        self.action_space = spaces.MultiDiscrete(
            [self.num_phases, self.num_duration_steps] * num_tls
        )

        # Update observation space
        obs_per_tls = self.max_lanes_per_tls * LANE_METRICS_COUNT + 3
        total_obs_dim = obs_per_tls * num_tls

        obs_low = self.observation_config.get("low", 0.0)
        obs_high = self.observation_config.get("high", 10000.0)

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(total_obs_dim,),
            dtype=np.float32,
        )

    def _collect_traffic_light_ids(self) -> None:
        """Collect traffic light IDs and their controlled lanes from SUMO."""
        assert self._conn is not None, TRACI_CONNECTION_ERROR
        self._tls_ids = list(self._conn.trafficlight.getIDList())

        # Cache controlled lanes for each TLS
        self._tls_controlled_lanes = {}
        for tls_id in self._tls_ids:
            try:
                lanes = list(self._conn.trafficlight.getControlledLanes(tls_id))
                # Remove duplicates while preserving order
                seen = set()
                unique_lanes = []
                for lane in lanes:
                    if lane not in seen:
                        seen.add(lane)
                        unique_lanes.append(lane)
                self._tls_controlled_lanes[tls_id] = unique_lanes
            except traci.TraCIException:
                self._tls_controlled_lanes[tls_id] = []

        self._update_action_space_for_multiple_tls()

    def _initialize_phase_state(self) -> None:
        """Initialize phase tracking state for all traffic lights."""
        for tls_id in self._tls_ids:
            self._pending_phase[tls_id] = None
            self._pending_duration[tls_id] = 0.0
            self._yellow_countdown[tls_id] = 0.0
            self._green_countdown[tls_id] = float(self.min_duration)

            # Get current phase and ensure we start with a green phase
            try:
                current = self._conn.trafficlight.getPhase(tls_id)
                if isinstance(current, tuple):
                    current = current[0] if len(current) > 0 else 0

                # Find a valid green phase
                if not self._is_green_phase(tls_id, current):
                    current = self._get_nearest_green_phase(tls_id, current)

                # Actually set the traffic light to the green phase
                self._conn.trafficlight.setPhase(tls_id, current)
                self._current_green_phase[tls_id] = current
            except traci.TraCIException:
                self._current_green_phase[tls_id] = 0

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
        self._initialize_phase_state()

        self._initialized = True
        self._step_count = 0
        self._start_time = time.time()

        # Reset reward tracking
        self._prev_waiting_count = 0
        self._prev_total_wait_time = 0.0

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
            action: Action to take (phase + duration per TLS)

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

        # Update phase timers (yellow countdown, green countdown)
        self._update_phase_timers()

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
            max_duration = self.observation_config.get("max_episode_duration")
            if max_duration is not None:
                if (time.time() - self._start_time) > float(max_duration):
                    truncated = True

        info = {
            "step": self._step_count,
            "time": _normalize_traci_value(self._conn.simulation.getTime()),
            "num_vehicles": len(self._conn.vehicle.getIDList()),
            "waiting_count": self._prev_waiting_count,
            "total_wait_time": self._prev_total_wait_time,
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
            self._tls_controlled_lanes = {}
            self._pending_phase = {}
            self._pending_duration = {}
            self._yellow_countdown = {}
            self._green_countdown = {}
            self._current_green_phase = {}
