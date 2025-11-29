"""
Multi-Agent Environment Wrapper for CARLA-SUMO Co-Simulation

Provides multi-agent RL interface where each intersection is an
independent agent. Supports centralized training with decentralized
execution (CTDE).
"""

from typing import Any
import numpy as np
import traci
from gymnasium import spaces

from .bridge import CarlaSumoGymEnv


class MultiAgentTrafficEnv:
    """
    Multi-agent wrapper for traffic light control at multiple intersections.

    Each intersection is treated as an independent agent. During training,
    agents can share neighbor observations (CTDE). During execution, each
    agent acts based on local observations only.
    """

    def __init__(
        self,
        sumo_cfg_file: str,
        num_agents: int | None = None,
        enable_ctde: bool = True,
        neighbor_radius: int = 1,
        observation_config: dict[str, Any] | None = None,
        action_config: dict[str, Any] | None = None,
        video_config: dict[str, Any] | None = None,
        device: str | None = None,
        **kwargs,
    ):
        """
        Initialize multi-agent traffic environment.

        Args:
            sumo_cfg_file: Path to SUMO configuration file
            num_agents: Number of agents (intersections). If None, auto-detect
            enable_ctde: Enable centralized training with decentralized exec
            neighbor_radius: Maximum distance for neighbor detection
            observation_config: Configuration for observation spaces
            action_config: Configuration for action spaces
            video_config: Configuration for video recording
            device: Compute device ('cuda', 'npu', 'cpu', or None for auto)
            **kwargs: Additional parameters passed to CarlaSumoGymEnv
        """
        self.sumo_cfg_file = sumo_cfg_file
        self.enable_ctde = enable_ctde
        self.neighbor_radius = neighbor_radius
        self.device = device

        self.base_env = CarlaSumoGymEnv(
            sumo_cfg_file=sumo_cfg_file,
            enable_rl_control=True,
            observation_config=observation_config or {},
            action_config=action_config or {},
            video_config=video_config or {},
            device=device,
            **kwargs,
        )

        self.num_agents = num_agents
        self.agent_ids: list[str] = []
        self.agent_observation_spaces: dict[str, spaces.Space] = {}
        self.agent_action_spaces: dict[str, spaces.Space] = {}
        self.neighbor_map: dict[str, list[str]] = {}

        self._initialized = False

    def _detect_agents(self) -> list[str]:
        """
        Detect traffic light IDs from SUMO simulation.

        Returns:
            List of traffic light IDs (agent IDs)
        """
        tls_ids = list(traci.trafficlight.getIDList())
        assert len(tls_ids) > 0, (
            "No traffic lights found in simulation. "
            "Ensure traffic lights are configured in SUMO network."
        )

        if self.num_agents is not None:
            assert len(tls_ids) == self.num_agents, (
                f"Expected {self.num_agents} agents, "
                f"found {len(tls_ids)} traffic lights."
            )

        return tls_ids

    def _build_neighbor_map(self) -> dict[str, list[str]]:
        """
        Build map of neighbors for each agent.

        Returns:
            Dictionary mapping agent ID to list of neighbor agent IDs
        """
        neighbor_map: dict[str, list[str]] = {}

        for agent_id in self.agent_ids:
            neighbors = self.base_env._get_neighbor_intersections(agent_id)
            neighbor_map[agent_id] = neighbors[: self.neighbor_radius]

        return neighbor_map

    def _get_agent_observation_space(self, agent_id: str) -> spaces.Box:
        """
        Get observation space for a single agent.

        Args:
            agent_id: Agent (traffic light) ID

        Returns:
            Observation space for this agent
        """
        base_shape = self.base_env.observation_space.shape
        assert base_shape is not None, "Observation space must have a shape"

        local_dim = base_shape[0]

        if self.enable_ctde:
            neighbor_count = len(self.neighbor_map.get(agent_id, []))
            neighbor_dim = neighbor_count * local_dim
            global_dim = 5
            total_dim = local_dim + neighbor_dim + global_dim
        else:
            total_dim = local_dim

        obs_space = self.base_env.observation_space
        assert hasattr(obs_space, "low") and hasattr(
            obs_space, "high"
        ), "Observation space must have low and high attributes"
        # Type narrowing for Box space
        if hasattr(obs_space, "low") and hasattr(obs_space, "high"):
            obs_low = obs_space.low[0]  # type: ignore
            obs_high = obs_space.high[0]  # type: ignore
        else:
            obs_low = 0.0
            obs_high = np.inf

        return spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(total_dim,),
            dtype=np.float32,
        )

    def _get_agent_action_space(self, agent_id: str) -> spaces.Discrete:
        """
        Get action space for a single agent.

        Args:
            agent_id: Agent (traffic light) ID

        Returns:
            Action space for this agent
        """
        num_phases = self.base_env._get_num_phases_from_tls(agent_id)
        return spaces.Discrete(num_phases)

    def _get_local_observation(self, agent_id: str) -> np.ndarray:
        """
        Get local observation for an agent (its own intersection state).

        Args:
            agent_id: Agent (traffic light) ID

        Returns:
            Local observation array
        """
        return self.base_env._get_intersection_observation(agent_id)

    def _get_neighbor_observations(self, agent_id: str) -> np.ndarray:
        """
        Get neighbor observations for an agent (CTDE mode).

        Args:
            agent_id: Agent (traffic light) ID

        Returns:
            Concatenated neighbor observations
        """
        neighbors = self.neighbor_map.get(agent_id, [])
        neighbor_obs = []

        for neighbor_id in neighbors:
            neighbor_obs.append(self._get_local_observation(neighbor_id))

        if not neighbor_obs:
            return np.array([], dtype=np.float32)

        return np.concatenate(neighbor_obs)

    def _get_simulation_time_feature(self) -> float:
        """Extract simulation time feature."""
        sim_time = traci.simulation.getTime()
        if isinstance(sim_time, tuple):
            sim_time = sim_time[0] if len(sim_time) > 0 else 0.0
        return float(sim_time)

    def _get_vehicle_count_feature(self) -> float:
        """Extract vehicle count feature."""
        vehicle_ids = traci.vehicle.getIDList()
        return float(len(vehicle_ids))

    def _get_total_waiting_time_feature(self) -> float:
        """Extract total waiting time feature."""
        vehicle_ids = traci.vehicle.getIDList()
        total_waiting = 0.0
        for veh_id in vehicle_ids:
            waiting = traci.vehicle.getWaitingTime(veh_id)
            if isinstance(waiting, tuple):
                waiting = waiting[0] if len(waiting) > 0 else 0.0
            total_waiting += float(waiting)
        return total_waiting

    def _get_agent_count_feature(self) -> float:
        """Extract agent count feature."""
        return float(len(self.agent_ids))

    def _get_average_phase_feature(self) -> float:
        """Extract average traffic light phase feature."""
        avg_phase = 0.0
        phase_count = 0
        for agent_id in self.agent_ids:
            try:
                phase = traci.trafficlight.getPhase(agent_id)
                if isinstance(phase, tuple):
                    phase = phase[0] if len(phase) > 0 else 0
                avg_phase += float(phase)
                phase_count += 1
            except traci.TraCIException:
                pass

        if phase_count > 0:
            avg_phase /= phase_count
        return float(avg_phase)

    def _get_global_observation(self) -> np.ndarray:
        """
        Get global network observation (for CTDE).

        Returns:
            Global observation array with 5 features:
            [sim_time, vehicle_count, total_waiting, agent_count, avg_phase]
        """
        try:
            features = [
                self._get_simulation_time_feature(),
                self._get_vehicle_count_feature(),
                self._get_total_waiting_time_feature(),
                self._get_agent_count_feature(),
                self._get_average_phase_feature(),
            ]
        except traci.TraCIException:
            features = [0.0] * 5

        return np.array(features, dtype=np.float32)

    def _get_agent_observation(self, agent_id: str) -> np.ndarray:
        """
        Get full observation for an agent (local + neighbor + global if CTDE).

        Args:
            agent_id: Agent (traffic light) ID

        Returns:
            Full observation array
        """
        local_obs = self._get_local_observation(agent_id)

        if self.enable_ctde:
            neighbor_obs = self._get_neighbor_observations(agent_id)
            global_obs = self._get_global_observation()

            if len(neighbor_obs) == 0:
                neighbor_obs = np.zeros(
                    local_obs.shape[0] * self.neighbor_radius, dtype=np.float32
                )

            full_obs = np.concatenate([local_obs, neighbor_obs, global_obs])
        else:
            full_obs = local_obs

        if self.device != "cpu":
            full_obs = self._move_observation_to_device(full_obs)

        return full_obs

    def _move_observation_to_device(self, obs: np.ndarray) -> np.ndarray:
        """
        Move observation to compute device (GPU/NPU) if available.

        Note: torch is imported inline here because it's an optional
        dependency. If PyTorch is not installed, this function returns
        the observation unchanged.

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

    def _get_agent_reward(self, agent_id: str) -> float:
        """
        Get reward for a single agent.

        Args:
            agent_id: Agent (traffic light) ID

        Returns:
            Reward value for this agent
        """
        return self.base_env._get_intersection_reward(agent_id)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """
        Reset the multi-agent environment.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            Tuple of (observations dict, infos dict)
        """
        self.base_env.reset(seed=seed, options=options)

        if not self._initialized:
            self.agent_ids = self._detect_agents()
            self.neighbor_map = self._build_neighbor_map()

            for agent_id in self.agent_ids:
                self.agent_observation_spaces[agent_id] = (
                    self._get_agent_observation_space(agent_id)
                )
                self.agent_action_spaces[agent_id] = (
                    self._get_agent_action_space(agent_id)
                )

            self._initialized = True

        observations = {}
        infos = {}

        for agent_id in self.agent_ids:
            observations[agent_id] = self._get_agent_observation(agent_id)
            infos[agent_id] = {
                "step": 0,
                "neighbors": self.neighbor_map.get(agent_id, []),
            }

        return observations, infos

    def step(self, actions: dict[str, Any]) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """
        Run one timestep of the multi-agent environment.

        Args:
            actions: Dictionary mapping agent ID to action

        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, infos)
        """
        assert (
            self._initialized
        ), "Environment not initialized. Call reset() first."

        combined_action = None
        if len(self.agent_ids) == 1:
            combined_action = actions.get(self.agent_ids[0], 0)
        else:
            action_list = [
                actions.get(agent_id, 0) for agent_id in self.agent_ids
            ]
            combined_action = np.array(action_list)

        _, _, done, truncated, _ = self.base_env.step(combined_action)

        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}

        for agent_id in self.agent_ids:
            observations[agent_id] = self._get_agent_observation(agent_id)
            rewards[agent_id] = self._get_agent_reward(agent_id)
            terminateds[agent_id] = done
            truncateds[agent_id] = truncated
            infos[agent_id] = {
                "step": self.base_env._step_count,
                "neighbors": self.neighbor_map.get(agent_id, []),
            }

        return observations, rewards, terminateds, truncateds, infos

    def render(self) -> None:
        """Render the environment."""
        self.base_env.render()

    def close(self) -> None:
        """Close the environment and clean up resources."""
        self.base_env.close()

    @property
    def observation_spaces(self) -> dict[str, spaces.Space]:
        """Get observation spaces for all agents."""
        return self.agent_observation_spaces

    @property
    def action_spaces(self) -> dict[str, spaces.Space]:
        """Get action spaces for all agents."""
        return self.agent_action_spaces
