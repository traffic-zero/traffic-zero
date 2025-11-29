"""
Multi-Agent PPO Training Script

Trains multiple traffic light agents using Stable Baselines3 PPO
with centralized training and decentralized execution (CTDE).
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import gymnasium as gym
from gymnasium import spaces
from torch import nn

from sim.carla.multi_agent_bridge import MultiAgentTrafficEnv
from agents.multi_agent_utils import detect_compute_device


class MultiAgentPPOWrapper(gym.Env):
    """
    Wrapper to convert multi-agent environment to single-agent for SB3.

    SB3 doesn't natively support multi-agent, so we wrap the multi-agent
    environment to present a single-agent interface during training.
    """

    def __init__(self, multi_agent_env: MultiAgentTrafficEnv):
        """
        Initialize wrapper.

        Args:
            multi_agent_env: Multi-agent traffic environment
        """
        super().__init__()
        self.multi_agent_env = multi_agent_env
        self.agent_ids: list[str] = []
        self._initialized = False
        # Initialize dummy spaces (will be updated after first reset)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(1)

    def reset(  # type: ignore[override]
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment and return batched observation."""
        obs_dict, info_dict = self.multi_agent_env.reset(
            seed=seed, options=options
        )

        if not self._initialized:
            self.agent_ids = sorted(obs_dict.keys())
            self._initialized = True
            # Update observation and action spaces based on actual agents
            total_obs_dim = sum(
                len(obs_dict[agent_id]) for agent_id in self.agent_ids
            )
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_obs_dim,),
                dtype=np.float32,
            )
            # Action space: one action per agent
            self.action_space = spaces.MultiDiscrete([4] * len(self.agent_ids))

        obs_list = [obs_dict[agent_id] for agent_id in self.agent_ids]
        batched_obs = np.concatenate(obs_list)

        return batched_obs, info_dict

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step environment with batched action."""
        num_agents = len(self.agent_ids)
        action_per_agent = action.reshape(num_agents, -1)

        actions_dict = {
            agent_id: int(action_per_agent[i][0])
            for i, agent_id in enumerate(self.agent_ids)
        }

        (
            obs_dict,
            rewards_dict,
            terminateds_dict,
            truncateds_dict,
            infos_dict,
        ) = self.multi_agent_env.step(actions_dict)

        obs_list = [obs_dict[agent_id] for agent_id in self.agent_ids]
        batched_obs = np.concatenate(obs_list)

        shared_reward = sum(rewards_dict.values())
        done = any(terminateds_dict.values()) or any(truncateds_dict.values())

        info = {
            "agent_rewards": rewards_dict,
            "agent_infos": infos_dict,
        }

        return batched_obs, shared_reward, done, False, info


def make_env(
    sumo_cfg_file: str,
    config: dict[str, Any],
    rank: int = 0,
    seed: int = 0,
) -> MultiAgentPPOWrapper:
    """
    Create a multi-agent environment.

    Args:
        sumo_cfg_file: Path to SUMO configuration file
        config: Configuration dictionary
        rank: Environment rank (for parallel training)
        seed: Random seed

    Returns:
        Wrapped multi-agent environment
    """
    env = MultiAgentTrafficEnv(
        sumo_cfg_file=sumo_cfg_file,
        num_agents=config.get("num_agents"),
        enable_ctde=config.get("enable_ctde", True),
        neighbor_radius=config.get("neighbor_radius", 1),
        observation_config=config.get("observation_config", {}),
        action_config=config.get("action_config", {}),
        video_config=config.get("video_config", {}),
        device=config.get("device"),
    )

    wrapped = MultiAgentPPOWrapper(env)
    return wrapped


def train(
    sumo_cfg_file: str,
    config: dict[str, Any],
    output_dir: str = "models/multi_agent_ppo",
    total_timesteps: int = 100000,
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
) -> None:
    """
    Train multi-agent PPO agents.

    Args:
        sumo_cfg_file: Path to SUMO configuration file
        config: Training configuration dictionary
        output_dir: Directory to save models and logs
        total_timesteps: Total training timesteps
        eval_freq: Evaluation frequency in timesteps
        n_eval_episodes: Number of episodes for evaluation
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = detect_compute_device(config.get("device"))
    print(f"Using device: {device}")

    n_envs = config.get("n_envs", 1)

    if n_envs == 1:
        env = make_env(sumo_cfg_file, config, seed=0)
        env = Monitor(env, str(output_path / "monitor"))
    else:
        env = SubprocVecEnv(
            [
                (
                    lambda rank, seed: lambda: Monitor(
                        make_env(sumo_cfg_file, config, rank=rank, seed=seed),
                        str(output_path / f"monitor_{seed}"),
                    )
                )(i, i)
                for i in range(n_envs)
            ]
        )

    policy_kwargs = config.get("policy_kwargs", {})

    # Convert activation function string to callable
    if "activation_fn" in policy_kwargs:
        activation_name = policy_kwargs["activation_fn"]
        if isinstance(activation_name, str):
            activation_map = {
                "tanh": nn.Tanh,
                "relu": nn.ReLU,
                "elu": nn.ELU,
                "leaky_relu": nn.LeakyReLU,
                "sigmoid": nn.Sigmoid,
            }
            if activation_name.lower() in activation_map:
                policy_kwargs["activation_fn"] = activation_map[
                    activation_name.lower()
                ]
            else:
                raise ValueError(
                    f"Unknown activation function: {activation_name}. "
                    f"Supported: {list(activation_map.keys())}"
                )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.get("learning_rate", 3e-4),
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("n_epochs", 10),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_range=config.get("clip_range", 0.2),
        ent_coef=config.get("ent_coef", 0.01),
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        tensorboard_log=str(output_path / "tensorboard"),
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=str(output_path / "checkpoints"),
        name_prefix="multi_agent_ppo",
    )

    eval_env = make_env(sumo_cfg_file, config, seed=42)
    eval_env = Monitor(eval_env, str(output_path / "eval_monitor"))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path / "best_model"),
        log_path=str(output_path / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Output directory: {output_path}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    model.save(str(output_path / "final_model"))

    print(f"Training complete. Model saved to {output_path / 'final_model'}")


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train multi-agent PPO for traffic light control"
    )

    parser.add_argument(
        "--sumo-cfg",
        type=str,
        required=True,
        help="Path to SUMO configuration file",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/multi_agent_ppo.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/multi_agent_ppo",
        help="Output directory for models and logs",
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="Total training timesteps",
    )

    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluation frequency in timesteps",
    )

    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=5,
        help="Number of episodes for evaluation",
    )

    args = parser.parse_args()

    import yaml

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(
        sumo_cfg_file=args.sumo_cfg,
        config=config,
        output_dir=args.output_dir,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
    )


if __name__ == "__main__":
    main()
