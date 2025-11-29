"""
Multi-Agent PPO Training Script

Trains multiple traffic light agents using Stable Baselines3 PPO
with centralized training and decentralized execution (CTDE).

Supports two modes:
- SUMO-only (default): Fast, lightweight training
- CARLA co-simulation: 3D visualization, camera support (--carla flag)
"""

import argparse
from pathlib import Path
from typing import Any

from gymnasium.wrappers import TimeLimit
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import gymnasium as gym
from gymnasium import spaces
from torch import nn

from sim.carla.multi_agent_bridge import MultiAgentTrafficEnv
from agents.multi_agent_utils import detect_compute_device


class TqdmCallback(BaseCallback):
    """
    Custom callback for tqdm progress bar during training.

    Provides detailed progress visualization with metrics like
    reward and episode length updated in real-time.
    """

    def __init__(self, total_timesteps: int):
        """
        Initialize tqdm callback.

        Args:
            total_timesteps: Total number of timesteps for training
        """
        super().__init__()
        self.pbar: tqdm | None = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self) -> None:
        """Called when training starts."""
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
        )

    def _on_step(self) -> bool:
        """
        Called after each environment step.

        Returns:
            True to continue training, False to stop
        """
        if self.pbar is not None:
            self.pbar.update(1)
            # Add metrics to tqdm postfix
            ep_buffer = self.model.ep_info_buffer
            if ep_buffer is not None and len(ep_buffer) > 0:
                last_ep = ep_buffer[-1]
                self.pbar.set_postfix(
                    {
                        "reward": f"{last_ep['r']:.2f}",
                        "ep_len": last_ep["l"],
                    }
                )
        return True

    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.pbar is not None:
            self.pbar.close()


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

        # Initialize environment once to determine correct spaces
        # This is required because SB3 checks spaces before first reset()
        obs_dict, _ = self.multi_agent_env.reset()
        self.agent_ids = sorted(obs_dict.keys())

        # Calculate total observation dimension
        total_obs_dim = sum(
            len(obs_dict[agent_id]) for agent_id in self.agent_ids
        )

        # Set proper observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32,
        )
        # Action space: one action per agent (4 phases each)
        self.action_space = spaces.MultiDiscrete([4] * len(self.agent_ids))

    def reset(  # type: ignore[override]
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment and return batched observation."""
        obs_dict, info_dict = self.multi_agent_env.reset(
            seed=seed, options=options
        )

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
    max_episode_steps: int = 1000,
    gui: bool = False,
) -> gym.Env:
    """
    Create a multi-agent environment.

    Args:
        sumo_cfg_file: Path to SUMO configuration file
        config: Configuration dictionary
        max_episode_steps: Maximum number of steps per episode
        gui: If True, use SUMO-GUI for visualization

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
        use_carla=config.get("use_carla", False),
        gui=gui,
    )

    wrapped = MultiAgentPPOWrapper(env)
    wrapped = TimeLimit(wrapped, max_episode_steps)

    return wrapped


def evaluate(
    sumo_cfg_file: str,
    config: dict[str, Any],
    model_path: str,
    n_episodes: int = 5,
) -> None:
    """
    Run trained model with SUMO-GUI visualization.

    Args:
        sumo_cfg_file: Path to SUMO configuration file
        config: Configuration dictionary
        model_path: Path to trained model
        n_episodes: Number of episodes to run
    """
    print(f"\nLoading model from {model_path}")
    model = PPO.load(model_path)

    print(f"Running {n_episodes} episode(s) with SUMO-GUI...")
    print("=" * 60)

    for episode in range(n_episodes):
        # Create environment with GUI enabled (fresh env each episode)
        env = make_env(sumo_cfg_file, config, gui=True)

        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += float(reward)
            step += 1
            done = done or truncated

        env.close()
        print(
            f"Episode {episode + 1}/{n_episodes}: "
            f"reward={total_reward:.2f}, steps={step}"
        )

    print("=" * 60)
    print("Evaluation complete.")


def train(
    sumo_cfg_file: str,
    config: dict[str, Any],
    output_dir: str = "models/multi_agent_ppo",
    total_timesteps: int = 100000,
    eval_freq: int = 10000,
    n_eval_episodes: int = 1,
    skip_eval: bool = False,
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
        skip_eval: If True, skip evaluation during training (faster)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = detect_compute_device(config.get("device"))
    print(f"Using device: {device}")

    n_envs = config.get("n_envs", 1)

    if n_envs == 1:
        env = make_env(sumo_cfg_file, config)
        env = Monitor(env, str(output_path / "monitor"))
    else:
        env = SubprocVecEnv(
            [
                (
                    lambda rank, seed: lambda: Monitor(
                        make_env(sumo_cfg_file, config),
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

    # Create tqdm callback for progress visualization
    tqdm_callback = TqdmCallback(total_timesteps)

    # Build callback list
    callbacks = [checkpoint_callback, tqdm_callback]

    if not skip_eval:
        eval_env = make_env(sumo_cfg_file, config)
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
        callbacks.append(eval_callback)

    use_carla = config.get("use_carla", False)
    mode = "CARLA co-simulation" if use_carla else "SUMO-only"

    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Output directory: {output_path}")
    print(f"Mode: {mode}")
    if skip_eval:
        print("Evaluation: disabled (--no-eval)")
    else:
        print(
            f"Evaluation: every {eval_freq} steps, {n_eval_episodes} episode(s)"
        )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,  # Disable SB3's built-in progress bar
    )

    model.save(str(output_path / "final_model"))

    print(f"\nTraining complete. Model saved to {output_path / 'final_model'}")


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
        help="Evaluation frequency in timesteps (default: 10000)",
    )

    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=1,
        help="Number of episodes for evaluation during training (default: 1)",
    )

    parser.add_argument(
        "--carla",
        action="store_true",
        help="Enable CARLA co-simulation (default: SUMO-only mode)",
    )

    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train, don't run inference with GUI afterwards",
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation with GUI (requires existing model)",
    )

    parser.add_argument(
        "--gui-episodes",
        type=int,
        default=1,
        help="Number of episodes to run in "
        "GUI mode after training (default: 1)",
    )

    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation during training (faster)",
    )

    args = parser.parse_args()

    import yaml

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config with CLI flag if --carla is provided
    if args.carla:
        config["use_carla"] = True

    output_path = Path(args.output_dir)

    # Find best available model (prioritize final > best > checkpoint)
    def find_model() -> Path | None:
        candidates = [
            output_path / "final_model.zip",
            output_path / "best_model" / "best_model.zip",
        ]
        # Also check checkpoints
        checkpoint_dir = output_path / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = sorted(
                checkpoint_dir.glob("rl_model_*_steps.zip"), reverse=True
            )
            candidates.extend(checkpoints)

        for path in candidates:
            if path.exists():
                return path
        return None

    model_path = find_model()

    if args.train_only:
        # Train-only mode: always train
        train(
            sumo_cfg_file=args.sumo_cfg,
            config=config,
            output_dir=args.output_dir,
            total_timesteps=args.total_timesteps,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            skip_eval=args.no_eval,
        )
    elif args.eval_only:
        # Eval-only mode: just run GUI with existing model
        if model_path:
            print(f"Found model at {model_path}")
            evaluate(
                sumo_cfg_file=args.sumo_cfg,
                config=config,
                model_path=str(model_path),
                n_episodes=args.gui_episodes,
            )
        else:
            print(f"Error: No trained model found in {output_path}")
            print("Run training first with --train-only")
    else:
        # Default mode: train first, then run GUI
        print("Default mode: Training + GUI evaluation")
        train(
            sumo_cfg_file=args.sumo_cfg,
            config=config,
            output_dir=args.output_dir,
            total_timesteps=args.total_timesteps,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            skip_eval=args.no_eval,
        )
        model_path = find_model()
        if model_path:
            evaluate(
                sumo_cfg_file=args.sumo_cfg,
                config=config,
                model_path=str(model_path),
                n_episodes=args.gui_episodes,
            )


if __name__ == "__main__":
    main()
