"""
Multi-Agent Utilities

Helper functions for multi-agent RL training, including agent management,
neighbor detection, device management, and metrics aggregation.
"""

from typing import Any
import numpy as np
import torch


def detect_compute_device(preferred: str | None = None) -> str:
    """
    Detect available compute device (GPU/NPU/CPU).

    Args:
        preferred: Preferred device ('cuda', 'npu', 'cpu') or None for auto

    Returns:
        Device string ('cuda', 'npu', or 'cpu')
    """
    if preferred and preferred.lower() in ["cuda", "npu", "cpu"]:
        return preferred

    if torch.cuda.is_available():
        return "cuda"
    if torch.xpu.is_available() or torch.mps.is_available():
        return "npu"

    return "cpu"


def get_device_torch(device: str | None = None) -> torch.device:
    """
    Get PyTorch device object.

    Args:
        device: Device string ('cuda', 'npu', 'cpu') or None for auto

    Returns:
        PyTorch device object
    """
    device_str = detect_compute_device(device)

    if device_str == "cuda":
        return torch.device("cuda")
    elif device_str == "npu":
        return torch.device("npu")
    else:
        return torch.device("cpu")


def move_to_device(
    data: np.ndarray | torch.Tensor, device: str | None = None
) -> torch.Tensor:
    """
    Move data to specified compute device.

    Args:
        data: NumPy array or PyTorch tensor
        device: Target device ('cuda', 'npu', 'cpu') or None for auto

    Returns:
        PyTorch tensor on target device
    """
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        tensor = data

    target_device = get_device_torch(device)
    return tensor.to(target_device)


def aggregate_multi_agent_metrics(
    metrics: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """
    Aggregate metrics across multiple agents.

    Args:
        metrics: Dictionary mapping agent ID to metrics dict

    Returns:
        Aggregated metrics dictionary
    """
    aggregated = {}

    if not metrics:
        return aggregated

    all_keys = set()
    for agent_metrics in metrics.values():
        all_keys.update(agent_metrics.keys())

    for key in all_keys:
        values = [
            float(agent_metrics.get(key, 0.0))
            for agent_metrics in metrics.values()
        ]

        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_min"] = float(np.min(values))
            aggregated[f"{key}_max"] = float(np.max(values))
            aggregated[f"{key}_sum"] = float(np.sum(values))

    return aggregated


def create_agent_id_mapping(agent_ids: list[str]) -> dict[str, int]:
    """
    Create mapping from agent ID to integer index.

    Args:
        agent_ids: List of agent IDs

    Returns:
        Dictionary mapping agent ID to index
    """
    return {agent_id: idx for idx, agent_id in enumerate(agent_ids)}


def batch_observations(
    observations: dict[str, np.ndarray], device: str | None = None
) -> torch.Tensor:
    """
    Batch observations from multiple agents into a single tensor.

    Args:
        observations: Dictionary mapping agent ID to observation array
        device: Target device ('cuda', 'npu', 'cpu') or None for auto

    Returns:
        Batched tensor of shape (num_agents, obs_dim)
    """
    if not observations:
        return torch.empty(0)

    obs_list = [
        observations[agent_id] for agent_id in sorted(observations.keys())
    ]
    batched = np.stack(obs_list)

    return move_to_device(batched, device)


def unbatch_actions(
    batched_actions: torch.Tensor | np.ndarray, agent_ids: list[str]
) -> dict[str, Any]:
    """
    Unbatch actions from tensor to per-agent dictionary.

    Args:
        batched_actions: Batched action tensor/array of shape (num_agents,)
        agent_ids: List of agent IDs in order

    Returns:
        Dictionary mapping agent ID to action
    """
    if isinstance(batched_actions, torch.Tensor):
        actions_array = batched_actions.cpu().numpy()
    else:
        actions_array = batched_actions

    return {
        agent_id: int(actions_array[i]) for i, agent_id in enumerate(agent_ids)
    }


def compute_shared_reward(
    rewards: dict[str, float], method: str = "sum"
) -> float:
    """
    Compute shared reward from individual agent rewards.

    Args:
        rewards: Dictionary mapping agent ID to reward
        method: Aggregation method ('sum', 'mean', 'min', 'max')

    Returns:
        Shared reward value
    """
    reward_values = list(rewards.values())

    if not reward_values:
        return 0.0

    if method == "sum":
        return float(np.sum(reward_values))
    elif method == "mean":
        return float(np.mean(reward_values))
    elif method == "min":
        return float(np.min(reward_values))
    elif method == "max":
        return float(np.max(reward_values))
    else:
        return float(np.mean(reward_values))
