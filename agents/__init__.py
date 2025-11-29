"""
Agents module.

Provides multi-agent RL utilities and training scripts for traffic
light control.
"""

from .multi_agent_utils import (
    detect_compute_device,
    get_device_torch,
    move_to_device,
    aggregate_multi_agent_metrics,
    create_agent_id_mapping,
    batch_observations,
    unbatch_actions,
    compute_shared_reward,
)

__all__ = [
    "detect_compute_device",
    "get_device_torch",
    "move_to_device",
    "aggregate_multi_agent_metrics",
    "create_agent_id_mapping",
    "batch_observations",
    "unbatch_actions",
    "compute_shared_reward",
]
