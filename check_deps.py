"""
Dependency check script for traffic-zero project.

Checks installation and basic functionality of required Python packages.
"""

import sys


print("🐍 Python version:", sys.version)

# --- Torch ---
try:
    import torch

    x = torch.rand(3, 3)
    print("✅ Torch OK, tensor:", x)
except Exception as e:
    print("❌ Torch FAIL:", e)

# --- Gymnasium ---
try:
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    obs = env.reset()
    print("✅ Gym OK, first obs:", obs)
    env.close()
except Exception as e:
    print("❌ Gym FAIL:", e)

# --- PettingZoo ---
try:
    import pettingzoo

    print("✅ PettingZoo OK, version:", pettingzoo.__version__)
except Exception as e:
    print("❌ PettingZoo FAIL:", e)

# --- MPE2 ---
try:
    from mpe2 import simple_v3

    env = simple_v3.env()
    env.reset()
    print("✅ MPE2 OK, agents:", env.agents[:3], "...")
    env.close()
except Exception as e:
    print("❌ MPE2 FAIL:", e)

# --- Ray + RLlib ---
try:
    import ray
    from ray import rllib

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    print("✅ Ray OK, version:", ray.__version__)
    print("   RLlib ID:", rllib.__name__)
    ray.shutdown()
except Exception as e:
    print("❌ Ray FAIL:", e)

# --- TraCI ---
try:
    import traci

    print("✅ TraCI ID:", traci.__name__)
except Exception as e:
    print("❌ TraCI FAIL:", e)

# --- SUMO ---
try:
    import sumolib

    print("✅ SUMO ID:", sumolib.__package__)
except Exception as e:
    print("❌ SUMO FAIL:", e)

# --- Dash ---
try:
    import dash

    print("✅ Dash OK, version:", dash.__version__)
except Exception as e:
    print("❌ Dash FAIL:", e)

print("All checks done.")
