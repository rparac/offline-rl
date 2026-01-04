"""
Simplified test script - just tests the training step to quickly identify errors.
"""

import numpy as np
import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from offline_rl.stable_baseline_practice.discrete_sac import DiscreteSAC
from offline_rl.torchrl_practice.env_util import setup_visual_minecraft

def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = setup_visual_minecraft()
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    print("Creating VisualMinecraft environment...")
    

    vec_env = SubprocVecEnv([make_env(i) for i in range(8)])
    print(f"Action space: {vec_env.action_space} (n={vec_env.action_space.n})")
    print(f"Observation space: {vec_env.observation_space}")
    print()
    
    print("Initializing DiscreteSAC...")
    model = DiscreteSAC(
        policy="DiscreteMlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        buffer_size=1000,
        learning_starts=20,
        batch_size=128,
        verbose=1,
        seed=42,
    )
    print("✓ Initialized")
    print()
    
    # Use learn() with a small number of steps to properly initialize everything
    # This ensures the logger is set up correctly
    model.learn(total_timesteps=int(2e5))
    print("✓ Training step succeeded!")

if __name__ == "__main__":
    main()

