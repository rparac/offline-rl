"""
Seems to be working for images just to reach one object
"""

import numpy as np
import gymnasium as gym
import torch
import wandb
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback
from offline_rl.stable_baseline_practice.discrete_sac import DiscreteSAC
from offline_rl.torchrl_practice.env_util import setup_visual_minecraft


# ============================================================================
# Hyperparameters (copied from offline_rl/torchrl_practice/sac_practice.py)
# ============================================================================
# Environment
NUM_ENVS = 8 

# Training
TOTAL_TIMESTEPS = 5_000_000  # total_frames
LEARNING_RATE = 3e-4
# Match TorchRL's buffer size exactly
BUFFER_SIZE = 10_000  # TorchRL uses 10,000
BATCH_SIZE = 512
# TorchRL starts training after collecting first batch (1000 frames)
LEARNING_STARTS = 1_000  # Match TorchRL's immediate training start

# SAC-specific
GAMMA = 0.99
TAU = 0.005
# Match TorchRL's "collect 1000 transitions, then do 10 SGD epochs with batch_size=512".
# In SB3 with VecEnv, each "step" produces NUM_ENVS transitions, so:
# transitions_per_update = TRAIN_FREQ[0] * NUM_ENVS
TRAIN_FREQ = (1000 // NUM_ENVS, "step")
GRADIENT_STEPS = 10  # number of gradient steps per update
TARGET_UPDATE_INTERVAL = 1
TARGET_ENTROPY = 0.1

# Policy
MAX_GRAD_NORM = 1.0
# Match TorchRL's architecture: single hidden layer of 16 units
# IMPORTANT: TorchRL defaults to num_qvalue_nets=2 (double-Q learning) even when
# you pass a single Q-network - it internally duplicates it!
POLICY_KWARGS = {
    # "normalized_image": True,
}

# Logging
WANDB_PROJECT = "offline-rl"
WANDB_RUN_NAME = "discrete-sac-test"
TENSORBOARD_LOG_DIR = "./tensorboard_logs/"

# Random seed
SEED = 42

# ============================================================================



def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = setup_visual_minecraft(image_env=True)
        env.reset(seed=seed + rank)
        # Wrap with Monitor to enable rollout metrics logging
        env = Monitor(env)
        return env
    set_random_seed(seed)
    return _init

def main():
    print("Creating VisualMinecraft environment...")
    
    vec_env = SubprocVecEnv([make_env(i, seed=SEED) for i in range(NUM_ENVS)])
    print(f"Action space: {vec_env.action_space} (n={vec_env.action_space.n})")
    print(f"Observation space: {vec_env.observation_space}")
    print()
    
    # Initialize wandb with tensorboard sync
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "learning_rate": LEARNING_RATE,
            "buffer_size": BUFFER_SIZE,
            "learning_starts": LEARNING_STARTS,
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "tau": TAU,
            "train_freq": TRAIN_FREQ,
            "gradient_steps": GRADIENT_STEPS,
            "target_update_interval": TARGET_UPDATE_INTERVAL,
            "num_envs": NUM_ENVS,
            "seed": SEED,
            "total_timesteps": TOTAL_TIMESTEPS,
        },
        sync_tensorboard=True,  # Sync tensorboard logs to wandb
        monitor_gym=True,
    )
    
    # Define metrics for wandb
    wandb.define_metric("global_step")
    global_step_metrics = ["rollout/*", "train/*", "time/*"]
    for metric in global_step_metrics:
        wandb.define_metric(metric, step_metric="global_step")
    
    print("Initializing DiscreteSAC...")
    model = DiscreteSAC(
        policy="DiscreteCnnPolicy",
        policy_kwargs=POLICY_KWARGS,
        env=vec_env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        target_entropy=TARGET_ENTROPY,
        critic_loss_type="smooth_l1",
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        ent_coef="auto_0.1", # 0.05,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        max_grad_norm=MAX_GRAD_NORM,
        verbose=1,
        seed=SEED,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        # replay_buffer_kwargs={
        #     "handle_timeout_termination": False,
        # }
    )
    print("✓ Initialized")
    print()
    
    # Create wandb callback
    wandb_callback = WandbCallback(
        # gradient_save_freq=1000,
        # model_save_path=f"models/{wandb.run.id}",
        verbose=2,
    )
    
    # Use learn() with a small number of steps to properly initialize everything
    # This ensures the logger is set up correctly
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=wandb_callback)
    print("✓ Training step succeeded!")
    
    wandb.finish()

if __name__ == "__main__":
    main()

