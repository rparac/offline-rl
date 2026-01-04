
from offline_rl.vlm.trainer.config import Config
from offline_rl.vlm.trainer.train import train, train_simple


config = """
env_name: VisualMinecraft-v0 # RL environment name
base_path: /data/private/rp218/offline-rl # Base path to save logs and checkpoints
seed: 42 # Seed for reproducibility
description: VisualMinecraft training using CLIP reward
tags: # Wandb tags
  - training
  - VisualMinecraft
  - CLIP
reward:
  name: clip
  pretrained_model: ViT-L-14/openai # CLIP model name
  # CLIP batch size per synchronous inference step.
  # Batch size must be divisible by n_workers (GPU count)
  # so that it can be shared among workers, and must be a divisor
  # of n_envs * episode_length so that all batches can be of the
  # same size (no support for variable batch size as of now.)
  batch_size: 100 # Originally 1600
  alpha: 0.5 # Alpha value of Baseline CLIP (CO-RELATE)
  target_prompts: # Description of the goal state
    - a humanoid robot kneeling
  baseline_prompts: # Description of the environment
    - a humanoid robot
  # Path to pre-saved model weights. When executing multiple runs,
  # mount a volume to this path to avoid downloading the model
  # weights multiple times.
  cache_dir: /data/private/rp218/open_clip
rl:
  policy_name: MlpPolicy
  n_steps: 100000 # Total number of simulation steps to be collected.
  n_envs_per_worker: 2 # Number of environments per worker (GPU)
  episode_length: 50 # Desired episode length
  learning_starts: 100 # Number of env steps to collect before training
  train_freq: 200 # Number of collected env steps between training iterations
  batch_size: 1024 # SAC buffer sample size per gradient step
  gradient_steps: 1 # Number of samples to collect from the buffer per training step
  tau: 0.005 # SAC target network update rate
  gamma: 0.99 # SAC discount factor
  learning_rate: 3e-4 # SAC optimizer learning rate
  buffer_size: 10000 # SAC buffer size
  rl_kwargs:
    exploration_initial_eps: 1.0
    exploration_final_eps: 0.1
    exploration_fraction: 0.7
logging:
  checkpoint_freq: 800 # Number of env steps between checkpoints
  video_freq: 800 # Number of env steps between videos
"""

def main():
    # train(config)
    train_simple(config)

if __name__ == "__main__":
    main()