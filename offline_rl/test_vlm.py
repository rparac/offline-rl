from PIL import Image
import torch
import numpy as np
from offline_rl.vlm.reward_model import load_reward_model_from_config, compute_rewards
from offline_rl.vlm.trainer.config import CLIPRewardConfig

batch_size = 1
num_workers = 1

baseline_prompts = ["a grid world with a robot, a pickaxe, a gem, a door, and a lava"]
target_prompts = ["a grid world with a robot, a pickaxe, a gem, a door, and a lava where the robot is trying to get the pickaxe"]

image = Image.open("debug_env_images/episode_0_step_0.png")
# Convert PIL image to numpy array, then to tensor
# Format: (H, W, C) -> (1, H, W, C) with batch dimension
image_array = np.array(image)
# Convert to tensor: (H, W, C) -> (1, H, W, C) with uint8 dtype
frames = torch.from_numpy(image_array).unsqueeze(0).to(torch.uint8)

config = CLIPRewardConfig(
    name="clip",
    pretrained_model="ViT-L-14/laion2b_s32b_b82k",
    batch_size=batch_size,
    alpha=0.1,
    target_prompts=target_prompts,
    baseline_prompts=baseline_prompts,
    cache_dir="vlm/cache",
)

# Load and move model to CUDA - keeping everything in float32 (default)
# This avoids dtype mismatches since embed_prompts() creates float32 buffers
# and autocast in CLIPEmbed.forward() outputs float32
clip_reward_model = load_reward_model_from_config(config)
clip_reward_model = clip_reward_model.to(torch.device("cuda"))
print("Success")

rewards = compute_rewards(clip_reward_model, frames, batch_size, num_workers, worker_frames_tensor=None)
print(rewards)