import gymnasium as gym
import torch
import ray
import time
import numpy as np

from env.visual_minecraft.fixed_len_env import GridWorldEnv
from ray_based_architecture.vlm_service import VLMService, _initialize_similarity_model

items = ["pickaxe", "lava", "door", "gem", "empty"]
formula = "(F c0)", 5, "task0: visit({1})".format(*items)
# formula = "F(c0 & F(c1))", 5, "task3: seq_visit({0}, {1})".format(*items)
kwargs = {
    "formula": formula,
    "render_mode": "rgb_array",
    "state_type": "image",
    "train": False,
    "use_dfa_state": True,
    "random_start": True,
}

env = gym.make(
    "VisualMinecraft-v0",
    **kwargs
)

obs, info = env.reset()

out_ref = []

obs_list = []

num_tests = 16
for i in range(num_tests):
    obs[0, 0, 0] = i % 256
    img_only = obs['image']
    obs_list.append(img_only)

img_tensor = torch.stack([torch.from_numpy(img) for img in obs_list])
img_tensor = img_tensor.to(torch.device("cuda"))
img_tensor = img_tensor.half()

similarity_model = _initialize_similarity_model()

labels = similarity_model.compute_labels(img_tensor)

print("Done")
