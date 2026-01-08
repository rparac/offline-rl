import gymnasium as gym
import ray
from ray import serve
import time
import numpy as np

from env.visual_minecraft.fixed_len_env import GridWorldEnv
from ray_based_architecture.vlm_service import VLMService


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

ray.init(
    runtime_env={
        "env_vars": {"RAY_DEBUG": "1"}
    }
)

# Setup ray actors collecting environment
vlm_handle = serve.run(VLMService.bind())

obs, info = env.reset()

out_ref = []

num_tests = 100000
for i in range(num_tests):
    obs[0, 0, 0] = i % 256
    ref = vlm_handle.compute_reward.remote(obs)
    out_ref.append(ref)

for ref in out_ref:
    out = ref.result()

print("Done")