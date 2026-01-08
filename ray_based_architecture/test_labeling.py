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

# Shutdown any existing Ray cluster to force reload of code
if ray.is_initialized():
    ray.shutdown()

ray.init(
    runtime_env={
        "env_vars": {"RAY_DEBUG": "1"}
    }
)

# Setup ray actors collecting environment
vlm_handle = serve.run(VLMService.bind(), name="default")

obs, info = env.reset()

# Use a sliding window approach to keep the pipeline full
num_tests = 100000
window_size = 500  # Keep 500 requests in flight at a time
pending_refs = []

start_time = time.time()
completed = 0

for i in range(num_tests):
    obs[0, 0, 0] = i % 256
    ref = vlm_handle.compute_reward.remote(obs)
    pending_refs.append(ref)
    
    # When window is full, wait for oldest requests to complete
    if len(pending_refs) >= window_size:
        # Wait for and process the oldest batch
        ready_refs = pending_refs[:100]
        for ref in ready_refs:
            _ = ref.result()
            completed += 1
        pending_refs = pending_refs[100:]
        
        # Print progress
        if completed % 1000 == 0:
            elapsed = time.time() - start_time
            throughput = completed / elapsed
            print(f"Completed: {completed}/{num_tests}, Throughput: {throughput:.1f} req/s")

# Process remaining requests
for ref in pending_refs:
    _ = ref.result()
    completed += 1

elapsed = time.time() - start_time
throughput = completed / elapsed
print(f"Done! Total time: {elapsed:.2f}s, Average throughput: {throughput:.1f} req/s")