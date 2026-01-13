
"""
Script for optimizing VLM usage
"""

import gymnasium as gym
import ray
from ray import serve
import time
import numpy as np
import os

from env.visual_minecraft.env import GridWorldEnv

from ray_based_architecture.actor import ExperienceCollector
from ray_based_architecture.shared_memory.sac_replay_buffer import SACReplayBuffer

# Set environment variable BEFORE Ray starts
os.environ["RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S"] = "0.5"

from env.visual_minecraft.fixed_len_env import GridWorldEnv
from ray_based_architecture.vlm_service import VLMService

def make_env():
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
    # Needs registering inside the ray process
    gym.envs.registration.register(
        id="VisualMinecraft-v0",
        entry_point="env.visual_minecraft.env:GridWorldEnv",
    )

    env = gym.make(
        "VisualMinecraft-v0",
        **kwargs
    )
    return env

# Shutdown any existing Ray cluster to force reload of code
if ray.is_initialized():
    ray.shutdown()

ray.init(
    address="auto",
    runtime_env={
        "env_vars": {
            "RAY_DEBUG": "1",
            "RAY_RECORD_REF_CREATION_SITES": "1",
        }
    }
)

# Replay buffer (filled by VLMService directly)
replay_buffer_name = "sac_replay_buffer"
replay_buffer_namespace = ray.get_runtime_context().namespace or "default"
replay_buffer = SACReplayBuffer.options(name=replay_buffer_name).remote(capacity=1_000_000, seed=0)

# Setup ray actors collecting environment
vlm_handle = serve.run(VLMService.bind(replay_buffer_name, replay_buffer_namespace), name="default")

actor = ExperienceCollector.remote(make_env, vlm_handle)

target_size = 200_000
poll_s = 5.0

collector_ref = actor.collect_n_episodes.remote(
    100000,
    max_inflight_submits=500,
    log_every_episodes=10,
)

# Periodically report buffer fill rate while collector runs.
start = time.time()
prev_t = start
prev_size = ray.get(replay_buffer.__len__.remote())
while True:
    done, _ = ray.wait([collector_ref], timeout=poll_s)

    now = time.time()
    size = ray.get(replay_buffer.__len__.remote())
    dt = max(now - prev_t, 1e-9)
    dsize = size - prev_size
    rate = dsize / dt
    elapsed = now - start
    remaining = max(target_size - size, 0)
    eta = (remaining / rate) if rate > 1e-9 else None

    print(
        f"[{elapsed:8.1f}s] replay_buffer size={size} "
        f"(+{dsize} in {dt:.1f}s => {rate:.1f} transitions/s) "
        f"eta_to_{target_size}={eta:.1f}s" if eta is not None else
        f"[{elapsed:8.1f}s] replay_buffer size={size} "
        f"(+{dsize} in {dt:.1f}s => {rate:.1f} transitions/s) "
        f"eta_to_{target_size}=inf"
    )

    prev_t, prev_size = now, size
    if done:
        break

ray.get(collector_ref)
print("Done. Final replay_buffer size:", ray.get(replay_buffer.__len__.remote()))
