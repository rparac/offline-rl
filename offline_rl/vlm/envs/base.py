from typing import Callable

import gymnasium
from gymnasium.wrappers import DtypeObservation
import numpy as np

from env.visual_minecraft.fixed_len_env import GridWorldEnv

def get_clip_rewarded_env_name(env_name: str) -> str:
    return "vlmrm/CLIPRewarded" + env_name


RENDER_DIM = {
    "VisualMinecraft-v0": (224, 224),
    "CartPole-v1": (400, 600),
    "MountainCarContinuous-v0": (400, 600),
    "Humanoid-v4": (480, 480),
}
class ToUintImage(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(3, 224, 224), dtype=np.uint8)
    
    def observation(self, obs):
        obs = np.clip(obs * 255.0, a_min=0, a_max=255).astype(np.uint8)
        return obs

def get_visual_minecraft_make_env(episode_length: int) -> Callable:

    def setup_visual_minecraft():
        env_id = "FixedLenVisualMinecraft-v0"

        items = ["pickaxe", "lava", "door", "gem", "empty"]
        formula = "(F c0)", 5, "task0: visit({1})".format(*items)
        kwargs = {
            "formula": formula,
            "episode_length": episode_length,
            "render_mode": "rgb_array",
            "state_type": "image",
            "train": False,
            "use_dfa_state": False,
            "random_start": True,
        }
        env = gymnasium.make(env_id, **kwargs)

        env = DtypeObservation(env, dtype=np.float32)
        env = ToUintImage(env)
        return env

    return setup_visual_minecraft


def get_make_env(
    env_name: str,
    *,
    render_mode: str = "rgb_array",
    **kwargs,
) -> Callable:
    def make_env_wrapper() -> gymnasium.Env:
        env: gymnasium.Env
        env = gymnasium.make(
            env_name,
            render_mode=render_mode,
            **kwargs,
        )
        return env

    return make_env_wrapper


def is_3d_env(env_name: str) -> bool:
    return env_name == "Humanoid-v4"