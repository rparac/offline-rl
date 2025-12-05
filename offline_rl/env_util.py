import gymnasium as gym
from tensordict import TensorDictBase
import torch
from torchrl.data import Composite
from torchrl.envs.transforms import Transform
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import DTypeCastTransform, GymWrapper

from env.visual_minecraft.env import GridWorldEnv
# from env.ltl_wrappers import LTLEnv
# from env.simple_ltl_env import SimpleLTLEnv

def _get_args_kwargs_visual_minecraft():
    items = ["pickaxe", "lava", "door", "gem", "empty"]
    formula = "(F c0)", 5, "task0: visit({1})".format(*items)
    # formula = "F(c0 & F(c1))", 5, "task3: seq_visit({0}, {1})".format(*items)
    kwargs = {
        "formula": formula,
        "render_mode": "rgb_array",
        "state_type": "symbolic",
        "train": False,
        "use_dfa_state": False,
        "random_start": True,
    }
    return kwargs

def setup_visual_minecraft():
    env_id = "VisualMinecraft-v0"
    _env = gym.make(
        env_id,
        **_get_args_kwargs_visual_minecraft()
    )
    return _env

# class ObsToFloat(Transform):
#     # def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
#     #     next_tensordict["observation"] = next_tensordict["observation"].float()
#     #     return next_tensordict

#     def __init__(self):
#         super().__init__(in_keys=["observation"], out_keys=["observation"])

#     def _apply_transform(self, obs):
#         return obs.float()

#     # def transform_output_spec(self, output_spec: Composite) -> Composite:
#     #     return output_spec.replace(observation=output_spec["observation"].float())


def setup_visual_minecraft_with_wrapper(device: torch.device = torch.device("cpu")):
    env_id = "VisualMinecraft-v0"
    categorical_action_encoding = True
    _env = GymEnv(
        env_id,
        categorical_action_encoding=categorical_action_encoding,
        **_get_args_kwargs_visual_minecraft(),
        device=device,
    )
    _env = _env.append_transform(
        DTypeCastTransform(dtype_out=torch.float32, dtype_in=torch.int64, in_keys=["observation"])
    )

    return _env


def setup_simple_ltl_env():
    env_id = "SimpleLTLEnv-v0"
    _env = gym.make(
        env_id,
    )
    _env = LTLEnv(
        _env,
        progression_mode="full",
        ltl_sampler="Sequence_2_4",
        intrinsic=0.0,
    )
    torchrl_env = GymWrapper(_env)

    return torchrl_env
