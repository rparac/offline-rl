import gymnasium as gym
from tensordict import TensorDictBase
import torch
from torchrl.data import Composite
from torchrl.envs.transforms import Transform
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import DTypeCastTransform, GymWrapper, RewardSum, StepCounter, TransformedEnv, step_mdp
import numpy as np


from env.visual_minecraft.env import GridWorldEnv
# Optional dependency (only used by `setup_simple_ltl_env`).
try:
    from env.ltl_wrappers import LTLEnv
except Exception:  # pragma: no cover
    LTLEnv = None  # type: ignore[assignment]
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
        "random_start": False,
    }
    return kwargs

def setup_visual_minecraft(image_env: bool = False, use_dfa_state: bool = False):
    env_id = "VisualMinecraft-v0"
    kwargs = _get_args_kwargs_visual_minecraft()
    if image_env:
        kwargs["state_type"] = "image"
        kwargs["normalize_env"] = False
        kwargs["random_start"] = True
    if use_dfa_state:
        kwargs["use_dfa_state"] = True
        items = ["pickaxe", "lava", "door", "gem", "empty"]
        kwargs["formula"] = "F(c0 & F(c1))", 5, "task3: seq_visit({0}, {1})".format(*items)
    _env = gym.make(
        env_id,
        **kwargs
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

def setup_frozen_lake_with_wrapper(device: torch.device = torch.device("cpu")):
    env_id = "FrozenLake-v1"
    map_name = "4x4"
    categorical_action_encoding = True
    _env = GymEnv(
        env_id,
        categorical_action_encoding=categorical_action_encoding,
        map_name=map_name,
        device=device,
    )
    _env = _env.append_transform(
        DTypeCastTransform(dtype_out=torch.float32, dtype_in=torch.int64, in_keys=["observation"])
    )
    # Used for logging purposes
    _env = _env.append_transform(
        RewardSum()
    )
    _env = _env.append_transform(
        StepCounter()
    )

    return _env


def setup_pendulum_with_wrapper(device: torch.device = torch.device("cpu")):
    env_id = "Pendulum-v1"
    _env = GymEnv(
        env_id,
        device=device,
        render_mode="rgb_array",
    )
    # Used for logging purposes
    _env = _env.append_transform(
        RewardSum()
    )
    _env = _env.append_transform(
        StepCounter()
    )
    return _env


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
    # Used for logging purposes
    _env = _env.append_transform(
        RewardSum()
    )
    _env = _env.append_transform(
        StepCounter()
    )

    return _env


def setup_simple_ltl_env():
    if LTLEnv is None:
        raise ImportError("LTLEnv could not be imported. Check `env/ltl_wrappers.py` and your PYTHONPATH.")
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

def record_policy_video(policy, max_steps=10, seed=42, device=None, render_env_fn=setup_visual_minecraft_with_wrapper):
    """
    Record a policy rollout as a video and return frames for logging.
    
    Args:
        policy: The policy to evaluate
        max_steps: Maximum number of steps in the rollout
        seed: Random seed for reproducibility
        device: Device to run policy on
        
    Returns:
        frames: Tensor of shape (T, C, H, W) representing video frames
    """
    
    render_env = render_env_fn(device=device)

    td = render_env.reset()

    # Img is a numpy array of shape (512, 512, 3)
    imgs = []
    img = render_env.render()
    imgs.append(img)
    
    done = False
    step_count = 0

    cumulative_reward = 0.0
    
    while not done and step_count < max_steps:
        device = td.device
        
        # Get action from policy using TensorDict format
        with torch.no_grad():
            td = policy(td.to(device))
        
        # Step environment
        td = render_env.step(td)

        # TorchRL convention: the post-step observation/reward/done live under the "next" key.
        # Move ("next", ...) entries to the root so the next policy call sees the updated state.
        cumulative_reward += float(td["next", "reward"].item())
        done = bool(td["next", "terminated"].item() or td["next", "truncated"].item())
        img = render_env.render()
        imgs.append(img)
        step_count += 1

        td = step_mdp(td, keep_other=False)
    
    render_env.close()

    # Convert frames to tensor format expected by TorchRL logger
    # Shape: (T, H, W, C) -> (T, C, H, W)
    frames_array = np.stack(imgs)  # Shape: (T, H, W, C)
    frames_array = frames_array.transpose(0, 3, 1, 2)  # Shape: (T, C, H, W)
    return frames_array, cumulative_reward

