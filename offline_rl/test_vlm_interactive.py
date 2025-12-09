import os
import gymnasium
from pathlib import Path
import torch
from gymnasium.wrappers import DtypeObservation
from PIL import Image
import numpy as np

from env.visual_minecraft.env import GridWorldEnv
from offline_rl.vlm.reward_model import compute_rewards, load_reward_model_from_config
from offline_rl.vlm.trainer.config import CLIPRewardConfig
    
batch_size = 1
num_workers = 1

class ImageObsWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
    
    def observation(self, obs):
        obs = np.clip(obs * 255.0, a_min=0, a_max=255).astype(np.uint8)
        return torch.from_numpy(obs).unsqueeze(0).to(torch.uint8)

def setup_visual_minecraft(device: torch.device = torch.device("cpu")):
    env_id = "VisualMinecraft-v0"

    items = ["pickaxe", "lava", "door", "gem", "empty"]
    formula = "(F c0)", 5, "task0: visit({1})".format(*items)
    kwargs = {
        "formula": formula,
        "render_mode": "human",
        "state_type": "image",
        "train": False,
        "use_dfa_state": False,
        "random_start": True,
    }
    env = gymnasium.make(env_id, **kwargs)

    env = DtypeObservation(env, dtype=np.float32)
    env = ImageObsWrapper(env)
    return env

def _initialize_reward_model():

    baseline_prompts = ["a grid world with a robot, a pickaxe, a gem, a door, and a lava"]
    target_prompts = ["a grid world with a robot, a pickaxe, a gem, a door, and a lava where the robot is trying to get the pickaxe"]
    config = CLIPRewardConfig(
        name="clip",
        pretrained_model="ViT-L-14/laion2b_s32b_b82k",
        batch_size=batch_size,
        alpha=0.1,
        target_prompts=target_prompts,
        baseline_prompts=baseline_prompts,
        cache_dir="vlm/cache",
    )

    clip_reward_model = load_reward_model_from_config(config)
    clip_reward_model = clip_reward_model.to(torch.device("cuda"))

    return clip_reward_model

def _compute_reward(clip_reward_model, frames):
    rewards = compute_rewards(clip_reward_model, frames, batch_size, num_workers, worker_frames_tensor=None)
    return rewards.item()

def main() -> None:
    # Create output directory for images
    output_dir = Path("debug_env_images")
    output_dir.mkdir(exist_ok=True)
    print(f"Images will be saved to: {output_dir.absolute()}")

    clip_reward_model = _initialize_reward_model()
    
    # Create environment using setup_visual_minecraft_with_wrapper
    env = setup_visual_minecraft(device=torch.device("cpu"))

    print("GridWorldEnv (via setup_visual_minecraft_with_wrapper) interactive session")
    print("Actions:")
    print("  w = up (2)")
    print("  s = down (0)")
    print("  d = right (1)")
    print("  a = left (3)")
    print("  r = reset episode")
    print("  q = quit")

    def print_step_result(step_idx, action_name, reward, terminated, truncated, info_dict):
        label = info_dict.get("label", None) if info_dict else None
        print(
            f"[step {step_idx}] action={action_name}, "
            f"reward={reward}, terminated={terminated}, truncated={truncated}, label={label}"
        )

    # Reset the environment (gymnasium returns tuple)
    obs, info = env.reset()
    reward = _compute_reward(clip_reward_model, obs)
    print(f"VLM reward is {reward}")

    
    step_idx = 0
    episode = 0
    # Save initial render
    
    terminated = False
    truncated = False

    try:
        while True:
            if terminated or truncated:
                print("Episode finished. Type 'r' to reset or 'q' to quit.")

            cmd = input("Command (w/a/s/d/r/q): ").strip().lower()

            if cmd == "q":
                break
            if cmd == "r":
                obs, info = env.reset()
                terminated = False
                truncated = False
                step_idx = 0
                episode += 1
                print("Environment reset.")
                # Save render after reset
                continue

            # Map commands to discrete actions used in the env
            if cmd == "w":
                action = 2  # up
                action_name = "UP"
            elif cmd == "s":
                action = 0  # down
                action_name = "DOWN"
            elif cmd == "d":
                action = 1  # right
                action_name = "RIGHT"
            elif cmd == "a":
                action = 3  # left
                action_name = "LEFT"
            else:
                print("Unknown command. Use w/a/s/d/r/q.")
                continue

            # Step the environment (gymnasium returns tuple)
            obs, reward, terminated, truncated, info = env.step(action)
            reward = _compute_reward(clip_reward_model, obs)
            print(f"VLM reward is {reward}")
            
            # Convert tensors to Python types if needed
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().numpy()
                # Remove batch dimension if present
                if obs.ndim > 1 and obs.shape[0] == 1:
                    obs = obs.squeeze(0)
            
            step_idx += 1

            print_step_result(step_idx, action_name, reward, terminated, truncated, info)
            

    finally:
        env.close()


if __name__ == "__main__":
    main()


