import os
from pathlib import Path
import torch
from tensordict import TensorDict
from PIL import Image
import numpy as np

from offline_rl.torchrl_practice.env_util import setup_visual_minecraft_with_wrapper

def save_render_image(env, output_dir: Path, step_idx: int, episode: int = 0):
    """Render the environment and save the image."""
    img = env.render()
    image = Image.fromarray(img)

    # Save image
    filename = output_dir / f"episode_{episode}_step_{step_idx}.png"
    image.save(filename)
    print(f"Saved image: {filename}")
    return filename


def main() -> None:
    # Create output directory for images
    output_dir = Path("debug_env_images")
    output_dir.mkdir(exist_ok=True)
    print(f"Images will be saved to: {output_dir.absolute()}")
    
    # Create environment using setup_visual_minecraft_with_wrapper
    env = setup_visual_minecraft_with_wrapper(device=torch.device("cpu"))

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

    # Reset the environment (TorchRL returns TensorDict)
    td = env.reset()
    obs = td["observation"]
    info = td.get("info", {})
    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()
    
    step_idx = 0
    episode = 0
    # Save initial render
    save_render_image(env, output_dir, step_idx, episode)
    
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
                td = env.reset()
                obs = td["observation"]
                info = td.get("info", {})
                if isinstance(obs, torch.Tensor):
                    obs = obs.cpu().numpy()
                terminated = False
                truncated = False
                step_idx = 0
                episode += 1
                print("Environment reset.")
                # Save render after reset
                save_render_image(env, output_dir, step_idx, episode)
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

            # Step the environment (TorchRL expects TensorDict with "action" key)
            action_tensor = torch.tensor(action, dtype=torch.long)
            action_td = TensorDict({"action": action_tensor}, batch_size=[])
            td = env.step(action_td)
            
            # Extract values from TensorDict
            # TorchRL typically structures step results with "next" key
            if "next" in td.keys():
                next_td = td["next"]
                obs = next_td.get("observation", td.get("observation"))
                reward = next_td.get("reward", torch.tensor(0.0))
                terminated = next_td.get("terminated", torch.tensor(False))
                truncated = next_td.get("truncated", torch.tensor(False))
                info = next_td.get("info", {})
            else:
                # Fallback if structure is different
                obs = td.get("observation")
                reward = td.get("reward", torch.tensor(0.0))
                terminated = td.get("terminated", torch.tensor(False))
                truncated = td.get("truncated", torch.tensor(False))
                info = td.get("info", {})
            
            # Convert tensors to Python types
            if isinstance(reward, torch.Tensor):
                reward = reward.item() if reward.numel() == 1 else float(reward)
            if isinstance(terminated, torch.Tensor):
                terminated = terminated.item() if terminated.numel() == 1 else bool(terminated)
            if isinstance(truncated, torch.Tensor):
                truncated = truncated.item() if truncated.numel() == 1 else bool(truncated)
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().numpy()
                # Remove batch dimension if present
                if obs.ndim > 1 and obs.shape[0] == 1:
                    obs = obs.squeeze(0)
            
            step_idx += 1

            # Save render after each step
            save_render_image(env, output_dir, step_idx, episode)

            print_step_result(step_idx, action_name, reward, terminated, truncated, info)
            print(obs)
            

    finally:
        env.close()


if __name__ == "__main__":
    main()


