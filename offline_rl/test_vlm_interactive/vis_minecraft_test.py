import os
import gymnasium
from pathlib import Path
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
import torch
from gymnasium.wrappers import DtypeObservation
from PIL import Image
import numpy as np

from env.visual_minecraft.fixed_len_env import GridWorldEnv
from offline_rl.vlm.open_clip.transform import image_transform
from offline_rl.vlm.success_detector import compute_similarities, load_similarity_model
from offline_rl.vlm.visual_minecraft_success_detector import compute_visual_minecraft_labels, compute_visual_minecraft_rewards, load_visual_minecraft_similarity_model, prompts_with_thresholds
    
batch_size = 1
num_workers = 1

class ImageObsWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(3, 224, 224), dtype=np.uint8)
    
    def observation(self, obs):
        obs = np.clip(obs * 255.0, a_min=0, a_max=255).astype(np.uint8)
        return torch.from_numpy(obs).unsqueeze(0).to(torch.uint8)

def setup_visual_minecraft(device: torch.device = torch.device("cpu")):
    env_id = "FixedLenVisualMinecraft-v0"

    items = ["pickaxe", "lava", "door", "gem", "empty"]
    formula = "(F c0)", 5, "task0: visit({1})".format(*items)
    kwargs = {
        "episode_length": 10,
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

def _visualize_image(obs, output_dir="debug_env_images", episode=0, step_idx=0):
    """
    Temporary. Visualize image that is passed to CLIP
    """
    transform = image_transform(image_size=224)
    std = OPENAI_DATASET_STD
    mean = OPENAI_DATASET_MEAN
    # output = (input - mean) / std
    # input = output * std + mean

    with torch.no_grad():
        transformed_obs = transform(obs).squeeze(0)
        for i in range(3):
            transformed_obs[i] = transformed_obs[i] * std[i] + mean[i]
        # Clamp to [0, 1] range
        transformed_obs = torch.clamp(transformed_obs, 0.0, 1.0)
        # Convert from (C, H, W) to (H, W, C) for PIL
        transformed_obs = transformed_obs.permute(1, 2, 0)
        # Convert to numpy array
        transformed_obs_np = transformed_obs.cpu().numpy()
        # Convert to PIL Image (expects values in [0, 1] for float)
        pil_image = Image.fromarray((transformed_obs_np * 255).astype(np.uint8))
        
        # Save the image
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        save_path = output_path / f"transformed_episode_{episode}_step_{step_idx}.png"
        pil_image.save(save_path)
        print(f"Saved transformed image to: {save_path}")
    
    return transformed_obs



def _initialize_similarity_model():
    model_name = "ViT-L-14/openai"
    
    similarity_model = load_visual_minecraft_similarity_model(
        model_name=model_name,
        cache_dir="/data/private/rp218/open_clip",
    )
    similarity_model = similarity_model.to(torch.device("cuda"))

    return similarity_model, list(prompts_with_thresholds.keys())

def _compute_labels(similarity_model, frames):
    labels = compute_visual_minecraft_labels(similarity_model, frames, batch_size, num_workers, worker_frames_tensor=None)
    return labels


def main() -> None:
    similarity_model, prompts = _initialize_similarity_model()
    
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
    print(f"\nPrompts: {prompts}")

    def print_step_result(step_idx, action_name, predicted_labels, terminated, truncated, info_dict):
        label = info_dict.get("label", None) if info_dict else None

        predicted_labels = predicted_labels.cpu().numpy().squeeze(0)
        np_prompts = np.array(prompts)
        predicted_prompts = np_prompts[predicted_labels]
        # similarities shape: (1, num_prompts) where first is goal, rest are negatives
        print(
            f"[step {step_idx}] action={action_name}, "
            f"terminated={terminated}, truncated={truncated}, gt_label={label}"
        )
        print(f"Predicted: ")
        for predicted_prompt in predicted_prompts:
            print(f"    {predicted_prompt}")
        print()

    # Reset the environment (gymnasium returns tuple)
    terminated = False
    truncated = False
    obs, info = env.reset()
    step_idx = 0
    episode = 0
    _visualize_image(obs, episode=episode, step_idx=step_idx)
    predicated_labels = _compute_labels(similarity_model, obs)
    print_step_result(step_idx, "RESET", predicated_labels, terminated, truncated, info)

    # Save initial render
    
    
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
                predicated_labels = _compute_labels(similarity_model, obs)
                print_step_result(step_idx, "RESET", predicated_labels, terminated, truncated, info)
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
            # obs_new = env.render()
            # from PIL import Image
            # Image.fromarray(obs_new).save(f"debug_env_images/2_episode_{episode}_step_{step_idx}.png")
            _visualize_image(obs, episode=episode, step_idx=step_idx)
            predicated_labels = _compute_labels(similarity_model, obs)
            vlm_predicted_reward = compute_visual_minecraft_rewards(similarity_model, obs, batch_size, num_workers, worker_frames_tensor=None)
            print(f"VLM predicted reward: {vlm_predicted_reward}")
            step_idx += 1

            print_step_result(step_idx, action_name, predicated_labels, terminated, truncated, info)
            
            # Print episode summary if episode ended
            if terminated or truncated:
                print("Episode finished.")
            

    finally:
        env.close()


if __name__ == "__main__":
    main()
