import os
import gymnasium
from pathlib import Path
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
import torch
from gymnasium.wrappers import DtypeObservation
from PIL import Image
import numpy as np

from env.visual_minecraft.env import GridWorldEnv
from offline_rl.vlm.open_clip.transform import image_transform
from offline_rl.vlm.success_detector import compute_similarities, load_similarity_model
    
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
    """
    goal_prompt: pickaxe missing
    negative_prompts: [
        all_cells_empty
        diamond not visible
        door not visible
        lava not visible
    ]

    """



    context = "A screenshot of a 2D pygame grid world. "
    goal_prompt = context + "A robot character standing over and partially hiding a pickaxe icon on a white background."
    negative_prompts = [
        context + "A robot character standing alone on a white tile with no other objects",
        context + "A robot character standing over and partially hiding a blue diamond gem icon on a white background",
        context + "A robot character standing over and partially hiding a red door icon on a white background",
        context + "A robot character standing on top of orange magma texture",
    ]

    goal_prompt = context + "A robot, blue diamond gem, opened double door, and orange magma texture are clearly visible in the image."
    negative_prompts = [
        context + "A robot, a pickaxe, blue diamond gem, opened double door, and orange magma texture are clearly visible in the image. ",
        context + "A robot, a pickaxe, opened double door, and orange magma texture are clearly visible in the image.",
        context + "A robot, a pickaxe, blue diamond gem, and orange magma texture are clearly visible in the image.",
        context + "A robot, a pickaxe, blue diamond gem, and opened double door are clearly visible in the image.",
    ]
    goal_prompt = "Grey and yellow pickaxe"
    negative_prompts = [
        "Blue diamond gem",
        "Open red double door",
        "Orange and yellow magma texture",
    ]


    # goal_prompt = "A robot standing on a square containing a pickaxe."
    # negative_prompts = [
    #     "A robot standing on a square containing a diamond gem.",
    #     "A robot standing on a square containing a red door.",
    #     "A robot standing on a square containing orange magma texture.",
    #     "A robot standing on an empty white square.",
    # ]



    model_name = "ViT-L-14/openai"
    # model_name = "ViT-B-32/openai"
    # model_name = "ViT-bigG-14/laion2b_s39b_b160k"
    
    similarity_model = load_similarity_model(
        model_name=model_name,
        goal_prompt=goal_prompt,
        negative_prompts=negative_prompts,
        cache_dir="/data/private/rp218/open_clip",
    )
    similarity_model = similarity_model.to(torch.device("cuda"))

    return similarity_model, goal_prompt, negative_prompts

def _compute_similarities(similarity_model, frames):
    similarities = compute_similarities(similarity_model, frames, batch_size, num_workers, worker_frames_tensor=None)
    return similarities

def main() -> None:
    similarity_model, goal_prompt, negative_prompts = _initialize_similarity_model()
    
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
    print(f"\nGoal prompt: {goal_prompt}")
    print(f"Negative prompts: {negative_prompts}")

    def print_step_result(step_idx, action_name, similarities, terminated, truncated, info_dict):
        label = info_dict.get("label", None) if info_dict else None
        # similarities shape: (1, num_prompts) where first is goal, rest are negatives
        goal_similarity = similarities[0, 0].item()
        print(
            f"[step {step_idx}] action={action_name}, "
            f"{goal_prompt}={goal_similarity:.4f}, terminated={terminated}, truncated={truncated}, label={label}"
        )
        # Optionally print all similarities
        print(f"  {goal_prompt}: {goal_similarity:.4f}")
        for i, neg_prompt in enumerate(negative_prompts):
            neg_similarity = similarities[0, i + 1].item()
            print(f"    {neg_prompt}: {neg_similarity:.4f}")
        print()
    
    def print_episode_summary(episode_num, episode_similarities):
        """Print min and second smallest values for each prompt at the end of an episode."""
        print(f"\n{'='*60}")
        print(f"Episode {episode_num} Summary - Min/Second Smallest Similarity Values")
        print(f"{'='*60}")
        for prompt, values in episode_similarities.items():
            if len(values) > 0:
                min_val = min(values)
                if len(values) > 1:
                    # Get second smallest by sorting and taking the second element
                    sorted_values = sorted(values)
                    second_smallest = sorted_values[1]
                    print(f"  {prompt}:")
                    print(f"    Min: {min_val:.4f}")
                    print(f"    Second Smallest: {second_smallest:.4f}")
                else:
                    print(f"  {prompt}:")
                    print(f"    Min: {min_val:.4f}")
                    print(f"    Second Smallest: N/A (only one value)")
            else:
                print(f"  {prompt}: No values recorded")
        print(f"{'='*60}\n")

    # Reset the environment (gymnasium returns tuple)
    obs, info = env.reset()
    step_idx = 0
    episode = 0
    _visualize_image(obs, episode=episode, step_idx=step_idx)
    similarities = _compute_similarities(similarity_model, obs)
    goal_similarity = similarities[0, 0].item()
    print(f"VLM {goal_prompt} is {goal_similarity:.4f}")
    # Save initial render
    
    terminated = False
    truncated = False
    
    # Track similarity values for each prompt during the episode
    episode_similarities = {
        goal_prompt: [],
        **{prompt: [] for prompt in negative_prompts}
    }
    
    # Track initial similarities
    episode_similarities[goal_prompt].append(similarities[0, 0].item())
    for i, neg_prompt in enumerate(negative_prompts):
        episode_similarities[neg_prompt].append(similarities[0, i + 1].item())

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
                # Reset tracking for new episode
                episode_similarities = {
                    goal_prompt: [],
                    **{prompt: [] for prompt in negative_prompts}
                }
                similarities = _compute_similarities(similarity_model, obs)
                goal_similarity = similarities[0, 0].item()
                print(f"VLM {goal_prompt} is {goal_similarity:.4f}")
                # Track initial similarities after reset
                episode_similarities[goal_prompt].append(similarities[0, 0].item())
                for i, neg_prompt in enumerate(negative_prompts):
                    episode_similarities[neg_prompt].append(similarities[0, i + 1].item())
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
            _visualize_image(obs, episode=episode, step_idx=step_idx)
            similarities = _compute_similarities(similarity_model, obs)
            goal_similarity = similarities[0, 0].item()
            print(f"VLM {goal_prompt} is {goal_similarity:.4f}")
            
            # Track similarities for this step
            episode_similarities[goal_prompt].append(similarities[0, 0].item())
            for i, neg_prompt in enumerate(negative_prompts):
                episode_similarities[neg_prompt].append(similarities[0, i + 1].item())
            
            # Convert tensors to Python types if needed
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().numpy()
                # Remove batch dimension if present
                if obs.ndim > 1 and obs.shape[0] == 1:
                    obs = obs.squeeze(0)
            
            step_idx += 1

            print_step_result(step_idx, action_name, similarities, terminated, truncated, info)
            
            # Print episode summary if episode ended
            if terminated or truncated:
                print_episode_summary(episode, episode_similarities)
            

    finally:
        env.close()


if __name__ == "__main__":
    main()
