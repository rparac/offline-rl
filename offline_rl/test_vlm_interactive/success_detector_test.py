"""
The paper `Vision-Language Models as a Source of Rewards`
computes a binary reward to highlight if a task is solved in the current frame.

We can use that in lieu of the labeling function (binary reward for sub-task completion).
To do this, we need to:
 1. Define a goal prompt
 2. Define negative prompts
 3. Compute cosine similarity between the goal prompt and the negative prompts
 4. Check if softmax is greater than a threshold
"""

from PIL import Image
import numpy as np
import torch

from offline_rl.vlm.success_detector import compute_similarities, load_similarity_model


goal_prompt = "robot is on top of a pickaxe"
negative_prompts = [
    "robot is on top of an empty cell",
    "robot is on top of a gem",
    "robot is on top of a door",
    "robot is on top of a lava",
]

batch_size = 1
num_workers = 1


image = Image.open("debug_env_images/episode_0_step_0.png")
# Convert PIL image to numpy array, then to tensor
# Format: (H, W, C) -> (1, H, W, C) with batch dimension
image_array = np.array(image)
# Convert to tensor: (H, W, C) -> (1, H, W, C) with uint8 dtype
frames = torch.from_numpy(image_array).unsqueeze(0).to(torch.uint8)


similarity_model = load_similarity_model(
    model_name="ViT-L-14/laion2b_s32b_b82k",
    goal_prompt=goal_prompt,
    negative_prompts=negative_prompts,
    cache_dir="vlm/cache",
)


similarity_model = similarity_model.to(torch.device("cuda"))

similarities = compute_similarities(similarity_model, frames, batch_size, num_workers, worker_frames_tensor=None)
print(similarities)