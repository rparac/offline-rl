import ray
from ray import serve
import numpy as np
import torch
import warnings

from offline_rl.vlm.visual_minecraft_success_detector import compute_visual_minecraft_similarities, load_visual_minecraft_similarity_model

# Ignore warnings about PyTorch not being writable, it is safe in this case
warnings.filterwarnings("ignore", category=UserWarning, message=".*not writable.*PyTorch.*")

# TODO: Need to pass the prompts ourselves

def _initialize_similarity_model():
    model_name = "ViT-L-14/openai"
    
    similarity_model = load_visual_minecraft_similarity_model(
        model_name=model_name,
        cache_dir="/data/private/rp218/open_clip",
    )
    similarity_model = similarity_model.to(torch.device("cuda"))

    return similarity_model


# Define the deployment (Consumer)
@serve.deployment(
    num_replicas=3,          # Use 3 GPUs (Worker Pool)
    ray_actor_options={"num_gpus": 1}, 
    max_ongoing_requests=100 # How many requests can sit in the queue per replica
)
class VLMService:
    def __init__(self):
        # Ray sets CUDA_VISIBLE_DEVICES, so "cuda" points to the assigned GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading VLM Model on {self.device}...")
        self.similarity_model = _initialize_similarity_model()
        self.similarity_model.to(self.device)
    
    # Batch of images is a list of observations (type is observation space)
    # Enable automatic batching
    @serve.batch(
        max_batch_size=32, 
        # batch_wait_timeout_s=0.1
    )
    async def compute_reward(self, batch_of_images) -> torch.Tensor:
        """
        Ray Serve automatically collects 'batch_of_images' from multiple agents.
        """
        # Run inference on the whole batch
        # rewards = self.model(batch_of_images)
        breakpoint()
        images_only = [obs['image'] for obs in batch_of_images]
        img_tensor = torch.stack([torch.from_numpy(image) for image in images_only])
        img_tensor = img_tensor.to(self.device, non_blocking=True).half()

        with torch.no_grad():
            labels = self.similarity_model.compute_labels(img_tensor)
        
        return labels.cpu()

    async def __call__(self, image_input):
        # This acts as the entry point for a single request
        return await self.compute_reward(image_input)

# This starts the "cluster" of consumers if run directly
if __name__ == "__main__":
    serve.run(VLMService.bind())