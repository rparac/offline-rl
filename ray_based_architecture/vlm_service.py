import ray
from ray import serve
from ray.util.metrics import Counter, Gauge, Histogram
import numpy as np
import torch
import warnings
import time

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
    max_ongoing_requests=500 # How many requests can sit in the queue per replica
)
class VLMService:
    def __init__(self):
        # Ray sets CUDA_VISIBLE_DEVICES, so "cuda" points to the assigned GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading VLM Model on {self.device}...")
        self.similarity_model = _initialize_similarity_model()
        self.similarity_model.to(self.device)
        
        # VLM-specific metrics (Ray already provides latency and request metrics)
        self.batch_size_gauge = Gauge(
            "vlm_batch_size",
            description="Current batch size (images per batch)"
        )
        self.throughput_gauge = Gauge(
            "vlm_throughput",
            description="Current throughput in images/sec"
        )
        
        # For periodic console logging
        self.batch_count = 0
        self.recent_batch_sizes = []
        self.recent_inference_times = []
    
    # Batch of images is a list of observations (type is observation space)
    # Enable automatic batching
    @serve.batch(
        max_batch_size=128,  # Increased batch size for better GPU utilization
        batch_wait_timeout_s=0.01  # Don't wait too long to fill batches
    )
    async def compute_reward(self, batch_of_images) -> torch.Tensor:
        """
        Ray Serve automatically collects 'batch_of_images' from multiple agents.
        """
        batch_size = len(batch_of_images)
        start_time = time.time()
        
        # Run inference on the whole batch
        images_only = [obs['image'] for obs in batch_of_images]
        img_tensor = torch.stack([torch.from_numpy(image) for image in images_only])
        img_tensor = img_tensor.to(self.device, non_blocking=True).half()

        with torch.no_grad():
            labels = self.similarity_model.compute_labels(img_tensor)
        
        # Calculate and record VLM-specific metrics
        inference_time_ms = (time.time() - start_time) * 1000
        throughput = batch_size / (inference_time_ms / 1000) if inference_time_ms > 0 else 0
        
        self.batch_size_gauge.set(batch_size)
        self.throughput_gauge.set(throughput)
        
        # Console logging (every 50 batches)
        self.batch_count += 1
        self.recent_batch_sizes.append(batch_size)
        self.recent_inference_times.append(inference_time_ms)
        
        if self.batch_count % 50 == 0:
            avg_batch_size = np.mean(self.recent_batch_sizes[-50:])
            avg_inference_time = np.mean(self.recent_inference_times[-50:])
            avg_throughput = avg_batch_size / (avg_inference_time / 1000)
            print(f"[GPU {self.device}] Batches: {self.batch_count} | "
                  f"Avg batch size: {avg_batch_size:.1f} | "
                  f"Avg inference time: {avg_inference_time:.1f}ms | "
                  f"Throughput: {avg_throughput:.1f} img/s")
        
        return labels.cpu()

    async def __call__(self, image_input):
        # This acts as the entry point for a single request
        return await self.compute_reward(image_input)

# This starts the "cluster" of consumers if run directly
if __name__ == "__main__":
    serve.run(VLMService.bind())