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

def _extract_image(obs):
    """
    Normalize an observation into a raw image array.

    Supports:
    - dict obs with an "image" key (common in wrapper-based envs)
    - raw numpy arrays (e.g., CHW uint8 from vectorized envs)
    """
    if isinstance(obs, dict):
        if "image" not in obs:
            raise KeyError(f"Expected dict observation to contain key 'image', got keys={list(obs.keys())}")
        return obs["image"]
    return obs

def initialize_similarity_model():
    model_name = "ViT-L-14/openai"
    
    similarity_model = load_visual_minecraft_similarity_model(
        model_name=model_name,
        cache_dir="/data/private/rp218/open_clip",
    )
    similarity_model = similarity_model.to(torch.device("cuda"))
    similarity_model = similarity_model.half()

    return similarity_model


# Define the deployment (Consumer)
@serve.deployment(
    num_replicas=3,
    ray_actor_options={"num_gpus": 1}, 
    max_ongoing_requests=500  # Large queue to keep GPU fed
)
class VLMService:
    def __init__(self, replay_buffer_name: str, replay_buffer_namespace: str):
        # Ray sets CUDA_VISIBLE_DEVICES, so "cuda" points to the assigned GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading VLM Model on {self.device}...", flush=True)
        self.similarity_model = initialize_similarity_model()
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

        # Avoid passing ActorHandles into Serve replicas (can hit deserialization bugs).
        # Instead, look it up by (name, namespace) from inside the replica process.
        try:
            self.replay_buffer_handle = ray.get_actor(replay_buffer_name, namespace=replay_buffer_namespace)
            print(f"[VLMService.__init__] Successfully found replay buffer '{replay_buffer_name}' in namespace '{replay_buffer_namespace}'", flush=True)
        except Exception as e:
            print(f"[VLMService.__init__] FATAL: Could not find replay buffer '{replay_buffer_name}' in namespace '{replay_buffer_namespace}': {e}", flush=True)
            raise

    # One request == one transition. Ray Serve batches requests across producers.
    @serve.batch(
        max_batch_size=128,
        batch_wait_timeout_s=0.05  # Short timeout to keep GPU busy
    )
    async def label_reward(self, observations, actions, next_observations, terminateds, truncateds):
        """
        Args are lists (one per request) due to Ray Serve batching:
        - observations: list[Any]
        - actions: list[Any]
        - next_observations: list[Any]
        - terminateds: list[bool]
        - truncateds: list[bool]
        """
        batch_size = len(next_observations)
        start_time = time.time()
        
        # Convert obs to raw images (supports dict obs or raw arrays)
        obs_images = [_extract_image(o) for o in observations]
        next_images = [_extract_image(o) for o in next_observations]
        
        # Stack numpy arrays into a single numpy array first (faster)
        obs_numpy = np.stack(obs_images, axis=0)
        next_numpy = np.stack(next_images, axis=0)
        
        # Convert to torch tensor and transfer to GPU in one go
        # Using pin_memory=True for faster CPU->GPU transfer
        img_tensor = torch.from_numpy(next_numpy).to(
            self.device, 
            dtype=torch.float16,  # Convert to half directly
            non_blocking=True
        )

        with torch.no_grad():
            labels = self.similarity_model.compute_labels(img_tensor)

        # Ensure we pass numpy arrays into the replay buffer actor.
        rewards = labels[:, 0].detach().float().cpu().numpy()
        
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
            avg_batch_size = np.mean(self.recent_batch_sizes)
            avg_inference_time = np.mean(self.recent_inference_times)
            avg_throughput = avg_batch_size / (avg_inference_time / 1000)
            print(f"[GPU {self.device}] Batches: {self.batch_count} | "
                  f"Avg batch size: {avg_batch_size:.1f} | "
                  f"Avg inference time: {avg_inference_time:.1f}ms | "
                  f"Throughput: {avg_throughput:.1f} img/s")
            self.recent_batch_sizes = []
            self.recent_inference_times = []
        
        # Fire-and-forget: write the whole batch into the replay buffer.
        # NOTE: Serve expects one return value per request; we return rewards per request below.
        self.replay_buffer_handle.add_batch.remote(
            observations=obs_numpy,
            next_observations=next_numpy,
            actions=np.asarray(actions, dtype=np.int64),
            rewards=rewards,
            terminateds=np.array(terminateds, dtype=np.bool_),
            truncateds=np.array(truncateds, dtype=np.bool_),
        )

        # One return per original request
        return list(rewards)

    async def add_to_buffer_with_labeling(self, observations, actions, next_observations, 
                                            terminateds, truncateds):
        """
        High-throughput batched endpoint (vector-env friendly).
        
        Reuses CLIP embeddings from BatchCLIPObsWrapper for both agent and reward computation.
        
        Args:
            observations: CLIP embeddings of observations (batch_size, 768)
            actions: actions taken (batch_size,)
            next_observations: CLIP embeddings of next observations (batch_size, 768)
            terminateds: terminal flags (batch_size,)
            truncateds: truncation flags (batch_size,)
        """
        start_time = time.time()
        batch_size = int(next_observations.shape[0])

        # Compute rewards from CLIP embeddings (already computed by BatchCLIPObsWrapper)
        embedding_tensor = torch.from_numpy(next_observations).to(
            self.device,
            dtype=torch.float16,
            non_blocking=True,
        )

        with torch.no_grad():
            similarities = self.similarity_model.forward(embedding_tensor)
            labels = self.similarity_model.labels_from_similarities(similarities)

        rewards = labels[:, 0].detach().float().cpu().numpy()

        inference_time_ms = (time.time() - start_time) * 1000
        throughput = batch_size / (inference_time_ms / 1000) if inference_time_ms > 0 else 0
        self.batch_size_gauge.set(batch_size)
        self.throughput_gauge.set(throughput)

        # Store CLIP embeddings in replay buffer for fast sampling
        try:
            ref = self.replay_buffer_handle.add_batch.remote(
                observations=observations,
                next_observations=next_observations,
                actions=np.asarray(actions, dtype=np.int64),
                rewards=rewards,
                terminateds=np.asarray(terminateds, dtype=np.bool_),
                truncateds=np.asarray(truncateds, dtype=np.bool_),
            )
            # Don't await - fire and forget. If this fails, we'll see it in Ray logs.
        except Exception as e:
            print(f"[VLMService] ERROR calling add_batch.remote: {e}", flush=True)
            raise

        # Return is optional for training loops; kept for debugging/metrics.
        return rewards

    
    # TODO: Delete this function if it is unused
    # Batch of images is a list of observations (type is observation space)
    # Enable automatic batching
    @serve.batch(
        max_batch_size=128,
        batch_wait_timeout_s=0.05  # Short timeout to keep GPU busy
    )
    async def compute_reward(self, batch_of_images) -> torch.Tensor:
        """
        Ray Serve automatically collects 'batch_of_images' from multiple agents.
        """
        batch_size = len(batch_of_images)
        start_time = time.time()
        
        # Optimized data transfer: Convert to tensor more efficiently
        images_only = [obs['image'] for obs in batch_of_images]
        
        # Stack numpy arrays into a single numpy array first (faster)
        img_numpy = np.stack(images_only, axis=0)
        
        # Convert to torch tensor and transfer to GPU in one go
        # Using pin_memory=True for faster CPU->GPU transfer
        img_tensor = torch.from_numpy(img_numpy).to(
            self.device, 
            dtype=torch.float16,  # Convert to half directly
            non_blocking=True
        )

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
            avg_batch_size = np.mean(self.recent_batch_sizes)
            avg_inference_time = np.mean(self.recent_inference_times)
            avg_throughput = avg_batch_size / (avg_inference_time / 1000)
            print(f"[GPU {self.device}] Batches: {self.batch_count} | "
                  f"Avg batch size: {avg_batch_size:.1f} | "
                  f"Avg inference time: {avg_inference_time:.1f}ms | "
                  f"Throughput: {avg_throughput:.1f} img/s")
            self.recent_batch_sizes = []
            self.recent_inference_times = []
        
        return labels.cpu()

    async def __call__(self, image_input):
        # This acts as the entry point for a single request
        return await self.compute_reward(image_input)

# This starts the "cluster" of consumers if run directly
if __name__ == "__main__":
    serve.run(VLMService.bind())