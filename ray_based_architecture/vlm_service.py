"""
Environment change:
 - observation becomes a dictionary
    - "image_embedding" is the CLIP embedding of the observation
    - "rm_state" current RM state

  - next observation could not be computed without a VLM
    - let's move its computation outside the VLM service and inside a env wrapper
"""


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
    num_replicas=1,
    ray_actor_options={"num_gpus": 1}, 
    max_ongoing_requests=1000  # Large queue since VLM is now very fast
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

    async def add_to_buffer_with_labeling(self, observations, actions, rewards, next_observations, 
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
            # dtype=torch.float32,
            non_blocking=True,
        )

        with torch.no_grad():
            similarities = self.similarity_model.forward(embedding_tensor)
            labels = self.similarity_model.labels_from_similarities(similarities)
            # if labels[:, 0].item():
            #     print("pickaxe")
            # elif labels[:, 1].item():
            #     print("gem")
            # elif labels[:, 2].item():
            #     print("door")
            # elif labels[:, 3].item():
            #     print("lava")

        custom_rewards = labels[:, 0].detach().float().cpu().numpy()

        assert custom_rewards.shape == rewards.shape
        assert (rewards == custom_rewards).all()

        inference_time_ms = (time.time() - start_time) * 1000
        throughput = batch_size / (inference_time_ms / 1000) if inference_time_ms > 0 else 0
        self.batch_size_gauge.set(batch_size)
        self.throughput_gauge.set(throughput)

        # Store CLIP embeddings in replay buffer for fast sampling
        self.replay_buffer_handle.add_batch.remote(
            observations=observations,
            next_observations=next_observations,
            actions=np.asarray(actions, dtype=np.int64),
            rewards=custom_rewards,
            terminateds=np.asarray(terminateds, dtype=np.bool_),
            truncateds=np.asarray(truncateds, dtype=np.bool_),
        )

        # Return is optional for training loops; kept for debugging/metrics.
        return custom_rewards

    
# This starts the "cluster" of consumers if run directly
if __name__ == "__main__":
    serve.run(VLMService.bind())