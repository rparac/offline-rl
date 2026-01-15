"""
Encode all observations using CLIP image encoder.
"""

import gymnasium as gym
import numpy as np
import torch

from ray_based_architecture.vlm_service import initialize_similarity_model

# TODO: make "cuda" a parameter in refactoring


class BatchCLIPObsWrapper(gym.vector.VectorObservationWrapper):
    """
    CLIP observation wrapper for vectorized environments.
    Works with any number of environments (including num_envs=1 for testing).
    """
    
    def __init__(self, env, device="cuda"):
        super().__init__(env)

        output_shape = (768,)  # CLIP image encoder size
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=output_shape,
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.num_envs, *output_shape),
            dtype=np.float32
        )

        self.device = "cuda"
        self._similarity_model = initialize_similarity_model()
        self._similarity_model.to(self.device).eval()

    def observations(self, observation):
        """Embed a batch of observations (num_envs, H, W, C) → (num_envs, 768)"""
        observation = np.transpose(observation, (0, 3, 1, 2))

        obs_tensor = torch.from_numpy(observation).to(
            self.device,
            dtype=torch.uint8,
            non_blocking=True
        )
        with torch.no_grad():
            embeddings = self._similarity_model.embed_images(obs_tensor)
        return embeddings.cpu().numpy()