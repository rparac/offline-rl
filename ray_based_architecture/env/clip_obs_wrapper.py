"""
Encode all observations using CLIP image encoder.
"""

import gymnasium as gym
import numpy as np
import torch

from ray_based_architecture.vlm_service import initialize_similarity_model

# TODO: make "cuda" a parameter in refactoring

class BatchCLIPObsWrapper(gym.vector.VectorObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        output_shape = (768,) # CLIP image encoder size
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
        # observation is a numpy array of shape (3, 224, 224)
        observation = torch.from_numpy(observation).to(
            self.device,
            dtype=torch.float16,
            non_blocking=True
        )
        with torch.no_grad():
            obs = self._similarity_model.embed_images(observation)
        out = obs.cpu().numpy()
        return out