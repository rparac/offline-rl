import gymnasium as gym
import numpy as np
import torch

from typing import Any

from offline_rl.vlm.visual_minecraft_success_detector import idx_to_vis_minecraft_prompt
from ray_based_architecture.vlm_service import initialize_similarity_model

class BatchCLIPObsAndLabelsWrapper(gym.vector.VectorWrapper):
    """
    A vector wrapper that passes observations through unchanged,
    Kept as one wrapper to save GPU memory.
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
        self._similarity_model = self._similarity_model.half()  # Ensure half precision

    def _observations(self, observation):
        """Embed a batch of observations (num_envs, H, W, C) → (num_envs, 768)"""
        observation = np.transpose(observation, (0, 3, 1, 2))

        obs_tensor = torch.from_numpy(observation).to(
            self.device,
            dtype=torch.uint8,
            non_blocking=True
        )
        with torch.no_grad():
            embeddings = self._similarity_model.embed_images(obs_tensor)
            # Ensure embeddings match model dtype (half precision)
            embeddings = embeddings.half()
        return embeddings
        # return embeddings.cpu().numpy()

    def _compute_labels(self, embedded_obs):
        with torch.no_grad():
            similarities = self._similarity_model.forward(embedded_obs)
            print(f"Similarities: {similarities}")
            labels = self._similarity_model.labels_from_similarities(similarities)

        return labels

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        converted_obs = self._observations(obs)
        info["labels"] = self._compute_labels(converted_obs).cpu().numpy()
        converted_obs = converted_obs.cpu().numpy()
        return converted_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        converted_obs = self._observations(obs)
        info["labels"] = self._compute_labels(converted_obs).cpu().numpy()
        converted_obs = converted_obs.cpu().numpy()
        return converted_obs, reward, terminated, truncated, info