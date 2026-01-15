"""
Distributed experience collector for SAC with CLIP embeddings.
Each collector actor runs on a separate GPU.
"""

import ray
import numpy as np
import torch
import gymnasium as gym
from collections import deque


@ray.remote(num_gpus=0.5)  # Share GPU across collectors
class ExperienceCollector:
    def __init__(self, make_env_fn, num_envs: int, vlm_service_handle, seed: int):
        """
        Args:
            make_env_fn: Function that creates a single environment
            num_envs: Number of parallel environments for this collector
            vlm_service_handle: Ray Serve handle for VLM service
            seed: Random seed for this collector
        """
        self.num_envs = num_envs
        self.vlm_service_handle = vlm_service_handle
        self.seed = seed
        
        # Create vectorized environments
        envs = [make_env_fn(seed + i) for i in range(num_envs)]
        self.envs = gym.vector.SyncVectorEnv(envs)
        
        # Policy will be set later via set_policy()
        self.policy = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.total_steps = 0
        
    def set_policy(self, policy_state_dict):
        """Update the policy with new weights from training loop."""
        # Create policy network if needed
        if self.policy is None:
            from ray_based_architecture.rl.discrete_sac import Actor
            # Create a dummy env to get observation/action spaces
            dummy_env = self.envs.envs[0]
            
            class DummyEnvs:
                def __init__(self, env):
                    self.single_observation_space = env.observation_space
                    self.single_action_space = env.action_space
            
            self.policy = Actor(DummyEnvs(dummy_env)).to(self.device)
        
        self.policy.load_state_dict(policy_state_dict)
        self.policy.eval()
    
    def collect_episodes(self, num_episodes: int, use_random_actions: bool = False):
        """
        Collect experience for a given number of episodes.
        
        Args:
            num_episodes: Number of episodes to collect
            use_random_actions: If True, use random actions (for initial data collection)
        """
        episodes_collected = 0
        obs, _ = self.envs.reset(seed=self.seed)
        
        episode_rewards_temp = np.zeros(self.num_envs)
        episode_lengths_temp = np.zeros(self.num_envs)
        
        while episodes_collected < num_episodes:
            # Select actions
            if use_random_actions or self.policy is None:
                actions = np.array([self.envs.single_action_space.sample() 
                                   for _ in range(self.num_envs)])
            else:
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).to(self.device)
                    actions, _, _ = self.policy.get_action(obs_tensor)
                    actions = actions.cpu().numpy()
            
            # Step environments
            next_obs, rewards, terminateds, truncateds, infos = self.envs.step(actions)
            
            # Send to VLM for reward labeling and storage
            self.vlm_service_handle.add_to_buffer_with_labeling.remote(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                terminateds=terminateds,
                truncateds=truncateds,
            )
            
            # Track statistics
            episode_rewards_temp += rewards
            episode_lengths_temp += 1
            self.total_steps += self.num_envs
            
            # Handle episode terminations
            if "episode" in infos:
                finished_episodes = infos["_episode"]
                if finished_episodes.any():
                    ep_rewards = infos["episode"]["r"][finished_episodes]
                    ep_lengths = infos["episode"]["l"][finished_episodes]
                    
                    for r, l in zip(ep_rewards, ep_lengths):
                        self.episode_rewards.append(float(r))
                        self.episode_lengths.append(float(l))
                        episodes_collected += 1
            
            obs = next_obs
            
            # Early exit if we've collected enough episodes
            if episodes_collected >= num_episodes:
                break
        
        return {
            "episodes_collected": episodes_collected,
            "total_steps": self.total_steps,
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "mean_length": np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
        }
    
    def collect_steps(self, num_steps: int, use_random_actions: bool = False):
        """
        Collect experience for a given number of steps.
        
        Args:
            num_steps: Number of environment steps to collect
            use_random_actions: If True, use random actions
        """
        steps_collected = 0
        
        if not hasattr(self, '_current_obs'):
            self._current_obs, _ = self.envs.reset(seed=self.seed)
        
        while steps_collected < num_steps:
            # Select actions
            if use_random_actions or self.policy is None:
                actions = np.array([self.envs.single_action_space.sample() 
                                   for _ in range(self.num_envs)])
            else:
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(self._current_obs).to(self.device)
                    actions, _, _ = self.policy.get_action(obs_tensor)
                    actions = actions.cpu().numpy()
            
            # Step environments
            next_obs, rewards, terminateds, truncateds, infos = self.envs.step(actions)
            
            # Send to VLM for reward labeling and storage
            self.vlm_service_handle.add_to_buffer_with_labeling.remote(
                observations=self._current_obs,
                actions=actions,
                next_observations=next_obs,
                terminateds=terminateds,
                truncateds=truncateds,
            )
            
            # Track statistics
            self.total_steps += self.num_envs
            steps_collected += self.num_envs
            
            # Handle episode terminations
            if "episode" in infos:
                finished_episodes = infos["_episode"]
                if finished_episodes.any():
                    ep_rewards = infos["episode"]["r"][finished_episodes]
                    ep_lengths = infos["episode"]["l"][finished_episodes]
                    
                    for r, l in zip(ep_rewards, ep_lengths):
                        self.episode_rewards.append(float(r))
                        self.episode_lengths.append(float(l))
            
            self._current_obs = next_obs
        
        return {
            "steps_collected": steps_collected,
            "total_steps": self.total_steps,
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "mean_length": np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
        }
    
    def get_stats(self):
        """Get current collection statistics."""
        return {
            "total_steps": self.total_steps,
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "mean_length": np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            "num_episodes": len(self.episode_rewards),
        }
