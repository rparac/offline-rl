"""
Wrapper which is used to directly augment RM information as part of the observation
Assumes, observation is a Box and the RM info is obtained as labels
"""
from typing import SupportsFloat, Any

import gymnasium
from gymnasium.core import WrapperActType, WrapperObsType
import numpy as np

from ray_based_architecture.reward_machine.reward_machine import RewardMachine
from ray_based_architecture.reward_machine.transition.deterministic_rm_transitioner import DeterministicRMTransitioner

class RMWrapper(gymnasium.vector.VectorWrapper):
    def __init__(self, env: gymnasium.vector.VectorWrapper, rm=None):
        super().__init__(env)

        _rm = rm if rm is not None else RewardMachine.default_rm()

        self.rm_transitioner = DeterministicRMTransitioner(_rm, num_envs=env.num_envs)
        # TODO: Remove later. For debugging only.
        self._prev_gt_state_indices = None
        self._prev_rm_state_indices = None
        # Track which environments terminated in the previous step
        # (needed because vectorized envs auto-reset terminated environments)
        self._prev_terminated = np.zeros(env.num_envs, dtype=bool)
        self._prev_truncated = np.zeros(env.num_envs, dtype=bool)

        num_states = len(self.rm_transitioner.rm.states)
        self.single_observation_space = gymnasium.spaces.Dict({
            "image_embedding": env.single_observation_space,
            "rm_state": gymnasium.spaces.Discrete(num_states),
        })

        self.observation_space = gymnasium.spaces.Dict({
            "image_embedding": env.observation_space,
            "rm_state": gymnasium.spaces.MultiDiscrete([num_states] * env.num_envs),
        })

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        rm_state = self.rm_transitioner.get_initial_state()
        obs, info = self.env.reset()
        rm_state = self.rm_transitioner.get_next_state(rm_state, info["labels"])
        self._curr_rm_state = rm_state
        
        # Convert RM state from one-hot to discrete indices
        rm_state_indices = np.argmax(rm_state, axis=1)

        self._prev_gt_state_indices = None
        self._prev_rm_state_indices = None
        self._prev_terminated = np.zeros(self.num_envs, dtype=bool)
        self._prev_truncated = np.zeros(self.num_envs, dtype=bool)
        
        # Return Dict observation
        new_obs = {
            "image_embedding": obs,
            "rm_state": rm_state_indices,
        }
        return new_obs, info

    def step(
            self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        info["original_reward"] = env_reward
        
        # In vectorized environments, when an episode terminates, the next step() automatically
        # resets that environment. The observation returned is from the reset state.
        # We need to reset RM state to initial for environments that terminated/truncated in the PREVIOUS step.
        # Check which environments were done in the previous step (these are now reset)
        prev_done_mask = self._prev_terminated | self._prev_truncated
        single_initial_rm_state = self.rm_transitioner.get_initial_state()[0]
        prev_rm_state = self._curr_rm_state
        prev_rm_state[prev_done_mask] = single_initial_rm_state

        # Compute next RM state from current state and labels
        rm_state = self.rm_transitioner.get_next_state(prev_rm_state, info["labels"])
        
        # Update tracking for next step
        self._prev_terminated = terminated.copy()
        self._prev_truncated = truncated.copy()
        
        rm_rewards = self.rm_transitioner.rm.get_reward(prev_rm_state, rm_state)
        
        self._curr_rm_state = rm_state
        
        # Convert RM state from one-hot to discrete indices
        rm_state_indices = np.argmax(rm_state, axis=1)
        
        # Return Dict observation
        new_obs = {
            "image_embedding": obs,
            "rm_state": rm_state_indices,
        }

        # We intentionally interrupt the episode if the RM believes the episode should be done

        return new_obs, rm_rewards, terminated, truncated, info