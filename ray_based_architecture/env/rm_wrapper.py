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

        rm_state = self.rm_transitioner.get_next_state(self._curr_rm_state, info["labels"])
        
        # TODO: clean this up. Possibly have a reward be returned by get_next_state.
        # Compute RM-based rewards
        rm_rewards = np.array([
            self.rm_transitioner.rm.get_reward(
                self.rm_transitioner.rm.states[np.argmax(self._curr_rm_state[i])],
                self.rm_transitioner.rm.states[np.argmax(rm_state[i])]
            )
            for i in range(self.num_envs)
        ], dtype=np.float32)
        
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