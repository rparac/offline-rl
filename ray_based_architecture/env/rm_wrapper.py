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

        self.rm_transitioner = DeterministicRMTransitioner(_rm)

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
        new_obs = np.concatenate((obs, rm_state))
        return new_obs, info

    def step(
            self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        rm_state = self.rm_transitioner.get_next_state(self._curr_rm_state, info["labels"])
        self._curr_rm_state = rm_state

        # We intentionally interrupt the episode if the RM believes the episode should be done
        if self.rm_transitioner.rm.is_state_terminal(rm_state) and not terminated:
            return np.concatenate((obs, rm_state)), reward, terminated, True, info

        return np.concatenate((obs, rm_state)), reward, terminated, truncated, info