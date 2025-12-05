import random, math, os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SimpleLTLEnv(gym.Env):

    def __init__(self, letters: str, max_num_steps: int):
        """
            letters:
                - (str) propositions
            timeout:
                - (int) maximum lenght of the episode
        """
        self.letters      = letters
        self.letter_types = list(set(letters))
        self.letter_types.sort()
        self.action_space = spaces.Discrete(len(self.letter_types))
        self.observation_space = spaces.Discrete(1)
        self.num_episodes = 0
        self.curr_step = 0
        self.max_num_steps = max_num_steps
        self.proposition = None

    def step(self, action):
        """
        This function executes an action in the environment.
        """
        self.curr_step += 1
        reward = 0.0
        truncated = self.curr_step > self.max_num_steps
        terminated = False
        obs = self._get_observation()
        self.proposition = action

        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        return self.observation_space.sample()

    def seed(self, seed=None):
        return

    def reset(self):
        self.curr_step = 0
        self.num_episodes += 1
        obs = self._get_observation()

        return obs

    def show(self):
        print("Events:", self.get_events(), "\tTimeout:", self.max_num_steps - self.curr_step)

    def get_events(self):
        return self.letter_types[self.proposition] if self.proposition != None else None

    def get_propositions(self):
        return self.letter_types

class SimpleLTLEnvDefault(SimpleLTLEnv):
    def __init__(self):
        # super().__init__(letters="abcdefghijkl", max_num_steps=75)
        super().__init__(letters="pgdl", max_num_steps=25)