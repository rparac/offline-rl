import gymnasium as gym

class FrozenLakeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, map_name: str = "4x4"):
        super().__init__(env)

        assert map_name in ["4x4", "8x8"]
        size = int(map_name[0])
        self.size = size
        self.observation_space = gym.spaces.MultiDiscrete([size, size])

    def observation(self, obs):
        return obs // self.size, obs % self.size