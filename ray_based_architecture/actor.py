
class ExperienceCollector:
    def __init__(self, make_env_fn, reward_labeller, replay_buffer):
        self.env = make_env_fn()
        self.reward_labeller = reward_labeller

        self.replay_buffer = replay_buffer

    async def run_forever(self):
        while True:
            pass

    async def collect_episode(self):
        obs, info = self.env.reset()
        done = False
        
        obs = []
        rewards = []
        next_obs = []
        terminateds = []
        truncateds = []
        actions = []

        reward_refs = []
        while not done:
            action = self.policy.act(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            reward_refs = self.reward_labeller.label_reward.remote(next_obs)

        rewards = [ref.result() for ref in reward_refs]

        self.replay_buffer.add_batch(
            obs=obs, 
            actions=actions, 
            rewards=rewards,
            next_obs=next_obs,
            terminateds=terminateds,
            truncateds=truncateds,
        )


        pass
