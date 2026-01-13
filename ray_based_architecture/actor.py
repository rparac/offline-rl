import ray
import time

# TODO: We should probably do two things:
#  1. Always submit the request of size submit_batch_size (wrap around if needed)
#  2. Simplify VLMService to process thing immediately instead of doing the batching itself
@ray.remote
class ExperienceCollector:
    def __init__(self, make_env_fn, reward_labeller):
        self.env = make_env_fn()
        self.reward_labeller = reward_labeller

        # self.replay_buffer = replay_buffer

    def collect_n_episodes(
        self,
        n: int,
        *,
        max_inflight_submits: int = 500,
        log_every_episodes: int = 10,
    ):
        start_time = time.time()
        total_submitted = 0  # transitions submitted to VLM (i.e., "converted obs")
        completed = 0
        inflight_submit_refs = []

        for ep in range(int(n)):
            observation, info = self.env.reset()
            done = False

            while not done:
                action = self.env.action_space.sample()
                next_observation, _reward, terminated, truncated, info = self.env.step(action)

                # One request == one transition. Ray Serve will batch across producers.
                ref = self.reward_labeller.label_reward.remote(
                    observations=observation,
                    actions=action,
                    next_observations=next_observation,
                    terminateds=bool(terminated),
                    truncateds=bool(truncated),
                )
                inflight_submit_refs.append(ref)
                total_submitted += 1

                if len(inflight_submit_refs) >= max_inflight_submits:
                    ready_refs = inflight_submit_refs[:100]
                    for ref in ready_refs:
                        _ = ref.result()
                        completed += 1
                    inflight_submit_refs = inflight_submit_refs[100:]

                observation = next_observation
                done = bool(terminated or truncated)

            if (ep + 1) % int(log_every_episodes) == 0 or (ep + 1) == int(n):
                elapsed = time.time() - start_time
                throughput = total_submitted / elapsed if elapsed > 0 else 0
                print(
                    f"Collected {ep + 1}/{int(n)} episodes | "
                    f"Submitted (converted obs): {total_submitted} | "
                    f"Completed: {completed} | "
                    f"Throughput: {throughput:.2f} obs/sec"
                )


        # Optionally wait for remaining writes to land before returning.
        if inflight_submit_refs:
            ray.get(inflight_submit_refs)
