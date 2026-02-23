import minigrid
import gymnasium as gym

env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="human")




observation, info = env.reset()

for _ in range(100):
    # Sample a random action (0-6: turn left, right, forward, pickup, drop, toggle, done)
    action = env.action_space.sample()
    
    # Modern Gymnasium returns 5 values
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()