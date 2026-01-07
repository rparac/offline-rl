import time
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("Pendulum-v1", render_mode="rgb_array")

obs = env.reset()

done = False

while not done:
    a = env.action_space.sample()
    print(f"Selected action: {a}")
    obs, reward, terminated, truncated, info = env.step(a)
    done = terminated or truncated
    img = env.render()
    plt.savefig("pendulum.png")

    time.sleep(1)

print("Done")