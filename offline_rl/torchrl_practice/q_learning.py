import os

from collections import defaultdict
from pathlib import Path
import numpy as np
import gymnasium as gym

from offline_rl.torchrl_practice.env_util import setup_visual_minecraft
from offline_rl.torchrl_practice.torch_rl_utils import generate_q_table


def epsilon_greedy_action(q_table, state, epsilon, action_space):
    """Select action using epsilon-greedy policy"""
    if np.random.random() < epsilon:
        return action_space.sample()  # Explore: random action
    else:
        # Exploit: choose best action
        return np.argmax(q_table[state[0], state[1], :])

def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    """
    Simple Q-learning algorithm
    
    Args:
        env: The environment
        num_episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon_start: Starting epsilon for exploration
        epsilon_end: Minimum epsilon value
        epsilon_decay: Decay rate for epsilon
    """
    # Initialize Q-table
    q_table = np.zeros((env.observation_space[0].n, env.observation_space[1].n, env.action_space.n))

    episode_rewards = []
    epsilon = epsilon_start

    for episode in range(num_episodes):
        obs, info = env.reset()

        epsilon = epsilon_start - (epsilon_start - epsilon_end) * episode / num_episodes

        total_reward = 0
        done = False
        while not done:
            action = epsilon_greedy_action(q_table, obs, epsilon, env.action_space)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = tuple(next_obs)

            if done:
                next_max_q = 0
            else:
                next_max_q = np.max(q_table[next_obs[0], next_obs[1], :])

            q_table[obs[0], obs[1], action] = q_table[obs[0], obs[1], action] + alpha * ((reward + gamma * next_max_q) - q_table[obs[0], obs[1], action])

            obs = next_obs
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
    
    return q_table, episode_rewards

# Main execution
if __name__ == "__main__":
    env = setup_visual_minecraft()
    
    print("Starting Q-learning training...")
    q_table, episode_rewards = q_learning(
        env,
        num_episodes=10000,
        alpha=0.1,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    print("\nTraining completed!")
    print(f"Final Q-table shape: {q_table.shape}")
    print(f"Average reward over all episodes: {np.mean(episode_rewards):.2f}")
    print(f"Average reward over last 100 episodes: {np.mean(episode_rewards[-100:]):.2f}")
    
    # Test the learned policy (greedy, no exploration)
    print("\nTesting learned policy...")
    obs, info = env.reset()
    state = tuple(obs)
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 100:
        action = np.argmax(q_table[state[0], state[1], :])  # Greedy action
        obs, reward, terminated, truncated, info = env.step(action)
        state = tuple(obs)
        total_reward += reward
        done = terminated or truncated
        steps += 1
    
    print(f"Test episode - Total reward: {total_reward}, Steps: {steps}")
    
    # Visualize Q-table
    print("\nCreating Q-table visualization...")
    generate_q_table(q_table, save_path="artifacts/q_table_visualization.png")


    save_dir = os.path.join(os.path.dirname(__file__), "offline_rl")
    save_path = Path("artifacts/q_table.npy")
    np.save(save_path, q_table)
    print(f"Q-table saved to: {save_path}")
