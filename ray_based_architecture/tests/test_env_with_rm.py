"""
Test environment with CLIP encoding, VLM reward labeling, and Reward Machine.
Manual keyboard control to play the environment and see RM-based rewards.
"""

import gymnasium as gym
import numpy as np
import ray
import time
import sys

from env.visual_minecraft.env import GridWorldEnv
from ray_based_architecture.env.clip_obs_and_labels_wrapper import BatchCLIPObsAndLabelsWrapper
from ray_based_architecture.env.rm_wrapper import RMWrapper
from ray_based_architecture.reward_machine.reward_machine import RewardMachine
from ray_based_architecture.shared_memory.sac_replay_buffer import SACReplayBuffer

import pygame


def create_test_env_with_rm(render_mode='human'):
    """Create a test environment with CLIP encoding and Reward Machine."""
    # Register the environment
    gym.envs.registration.register(
        id="VisualMinecraft-v0",
        entry_point="env.visual_minecraft.env:GridWorldEnv",
    )
    
    # Environment configuration
    items = ["pickaxe", "lava", "door", "gem", "empty"]
    # formula = "(F c0)", 5, "task0: visit({1})".format(*items)
    formula = "F(c0 & F(c1))", 5, "task3: seq_visit({0}, {1})".format(*items)
    kwargs = {
        "formula": formula,
        "render_mode": render_mode,
        "state_type": "image",
        "normalize_env": False,
        "train": False,
        "use_dfa_state": False,
        "random_start": False,
    }
    
    # Create environment
    env = gym.make("VisualMinecraft-v0", **kwargs)
    
    # Wrap in vector env (single env) so we can use BatchCLIPObsAndLabelsWrapper
    env = gym.vector.SyncVectorEnv([lambda: env])
    
    # Wrap with CLIP encoder and labels (converts images to embeddings and computes labels)
    env = BatchCLIPObsAndLabelsWrapper(env)
    
    # Create Reward Machine (same as in discrete_sac_with_rm.py)
    # u0 (initial) -> u1 (got pickaxe) -> uacc (got gem)
    rm = RewardMachine()
    rm.add_states(["u0", "u1", "uacc"])
    rm.set_u0("u0")
    rm.set_uacc("uacc")
    
    # Add transitions based on CLIP labels
    # Label order from visual_minecraft_success_detector:
    # 0: "Grey and yellow pickaxe"
    # 1: "Blue diamond gem"
    # 2: "Open red double door"
    # 3: "Orange and yellow magma texture"
    # u0 --[Grey and yellow pickaxe]--> u1
    # u1 --[Blue diamond gem]--> uacc
    rm.add_transition("u0", "u1", ("Grey and yellow pickaxe",))
    rm.add_transition("u1", "uacc", ("Orange and yellow magma texture",))
    
    # Build transition matrix with label order matching visual_minecraft_success_detector
    label_order = ["Grey and yellow pickaxe", "Blue diamond gem", "Open red double door", "Orange and yellow magma texture"]
    rm.build_transition_matrix(label_order)
    
    print(f"\n✓ Reward Machine initialized with {len(rm.states)} states:")
    print(rm)
    
    # Wrap with RM wrapper to handle RM state transitions and rewards
    env = RMWrapper(env, rm=rm)
    
    return env, rm


def get_manual_action():
    """Get action from keyboard input using pygame."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return -1  # Signal to quit
        if event.type == pygame.KEYDOWN:
            # Arrow keys or WASD
            if event.key == pygame.K_UP or event.key == pygame.K_w:
                return 2  # Up (was swapped with Down)
            elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                return 1  # Right
            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                return 0  # Down (was swapped with Up)
            elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                return 3  # Left
            elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                return -1  # Quit
    return None


def format_labels(labels, label_names):
    """Format binary labels as a readable string."""
    active_labels = [label_names[i] for i, active in enumerate(labels) if active]
    if not active_labels:
        return "none"
    return ", ".join(active_labels)


def test_env_with_rm(num_episodes=3):
    """
    Test environment with VLM reward labeling and Reward Machine using keyboard control.
    
    Args:
        num_episodes: Number of episodes to play
    """
    print("=" * 80)
    print("Play Environment with Reward Machine")
    print("Controls: Arrow Keys or WASD | ESC/Q to quit")
    print("=" * 80)
    
    # Connect to Ray cluster
    try:
        ray.init(address='auto', ignore_reinit_error=True, log_to_driver=True)
        print(f"✓ Connected to Ray cluster")
        print(f"  Namespace: {ray.get_runtime_context().namespace or 'default'}")
    except Exception as e:
        print(f"✗ Failed to connect to Ray cluster: {e}")
        print("  Starting local Ray...")
        ray.init(ignore_reinit_error=True, log_to_driver=True)
    
    # Create replay buffer (VLM service needs this)
    replay_buffer_name = "test_replay_buffer"
    replay_buffer_namespace = ray.get_runtime_context().namespace or "default"
    
    print(f"\n✓ Creating replay buffer...")
    rb = SACReplayBuffer.options(
        name=replay_buffer_name,
        num_cpus=1,
        max_concurrency=4,
    ).remote(capacity=1000, seed=42)  # Small buffer for testing
    print(f"  Created '{replay_buffer_name}' in namespace '{replay_buffer_namespace}'")
    
    # Check GPU availability
    available_resources = ray.available_resources()
    available_gpus = available_resources.get('GPU', 0)
    print(f"  Available GPUs: {available_gpus}")
    
    if available_gpus < 1:
        print(f"\n✗ Not enough GPUs available (need 1, have {available_gpus})")
        print("  Either free up a GPU or modify VLMService deployment to use fewer GPUs")
        ray.kill(rb)
        return
    
    # Create environment with RM
    env, rm = create_test_env_with_rm(render_mode='human')
    print(f"\n✓ Created environment with Reward Machine")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Actions: 0=Down, 1=Right, 2=Up, 3=Left")
    
    # Label names for display
    label_names = ["Grey and yellow pickaxe", "Blue diamond gem", "Open red double door", "Orange and yellow magma texture"]
    
    # Initialize pygame for keyboard input
    pygame.init()
    print("\n✓ Keyboard control initialized")
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"\n{'─' * 80}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'─' * 80}")
        
        obs, info = env.reset()
        print(f"Initial observation: Dict with 'image_embedding' shape {obs['image_embedding'].shape} and 'rm_state' {obs['rm_state']}")
        
        # Get initial RM state
        rm_state_idx = obs['rm_state'][0]
        rm_state_name = list(rm.states)[rm_state_idx]
        print(f"Initial RM state: {rm_state_name} (index {rm_state_idx})")
        
        done = False
        step_count = 0
        episode_reward = 0
        rm_state_history = [rm_state_name]
        
        while not done:
            # Render
            env.render()
            
            # Get action from keyboard
            action = None
            while action is None:
                action = get_manual_action()
                if action == -1:  # Quit signal
                    print("\nQuitting...")
                    env.close()
                    pygame.quit()
                    return
                time.sleep(0.01)  # Small delay to avoid busy waiting
            
            # VectorEnv expects array of actions
            actions_array = np.array([action])
            next_obs, rm_reward, terminated, truncated, info = env.step(actions_array)
            
            # Extract single values from batch dimension
            # rm_reward = rm_reward[0]
            single_rm_reward = rm_reward[0]
            terminated = terminated[0]
            truncated = truncated[0]
            
            # Get current RM state
            current_rm_state_idx = next_obs['rm_state'][0]
            current_rm_state_name = list(rm.states)[current_rm_state_idx]
            
            # Get labels
            labels = info["labels"][0]  # Binary vector
            labels_str = format_labels(labels, label_names)
            
            # Check if RM state changed
            state_changed = current_rm_state_name != rm_state_history[-1]
            
            rb.add_batch.remote(
                observations=obs,
                rewards=rm_reward,
                actions=actions_array,
                next_observations=next_obs,
                terminateds=np.array([terminated]),
                truncateds=np.array([truncated]),
            )
            
            # Display step information
            state_change_str = f" → {current_rm_state_name}" if state_changed else ""
            print(f"Step {step_count:3d} | Action: {action} | "
                  f"RM state: {rm_state_history[-1]}{state_change_str} | "
                  f"RM reward: {single_rm_reward:.3f} | "
                  f"Labels: [{labels_str}]")
            
            if state_changed:
                print(f"         ⚡ RM transition: {rm_state_history[-1]} → {current_rm_state_name}")
                rm_state_history.append(current_rm_state_name)
            
            episode_reward += single_rm_reward
            
            obs = next_obs
            step_count += 1
            done = terminated or truncated
            
            # Safety limit
            if step_count > 100:
                print("  (Reached step limit)")
                break
        
        print(f"\nEpisode finished:")
        print(f"  Steps: {step_count}")
        print(f"  Total RM reward: {episode_reward:.3f}")
        print(f"  RM state path: {' → '.join(rm_state_history)}")
        if rm_state_history[-1] == "uacc":
            print(f"  ✓ Reached accepting state!")
        else:
            print(f"  ✗ Did not reach accepting state (ended in {rm_state_history[-1]})")
        
        if episode < num_episodes - 1:
            print("\nPress any arrow key to start next episode...")
            waiting = True
            while waiting:
                action = get_manual_action()
                if action == -1:
                    print("\nQuitting...")
                    env.close()
                    pygame.quit()
                    return
                elif action is not None:
                    waiting = False
                time.sleep(0.01)
    
    env.close()
    pygame.quit()
    
    # Cleanup Ray resources
    print(f"\nCleaning up...")
    
    try:
        ray.kill(rb)
        print("  Deleted replay buffer")
    except Exception:
        pass
    
    print(f"\n{'=' * 80}")
    print("Test completed!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Play environment with keyboard control and see RM-based rewards")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play")
    
    args = parser.parse_args()
    
    test_env_with_rm(num_episodes=args.episodes)
