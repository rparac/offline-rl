"""
Test environment with CLIP encoding and VLM reward labeling.
Manual keyboard control to play the environment and see VLM rewards.
"""

import gymnasium as gym
import numpy as np
import ray
from ray import serve
import time
import sys

from env.visual_minecraft.env import GridWorldEnv
from ray_based_architecture.env.clip_obs_wrapper import BatchCLIPObsWrapper
from ray_based_architecture.vlm_service import VLMService
from ray_based_architecture.shared_memory.sac_replay_buffer import SACReplayBuffer

# Try to import pygame for keyboard input
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available, manual control disabled")


def create_test_env(render_mode='human'):
    """Create a test environment with CLIP encoding."""
    # Register the environment
    gym.envs.registration.register(
        id="VisualMinecraft-v0",
        entry_point="env.visual_minecraft.env:GridWorldEnv",
    )
    
    # Environment configuration
    items = ["pickaxe", "lava", "door", "gem", "empty"]
    formula = "(F c0)", 5, "task0: visit({1})".format(*items)
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
    
    # Wrap in vector env (single env) so we can use BatchCLIPObsWrapper
    env = gym.vector.SyncVectorEnv([lambda: env])
    
    # Wrap with CLIP encoder (converts images to embeddings)
    env = BatchCLIPObsWrapper(env)
    
    return env


def get_manual_action():
    """Get action from keyboard input using pygame."""
    if not PYGAME_AVAILABLE:
        return None
    
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


def test_env_with_vlm(num_episodes=3):
    """
    Test environment with VLM reward labeling using keyboard control.
    
    Args:
        num_episodes: Number of episodes to play
    """
    print("=" * 60)
    print("Play Environment with VLM Reward Labeling")
    print("Controls: Arrow Keys or WASD | ESC/Q to quit")
    print("=" * 60)
    
    if not PYGAME_AVAILABLE:
        print("\n✗ This script requires pygame. Install with: pip install pygame")
        return
    
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
    
    # Start VLM service
    try:
        print(f"\n✓ Starting VLM service...")
        serve_app_name = "test_vlm_service"
        
        # Use serve.run with blocking=True to catch initialization errors
        vlm_handle = serve.run(
            VLMService.bind(replay_buffer_name, replay_buffer_namespace), 
            name=serve_app_name,
            route_prefix="/test_vlm"
        )
        
        # Wait a bit for deployment to initialize
        time.sleep(3)
        
        # Check deployment status
        status = serve.status()
        if serve_app_name in status.applications:
            app_status = status.applications[serve_app_name]
            print(f"  VLM service status: {app_status.status}")
            if app_status.status != "RUNNING":
                print(f"  Warning: App status is {app_status.status}, message: {app_status.message}")
        
        print(f"  VLM service '{serve_app_name}' is ready")
    except Exception as e:
        print(f"\n✗ Failed to start VLM service: {e}")
        print(f"  Error type: {type(e).__name__}")
        
        # Try to get more details from serve status
        try:
            status = serve.status()
            print(f"  Serve status: {status}")
        except Exception:
            pass
        
        ray.kill(rb)
        return
    
    # Create environment
    env = create_test_env(render_mode='human')
    print(f"\n✓ Created environment")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Actions: 0=Down, 1=Right, 2=Up, 3=Left")
    
    # Initialize pygame for keyboard input
    pygame.init()
    print("\n✓ Keyboard control initialized")
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"\n{'─' * 60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'─' * 60}")
        
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")  # Will be (1, 768) for single env
        
        done = False
        step_count = 0
        episode_reward = 0
        
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
            next_obs, env_reward, terminated, truncated, info = env.step(actions_array)
            
            # Extract single values from batch dimension
            env_reward = env_reward[0]
            terminated = terminated[0]
            truncated = truncated[0]
            
            # Get VLM reward
            vlm_start = time.time()
            vlm_rewards = vlm_handle.add_to_buffer_with_labeling.remote(
                observations=obs,
                actions=actions_array,
                next_observations=next_obs,
                terminateds=np.array([terminated]),
                truncateds=np.array([truncated]),
            ).result()
            
            vlm_reward = vlm_rewards[0]
            vlm_time_ms = (time.time() - vlm_start) * 1000
            
            print(f"Step {step_count:3d} | Action: {action} | "
                  f"Env reward: {env_reward:.3f} | VLM reward: {vlm_reward:.3f} | "
                  f"VLM time: {vlm_time_ms:.1f}ms")
            
            episode_reward += vlm_reward
            
            obs = next_obs
            step_count += 1
            done = terminated or truncated
            
            # Safety limit
            if step_count > 100:
                print("  (Reached step limit)")
                break
        
        print(f"\nEpisode finished:")
        print(f"  Steps: {step_count}")
        print(f"  Total VLM reward: {episode_reward:.3f}")
        
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
        serve.delete("test_vlm_service")
        print("  Deleted VLM service")
    except Exception:
        pass
    
    try:
        ray.kill(rb)
        print("  Deleted replay buffer")
    except Exception:
        pass
    
    print(f"\n{'=' * 60}")
    print("Test completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Play environment with keyboard control and see VLM rewards")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play")
    
    args = parser.parse_args()
    
    test_env_with_vlm(num_episodes=args.episodes)
