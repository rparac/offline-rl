import os
import sys
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import imageio

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import gymnasium as gym

# Add project root to path to import env
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from env.visual_minecraft.env import GridWorldEnv

# Directory for saving episode frames
FRAMES_DIR = Path(__file__).parent / "interactive_test"


def load_roboreward_model():
    """Load and return the RoboReward model and processor."""
    print("Loading RoboReward model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "teetone/RoboReward-8B", dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("teetone/RoboReward-8B")
    print("Model loaded successfully!")
    return model, processor


def get_task_prompt():
    """Return the task description prompt for RoboReward."""
    return """
Given the task, assign a discrete progress score reward (1,2,3,4,5) for the robot in the video in the format: ANSWER: <score>
Rubric for end-of-episode progress (judge only the final state without time limits):
1 - No Success: Final state shows no goal-relevant change for the command.
2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.
3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.
4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.
5 - Perfect Completion: Final state satisfies all requirements.

Task: 
You are a robot in a grid world which consists of a blue diamond gem, a yellow-grey pickaxe, a yellow magma texture, and a red open door. 
Your task is to reach the yellow-grey pickaxe object.
"""


def setup_environment():
    """Setup and return the VisualMinecraft environment."""
    items = ["pickaxe", "lava", "door", "gem", "empty"]
    formula = "(F c0)", 5, "task0: visit({1})".format(*items)
    env = GridWorldEnv(
        formula=formula,
        render_mode="rgb_array",  # Use rgb_array to get frames
        state_type="image",
        use_dfa_state=False,
        train=False,  # Set to False to show robot during navigation
        size=4,
        random_start=False,
        normalize_env=False  # Keep raw images for VLM
    )
    return env


def get_action_from_input():
    """
    Get action from user input.
    Returns action: 0=DOWN, 1=RIGHT, 2=UP, 3=LEFT, or None to quit.
    """
    print("\nAction: [w=UP, s=DOWN, a=LEFT, d=RIGHT, q=QUIT]")
    try:
        key = input().strip().lower()
        if key == 'w':
            return 2  # UP
        elif key == 's':
            return 0  # DOWN
        elif key == 'a':
            return 3  # LEFT
        elif key == 'd':
            return 1  # RIGHT
        elif key == 'q':
            return None
        else:
            print("Invalid key. Use w/s/a/d for movement, q to quit.")
            return get_action_from_input()
    except (EOFError, KeyboardInterrupt):
        return None


def clear_frames_directory():
    """Clear the frames directory and recreate it."""
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cleared and created frames directory: {FRAMES_DIR}")


def save_frame(frame_array, frame_path):
    """Save a frame array to a file."""
    img = Image.fromarray(frame_array)
    img.save(frame_path)


def create_video_from_frames(frame_paths, output_video_path, fps=2):
    """
    Create a video from a list of frame image paths.
    
    Args:
        frame_paths: List of paths to frame images
        output_video_path: Path where the video will be saved
        fps: Frames per second for the video
    
    Returns:
        Path to the created video
    """
    print(f"Creating video from {len(frame_paths)} frames...")
    
    # Sort frame paths by frame number to ensure correct order
    def get_frame_number(path):
        # Extract frame number from path like "frame_0.png" -> 0
        filename = Path(path).stem
        return int(filename.split('_')[1])
    
    sorted_frame_paths = sorted(frame_paths, key=get_frame_number)
    
    # Read all frames
    frames = []
    for frame_path in sorted_frame_paths:
        img = Image.open(frame_path)
        frames.append(np.array(img))
    
    # Create video using imageio
    # Note: Requires imageio-ffmpeg plugin for MP4 support: pip install imageio-ffmpeg
    imageio.mimsave(
        str(output_video_path),
        frames,
        fps=fps,
        format='FFMPEG',
        codec='libx264'
    )
    
    print(f"Video saved to {output_video_path}")
    return output_video_path


def evaluate_with_roboreward(model, processor, frame_paths, task_prompt, video_path):
    """
    Evaluate the episode video with RoboReward model.
    
    Args:
        model: The RoboReward model
        processor: The processor for the model
        frame_paths: List of paths to frame images (used to create video)
        task_prompt: The task description prompt
        video_path: Path to the video file
    
    Returns:
        The model's output text
    """
    # Prepare a multimodal chat message with video + text
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": task_prompt}
            ],
        }
    ]
    
    # Preprocess: convert to model inputs
    inputs = processor(
        text=[processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )],
        videos=[str(video_path)],  # video input
        padding=True,
        return_tensors="pt",
        # do_resize=True,
    )
    
    # Move to GPU if available
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate output
    print("\nEvaluating episode with RoboReward model...")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
    )
    
    # Decode the text response
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text


def main():
    """Main function to run the interactive environment with RoboReward evaluation."""
    # Load model
    model, processor = load_roboreward_model()
    task_prompt = get_task_prompt()
    
    # Setup environment
    env = setup_environment()
    
    print("\n" + "="*60)
    print("VisualMinecraft Environment - Interactive Navigation")
    print("="*60)
    print("Controls: w=UP, s=DOWN, a=LEFT, d=RIGHT, q=QUIT")
    print("Goal: Reach the pickaxe (yellow-grey object)")
    print("="*60)
    
    episode_num = 0
    
    while True:
        # Reset environment
        obs, info = env.reset()
        episode_num += 1
        frames = []
        frame_paths = []
        
        print(f"\n--- Episode {episode_num} ---")
        print("Episode started. Navigate to the pickaxe!")
        
        # Clear and setup frames directory for this episode
        clear_frames_directory()
        
        # Get initial frame
        initial_frame = env.render()
        if initial_frame is not None:
            frame_path = FRAMES_DIR / f"frame_0.png"
            save_frame(initial_frame, str(frame_path))
            frames.append(initial_frame)
            frame_paths.append(str(frame_path))
            print(f"Step 0: Frame saved to {frame_path}")
        
        step = 0
        done = False
        user_quit = False
        
        while not done:
            # Get action from user
            action = get_action_from_input()
            if action is None:
                print("Quitting current episode...")
                user_quit = True
                break
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            done = terminated or truncated
            
            # Get and save frame
            frame = env.render()
            if frame is not None:
                frame_path = FRAMES_DIR / f"frame_{step}.png"
                save_frame(frame, str(frame_path))
                frames.append(frame)
                frame_paths.append(str(frame_path))
                
                # Show agent location if available
                agent_loc = info.get('gt_labels', None)
                if agent_loc is not None:
                    has_pickaxe, has_gem, has_exit, has_lava = agent_loc
                    location_desc = []
                    if has_pickaxe:
                        location_desc.append("PICKAXE")
                    if has_gem:
                        location_desc.append("GEM")
                    if has_exit:
                        location_desc.append("EXIT")
                    if has_lava:
                        location_desc.append("LAVA")
                    loc_str = f" at {'/'.join(location_desc)}" if location_desc else ""
                    print(f"Step {step}: Action taken, frame saved. Reward: {reward}{loc_str}")
                else:
                    print(f"Step {step}: Action taken, frame saved. Reward: {reward}")
            
            # Check if episode is done
            if terminated:
                print(f"\nEpisode completed successfully! (Terminated at step {step})")
            elif truncated:
                print(f"\nEpisode timed out! (Truncated at step {step})")
        
        # Episode ended (either naturally or by user quit) - create video and evaluate with RoboReward
        if len(frames) > 0:
            print(f"\nEpisode ended with {len(frames)} frames.")
            
            # Create video from frames
            video_path = FRAMES_DIR / "episode_video.mp4"
            create_video_from_frames(frame_paths, video_path, fps=2)
            
            print("Evaluating with RoboReward model...")
            output_text = evaluate_with_roboreward(model, processor, frame_paths, task_prompt, video_path)
            
            print("\n" + "="*60)
            print("RoboReward Evaluation Result:")
            print("="*60)
            print(output_text[0] if isinstance(output_text, list) else output_text)
            print("="*60)
        else:
            print("\nNo frames collected. Skipping evaluation.")
        
        # If user quit, exit the main loop
        if user_quit:
            break
        
        # Ask if user wants to continue
        print("\nStart a new episode? [y/n]")
        try:
            response = input().strip().lower()
            if response != 'y':
                break
        except (EOFError, KeyboardInterrupt):
            break
    
    env.close()
    print("\nExiting. Goodbye!")


if __name__ == "__main__":
    main()
