import sys
from typing import Optional

from PIL import Image

from quick_experiments.shapes_experiment.env import make_env, make_test_env


KEY_TO_ACTION = {
    "w": 0,  # up
    "s": 1,  # down
    "a": 2,  # left
    "d": 3,  # right
}


def get_action_from_input() -> Optional[int]:
    """Read a single line from stdin and map it to an action.

    Returns:
        int action in [0, 3] or None to quit/skip.
    """
    try:
        key = input("Move (w/a/s/d, q to quit, ENTER to repeat last): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if key == "":
        return -1  # sentinel: repeat last
    if key == "q":
        return None
    return KEY_TO_ACTION.get(key, -2)  # -2 means invalid, ignore


def main():
    env = make_test_env(render_mode="human")
    obs, info = env.reset()
    env.render()

    last_action = 0  # default action (up)

    try:
        while True:
            action = get_action_from_input()
            if action is None:
                print("Exiting.")
                break

            if action == -1:
                # Repeat last action
                action = last_action
            elif action == -2:
                print("Invalid key. Use w/a/s/d, q to quit.")
                continue

            last_action = action

            obs, reward, terminated, truncated, info = env.step(action)
            # Image.fromarray(obs).save("obs.png")
            env.render()

            if reward != 0.0:
                print(f"Reward: {reward}")

            if terminated or truncated:
                print("Episode finished. Resetting environment.")
                obs, info = env.reset()
                env.render()
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())

