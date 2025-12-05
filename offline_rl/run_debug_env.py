import gymnasium as gym

from env.visual_minecraft.debug_env import DebugGridWorldEnv


def build_env(render_mode: str = "human") -> gym.Env:
    """
    Create a small DebugGridWorldEnv instance with a simple LTL formula.

    The formula below just says: "eventually visit c0", where c0 is mapped to
    the only proposition used in this debug env (the pickaxe).
    """
    items = ["pickaxe"]
    formula = ("(F c0)", 1, "debug_task: visit({0})".format(*items))

    env = DebugGridWorldEnv(
        formula=formula,
        render_mode=render_mode,
        state_type="symbolic",
        train=False,
    )
    return env


def main() -> None:
    env = build_env(render_mode="human")

    print("DebugGridWorldEnv interactive session")
    print("Actions:")
    print("  w = up (2)")
    print("  s = down (0)")
    print("  d = right (1)")
    print("  a = left (3)")
    print("  r = reset episode")
    print("  q = quit")

    obs, info = env.reset()
    terminated = False
    truncated = False

    def print_step_result(step_idx, action_name, reward, terminated, truncated, info_dict):
        label = info_dict.get("label", None)
        print(
            f"[step {step_idx}] action={action_name}, "
            f"reward={reward}, terminated={terminated}, truncated={truncated}, label={label}"
        )

    step_idx = 0

    try:
        while True:
            if terminated or truncated:
                print("Episode finished. Type 'r' to reset or 'q' to quit.")

            cmd = input("Command (w/a/s/d/r/q): ").strip().lower()

            if cmd == "q":
                break
            if cmd == "r":
                obs, info = env.reset()
                terminated = False
                truncated = False
                step_idx = 0
                print("Environment reset.")
                continue

            # Map commands to discrete actions used in the env
            if cmd == "w":
                action = 2  # up
                action_name = "UP"
            elif cmd == "s":
                action = 0  # down
                action_name = "DOWN"
            elif cmd == "d":
                action = 1  # right
                action_name = "RIGHT"
            elif cmd == "a":
                action = 3  # left
                action_name = "LEFT"
            else:
                print("Unknown command. Use w/a/s/d/r/q.")
                continue

            obs, reward, terminated, truncated, info = env.step(action)
            step_idx += 1
            print_step_result(step_idx, action_name, reward, terminated, truncated, info)

    finally:
        env.close()


if __name__ == "__main__":
    main()


