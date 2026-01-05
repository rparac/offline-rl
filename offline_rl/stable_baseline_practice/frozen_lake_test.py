import argparse
import random
from typing import Optional, Tuple

import gymnasium as gym

from offline_rl.stable_baseline_practice.frozen_lake_utils import FrozenLakeObsWrapper


def _safe_make_env(render_mode: str, is_slippery: bool, map_name: str):
    """Create FrozenLake with a best-effort render mode.

    If "human" rendering fails (common on headless setups), we fall back to "ansi".
    """
    try:
        env = gym.make(
            "FrozenLake-v1",
            render_mode=render_mode,
            is_slippery=is_slippery,
            map_name=map_name,
        )
        return FrozenLakeObsWrapper(env, map_name=map_name)
    except Exception as e:
        if render_mode != "ansi":
            print(f'Warning: render_mode="{render_mode}" failed ({e}). Falling back to "ansi".')
            return gym.make(
                "FrozenLake-v1",
                render_mode="ansi",
                is_slippery=is_slippery,
                map_name=map_name,
            )
        raise


def _render_text(env, obs: int) -> None:
    """Terminal-friendly rendering that marks the agent position."""
    unwrapped = env.unwrapped
    desc = getattr(unwrapped, "desc", None)
    nrow = getattr(unwrapped, "nrow", None)
    ncol = getattr(unwrapped, "ncol", None)
    if desc is None or nrow is None or ncol is None:
        return

    r, c = divmod(int(obs), int(ncol))
    rows = []
    for rr in range(int(nrow)):
        cells = []
        for cc in range(int(ncol)):
            ch = desc[rr, cc]
            if isinstance(ch, (bytes, bytearray)):
                ch = ch.decode("utf-8")
            elif isinstance(ch, (int,)):
                ch = chr(ch)
            if rr == r and cc == c:
                cells.append("A")
            else:
                cells.append(ch)
        rows.append(" ".join(cells))
    print("\n".join(rows))


def _parse_action(s: str) -> Optional[int]:
    s = s.strip().lower()
    if not s:
        return None
    # FrozenLake: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
    mapping = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "a": 0,
        "s": 1,
        "d": 2,
        "w": 3,
        "left": 0,
        "down": 1,
        "right": 2,
        "up": 3,
        "l": 0,
        "j": 1,
        "r": 2,
        "u": 3,
    }
    return mapping.get(s)


def _reset(env, seed: Optional[int]) -> Tuple[Tuple[int, int], dict]:
    seed = random.randint(0, 10)
    obs, info = env.reset(seed=seed)
    if env.render_mode == "ansi":
        print(env.render())
        _render_text(env, obs)
    return obs, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Play FrozenLake-v1 interactively.")
    parser.add_argument("--map-name", default="4x4", choices=["4x4", "8x8"])
    parser.add_argument("--slippery", action="store_true", help="Use slippery dynamics (harder).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", default="human", choices=["human", "ansi"])
    args = parser.parse_args()

    env = _safe_make_env(args.render, is_slippery=args.slippery, map_name=args.map_name)

    print("Controls: [W/A/S/D] or [0/1/2/3] (0=LEFT,1=DOWN,2=RIGHT,3=UP)")
    print("Other: [r] reset, [q] quit")
    print(f"Config: map={args.map_name}, slippery={args.slippery}, render={env.render_mode}")

    obs, _ = _reset(env, seed=args.seed)

    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd in {"q", "quit", "exit"}:
                break
            if cmd in {"r", "reset"}:
                obs, _ = _reset(env, seed=args.seed)
                continue

            action = _parse_action(cmd)
            if action is None:
                print("Invalid input. Use WASD/0-3, or r/q.")
                continue

            obs, reward, terminated, truncated, info = env.step(action)
            if env.render_mode == "ansi":
                print(env.render())
                _render_text(env, obs)
            print(f"obs={obs} reward={reward} terminated={terminated} truncated={truncated} info={info}")

            if terminated or truncated:
                print("Episode done. Press [r] to reset or [q] to quit.")
    finally:
        env.close()


if __name__ == "__main__":
    main()