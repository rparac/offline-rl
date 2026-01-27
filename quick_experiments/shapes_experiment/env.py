import numpy as np
import matplotlib

# Try to switch to an interactive backend if the default is non-interactive (e.g. Agg).
if matplotlib.get_backend().lower() == "agg":
    for _backend in ("TkAgg", "Qt5Agg", "GTK3Agg"):
        try:
            matplotlib.use(_backend, force=True)
            break
        except Exception:
            continue

import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces


class SimpleShapesEnv(gym.Env):
    """A tiny image-based environment with:

    - Agent: green right triangle
    - Goal: red square
    - Observation: RGB image (H x W x 3) of the whole scene
    - Actions: 0=up, 1=down, 2=left, 3=right
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        agent_size: int = 10,
        goal_size: int = 10,
        step_size: int = 4,
        max_steps: int = 200,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.width = width
        self.height = height
        self.agent_size = agent_size
        self.goal_size = goal_size
        self.step_size = step_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # 4x4 logical grid over the image
        self.grid_size = 4
        self.cell_w = self.width // self.grid_size
        self.cell_h = self.height // self.grid_size

        if self.render_mode is not None:
            assert self.render_mode in self.metadata["render_modes"], (
                f"Invalid render_mode {self.render_mode}. "
                f"Valid modes: {self.metadata['render_modes']}"
            )

        # Discrete movement actions
        self.action_space = spaces.Discrete(4)  # up, down, left, right

        # Observation is the full RGB image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8,
        )

        self.rng = np.random.default_rng()

        # Positions are stored as (x, y) of the top-left corner
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([0, 0], dtype=np.int32)
        # Also track positions in grid coordinates (i, j), 0..3
        self.agent_cell = np.array([0, 0], dtype=np.int32)
        self.goal_cell = np.array([0, 0], dtype=np.int32)
        self.step_count = 0
        self._fig = None
        self._ax = None
        self._im = None

    def _cell_to_position(self, cell: np.ndarray, size: int) -> np.ndarray:
        """Convert a grid cell (i, j) into a top-left pixel position centered in that cell."""
        i, j = int(cell[0]), int(cell[1])
        cell_x0 = i * self.cell_w
        cell_y0 = j * self.cell_h
        x = cell_x0 + (self.cell_w - size) // 2
        y = cell_y0 + (self.cell_h - size) // 2
        x = int(np.clip(x, 0, self.width - size))
        y = int(np.clip(y, 0, self.height - size))
        return np.array([x, y], dtype=np.int32)

    def _non_overlapping_positions(self) -> None:
        """Sample agent and goal so they start in different grid cells."""
        while True:
            agent_cell = self.rng.integers(0, self.grid_size, size=2, dtype=np.int32)
            goal_cell = self.rng.integers(0, self.grid_size, size=2, dtype=np.int32)

            if not np.array_equal(agent_cell, goal_cell):
                self.agent_cell = agent_cell
                self.goal_cell = goal_cell
                self.agent_pos = self._cell_to_position(self.agent_cell, self.agent_size)
                self.goal_pos = self._cell_to_position(self.goal_cell, self.goal_size)
                return

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self._non_overlapping_positions()
        obs = self._render_image()
        info = {}
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"

        self.step_count += 1

        # Move agent one cell on the 4x4 grid
        if action == 0 and self.agent_cell[1] > 0:  # up
            self.agent_cell[1] -= 1
        elif action == 1 and self.agent_cell[1] < self.grid_size - 1:  # down
            self.agent_cell[1] += 1
        elif action == 2 and self.agent_cell[0] > 0:  # left
            self.agent_cell[0] -= 1
        elif action == 3 and self.agent_cell[0] < self.grid_size - 1:  # right
            self.agent_cell[0] += 1

        # Update pixel position to stay centered in the new cell
        self.agent_pos = self._cell_to_position(self.agent_cell, self.agent_size)

        # Check for termination
        terminated = self._is_on_goal()
        truncated = self.step_count >= self.max_steps
        reward = 1.0 if terminated else 0.0

        obs = self._render_image()
        info = {}

        return obs, reward, terminated, truncated, info

    def _is_on_goal(self) -> bool:
        """Return True if the agent is in the same grid cell as the goal."""
        return bool(np.array_equal(self.agent_cell, self.goal_cell))

    def _render_image(self) -> np.ndarray:
        """Render a simple RGB image with a triangle agent and square goal."""
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255  # white background

        # Draw a light 4x4 grid to make positions easier to see
        grid_color = np.array([200, 200, 200], dtype=np.uint8)
        cell_w = self.width // 4
        cell_h = self.height // 4
        # Vertical lines
        for i in range(1, 4):
            x = i * cell_w
            if 0 <= x < self.width:
                img[:, max(0, x - 1) : min(self.width, x + 1)] = grid_color
        # Horizontal lines
        for j in range(1, 4):
            y = j * cell_h
            if 0 <= y < self.height:
                img[max(0, y - 1) : min(self.height, y + 1), :] = grid_color

        # Draw goal(s)
        # Base class: single red square
        if isinstance(self, TwoGoalShapesEnv):
            # Goal 1: purple square
            gx1, gy1 = self.goal_pos
            gs1 = self.goal_size
            img[gy1 : gy1 + gs1, gx1 : gx1 + gs1] = np.array([160, 32, 240], dtype=np.uint8)

            # Goal 2: red cross inside its bounding box
            gx2, gy2 = self.goal2_pos
            s2 = self.goal2_size
            cx = gx2 + s2 // 2
            cy = gy2 + s2 // 2
            thickness = max(1, s2 // 5)

            # Horizontal bar of the cross
            y_start = max(0, cy - thickness // 2)
            y_end = min(self.height, cy + (thickness + 1) // 2)
            x_start = max(0, gx2)
            x_end = min(self.width, gx2 + s2)
            img[y_start:y_end, x_start:x_end] = np.array([255, 0, 0], dtype=np.uint8)

            # Vertical bar of the cross
            x_start_v = max(0, cx - thickness // 2)
            x_end_v = min(self.width, cx + (thickness + 1) // 2)
            y_start_v = max(0, gy2)
            y_end_v = min(self.height, gy2 + s2)
            img[y_start_v:y_end_v, x_start_v:x_end_v] = np.array([255, 0, 0], dtype=np.uint8)
        else:
            # Original: single red square
            gx, gy = self.goal_pos
            gs = self.goal_size
            img[gy : gy + gs, gx : gx + gs] = np.array([255, 0, 0], dtype=np.uint8)

        # Draw agent as a green right triangle inside its bounding box
        ax, ay = self.agent_pos
        s = self.agent_size
        for dy in range(s):
            # Triangle grows from left edge: x from 0 to dy
            max_dx = min(dy, s - 1)
            y = ay + dy
            if 0 <= y < self.height:
                x_start = ax
                x_end = ax + max_dx + 1
                x_start_clipped = max(0, x_start)
                x_end_clipped = min(self.width, x_end)
                if x_start_clipped < x_end_clipped:
                    img[y, x_start_clipped:x_end_clipped] = np.array([0, 200, 0], dtype=np.uint8)

        return img

    def render(self):
        """Render the environment.

        - If render_mode == 'rgb_array' or None: return an RGB image (H, W, 3).
        - If render_mode == 'human': show/update a matplotlib window, return None.
        """
        img = self._render_image()

        if self.render_mode is None or self.render_mode == "rgb_array":
            return img

        if self.render_mode == "human":
            if self._fig is None:
                self._fig, self._ax = plt.subplots()
                self._im = self._ax.imshow(img)
                self._ax.set_axis_off()
                plt.ion()
                self._fig.canvas.draw()
            else:
                self._im.set_data(img)
                self._fig.canvas.draw_idle()
            plt.pause(1.0 / self.metadata["render_fps"])
            return None

        # Fallback: just return the image
        return img



class TwoGoalShapesEnv(SimpleShapesEnv):
    """Like SimpleShapesEnv but with two goals:

    - Yellow square
    - Red circle

    Reaching either object yields reward 1.0 and ends the episode.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Second goal (same size as first)
        self.goal2_size = self.goal_size
        self.goal2_cell = np.array([0, 0], dtype=np.int32)
        self.goal2_pos = np.array([0, 0], dtype=np.int32)

    def _non_overlapping_positions(self) -> None:
        """Sample agent and two goals in three different grid cells."""
        while True:
            agent_cell = self.rng.integers(0, self.grid_size, size=2, dtype=np.int32)
            goal1_cell = self.rng.integers(0, self.grid_size, size=2, dtype=np.int32)
            goal2_cell = self.rng.integers(0, self.grid_size, size=2, dtype=np.int32)

            # Ensure all three are in distinct cells
            if (
                not np.array_equal(agent_cell, goal1_cell)
                and not np.array_equal(agent_cell, goal2_cell)
                and not np.array_equal(goal1_cell, goal2_cell)
            ):
                self.agent_cell = agent_cell
                self.goal_cell = goal1_cell
                self.goal2_cell = goal2_cell

                self.agent_pos = self._cell_to_position(self.agent_cell, self.agent_size)
                self.goal_pos = self._cell_to_position(self.goal_cell, self.goal_size)
                self.goal2_pos = self._cell_to_position(self.goal2_cell, self.goal2_size)
                return

    def _is_on_goal(self) -> bool:
        """Return True if the agent is in the same grid cell as any goal."""
        on_goal1 = np.array_equal(self.agent_cell, self.goal_cell)
        on_goal2 = np.array_equal(self.agent_cell, self.goal2_cell)
        return bool(on_goal1 or on_goal2)


    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._im = None


def make_env(**kwargs) -> SimpleShapesEnv:
    """Convenience constructor so you can do:

    env = make_env()
    """

    return SimpleShapesEnv(**kwargs)

def make_test_env(**kwargs) -> SimpleShapesEnv:
    """Convenience constructor so you can do:

    env = make_test_env()
    """

    return TwoGoalShapesEnv(**kwargs)