import matplotlib
matplotlib.use('Agg')

from pathlib import Path
import warnings
import numpy as np
from tensordict.nn import TensorDictModule
import torch
from torch import nn
import torchrl
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, ReplayBuffer
from torchrl.envs.utils import RandomPolicy
from tensordict import TensorDict
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from offline_rl.env_util import setup_visual_minecraft_with_wrapper

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

env = setup_visual_minecraft_with_wrapper()
env.set_seed(seed)

max_episode_steps = 100

# Load Q-table and create a policy based on it
q_table_path = Path("artifacts/q_table.npy")
q_table = np.load(q_table_path)
print(f"Loaded Q-table with shape: {q_table.shape}")

class QTablePolicy(nn.Module):
    """Policy that selects actions using argmax over a Q-table."""
    
    def __init__(self, q_table):
        super().__init__()
        # Register as buffer so it moves with the model to the correct device
        self.register_buffer('q_table', torch.from_numpy(q_table).float())
        # Q-table shape: (4, 4, 4) = (row, col, action)
        
    def forward(self, observation):
        """
        Args:
            observation: observation tensor with shape [batch, 2, 4] where each dimension is one-hot encoded
                        observation[:, 0] is row one-hot, observation[:, 1] is col one-hot

            # shape [batch, 2]
        Returns:
            action: [batch] tensor with action indices
        """
        
        squeezed = False
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
            squeezed = True

        row_indices = observation[:, 0] # [batch]
        row_indices = row_indices.long()
        col_indices = observation[:, 1] # [batch]
        col_indices = col_indices.long()
        
        # Use advanced indexing to get Q-values for all states at once
        # q_table[row_indices, col_indices, :] gives [batch, num_actions]
        q_values = self.q_table[row_indices, col_indices, :]  # [batch, num_actions]
        
        # Select action with highest Q-value
        actions = q_values.argmax(dim=-1)  # [batch]
        
        if squeezed:
            actions = actions.squeeze(0)

        return actions


if __name__ == "__main__":
    # Create the policy module
    q_table_policy_module = QTablePolicy(q_table)

    # Wrap it in a TensorDictModule to work with torchrl
    policy = TensorDictModule(
        q_table_policy_module,
        in_keys=["observation"],
        out_keys=["action"],
    )

    collector = SyncDataCollector(
        env, 
        policy, 
        frames_per_batch=100,
        total_frames=1000,
        device="cuda",
    )

    storage = LazyMemmapStorage(max_size=10000)
    replay_buffer = ReplayBuffer(storage=storage)

    for i, data in enumerate(collector):
        replay_buffer.extend(data)
        print(f"Collected batch {i}, Buffer size: {len(replay_buffer)}")

    dataset_path = Path("torch_dataset")
    replay_buffer.dump(dataset_path)

    print("Data collection complete")


    def visualize_state_action_heatmaps(
        replay_buffer: ReplayBuffer,
        grid_size: int = 4,
        num_actions: int = 4,
        save_path: str = "offline_rl/imgs/state_action_heatmaps.png",
    ):
        """Visualize state visitation counts split by action as 4 heatmaps (one per action)."""

        total = len(replay_buffer)
        if total == 0:
            print("Replay buffer is empty, skipping state-action heatmap visualization.")
            return

        # Sample the whole buffer (or a subset if you later want)
        batch = replay_buffer[:]

        obs = batch["observation"]  # [N, 2, 4]
        actions = batch["action"]   # [N] or [N, 1]

        # Ensure actions are 1D
        if actions.dim() > 1:
            actions = actions.squeeze(-1)

        # Decode row/col from one-hot
        rows = obs[:, 0].cpu().numpy()
        cols = obs[:, 1].cpu().numpy()
        acts = actions.cpu().numpy()

        # Count visits: [grid_size, grid_size, num_actions]
        counts = np.zeros((grid_size, grid_size, num_actions), dtype=np.int32)
        for r, c, a in zip(rows, cols, acts):
            if 0 <= r < grid_size and 0 <= c < grid_size and 0 <= a < num_actions:
                counts[int(r), int(c), int(a)] += 1

        # Plot 2x2 heatmaps, one per action
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes = axes.flatten()

        # Swap UP and DOWN labels to match the updated Q-table visualization
        action_names = ["RIGHT (0)", "DOWN (1)", "LEFT (2)", "UP (3)"]

        # Shared color scale across all actions
        vmin = 0
        vmax = counts.max() if counts.max() > 0 else 1

        for a in range(num_actions):
            ax = axes[a]
            grid = counts[:, :, a]

            im = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Annotate counts
            for r in range(grid_size):
                for c in range(grid_size):
                    val = grid[r, c]
                    if val > 0:
                        ax.text(
                            c,
                            r,
                            str(val),
                            ha="center",
                            va="center",
                            color="white" if val < vmax * 0.5 else "black",
                            fontsize=8,
                            fontweight="bold",
                        )

            ax.set_title(f"Action {action_names[a]}")
            ax.set_xticks(range(grid_size))
            ax.set_yticks(range(grid_size))
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"State-action heatmaps saved to {save_path}")

    # Create visualizations after dataset creation
    visualize_state_action_heatmaps(replay_buffer)