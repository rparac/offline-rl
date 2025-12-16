import io
from PIL import Image
import matplotlib
matplotlib.use('Agg')

from pathlib import Path
from tensordict.nn import TensorDictModule
import torch
from torch import nn
from torch.distributions import Categorical
from torchrl.data import LazyMemmapStorage, ReplayBuffer
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, QValueModule
from torchrl.objectives import DQNLoss, DiscreteIQLLoss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensordict import TensorDict

import numpy as np

from offline_rl.torchrl_practice.env_util import setup_visual_minecraft_with_wrapper
from offline_rl.torchrl_practice.networks import ActorNet, QNet, VNet


@torch.no_grad()
def evaluate_policy(env, policy, num_eval_episodes=2):
    """Calculate the mean cumulative reward over multiple episodes."""
    episode_rewards = []

    for _ in range(num_eval_episodes):
        eval_td = env.rollout(max_steps=20, policy=policy, auto_cast_to_device=True)
        episode_rewards.append(eval_td["next", "reward"].sum().item())

    return np.mean(episode_rewards)

def define_network(env, device):
    value_net = ValueOperator(
        in_keys=["observation"],
        out_keys=["state_value"],
        module=VNet()
    )

    # TODO: ValueModule has previously been used for IQL.

    # Use QValueModule instead of ValueOperator for DQN compatibility
    # QValueModule outputs ['action', 'action_value', 'chosen_action_value']
    q_net_base = TensorDictModule(
        QNet(),
        in_keys=["observation"],
        out_keys=["state_action_value"],
    )
    q_net = QValueModule(
        action_value_key="state_action_value",
        action_space="categorical",
    )

    actor = ProbabilisticActor(
        module=TensorDictModule(
            ActorNet(),
            in_keys=["observation"],
            out_keys=["logits"],
        ),
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        # return_log_prob=True,
    )  

    # Combine q_net_base and q_net into a single sequential module
    from tensordict.nn import TensorDictSequential as TDSeq
    q_net_combined = TDSeq(q_net_base, q_net)
    
    model = torch.nn.ModuleList([actor, q_net_combined, value_net]).to(device)

    # Initialize the model (since MLPs are Lazy Linear modules)
    with torch.no_grad():
        tensordict = env.reset().to(device)

        # model[1](tensordict)

        tensordict = tensordict.unsqueeze(0)
        for net in model:
            tensordict = net(tensordict)
            print(tensordict)
    return model

def define_loss_module_q_learning(model, gamma=0.99):
    loss_module = DQNLoss(
        model[1],
        loss_function="l2",
        action_space="categorical",
    )
    loss_module.set_keys(action_value="state_action_value")
    loss_module.make_value_estimator(gamma=gamma)
    return loss_module

def define_loss_module(model):
    # loss_module = IQLLoss(
    loss_module = DiscreteIQLLoss(
        model[0],
        model[1],
        action_space="categorical",
        value_network=model[2],
        loss_function="l2",
        temperature=0.1,
        expectile=0.99,
    )
    loss_module.make_value_estimator(gamma=0.9)
    return loss_module

def setup_data():
    dataset_path = Path("torch_dataset")
    storage = LazyMemmapStorage(max_size=10000)
    replay_buffer = ReplayBuffer(storage=storage)
    replay_buffer.load(dataset_path)
    return replay_buffer

def _pyplt_to_pil(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img

def _generate_q_table(q_table: np.ndarray) -> Image.Image:
    """
    Core plotting logic for visualizing a Q-table.
    Uses the 'gold standard' visualization originally defined in `offline_rl/q_learning.py`.
    """
    # Action names mapping based on env.py step() function:
    # Swapped UP and DOWN to match the debug rendering
    action_names = {
        0: 'RIGHT (→)',
        3: 'UP (↑)',
        2: 'LEFT (←)',
        1: 'DOWN (↓)',
    }

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    # Find global min and max for consistent color scaling
    vmin = q_table.min()
    vmax = q_table.max()

    # Create a heatmap for each action
    for action in range(q_table.shape[-1]):
        ax = axes[action]
        q_values = q_table[:, :, action]

        # Create heatmap
        im = ax.imshow(q_values, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)

        # Add colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Q(s,a)', rotation=270, labelpad=15, fontsize=10)

        # Set ticks and labels
        grid_size_y, grid_size_x = q_values.shape
        ax.set_xticks(np.arange(grid_size_x))
        ax.set_yticks(np.arange(grid_size_y))
        ax.set_xticklabels(np.arange(grid_size_x))
        ax.set_yticklabels(np.arange(grid_size_y))
        ax.set_xlabel('Column (X)', fontsize=11)
        ax.set_ylabel('Row (Y)', fontsize=11)
        ax.set_title(f'Q(s, a) for Action {action}: {action_names[action]}', fontsize=12, pad=15, weight='bold')

        # Add text annotations with values
        for i in range(grid_size_y):
            for j in range(grid_size_x):
                value = q_values[i, j]
                text_color = 'white' if value < (vmin + vmax) / 2 else 'black'
                ax.text(j, i, f'{value:.2f}',
                        ha="center", va="center", color=text_color, fontsize=10, weight='bold')

    plt.suptitle('Q-Table for Each Action', fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    img = _pyplt_to_pil(plt)
    plt.close()
    return img


@torch.no_grad()
def generate_q_table(source, grid_size: int = 4, num_actions: int = 4) -> Image.Image:
    """
    Unified Q-table visualization.

    - If `source` is a numpy array, it is treated as a precomputed Q-table with
      shape (grid_y, grid_x, num_actions).
    - If `source` is a Q-network (ValueOperator / nn.Module), we compute the Q-table
      over all discrete (row, col) states and then plot using the same visualization.
    """
    # Case 1: already a Q-table (numpy array)
    if isinstance(source, np.ndarray):
        return _generate_q_table(source)

    q_net = source

    # Create all possible states (row, col) pairs
    states = []
    for row in range(grid_size):
        for col in range(grid_size):
            states.append((row, col))

    # Create observations: shape [num_states, 2, grid_size] where each dimension is one-hot
    observations = []
    for row, col in states:
        obs = torch.tensor([row, col], dtype=torch.float32)
        observations.append(obs)

    # Get device from the first module in the sequential (or the module itself if not sequential)
    device = next(q_net[0].parameters()).device if hasattr(q_net, '__getitem__') else next(q_net.parameters()).device
    observations = torch.stack(observations).to(device)  # [num_states, 2, grid_size]

    # Compute Q values for all state-action pairs
    q_values = torch.zeros(len(states), num_actions)

    td = TensorDict({
        "observation": observations,
    }, batch_size=[len(states)])

    q_net(td)
    # TODO: should this be state_action_value?
    q_values = td["state_action_value"].squeeze(-1).cpu()  # [num_states]

    # Reshape into (grid_size, grid_size, num_actions) Q-table
    q_table = np.zeros((grid_size, grid_size, num_actions), dtype=np.float32)
    for idx, (row, col) in enumerate(states):
        q_table[row, col, :] = q_values[idx].cpu().numpy()

    return _generate_q_table(q_table)


def _generate_v_table(v_table: np.ndarray) -> Image.Image:
    """
    Plot a state-value table V(s) as a single heatmap, styled similarly to the Q-table plots.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    vmin = v_table.min()
    vmax = v_table.max()

    im = ax.imshow(v_table, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("V(s)", rotation=270, labelpad=15, fontsize=10)

    grid_size_y, grid_size_x = v_table.shape
    ax.set_xticks(np.arange(grid_size_x))
    ax.set_yticks(np.arange(grid_size_y))
    ax.set_xticklabels(np.arange(grid_size_x))
    ax.set_yticklabels(np.arange(grid_size_y))
    ax.set_xlabel("Column (X)", fontsize=11)
    ax.set_ylabel("Row (Y)", fontsize=11)
    ax.set_title("State Value Function V(s)", fontsize=14, weight="bold", pad=15)

    # Add text annotations
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            value = v_table[i, j]
            text_color = "white" if value < (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color=text_color, fontsize=10, weight="bold")

    plt.tight_layout()
    img = _pyplt_to_pil(plt)
    plt.close()
    return img


@torch.no_grad()
def generate_v_table(source, grid_size: int = 4) -> Image.Image:
    """
    Visualize a state-value function V(s) over a discrete grid world.

    - If `source` is a numpy array, it is treated as a precomputed V-table with
      shape (grid_y, grid_x).
    - If `source` is a value network (ValueOperator / nn.Module), we compute V(s)
      over all discrete (row, col) states and then plot using the same style.
    """
    # Case 1: already a V-table (numpy array)
    if isinstance(source, np.ndarray):
        return _generate_v_table(source)

    v_net = source

    # Create all possible states (row, col) pairs
    states = []
    for row in range(grid_size):
        for col in range(grid_size):
            states.append((row, col))

    # Create observations: shape [num_states, 2, grid_size] where each dimension is one-hot
    observations = []
    for row, col in states:
        obs = torch.tensor([row, col], dtype=torch.float32)
        observations.append(obs)

    observations = torch.stack(observations).to(source.device)  # [num_states, 2, grid_size]

    # Compute V values for all states
    td = TensorDict({"observation": observations}, batch_size=[len(states)])
    v_net(td)
    v_values = td["state_value"].squeeze(-1).cpu()  # [num_states]

    # Reshape into (grid_size, grid_size) V-table
    v_table = np.zeros((grid_size, grid_size), dtype=np.float32)
    for idx, (row, col) in enumerate(states):
        v_table[row, col] = v_values[idx].item()

    return _generate_v_table(v_table)
