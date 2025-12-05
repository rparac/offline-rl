import numpy as np
from tensordict.nn import TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import ExplorationType
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import SoftUpdate
from tqdm.auto import tqdm
from pathlib import Path
import torch
import wandb
import time

from offline_rl.torch_rl_utils import define_loss_module_q_learning, define_network, define_loss_module, evaluate_policy, setup_data, setup_visual_minecraft_with_wrapper, visualize_q_table, visualize_v_table


device = "cuda"

# TODO: probably remove this
env = setup_visual_minecraft_with_wrapper()
model = define_network(env, device)
model = model.to(device)


action_spec = env.action_spec
q_actor = QValueActor(
    module=model[1], 
    in_keys=["observation"],
    action_space="categorical",
)
egreedy_module = EGreedyModule(
    spec=action_spec,
    eps_init=1.0,
    eps_end=0.1,
    annealing_num_steps=5000,
)
exploring_policy = TensorDictSequential(
    q_actor, egreedy_module
)
gamma = 0.99  # Discount factor
loss_module = define_loss_module_q_learning(model, gamma=gamma)
target_net_updater = SoftUpdate(loss_module, tau=0.005)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=10000))

iterations = 50_000  # Set to 50_000 to reproduce the results below
frames_per_batch = 10
eval_interval = 500
batch_size = 32
learning_rate = 0.0003
tau = 0.005  # Target network soft update coefficient

num_batched_iterations = iterations // frames_per_batch

# Initialize wandb
wandb.init(
    project="vq-vae-dqn",
    config={
        "iterations": iterations,
        "frames_per_batch": frames_per_batch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "tau": tau,
        "eps_init": 1.0,
        "eps_end": 0.05,
        "annealing_num_steps": 10000,
        "eval_interval": eval_interval,
        "replay_buffer_size": 10000,
    },
    name=f"dqn_gamma{gamma}_lr{learning_rate}",
)

collector = SyncDataCollector(
    env,
    policy=exploring_policy,
    frames_per_batch=frames_per_batch,
    total_frames=iterations,
    device="cuda",
)

loss_logs = []
start_time = time.time()
total_frames = 0

for i, data in enumerate(tqdm(collector, total=num_batched_iterations)):
    replay_buffer.extend(data)
    
    # Log episode statistics from data
    if "next" in data.keys() and "reward" in data["next"].keys():
        episode_rewards = data["next", "reward"]
        if episode_rewards.numel() > 0:
            wandb.log({
                "episode/reward_mean": episode_rewards.mean().item(),
            }, step=i)
    
    # Log done statistics
    if "next" in data.keys() and "done" in data["next"].keys():
        dones = data["next", "done"]
        if dones.numel() > 0:
            done_rate = dones.float().mean().item()
            wandb.log({"episode/done_rate": done_rate}, step=i)


    optimizer.zero_grad()
    sampled_tensordict = replay_buffer.sample(batch_size=batch_size).to(device)

    loss_td = loss_module(sampled_tensordict)
    loss_value = loss_td["loss"]
    loss_logs.append(loss_value.item())
    
    # Log all metrics from loss TensorDict (torchrl provides additional metrics)
    log_dict = {
        "train/loss": loss_value.item(),
        "train/step": i,
        "train/frames": (i + 1) * frames_per_batch,
        "train/replay_buffer_size": len(replay_buffer),
    }

    # Log epsilon value
    wandb.log({
        "train/epsilon": egreedy_module.eps.item(),
    }, step=i)
    
    # Add any other metrics from loss_td
    for key in loss_td.keys():
        if key != "loss" and isinstance(loss_td[key], torch.Tensor):
            if loss_td[key].numel() == 1:
                log_dict[f"train/{key}"] = loss_td[key].item()
    
    # Log gradient norm if available
    total_norm = 0.0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    if param_count > 0:
        total_norm = total_norm ** (1. / 2)
        log_dict["train/grad_norm"] = total_norm
    
    wandb.log(log_dict, step=i)

    loss_value.backward()
    optimizer.step()
    target_net_updater.step()
    egreedy_module.step()
    
    total_frames += frames_per_batch

    if i % eval_interval == 0:
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            # Visualize Q-table at the end of training
            print("Generating Q-table visualization...")
            viz_path = f"offline_rl/imgs/deep_q_table_visualization_{i}.png"
            visualize_q_table(model[1], grid_size=4, num_actions=4, save_path=viz_path)
            
            # Log Q-table visualization to wandb
            if Path(viz_path).exists():
                wandb.log({
                    "q_table_visualization": wandb.Image(viz_path),
                }, step=i)
            
            # Evaluate policy if evaluate_policy function is available
            eval_reward = evaluate_policy(env, exploring_policy, num_eval_episodes=2)
            wandb.log({
                "eval/mean_reward": eval_reward,
            }, step=i)

# Log final statistics
elapsed_time = time.time() - start_time

wandb.finish()
print(f"Training complete! Total time: {elapsed_time:.2f} seconds")