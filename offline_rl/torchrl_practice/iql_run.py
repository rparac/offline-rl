import numpy as np
from torchrl.objectives import SoftUpdate
from torchrl.record.loggers.wandb import WandbLogger
from tqdm.auto import tqdm
from pathlib import Path
import torch
import wandb

from offline_rl.torchrl_practice.torch_rl_utils import define_network, define_loss_module, evaluate_policy, setup_data, setup_visual_minecraft_with_wrapper, generate_q_table, generate_v_table


device = "cuda"

# TODO: probably remove this
env = setup_visual_minecraft_with_wrapper()
model = define_network(env, device)
model = model.to(device)


loss_module = define_loss_module(model)

batch_size = 256
lr = 3e-4
tau = 0.005
target_net_updater = SoftUpdate(loss_module, tau=tau)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
replay_buffer = setup_data()

iterations = 50_000  # Set to 50_000 to reproduce the results below
eval_interval = 10_000

# Initialize TorchRL WandbLogger
logger = WandbLogger(
    exp_name="iql-training",
    project="offline-rl-iql",
    save_dir="./logs",
    offline=False,
)

# Log hyperparameters
logger.log_hparams({
    "batch_size": batch_size,
    "lr": lr,
    "tau": tau,
    "iterations": iterations,
    "eval_interval": eval_interval,
    "device": device,
})

loss_logs = []
eval_reward_logs = []
pbar = tqdm(range(iterations))

for i in pbar:
    # 1) Sample data from the dataset
    data = replay_buffer.sample(batch_size=batch_size)
    data = data.to(device)

    optimizer.zero_grad()
    # 2) Compute loss l = L_V + L_Q + L_pi
    loss_dict = loss_module(data)
    loss = loss_dict["loss_value"] + loss_dict["loss_qvalue"] + loss_dict["loss_actor"]
    loss_logs.append(loss.item())

    # 3) Backpropagate the gradients
    loss.backward()
    
    # Log gradient norm
    total_norm = 0.0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    if param_count > 0:
        total_norm = total_norm ** (1. / 2)
        logger.log_scalar("train/grad_norm", total_norm, step=i)
    
    optimizer.step()  # Update V(s), Q(a, s), pi(a|s)
    target_net_updater.step()  # Update the target Q-network
    
    # Log loss components
    logger.log_scalar("train/loss_total", loss.item(), step=i)
    logger.log_scalar("train/loss_value", loss_dict["loss_value"].item(), step=i)
    logger.log_scalar("train/loss_qvalue", loss_dict["loss_qvalue"].item(), step=i)
    logger.log_scalar("train/loss_actor", loss_dict["loss_actor"].item(), step=i)
    logger.log_scalar("train/step", i, step=i)

    # Evaluate the policy
    if i % eval_interval == 0:
        # Visualize Q-table at the end of training
        print("Generating Q-table visualization...")
        q_table_img = generate_q_table(model[1], grid_size=4, num_actions=4)
        
        eval_reward = evaluate_policy(env, model[0])
        eval_reward_logs.append(eval_reward)
        
        v_table_path = f"offline_rl/imgs/new_v_table_visualization_{i}.png"
        v_table_img = generate_v_table(model[2], grid_size=4)
        
        # Log evaluation metrics
        logger.log_scalar("eval/mean_reward", eval_reward, step=i)
        logger.log_scalar("eval/step", i, step=i)
        
        # Log visualizations using wandb (via logger's experiment object)
        if q_table_img is not None:
            logger.experiment.log({
                "eval/q_table_visualization": wandb.Image(q_table_img),
            }, step=i)
        if v_table_img is not None:
            logger.experiment.log({
                "eval/v_table_visualization": wandb.Image(v_table_img),
            }, step=i)
        
        pbar.set_description(
            f"Loss: {loss_logs[-1]:.1f}, Avg return: {eval_reward_logs[-1]:.1f}"
        )

pbar.close()

# Visualize Q-table at the end of training
print("Generating Q-table visualization...")
final_q_table_img = generate_q_table(model[1], grid_size=4, num_actions=4)

# Log final visualization
if final_q_table_img is not None:
    logger.experiment.log({
        "final/q_table_visualization": wandb.Image(final_q_table_img),
    }, step=iterations)

print("Training complete!")