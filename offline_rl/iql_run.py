import numpy as np
from torchrl.objectives import SoftUpdate
from tqdm.auto import tqdm
from pathlib import Path
import torch

from offline_rl.torch_rl_utils import define_network, define_loss_module, evaluate_policy, setup_data, setup_visual_minecraft_with_wrapper, visualize_q_table, visualize_v_table


device = "cuda"

# TODO: probably remove this
env = setup_visual_minecraft_with_wrapper()
model = define_network(env, device)
model = model.to(device)


loss_module = define_loss_module(model)

target_net_updater = SoftUpdate(loss_module, tau=0.005)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
replay_buffer = setup_data()

iterations = 10_000  # Set to 50_000 to reproduce the results below
eval_interval = 1_000

loss_logs = []
eval_reward_logs = []
pbar = tqdm(range(iterations))

for i in pbar:
    # 1) Sample data from the dataset
    data = replay_buffer.sample(batch_size=32)
    data = data.to(device)

    optimizer.zero_grad()
    # 2) Compute loss l = L_V + L_Q + L_pi
    loss_dict = loss_module(data)
    loss = loss_dict["loss_value"] + loss_dict["loss_qvalue"] + loss_dict["loss_actor"]
    loss_logs.append(loss.item())

    # 3) Backpropagate the gradients
    loss.backward()
    optimizer.step()  # Update V(s), Q(a, s), pi(a|s)
    target_net_updater.step()  # Update the target Q-network

    # Evaluate the policy
    if i % eval_interval == 0:
        # Visualize Q-table at the end of training
        print("Generating Q-table visualization...")
        visualize_q_table(model[1], grid_size=4, num_actions=4, save_path=f"offline_rl/imgs/new_q_table_visualization_{i}.png")
        eval_reward_logs.append(evaluate_policy(env, model[0]))
        pbar.set_description(
            f"Loss: {loss_logs[-1]:.1f}, Avg return: {eval_reward_logs[-1]:.1f}"
        )
        visualize_v_table(model[2], grid_size=4, save_path=f"offline_rl/imgs/new_v_table_visualization_{i}.png")
        pbar.set_description(
            f"Loss: {loss_logs[-1]:.1f}"
        )

pbar.close()

# Visualize Q-table at the end of training
print("Generating Q-table visualization...")
visualize_q_table(model[1], grid_size=4, num_actions=4, save_path="offline_rl/imgs/final_q_table_visualization.png")
print("Training complete!")