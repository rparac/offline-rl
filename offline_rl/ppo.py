import warnings

from PIL import Image
from torch.distributions import Categorical

from offline_rl.env_util import setup_visual_minecraft_with_wrapper, setup_visual_minecraft
from offline_rl.networks import ActorNet, VNet
warnings.filterwarnings("ignore")
from torch import multiprocessing


from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import numpy as np
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import MultiSyncDataCollector, MultiaSyncDataCollector, SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.envs import ParallelEnv
from torchrl.modules import ProbabilisticActor
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import wandb

def record_policy_video(policy, max_steps=10, seed=42):
    """
    Record a policy rollout as a video and return frames for wandb logging.
    
    Args:
        policy: The policy to evaluate
        max_steps: Maximum number of steps in the rollout
        seed: Random seed for reproducibility
        
    Returns:
        frames: Numpy array of shape (T, H, W, C) representing video frames
    """
    
    # Create a separate environment for rendering with rgb_array mode
    render_env = setup_visual_minecraft_with_wrapper(device=torch.device("cpu"))

    td = render_env.reset()

    # Img is a numpy array of shape (512, 512, 3)
    imgs = []
    img = render_env.render()
    imgs.append(img)
    
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        
        # Get action from policy using TensorDict format
        with torch.no_grad():
            action_td = policy(td.to(device))
        
        # Step environment
        td = render_env.step(action_td)
        
        done = td["next", "terminated"].item() or td["next", "truncated"].item()
        img = render_env.render()
        imgs.append(img)
        step_count += 1
    
    render_env.close()

    # Convert frames to numpy array format expected by wandb
    frames_array = np.stack(imgs)  # Shape: (T, H, W, C)
    # wandb.Video expects shape (T, H, W, C) where T is time, H/W are height/width, C is channels
    frames_array = frames_array.transpose(0, 3, 1, 2)

    video = wandb.Video(frames_array, fps=4, format="mp4")
    return video


if __name__ == "__main__":

    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    num_workers = 4
    num_envs_per_worker = 4  # Number of parallel environments per worker for vectorization
    lr = 3e-4
    max_grad_norm = 1.0

    # frames_per_batch = 8192
    frames_per_batch = 128
    # For a complete training, bring the number of frames up to 1M
    # total_frames = 50_000
    total_frames = 5_000_000

    sub_batch_size = 8192 # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 10  # optimization steps per batch of data collected
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 0.1

    env = setup_visual_minecraft_with_wrapper(device=device)


    actor_net = ActorNet().to(device)
    policy = ProbabilisticActor(
        module=TensorDictModule(
            actor_net,
            in_keys=["observation"],
            out_keys=["logits"],
        ),
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        return_log_prob=True,
    )

    value_net = VNet().to(device)
    value_module = TensorDictModule(
        value_net,
        in_keys=["observation"],
        out_keys=["state_value"],
    )

    # collector = SyncDataCollector(
    #     create_env_fn=lambda: setup_visual_minecraft_with_wrapper(device=device),
    #     policy=policy,
    #     frames_per_batch=frames_per_batch,
    #     total_frames=total_frames,
    #     split_trajs=True,
    #     device=device,
    # )

    # Create vectorized environment factory function
    def create_vectorized_env():
        """Create a vectorized environment with multiple parallel instances."""
        # Use CPU for environments in worker processes; collector handles device placement
        return ParallelEnv(
            num_workers=num_envs_per_worker,
            create_env_fn=lambda: setup_visual_minecraft_with_wrapper(device=torch.device("cpu")),
        )

    # collector = MultiaSyncDataCollector(
    #     create_env_fn=create_vectorized_env,
    collector = MultiSyncDataCollector(
        create_env_fn=create_vectorized_env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=True, # split trajectories into an episode view
        reset_at_each_batch=True, # do not make episode collection stateful;  this would make metrics tracking more difficult
        device=device,
        num_workers=num_workers,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(
        value_network=value_module,
        gamma=gamma,
        lmbda=lmbda,
        average_gae=True,
        device=device,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coef=0.5,
        loss_critic_type="smooth_l1",
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    logs = defaultdict(list)
    # pbar = tqdm(total=total_frames * frames_per_batch)
    eval_str = ""



    # wandb_video = record_policy_video(policy, max_steps=100, seed=42)
    # print("Done")

    # Initialize wandb
    wandb.init(
        project="offline-rl-ppo",
        config={
            "lr": lr,
            "max_grad_norm": max_grad_norm,
            "frames_per_batch": frames_per_batch,
            "total_frames": total_frames,
            "sub_batch_size": sub_batch_size,
            "num_epochs": num_epochs,
            "clip_epsilon": clip_epsilon,
            "gamma": gamma,
            "lmbda": lmbda,
            "entropy_eps": entropy_eps,
            "device": str(device),
            "num_workers": num_workers,
            "num_envs_per_worker": num_envs_per_worker,
        }
    )

    num_steps_lifetime = 0
    num_terminated_lifetime = 0
    num_truncated_lifetime = 0

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in tqdm(enumerate(collector)):
        # Count number of agent steps sampled
        num_steps = frames_per_batch
        
        # Count number of episodes sampled (episodes that ended)
        num_terminated = tensordict_data["next", "terminated"].sum().item()
        num_truncated = tensordict_data["next", "truncated"].sum().item()

        num_steps_lifetime += num_steps
        num_terminated_lifetime += num_terminated
        num_truncated_lifetime += num_truncated
        
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            tensordict_data = tensordict_data.to(device)
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()
                
                # Log loss components to wandb
                wandb.log({
                    "train/loss_objective": loss_vals["loss_objective"].item(),
                    "train/loss_critic": loss_vals["loss_critic"].item(),
                    "train/loss_entropy": loss_vals["loss_entropy"].item(),
                    "train/loss_total": loss_value.item(),
                })

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        # pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        
        # Log training metrics to wandb
        wandb.log({
            "train/reward": logs["reward"][-1],
            "train/learning_rate": logs["lr"][-1],
            "train/batch": i,
            "train/num_steps_sampled_batch": num_steps,
            "train/num_steps_sampled_lifetime": num_steps_lifetime,
            "train/num_terminated_sampled_batch": num_terminated,
            "train/num_terminated_sampled_lifetime": num_terminated_lifetime,
            "train/num_truncated_sampled_batch": num_truncated,
            "train/num_truncated_sampled_lifetime": num_truncated_lifetime,
        })
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(1000, policy)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f})"
                )
                
                # Record policy video for visualization
                try:
                    video = record_policy_video(policy, max_steps=10, seed=42)
                    # Log video to wandb
                    # wandb.Video expects shape (T, H, W, C) or (T, C, H, W)
                    # Our frames are already in (T, H, W, C) format
                    wandb.log({
                        "eval/reward_mean": logs["eval reward"][-1],
                        "eval/reward_sum": logs["eval reward (sum)"][-1],
                        "eval/batch": i,
                        "eval/policy_video": video,
                    })
                except Exception as e:
                    # If video recording fails, still log other metrics
                    print(f"Warning: Failed to record policy video: {e}")
                    wandb.log({
                        "eval/reward_mean": logs["eval reward"][-1],
                        "eval/reward_sum": logs["eval reward (sum)"][-1],
                        "eval/batch": i,
                    })
                
                del eval_rollout
        # pbar.set_description(", ".join([eval_str, cum_reward_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()

    # Finish wandb run
    wandb.finish()