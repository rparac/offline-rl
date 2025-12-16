import warnings

from PIL import Image
from torch.distributions import Categorical

from offline_rl.torchrl_practice.env_util import setup_visual_minecraft_with_wrapper, setup_visual_minecraft
from offline_rl.torchrl_practice.networks import ActorNet, VNet
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
from torchrl.record.loggers.wandb import WandbLogger
from tqdm import tqdm

def record_policy_video(policy, max_steps=10, seed=42, device=None):
    """
    Record a policy rollout as a video and return frames for logging.
    
    Args:
        policy: The policy to evaluate
        max_steps: Maximum number of steps in the rollout
        seed: Random seed for reproducibility
        device: Device to run policy on
        
    Returns:
        frames: Tensor of shape (T, C, H, W) representing video frames
    """
    
    # Create a separate environment for rendering with rgb_array mode
    render_env = setup_visual_minecraft_with_wrapper(device=device)

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

    # Convert frames to tensor format expected by TorchRL logger
    # Shape: (T, H, W, C) -> (T, C, H, W)
    frames_array = np.stack(imgs)  # Shape: (T, H, W, C)
    frames_array = frames_array.transpose(0, 3, 1, 2)  # Shape: (T, C, H, W)
    return frames_array


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

    frames_per_batch = 8192
    # frames_per_batch = 128
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



    # Initialize TorchRL WandbLogger
    logger = WandbLogger(
        exp_name="ppo-training",
        project="offline-rl-ppo",
        save_dir="./logs",
        offline=False,
    )
    
    # Log hyperparameters
    logger.log_hparams({
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
    })

    num_steps_lifetime = 0
    num_terminated_lifetime = 0
    num_truncated_lifetime = 0

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in tqdm(enumerate(collector)):
        # Count number of agent steps sampled
        num_steps = frames_per_batch
        num_steps_lifetime += num_steps

        done_mask = tensordict_data["next", "done"]

        mean_episode_reward_batch = tensordict_data["next", "episode_reward"][done_mask].mean().item()
        mean_episode_steps_batch = tensordict_data["next", "step_count"][done_mask].float().mean().item()

        num_truncated_batch = tensordict_data["next", "truncated"].sum().item()
        num_terminated_batch = tensordict_data["next", "terminated"].sum().item()
        

        num_steps_lifetime += num_steps
        
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
                
                # Log loss components
                logger.log_scalar("train/loss_objective", loss_vals["loss_objective"].item(), step=i)
                logger.log_scalar("train/loss_critic", loss_vals["loss_critic"].item(), step=i)
                logger.log_scalar("train/loss_entropy", loss_vals["loss_entropy"].item(), step=i)
                logger.log_scalar("train/loss_total", loss_value.item(), step=i)

        logs["episode_reward"].append(mean_episode_reward_batch)
        # pbar.update(tensordict_data.numel())
        logs["episode_steps"].append(mean_episode_steps_batch)
        logs["lr"].append(optim.param_groups[0]["lr"])
        
        # Log training metrics
        logger.log_scalar("train/episode_reward_mean", logs["episode_reward"][-1], step=i)
        logger.log_scalar("train/learning_rate", logs["lr"][-1], step=i)
        logger.log_scalar("train/episode_steps_mean", logs["episode_steps"][-1], step=i)
        logger.log_scalar("train/batch", i, step=i)
        logger.log_scalar("train/num_steps_sampled_batch", frames_per_batch, step=i)
        logger.log_scalar("train/num_steps_sampled_lifetime", num_steps_lifetime, step=i)
        logger.log_scalar("train/num_terminated_sampled_batch", num_terminated_batch, step=i)
        logger.log_scalar("train/num_truncated_sampled_batch", num_truncated_batch, step=i)
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
                done_mask = eval_rollout["next", "done"]
                eval_reward_mean = eval_rollout["next", "episode_reward"][done_mask].mean().item()
                logs["eval episode reward"].append(eval_reward_mean)
                
                # Record policy video for visualization
                try:
                    video_frames = record_policy_video(policy, max_steps=10, seed=42, device=device)
                    logger.log_video("eval/policy_video", video_frames, step=i, fps=4)
                except Exception as e:
                    # If video recording fails, still log other metrics
                    print(f"Warning: Failed to record policy video: {e}")
                
                # Log evaluation metrics
                logger.log_scalar("eval/episode_reward_mean", logs["eval episode reward"][-1], step=i)
                logger.log_scalar("eval/batch", i, step=i)
                
                del eval_rollout

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()
