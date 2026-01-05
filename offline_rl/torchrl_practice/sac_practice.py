import warnings

from PIL import Image
from torch.distributions import Categorical
from torchrl.data import RandomSampler

from offline_rl.torchrl_practice.env_util import record_policy_video, setup_visual_minecraft_with_wrapper, setup_visual_minecraft
from offline_rl.torchrl_practice.networks import ActorNet, QNet, VNet
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
from torchrl.objectives import ClipPPOLoss, DiscreteSACLoss, SoftUpdate
from torchrl.objectives.value import GAE
from torchrl.record.loggers.wandb import WandbLogger
from tqdm import tqdm



# Create vectorized environment factory function
def create_vectorized_env():
    """Create a vectorized environment with multiple parallel instances."""
    # Use CPU for environments in worker processes; collector handles device placement
    return ParallelEnv(
        num_workers=num_envs_per_worker,
        create_env_fn=lambda: setup_visual_minecraft_with_wrapper(device=torch.device("cpu")),
    )

if __name__ == "__main__":
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    num_workers = 4
    num_envs_per_worker = 4
    num_epochs = 10

    buffer_size = 10000

    frames_before_update = 1000 # 10000
    frames_per_batch = 512 # 8192
    total_frames = 5_000_000

    learning_rate = 3e-4
    gamma = 0.99
    tau = 0.005
    max_grad_norm = 1.0

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
    ).to(device)

    qvalue = TensorDictModule(
        module=QNet(),
        in_keys=["observation"],
        out_keys=["action_value"]
    ).to(device)

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
        frames_per_batch=frames_before_update,
        total_frames=total_frames,
        device=device,
        num_workers=num_workers,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=buffer_size),
        sampler=RandomSampler(),
    )

    loss_module = DiscreteSACLoss(
        actor_network=policy,
        qvalue_network=qvalue,
        action_space="categorical",
        num_actions=env.action_spec.n,
    )
    loss_module.make_value_estimator(gamma=gamma)

    target_net_updater = SoftUpdate(loss_module, tau=tau)

    optim = torch.optim.Adam(loss_module.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    logs = defaultdict(list)
    # pbar = tqdm(total=total_frames * frames_per_batch)
    eval_str = ""



    # Initialize TorchRL WandbLogger
    logger = WandbLogger(
        exp_name="sac-training",
        project="offline-rl-sac",
        save_dir="./logs",
        offline=False,
    )
    
    # Log hyperparameters for SAC
    logger.log_hparams({
        "learning_rate": learning_rate,
        "tau": tau,
        "frames_per_batch": frames_per_batch,
        "total_frames": total_frames,
        "buffer_size": buffer_size,
        "gamma": gamma,
        "device": str(device),
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
        replay_buffer.extend(tensordict_data.cpu())
        
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(num_epochs):
            tensordict_data = replay_buffer.sample(frames_per_batch).to(device)
            loss_vals = loss_module(tensordict_data)

            loss_value = (
                loss_vals["loss_actor"]
                + loss_vals["loss_qvalue"]
                + loss_vals["loss_alpha"]
            )
            loss_value.backward()

            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

            
            # Log loss components
            logger.log_scalar("train/total_loss", loss_value.item(), step=i)
            logger.log_scalar("train/loss_actor", loss_vals["loss_actor"].item(), step=i)
            logger.log_scalar("train/loss_qvalue", loss_vals["loss_qvalue"].item(), step=i)
            logger.log_scalar("train/loss_alpha", loss_vals["loss_alpha"].item(), step=i)
            logger.log_scalar("train/entropy", loss_vals["entropy"].item(), step=i)
            logger.log_scalar("train/alpha", loss_vals["alpha"].item(), step=i)

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
                # eval_rollout = env.rollout(10, policy)
                # done_mask = eval_rollout["next", "done"]
                # eval_reward_mean = eval_rollout["next", "episode_reward"][done_mask].mean().item()
                # logs["eval episode reward"].append(eval_reward_mean)
                
                # Record policy video for visualization
                try:
                    video_frames, cummulative_reward = record_policy_video(policy, max_steps=10, seed=42, device=device)
                    logs["eval episode reward"].append(cummulative_reward)
                    logger.log_video("eval/policy_video", video_frames, step=i, fps=4)
                except Exception as e:
                    # If video recording fails, still log other metrics
                    print(f"Warning: Failed to record policy video: {e}")
                
                # Log evaluation metrics
                logger.log_scalar("eval/episode_reward_mean", logs["eval episode reward"][-1], step=i)
                logger.log_scalar("eval/batch", i, step=i)
                
                # del eval_rollout

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()
