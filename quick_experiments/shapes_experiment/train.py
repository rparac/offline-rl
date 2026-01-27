# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
from collections import deque
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from offline_rl.clean_rl.buffers import ReplayBuffer
from quick_experiments.shapes_experiment.env import make_env, make_test_env, TwoGoalShapesEnv

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "QuickShapes-v0"
    """the id of the environment"""
    total_timesteps: int = 100_000
    """total timesteps of the experiments"""
    buffer_size: int = 10_000
    """the replay memory buffer size"""  # smaller than in original paper but evaluation is done only for 100k steps anyway
    gamma: float = 0.9
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 1_000
    """timestep to start learning"""
    policy_lr: float = 2.14965825658665e-05
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1.2729682271260388e-04
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 1
    """the frequency of training updates"""
    target_update_frequency: int = 16
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    starting_alpha: float = 0.04199973365314628
    """the starting value of the entropy coefficient with autotune"""
    target_entropy_scale: float = 0.3140753169818932
    """coefficient for scaling the autotune entropy target"""
    num_envs: int = 8
    """the number of environments to run in parallel"""


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_env_fn(capture_video, idx, run_name):
    def thunk():
        env = make_env(render_mode="rgb_array")
        if capture_video and idx == 0:
            trigger = lambda t: t % 1000 == 0
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=trigger)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
        h, w, c = env.observation_space.shape
        env = gym.wrappers.TransformObservation(
            env, 
            lambda obs: torch.from_numpy(np.transpose(obs, (2, 0, 1))).float() / 255.0,
            observation_space=gym.spaces.Box(
                low=0.0, 
                high=1.0, 
                shape=(c, h, w), 
                dtype=np.float32
            )
        )
        return env
    return thunk


def evaluate(actor: nn.Module, device: torch.device, global_step: int, writer: SummaryWriter) -> None:
    """Evaluate the current policy on the TwoGoalShapesEnv.

    Runs 10 episodes and reports how often the agent reaches the purple square vs red cross.
    """
    # Build evaluation env with same wrappers as training (TimeLimit + observation transform)
    env: gym.Env = make_test_env(render_mode="rgb_array")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
    h, w, c = env.observation_space.shape
    env = gym.wrappers.TransformObservation(
        env,
        lambda obs: torch.from_numpy(np.transpose(obs, (2, 0, 1))).float() / 255.0,
        observation_space=gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(c, h, w),
            dtype=np.float32,
        ),
    )
    # The unwrapped environment is the TwoGoalShapesEnv we care about for counting hits
    assert isinstance(env.unwrapped, TwoGoalShapesEnv)

    num_episodes = 10
    purple_hits = 0
    red_hits = 0

    try:
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                # Obs is already NCHW float32 in [0, 1] thanks to the wrapper
                if isinstance(obs, torch.Tensor):
                    obs_t = obs.unsqueeze(0).to(device)
                else:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits = actor(obs_t)
                    action = logits.argmax(dim=1).cpu().numpy()[0]

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            # After episode ends, check which goal cell the agent ended in on the unwrapped env
            if np.array_equal(env.unwrapped.agent_cell, env.unwrapped.goal_cell):
                purple_hits += 1
            elif np.array_equal(env.unwrapped.agent_cell, env.unwrapped.goal2_cell):
                red_hits += 1
    finally:
        env.close()

    purple_rate = purple_hits / num_episodes
    red_rate = red_hits / num_episodes

    # Log to TensorBoard
    writer.add_scalar("eval/purple_goal_rate", purple_rate, global_step)
    writer.add_scalar("eval/red_goal_rate", red_rate, global_step)

    # Also print to stdout for quick inspection
    print(
        f"[eval @ step {global_step}] "
        f"purple_square: {purple_hits}/{num_episodes}, red_cross: {red_hits}/{num_episodes}"
    )

# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=4, stride=2)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 64))
        self.fc_q = layer_init(nn.Linear(64, envs.single_action_space.n))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals



class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=4, stride=2)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 64))
        self.fc_logits = layer_init(nn.Linear(64, envs.single_action_space.n))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        # W&B sweeps typically use CLI-style hyphenated names (e.g. `policy-lr`),
        # while our Python dataclass uses underscore names (e.g. `policy_lr`).
        # If we log `vars(args)` directly, W&B ends up showing both variants.
        wandb_config = {k.replace("_", "-"): v for k, v in vars(args).items()}
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=wandb_config,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    seen_rewards = deque(maxlen=100)
    seen_lengths = deque(maxlen=100)


    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = [make_env_fn(capture_video=args.capture_video, idx=i, run_name=run_name) for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(envs)
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.tensor(np.log(args.starting_alpha), device=device, requires_grad=True)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    can_log = False
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos:
            terminated_episodes = infos["_episode"]

            rewards_to_log = infos["episode"]["r"][terminated_episodes].tolist()
            seen_rewards.extend(rewards_to_log)
            lengths_to_log = infos["episode"]["l"][terminated_episodes].tolist()
            seen_lengths.extend(lengths_to_log)


            avg_reward = sum(seen_rewards) / len(seen_rewards)
            avg_length = sum(seen_lengths) / len(seen_lengths)
            writer.add_scalar("charts/episodic_return", avg_reward, global_step)
            writer.add_scalar("charts/episodic_length", avg_length, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        # real_next_obs = next_obs.copy()
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                can_log = True
                data = rb.sample(args.batch_size)
                # CRITIC training
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target)

                # use Q-values only for the taken actions
                qf1_values = qf1(data.observations)
                qf2_values = qf2(data.observations)
                qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
                qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                _, log_pi, action_probs = actor.get_action(data.observations)
                with torch.no_grad():
                    qf1_values = qf1(data.observations)
                    qf2_values = qf2(data.observations)
                    min_qf_values = torch.min(qf1_values, qf2_values)
                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    # reuse action probabilities for temperature loss
                    alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_update_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0 and can_log:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("steps_per_second:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/steps_per_second", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        # Periodic evaluation on TwoGoalShapesEnv
        if global_step > 0 and global_step % 1000 == 0:
            evaluate(actor, device, global_step, writer)

    envs.close()
    writer.close()