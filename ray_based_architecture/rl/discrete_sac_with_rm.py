# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
from collections import deque
import random
import time
from dataclasses import dataclass

import gymnasium as gym
from tqdm import trange
import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from env.visual_minecraft.env import GridWorldEnv
from ray_based_architecture.env.clip_obs_and_labels_wrapper import BatchCLIPObsAndLabelsWrapper
from ray_based_architecture.env.rm_wrapper import RMWrapper
from ray_based_architecture.shared_memory.sac_replay_buffer import SACReplayBuffer
from ray_based_architecture.reward_machine.reward_machine import RewardMachine
from offline_rl.vlm.visual_minecraft_success_detector import (
    LABEL_BLUE_DIAMOND_GEM,
    LABEL_OPEN_RED_DOUBLE_DOOR,
    VISUAL_MINECRAFT_LABEL_ORDER,
    LABEL_GREY_YELLOW_PICKAXE,
    LABEL_ORANGE_YELLOW_MAGMA,
)



@dataclass
class Args:
    exp_name: str = "sac_visual_minecraft"
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
    env_id: str = "VisualMinecraft-v0"
    """the id of the environment"""
    total_timesteps: int = 100_000
    """total timesteps of the experiments"""
    buffer_size: int = 10_000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 512
    """the batch size of sample from the reply memory"""
    learning_starts: int = 2000
    """timestep to start learning"""
    policy_lr: float = 0.00004127483240070165
    """the learning rate of the policy network optimizer"""
    q_lr: float = 0.00019077115118374603
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 1
    """the frequency of training updates"""
    target_update_frequency: int = 64
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    starting_alpha: float = 0.2722738446422038
    """the starting value of the entropy coefficient with autotune"""
    target_entropy_scale: float = 0.2232439060872044
    """coefficient for scaling the autotune entropy target"""
    num_envs: int = 8
    """the number of environments to run in parallel"""
    max_inflight_label_requests: int = 2000
    """max number of outstanding reward-labeling requests to Ray Serve (backpressure)"""
    replay_buffer_num_cpus: int = 1
    """CPU resources reserved for the replay buffer actor"""
    replay_buffer_max_concurrency: int = 16
    """max concurrent requests to replay buffer (sample() gets priority via internal condition variables)"""
    use_multiple_agents: bool = False
    """If True, initializes/shuts down Ray for each run (needed for parallel sweeps). If False, uses existing Ray cluster."""

def make_env(env_id, seed, idx, capture_video, run_name):

    def thunk():
        items = ["pickaxe", "lava", "door", "gem", "empty"]
        # formula = "(F c0)", 5, "task0: visit({1})".format(*items)
        # formula = "F(c0 & F(c1))", 5, "task3: seq_visit({0}, {1})".format(*items)
        formula = "(F (c0 & F c2)) & (F (c3 & F c2))", 5, "pickaxe or gem then door"

        kwargs = {
            "formula": formula,
            "render_mode": "rgb_array",
            "state_type": "image",
            "normalize_env": False,
            "train": False,
            "use_dfa_state": False,
            "random_start": False,
        }

        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **kwargs)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **kwargs)
        env.action_space.seed(seed)
        return env

    return thunk


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
# NOTE: Using separate encoders for Actor and Critics (no shared parameters)
# Observations are Dict with "image_embedding" (CLIP, 768) and "rm_state" (discrete index)
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        clip_dim = envs.single_observation_space["image_embedding"].shape[0]
        num_rm_states = envs.single_observation_space["rm_state"].n
        
        self.clip_encoder = nn.Sequential(
            layer_init(nn.Linear(clip_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )
        # RM state is discrete, so use embedding
        self.rm_embedding = nn.Embedding(num_rm_states, 32)
        self.fc1 = layer_init(nn.Linear(128 + 32, 64))
        self.fc_q = layer_init(nn.Linear(64, envs.single_action_space.n))

    def forward(self, x):
        # x is a Dict with "image_embedding" and "rm_state"
        clip_obs = x["image_embedding"]
        rm_state = x["rm_state"].long()  # Convert to long for embedding
        
        clip_features = self.clip_encoder(clip_obs)
        rm_features = self.rm_embedding(rm_state)
        
        # Concatenate features
        combined = torch.cat([clip_features, rm_features], dim=1)
        x = F.relu(self.fc1(combined))
        q_vals = self.fc_q(x)
        return q_vals



class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        clip_dim = envs.single_observation_space["image_embedding"].shape[0]
        num_rm_states = envs.single_observation_space["rm_state"].n
        
        self.clip_encoder = nn.Sequential(
            layer_init(nn.Linear(clip_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )
        # RM state is discrete, so use embedding
        self.rm_embedding = nn.Embedding(num_rm_states, 32)
        self.fc1 = layer_init(nn.Linear(128 + 32, 64))
        self.fc_logits = layer_init(nn.Linear(64, envs.single_action_space.n))

    def forward(self, x):
        # x is a Dict with "image_embedding" and "rm_state"
        clip_obs = x["image_embedding"]
        rm_state = x["rm_state"].long()  # Convert to long for embedding
        
        clip_features = self.clip_encoder(clip_obs)
        rm_features = self.rm_embedding(rm_state)
        
        # Concatenate features
        combined = torch.cat([clip_features, rm_features], dim=1)
        x = F.relu(self.fc1(combined))
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


def train(args: Args):
    # Initialize Ray with local resources if running parallel sweeps
    # For single runs with a persistent cluster, skip this (use existing Ray instance)
    if args.use_multiple_agents:
        ray.init(ignore_reinit_error=True, log_to_driver=False)
        print("[Setup] Initialized local Ray instance for parallel sweep agent")
    
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
    seen_original_rewards = deque(maxlen=100)


    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = [make_env(args.env_id, args.seed, i, args.capture_video, run_name) for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(envs)
    envs = BatchCLIPObsAndLabelsWrapper(envs)
    
    # Create reward machine with 3 states:
    # u0 (initial) -> u1 (got pickaxe) -> uacc (got gem)
    rm = RewardMachine()
    rm.add_states(["u0", "u1", "u2", "u3", "uacc"])
    rm.set_u0("u0")
    rm.set_uacc("uacc")
    
    # u0 --[Grey and yellow pickaxe]--> u1
    # u1 --[Orange and yellow magma texture]--> uacc
    rm.add_transition("u0", "u1", (LABEL_GREY_YELLOW_PICKAXE,))
    rm.add_transition("u0", "u2", (LABEL_BLUE_DIAMOND_GEM,))
    rm.add_transition("u1", "u3", (LABEL_BLUE_DIAMOND_GEM,))
    rm.add_transition("u2", "u3", (LABEL_GREY_YELLOW_PICKAXE,))
    rm.add_transition("u3", "uacc", (LABEL_OPEN_RED_DOUBLE_DOOR,))
    
    # Build transition matrix with label order matching visual_minecraft_success_detector
    rm.build_transition_matrix(VISUAL_MINECRAFT_LABEL_ORDER)
    
    # Build transition matrix with label order matching visual_minecraft_success_detector
    rm.build_transition_matrix(VISUAL_MINECRAFT_LABEL_ORDER)
    
    print(f"[Setup] Reward machine with {len(rm.states)} states initialized")
    print(rm)
    
    # Wrap environment with RM wrapper to handle RM state transitions and rewards
    # RMWrapper adds RM state to observations as a Dict with "image_embedding" and "rm_state"
    envs = RMWrapper(envs, rm=rm)
    
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

    # rb = ReplayBuffer(
    #     args.buffer_size,
    #     envs.single_observation_space,
    #     envs.single_action_space,
    #     device,
    #     n_envs=args.num_envs,
    #     handle_timeout_termination=False,
    # )
    replay_buffer_name = "sac_replay_buffer"
    replay_buffer_namespace = ray.get_runtime_context().namespace or "default"
    
    # Create replay buffer with high concurrency to allow sample() + add_batch() to overlap
    # The separate locks inside SACReplayBuffer ensure sample() doesn't wait for add_batch()
    rb = SACReplayBuffer.options(
        name=replay_buffer_name,
        num_cpus=args.replay_buffer_num_cpus,
        max_concurrency=args.replay_buffer_max_concurrency,
    ).remote(capacity=args.buffer_size, seed=args.seed)
    print(f"[Setup] Replay buffer actor '{replay_buffer_name}' created in namespace '{replay_buffer_namespace}' with max_concurrency={args.replay_buffer_max_concurrency}", flush=True)
    
    start_time = time.time()
    inflight_label_refs = deque()

    try:
        # TRY NOT TO MODIFY: start the game
        obs, info = envs.reset(seed=args.seed)
        can_log = False
        for global_step in trange(args.total_timesteps, desc="Training Progress"):
            # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                # Convert Dict observation to tensors
                obs_dict = {
                    "image_embedding": torch.as_tensor(obs["image_embedding"], device=device, dtype=torch.float32),
                    "rm_state": torch.as_tensor(obs["rm_state"], device=device, dtype=torch.int64),
                }
                actions, _, _ = actor.get_action(obs_dict)
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            # RMWrapper handles RM state transitions and computes RM-based rewards automatically
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            
            # Add transitions to replay buffer
            ref = rb.add_batch.remote(
                observations=obs,
                actions=actions,
                rewards=rewards,
                next_observations=next_obs,
                terminateds=terminations,
                truncateds=truncations,
            )
            inflight_label_refs.append(ref)
            # Backpressure: occasionally drain so we don't build an unbounded queue.
            if len(inflight_label_refs) >= args.max_inflight_label_requests:
                # For Ray actors, use ray.get to wait for completion of the oldest request.
                ray.get(inflight_label_refs.popleft())
            # Periodically log buffer fill.
            if global_step % 1000 == 0:
                rb_size = ray.get(rb.__len__.remote())
                print(f"[Step {global_step}] replay_buffer size={rb_size}, inflight_label_refs={len(inflight_label_refs)}", flush=True)
                writer.add_scalar("replay_buffer/size", rb_size, global_step)
                writer.add_scalar("replay_buffer/label_inflight", len(inflight_label_refs), global_step)

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

                if "original_reward" in infos:
                    original_rewards_to_log = infos["original_reward"][terminated_episodes].tolist()
                    seen_original_rewards.extend(original_rewards_to_log)
                    avg_original_reward = sum(seen_original_rewards) / len(seen_original_rewards)
                    writer.add_scalar("charts/episodic_original_return", avg_original_reward, global_step)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                if global_step % args.update_frequency == 0:
                    can_log = True
                    # This ray.get() is BLOCKING and can stall if replay buffer is slow/empty
                    sample_start = time.time()
                    batch = ray.get(rb.sample.remote(args.batch_size))
                    sample_time_ms = (time.time() - sample_start) * 1000
                    if sample_time_ms > 100:  # Log if sampling takes >100ms
                        print(f"[Step {global_step}] WARNING: rb.sample took {sample_time_ms:.1f}ms (blocking!)", flush=True)
                    if not batch:
                        if global_step % 100 == 0:
                            print(f"[Step {global_step}] WARNING: replay buffer returned empty batch, skipping training", flush=True)
                        continue
                    # Handle Dict observations
                    observations = {
                        "image_embedding": torch.as_tensor(batch["observations"]["image_embedding"], device=device, dtype=torch.float32),
                        "rm_state": torch.as_tensor(batch["observations"]["rm_state"], device=device, dtype=torch.int64),
                    }
                    next_observations = {
                        "image_embedding": torch.as_tensor(batch["next_observations"]["image_embedding"], device=device, dtype=torch.float32),
                        "rm_state": torch.as_tensor(batch["next_observations"]["rm_state"], device=device, dtype=torch.int64),
                    }
                    actions_t = torch.as_tensor(batch["actions"], device=device, dtype=torch.int64).view(-1, 1)
                    rewards_t = torch.as_tensor(batch["rewards"], device=device, dtype=torch.float32).view(-1, 1)
                    dones_t = torch.as_tensor(
                        np.logical_or(batch["terminateds"], batch["truncateds"]),
                        device=device,
                        dtype=torch.float32,
                    ).view(-1, 1)

                    # CRITIC training
                    with torch.no_grad():
                        _, next_state_log_pi, next_state_action_probs = actor.get_action(next_observations)
                        qf1_next_target = qf1_target(next_observations)
                        qf2_next_target = qf2_target(next_observations)
                        # we can use the action probabilities instead of MC sampling to estimate the expectation
                        min_qf_next_target = next_state_action_probs * (
                            torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                        )
                        # adapt Q-target for discrete Q-function
                        min_qf_next_target = min_qf_next_target.sum(dim=1)
                        next_q_value = rewards_t.flatten() + (1 - dones_t.flatten()) * args.gamma * (min_qf_next_target)

                    # use Q-values only for the taken actions
                    qf1_values = qf1(observations)
                    qf2_values = qf2(observations)
                    qf1_a_values = qf1_values.gather(1, actions_t).view(-1)
                    qf2_a_values = qf2_values.gather(1, actions_t).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    q_optimizer.step()

                    # ACTOR training
                    _, log_pi, action_probs = actor.get_action(observations)
                    with torch.no_grad():
                        qf1_values = qf1(observations)
                        qf2_values = qf2(observations)
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
    
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Interrupted by user, cleaning up...")
    finally:
        # Clean shutdown: close resources and tear down Ray actors/Serve deployments
        print("Closing environments and writer...")
        envs.close()
        writer.close()
        
        # Kill the replay buffer actor
        try:
            print(f"Killing replay buffer actor '{replay_buffer_name}'...")
            ray.kill(rb)
        except Exception as e:
            print(f"Warning: failed to kill replay buffer actor: {e}")
        
        # Shutdown Ray if we initialized it for parallel sweeps
        if args.use_multiple_agents:
            try:
                print("Shutting down Ray...")
                ray.shutdown()
            except Exception as e:
                print(f"Warning: failed to shutdown Ray: {e}")
        
        print("Cleanup complete.")

if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)