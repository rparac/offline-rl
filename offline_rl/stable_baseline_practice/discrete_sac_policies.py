import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Optional, TypeVar, Union

import numpy as np
import torch.nn.functional as F
from stable_baselines3.common.policies import BaseModel, BasePolicy
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor

SelfDiscreteCritic = TypeVar("SelfDiscreteCritic", bound="DiscreteCritic")


class DiscreteCritic(BaseModel):
    """
    Critic network(s) for Discrete Soft Actor-Critic (DSAC).
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        # For discrete actions, we need Q-values for ALL actions, not just action_dim
        # get_action_dim() returns 1 for Discrete, but we need action_space.n outputs
        num_actions = self.action_space.n

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: list[nn.Module] = []
        for idx in range(n_critics):
            q_net_list = create_mlp(features_dim, num_actions, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor) -> tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        return tuple(q_net(features) for q_net in self.q_networks)


class DiscreteActor(BasePolicy):
    """
    Actor network (policy) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        # For discrete actions, we need logits for ALL actions
        # get_action_dim() returns 1 for Discrete, but CategoricalDistribution needs action_space.n
        num_actions = self.action_space.n
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        self.action_dist = CategoricalDistribution(num_actions)  # type: ignore[assignment]
        self.logits = nn.Linear(last_layer_dim, num_actions)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        logits = self.logits(self.latent_pi(features))

        return self.action_dist.actions_from_params(logits, deterministic=deterministic)

    def action_log_prob(self, obs: PyTorchObs) -> th.Tensor:
        """
        Returns log probabilities for ALL actions.
        This is needed for discrete SAC which requires the full distribution.
        
        Returns:
            log_probs: Tensor of shape [batch_size, num_actions] with log probabilities
        """
        features = self.extract_features(obs, self.features_extractor)
        logits = self.logits(self.latent_pi(features))
        # Return log probabilities for ALL actions (needed for discrete SAC)
        return F.log_softmax(logits, dim=-1)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)

class DiscreteSACPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for Discrete SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    :param use_sde: Whether to use State Dependent Exploration (SDE) instead of action noise
        (currently not implemented for discrete SAC)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        (currently not implemented for discrete SAC)
    :param use_sde_at_warmup: Whether to use SDE during warm-up phase
        (currently not implemented for discrete SAC)
    """

    actor: DiscreteActor
    critic: DiscreteCritic
    critic_target: DiscreteCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()

        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = list(self.critic.parameters())

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DiscreteActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return DiscreteActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DiscreteCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return DiscreteCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


DiscreteMlpPolicy = DiscreteSACPolicy


class DiscreteCnnPolicy(DiscreteSACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
        )


class DiscreteMultiInputPolicy(DiscreteSACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
        )
