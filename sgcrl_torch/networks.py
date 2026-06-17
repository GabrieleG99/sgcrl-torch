"""PyTorch networks for SGCRL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from sgcrl_torch.distributions import normal_tanh_from_raw


def _init_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: Sequence[int],
        output_dim: int,
        activate_final: bool = False,
    ):
        super().__init__()
        sizes = [input_dim, *hidden_layer_sizes, output_dim]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2 or activate_final:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.apply(_init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvTorso(nn.Module):
    """Small Atari-style torso for optional image observations."""

    def __init__(self, in_channels: int = 6, output_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            flat_dim = self.net(dummy).shape[-1]
        self.proj = nn.Linear(flat_dim, output_dim)
        self.apply(_init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.proj(self.net(x)))


class PolicyNetwork(nn.Module):
    """Stochastic tanh-normal actor."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_layer_sizes: Tuple[int, ...] = (256, 256),
        actor_min_std: float = 1e-6,
        obs_dim: Optional[int] = None,
        use_image_obs: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.actor_min_std = actor_min_std
        self.use_image_obs = use_image_obs
        self.obs_dim = obs_dim

        if use_image_obs:
            if obs_dim is None:
                raise ValueError("obs_dim is required for image observations")
            self.torso = ConvTorso(in_channels=6)
            body_input_dim = 512
        else:
            self.torso = None
            body_input_dim = observation_dim

        layers = []
        last_dim = body_input_dim
        for width in hidden_layer_sizes:
            layers.append(nn.Linear(last_dim, width))
            layers.append(nn.ReLU())
            last_dim = width
        self.body = nn.Sequential(*layers)
        self.loc_layer = nn.Linear(last_dim, action_dim)
        self.scale_layer = nn.Linear(last_dim, action_dim)
        self.apply(_init_linear)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        if self.use_image_obs:
            state = obs[:, : self.obs_dim].reshape(-1, 64, 64, 3) / 255.0
            goal = obs[:, self.obs_dim :].reshape(-1, 64, 64, 3) / 255.0
            x = torch.cat([state, goal], dim=-1).permute(0, 3, 1, 2)
            return self.torso(x)
        return obs

    def forward(self, obs: torch.Tensor):
        x = self._features(obs)
        x = self.body(x)
        raw_loc = self.loc_layer(x)
        raw_scale = self.scale_layer(x)
        return normal_tanh_from_raw(raw_loc, raw_scale, self.actor_min_std)

    @torch.no_grad()
    def act(self, obs: np.ndarray, device: torch.device, deterministic: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        dist = self(obs_tensor)
        action = dist.mode() if deterministic else dist.sample()
        return action.squeeze(0).cpu().numpy()


class ContrastiveQNetwork(nn.Module):
    """Contrastive critic that scores all state-action/goal pairs in a batch."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        repr_dim: int = 64,
        hidden_layer_sizes: Tuple[int, ...] = (256, 256),
        repr_norm: bool = False,
        twin_q: bool = False,
        use_image_obs: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.repr_dim = int(repr_dim)
        self.repr_norm = repr_norm
        self.twin_q = twin_q
        self.use_image_obs = use_image_obs

        if use_image_obs:
            self.state_torso = ConvTorso(in_channels=3, output_dim=512)
            self.goal_torso = ConvTorso(in_channels=3, output_dim=512)
            state_dim = 512
            goal_in_dim = 512
        else:
            self.state_torso = None
            self.goal_torso = None
            state_dim = obs_dim
            goal_in_dim = goal_dim

        self.sa_encoder = MLP(state_dim + action_dim, hidden_layer_sizes, self.repr_dim)
        self.g_encoder = MLP(goal_in_dim, hidden_layer_sizes, self.repr_dim)
        if twin_q:
            self.sa_encoder2 = MLP(state_dim + action_dim, hidden_layer_sizes, self.repr_dim)
            self.g_encoder2 = MLP(goal_in_dim, hidden_layer_sizes, self.repr_dim)
        else:
            self.sa_encoder2 = None
            self.g_encoder2 = None

    def _split_obs(self, obs: torch.Tensor):
        state = obs[:, : self.obs_dim]
        goal = obs[:, self.obs_dim :]
        if self.use_image_obs:
            state = state.reshape(-1, 64, 64, 3).permute(0, 3, 1, 2) / 255.0
            goal = goal.reshape(-1, 64, 64, 3).permute(0, 3, 1, 2) / 255.0
            state = self.state_torso(state)
            goal = self.goal_torso(goal)
        return state, goal

    def _encode(self, obs: torch.Tensor, action: torch.Tensor, second: bool = False):
        state, goal = self._split_obs(obs)
        if second:
            sa_repr = self.sa_encoder2(torch.cat([state, action], dim=-1))
            g_repr = self.g_encoder2(goal)
        else:
            sa_repr = self.sa_encoder(torch.cat([state, action], dim=-1))
            g_repr = self.g_encoder(goal)

        if self.repr_norm:
            sa_repr = sa_repr / torch.clamp(sa_repr.norm(dim=1, keepdim=True), min=1e-6)
            g_repr = g_repr / torch.clamp(g_repr.norm(dim=1, keepdim=True), min=1e-6)
        return sa_repr, g_repr

    @staticmethod
    def combine_repr(sa_repr: torch.Tensor, g_repr: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ik,jk->ij", sa_repr, g_repr)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        sa_repr, g_repr = self._encode(obs, action, second=False)
        critic_val = self.combine_repr(sa_repr, g_repr)
        if self.twin_q:
            sa_repr2, g_repr2 = self._encode(obs, action, second=True)
            critic_val2 = self.combine_repr(sa_repr2, g_repr2)
            critic_val = torch.stack([critic_val, critic_val2], dim=-1)
            sa_repr, g_repr = sa_repr2, g_repr2
        return critic_val, sa_repr, g_repr


@dataclass
class NetworkBundle:
    policy: PolicyNetwork
    critic: ContrastiveQNetwork


def make_networks(config) -> NetworkBundle:
    return NetworkBundle(
        policy=PolicyNetwork(
            observation_dim=config.resolved_observation_dim(),
            action_dim=config.action_dim,
            hidden_layer_sizes=config.hidden_layer_sizes,
            obs_dim=config.obs_dim,
            use_image_obs=config.use_image_obs,
        ),
        critic=ContrastiveQNetwork(
            obs_dim=config.obs_dim,
            goal_dim=config.resolved_goal_dim(),
            action_dim=config.action_dim,
            repr_dim=int(config.repr_dim),
            hidden_layer_sizes=config.hidden_layer_sizes,
            repr_norm=config.repr_norm,
            twin_q=config.twin_q,
            use_image_obs=config.use_image_obs,
        ),
    )
