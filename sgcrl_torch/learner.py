"""PyTorch implementation of the SGCRL learner."""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from sgcrl_torch.config import ContrastiveConfig
from sgcrl_torch.networks import make_networks
from sgcrl_torch.replay import TransitionBatch
from sgcrl_torch.utils import atomic_torch_save, obs_to_goal_tensor


class ContrastiveLearner:
    """Owns policy/critic optimizers and performs SGCRL updates."""

    def __init__(self, config: ContrastiveConfig, device: torch.device):
        self.config = config
        self.device = device
        bundle = make_networks(config)
        self.policy = bundle.policy.to(device)
        self.critic = bundle.critic.to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        self.target_critic.eval()

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.actor_learning_rate, eps=1e-7
        )
        self.q_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.learning_rate, eps=1e-7
        )

        self.adaptive_entropy = config.entropy_coefficient is None
        if self.adaptive_entropy:
            self.log_alpha = torch.nn.Parameter(torch.zeros((), device=device))
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None

        self.num_sgd_steps = 0

    def alpha_value(self) -> torch.Tensor:
        if self.adaptive_entropy:
            return self.log_alpha.detach().exp()
        return torch.as_tensor(float(self.config.entropy_coefficient), device=self.device)

    def _obs_to_goal(self, obs: torch.Tensor) -> torch.Tensor:
        return obs_to_goal_tensor(obs, self.config.start_index, self.config.end_index)

    def alpha_loss(self, transitions: TransitionBatch) -> torch.Tensor:
        assert self.adaptive_entropy
        with torch.no_grad():
            dist = self.policy(transitions.observation)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            target = (-log_prob - self.config.target_entropy).detach()
        return (self.log_alpha.exp() * target).mean()

    def critic_loss(self, transitions: TransitionBatch):
        batch_size = transitions.observation.shape[0]
        obs = transitions.observation
        next_obs = transitions.next_observation

        if self.config.use_td:
            state, goal = torch.split(obs, [self.config.obs_dim, self.config.resolved_goal_dim()], dim=1)
            next_state = next_obs[:, : self.config.obs_dim]
            if self.config.add_mc_to_td:
                next_fraction = (1 - self.config.discount) / ((1 - self.config.discount) + 1)
                num_next = int(batch_size * next_fraction)
                new_goal = torch.cat([self._obs_to_goal(next_state[:num_next]), goal[num_next:]], dim=0)
            else:
                new_goal = self._obs_to_goal(next_state)
            obs = torch.cat([state, new_goal], dim=1)

        eye = torch.eye(batch_size, device=self.device)
        logits, _, _ = self.critic(obs, transitions.action)

        if self.config.use_td:
            if logits.dim() != 3:
                raise ValueError("TD learning requires twin_q=True")

            _, goal = torch.split(obs, [self.config.obs_dim, self.config.resolved_goal_dim()], dim=1)
            next_state = transitions.next_observation[:, : self.config.obs_dim]
            goal_indices = torch.roll(torch.arange(batch_size, device=self.device), shifts=-1)
            shuffled_goal = goal[goal_indices]
            td_next_obs = torch.cat([next_state, shuffled_goal], dim=1)

            with torch.no_grad():
                next_action = self.policy(td_next_obs).sample()
                next_q, _, _ = self.target_critic(td_next_obs, next_action)
                next_q = torch.sigmoid(next_q)
                next_v = torch.min(next_q, dim=-1).values
                next_v = torch.diagonal(next_v, 0)
                w = next_v / torch.clamp(1.0 - next_v, min=1e-6)
                w = torch.clamp(w, 0.0, 20.0)

            idx = torch.arange(batch_size, device=self.device)
            pos_logits = logits[idx, idx, :]
            neg_logits = logits[idx, goal_indices, :]
            loss_pos = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits), reduction="none"
            )
            loss_neg1 = w[:, None] * F.binary_cross_entropy_with_logits(
                neg_logits, torch.ones_like(neg_logits), reduction="none"
            )
            loss_neg2 = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits), reduction="none"
            )
            if self.config.add_mc_to_td:
                loss = (1 + (1 - self.config.discount)) * loss_pos
                loss = loss + self.config.discount * loss_neg1 + 2 * loss_neg2
            else:
                loss = (1 - self.config.discount) * loss_pos
                loss = loss + self.config.discount * loss_neg1 + loss_neg2
            logits_for_metrics = logits.mean(dim=-1)
        else:
            if logits.dim() == 3:
                losses = []
                for i in range(logits.shape[-1]):
                    losses.append(self._mc_loss(logits[:, :, i], eye))
                loss = torch.stack(losses, dim=-1).mean(dim=-1)
                logits_for_metrics = logits.mean(dim=-1)
            else:
                loss = self._mc_loss(logits, eye)
                logits_for_metrics = logits

        loss = loss.mean()
        metrics = self._critic_metrics(logits_for_metrics, eye)
        return loss, metrics

    def _mc_loss(self, logits: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
        if self.config.use_cpc:
            targets = torch.arange(logits.shape[0], device=self.device)
            ce = F.cross_entropy(logits, targets, reduction="none")
            return ce + 0.01 * torch.logsumexp(logits, dim=1).pow(2)
        return F.binary_cross_entropy_with_logits(logits, eye, reduction="none")

    def _critic_metrics(self, logits: torch.Tensor, eye: torch.Tensor) -> Dict[str, torch.Tensor]:
        correct = logits.argmax(dim=1) == eye.argmax(dim=1)
        pos_denom = eye.sum().clamp_min(1.0)
        neg_mask = 1.0 - eye
        neg_denom = neg_mask.sum().clamp_min(1.0)
        return {
            "binary_accuracy": ((logits > 0) == (eye > 0)).float().mean(),
            "categorical_accuracy": correct.float().mean(),
            "logits_pos": (logits * eye).sum() / pos_denom,
            "logits_neg": (logits * neg_mask).sum() / neg_denom,
            "logsumexp": torch.logsumexp(logits, dim=1).pow(2).mean(),
        }

    def actor_loss(self, transitions: TransitionBatch, alpha: torch.Tensor):
        obs = transitions.observation
        state = obs[:, : self.config.obs_dim]
        goal = obs[:, self.config.obs_dim :]

        if self.config.random_goals == 0.0:
            new_state = state
            new_goal = goal
        elif self.config.random_goals == 0.5:
            new_state = torch.cat([state, state], dim=0)
            new_goal = torch.cat([goal, torch.roll(goal, shifts=1, dims=0)], dim=0)
        else:
            if self.config.random_goals != 1.0:
                raise ValueError("random_goals must be one of 0.0, 0.5, 1.0")
            new_state = state
            new_goal = torch.roll(goal, shifts=1, dims=0)

        new_obs = torch.cat([new_state, new_goal], dim=1)
        dist = self.policy(new_obs)
        action, log_prob = dist.rsample_and_log_prob()
        q_action, _, _ = self.critic(new_obs, action)
        if q_action.dim() == 3:
            q_action = torch.min(q_action, dim=-1).values
        actor_loss = -torch.diagonal(q_action, 0)
        approx_entropy = -log_prob
        if self.config.use_action_entropy:
            actor_loss = actor_loss - alpha * approx_entropy
        metrics = {"entropy_mean": approx_entropy.mean()}
        return actor_loss.mean(), metrics

    def update(self, transitions: TransitionBatch) -> Dict[str, float]:
        transitions = transitions.to(self.device)

        alpha = self.alpha_value()
        alpha_loss = self.alpha_loss(transitions) if self.adaptive_entropy else None

        # Match the JAX code's immutable semantics: actor gradients see the
        # critic parameters from the start of this update, not the just-updated
        # critic. We compute those policy gradients before q_optimizer.step().
        self.policy_optimizer.zero_grad(set_to_none=True)
        for param in self.critic.parameters():
            param.requires_grad_(False)
        actor_loss, actor_metrics = self.actor_loss(transitions, alpha)
        actor_loss.backward()
        for param in self.critic.parameters():
            param.requires_grad_(True)

        critic_loss, critic_metrics = self.critic_loss(transitions)
        self.q_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.q_optimizer.step()
        self._soft_update_target()

        self.policy_optimizer.step()

        metrics = {
            "critic_loss": critic_loss.detach(),
            "actor_loss": actor_loss.detach(),
            **critic_metrics,
            **actor_metrics,
        }

        if self.adaptive_entropy:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            metrics["alpha_loss"] = alpha_loss.detach()
            metrics["alpha"] = self.log_alpha.detach().exp()

        self.num_sgd_steps += 1
        return {key: float(value.detach().cpu()) for key, value in metrics.items()}

    @torch.no_grad()
    def _soft_update_target(self):
        tau = self.config.tau
        for target_param, source_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)

    def policy_state_cpu(self) -> Dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in self.policy.state_dict().items()}

    def policy_state_numpy(self) -> Dict[str, object]:
        return {key: value.detach().cpu().numpy().copy() for key, value in self.policy.state_dict().items()}

    def state_dict(self) -> Dict[str, object]:
        state = {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "num_sgd_steps": self.num_sgd_steps,
            "config": self.config,
        }
        if self.adaptive_entropy:
            state["log_alpha"] = self.log_alpha.detach()
            state["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        return state

    def save(self, path: str) -> None:
        atomic_torch_save(self.state_dict(), path)
