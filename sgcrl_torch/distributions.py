"""Torch distributions used by the SGCRL policy."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class TanhNormal:
    """Diagonal Normal transformed through tanh.

    This mirrors the Haiku/TensorFlow Probability NormalTanhDistribution used
    by the original policy network.
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, threshold: float = 0.999):
        self.loc = loc
        self.scale = scale
        self.threshold = threshold
        self.base_dist = torch.distributions.Normal(loc, scale)

    def rsample(self) -> torch.Tensor:
        return torch.tanh(self.base_dist.rsample())

    def sample(self) -> torch.Tensor:
        return torch.tanh(self.base_dist.sample())

    def mode(self) -> torch.Tensor:
        return torch.tanh(self.loc)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(action, -self.threshold, self.threshold)
        pre_tanh = atanh(action)
        log_prob = self.base_dist.log_prob(pre_tanh)
        correction = torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6))
        return (log_prob - correction).sum(dim=-1)

    def rsample_and_log_prob(self):
        pre_tanh = self.base_dist.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = self.base_dist.log_prob(pre_tanh)
        correction = torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6))
        return action, (log_prob - correction).sum(dim=-1)

    def entropy_estimate(self) -> torch.Tensor:
        action = self.rsample()
        return -self.log_prob(action)


def normal_tanh_from_raw(raw_loc: torch.Tensor, raw_scale: torch.Tensor, min_scale: float):
    loc = 10.0 * torch.tanh(raw_loc / 10.0)
    scale = F.softplus(raw_scale) + min_scale
    return TanhNormal(loc, scale)
