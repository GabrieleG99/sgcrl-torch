"""Shared utilities for the PyTorch SGCRL implementation."""

from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def obs_to_goal_1d(obs: np.ndarray, start_index: int, end_index: int) -> np.ndarray:
    assert len(obs.shape) == 1
    return obs_to_goal_2d(obs[None], start_index, end_index)[0]


def obs_to_goal_2d(obs: np.ndarray, start_index: int, end_index: int) -> np.ndarray:
    assert len(obs.shape) == 2
    if end_index == -1:
        return obs[:, start_index:]
    return obs[:, start_index:end_index]


def obs_to_goal_tensor(obs: torch.Tensor, start_index: int, end_index: int) -> torch.Tensor:
    if end_index == -1:
        return obs[:, start_index:]
    return obs[:, start_index:end_index]


def action_dim(action_space) -> int:
    return int(np.prod(action_space.shape))


def flatten_action(action: np.ndarray) -> np.ndarray:
    return np.asarray(action, dtype=np.float32).reshape(-1)


def atomic_torch_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


class SuccessObserver:
    """Tracks whether any reward in an episode was positive."""

    def __init__(self):
        self._rewards = []
        self._success = []

    def observe_first(self, observation):
        del observation
        if self._rewards:
            self._success.append(float(np.sum(self._rewards) >= 1))
        self._rewards = []

    def observe(self, observation, action, reward, done, info):
        del observation, action, done, info
        self._rewards.append(reward)

    def get_metrics(self) -> Dict[str, float]:
        success = float(np.sum(self._rewards) >= 1)
        recent = self._success[-1000:]
        return {
            "success": success,
            "success_1000": float(np.mean(recent)) if recent else success,
        }


class DistanceObserver:
    """Measures L2 distance between achieved goal coordinates and goal."""

    def __init__(self, obs_dim: int, start_index: int, end_index: int, smooth: bool = True):
        self._distances = []
        self._history = {}
        self._obs_dim = obs_dim
        self._start_index = start_index
        self._end_index = end_index
        self._smooth = smooth

    def _distance(self, observation) -> float:
        obs = observation[: self._obs_dim]
        goal = observation[self._obs_dim :]
        achieved = obs_to_goal_1d(obs, self._start_index, self._end_index)
        return float(np.linalg.norm(achieved - goal))

    def observe_first(self, observation):
        if self._smooth and self._distances:
            for key, value in self._current_metrics().items():
                self._history.setdefault(key, []).append(value)
        self._distances = [self._distance(observation)]

    def observe(self, observation, action, reward, done, info):
        del action, reward, done, info
        self._distances.append(self._distance(observation))

    def _current_metrics(self) -> Dict[str, float]:
        return {
            "init_dist": self._distances[0],
            "final_dist": self._distances[-1],
            "delta_dist": self._distances[0] - self._distances[-1],
            "min_dist": min(self._distances),
        }

    def get_metrics(self) -> Dict[str, float]:
        metrics = self._current_metrics()
        if self._smooth:
            for key, values in self._history.items():
                for size in (10, 100, 1000):
                    metrics[f"{key}_{size}"] = float(np.nanmean(values[-size:]))
        return metrics
