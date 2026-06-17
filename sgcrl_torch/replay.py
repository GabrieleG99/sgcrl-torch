"""Episode replay with SGCRL future-goal relabeling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from queue import Empty
from typing import Deque, Dict, Optional

import numpy as np
import torch

from sgcrl_torch.utils import obs_to_goal_2d


@dataclass
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    discounts: np.ndarray

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    @property
    def episode_return(self) -> float:
        return float(np.sum(self.rewards))

    @property
    def success(self) -> float:
        return float(self.episode_return >= 1.0)


@dataclass
class TransitionBatch:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    discount: torch.Tensor
    next_observation: torch.Tensor
    next_action: torch.Tensor

    def to(self, device: torch.device) -> "TransitionBatch":
        return TransitionBatch(
            observation=self.observation.to(device),
            action=self.action.to(device),
            reward=self.reward.to(device),
            discount=self.discount.to(device),
            next_observation=self.next_observation.to(device),
            next_action=self.next_action.to(device),
        )


class EpisodeReplayBuffer:
    """Replay buffer storing complete episodes and sampling relabeled transitions."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        start_index: int,
        end_index: int,
        max_replay_size: int,
        max_episode_steps: int,
        discount: float,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.start_index = start_index
        self.end_index = end_index
        self.discount = discount
        max_episodes = max(1, max_replay_size // max(1, max_episode_steps))
        self._episodes: Deque[Episode] = deque(maxlen=max_episodes)
        self._num_steps = 0
        self._num_episodes_added = 0
        self._num_successes = 0
        self._success_history: Deque[float] = deque(maxlen=1000)
        self._return_history: Deque[float] = deque(maxlen=1000)
        self._length_history: Deque[float] = deque(maxlen=1000)
        self._latest_episode_metrics = {
            "success": 0.0,
            "return": 0.0,
            "length": 0.0,
        }
        self._rng = np.random.default_rng(seed)

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_episodes_added(self) -> int:
        return self._num_episodes_added

    @property
    def num_successes(self) -> int:
        return self._num_successes

    @property
    def success_rate(self) -> float:
        if self._num_episodes_added == 0:
            return 0.0
        return self._num_successes / self._num_episodes_added

    @property
    def success_rate_1000(self) -> float:
        if not self._success_history:
            return 0.0
        return float(np.mean(self._success_history))

    @property
    def average_return_1000(self) -> float:
        if not self._return_history:
            return 0.0
        return float(np.mean(self._return_history))

    @property
    def average_length_1000(self) -> float:
        if not self._length_history:
            return 0.0
        return float(np.mean(self._length_history))

    def add_episode(self, episode: Dict[str, np.ndarray] | Episode) -> None:
        if isinstance(episode, dict):
            episode = Episode(
                observations=np.asarray(episode["observations"], dtype=np.float32),
                actions=np.asarray(episode["actions"], dtype=np.float32),
                rewards=np.asarray(episode["rewards"], dtype=np.float32),
                discounts=np.asarray(episode["discounts"], dtype=np.float32),
            )
        if episode.length <= 0:
            return
        if episode.observations.shape[0] != episode.length + 1:
            raise ValueError("episodes must contain T+1 observations and T actions")
        if len(self._episodes) == self._episodes.maxlen:
            self._num_steps -= self._episodes[0].length
        self._episodes.append(episode)
        self._num_steps += episode.length
        self._num_episodes_added += 1
        success = episode.success
        episode_return = episode.episode_return
        self._num_successes += int(success)
        self._success_history.append(success)
        self._return_history.append(episode_return)
        self._length_history.append(float(episode.length))
        self._latest_episode_metrics = {
            "success": success,
            "return": episode_return,
            "length": float(episode.length),
        }

    def drain_queue(self, episode_queue, max_items: Optional[int] = None) -> int:
        drained = 0
        while max_items is None or drained < max_items:
            try:
                item = episode_queue.get_nowait()
            except Empty:
                break
            self.add_episode(item)
            drained += 1
        return drained

    def environment_metrics(self) -> Dict[str, float]:
        """Metrics for episodes completed by actors in the simulated env."""
        return {
            "environment_episodes": float(self._num_episodes_added),
            "environment_successes": float(self._num_successes),
            "environment_success_rate": self.success_rate,
            "environment_success_1000": self.success_rate_1000,
            "environment_episode_return_1000": self.average_return_1000,
            "environment_episode_length_1000": self.average_length_1000,
            "environment_latest_success": self._latest_episode_metrics["success"],
            "environment_latest_return": self._latest_episode_metrics["return"],
            "environment_latest_episode_length": self._latest_episode_metrics["length"],
        }

    def _sample_episode(self) -> Episode:
        if not self._episodes:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        return self._episodes[int(self._rng.integers(0, len(self._episodes)))]

    def _sample_future_index(self, t: int, length: int) -> int:
        future = np.arange(t + 1, length + 1)
        offsets = future - t
        probs = self.discount ** offsets.astype(np.float32)
        probs = probs / probs.sum()
        return int(self._rng.choice(future, p=probs))

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        observations = []
        next_observations = []
        actions = []
        next_actions = []
        rewards = []
        discounts = []

        for _ in range(batch_size):
            episode = self._sample_episode()
            t = int(self._rng.integers(0, episode.length))
            future_idx = self._sample_future_index(t, episode.length)

            state = episode.observations[t, : self.obs_dim]
            next_state = episode.observations[t + 1, : self.obs_dim]
            future_state = episode.observations[future_idx : future_idx + 1, : self.obs_dim]
            goal = obs_to_goal_2d(future_state, self.start_index, self.end_index)[0]

            observations.append(np.concatenate([state, goal], axis=0))
            next_observations.append(np.concatenate([next_state, goal], axis=0))
            actions.append(episode.actions[t])
            next_actions.append(episode.actions[min(t + 1, episode.length - 1)])
            rewards.append(episode.rewards[t])
            discounts.append(episode.discounts[t])

        return TransitionBatch(
            observation=torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=device),
            action=torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device),
            reward=torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=device),
            discount=torch.as_tensor(np.asarray(discounts), dtype=torch.float32, device=device),
            next_observation=torch.as_tensor(np.asarray(next_observations), dtype=torch.float32, device=device),
            next_action=torch.as_tensor(np.asarray(next_actions), dtype=torch.float32, device=device),
        )
