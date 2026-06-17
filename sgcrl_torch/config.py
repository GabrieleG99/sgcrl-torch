"""Configuration for the PyTorch SGCRL implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np


@dataclass
class ContrastiveConfig:
    """Configuration options mirroring the original JAX SGCRL config."""

    add_uid: bool = True
    time_delta_minutes: int = 5
    log_dir: str = "logs/"
    env_name: str = ""
    alg_name: str = ""
    seed: int = 0
    max_number_of_steps: int = 8_000_000
    num_actors: int = 4

    # Environment options.
    fix_goals: bool = False

    # Loss options.
    batch_size: int = 256
    actor_learning_rate: float = 3e-4
    learning_rate: float = 3e-4
    reward_scale: float = 1.0
    discount: float = 0.99
    n_step: int = 1
    tau: float = 0.005
    hidden_layer_sizes: Tuple[int, ...] = (256, 256)

    # Entropy options.
    use_action_entropy: bool = True
    entropy_coefficient: Optional[float] = None
    target_entropy: float = 0.0

    # Replay options.
    min_replay_size: int = 10_000
    max_replay_size: int = 1_000_000
    replay_table_name: str = "default_table"
    prefetch_size: int = 4
    num_parallel_calls: Optional[int] = 4
    samples_per_insert: float = 256.0
    samples_per_insert_tolerance_rate: float = 0.1
    num_sgd_steps_per_step: int = 64

    # Training options.
    no_repr: bool = False
    repr_dim: Union[int, str] = 64
    use_random_actor: bool = True
    repr_norm: bool = False
    use_cpc: bool = False
    local: bool = False
    use_td: bool = False
    twin_q: bool = False
    use_image_obs: bool = False
    random_goals: float = 0.5
    jit: bool = False
    add_mc_to_td: bool = False
    resample_neg_actions: bool = False

    # Filled from the environment.
    obs_dim: int = -1
    goal_dim: int = -1
    action_dim: int = -1
    max_episode_steps: int = -1
    start_index: int = 0
    end_index: int = -1

    # PyTorch/distributed runtime options.
    device: str = "auto"
    actor_device: str = "cpu"
    actor_update_period: int = 100
    episode_queue_size: int = 128
    start_method: str = "spawn"
    log_every: float = 10.0
    eval_every: int = 0
    eval_episodes: int = 5
    checkpoint_every: int = 100
    rate_limit_actors: bool = True
    rate_limit_error_buffer: int = -1

    def resolved_goal_dim(self) -> int:
        if self.goal_dim > 0:
            return self.goal_dim
        if self.obs_dim <= 0:
            raise ValueError("obs_dim must be set before resolving goal_dim")
        if self.end_index == -1:
            return self.obs_dim - self.start_index
        return self.end_index - self.start_index

    def resolved_observation_dim(self) -> int:
        return self.obs_dim + self.resolved_goal_dim()

    def apply_algorithm(self, alg_name: str) -> None:
        """Apply the same algorithm flags as lp_contrastive.py."""
        self.alg_name = alg_name
        if alg_name == "contrastive_cpc":
            self.use_cpc = True
        elif alg_name == "contrastive_nce":
            self.use_cpc = False
        elif alg_name == "c_learning":
            self.use_td = True
            self.twin_q = True
        elif alg_name == "nce+c_learning":
            self.use_td = True
            self.twin_q = True
            self.add_mc_to_td = True
        else:
            raise NotImplementedError(f"Unknown method: {alg_name}")


def target_entropy_from_action_space(action_space, target_entropy_per_dimension=None):
    """Matches the original target entropy heuristic for bounded actions."""
    if not hasattr(action_space, "shape"):
        raise ValueError(f"Only bounded Box-like spaces are supported: {action_space}")

    num_actions = int(np.prod(action_space.shape))
    if target_entropy_per_dimension is None:
        low = np.asarray(action_space.low)
        high = np.asarray(action_space.high)
        if not np.allclose(low, -1.0):
            raise ValueError(f"Minimum expected to be -1, got: {low}")
        if not np.allclose(high, 1.0):
            raise ValueError(f"Maximum expected to be 1, got: {high}")
        return -num_actions
    return target_entropy_per_dimension * num_actions
