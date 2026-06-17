"""Actor and evaluation loops for the PyTorch SGCRL runtime."""

from __future__ import annotations

import time
from queue import Empty
from typing import Dict

import numpy as np
import torch

from sgcrl_torch.networks import PolicyNetwork
import sgcrl_torch.env_utils as env_utils
from sgcrl_torch.utils import DistanceObserver, SuccessObserver, flatten_action, resolve_device, set_global_seeds


def _load_policy_if_available(policy: PolicyNetwork, weight_queue, current_version: int, device: torch.device) -> int:
    latest = None
    while True:
        try:
            latest = weight_queue.get_nowait()
        except Empty:
            break
    if latest is None:
        return current_version
    version, state = latest
    if version <= current_version:
        return current_version
    state = {key: torch.as_tensor(value) for key, value in state.items()}
    policy.load_state_dict(state)
    policy.to(device)
    policy.eval()
    return version


def _random_action(env):
    return np.asarray(env.action_space.sample(), dtype=np.float32)


def _read_counter(counter) -> int:
    with counter.get_lock():
        return int(counter.value)


def _wait_for_rate_limiter(config, actor_steps, learner_samples, stop_event) -> None:
    if not config.rate_limit_actors:
        return
    error_buffer = config.rate_limit_error_buffer
    if error_buffer < 0:
        error_buffer = int(config.min_replay_size * config.samples_per_insert_tolerance_rate)

    while not stop_event.is_set():
        inserted = _read_counter(actor_steps)
        if inserted < config.min_replay_size:
            return
        sampled = _read_counter(learner_samples)
        allowed_inserted = config.min_replay_size + error_buffer
        allowed_inserted += int(sampled / max(config.samples_per_insert, 1e-6))
        if inserted < allowed_inserted:
            return
        time.sleep(0.01)


def actor_loop(
    actor_id: int,
    config,
    fixed_start_end,
    episode_queue,
    weight_queue,
    actor_steps,
    learner_samples,
    stop_event,
):
    """Collect episodes asynchronously and push them to the learner queue."""
    set_global_seeds(config.seed + 10_000 + actor_id)
    device = resolve_device(config.actor_device)
    env, _ = env_utils.make_environment(
        config.env_name,
        config.start_index,
        config.end_index,
        seed=config.seed + actor_id,
        fixed_start_end=fixed_start_end,
    )
    policy = PolicyNetwork(
        observation_dim=config.resolved_observation_dim(),
        action_dim=config.action_dim,
        hidden_layer_sizes=config.hidden_layer_sizes,
        obs_dim=config.obs_dim,
        use_image_obs=config.use_image_obs,
    ).to(device)
    policy.eval()

    version = -1
    obs = env.reset(seed=config.seed + actor_id)
    observations = [np.asarray(obs, dtype=np.float32)]
    actions = []
    rewards = []
    discounts = []
    steps_since_sync = config.actor_update_period

    while not stop_event.is_set():
        if steps_since_sync >= config.actor_update_period:
            version = _load_policy_if_available(policy, weight_queue, version, device)
            steps_since_sync = 0

        use_random = config.use_random_actor and version < 1
        if use_random:
            action = _random_action(env)
        else:
            action = policy.act(obs, device=device, deterministic=False)
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)

        next_obs, reward, done, info = env.step(action)
        del info
        with actor_steps.get_lock():
            actor_steps.value += 1
            reached_limit = actor_steps.value >= config.max_number_of_steps

        actions.append(flatten_action(action))
        rewards.append(float(reward))
        discounts.append(0.0 if done else 1.0)
        observations.append(np.asarray(next_obs, dtype=np.float32))
        steps_since_sync += 1
        obs = next_obs

        if done or reached_limit:
            episode_queue.put(
                {
                    "observations": np.asarray(observations, dtype=np.float32),
                    "actions": np.asarray(actions, dtype=np.float32),
                    "rewards": np.asarray(rewards, dtype=np.float32),
                    "discounts": np.asarray(discounts, dtype=np.float32),
                }
            )
            if reached_limit:
                break
            _wait_for_rate_limiter(config, actor_steps, learner_samples, stop_event)
            if stop_event.is_set():
                break
            obs = env.reset()
            observations = [np.asarray(obs, dtype=np.float32)]
            actions = []
            rewards = []
            discounts = []


@torch.no_grad()
def evaluate_policy(
    config,
    policy_state: Dict[str, torch.Tensor],
    fixed_start_end,
    episodes: int,
    device: torch.device,
) -> Dict[str, float]:
    policy = PolicyNetwork(
        observation_dim=config.resolved_observation_dim(),
        action_dim=config.action_dim,
        hidden_layer_sizes=config.hidden_layer_sizes,
        obs_dim=config.obs_dim,
        use_image_obs=config.use_image_obs,
    ).to(device)
    policy.load_state_dict(policy_state)
    policy.eval()

    metric_values = []
    for episode_idx in range(episodes):
        env, _ = env_utils.make_environment(
            config.env_name,
            config.start_index,
            config.end_index,
            seed=config.seed + 50_000 + episode_idx,
            fixed_start_end=fixed_start_end,
        )
        obs = env.reset(seed=config.seed + 50_000 + episode_idx)
        success_observer = SuccessObserver()
        distance_observer = DistanceObserver(config.obs_dim, config.start_index, config.end_index)
        success_observer.observe_first(obs)
        distance_observer.observe_first(obs)
        done = False
        while not done:
            action = policy.act(obs, device=device, deterministic=True)
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            obs, reward, done, info = env.step(action)
            success_observer.observe(obs, action, reward, done, info)
            distance_observer.observe(obs, action, reward, done, info)
        metrics = {}
        metrics.update(success_observer.get_metrics())
        metrics.update(distance_observer.get_metrics())
        metric_values.append(metrics)

    out = {}
    for key in metric_values[0]:
        out[f"eval/{key}"] = float(np.mean([metrics[key] for metrics in metric_values]))
    return out
