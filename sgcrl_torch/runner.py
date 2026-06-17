"""Training runtime for the PyTorch SGCRL implementation."""

from __future__ import annotations

import os
import time
from queue import Full
from typing import Dict

import numpy as np
import torch
import torch.multiprocessing as mp

import sgcrl_torch.env_utils as env_utils
from sgcrl_torch.actors import actor_loop, evaluate_policy
from sgcrl_torch.config import ContrastiveConfig
from sgcrl_torch.learner import ContrastiveLearner
from sgcrl_torch.logging_utils import Logger
from sgcrl_torch.replay import EpisodeReplayBuffer
from sgcrl_torch.utils import action_dim, resolve_device, set_global_seeds


FIXED_GOAL_DICT: Dict[str, object] = {
    "point_Spiral11x11": [
        np.array([5, 5], dtype=float),
        np.array([10, 10], dtype=float),
    ],
    "sawyer_bin": np.array([0.12, 0.7, 0.02]),
    "sawyer_box": np.array([0.0, 0.75, 0.133]),
    "sawyer_peg": np.array([-0.3, 0.6, 0.0]),
}


def _fixed_goal_for(config: ContrastiveConfig):
    if not config.fix_goals:
        return None
    return FIXED_GOAL_DICT.get(config.env_name)


def _eval_fixed_goal_for(config: ContrastiveConfig):
    return FIXED_GOAL_DICT.get(config.env_name)


def prepare_config(config: ContrastiveConfig) -> ContrastiveConfig:
    fixed_start_end = _fixed_goal_for(config)
    env, obs_dim = env_utils.make_environment(
        config.env_name,
        config.start_index,
        config.end_index,
        seed=config.seed,
        fixed_start_end=fixed_start_end,
    )
    low = np.asarray(env.action_space.low)
    high = np.asarray(env.action_space.high)
    if not np.allclose(low, -1.0) or not np.allclose(high, 1.0):
        raise ValueError("SGCRL expects actions bounded in [-1, 1]")

    config.obs_dim = obs_dim
    config.goal_dim = config.resolved_goal_dim()
    config.action_dim = action_dim(env.action_space)
    step_limit = getattr(env, "_step_limit", None)
    if step_limit is None:
        step_limit = getattr(getattr(env, "_environment", None), "_step_limit", None)
    config.max_episode_steps = int(step_limit or config.max_episode_steps)
    return config


def _publish_policy(weight_queues, version: int, state) -> None:
    for queue in weight_queues:
        while True:
            try:
                queue.put_nowait((version, state))
                break
            except Full:
                try:
                    queue.get_nowait()
                except Exception:
                    pass


def run_training(config: ContrastiveConfig) -> None:
    """Run asynchronous actor/learner SGCRL training."""
    set_global_seeds(config.seed)
    config = prepare_config(config)
    device = resolve_device(config.device)

    run_dir = os.path.join(config.log_dir, f"{config.alg_name}_{config.env_name}_{config.seed}")
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger("learner", run_dir, add_uid=config.add_uid, time_delta=config.log_every)

    replay = EpisodeReplayBuffer(
        obs_dim=config.obs_dim,
        goal_dim=config.goal_dim,
        start_index=config.start_index,
        end_index=config.end_index,
        max_replay_size=config.max_replay_size,
        max_episode_steps=config.max_episode_steps + 1,
        discount=config.discount,
        seed=config.seed,
    )
    learner = ContrastiveLearner(config, device)

    ctx = mp.get_context(config.start_method)
    episode_queue = ctx.Queue(maxsize=config.episode_queue_size)
    actor_steps = ctx.Value("i", 0)
    learner_samples = ctx.Value("q", 0)
    stop_event = ctx.Event()

    fixed_start_end = _fixed_goal_for(config)
    actors = []
    weight_queues = []
    initial_policy_state = learner.policy_state_numpy()
    for actor_id in range(config.num_actors):
        weight_queue = ctx.Queue(maxsize=2)
        weight_queue.put((0, initial_policy_state))
        weight_queues.append(weight_queue)
        process = ctx.Process(
            target=actor_loop,
            args=(
                actor_id,
                config,
                fixed_start_end,
                episode_queue,
                weight_queue,
                actor_steps,
                learner_samples,
                stop_event,
            ),
        )
        process.daemon = True
        process.start()
        actors.append(process)

    learner_steps = 0
    last_checkpoint_step = 0
    try:
        while True:
            replay.drain_queue(episode_queue, max_items=64)
            with actor_steps.get_lock():
                current_actor_steps = actor_steps.value
            done_collecting = current_actor_steps >= config.max_number_of_steps

            if replay.num_steps < config.min_replay_size:
                with learner_samples.get_lock():
                    current_learner_samples = learner_samples.value
                logger.write(
                    {
                        "actor_steps": current_actor_steps,
                        "learner_samples": current_learner_samples,
                        "sample_to_insert_ratio": current_learner_samples / max(current_actor_steps, 1),
                        "replay_steps": replay.num_steps,
                        "replay_episodes": replay.num_episodes,
                        "learner_steps": learner_steps,
                        **replay.environment_metrics(),
                    }
                )
                if done_collecting:
                    break
                time.sleep(0.1)
                continue

            started = time.time()
            metrics = {}
            for _ in range(config.num_sgd_steps_per_step):
                replay.drain_queue(episode_queue, max_items=16)
                batch = replay.sample(config.batch_size, device)
                metrics = learner.update(batch)
                with learner_samples.get_lock():
                    learner_samples.value += config.batch_size
            elapsed = max(time.time() - started, 1e-6)
            learner_steps += 1
            with learner_samples.get_lock():
                current_learner_samples = learner_samples.value

            _publish_policy(weight_queues, learner_steps, learner.policy_state_numpy())

            metrics.update(
                {
                    "actor_steps": current_actor_steps,
                    "learner_steps": learner_steps,
                    "sgd_steps": learner.num_sgd_steps,
                    "learner_samples": current_learner_samples,
                    "sample_to_insert_ratio": current_learner_samples / max(current_actor_steps, 1),
                    "replay_steps": replay.num_steps,
                    "replay_episodes": replay.num_episodes,
                    "steps_per_second": config.num_sgd_steps_per_step / elapsed,
                    **replay.environment_metrics(),
                }
            )

            if config.eval_every and learner_steps % config.eval_every == 0:
                eval_metrics = evaluate_policy(
                    config,
                    learner.policy_state_cpu(),
                    _eval_fixed_goal_for(config),
                    config.eval_episodes,
                    device,
                )
                metrics.update(eval_metrics)

            logger.write(metrics)

            if config.checkpoint_every and learner_steps - last_checkpoint_step >= config.checkpoint_every:
                learner.save(os.path.join(run_dir, "checkpoints", f"learner_{learner_steps}.pt"))
                last_checkpoint_step = learner_steps

            if done_collecting:
                break

        replay.drain_queue(episode_queue, max_items=None)
        learner.save(os.path.join(run_dir, "checkpoints", "learner_final.pt"))
    finally:
        stop_event.set()
        for process in actors:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        logger.close()
