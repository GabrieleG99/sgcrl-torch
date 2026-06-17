#!/usr/bin/env python
"""Run SGCRL with the PyTorch actor/learner runtime."""

from __future__ import annotations

import argparse
import contextlib
import os
import re

import torch

import sgcrl_torch.learner as learner_module
from sgcrl_torch.config import ContrastiveConfig
from sgcrl_torch.runner import run_training


def _hidden_layers(value: str):
    return tuple(int(part) for part in value.split(",") if part)


def _torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_warm_start(learner, checkpoint_path: str) -> None:
    checkpoint = _torch_load(checkpoint_path, learner.device)
    learner.policy.load_state_dict(checkpoint["policy"])
    learner.critic.load_state_dict(checkpoint["critic"])
    learner.target_critic.load_state_dict(checkpoint["target_critic"])
    learner.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
    learner.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
    learner.num_sgd_steps = int(checkpoint.get("num_sgd_steps", learner.num_sgd_steps))

    if learner.adaptive_entropy:
        if "log_alpha" not in checkpoint or "alpha_optimizer" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain adaptive entropy state, but this run uses --adaptive_entropy."
            )
        learner.log_alpha.data.copy_(checkpoint["log_alpha"].to(learner.device))
        learner.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])


def _checkpoint_step_offset(checkpoint_path: str) -> int:
    match = re.fullmatch(r"learner_(\d+)\.pt", os.path.basename(checkpoint_path))
    if match is None:
        return 0
    return int(match.group(1))


def _offset_checkpoint_path(path: str, offset: int) -> str:
    if offset <= 0:
        return path
    basename = os.path.basename(path)
    match = re.fullmatch(r"learner_(\d+)\.pt", basename)
    if match is None:
        return path
    next_step = int(match.group(1)) + offset
    return os.path.join(os.path.dirname(path), f"learner_{next_step}.pt")


@contextlib.contextmanager
def patched_warm_start(checkpoint_path: str | None):
    if not checkpoint_path:
        yield
        return

    original_init = learner_module.ContrastiveLearner.__init__
    original_save = learner_module.ContrastiveLearner.save
    checkpoint_step_offset = _checkpoint_step_offset(checkpoint_path)

    def _init_with_warm_start(self, config, device):
        original_init(self, config, device)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Warm-start checkpoint does not exist: {checkpoint_path}")
        _load_warm_start(self, checkpoint_path)
        print(f"Warm-started learner from {checkpoint_path}")
        if checkpoint_step_offset > 0:
            print(f"Offsetting future numbered checkpoint names by {checkpoint_step_offset} learner steps")

    def _save_with_warm_start_offset(self, path):
        return original_save(self, _offset_checkpoint_path(path, checkpoint_step_offset))

    learner_module.ContrastiveLearner.__init__ = _init_with_warm_start
    learner_module.ContrastiveLearner.save = _save_with_warm_start_offset
    try:
        yield
    finally:
        learner_module.ContrastiveLearner.__init__ = original_init
        learner_module.ContrastiveLearner.save = original_save


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch SGCRL")
    parser.add_argument("--log_dir_path", default="logs/", help="Where to log metrics")
    parser.add_argument("--time_delta_minutes", type=int, default=5, help="Checkpoint cadence metadata")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--add_uid", action="store_true", help="Save logs in a unique subdirectory")
    parser.add_argument("--alg", default="contrastive_cpc", choices=["contrastive_nce", "contrastive_cpc", "c_learning", "nce+c_learning"])
    parser.add_argument("--env", default="sawyer_bin", help="sawyer_bin, sawyer_box, sawyer_peg, point_Spiral11x11")
    parser.add_argument("--num_steps", type=int, default=8_000_000, help="Maximum actor environment steps")
    parser.add_argument("--sample_goals", action="store_true", help="Sample goals instead of using the fixed single goal")
    parser.add_argument(
        "--warm_start_checkpoint",
        default=None,
        help="Load policy, critic, target critic, optimizers, and alpha state from a learner checkpoint",
    )

    parser.add_argument("--num_actors", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--actor_device", default="cpu")
    parser.add_argument("--start_method", default="spawn", choices=["spawn", "fork", "forkserver"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--min_replay_size", type=int, default=10_000)
    parser.add_argument("--max_replay_size", type=int, default=1_000_000)
    parser.add_argument("--num_sgd_steps_per_step", type=int, default=64)
    parser.add_argument("--samples_per_insert", type=float, default=256.0)
    parser.add_argument("--actor_update_period", type=int, default=100)
    parser.add_argument("--episode_queue_size", type=int, default=128)
    parser.add_argument("--no_rate_limit_actors", action="store_true", help="Disable Reverb-style actor backpressure")
    parser.add_argument("--rate_limit_error_buffer", type=int, default=-1, help="Extra actor steps allowed beyond the sample/insert ratio")
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--log_every", type=float, default=10.0)

    parser.add_argument("--actor_learning_rate", type=float, default=3e-4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--hidden_layer_sizes", type=_hidden_layers, default=(256, 256))
    parser.add_argument("--repr_dim", type=int, default=64)
    parser.add_argument("--repr_norm", action="store_true")
    parser.add_argument("--random_goals", type=float, default=0.5)
    parser.add_argument("--entropy_coefficient", type=float, default=0.0)
    parser.add_argument("--adaptive_entropy", action="store_true", help="Use learned alpha instead of fixed entropy_coefficient")
    parser.add_argument("--target_entropy", type=float, default=0.0)
    parser.add_argument("--no_action_entropy", action="store_true")
    parser.add_argument("--no_random_actor", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ContrastiveConfig(
        seed=args.seed,
        add_uid=args.add_uid,
        time_delta_minutes=args.time_delta_minutes,
        log_dir=args.log_dir_path,
        env_name=args.env,
        max_number_of_steps=args.num_steps,
        fix_goals=not args.sample_goals,
        num_actors=args.num_actors,
        device=args.device,
        actor_device=args.actor_device,
        start_method=args.start_method,
        batch_size=args.batch_size,
        min_replay_size=args.min_replay_size,
        max_replay_size=args.max_replay_size,
        num_sgd_steps_per_step=args.num_sgd_steps_per_step,
        samples_per_insert=args.samples_per_insert,
        actor_update_period=args.actor_update_period,
        episode_queue_size=args.episode_queue_size,
        rate_limit_actors=not args.no_rate_limit_actors,
        rate_limit_error_buffer=args.rate_limit_error_buffer,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        log_every=args.log_every,
        actor_learning_rate=args.actor_learning_rate,
        learning_rate=args.learning_rate,
        discount=args.discount,
        tau=args.tau,
        hidden_layer_sizes=args.hidden_layer_sizes,
        repr_dim=args.repr_dim,
        repr_norm=args.repr_norm,
        random_goals=args.random_goals,
        entropy_coefficient=None if args.adaptive_entropy else args.entropy_coefficient,
        target_entropy=args.target_entropy,
        use_action_entropy=not args.no_action_entropy,
        use_random_actor=not args.no_random_actor,
    )
    config.apply_algorithm(args.alg)
    print(f"Using env {args.env}...")
    print(f"Using alg {args.alg}...")
    print(f"Using random seed {args.seed}...")
    with patched_warm_start(args.warm_start_checkpoint):
        run_training(config)


if __name__ == "__main__":
    print("Starting SGCRL training...")
    main()
