#!/usr/bin/env python3
"""Evaluate SGCRL robustness under object perturbations.

This reproduces the paper's "Further training: agent develops robustness and
self-recovery" perturbation experiment for Sawyer box and Sawyer peg policies.
It loads one or more trained checkpoints, perturbs the manipulated object by up
to 5 cm either at reset or halfway through the episode, and records whether the
policy still reaches the fixed goal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

# Set this before importing Gymnasium/Meta-World/MuJoCo modules.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import sgcrl_torch.env_utils as env_utils
from render_sawyer_box import (
    build_policy,
    render_display,
    render_frame,
    torch_load,
    unwrap_env,
    write_video,
)
from sgcrl_torch.runner import FIXED_GOAL_DICT
from sgcrl_torch.utils import obs_to_goal_1d, resolve_device, set_global_seeds


SUPPORTED_ENVS = {"sawyer_box", "sawyer_peg"}
CSV_FIELDS = [
    "label",
    "checkpoint",
    "checkpoint_step",
    "env_name",
    "setting",
    "episode",
    "seed",
    "steps",
    "success",
    "success_before_perturb",
    "success_after_perturb",
    "first_success_step",
    "first_success_after_perturb_step",
    "recovery_steps",
    "reset_dist",
    "pre_perturb_dist",
    "post_perturb_dist",
    "final_dist",
    "min_dist",
    "post_perturb_min_dist",
    "perturb_step",
    "requested_dx",
    "requested_dy",
    "requested_dz",
    "observed_dx",
    "observed_dy",
    "observed_dz",
    "observed_displacement",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run static and dynamic object perturbation evaluations for trained "
            "SGCRL Sawyer box/peg checkpoints."
        )
    )
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Learner checkpoint paths to evaluate.")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for checkpoints, e.g. early further. Must match --checkpoints length.",
    )
    parser.add_argument("--output_dir", default="renders/self_recovery_perturbations")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["static", "dynamic"],
        choices=["none", "static", "dynamic"],
        help="none is an unperturbed baseline; static perturbs after reset; dynamic perturbs mid-episode.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Defaults to each checkpoint's training seed.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample_goals", action="store_true", help="Sample environment goals instead of fixed goals.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using policy means.")
    parser.add_argument("--perturbation_min", type=float, default=0.0)
    parser.add_argument("--perturbation_max", type=float, default=0.05)
    parser.add_argument(
        "--offset_mode",
        choices=["positive", "signed", "signed_xy"],
        default="positive",
        help=(
            "positive samples each axis from [min, max], matching the paper text; "
            "signed uses random signs on all axes; signed_xy signs x/y and keeps z positive."
        ),
    )
    parser.add_argument(
        "--dynamic_perturb_step",
        type=int,
        default=None,
        help="Step at which dynamic perturbations are applied. Defaults to half the episode limit.",
    )
    parser.add_argument(
        "--render_episodes",
        type=int,
        default=0,
        help="Write videos for the first N episodes of each checkpoint/setting.",
    )
    parser.add_argument("--display", action="store_true", help="Also call env.render() each step.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera_name", default="corner2")
    parser.add_argument(
        "--render_backend",
        choices=("auto", "wrapper", "sim", "mujoco_renderer"),
        default="auto",
    )
    parser.add_argument("--format", choices=("gif", "mp4"), default="gif")
    return parser.parse_args()


def checkpoint_step(path: Path) -> Optional[int]:
    match = re.fullmatch(r"learner_(\d+)\.pt", path.name)
    if match is None:
        return None
    return int(match.group(1))


def default_label(path: Path) -> str:
    if len(path.parts) >= 3 and path.parts[-2] == "checkpoints":
        return f"{path.parts[-3]}_{path.stem}"
    return path.stem


def validate_args(args: argparse.Namespace) -> list[str]:
    checkpoint_paths = [Path(path) for path in args.checkpoints]
    missing = [str(path) for path in checkpoint_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))

    if args.labels is None or len(args.labels) == 0:
        return [default_label(path) for path in checkpoint_paths]
    if len(args.labels) != len(checkpoint_paths):
        raise ValueError("--labels must have the same length as --checkpoints")
    return list(args.labels)


def goal_distance(observation: np.ndarray, config) -> float:
    state = observation[: config.obs_dim]
    goal = observation[config.obs_dim :]
    achieved = obs_to_goal_1d(state, config.start_index, config.end_index)
    return float(np.linalg.norm(achieved - goal))


def current_observation(env) -> np.ndarray:
    """Read the current SGCRL observation after directly mutating MuJoCo state."""
    unwrapped = unwrap_env(env)
    if not hasattr(unwrapped, "_get_sgcrl_obs"):
        raise TypeError("Expected a Sawyer SGCRL environment with _get_sgcrl_obs().")
    full_observation = unwrapped._get_sgcrl_obs()
    if hasattr(env, "_convert_observation"):
        return env._convert_observation(full_observation)
    return np.asarray(full_observation, dtype=np.float32)


def sample_offset(
    rng: np.random.Generator,
    min_delta: float,
    max_delta: float,
    mode: str,
) -> np.ndarray:
    if min_delta < 0.0 or max_delta < 0.0:
        raise ValueError("Perturbation magnitudes must be non-negative.")
    if min_delta > max_delta:
        raise ValueError("--perturbation_min must be <= --perturbation_max")

    offset = rng.uniform(min_delta, max_delta, size=3)
    if mode == "signed":
        offset *= rng.choice([-1.0, 1.0], size=3)
    elif mode == "signed_xy":
        offset[:2] *= rng.choice([-1.0, 1.0], size=2)
    elif mode != "positive":
        raise ValueError(f"Unknown offset mode: {mode}")
    return offset.astype(np.float64)


def perturb_object_qpos(env, offset: np.ndarray) -> dict[str, object]:
    """Translate the manipulated object by adding offset to MuJoCo qpos[9:12]."""
    unwrapped = unwrap_env(env)
    if not hasattr(unwrapped, "_get_pos_objects"):
        raise TypeError("Expected a Sawyer environment with _get_pos_objects().")
    if not hasattr(unwrapped, "data") or not hasattr(unwrapped, "set_state"):
        raise TypeError("Expected a MuJoCo environment exposing data and set_state().")

    before = np.asarray(unwrapped._get_pos_objects(), dtype=np.float64).copy()
    qpos = unwrapped.data.qpos.flat.copy()
    qvel = unwrapped.data.qvel.flat.copy()
    if qpos.shape[0] < 12:
        raise ValueError(f"Cannot perturb object: qpos has only {qpos.shape[0]} entries.")

    qpos[9:12] += offset
    if qvel.shape[0] > 9:
        qvel[9 : min(15, qvel.shape[0])] = 0.0
    unwrapped.set_state(qpos, qvel)

    after = np.asarray(unwrapped._get_pos_objects(), dtype=np.float64).copy()
    observed = after - before
    return {
        "object_before": before,
        "object_after": after,
        "observed_offset": observed,
        "observed_displacement": float(np.linalg.norm(observed)),
    }


def close_env(env) -> None:
    close = getattr(env, "close", None)
    if close is not None:
        close()


def normalize_optional(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def run_episode(
    *,
    env,
    policy,
    config,
    device: torch.device,
    rng: np.random.Generator,
    setting: str,
    episode_idx: int,
    episode_seed: int,
    args: argparse.Namespace,
    label: str,
    checkpoint_path: Path,
    should_render: bool,
    video_dir: Path,
) -> dict[str, object]:
    obs = env.reset(seed=episode_seed)
    reset_dist = goal_distance(obs, config)
    distances = [reset_dist]
    post_perturb_distances: list[float] = []
    perturb_step = None
    pre_perturb_dist = None
    post_perturb_dist = None
    requested_offset = np.zeros(3, dtype=np.float64)
    observed_offset = np.zeros(3, dtype=np.float64)
    observed_displacement = 0.0

    max_steps = int(getattr(env, "_step_limit", config.max_episode_steps))
    dynamic_step = args.dynamic_perturb_step
    if dynamic_step is None:
        dynamic_step = max_steps // 2
    dynamic_step = max(0, min(int(dynamic_step), max_steps))

    frames = []
    if should_render:
        frames.append(render_frame(env, args.width, args.height, args.camera_name, args.render_backend))
    if args.display:
        render_display(env)

    if setting == "static":
        perturb_step = 0
        pre_perturb_dist = reset_dist
        requested_offset = sample_offset(
            rng,
            args.perturbation_min,
            args.perturbation_max,
            args.offset_mode,
        )
        perturb_info = perturb_object_qpos(env, requested_offset)
        observed_offset = perturb_info["observed_offset"]
        observed_displacement = perturb_info["observed_displacement"]
        obs = current_observation(env)
        post_perturb_dist = goal_distance(obs, config)
        distances.append(post_perturb_dist)
        post_perturb_distances.append(post_perturb_dist)
        if should_render:
            frames.append(render_frame(env, args.width, args.height, args.camera_name, args.render_backend))

    done = False
    steps = 0
    success = False
    success_before_perturb = False
    success_after_perturb = False
    first_success_step = None
    first_success_after_perturb_step = None

    while not done:
        if setting == "dynamic" and perturb_step is None and steps >= dynamic_step:
            perturb_step = steps
            pre_perturb_dist = distances[-1]
            requested_offset = sample_offset(
                rng,
                args.perturbation_min,
                args.perturbation_max,
                args.offset_mode,
            )
            perturb_info = perturb_object_qpos(env, requested_offset)
            observed_offset = perturb_info["observed_offset"]
            observed_displacement = perturb_info["observed_displacement"]
            obs = current_observation(env)
            post_perturb_dist = goal_distance(obs, config)
            distances.append(post_perturb_dist)
            post_perturb_distances.append(post_perturb_dist)
            if should_render:
                frames.append(render_frame(env, args.width, args.height, args.camera_name, args.render_backend))

        action = policy.act(obs, device=device, deterministic=not args.stochastic)
        action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
        obs, reward, done, info = env.step(action)
        del info
        steps += 1

        dist = goal_distance(obs, config)
        distances.append(dist)
        if perturb_step is not None:
            post_perturb_distances.append(dist)

        step_success = float(reward) > 0.0
        if step_success:
            success = True
            if first_success_step is None:
                first_success_step = steps
            if perturb_step is None:
                success_before_perturb = True
            else:
                success_after_perturb = True
                if first_success_after_perturb_step is None:
                    first_success_after_perturb_step = steps

        if should_render:
            frames.append(render_frame(env, args.width, args.height, args.camera_name, args.render_backend))
        if args.display:
            render_display(env)

    if setting == "none":
        success_after_perturb = success
    elif perturb_step is None:
        perturb_step = steps

    recovery_steps = None
    if first_success_after_perturb_step is not None and perturb_step is not None:
        recovery_steps = first_success_after_perturb_step - perturb_step

    if should_render:
        video_path = video_dir / f"{label}_{setting}_episode_{episode_idx:03d}.{args.format}"
        write_video(video_path, frames, args.fps)
        print(f"Wrote {video_path}")

    return {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step(checkpoint_path),
        "env_name": config.env_name,
        "setting": setting,
        "episode": episode_idx,
        "seed": episode_seed,
        "steps": steps,
        "success": float(success),
        "success_before_perturb": float(success_before_perturb),
        "success_after_perturb": float(success_after_perturb),
        "first_success_step": first_success_step,
        "first_success_after_perturb_step": first_success_after_perturb_step,
        "recovery_steps": recovery_steps,
        "reset_dist": reset_dist,
        "pre_perturb_dist": pre_perturb_dist,
        "post_perturb_dist": post_perturb_dist,
        "final_dist": distances[-1],
        "min_dist": min(distances),
        "post_perturb_min_dist": min(post_perturb_distances) if post_perturb_distances else min(distances),
        "perturb_step": perturb_step,
        "requested_dx": float(requested_offset[0]),
        "requested_dy": float(requested_offset[1]),
        "requested_dz": float(requested_offset[2]),
        "observed_dx": float(observed_offset[0]),
        "observed_dy": float(observed_offset[1]),
        "observed_dz": float(observed_offset[2]),
        "observed_displacement": observed_displacement,
    }


def mean_present(rows: Iterable[dict[str, object]], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["label"], row["checkpoint"], row["env_name"], row["setting"])].append(row)

    summary = []
    for (label, checkpoint, env_name, setting), group_rows in sorted(groups.items()):
        summary.append(
            {
                "label": label,
                "checkpoint": checkpoint,
                "checkpoint_step": group_rows[0]["checkpoint_step"],
                "env_name": env_name,
                "setting": setting,
                "episodes": len(group_rows),
                "success_rate": mean_present(group_rows, "success"),
                "success_after_perturb_rate": mean_present(group_rows, "success_after_perturb"),
                "mean_recovery_steps": mean_present(group_rows, "recovery_steps"),
                "mean_reset_dist": mean_present(group_rows, "reset_dist"),
                "mean_post_perturb_dist": mean_present(group_rows, "post_perturb_dist"),
                "mean_final_dist": mean_present(group_rows, "final_dist"),
                "mean_min_dist": mean_present(group_rows, "min_dist"),
                "mean_post_perturb_min_dist": mean_present(group_rows, "post_perturb_min_dist"),
                "mean_observed_displacement": mean_present(group_rows, "observed_displacement"),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: normalize_optional(row.get(field)) for field in fields})


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_outputs(output_dir: Path, rows: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "episodes.csv", rows, CSV_FIELDS)
    write_csv(output_dir / "summary.csv", summary, list(summary[0].keys()) if summary else [])
    with (output_dir / "summary.json").open("w") as file:
        json.dump(json_safe(summary), file, indent=2, allow_nan=False)


def evaluate_checkpoint(
    checkpoint_path: Path,
    label: str,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    checkpoint = torch_load(checkpoint_path, device)
    config = checkpoint["config"]
    config.device = args.device
    if config.env_name not in SUPPORTED_ENVS:
        raise ValueError(
            f"{checkpoint_path} uses env {config.env_name!r}; this experiment supports "
            f"{', '.join(sorted(SUPPORTED_ENVS))}."
        )

    seed = int(config.seed if args.seed is None else args.seed)
    set_global_seeds(seed)
    policy = build_policy(config, checkpoint, device)
    fixed_goal = None if args.sample_goals else FIXED_GOAL_DICT.get(config.env_name)

    rows = []
    for setting in args.settings:
        video_dir = output_dir / "videos" / label / setting
        if args.render_episodes > 0:
            video_dir.mkdir(parents=True, exist_ok=True)
        for episode_idx in range(args.episodes):
            episode_seed = seed + 50_000 + episode_idx
            rng = np.random.default_rng(seed + 900_000 + episode_idx)
            env, _ = env_utils.make_environment(
                config.env_name,
                config.start_index,
                config.end_index,
                seed=episode_seed,
                fixed_start_end=fixed_goal,
            )
            try:
                row = run_episode(
                    env=env,
                    policy=policy,
                    config=config,
                    device=device,
                    rng=rng,
                    setting=setting,
                    episode_idx=episode_idx,
                    episode_seed=episode_seed,
                    args=args,
                    label=label,
                    checkpoint_path=checkpoint_path,
                    should_render=episode_idx < args.render_episodes,
                    video_dir=video_dir,
                )
                rows.append(row)
                print(
                    "label={label} setting={setting} episode={episode} "
                    "success={success:.0f} success_after_perturb={success_after_perturb:.0f} "
                    "final_dist={final_dist:.4f} post_perturb_min_dist={post_perturb_min_dist:.4f}".format(
                        **row
                    )
                )
            finally:
                close_env(env)
    return rows


def main() -> None:
    args = parse_args()
    labels = validate_args(args)
    checkpoint_paths = [Path(path) for path in args.checkpoints]
    output_dir = Path(args.output_dir)
    device = resolve_device(args.device)

    all_rows = []
    for checkpoint_path, label in zip(checkpoint_paths, labels):
        all_rows.extend(evaluate_checkpoint(checkpoint_path, label, args, device, output_dir))

    summary = build_summary(all_rows)
    write_outputs(output_dir, all_rows, summary)
    print(f"Wrote {output_dir / 'episodes.csv'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")
    for row in summary:
        print(
            "summary label={label} setting={setting} success_rate={success_rate:.3f} "
            "success_after_perturb_rate={success_after_perturb_rate:.3f} "
            "mean_recovery_steps={mean_recovery_steps:.2f}".format(**row)
        )


if __name__ == "__main__":
    main()
