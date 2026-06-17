#!/usr/bin/env python3
"""Visualize the Appendix A goal-encoder norm experiment.

This script probes an early critic checkpoint before the first success. It
uniformly samples candidate target positions x_i, converts each x_i into a goal
state where the object is at x_i and the end effector is held at the fixed
grasping offset from x_i, then plots ||psi(x_i)||_2^2 for the critic goal
encoder.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import sgcrl_torch.env_utils as env_utils
from sgcrl_torch.networks import ContrastiveQNetwork, PolicyNetwork
from sgcrl_torch.runner import FIXED_GOAL_DICT
from sgcrl_torch.utils import obs_to_goal_2d, resolve_device, set_global_seeds


DEFAULT_CHECKPOINT = "logs/contrastive_cpc_sawyer_bin_0/checkpoints/learner_final.pt"
DEFAULT_WORKSPACE_BOUNDS = {
    "sawyer_bin": ((-0.30, 0.15), (0.55, 0.9), (0.02, 0.15)),
    "sawyer_box": ((-0.25, 0.25), (0.55, 0.9), (0.02, 0.16)),
    "sawyer_peg": ((-0.45, 0.05), (0.45, 0.85), (0.0, 0.2)),
}
DEFAULT_GRASP_OFFSETS = {
    "sawyer_bin": (0.0, 0.0, 0.03),
    "sawyer_box": (0.0, 0.0, 0.03),
    "sawyer_peg": (0.13, 0.0, 0.03),
}
DEFAULT_GRIPPER = 0.4
DEFAULT_BOX_QUAT = (0.707, 0.0, 0.0, 0.707)


def parse_pair(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated values, e.g. -0.25,0.25")
    lo, hi = float(parts[0]), float(parts[1])
    if hi < lo:
        raise argparse.ArgumentTypeError("Upper bound must be greater than or equal to lower bound")
    return lo, hi


def parse_triple(value: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values, e.g. 0,0,0.03")
    return float(parts[0]), float(parts[1]), float(parts[2])


def parse_quat(value: str) -> tuple[float, float, float, float]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Expected four comma-separated values, e.g. 0.707,0,0,0.707")
    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample many synthetic goal states x_i and plot the squared norm of "
            "the critic goal-state encoder, ||psi(x_i)||_2^2."
        )
    )
    parser.add_argument("--checkpoint_path", "--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", default="renders/norm_visualizer")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None, help="Defaults to the checkpoint seed.")
    parser.add_argument("--num_samples", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--num_rollouts", type=int, default=5, help="Policy rollouts to overlay; use 0 to disable.")
    parser.add_argument("--max_episode_length", type=int, default=None)
    parser.add_argument("--xlim", type=parse_pair, default=None, help="Uniform x-position bounds as min,max.")
    parser.add_argument("--ylim", type=parse_pair, default=None, help="Uniform y-position bounds as min,max.")
    parser.add_argument("--zlim", type=parse_pair, default=None, help="Uniform z-position bounds as min,max.")
    parser.add_argument(
        "--ee_offset",
        type=parse_triple,
        default=None,
        help="End-effector offset from sampled object position x_i. Defaults to the env grasping offset.",
    )
    parser.add_argument("--gripper", type=float, default=DEFAULT_GRIPPER)
    parser.add_argument(
        "--box_quat",
        type=parse_quat,
        default=DEFAULT_BOX_QUAT,
        help="Object quaternion used when probing sawyer_box checkpoints.",
    )
    parser.add_argument("--point_size", type=float, default=4.0)
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to checkpoint stem.")
    return parser.parse_args()


def torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_critic(config, checkpoint: dict, device: torch.device) -> ContrastiveQNetwork:
    critic = ContrastiveQNetwork(
        obs_dim=config.obs_dim,
        goal_dim=config.resolved_goal_dim(),
        action_dim=config.action_dim,
        hidden_layer_sizes=config.hidden_layer_sizes,
        repr_dim=int(config.repr_dim),
        repr_norm=config.repr_norm,
        twin_q=config.twin_q,
        use_image_obs=config.use_image_obs,
    ).to(device)
    critic.load_state_dict(checkpoint["critic"])
    critic.eval()
    return critic


def build_policy(config, checkpoint: dict, device: torch.device) -> Optional[PolicyNetwork]:
    if "policy" not in checkpoint:
        return None
    policy = PolicyNetwork(
        observation_dim=config.resolved_observation_dim(),
        action_dim=config.action_dim,
        hidden_layer_sizes=config.hidden_layer_sizes,
        obs_dim=config.obs_dim,
        use_image_obs=config.use_image_obs,
    ).to(device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    return policy


def bounds_for_env(
    env_name: str,
    xlim: Optional[tuple[float, float]],
    ylim: Optional[tuple[float, float]],
    zlim: Optional[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    default = DEFAULT_WORKSPACE_BOUNDS.get(env_name)
    if default is None and (xlim is None or ylim is None or zlim is None):
        raise ValueError(
            f"No default sampling bounds are known for {env_name!r}; pass --xlim, --ylim, and --zlim."
        )
    return xlim or default[0], ylim or default[1], zlim or default[2]


def grasp_offset_for_env(env_name: str, ee_offset: Optional[tuple[float, float, float]]) -> np.ndarray:
    if ee_offset is not None:
        return np.asarray(ee_offset, dtype=np.float32)
    return np.asarray(DEFAULT_GRASP_OFFSETS.get(env_name, (0.0, 0.0, 0.03)), dtype=np.float32)


def sample_positions(
    rng: np.random.Generator,
    num_samples: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
) -> np.ndarray:
    low = np.asarray([xlim[0], ylim[0], zlim[0]], dtype=np.float32)
    high = np.asarray([xlim[1], ylim[1], zlim[1]], dtype=np.float32)
    return rng.uniform(low=low, high=high, size=(num_samples, 3)).astype(np.float32)


def states_from_positions(
    config,
    positions: np.ndarray,
    ee_offset: np.ndarray,
    gripper: float,
    box_quat: tuple[float, float, float, float],
) -> np.ndarray:
    hand_positions = positions + ee_offset[None]
    gripper_column = np.full((positions.shape[0], 1), gripper, dtype=np.float32)

    if config.env_name in ("sawyer_bin", "sawyer_peg"):
        states = np.concatenate([hand_positions, gripper_column, positions], axis=1)
    elif config.env_name == "sawyer_box":
        quat = np.repeat(np.asarray(box_quat, dtype=np.float32)[None], positions.shape[0], axis=0)
        states = np.concatenate([hand_positions, gripper_column, positions, quat], axis=1)
    else:
        raise NotImplementedError(
            "The goal-state construction in this visualizer is implemented for sawyer_bin, "
            "sawyer_box, and sawyer_peg checkpoints."
        )

    if states.shape[1] != config.obs_dim:
        raise ValueError(f"Synthetic state dim {states.shape[1]} does not match checkpoint obs_dim={config.obs_dim}.")
    return states.astype(np.float32)


def close_env(env) -> None:
    close = getattr(env, "close", None)
    if close is not None:
        close()


def object_positions_from_observations(observations: np.ndarray) -> np.ndarray:
    if observations.shape[1] < 7:
        raise ValueError("Sawyer rollout observations are expected to contain object xyz at state indices 4:7.")
    return observations[:, 4:7]


def fixed_goal_position(config) -> Optional[np.ndarray]:
    goal = FIXED_GOAL_DICT.get(config.env_name)
    if goal is None:
        return None
    return np.asarray(goal, dtype=np.float32).reshape(-1)[:3]


def rollout_policy(
    config,
    policy: Optional[PolicyNetwork],
    device: torch.device,
    seed: int,
    num_rollouts: int,
    max_episode_length: Optional[int],
) -> list[np.ndarray]:
    if policy is None or num_rollouts <= 0:
        return []

    trajectories = []
    fixed_goal = fixed_goal_position(config)
    limit = int(max_episode_length or config.max_episode_steps)
    for rollout_idx in range(num_rollouts):
        episode_seed = seed + 20_000 + rollout_idx
        env, _ = env_utils.make_environment(
            config.env_name,
            config.start_index,
            config.end_index,
            seed=episode_seed,
            fixed_start_end=fixed_goal,
        )
        obs = env.reset(seed=episode_seed)
        observations = [np.asarray(obs, dtype=np.float32).copy()]

        done = False
        steps = 0
        while not done and steps < limit:
            action = policy.act(obs, device=device, deterministic=True)
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            obs, _, done, _ = env.step(action)
            observations.append(np.asarray(obs, dtype=np.float32).copy())
            steps += 1

        close_env(env)
        trajectories.append(object_positions_from_observations(np.stack(observations, axis=0)))
    return trajectories


@torch.no_grad()
def goal_encoder_norm_squared(
    critic: ContrastiveQNetwork,
    goal_states: np.ndarray,
    config,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    goal_inputs = obs_to_goal_2d(goal_states, config.start_index, config.end_index)
    norms = []
    for start in range(0, goal_inputs.shape[0], batch_size):
        batch = torch.as_tensor(goal_inputs[start : start + batch_size], dtype=torch.float32, device=device)
        encoded = critic.g_encoder(batch)
        if critic.repr_norm:
            encoded = encoded / torch.clamp(encoded.norm(dim=1, keepdim=True), min=1e-6)
        norms.append(torch.log(torch.sum(encoded * encoded, dim=1)).cpu().numpy())
    return np.concatenate(norms, axis=0)


def write_plot(
    output_path: Path,
    positions: np.ndarray,
    norm_squared: np.ndarray,
    rollout_trajectories: list[np.ndarray],
    fixed_goal: Optional[np.ndarray],
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    title: str,
    point_size: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    xlim, ylim, zlim = bounds
    panels = (
        (axes[0], positions[:, 0], positions[:, 1], xlim, ylim, "x", "y"),
        (axes[1], positions[:, 0], positions[:, 2], xlim, zlim, "x", "z"),
    )
    vivid_colors = np.asarray(
        [
            "#ff1744",
            "#00c853",
            "#2979ff",
            "#ff9100",
            "#d500f9",
            "#00e5ff",
            "#ffea00",
            "#f50057",
            "#76ff03",
            "#651fff",
        ]
    )
    colors = vivid_colors[np.arange(max(len(rollout_trajectories), 1)) % len(vivid_colors)]
    for panel_idx, (axis, horizontal, vertical, hlim, vlim, xlabel, ylabel) in enumerate(panels):
        scatter = axis.scatter(
            horizontal,
            vertical,
            c=norm_squared,
            s=point_size,
            cmap="viridis",
            linewidths=0,
            alpha=0.85,
        )
        for trajectory, color in zip(rollout_trajectories, colors):
            trajectory_horizontal = trajectory[:, 0]
            trajectory_vertical = trajectory[:, 1 if panel_idx == 0 else 2]
            (line,) = axis.plot(
                trajectory_horizontal,
                trajectory_vertical,
                color=color,
                linewidth=2.6,
                alpha=1.0,
                zorder=3,
            )
            line.set_path_effects(
                [
                    path_effects.Stroke(linewidth=4.0, foreground="black", alpha=0.65),
                    path_effects.Normal(),
                ]
            )
            axis.scatter(
                trajectory_horizontal[0],
                trajectory_vertical[0],
                marker="D",
                s=58,
                c="red",
                edgecolors="white",
                linewidths=0.9,
                zorder=5,
            )
        if fixed_goal is not None:
            axis.scatter(
                fixed_goal[0],
                fixed_goal[1 if panel_idx == 0 else 2],
                marker="*",
                s=220,
                c="gold",
                edgecolors="black",
                linewidths=0.9,
                zorder=6,
            )
        axis.set_xlim(*hlim)
        axis.set_ylim(*vlim)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_aspect("auto")
        colorbar = fig.colorbar(scatter, ax=axis)
        colorbar.set_label(r"log $||\psi(x_i)||_2^2$")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint = torch_load(checkpoint_path, device)
    config = checkpoint["config"]
    config.device = args.device
    seed = config.seed if args.seed is None else args.seed
    set_global_seeds(seed)

    critic = build_critic(config, checkpoint, device)
    policy = build_policy(config, checkpoint, device)
    bounds = bounds_for_env(config.env_name, args.xlim, args.ylim, args.zlim)
    ee_offset = grasp_offset_for_env(config.env_name, args.ee_offset)
    rng = np.random.default_rng(seed)
    positions = sample_positions(rng, args.num_samples, *bounds)
    goal_states = states_from_positions(config, positions, ee_offset, args.gripper, args.box_quat)
    norm_squared = goal_encoder_norm_squared(critic, goal_states, config, device, args.batch_size)
    rollout_trajectories = rollout_policy(
        config,
        policy,
        device,
        seed,
        args.num_rollouts,
        args.max_episode_length,
    )
    fixed_goal = fixed_goal_position(config)

    prefix = args.prefix or checkpoint_path.stem
    png_path = output_dir / f"{prefix}_goal_encoder_norm_squared.png"
    npz_path = output_dir / f"{prefix}_goal_encoder_norm_squared.npz"
    write_plot(
        png_path,
        positions,
        norm_squared,
        rollout_trajectories,
        fixed_goal,
        bounds,
        title=f"{config.env_name} | {checkpoint_path.name} | pre-success goal encoder",
        point_size=args.point_size,
    )
    np.savez_compressed(
        npz_path,
        positions=positions,
        goal_states=goal_states,
        goal_encoder_norm_squared=norm_squared,
        rollout_trajectories=np.array(rollout_trajectories, dtype=object),
        fixed_goal=fixed_goal,
        xlim=np.asarray(bounds[0], dtype=np.float32),
        ylim=np.asarray(bounds[1], dtype=np.float32),
        zlim=np.asarray(bounds[2], dtype=np.float32),
        ee_offset=ee_offset,
        gripper=np.asarray(args.gripper, dtype=np.float32),
        checkpoint=str(checkpoint_path),
    )

    print(f"Wrote {png_path}")
    print(f"Wrote {npz_path}")
    print(
        "norm_squared: "
        f"min={float(norm_squared.min()):.6g} "
        f"mean={float(norm_squared.mean()):.6g} "
        f"max={float(norm_squared.max()):.6g}"
    )


if __name__ == "__main__":
    main()
