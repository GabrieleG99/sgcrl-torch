#!/usr/bin/env python
"""Evaluate state-action representation geometry for a trained SGCRL critic.

This script implements an evaluation-only version of the state-action
representation experiment: sample state-action pairs, compute the critic's
phi(s, a), then test whether nearest neighbors in representation space produce
similar transition effects.

For point_* environments, states are sampled directly from free maze cells and
transition effects are computed with action noise disabled. For Sawyer
environments, the script falls back to sampling real one-step transitions from
the environment with random actions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import sgcrl_torch.env_utils as env_utils
from sgcrl_torch.config import ContrastiveConfig
from sgcrl_torch.networks import ContrastiveQNetwork
from sgcrl_torch.runner import FIXED_GOAL_DICT, prepare_config
from sgcrl_torch.utils import obs_to_goal_2d, resolve_device, set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether critic state-action representations phi(s,a) cluster "
            "state-action pairs with similar transition effects."
        )
    )
    parser.add_argument("--checkpoint_path", "--checkpoint", required=True)
    parser.add_argument("--output_dir", default="renders/state_action_repr_geometry")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None, help="Defaults to the checkpoint seed.")
    parser.add_argument("--env", default=None, help="Override env_name from the checkpoint config.")
    parser.add_argument("--num_samples", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--nearest_k", type=int, default=10)
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument(
        "--actions_per_state",
        type=int,
        default=16,
        help="Only used for point_* environments; samples evenly spaced action directions per state.",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=None,
        help="Only used for non-point environments. Defaults to --num_samples.",
    )
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to checkpoint stem.")
    return parser.parse_args()


def torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def config_from_checkpoint(checkpoint: dict, env_override: str | None) -> ContrastiveConfig:
    config = checkpoint.get("config")
    if config is None:
        if env_override is None:
            raise ValueError("Checkpoint has no config; pass --env so dimensions can be prepared.")
        config = ContrastiveConfig(env_name=env_override)
        config.apply_algorithm("contrastive_cpc")
    if env_override is not None:
        config.env_name = env_override
    if config.obs_dim <= 0 or config.action_dim <= 0 or config.goal_dim <= 0:
        config = prepare_config(config)
    return config


def build_critic(config: ContrastiveConfig, checkpoint: dict, device: torch.device) -> ContrastiveQNetwork:
    critic = ContrastiveQNetwork(
        obs_dim=config.obs_dim,
        goal_dim=config.resolved_goal_dim(),
        action_dim=config.action_dim,
        repr_dim=int(config.repr_dim),
        hidden_layer_sizes=config.hidden_layer_sizes,
        repr_norm=config.repr_norm,
        twin_q=config.twin_q,
        use_image_obs=config.use_image_obs,
    ).to(device)
    try:
        critic.load_state_dict(checkpoint["critic"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not load checkpoint into ContrastiveQNetwork. This script requires "
            "a product/representation critic checkpoint, not a monolithic critic checkpoint."
        ) from exc
    critic.eval()
    return critic


def fixed_goal_for_env(env_name: str):
    return FIXED_GOAL_DICT.get(env_name)


def unwrap_env(env):
    while hasattr(env, "_environment"):
        env = env._environment
    return env


def sample_point_state_actions(
    config: ContrastiveConfig,
    rng: np.random.Generator,
    num_samples: int,
    actions_per_state: int,
    action_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    env, _ = env_utils.make_environment(
        config.env_name,
        config.start_index,
        config.end_index,
        seed=config.seed,
        fixed_start_end=fixed_goal_for_env(config.env_name),
    )
    base_env = unwrap_env(env)
    base_env._action_noise = 0.0

    walls = base_env.walls
    free_cells = np.argwhere(walls == 0)
    if free_cells.size == 0:
        raise ValueError(f"No free cells found for {config.env_name}.")

    num_states = max(1, int(np.ceil(num_samples / max(actions_per_state, 1))))
    cell_indices = rng.integers(0, len(free_cells), size=num_states)
    states = free_cells[cell_indices].astype(np.float32) + rng.uniform(0.1, 0.9, size=(num_states, 2))

    angles = np.linspace(0.0, 2.0 * np.pi, actions_per_state, endpoint=False, dtype=np.float32)
    action_template = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    action_template *= float(action_scale)

    sampled_states = np.repeat(states, actions_per_state, axis=0)[:num_samples]
    actions = np.tile(action_template, (num_states, 1))[:num_samples]
    goals = np.zeros((sampled_states.shape[0], config.resolved_goal_dim()), dtype=np.float32)
    observations = np.concatenate([sampled_states, goals], axis=1).astype(np.float32)

    effects = []
    for state, action in zip(sampled_states, actions):
        base_env.state = state.astype(float).copy()
        base_env.goal = np.zeros(2, dtype=float)
        base_env._timestep = 0
        next_obs, _, _, _ = base_env.step(action.astype(np.float32))
        effects.append(next_obs[: config.obs_dim] - state)

    close = getattr(env, "close", None)
    if close is not None:
        close()

    effects = np.asarray(effects, dtype=np.float32)
    achieved_states = sampled_states.astype(np.float32)
    return observations, actions.astype(np.float32), effects, achieved_states


def sample_rollout_state_actions(
    config: ContrastiveConfig,
    rng: np.random.Generator,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observations = []
    actions = []
    effects = []
    achieved_states = []
    fixed_goal = fixed_goal_for_env(config.env_name)
    env_seed = int(config.seed)
    env, _ = env_utils.make_environment(
        config.env_name,
        config.start_index,
        config.end_index,
        seed=env_seed,
        fixed_start_end=fixed_goal,
    )
    obs = env.reset(seed=env_seed)

    while len(observations) < num_samples:
        action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        next_obs, _, done, _ = env.step(action)

        state = np.asarray(obs[: config.obs_dim], dtype=np.float32)
        next_state = np.asarray(next_obs[: config.obs_dim], dtype=np.float32)
        achieved = obs_to_goal_2d(state[None], config.start_index, config.end_index)[0]
        next_achieved = obs_to_goal_2d(next_state[None], config.start_index, config.end_index)[0]

        observations.append(np.asarray(obs, dtype=np.float32))
        actions.append(action.reshape(-1).astype(np.float32))
        effects.append((next_achieved - achieved).astype(np.float32))
        achieved_states.append(achieved.astype(np.float32))

        obs = env.reset() if done else next_obs

    close = getattr(env, "close", None)
    if close is not None:
        close()

    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(effects, dtype=np.float32),
        np.asarray(achieved_states, dtype=np.float32),
    )


@torch.no_grad()
def encode_state_actions(
    critic: ContrastiveQNetwork,
    observations: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, observations.shape[0], batch_size):
        end = min(start + batch_size, observations.shape[0])
        obs_tensor = torch.as_tensor(observations[start:end], dtype=torch.float32, device=device)
        action_tensor = torch.as_tensor(actions[start:end], dtype=torch.float32, device=device)
        sa_repr, _ = critic._encode(obs_tensor, action_tensor, second=False)
        outputs.append(sa_repr.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def pca_2d(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    embedding = centered @ vh[:2].T
    variances = singular_values**2
    explained = variances[:2] / max(float(variances.sum()), 1e-12)
    return embedding.astype(np.float32), explained.astype(np.float32)


def pairwise_squared_distances(values: np.ndarray) -> np.ndarray:
    norms = np.sum(values * values, axis=1, keepdims=True)
    dists = norms + norms.T - 2.0 * values @ values.T
    return np.maximum(dists, 0.0)


def vector_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(denom, 1e-8)


def nearest_neighbor_metrics(
    representations: np.ndarray,
    actions: np.ndarray,
    effects: np.ndarray,
    nearest_k: int,
    rng: np.random.Generator,
) -> dict:
    num_samples = representations.shape[0]
    k = min(nearest_k, num_samples - 1)
    if k <= 0:
        raise ValueError("Need at least two samples for nearest-neighbor analysis.")

    repr_dists = pairwise_squared_distances(representations)
    np.fill_diagonal(repr_dists, np.inf)
    nn_indices = np.argpartition(repr_dists, kth=k - 1, axis=1)[:, :k]
    query_indices = np.repeat(np.arange(num_samples), k)
    flat_nn = nn_indices.reshape(-1)

    random_indices = rng.integers(0, num_samples - 1, size=query_indices.shape[0])
    random_indices = random_indices + (random_indices >= query_indices)

    nn_effect_cos = vector_cosine(effects[query_indices], effects[flat_nn])
    random_effect_cos = vector_cosine(effects[query_indices], effects[random_indices])
    nn_action_cos = vector_cosine(actions[query_indices], actions[flat_nn])
    random_action_cos = vector_cosine(actions[query_indices], actions[random_indices])
    nn_effect_dist = np.linalg.norm(effects[query_indices] - effects[flat_nn], axis=1)
    random_effect_dist = np.linalg.norm(effects[query_indices] - effects[random_indices], axis=1)

    return {
        "nearest_indices": nn_indices,
        "nn_effect_cosine": nn_effect_cos,
        "random_effect_cosine": random_effect_cos,
        "nn_action_cosine": nn_action_cos,
        "random_action_cosine": random_action_cos,
        "nn_effect_distance": nn_effect_dist,
        "random_effect_distance": random_effect_dist,
        "mean_nn_effect_cosine": float(np.mean(nn_effect_cos)),
        "mean_random_effect_cosine": float(np.mean(random_effect_cos)),
        "mean_nn_action_cosine": float(np.mean(nn_action_cos)),
        "mean_random_action_cosine": float(np.mean(random_action_cos)),
        "mean_nn_effect_distance": float(np.mean(nn_effect_dist)),
        "mean_random_effect_distance": float(np.mean(random_effect_dist)),
    }


def angle_or_zero(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[1] < 2:
        return np.zeros(vectors.shape[0], dtype=np.float32)
    return np.arctan2(vectors[:, 1], vectors[:, 0])


def save_pca_plot(
    path: Path,
    embedding: np.ndarray,
    color_values: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap: str = "twilight",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=color_values, s=8, alpha=0.75, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label=colorbar_label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_histogram(path: Path, metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bins = np.linspace(-1.0, 1.0, 61)
    ax.hist(metrics["random_effect_cosine"], bins=bins, alpha=0.55, density=True, label="random pairs")
    ax.hist(metrics["nn_effect_cosine"], bins=bins, alpha=0.65, density=True, label="repr nearest neighbors")
    ax.set_xlabel("transition-effect cosine similarity")
    ax.set_ylabel("density")
    ax.set_title("Do representation neighbors have similar effects?")
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_distance_scatter(path: Path, representations: np.ndarray, effects: np.ndarray, rng: np.random.Generator) -> None:
    num_samples = representations.shape[0]
    num_pairs = min(20_000, num_samples * 20)
    idx_a = rng.integers(0, num_samples, size=num_pairs)
    idx_b = rng.integers(0, num_samples - 1, size=num_pairs)
    idx_b = idx_b + (idx_b >= idx_a)
    repr_dist = np.linalg.norm(representations[idx_a] - representations[idx_b], axis=1)
    effect_dist = np.linalg.norm(effects[idx_a] - effects[idx_b], axis=1)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(repr_dist, effect_dist, s=5, alpha=0.25)
    ax.set_xlabel("representation distance")
    ax.set_ylabel("transition-effect distance")
    ax.set_title("Representation distance vs effect distance")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint = torch_load(checkpoint_path, device)
    config = config_from_checkpoint(checkpoint, args.env)
    seed = int(config.seed if args.seed is None else args.seed)
    set_global_seeds(seed)
    rng = np.random.default_rng(seed)
    critic = build_critic(config, checkpoint, device)

    if config.use_image_obs:
        raise NotImplementedError("This visualizer currently supports vector-observation critics only.")

    if config.env_name.startswith("point_"):
        observations, actions, effects, achieved_states = sample_point_state_actions(
            config,
            rng,
            args.num_samples,
            args.actions_per_state,
            args.action_scale,
        )
    else:
        observations, actions, effects, achieved_states = sample_rollout_state_actions(
            config,
            rng,
            int(args.rollout_steps or args.num_samples),
        )

    representations = encode_state_actions(critic, observations, actions, device, args.batch_size)
    pca_embedding, explained = pca_2d(representations)
    metrics = nearest_neighbor_metrics(representations, actions, effects, args.nearest_k, rng)

    prefix = args.prefix or checkpoint_path.stem
    action_angle = angle_or_zero(actions)
    effect_angle = angle_or_zero(effects)
    effect_speed = np.linalg.norm(effects, axis=1)

    save_pca_plot(
        output_dir / f"{prefix}_sa_repr_pca_by_action.png",
        pca_embedding,
        action_angle,
        "State-action representation PCA colored by action angle",
        "action angle",
    )
    save_pca_plot(
        output_dir / f"{prefix}_sa_repr_pca_by_effect_angle.png",
        pca_embedding,
        effect_angle,
        "State-action representation PCA colored by transition-effect angle",
        "effect angle",
    )
    save_pca_plot(
        output_dir / f"{prefix}_sa_repr_pca_by_effect_speed.png",
        pca_embedding,
        effect_speed,
        "State-action representation PCA colored by transition-effect speed",
        "effect speed",
        cmap="viridis",
    )
    save_histogram(output_dir / f"{prefix}_nn_effect_cosine_hist.png", metrics)
    save_distance_scatter(output_dir / f"{prefix}_repr_vs_effect_distance.png", representations, effects, rng)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "env_name": config.env_name,
        "seed": seed,
        "num_samples": int(representations.shape[0]),
        "repr_dim": int(representations.shape[1]),
        "pca_explained_variance_pc1": float(explained[0]),
        "pca_explained_variance_pc2": float(explained[1]),
        "nearest_k": int(min(args.nearest_k, representations.shape[0] - 1)),
        "mean_effect_speed": float(np.mean(effect_speed)),
        "std_effect_speed": float(np.std(effect_speed)),
        **{
            key: value
            for key, value in metrics.items()
            if key.startswith("mean_")
        },
    }
    with open(output_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    np.savez_compressed(
        output_dir / f"{prefix}_data.npz",
        representations=representations,
        pca_embedding=pca_embedding,
        observations=observations,
        actions=actions,
        effects=effects,
        achieved_states=achieved_states,
        nearest_indices=metrics["nearest_indices"],
        nn_effect_cosine=metrics["nn_effect_cosine"],
        random_effect_cosine=metrics["random_effect_cosine"],
        nn_action_cosine=metrics["nn_action_cosine"],
        random_action_cosine=metrics["random_action_cosine"],
    )

    print(f"Saved state-action representation geometry visualizations to {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
