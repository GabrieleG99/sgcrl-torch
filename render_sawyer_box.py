#!/usr/bin/env python3
"""Render rollouts from the final Sawyer box SGCRL checkpoint."""

from __future__ import annotations

import argparse
import os

# Set this before importing Gymnasium/Metaworld/MuJoCo modules. If it is set
# after those imports, headless rgb_array rendering can fail to create a GL context.
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

import sgcrl_torch.env_utils as env_utils
from sgcrl_torch.networks import PolicyNetwork
from sgcrl_torch.runner import FIXED_GOAL_DICT
from sgcrl_torch.utils import DistanceObserver, SuccessObserver, resolve_device, set_global_seeds


DEFAULT_CHECKPOINT = "logs/contrastive_cpc_sawyer_bin_42/checkpoints/learner_140000.pt"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the Sawyer box final checkpoint and render deterministic trials."
    )
    env_name = "_".join(DEFAULT_CHECKPOINT.split("/")[1].split("_")[-3:-1])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", default=f"renders/{env_name}_renders")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None, help="Defaults to the checkpoint seed.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample_goals", action="store_true", help="Sample goals instead of using the fixed eval goal.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using policy means.")
    parser.add_argument("--display", action="store_true", help="Also call env.render() each step for interactive viewers.")
    parser.add_argument("--no_video", action="store_true", help="Run trials without writing videos.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera_name", default="corner2")
    parser.add_argument(
        "--render_backend",
        choices=("auto", "wrapper", "sim", "mujoco_renderer"),
        default="auto",
        help="Renderer to use for saved frames. auto prefers camera-aware renderers when camera_name is set.",
    )
    parser.add_argument("--list_cameras", action="store_true", help="Print available MuJoCo camera names and exit.")
    parser.add_argument("--format", choices=("gif", "mp4"), default="gif")
    return parser.parse_args()


def torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_policy(config, checkpoint: dict, device: torch.device) -> PolicyNetwork:
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


def unwrap_env(env):
    unwrapped = env
    while hasattr(unwrapped, "_environment"):
        unwrapped = unwrapped._environment
    return unwrapped


def get_camera_names(unwrapped) -> list[str]:
    model = getattr(unwrapped, "model", None)
    sim = getattr(unwrapped, "sim", None)
    if model is None and sim is not None:
        model = getattr(sim, "model", None)
    if model is None:
        return []

    names = getattr(model, "camera_names", None)
    if names is not None:
        return [name.decode() if isinstance(name, bytes) else str(name) for name in names]

    name_bytes = getattr(model, "names", None)
    camera_name_addresses = getattr(model, "name_camadr", None)
    if name_bytes is not None and camera_name_addresses is not None:
        raw_names = bytes(name_bytes)
        camera_names = []
        for address in camera_name_addresses:
            end = raw_names.find(b"\0", int(address))
            if end >= 0:
                camera_names.append(raw_names[int(address) : end].decode())
        return [name for name in camera_names if name]

    ncam = int(getattr(model, "ncam", 0))
    if ncam and hasattr(model, "id2name"):
        camera_names = []
        for camera_id in range(ncam):
            try:
                camera_names.append(model.id2name(camera_id, "camera"))
            except (AttributeError, TypeError, ValueError):
                continue
        return [name for name in camera_names if name]
    return []


def camera_id_from_name(unwrapped, camera_name: Optional[str]) -> Optional[int]:
    if camera_name is None:
        return None

    camera_names = get_camera_names(unwrapped)
    if camera_names:
        try:
            return camera_names.index(camera_name)
        except ValueError:
            return None

    model = getattr(unwrapped, "model", None)
    sim = getattr(unwrapped, "sim", None)
    if model is None and sim is not None:
        model = getattr(sim, "model", None)
    if model is not None and hasattr(model, "camera_name2id"):
        try:
            return int(model.camera_name2id(camera_name))
        except (AttributeError, TypeError, ValueError):
            return None
    return None


def validate_camera_name(unwrapped, camera_name: Optional[str]) -> None:
    if camera_name is None:
        return
    camera_names = get_camera_names(unwrapped)
    if camera_names and camera_name not in camera_names:
        available = ", ".join(camera_names)
        raise ValueError(f"Camera {camera_name!r} was not found. Available cameras: {available}")


def normalize_frame(frame, flip_vertical: bool = False) -> Optional[np.ndarray]:
    if frame is None:
        return None
    frame = np.asarray(frame)
    if frame.size == 0 or frame.ndim < 2:
        return None
    if flip_vertical:
        frame = frame[::-1]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and frame.max(initial=0.0) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def render_with_wrapper(env, width: int, height: int, camera_name: Optional[str]) -> Optional[np.ndarray]:
    resolution = (width, height)
    attempts = (
        lambda: env.render(camera_name=camera_name, resolution=resolution),
        lambda: env.render(offscreen=True, camera_name=camera_name, resolution=resolution),
        lambda: env.render(mode="rgb_array", camera_name=camera_name, resolution=resolution),
        lambda: env.render(),
    )
    for attempt in attempts:
        try:
            frame = attempt()
        except Exception:
            continue
        frame = normalize_frame(frame)
        if frame is not None:
            return frame
    return None


def render_with_sim(unwrapped, width: int, height: int, camera_name: Optional[str]) -> Optional[np.ndarray]:
    sim = getattr(unwrapped, "sim", None)
    if sim is None or not hasattr(sim, "render"):
        return None

    attempts = (
        lambda: sim.render(width=width, height=height, camera_name=camera_name),
        lambda: sim.render(width, height, camera_name=camera_name),
        lambda: sim.render(width=width, height=height),
    )
    for attempt in attempts:
        try:
            frame = attempt()
        except Exception:
            continue
        frame = normalize_frame(frame, flip_vertical=True)
        if frame is not None:
            return frame
    return None


def render_with_mujoco_renderer(unwrapped, width: int, height: int, camera_name: Optional[str]) -> Optional[np.ndarray]:
    renderer = getattr(unwrapped, "mujoco_renderer", None)
    if renderer is None or not hasattr(renderer, "render"):
        return None

    old_camera_id = getattr(renderer, "camera_id", None)
    camera_id = camera_id_from_name(unwrapped, camera_name)
    if camera_id is not None and hasattr(renderer, "camera_id"):
        renderer.camera_id = camera_id

    try:
        attempts = (
            lambda: renderer.render("rgb_array"),
            lambda: renderer.render(render_mode="rgb_array"),
        )
        for attempt in attempts:
            try:
                frame = attempt()
            except Exception:
                continue
            frame = normalize_frame(frame, flip_vertical=True)
            if frame is not None:
                if frame.shape[0] != height or frame.shape[1] != width:
                    # Gymnasium's MujocoRenderer uses the environment's configured
                    # render size. Keep the frame instead of treating size mismatch
                    # as a render failure.
                    pass
                return frame
    finally:
        if old_camera_id is not None and hasattr(renderer, "camera_id"):
            renderer.camera_id = old_camera_id
    return None


def render_frame(
    env,
    width: int,
    height: int,
    camera_name: Optional[str],
    render_backend: str = "auto",
) -> Optional[np.ndarray]:
    unwrapped = unwrap_env(env)
    validate_camera_name(unwrapped, camera_name)

    renderers = {
        "wrapper": lambda: render_with_wrapper(env, width, height, camera_name),
        "sim": lambda: render_with_sim(unwrapped, width, height, camera_name),
        "mujoco_renderer": lambda: render_with_mujoco_renderer(unwrapped, width, height, camera_name),
    }
    if render_backend != "auto":
        return renderers[render_backend]()

    if camera_name is not None:
        backend_order = ("mujoco_renderer", "sim", "wrapper")
    else:
        backend_order = ("wrapper", "mujoco_renderer", "sim")

    for name in backend_order:
        frame = renderers[name]()
        if frame is not None:
            return frame
    return None


def render_display(env) -> None:
    try:
        env.render()
    except (AttributeError, TypeError, ValueError):
        pass


def write_video(path: Path, frames: Iterable[np.ndarray], fps: int) -> None:
    frames = [np.asarray(frame, dtype=np.uint8) for frame in frames if frame is not None]
    if not frames:
        raise RuntimeError(f"No renderable frames were captured, so {path} was not written.")
    try:
        import imageio.v3 as iio

        iio.imwrite(path, frames, fps=fps)
    except ImportError as exc:
        fallback = path.with_suffix(".npz")
        np.savez_compressed(fallback, frames=np.stack(frames, axis=0))
        raise RuntimeError(
            f"imageio is not installed, so frames were saved to {fallback} instead. "
            "Install imageio[ffmpeg] to write mp4 videos."
        ) from exc


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint = torch_load(checkpoint_path, device)
    config = checkpoint["config"]
    config.device = args.device
    seed = config.seed if args.seed is None else args.seed
    set_global_seeds(seed)

    if config.env_name != "sawyer_box":
        print(f"Checkpoint env is {config.env_name!r}; rendering that env instead of sawyer_box.")

    policy = build_policy(config, checkpoint, device)
    fixed_goal = None if args.sample_goals else FIXED_GOAL_DICT.get(config.env_name)

    metrics = []
    for episode_idx in range(args.episodes):
        episode_seed = seed + 50_000 + episode_idx
        env, _ = env_utils.make_environment(
            config.env_name,
            config.start_index,
            config.end_index,
            seed=episode_seed,
            fixed_start_end=fixed_goal,
        )
        obs = env.reset(seed=episode_seed)
        if args.list_cameras:
            camera_names = get_camera_names(unwrap_env(env))
            if camera_names:
                print("Available cameras: " + ", ".join(camera_names))
            else:
                print("No MuJoCo camera names were discoverable from this environment.")
            close = getattr(env, "close", None)
            if close is not None:
                close()
            return

        success_observer = SuccessObserver()
        distance_observer = DistanceObserver(config.obs_dim, config.start_index, config.end_index)
        success_observer.observe_first(obs)
        distance_observer.observe_first(obs)

        frames = []
        if not args.no_video:
            frames.append(render_frame(env, args.width, args.height, args.camera_name, args.render_backend))
        if args.display:
            render_display(env)

        done = False
        steps = 0
        total_reward = 0.0
        while not done:
            action = policy.act(obs, device=device, deterministic=not args.stochastic)
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            success_observer.observe(obs, action, reward, done, info)
            distance_observer.observe(obs, action, reward, done, info)
            if not args.no_video:
                frames.append(render_frame(env, args.width, args.height, args.camera_name, args.render_backend))
            if args.display:
                render_display(env)

        episode_metrics = {
            "episode": episode_idx,
            "steps": steps,
            "return": total_reward,
            **success_observer.get_metrics(),
            **distance_observer.get_metrics(),
        }
        metrics.append(episode_metrics)

        if not args.no_video:
            video_path = output_dir / f"trial_{episode_idx:03d}.{args.format}"
            write_video(video_path, frames, args.fps)
            print(f"Wrote {video_path}")
        print(
            "trial={episode} steps={steps} return={return:.1f} success={success:.0f} "
            "final_dist={final_dist:.4f} min_dist={min_dist:.4f}".format(**episode_metrics)
        )

        close = getattr(env, "close", None)
        if close is not None:
            close()

    success = np.mean([item["success"] for item in metrics])
    final_dist = np.mean([item["final_dist"] for item in metrics])
    min_dist = np.mean([item["min_dist"] for item in metrics])
    print(f"mean_success={success:.3f} mean_final_dist={final_dist:.4f} mean_min_dist={min_dist:.4f}")


if __name__ == "__main__":
    main()
