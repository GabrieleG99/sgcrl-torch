"""Environment loading and wrappers for the PyTorch SGCRL port."""

from __future__ import annotations

import os
from typing import Sequence

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym

import numpy as np

import sgcrl_torch.point_env as point_env
from sgcrl_torch.utils import obs_to_goal_1d

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def euler2quat(euler):
    """Convert Euler angles to quaternions."""
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, f"Invalid shape euler {euler}"

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def _metaworld_envs_v3():
    try:
        from metaworld import envs as metaworld_envs
    except ImportError as exc:  # pragma: no cover - depends on local mujoco setup.
        raise ImportError(
            "Meta-World v3 is required for sawyer_* environments. Install it with "
            "`pip install metaworld` and use the current MuJoCo-backed v3 release."
        ) from exc
    required = (
        "SawyerBinPickingEnvV3",
        "SawyerBoxCloseEnvV3",
        "SawyerPegInsertionSideEnvV3",
    )
    missing = [name for name in required if not hasattr(metaworld_envs, name)]
    if missing:
        raise ImportError(
            "This PyTorch port supports only Meta-World v3 sawyer environments. "
            f"Missing v3 classes: {', '.join(missing)}"
        )
    return metaworld_envs


def _body_xpos(env, body_name):
    """Return a body position using the MuJoCo API used by Meta-World v3."""
    return env.data.body(body_name).xpos


def _sgcrl_observation_space(obs_dim):
    return gym.spaces.Box(
        low=np.full(2 * obs_dim, -np.inf),
        high=np.full(2 * obs_dim, np.inf),
        dtype=np.float32,
    )


def _make_sawyer_bin_class():
    base = _metaworld_envs_v3().SawyerBinPickingEnvV3

    class SawyerBin(base):
        def __init__(self, fixed_start_end=None):
            self._goal = np.zeros(3)
            super().__init__()
            self._partially_observable = False
            self._freeze_rand_vec = False
            self._set_task_called = True
            self._fixed_start_end = fixed_start_end
            self.reset()
            self.observation_space = _sgcrl_observation_space(7)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed, options=options)
            pos1 = _body_xpos(self, "bin_goal").copy()
            pos1 += np.random.uniform(-0.05, 0.05, 3)
            pos2 = self._get_pos_objects().copy()
            if self._fixed_start_end is not None:
                self._goal = self._fixed_start_end
            else:
                t = np.random.random()
                self._goal = t * pos1 + (1 - t) * pos2
                self._goal[2] = np.random.uniform(0.03, 0.12)
            self._target_pos = self._goal
            return self._get_sgcrl_obs()

        def step(self, action):
            super().step(action)
            obj_pos = self._get_pos_objects()
            dist = np.linalg.norm(self._goal - obj_pos)
            reward = float(dist < 0.05)
            return self._get_sgcrl_obs(), reward, False, {}

        def _get_sgcrl_obs(self):
            pos_hand = self.get_endeff_pos()
            finger_right = self._get_site_pos("rightEndEffector")
            finger_left = self._get_site_pos("leftEndEffector")
            gripper = np.linalg.norm(finger_right - finger_left)
            gripper = np.clip(gripper / 0.1, 0.0, 1.0)
            obs = np.concatenate((pos_hand, [gripper], self._get_pos_objects()))
            goal = np.concatenate([self._goal + np.array([0.0, 0.0, 0.03]), [0.4], self._goal])
            return np.concatenate([obs, goal]).astype(np.float32)

    return SawyerBin


def _make_sawyer_box_class():
    base = _metaworld_envs_v3().SawyerBoxCloseEnvV3

    class SawyerBox(base):
        def __init__(self, fixed_start_end=None):
            self._goal_pos = np.zeros(3)
            self._goal_quat = np.zeros(4)
            super().__init__()
            self._fixed_start_end = fixed_start_end
            self._set_task_called = True
            self._partially_observable = False
            self._freeze_rand_vec = False
            self.reset()
            self.observation_space = _sgcrl_observation_space(11)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed, options=options)
            pos1 = self._target_pos.copy()
            pos2 = self._get_pos_objects().copy()
            if self._fixed_start_end is not None:
                self._goal_pos = pos1
            else:
                t = np.random.random()
                self._goal_pos = t * pos1 + (1 - t) * pos2
            self._goal_quat = np.array([0.707, 0, 0, 0.707])
            self._target_pos = self._goal_pos
            return self._get_sgcrl_obs()

        def step(self, action):
            super().step(action)
            obj_pos = self._get_pos_objects()
            obj_quat = self._get_quat_objects()
            dist_pos = np.linalg.norm(self._goal_pos - obj_pos)
            dist_quat = np.linalg.norm(self._goal_quat - obj_quat)
            reward = float(dist_pos < 0.08 and dist_quat < 0.08)
            return self._get_sgcrl_obs(), reward, False, {}

        def _get_sgcrl_obs(self):
            pos_hand = self.get_endeff_pos()
            finger_right = self._get_site_pos("rightEndEffector")
            finger_left = self._get_site_pos("leftEndEffector")
            gripper = np.linalg.norm(finger_right - finger_left)
            gripper = np.clip(gripper / 0.1, 0.0, 1.0)
            obj_pos = self._get_pos_objects()
            obj_quat = self._get_quat_objects()
            obs = np.concatenate((pos_hand, [gripper], obj_pos, obj_quat))
            goal = np.concatenate(
                [self._goal_pos + np.array([0.0, 0.0, 0.03]), [0.4], self._goal_pos, self._goal_quat]
            )
            return np.concatenate([obs, goal]).astype(np.float32)

    return SawyerBox


def _make_sawyer_peg_class():
    base = _metaworld_envs_v3().SawyerPegInsertionSideEnvV3

    class SawyerPeg(base):
        def __init__(self, fixed_start_end=None):
            self._goal_pos = np.zeros(3)
            super().__init__()
            self._fixed_start_end = fixed_start_end
            self._set_task_called = True
            self._partially_observable = False
            self._freeze_rand_vec = False
            self.reset()
            self.observation_space = _sgcrl_observation_space(7)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed, options=options)
            pos1 = self._target_pos.copy()
            pos2 = self._get_site_pos("pegHead")
            if self._fixed_start_end is not None:
                self._goal_pos = pos1
            else:
                t = np.random.random()
                self._goal_pos = t * pos1 + (1 - t) * pos2
            self._target_pos = self._goal_pos
            return self._get_sgcrl_obs()

        def step(self, action):
            super().step(action)
            obj_head = self._get_site_pos("pegHead")
            scale = np.array([1.0, 2.0, 2.0])
            dist_pos = float(np.linalg.norm((obj_head - self._goal_pos) * scale))
            reward = float(dist_pos < 0.07)
            return self._get_sgcrl_obs(), reward, False, {}

        def _get_sgcrl_obs(self):
            pos_hand = self.get_endeff_pos()
            finger_right = self._get_site_pos("rightEndEffector")
            finger_left = self._get_site_pos("leftEndEffector")
            gripper = np.linalg.norm(finger_right - finger_left)
            gripper = np.clip(gripper / 0.1, 0.0, 1.0)
            obj_pos_head = self._get_site_pos("pegHead")
            obs = np.concatenate((pos_hand, [gripper], obj_pos_head))
            goal = np.concatenate([self._goal_pos + np.array([0.13, 0.0, 0.03]), [0.4], self._goal_pos])
            return np.concatenate([obs, goal]).astype(np.float32)

    return SawyerPeg


def load(env_name: str, fixed_start_end=None):
    """Load the underlying Gym-style environment and metadata."""
    kwargs = {}
    if env_name == "sawyer_bin":
        env_class = _make_sawyer_bin_class()
        max_episode_steps = 150
        kwargs["fixed_start_end"] = fixed_start_end
    elif env_name == "sawyer_box":
        env_class = _make_sawyer_box_class()
        max_episode_steps = 150
        kwargs["fixed_start_end"] = fixed_start_end
    elif env_name == "sawyer_peg":
        env_class = _make_sawyer_peg_class()
        max_episode_steps = 150
        kwargs["fixed_start_end"] = fixed_start_end
    elif env_name.startswith("point_"):
        env_class = point_env.PointEnv
        kwargs["walls"] = env_name.split("_")[-1]
        kwargs["fixed_start_end"] = fixed_start_end
        max_episode_steps = 100 if "11x11" in env_name else 50
    else:
        raise NotImplementedError(f"Unsupported environment: {env_name}")

    gym_env = env_class(**kwargs)
    obs_dim = gym_env.observation_space.shape[0] // 2
    return gym_env, obs_dim, max_episode_steps


def _reset_env(env, seed=None):
    try:
        result = env.reset(seed=seed)
    except TypeError:
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def _step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = result
    return obs, reward, bool(done), info


class StepLimitWrapper:
    """Small Gym-style equivalent of Acme's StepLimitWrapper."""

    def __init__(self, environment, step_limit: int):
        self._environment = environment
        self._step_limit = step_limit
        self._elapsed_steps = 0
        self.action_space = environment.action_space
        self.observation_space = environment.observation_space

    def reset(self, seed=None):
        self._elapsed_steps = 0
        return _reset_env(self._environment, seed=seed)

    def step(self, action):
        obs, reward, done, info = _step_env(self._environment, action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._step_limit:
            done = True
            info = dict(info)
            info["TimeLimit.truncated"] = True
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self._environment, name)


class ObservationFilterWrapper:
    """Expose the full state and only the selected goal coordinates."""

    def __init__(self, environment, idx: Sequence[int]):
        self._environment = environment
        self._idx = np.asarray(idx, dtype=np.int64)
        low = np.asarray(environment.observation_space.low)[self._idx]
        high = np.asarray(environment.observation_space.high)[self._idx]
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = environment.action_space
        self._step_limit = getattr(environment, "_step_limit", None)

    def _convert_observation(self, observation):
        return np.asarray(observation, dtype=np.float32)[self._idx]

    def reset(self, seed=None):
        return self._convert_observation(_reset_env(self._environment, seed=seed))

    def step(self, action):
        obs, reward, done, info = _step_env(self._environment, action)
        return self._convert_observation(obs), reward, done, info

    def __getattr__(self, name):
        return getattr(self._environment, name)


def make_environment(
    env_name: str,
    start_index: int,
    end_index: int,
    seed: int,
    fixed_start_end=None,
):
    """Create the filtered Gym-style environment and return it with obs_dim."""
    np.random.seed(seed)
    gym_env, obs_dim, max_episode_steps = load(env_name, fixed_start_end)
    goal_indices = obs_dim + obs_to_goal_1d(np.arange(obs_dim), start_index, end_index)
    indices = np.concatenate([np.arange(obs_dim), goal_indices])
    env = StepLimitWrapper(gym_env, step_limit=max_episode_steps)
    env = ObservationFilterWrapper(env, indices)
    return env, obs_dim
