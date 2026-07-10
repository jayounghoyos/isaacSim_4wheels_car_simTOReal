"""DriveToCube: a Gymnasium env where a differential-drive car learns to reach a target cube.

Observation (8-dim, egocentric — mirrors the Leatherback design):
  [ distance_to_cube,
    cos(heading_error), sin(heading_error),   # which way to turn (wrap-safe)
    forward_vel_body, lateral_vel_body,       # body-frame velocity
    yaw_rate,
    last_left_action, last_right_action ]
Action (2-dim, continuous in [-1, 1]): [left_wheel, right_wheel] -> scaled to wheel velocity.
Reward: progress toward cube (dense) + small heading bonus + big reach bonus - tiny control cost.
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

_XML = os.path.join(os.path.dirname(__file__), "car.xml")


def _yaw_from_quat(q):
    # q = [w, x, y, z]
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class DriveToCubeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, max_steps=300, frame_skip=10):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(_XML)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.wheel_vel_scale = 25.0          # action [-1,1] -> +/-25 rad/s
        self.reach_tol = 0.45                 # within this distance = success
        self.render_mode = render_mode
        self._renderer = None
        self._steps = 0

        # 8-dim observation, 2-dim action
        high = np.array([np.inf] * 8, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._last_action = np.zeros(2, dtype=np.float32)

    # ---- helpers ----
    def _car_xy(self):
        return self.data.qpos[0:2].copy()

    def _car_yaw(self):
        return _yaw_from_quat(self.data.qpos[3:7])

    def _target_xy(self):
        return np.array(self.data.mocap_pos[0][:2])

    def _obs(self):
        car_xy = self._car_xy()
        yaw = self._car_yaw()
        to_target = self._target_xy() - car_xy
        dist = float(np.linalg.norm(to_target))
        target_dir = np.arctan2(to_target[1], to_target[0])
        heading_err = np.arctan2(np.sin(target_dir - yaw), np.cos(target_dir - yaw))

        # world linear velocity -> body frame
        vx_w, vy_w = self.data.qvel[0], self.data.qvel[1]
        c, s = np.cos(-yaw), np.sin(-yaw)
        vx_b = c * vx_w - s * vy_w
        vy_b = s * vx_w + c * vy_w
        yaw_rate = self.data.qvel[5]

        return np.array([
            dist,
            np.cos(heading_err), np.sin(heading_err),
            vx_b, vy_b, yaw_rate,
            self._last_action[0], self._last_action[1],
        ], dtype=np.float32)

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # car at origin, random heading
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[0:3] = [0.0, 0.0, 0.06]
        self.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        # random target: 2-4 m away, any direction
        r = self.np_random.uniform(2.0, 4.0)
        ang = self.np_random.uniform(-np.pi, np.pi)
        self.data.mocap_pos[0] = [r * np.cos(ang), r * np.sin(ang), 0.15]
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._last_action[:] = 0.0
        self._prev_dist = float(np.linalg.norm(self._target_xy() - self._car_xy()))
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.data.ctrl[0] = action[0] * self.wheel_vel_scale
        self.data.ctrl[1] = action[1] * self.wheel_vel_scale
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        dist = float(np.linalg.norm(self._target_xy() - self._car_xy()))
        obs = self._obs()
        heading_err = np.arctan2(obs[2], obs[1])  # sin, cos -> angle

        progress = self._prev_dist - dist
        heading_rew = np.exp(-abs(heading_err) / 0.5)
        ctrl_cost = 0.001 * float(np.sum(action ** 2))
        reached = dist < self.reach_tol
        reward = 5.0 * progress + 0.02 * heading_rew - ctrl_cost + (10.0 if reached else 0.0)

        self._prev_dist = dist
        self._last_action = action
        terminated = bool(reached)
        truncated = self._steps >= self.max_steps
        return obs, float(reward), terminated, truncated, {"dist": dist, "is_success": reached}

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=720, width=1280)
        cam = mujoco.MjvCamera()
        # follow the car from behind/above
        cx, cy = self._car_xy()
        cam.lookat[:] = [cx, cy, 0.0]
        cam.distance = 4.0
        cam.azimuth = np.degrees(self._car_yaw()) + 180.0
        cam.elevation = -30
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
