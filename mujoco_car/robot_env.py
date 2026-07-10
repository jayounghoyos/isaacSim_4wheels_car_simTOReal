"""DriveToCube for YOUR Onshape robot (robot/train_scene.xml).

Same egocentric design as the demo env, adapted to:
  - a free-floating chassis (part_1_freejoint) read for pose/velocity,
  - 4-wheel skid-steer: action[0]=left pair (fl,rl), action[1]=right pair (fr,rr),
  - the robot's real scale (~1.5 m, ~2 m/s) -> cube spawned 4-8 m away, reach tol ~1.1 m.
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

_XML = os.path.join(os.path.dirname(__file__), "..", "robot", "train_scene.xml")
_REST_Z = 0.20  # captured resting chassis height


def _yaw_from_quat(q):
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class RobotDriveToCubeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, max_steps=400, frame_skip=25):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(_XML))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.wheel_vel_scale = 12.0       # action [-1,1] -> +/-12 rad/s
        self.reach_tol = 1.1
        self.render_mode = render_mode
        self._renderer = None

        self._fj = self.model.joint("part_1_freejoint").qposadr[0]
        self._fjd = self.model.joint("part_1_freejoint").dofadr[0]
        self._left = [self.model.actuator(n).id for n in ("wheel_fl", "wheel_rl")]
        self._right = [self.model.actuator(n).id for n in ("wheel_fr", "wheel_rr")]

        high = np.array([np.inf] * 8, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._last_action = np.zeros(2, dtype=np.float32)

    def _car_xy(self):
        return self.data.qpos[self._fj:self._fj + 2].copy()

    def _car_yaw(self):
        return _yaw_from_quat(self.data.qpos[self._fj + 3:self._fj + 7])

    def _target_xy(self):
        return np.array(self.data.mocap_pos[0][:2])

    def _obs(self):
        car_xy = self._car_xy()
        yaw = self._car_yaw()
        to_t = self._target_xy() - car_xy
        dist = float(np.linalg.norm(to_t))
        heading_err = np.arctan2(np.sin(np.arctan2(to_t[1], to_t[0]) - yaw),
                                 np.cos(np.arctan2(to_t[1], to_t[0]) - yaw))
        vx_w, vy_w = self.data.qvel[self._fjd], self.data.qvel[self._fjd + 1]
        c, s = np.cos(-yaw), np.sin(-yaw)
        vx_b, vy_b = c * vx_w - s * vy_w, s * vx_w + c * vy_w
        yaw_rate = self.data.qvel[self._fjd + 5]
        return np.array([dist, np.cos(heading_err), np.sin(heading_err),
                         vx_b, vy_b, yaw_rate, self._last_action[0], self._last_action[1]],
                        dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[self._fj:self._fj + 3] = [0.0, 0.0, _REST_Z]
        self.data.qpos[self._fj + 3:self._fj + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        r = self.np_random.uniform(2.5, 5.0)
        ang = self.np_random.uniform(-np.pi, np.pi)
        self.data.mocap_pos[0] = [r * np.cos(ang), r * np.sin(ang), 0.3]
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._last_action[:] = 0.0
        self._prev_dist = float(np.linalg.norm(self._target_xy() - self._car_xy()))
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        # left wheels spin about -Y, right about +Y (mirrored axles): negate left so
        # action=[+1,+1] drives forward, [+1,-1] turns.
        for a in self._left:
            self.data.ctrl[a] = -action[0] * self.wheel_vel_scale
        for a in self._right:
            self.data.ctrl[a] = action[1] * self.wheel_vel_scale
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        dist = float(np.linalg.norm(self._target_xy() - self._car_xy()))
        obs = self._obs()
        heading_err = np.arctan2(obs[2], obs[1])
        progress = self._prev_dist - dist
        heading_rew = np.exp(-abs(heading_err) / 0.5)
        ctrl_cost = 0.001 * float(np.sum(action ** 2))
        reached = dist < self.reach_tol
        reward = 2.0 * progress + 0.02 * heading_rew - ctrl_cost + (15.0 if reached else 0.0)

        self._prev_dist = dist
        self._last_action = action
        # also fail if the robot flips (z too low/high)
        flipped = not (0.05 < self.data.qpos[self._fj + 2] < 1.5)
        terminated = bool(reached)
        truncated = bool(self._steps >= self.max_steps or flipped)
        return obs, float(reward), terminated, truncated, {"dist": dist, "is_success": reached}

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=720, width=1280)
        cam = mujoco.MjvCamera()
        cx, cy = self._car_xy()
        cam.lookat[:] = [cx, cy, 0.2]
        cam.distance = 8.0
        cam.azimuth = np.degrees(self._car_yaw()) + 180.0
        cam.elevation = -28
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
