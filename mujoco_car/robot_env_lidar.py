"""DriveToCube WITH obstacles + 2D LIDAR, inside a walled arena — for YOUR Onshape robot.

Extends the open-floor task (mujoco_car/robot_env.py) with:
  - a walled cage + solid obstacles (robot/train_scene_lidar.xml),
  - a simulated 2D LIDAR (mj_ray) added to the observation,
  - a difficulty curriculum: n_obstacles (1..3) and an optional moving obstacle,
  - a collision penalty (walls/obstacles) so the robot learns to avoid them.

Observation (8 + n_rays):
  [ dist, cos(head_err), sin(head_err), vx_b, vy_b, yaw_rate, last_L, last_R,   # base 8 (privileged cube)
    r_0 .. r_{n_rays-1} ]                                                        # LIDAR, normalized 0..1
Action (2, continuous): [left_pair, right_pair] wheel velocity (mirrored-axle sign handled here).

LIDAR correctness (the two user concerns):
  * mj_ray is called with flg_static=1 so STATIC walls are detected (flg_static=0 = "ray through wall").
  * a geomgroup mask restricts sensing to group 5 (walls+obstacles) — floor/robot/goal are ignored.
  * mj_ray returns the NEAREST surface -> the FRONT face for a ray starting outside a convex box.
    We keep the origin outside obstacles by terminating on collision (robot never rests inside a geom).
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from mujoco_car.robot_env import _yaw_from_quat

_XML = os.path.join(os.path.dirname(__file__), "..", "robot", "train_scene_lidar.xml")
_REST_Z = 0.20
_LIDAR_GROUP = 5          # geom group that the LIDAR senses (walls + obstacles)
_OBS_HALF = 0.3           # obstacle box half-size (xy) — smaller = wider gaps for the big robot
_OBS_Z = 0.5              # obstacle center z (half-height) => bottom on floor, no levitation
_ARENA = 6.3              # keep everything within +/- this (walls inner face ~7.0)
_PARK = np.array([100.0, 100.0, 0.5])   # where unused obstacles are parked


class RobotLidarEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, max_steps=500, frame_skip=25,
                 n_obstacles=1, moving=False, n_rays=24, lidar_max=8.0):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(_XML))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.wheel_vel_scale = 12.0
        self.reach_tol = 1.1
        self.render_mode = render_mode
        self._renderer = None

        # curriculum params
        self.n_obstacles = int(n_obstacles)
        self.moving = bool(moving)
        self.n_rays = int(n_rays)
        self.lidar_max = float(lidar_max)

        # ids / indices
        self._fj = self.model.joint("part_1_freejoint").qposadr[0]
        self._fjd = self.model.joint("part_1_freejoint").dofadr[0]
        self._left = [self.model.actuator(n).id for n in ("wheel_fl", "wheel_rl")]
        self._right = [self.model.actuator(n).id for n in ("wheel_fr", "wheel_rr")]
        self._robot_body = self.model.body("part_1").id
        self._lidar_site = self.model.site("lidar").id
        self._obs_mocap = [self.model.body_mocapid[self.model.body(n).id] for n in ("obs1", "obs2", "obs3")]
        self._target_mocap = self.model.body_mocapid[self.model.body("target").id]

        # collision sets: robot collision geoms (group 3) vs LIDAR/obstacle geoms (group 5)
        self._robot_geoms = {g for g in range(self.model.ngeom)
                             if self.model.geom_group[g] == 3}
        self._solid_geoms = {g for g in range(self.model.ngeom)
                             if self.model.geom_group[g] == _LIDAR_GROUP}

        # precomputed ray base angles (ray 0 = robot forward, full circle)
        self._ray_base = np.linspace(0.0, 2 * np.pi, self.n_rays, endpoint=False)
        self._geomgroup = np.zeros(6, dtype=np.uint8)
        self._geomgroup[_LIDAR_GROUP] = 1
        self._gid = np.zeros(1, dtype=np.int32)

        obs_dim = 8 + self.n_rays
        high = np.array([np.inf] * obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._last_action = np.zeros(2, dtype=np.float32)

    # ---- curriculum ----
    def set_stage(self, n_obstacles, moving):
        self.n_obstacles = int(n_obstacles)
        self.moving = bool(moving)

    # ---- helpers ----
    def _car_xy(self):
        return self.data.qpos[self._fj:self._fj + 2].copy()

    def _car_yaw(self):
        return _yaw_from_quat(self.data.qpos[self._fj + 3:self._fj + 7])

    def _target_xy(self):
        return np.array(self.data.mocap_pos[self._target_mocap][:2])

    def _lidar(self):
        origin = np.array(self.data.site_xpos[self._lidar_site], dtype=np.float64)
        yaw = self._car_yaw()
        ranges = np.ones(self.n_rays, dtype=np.float32)   # 1.0 = clear / max range
        for i, base in enumerate(self._ray_base):
            ang = yaw + base
            vec = np.array([np.cos(ang), np.sin(ang), 0.0])
            dist = mujoco.mj_ray(self.model, self.data, origin, vec,
                                 self._geomgroup, 1, self._robot_body, self._gid)
            if dist >= 0.0:
                ranges[i] = min(dist, self.lidar_max) / self.lidar_max
        self._last_min_range = float(ranges.min())   # closest obstacle/wall (normalized 0..1)
        return ranges

    def _obs(self):
        car_xy = self._car_xy()
        yaw = self._car_yaw()
        to_t = self._target_xy() - car_xy
        dist = float(np.linalg.norm(to_t))
        target_dir = np.arctan2(to_t[1], to_t[0])
        heading_err = np.arctan2(np.sin(target_dir - yaw), np.cos(target_dir - yaw))
        vx_w, vy_w = self.data.qvel[self._fjd], self.data.qvel[self._fjd + 1]
        c, s = np.cos(-yaw), np.sin(-yaw)
        vx_b, vy_b = c * vx_w - s * vy_w, s * vx_w + c * vy_w
        yaw_rate = self.data.qvel[self._fjd + 5]
        base = np.array([dist, np.cos(heading_err), np.sin(heading_err),
                         vx_b, vy_b, yaw_rate, self._last_action[0], self._last_action[1]],
                        dtype=np.float32)
        return np.concatenate([base, self._lidar()]).astype(np.float32)

    # ---- obstacle placement (rejection sampling; concern #2) ----
    def _place_obstacles(self, robot_xy, cube_xy):
        d_robot_min = 2.0     # keep obstacles off the robot spawn
        d_cube_min = 2.0      # don't block the goal
        d_pair_min = 3.0      # gap wide enough for the ~1.9 m robot to pass between (smaller obstacles)
        placed = []
        line = cube_xy - robot_xy
        line_len = np.linalg.norm(line) + 1e-6
        u = line / line_len                      # along corridor
        perp = np.array([-u[1], u[0]])           # perpendicular
        for _ in range(self.n_obstacles):
            p = None
            for _try in range(200):
                t = self.np_random.uniform(0.3, 0.78)          # fraction along corridor
                lat = self.np_random.uniform(-2.2, 2.2)        # lateral offset
                cand = robot_xy + u * (t * line_len) + perp * lat
                if np.max(np.abs(cand)) > _ARENA:
                    continue
                if np.linalg.norm(cand - robot_xy) < d_robot_min:
                    continue
                if np.linalg.norm(cand - cube_xy) < d_cube_min:
                    continue
                if any(np.linalg.norm(cand - q) < d_pair_min for q in placed):
                    continue
                p = cand
                break
            if p is not None:
                placed.append(p)
        return placed

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[self._fj:self._fj + 3] = [0.0, 0.0, _REST_Z]
        self.data.qpos[self._fj + 3:self._fj + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

        # cube spawn range grows a little with difficulty (room for obstacles between)
        r_lo, r_hi = (3.5, 5.5) if self.n_obstacles <= 1 else (4.0, 6.0)
        r = self.np_random.uniform(r_lo, r_hi)
        ang = self.np_random.uniform(-np.pi, np.pi)
        cube_xy = np.array([r * np.cos(ang), r * np.sin(ang)])
        self.data.mocap_pos[self._target_mocap] = [cube_xy[0], cube_xy[1], 0.02]  # tower base on floor

        # place obstacles between robot (origin) and cube; park unused ones
        placed = self._place_obstacles(np.zeros(2), cube_xy)
        for k, mid in enumerate(self._obs_mocap):
            if k < len(placed):
                self.data.mocap_pos[mid] = [placed[k][0], placed[k][1], _OBS_Z]
            else:
                self.data.mocap_pos[mid] = _PARK
        # remember moving-obstacle anchor (first obstacle) for the moving stage
        self._moving_anchor = np.array(placed[0]) if (self.moving and placed) else None
        self._moving_perp = None
        if self._moving_anchor is not None:
            u = cube_xy / (np.linalg.norm(cube_xy) + 1e-6)
            self._moving_perp = np.array([-u[1], u[0]])

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._t = 0.0
        self._last_action[:] = 0.0
        self._prev_dist = float(np.linalg.norm(self._target_xy() - self._car_xy()))
        return self._obs(), {}

    def _collided(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in self._robot_geoms and g2 in self._solid_geoms) or \
               (g2 in self._robot_geoms and g1 in self._solid_geoms):
                return True
        return False

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        for a in self._left:
            self.data.ctrl[a] = -action[0] * self.wheel_vel_scale   # mirrored axle
        for a in self._right:
            self.data.ctrl[a] = action[1] * self.wheel_vel_scale

        dt = self.frame_skip * self.model.opt.timestep
        # animate moving obstacle (kept at z=_OBS_Z: no levitation)
        if self._moving_anchor is not None:
            self._t += dt
            amp, period = 1.6, 5.0
            lat = amp * np.sin(2 * np.pi * self._t / period)
            p = self._moving_anchor + self._moving_perp * lat
            p = np.clip(p, -_ARENA, _ARENA)
            self.data.mocap_pos[self._obs_mocap[0]] = [p[0], p[1], _OBS_Z]

        collided = False
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self._collided():
                collided = True
        self._steps += 1

        dist = float(np.linalg.norm(self._target_xy() - self._car_xy()))
        obs = self._obs()
        heading_err = np.arctan2(obs[2], obs[1])
        progress = self._prev_dist - dist
        heading_rew = np.exp(-abs(heading_err) / 0.5)
        ctrl_cost = 0.001 * float(np.sum(action ** 2))
        reached = dist < self.reach_tol
        # PROXIMITY penalty from the LIDAR: a smooth, increasing cost as the robot nears an
        # obstacle/wall (BEFORE contact) -> gives a gradient to steer away early and detour.
        # min_range is normalized (0..1); threshold 0.22 ~= 1.75 m at lidar_max=8 m.
        prox_pen = 5.0 * max(0.0, 0.22 - getattr(self, "_last_min_range", 1.0))
        reward = 2.0 * progress + 0.02 * heading_rew - ctrl_cost - prox_pen + (15.0 if reached else 0.0)
        if collided:
            reward -= 2.5   # penalize contact but DON'T end the episode — let it learn to back off.
                            # (kinematic obstacles already stop the robot, so lingering earns no progress;
                            #  the chassis collides before the LIDAR origin could enter a box -> no back-face)

        self._prev_dist = dist
        self._last_action = action
        flipped = not (0.05 < self.data.qpos[self._fj + 2] < 1.5)
        terminated = bool(reached)
        truncated = bool(self._steps >= self.max_steps or flipped)
        info = {"dist": dist, "is_success": reached, "collided": collided}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=720, width=1280)
            self._scene_opt = mujoco.MjvOption()
            self._scene_opt.geomgroup[_LIDAR_GROUP] = 1   # show walls + obstacles
        cam = mujoco.MjvCamera()
        cx, cy = self._car_xy()
        cam.lookat[:] = [cx, cy, 0.2]
        cam.distance = 9.0
        cam.azimuth = np.degrees(self._car_yaw()) + 180.0
        cam.elevation = -35
        self._renderer.update_scene(self.data, camera=cam, scene_option=self._scene_opt)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
