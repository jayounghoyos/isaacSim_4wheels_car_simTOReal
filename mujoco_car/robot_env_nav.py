"""Pose-based navigation (A->B) with LIDAR + onboard CAMERA senses — for YOUR Onshape robot.

Autonomous-driving style: the robot is COMMANDED to a goal pose (given in the observation, like a GPS
waypoint) and must drive there from anywhere in a big arena while avoiding obstacles.
  - The commanded goal (relative vector) is a legitimate destination input — NOT privileged obstacle info.
  - LIDAR (24 rays) = primary obstacle sense.
  - Camera (64x64 RGB from robot_cam) = auxiliary sense that helps when the goal/obstacles are in view.

Observation = Dict:
  "vec": Box(32) = [dist, cos(head_err), sin(head_err), vx_b, vy_b, yaw_rate, last_L, last_R] + 24 LIDAR
  "img": Box(0,255,(64,64,3), uint8) = forward camera
Action = Box(2) = [left_pair, right_pair] wheel velocity (mirrored-axle sign handled here).

Task setup: robot spawns at a RANDOM pose anywhere, goal at a RANDOM pose anywhere (>= min journey).
Obstacles (up to 4): 60% placed near the goal (guard the approach), 40% random across the arena.
No moving obstacles this round.
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from mujoco_car.robot_env import _yaw_from_quat

_XML = os.path.join(os.path.dirname(__file__), "..", "robot", "train_scene_nav.xml")
_REST_Z = 0.20
_LIDAR_GROUP = 5
_OBS_HALF = 0.3
_OBS_Z = 0.5
_ARENA = 13.0             # keep robot/goal/obstacles within +/- this (walls inner face ~15.0)
_PARK = np.array([200.0, 200.0, 0.5])
_IMG = 64                 # camera resolution (square)


class RobotNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, max_steps=400, frame_skip=25,
                 n_obstacles=1, n_rays=24, lidar_max=8.0, goal_lo=2.5, goal_hi=5.0):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(_XML))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.wheel_vel_scale = 10.0    # ~1.8 m/s top; reward penalties (not speed) do the smoothing
        self.reach_tol = 1.1
        self.render_mode = render_mode
        self.n_obstacles = int(n_obstacles)
        self.n_rays = int(n_rays)
        self.lidar_max = float(lidar_max)
        self.goal_lo = float(goal_lo)      # distance curriculum: goal spawned in [goal_lo, goal_hi] m
        self.goal_hi = float(goal_hi)

        self._fj = self.model.joint("part_1_freejoint").qposadr[0]
        self._fjd = self.model.joint("part_1_freejoint").dofadr[0]
        self._left = [self.model.actuator(n).id for n in ("wheel_fl", "wheel_rl")]
        self._right = [self.model.actuator(n).id for n in ("wheel_fr", "wheel_rr")]
        self._robot_body = self.model.body("part_1").id
        self._lidar_site = self.model.site("lidar").id
        self._cam_id = self.model.camera("robot_cam").id
        self._obs_mocap = [self.model.body_mocapid[self.model.body(n).id]
                           for n in ("obs1", "obs2", "obs3", "obs4")]
        self._target_mocap = self.model.body_mocapid[self.model.body("target").id]
        self._robot_geoms = {g for g in range(self.model.ngeom) if self.model.geom_group[g] == 3}
        self._solid_geoms = {g for g in range(self.model.ngeom) if self.model.geom_group[g] == _LIDAR_GROUP}

        self._ray_base = np.linspace(0.0, 2 * np.pi, self.n_rays, endpoint=False)
        self._geomgroup = np.zeros(6, dtype=np.uint8); self._geomgroup[_LIDAR_GROUP] = 1
        self._gid = np.zeros(1, dtype=np.int32)

        # camera renderer (lazy — created in the worker process on first use)
        self._cam_renderer = None
        self._cam_opt = mujoco.MjvOption(); self._cam_opt.geomgroup[_LIDAR_GROUP] = 1
        self._third_renderer = None

        self.observation_space = spaces.Dict({
            "vec": spaces.Box(-np.inf, np.inf, (8 + self.n_rays,), np.float32),
            "img": spaces.Box(0, 255, (_IMG, _IMG, 3), np.uint8),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._last_action = np.zeros(2, dtype=np.float32)

    def set_stage(self, n_obstacles, goal_lo=None, goal_hi=None):
        self.n_obstacles = int(n_obstacles)
        if goal_lo is not None:
            self.goal_lo = float(goal_lo)
        if goal_hi is not None:
            self.goal_hi = float(goal_hi)

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
        ranges = np.ones(self.n_rays, dtype=np.float32)
        for i, base in enumerate(self._ray_base):
            ang = yaw + base
            vec = np.array([np.cos(ang), np.sin(ang), 0.0])
            dist = mujoco.mj_ray(self.model, self.data, origin, vec,
                                 self._geomgroup, 1, self._robot_body, self._gid)
            if dist >= 0.0:
                ranges[i] = min(dist, self.lidar_max) / self.lidar_max
        self._last_min_range = float(ranges.min())
        return ranges

    def _camera(self):
        if self._cam_renderer is None:
            self._cam_renderer = mujoco.Renderer(self.model, _IMG, _IMG)
        self._cam_renderer.update_scene(self.data, camera=self._cam_id, scene_option=self._cam_opt)
        return self._cam_renderer.render().astype(np.uint8)

    def _obs(self):
        car_xy = self._car_xy(); yaw = self._car_yaw()
        to_t = self._target_xy() - car_xy
        dist = float(np.linalg.norm(to_t))
        target_dir = np.arctan2(to_t[1], to_t[0])
        heading_err = np.arctan2(np.sin(target_dir - yaw), np.cos(target_dir - yaw))
        vx_w, vy_w = self.data.qvel[self._fjd], self.data.qvel[self._fjd + 1]
        c, s = np.cos(-yaw), np.sin(-yaw)
        vx_b, vy_b = c * vx_w - s * vy_w, s * vx_w + c * vy_w
        yaw_rate = self.data.qvel[self._fjd + 5]
        vec = np.concatenate([
            np.array([dist, np.cos(heading_err), np.sin(heading_err), vx_b, vy_b, yaw_rate,
                      self._last_action[0], self._last_action[1]], dtype=np.float32),
            self._lidar()]).astype(np.float32)
        return {"vec": vec, "img": self._camera()}

    # ---- obstacle placement: 60% near goal, 40% random ----
    def _place_obstacles(self, robot_xy, goal_xy):
        d_robot_min, d_goal_min, d_pair_min = 2.2, 1.6, 2.8
        placed = []
        for _ in range(self.n_obstacles):
            for _try in range(300):
                if self.np_random.random() < 0.72:                      # ~60% ACCEPTED near goal
                    # (sampled higher than 60% because near-goal candidates are rejected more often)
                    r = self.np_random.uniform(1.8, 3.5)
                    a = self.np_random.uniform(-np.pi, np.pi)
                    cand = goal_xy + r * np.array([np.cos(a), np.sin(a)])
                else:                                                    # 40% random in the arena
                    cand = self.np_random.uniform(-_ARENA, _ARENA, size=2)
                if np.max(np.abs(cand)) > _ARENA:
                    continue
                if np.linalg.norm(cand - robot_xy) < d_robot_min:
                    continue
                if np.linalg.norm(cand - goal_xy) < d_goal_min:
                    continue
                if any(np.linalg.norm(cand - q) < d_pair_min for q in placed):
                    continue
                placed.append(cand); break
        return placed

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # random robot pose anywhere
        robot_xy = self.np_random.uniform(-_ARENA, _ARENA, size=2)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[self._fj:self._fj + 3] = [robot_xy[0], robot_xy[1], _REST_Z]
        self.data.qpos[self._fj + 3:self._fj + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        # goal at a distance drawn from the curriculum range [goal_lo, goal_hi], random direction,
        # kept inside the arena (sample distance+angle directly so near-goal ranges stay efficient)
        goal_xy = robot_xy.copy()
        for _ in range(200):
            dgoal = self.np_random.uniform(self.goal_lo, self.goal_hi)
            ang = self.np_random.uniform(-np.pi, np.pi)
            cand = robot_xy + dgoal * np.array([np.cos(ang), np.sin(ang)])
            if np.max(np.abs(cand)) <= _ARENA:
                goal_xy = cand
                break
        self.data.mocap_pos[self._target_mocap] = [goal_xy[0], goal_xy[1], 0.02]
        # obstacles
        placed = self._place_obstacles(robot_xy, goal_xy)
        for k, mid in enumerate(self._obs_mocap):
            self.data.mocap_pos[mid] = ([placed[k][0], placed[k][1], _OBS_Z]
                                        if k < len(placed) else _PARK)
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._last_action[:] = 0.0
        self._prev_dist = float(np.linalg.norm(self._target_xy() - self._car_xy()))
        return self._obs(), {}

    def _collided(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in self._robot_geoms and c.geom2 in self._solid_geoms) or \
               (c.geom2 in self._robot_geoms and c.geom1 in self._solid_geoms):
                return True
        return False

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        for a in self._left:
            self.data.ctrl[a] = -action[0] * self.wheel_vel_scale   # mirrored axle
        for a in self._right:
            self.data.ctrl[a] = action[1] * self.wheel_vel_scale
        collided = False
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self._collided():
                collided = True
        self._steps += 1

        dist = float(np.linalg.norm(self._target_xy() - self._car_xy()))
        obs = self._obs()
        heading_err = np.arctan2(obs["vec"][2], obs["vec"][1])
        progress = self._prev_dist - dist
        heading_rew = np.exp(-abs(heading_err) / 0.5)
        ctrl_cost = 0.001 * float(np.sum(action ** 2))
        prox_pen = 5.0 * max(0.0, 0.22 - getattr(self, "_last_min_range", 1.0))
        # SMOOTH-MOTION shaping: discourage the "hop + slide sideways" cheat, reward car-like driving
        vy_b = float(obs["vec"][4])                                  # body-frame lateral velocity
        vz = float(self.data.qvel[self._fjd + 2])                    # vertical velocity (hopping)
        lateral_pen = 1.2 * abs(vy_b)                                # move ALONG your heading, don't slide (strong)
        bounce_pen = 0.4 * abs(vz)                                   # don't hop
        jerk_pen = 0.05 * float(np.sum((action - self._last_action) ** 2))  # smooth, non-twitchy control
        reached = dist < self.reach_tol
        reward = (2.0 * progress + 0.02 * heading_rew - ctrl_cost - prox_pen
                  - lateral_pen - bounce_pen - jerk_pen + (15.0 if reached else 0.0))
        if collided:
            reward -= 2.5

        self._prev_dist = dist
        self._last_action = action
        flipped = not (0.05 < self.data.qpos[self._fj + 2] < 1.5)
        terminated = bool(reached)
        truncated = bool(self._steps >= self.max_steps or flipped)
        return obs, float(reward), terminated, truncated, {"dist": dist, "is_success": reached,
                                                            "collided": collided}

    def render(self):
        # third-person chase view (for filmstrips / recordings)
        if self._third_renderer is None:
            self._third_renderer = mujoco.Renderer(self.model, height=720, width=1280)
        cam = mujoco.MjvCamera()
        cx, cy = self._car_xy()
        cam.lookat[:] = [cx, cy, 0.2]; cam.distance = 10.0
        cam.azimuth = np.degrees(self._car_yaw()) + 180.0; cam.elevation = -32
        self._third_renderer.update_scene(self.data, camera=cam, scene_option=self._cam_opt)
        return self._third_renderer.render()

    def close(self):
        for r in (self._cam_renderer, self._third_renderer):
            if r is not None:
                r.close()
        self._cam_renderer = self._third_renderer = None
