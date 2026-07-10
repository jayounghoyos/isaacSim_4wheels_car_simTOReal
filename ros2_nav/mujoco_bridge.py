#!/usr/bin/env python3
"""ROS 2 <-> MuJoCo bridge — the SIM STAND-IN for the real robot's sensors and motors.

Runs the MuJoCo nav scene and exposes the SAME ROS 2 topics the real robot will:
  publishes /scan, /odom, /camera/image_raw, /goal_pose   (its "sensors")
  subscribes /cmd_vel                                      (its "motors")
Self-contained (only mujoco + numpy) so it mirrors the minimal on-robot dependency set.

  NOTE ON /odom = the sim-to-real gap: here /odom is PERFECT (from the physics engine). On the
  real robot this topic must be produced by wheel-encoder odometry + an IMU. The policy node is
  identical either way — that is the whole point of routing through ROS 2.

Run (ROS 2 sourced + py3.12 venv active), from the repo root:
  MUJOCO_GL=egl python ros2_nav/mujoco_bridge.py
"""
import os
import math
import numpy as np
import mujoco

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped

XML = os.path.join(os.path.dirname(__file__), "..", "robot", "train_scene_nav.xml")
LIDAR_GROUP = 5
N_SCAN = 180                 # publish a realistic 2-degree LaserScan; policy node resamples to 24
IMG = 64
REST_Z = 0.20
ARENA = 13.0
OBS_Z = 0.5
WHEEL_RADIUS = 0.184
WHEEL_BASE = 1.58
CTRL_HZ = 20.0
FRAME_SKIP = 25              # 25 * dt(0.002) = 0.05s = 20 Hz, matches training


class MujocoBridge(Node):
    def __init__(self):
        super().__init__("mujoco_bridge")
        os.environ.setdefault("MUJOCO_GL", "egl")
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(XML))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(0)

        self.fj = self.model.joint("part_1_freejoint").qposadr[0]
        self.fjd = self.model.joint("part_1_freejoint").dofadr[0]
        self.left = [self.model.actuator(n).id for n in ("wheel_fl", "wheel_rl")]
        self.right = [self.model.actuator(n).id for n in ("wheel_fr", "wheel_rr")]
        self.robot_body = self.model.body("part_1").id
        self.lidar_site = self.model.site("lidar").id
        self.cam_id = self.model.camera("robot_cam").id
        self.obs_mocap = [self.model.body_mocapid[self.model.body(n).id] for n in ("obs1", "obs2", "obs3", "obs4")]
        self.target_mocap = self.model.body_mocapid[self.model.body("target").id]
        self.geomgroup = np.zeros(6, np.uint8); self.geomgroup[LIDAR_GROUP] = 1
        self.gid = np.zeros(1, np.int32)

        self.renderer = mujoco.Renderer(self.model, IMG, IMG)
        self.cam_opt = mujoco.MjvOption(); self.cam_opt.geomgroup[LIDAR_GROUP] = 1

        self.cmd = np.zeros(2)          # (v, w) from /cmd_vel
        self.n_obstacles = int(self.declare_parameter("n_obstacles", 2).value)
        self._reset()

        self.scan_pub = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.img_pub = self.create_publisher(Image, "/camera/image_raw", qos_profile_sensor_data)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)
        self.timer = self.create_timer(1.0 / CTRL_HZ, self._tick)
        self.steps = 0

    # ---- episode ----
    def _reset(self):
        mujoco.mj_resetData(self.model, self.data)
        robot = self.rng.uniform(-ARENA, ARENA, 2)
        yaw = self.rng.uniform(-math.pi, math.pi)
        self.data.qpos[self.fj:self.fj + 3] = [robot[0], robot[1], REST_Z]
        self.data.qpos[self.fj + 3:self.fj + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for _ in range(200):
            goal = self.rng.uniform(-ARENA, ARENA, 2)
            if 6.0 <= np.linalg.norm(goal - robot) <= 14.0:
                break
        self.goal = goal
        self.data.mocap_pos[self.target_mocap] = [goal[0], goal[1], 0.02]
        placed = []
        for k, mid in enumerate(self.obs_mocap):
            if k < self.n_obstacles:
                for _ in range(200):
                    if self.rng.random() < 0.6:
                        r = self.rng.uniform(1.8, 3.5); a = self.rng.uniform(-math.pi, math.pi)
                        c = goal + r * np.array([math.cos(a), math.sin(a)])
                    else:
                        c = self.rng.uniform(-ARENA, ARENA, 2)
                    if (np.max(np.abs(c)) <= ARENA and np.linalg.norm(c - robot) > 2.2
                            and np.linalg.norm(c - goal) > 1.6
                            and all(np.linalg.norm(c - q) > 2.8 for q in placed)):
                        placed.append(c); self.data.mocap_pos[mid] = [c[0], c[1], OBS_Z]; break
                else:
                    self.data.mocap_pos[mid] = [200, 200, OBS_Z]
            else:
                self.data.mocap_pos[mid] = [200, 200, OBS_Z]
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.get_logger().info(f"episode reset: goal ({goal[0]:.1f},{goal[1]:.1f}), {self.n_obstacles} obstacles")

    def _on_cmd(self, m: Twist):
        self.cmd = np.array([m.linear.x, m.angular.z])

    # ---- helpers ----
    def _yaw(self):
        q = self.data.qpos[self.fj + 3:self.fj + 7]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def _apply_cmd(self):
        v, w = self.cmd
        vL = (v - w * WHEEL_BASE / 2) / WHEEL_RADIUS      # desired forward wheel speed (rad/s)
        vR = (v + w * WHEEL_BASE / 2) / WHEEL_RADIUS
        for a in self.left:
            self.data.ctrl[a] = -vL                       # mirrored axle (left spins about -Y)
        for a in self.right:
            self.data.ctrl[a] = vR

    def _publish(self, stamp):
        # /odom (PERFECT here; from encoders+IMU on the real robot)
        x, y, yaw = self.data.qpos[self.fj], self.data.qpos[self.fj + 1], self._yaw()
        vx_w, vy_w = self.data.qvel[self.fjd], self.data.qvel[self.fjd + 1]
        c, s = math.cos(-yaw), math.sin(-yaw)
        od = Odometry(); od.header.stamp = stamp; od.header.frame_id = "odom"; od.child_frame_id = "base_link"
        od.pose.pose.position.x = float(x); od.pose.pose.position.y = float(y)
        od.pose.pose.orientation.z = float(math.sin(yaw / 2)); od.pose.pose.orientation.w = float(math.cos(yaw / 2))
        od.twist.twist.linear.x = float(c * vx_w - s * vy_w)   # body-frame
        od.twist.twist.linear.y = float(s * vx_w + c * vy_w)
        od.twist.twist.angular.z = float(self.data.qvel[self.fjd + 5])
        self.odom_pub.publish(od)

        # /scan (180 rays over full circle, robot-forward = angle 0)
        origin = np.array(self.data.site_xpos[self.lidar_site], np.float64)
        rng = []
        for i in range(N_SCAN):
            a = yaw + 2 * math.pi * i / N_SCAN
            vec = np.array([math.cos(a), math.sin(a), 0.0])
            d = mujoco.mj_ray(self.model, self.data, origin, vec, self.geomgroup, 1, self.robot_body, self.gid)
            rng.append(d if d >= 0 else float("inf"))
        sc = LaserScan(); sc.header.stamp = stamp; sc.header.frame_id = "lidar"
        sc.angle_min = 0.0; sc.angle_max = float(2 * math.pi * (N_SCAN - 1) / N_SCAN)
        sc.angle_increment = float(2 * math.pi / N_SCAN); sc.range_min = 0.0; sc.range_max = 30.0
        sc.ranges = rng
        self.scan_pub.publish(sc)

        # /camera/image_raw (rgb8)
        self.renderer.update_scene(self.data, camera=self.cam_id, scene_option=self.cam_opt)
        frame = self.renderer.render().astype(np.uint8)
        im = Image(); im.header.stamp = stamp; im.header.frame_id = "camera"
        im.height, im.width, im.encoding = IMG, IMG, "rgb8"; im.step = IMG * 3
        im.data = frame.tobytes()
        self.img_pub.publish(im)

        # /goal_pose
        gp = PoseStamped(); gp.header.stamp = stamp; gp.header.frame_id = "odom"
        gp.pose.position.x = float(self.goal[0]); gp.pose.position.y = float(self.goal[1])
        self.goal_pub.publish(gp)

    def _tick(self):
        self._apply_cmd()
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        self._publish(self.get_clock().now().to_msg())
        dist = float(np.linalg.norm(self.goal - self.data.qpos[self.fj:self.fj + 2]))
        if dist < 1.1:
            self.get_logger().info(f"REACHED goal (dist {dist:.2f}) — new episode")
            self._reset()
        elif self.steps >= 500:
            self.get_logger().info(f"timeout (dist {dist:.2f}) — new episode")
            self._reset()


def main():
    rclpy.init()
    node = MujocoBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
