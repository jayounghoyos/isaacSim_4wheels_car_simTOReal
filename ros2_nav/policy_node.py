#!/usr/bin/env python3
"""ROS 2 policy node — runs the trained nav policy from sensor topics. HARDWARE-AGNOSTIC.

Subscribes:
  /scan              (sensor_msgs/LaserScan)   2D LIDAR
  /odom              (nav_msgs/Odometry)       pose + body-frame twist  <-- the sim-to-real "gap" topic
  /camera/image_raw  (sensor_msgs/Image, rgb8) forward camera
  /goal_pose         (geometry_msgs/PoseStamped) commanded waypoint (map/odom frame)
Publishes:
  /cmd_vel           (geometry_msgs/Twist)     linear.x, angular.z for a diff-drive base

This SAME node runs in simulation (topics from mujoco_bridge.py) and on the real robot
(topics from the LIDAR driver, wheel-encoder/IMU odometry, and camera driver). Only the
producers of the topics change — the policy does not.

Run (ROS 2 sourced + the py3.12 venv active):
  python ros2_nav/policy_node.py --ros-args -p onnx:=models/policy_nav.onnx
"""
import os
import numpy as np
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped

N_RAYS = 24
LIDAR_MAX = 8.0
IMG = 64
REACH_TOL = 1.1
# robot kinematics (from robot.xml): wheel radius ~0.184 m, track width ~ 2*0.895 m between wheel rows.
WHEEL_RADIUS = 0.184
WHEEL_BASE = 1.58
WHEEL_VEL_SCALE = 10.0     # action[-1,1] -> rad/s (matches RobotNavEnv)


def yaw_from_quat(x, y, z, w):
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class PolicyNode(Node):
    def __init__(self):
        super().__init__("nav_policy")
        default_onnx = os.path.join(os.path.dirname(__file__), "..", "models", "policy_nav.onnx")
        onnx_path = self.declare_parameter("onnx", default_onnx).value
        self.rate_hz = float(self.declare_parameter("rate", 20.0).value)
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.get_logger().info(f"loaded ONNX policy: {onnx_path}")

        # latest sensor state
        self.scan = None
        self.pose = None          # (x, y, yaw)
        self.twist = None         # (vx_body, vy_body, yaw_rate)
        self.img = np.zeros((IMG, IMG, 3), np.uint8)
        self.goal = None          # (x, y)
        self.last_action = np.zeros(2, np.float32)

        self.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/image_raw", self._on_img, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        # EVENT-DRIVEN: act on each /odom (the time-critical pose+velocity). One action per odom
        # update -> exactly one-step latency, matching the training control loop. On the real robot
        # /odom arrives at the encoder/IMU rate; scan+image are used as latest-available.
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)

    # ---- subscribers ----
    def _on_scan(self, m: LaserScan):
        self.scan = m

    def _on_odom(self, m: Odometry):
        p = m.pose.pose
        yaw = yaw_from_quat(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
        self.pose = (p.position.x, p.position.y, yaw)
        t = m.twist.twist
        self.twist = (t.linear.x, t.linear.y, t.angular.z)   # assumed body-frame (diff-drive convention)
        self._tick()                                          # act immediately on fresh odom

    def _on_img(self, m: Image):
        # expect rgb8; reshape and nearest-resize to IMG x IMG
        buf = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        yi = (np.linspace(0, m.height - 1, IMG)).astype(int)
        xi = (np.linspace(0, m.width - 1, IMG)).astype(int)
        self.img = buf[yi][:, xi].copy()

    def _on_goal(self, m: PoseStamped):
        g = (m.pose.position.x, m.pose.position.y)
        if self.goal is None or abs(g[0] - self.goal[0]) > 1e-3 or abs(g[1] - self.goal[1]) > 1e-3:
            self.get_logger().info(f"new goal: ({g[0]:.2f}, {g[1]:.2f})")   # log only on change
        self.goal = g

    # ---- LIDAR resample: LaserScan -> our 24 rays (ray 0 = robot forward, full circle) ----
    def _lidar_vec(self):
        ranges = np.ones(N_RAYS, np.float32)
        if self.scan is None:
            return ranges
        s = self.scan
        scan_r = np.asarray(s.ranges, np.float32)
        n = len(scan_r)
        if n == 0:
            return ranges
        want = np.linspace(0.0, 2 * np.pi, N_RAYS, endpoint=False)   # relative angles, forward=0
        for i, a in enumerate(want):
            # nearest LaserScan index for this angle (scan angles: angle_min + k*angle_increment)
            k = int(round((a - s.angle_min) / s.angle_increment)) % n if s.angle_increment else 0
            d = scan_r[k]
            if np.isfinite(d) and d > 0:
                ranges[i] = min(d, LIDAR_MAX) / LIDAR_MAX
        return ranges

    def _build_obs(self):
        if self.pose is None or self.goal is None:
            return None
        x, y, yaw = self.pose
        gx, gy = self.goal
        dx, dy = gx - x, gy - y
        dist = float(np.hypot(dx, dy))
        target_dir = np.arctan2(dy, dx)
        head_err = np.arctan2(np.sin(target_dir - yaw), np.cos(target_dir - yaw))
        vx_b, vy_b, yaw_rate = self.twist if self.twist else (0.0, 0.0, 0.0)
        vec = np.concatenate([
            np.array([dist, np.cos(head_err), np.sin(head_err), vx_b, vy_b, yaw_rate,
                      self.last_action[0], self.last_action[1]], np.float32),
            self._lidar_vec()]).astype(np.float32)
        img = self.img.transpose(2, 0, 1)[None].astype(np.float32)   # 1,C,H,W
        return img, vec[None], dist

    def _tick(self):
        built = self._build_obs()
        if built is None:
            return                       # waiting for odom + goal
        img, vec, dist = built
        if dist < REACH_TOL:             # arrived -> stop
            self.cmd_pub.publish(Twist())
            return
        action = self.sess.run(["action"], {"img": img, "vec": vec})[0][0]
        action = np.clip(action, -1.0, 1.0)
        self.last_action = action.astype(np.float32)
        # policy action = [left_pair, right_pair] normalized -> wheel rad/s -> diff-drive Twist
        vL = action[0] * WHEEL_VEL_SCALE
        vR = action[1] * WHEEL_VEL_SCALE
        v = WHEEL_RADIUS * (vL + vR) / 2.0
        w = WHEEL_RADIUS * (vR - vL) / WHEEL_BASE
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
