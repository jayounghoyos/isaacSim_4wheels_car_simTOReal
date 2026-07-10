#!/usr/bin/env python3
"""Sim-to-real MIRROR streamer (runs on the PC).

Runs the MuJoCo sim + trained policy in a live viewer, and streams each [left, right] action over UDP
to the Jetson Nano's mirror_server, so the real wheels mirror the sim robot in real time.

Run on the PC (needs the display for the viewer):
  PYTHONPATH=$PWD python jetson/mirror_stream.py --host 192.168.1.12 [--stage 2] [--no-view]

The action sent is the policy's raw [left, right] in [-1, 1] — exactly what motor_driver expects.
(If the real robot drives backwards or a side is reversed, swap that motor pair's wires on the L298N.)
"""
import os
import sys
import time
import socket
import struct
import numpy as np
import mujoco
import onnxruntime as ort
from mujoco_car.robot_env_nav import RobotNavEnv, _LIDAR_GROUP

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(HERE, "..", "mujoco_car", "policy_nav.onnx")
PKT = struct.Struct("!ff")

host = sys.argv[sys.argv.index("--host") + 1] if "--host" in sys.argv else "192.168.1.12"
port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 5005
stage = int(sys.argv[sys.argv.index("--stage") + 1]) if "--stage" in sys.argv else 2
view = "--no-view" not in sys.argv

STAGE_NOBS = [0, 1, 2, 3, 4]
env = RobotNavEnv(n_obstacles=STAGE_NOBS[stage])
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
dt = env.frame_skip * env.model.opt.timestep


def act(obs):
    img = obs["img"].transpose(2, 0, 1)[None].astype(np.float32)
    v = obs["vec"][None].astype(np.float32)
    return np.clip(sess.run(["action"], {"img": img, "vec": v})[0][0], -1.0, 1.0)


def send(a):
    sock.sendto(PKT.pack(float(a[0]), float(a[1])), (host, port))


def run(viewer=None):
    obs, _ = env.reset()
    ep = 0
    print(f"streaming [left,right] -> {host}:{port} at {1/dt:.0f} Hz  (Ctrl-C to stop)")
    while viewer is None or viewer.is_running():
        t0 = time.time()
        a = act(obs)
        send(a)                                   # <-- real robot mirrors this
        obs, r, term, trunc, info = env.step(a)
        if viewer is not None:
            viewer.sync()
        if term or trunc:
            ep += 1
            print(f"  sim episode {ep}: {'REACHED' if info.get('is_success') else 'ended'} (dist {info['dist']:.2f})")
            obs, _ = env.reset()
        time.sleep(max(0, dt - (time.time() - t0)))  # pace to real time (20 Hz)


def main():
    print(f"MIRROR: sim policy -> Jetson Nano {host}:{port}")
    try:
        if view:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                viewer.opt.geomgroup[_LIDAR_GROUP] = 1
                run(viewer)
        else:
            run(None)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(5):
            sock.sendto(PKT.pack(0.0, 0.0), (host, port))  # tell the robot to STOP
            time.sleep(0.02)
        print("\nsent STOP, exiting.")


if __name__ == "__main__":
    main()
