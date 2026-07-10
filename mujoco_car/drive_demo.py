"""Manual drive demo in the GUI — scripted forward/turn so you can SEE the physics
(no trained policy). Drives forward a few seconds, turns, repeats. Run on your desktop:
    PYTHONPATH=$PWD python mujoco_car/drive_demo.py
"""
import time
import numpy as np
import mujoco, mujoco.viewer
from mujoco_car.robot_env import RobotDriveToCubeEnv

env = RobotDriveToCubeEnv()
obs, _ = env.reset()
dt = env.frame_skip * env.model.opt.timestep
t = 0
with mujoco.viewer.launch_passive(env.model, env.data) as v:
    while v.is_running():
        phase = (t * dt) % 6.0
        a = np.array([1.0, 1.0]) if phase < 4.0 else np.array([1.0, -1.0])  # forward, then turn
        obs, r, term, trunc, info = env.step(a)
        v.sync(); time.sleep(dt); t += 1
        if term or trunc:
            obs, _ = env.reset()
