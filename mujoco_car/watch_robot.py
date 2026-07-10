"""Watch YOUR trained robot drive to the cube in MuJoCo's interactive viewer.

Run in a terminal on your desktop:
    conda activate isaaclab
    cd ~/personalProjects/personalRobot
    PYTHONPATH=$PWD python mujoco_car/watch_robot.py
Add --random to watch the untrained robot.
"""
import os, sys, time
import numpy as np
import mujoco, mujoco.viewer
from mujoco_car.robot_env import RobotDriveToCubeEnv

HERE = os.path.dirname(__file__)
use_random = "--random" in sys.argv

env = RobotDriveToCubeEnv()
predict = None
if not use_random:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    model = PPO.load(os.path.join(HERE, "ppo_robot_drivetocube"), device="cpu")
    vec = VecNormalize.load(os.path.join(HERE, "vecnormalize_robot.pkl"),
                            DummyVecEnv([lambda: RobotDriveToCubeEnv()]))
    vec.training = False
    predict = lambda o: model.predict(vec.normalize_obs(o), deterministic=True)[0]

obs, _ = env.reset()
dt = env.frame_skip * env.model.opt.timestep
ep = 0
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        action = env.action_space.sample() if use_random else predict(obs)
        obs, r, term, trunc, info = env.step(action)
        viewer.sync()
        time.sleep(dt)
        if term or trunc:
            ep += 1
            print(f"episode {ep}: {'REACHED' if info.get('is_success') else 'missed'} (dist {info['dist']:.2f})")
            obs, _ = env.reset()
            time.sleep(0.5)
