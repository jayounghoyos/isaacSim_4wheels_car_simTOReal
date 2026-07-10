"""Watch the trained car drive to the cube in MuJoCo's interactive viewer (real-time window).

Run this in a terminal ON YOUR DESKTOP (so the window opens on your monitor):

    conda activate isaaclab
    cd ~/personalProjects/personalRobot
    PYTHONPATH=$PWD python mujoco_car/watch.py

Controls: drag to orbit the camera, scroll to zoom. Close the window to quit.
Pass --random to watch the untrained (random) policy instead.
"""
import os, sys, time
# NOTE: do NOT force MUJOCO_GL=egl here — the interactive viewer needs an on-screen GL context.
import numpy as np
import mujoco
import mujoco.viewer
from mujoco_car.env import DriveToCubeEnv

HERE = os.path.dirname(__file__)
use_random = "--random" in sys.argv

env = DriveToCubeEnv()
predict = None
vec = None
if not use_random:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    model = PPO.load(os.path.join(HERE, "ppo_drivetocube"), device="cpu")
    vec = VecNormalize.load(os.path.join(HERE, "vecnormalize.pkl"),
                            DummyVecEnv([lambda: DriveToCubeEnv()]))
    vec.training = False
    predict = lambda o: model.predict(vec.normalize_obs(o), deterministic=True)[0]

obs, _ = env.reset()
dt = env.frame_skip * env.model.opt.timestep
episode = 0
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        action = env.action_space.sample() if use_random else predict(obs)
        obs, r, term, trunc, info = env.step(action)
        viewer.sync()
        time.sleep(dt)  # real-time pacing
        if term or trunc:
            episode += 1
            print(f"episode {episode}: {'REACHED' if info.get('is_success') else 'missed'} "
                  f"(dist {info['dist']:.2f}) — resetting")
            obs, _ = env.reset()
            time.sleep(0.5)
