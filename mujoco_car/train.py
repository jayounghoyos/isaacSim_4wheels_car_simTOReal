"""Train PPO (stable-baselines3) on DriveToCube. Saves model + VecNormalize stats + TensorBoard logs.

Usage: python mujoco_car/train.py [total_timesteps] [n_envs]
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from mujoco_car.env import DriveToCubeEnv

HERE = os.path.dirname(__file__)
LOGDIR = os.path.join(HERE, "runs")
os.makedirs(LOGDIR, exist_ok=True)


def make_env(seed):
    def _f():
        env = DriveToCubeEnv()
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _f


def main():
    total_timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
    n_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    venv = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", venv,
        policy_kwargs=dict(net_arch=[64, 64]),
        n_steps=1024, batch_size=2048, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
        clip_range=0.2, ent_coef=0.0,
        device="cpu",                       # small MLP -> CPU is faster than GPU
        tensorboard_log=LOGDIR, verbose=1,
    )
    print(f"TRAIN_START timesteps={total_timesteps} n_envs={n_envs}")
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    model.save(os.path.join(HERE, "ppo_drivetocube"))
    venv.save(os.path.join(HERE, "vecnormalize.pkl"))
    print("TRAIN_DONE saved model + vecnormalize")
    venv.close()


if __name__ == "__main__":
    main()
