"""Continue training the robot PPO policy from the saved checkpoint (improves reliability).

Usage: python mujoco_car/continue_train_robot.py [extra_timesteps] [n_envs]
Loads ppo_robot_drivetocube + vecnormalize_robot.pkl, trains more, saves back.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from mujoco_car.robot_env import RobotDriveToCubeEnv

HERE = os.path.dirname(__file__)


def make_env(seed):
    def _f():
        e = Monitor(RobotDriveToCubeEnv()); e.reset(seed=seed); return e
    return _f


def main():
    extra = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    n_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    venv = SubprocVecEnv([make_env(100 + i) for i in range(n_envs)])
    venv = VecNormalize.load(os.path.join(HERE, "vecnormalize_robot.pkl"), venv)
    venv.training = True
    model = PPO.load(os.path.join(HERE, "ppo_robot_drivetocube"), env=venv, device="cpu",
                     tensorboard_log=os.path.join(HERE, "runs_robot"))
    print(f"CONTINUE_START extra={extra}")
    model.learn(total_timesteps=extra, reset_num_timesteps=False, progress_bar=False)
    model.save(os.path.join(HERE, "ppo_robot_drivetocube"))
    venv.save(os.path.join(HERE, "vecnormalize_robot.pkl"))
    print("CONTINUE_DONE saved improved robot model")
    venv.close()


if __name__ == "__main__":
    main()
