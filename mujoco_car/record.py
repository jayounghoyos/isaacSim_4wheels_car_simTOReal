"""Roll out a trained (or random) policy and save an MP4 so you can WATCH the car.

Usage: python mujoco_car/record.py [out.mp4] [n_episodes] [--random]
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import imageio
from mujoco_car.env import DriveToCubeEnv

HERE = os.path.dirname(__file__)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/tmp/drive.mp4"
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 3
    use_random = "--random" in sys.argv

    env = DriveToCubeEnv(render_mode="rgb_array")

    policy = None
    vec = None
    if not use_random:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        policy = PPO.load(os.path.join(HERE, "ppo_drivetocube"), device="cpu")
        vec = VecNormalize.load(os.path.join(HERE, "vecnormalize.pkl"),
                                DummyVecEnv([lambda: DriveToCubeEnv()]))
        vec.training = False

    frames, successes = [], 0
    for ep in range(n_ep):
        obs, _ = env.reset(seed=100 + ep)
        done = False
        while not done:
            if use_random:
                action = env.action_space.sample()
            else:
                norm_obs = vec.normalize_obs(obs)
                action, _ = policy.predict(norm_obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            frames.append(env.render())
            done = term or trunc
        successes += int(info.get("is_success", False))
        print(f"episode {ep}: {'REACHED' if info.get('is_success') else 'missed'} (final dist {info['dist']:.2f})")

    imageio.mimsave(out, frames, fps=20)
    print(f"RECORD_DONE successes={successes}/{n_ep} -> {out} ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
