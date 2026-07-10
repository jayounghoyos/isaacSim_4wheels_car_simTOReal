"""Curriculum PPO for pose-based navigation with LIDAR + camera (MultiInputPolicy).

Stages: 0 obstacles (warmup) -> 1 -> 2 -> 3 -> 4. Auto-advances on rolling success; saves model +
filmstrip at each promotion and a '_latest' checkpoint every rollout for the live viewer.

Usage: PYTHONPATH=$PWD MUJOCO_GL=egl python mujoco_car/train_nav_curriculum.py [total] [--resume --start-stage K]
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from mujoco_car.robot_env_nav import RobotNavEnv

from mujoco_car.nav_config import (
    STAGES, SUCCESS_THRESHOLD, MIN_EPS_PER_STAGE, MAX_STEPS_PER_STAGE, N_ENVS, MODEL, VEC, LOGDIR, FILMSTRIP_DIR)

HERE = os.path.dirname(__file__)
os.makedirs(LOGDIR, exist_ok=True)


def make_env(seed, cfg):
    _, nobs, lo, hi = cfg
    def _f():
        env = Monitor(RobotNavEnv(n_obstacles=nobs, goal_lo=lo, goal_hi=hi), info_keywords=("is_success",))
        env.reset(seed=seed)
        return env
    return _f


def save_filmstrip(model, vecnorm, cfg, n_eval=8):
    stage_name, nobs, lo, hi = cfg
    env = RobotNavEnv(render_mode="rgb_array", n_obstacles=nobs, goal_lo=lo, goal_hi=hi)
    succ, best = 0, None
    for ep in range(n_eval):
        obs, _ = env.reset(seed=1000 + ep)
        frames, done = [], False
        while not done:
            a, _ = model.predict(vecnorm.normalize_obs(obs), deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            frames.append(env.render()); done = term or trunc
        if info.get("is_success"):
            succ += 1; best = best or frames
    best = best or frames
    idx = np.linspace(0, len(best) - 1, 5).astype(int)
    strip = np.concatenate([np.asarray(Image.fromarray(best[i]).resize((384, 216))) for i in idx], axis=1)
    out = os.path.join(FILMSTRIP_DIR, f"filmstrip_nav_{stage_name}.png")
    Image.fromarray(strip).save(out); env.close()
    print(f"[FILMSTRIP] {stage_name}: eval {succ}/{n_eval} -> {out}", flush=True)


class CurriculumCallback(BaseCallback):
    def __init__(self, venv):
        super().__init__()
        self.venv = venv; self.stage = 0; self.stage_start_steps = 0

    def _on_training_start(self):
        self.stage_start_steps = self.num_timesteps

    def _success(self):
        buf = self.model.ep_info_buffer
        vals = [e.get("is_success", 0.0) for e in buf if "is_success" in e]
        return (float(np.mean(vals)), len(vals)) if vals else (0.0, 0)

    def on_rollout_end(self):
        try:
            self.model.save(MODEL + "_latest"); self.venv.save(VEC.replace(".pkl", "_latest.pkl"))
        except Exception:
            pass
        sr, n = self._success()
        self.logger.record("curriculum/stage", self.stage)
        self.logger.record("curriculum/success_rate", sr)
        in_stage = self.num_timesteps - self.stage_start_steps
        promote = (sr >= SUCCESS_THRESHOLD and n >= MIN_EPS_PER_STAGE) or (in_stage >= MAX_STEPS_PER_STAGE)
        if promote and self.stage < len(STAGES) - 1:
            cfg = STAGES[self.stage]
            print(f"[PROMOTE] '{cfg[0]}' at success={sr:.2f} ({in_stage} steps)", flush=True)
            self.model.save(MODEL); self.venv.save(VEC)
            save_filmstrip(self.model, self.venv, cfg)
            self.stage += 1; self.stage_start_steps = self.num_timesteps
            nxt = STAGES[self.stage]
            self.venv.env_method("set_stage", nxt[1], nxt[2], nxt[3])
            print(f"[STAGE] -> '{nxt[0]}' (n_obstacles={nxt[1]}, goal {nxt[2]}-{nxt[3]}m)", flush=True)

    def _on_step(self):
        return True


def main():
    args = sys.argv[1:]
    resume = "--resume" in args
    start_stage = int(args[args.index("--start-stage") + 1]) if "--start-stage" in args else 0
    nums = [a for a in args if a.isdigit()]
    total = int(nums[0]) if nums else 6_000_000

    stage_cfg = STAGES[start_stage]
    venv = SubprocVecEnv([make_env(i, stage_cfg) for i in range(N_ENVS)])
    if resume and os.path.exists(MODEL + ".zip") and os.path.exists(VEC):
        venv = VecNormalize.load(VEC, venv); venv.training = True; venv.norm_reward = True
        model = PPO.load(MODEL, env=venv, device="cuda", tensorboard_log=LOGDIR)
        print(f"RESUME from {MODEL} at stage {start_stage} ({STAGES[start_stage][0]})", flush=True)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_obs_keys=["vec"], norm_reward=True, clip_obs=10.0)
        model = PPO("MultiInputPolicy", venv, policy_kwargs=dict(net_arch=[256, 256]),
                    n_steps=512, batch_size=1536, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                    learning_rate=3e-4, clip_range=0.2, ent_coef=0.005, device="cuda",
                    tensorboard_log=LOGDIR, verbose=1)
    cb = CurriculumCallback(venv); cb.stage = start_stage
    print(f"TRAIN_START nav total={total} start_stage={STAGES[start_stage][0]} resume={resume} n_envs={N_ENVS}", flush=True)
    model.learn(total_timesteps=total, callback=cb, progress_bar=False, reset_num_timesteps=True)
    model.save(MODEL); venv.save(VEC)
    save_filmstrip(model, venv, STAGES[cb.stage])
    print("TRAIN_DONE saved ppo_robot_nav", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
