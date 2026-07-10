"""Curriculum PPO for the LIDAR obstacle-avoidance task.

Stages: 1 obstacle -> 2 -> 3 -> 1 MOVING obstacle. Auto-advances when the rolling success
rate crosses a threshold; saves the model + a filmstrip eval PNG at every promotion so you
can watch progress. Cube position stays privileged (LIDAR is for avoidance).

Usage: PYTHONPATH=$PWD MUJOCO_GL=egl python mujoco_car/train_curriculum.py [total_timesteps]
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from mujoco_car.robot_env_lidar import RobotLidarEnv

HERE = os.path.dirname(__file__)
LOGDIR = os.path.join(HERE, "runs_lidar")
os.makedirs(LOGDIR, exist_ok=True)

STAGES = [
    ("0_warmup",    dict(n_obstacles=0, moving=False)),   # learn to reach with the 32-d LIDAR obs first
    ("1_obstacle",  dict(n_obstacles=1, moving=False)),
    ("2_obstacles", dict(n_obstacles=2, moving=False)),
    ("3_obstacles", dict(n_obstacles=3, moving=False)),
    ("1_moving",    dict(n_obstacles=1, moving=True)),
]
SUCCESS_THRESHOLD = 0.60   # rolling success (with exploration) runs ~20 pts below deterministic eval
MIN_EPS_PER_STAGE = 60          # require this many episodes before a promotion counts
MAX_STEPS_PER_STAGE = 2_000_000  # promote anyway if stuck this long


def make_env(seed, **stage):
    def _f():
        env = Monitor(RobotLidarEnv(**stage), info_keywords=("is_success",))
        env.reset(seed=seed)
        return env
    return _f


def save_filmstrip(model, vecnorm, stage_name, n_eval=8):
    """Run deterministic eval episodes; save a 5-frame filmstrip of a successful one."""
    env = RobotLidarEnv(render_mode="rgb_array", **dict(STAGES[[s[0] for s in STAGES].index(stage_name)][1]))
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
    out = os.path.join(HERE, f"filmstrip_{stage_name}.png")
    Image.fromarray(strip).save(out)
    env.close()
    print(f"[FILMSTRIP] {stage_name}: eval success {succ}/{n_eval} -> {out}", flush=True)
    return succ / n_eval


class CurriculumCallback(BaseCallback):
    def __init__(self, venv, verbose=1):
        super().__init__(verbose)
        self.venv = venv
        self.stage = 0
        self.stage_start_steps = 0

    def _success_rate(self):
        buf = self.model.ep_info_buffer
        vals = [e.get("is_success", 0.0) for e in buf if "is_success" in e]
        return (float(np.mean(vals)), len(vals)) if vals else (0.0, 0)

    def _rollout_end(self):
        sr, n = self._success_rate()
        steps_in_stage = self.num_timesteps - self.stage_start_steps
        self.logger.record("curriculum/stage", self.stage)
        self.logger.record("curriculum/success_rate", sr)
        promote = (sr >= SUCCESS_THRESHOLD and n >= MIN_EPS_PER_STAGE) or \
                  (steps_in_stage >= MAX_STEPS_PER_STAGE)
        if promote and self.stage < len(STAGES) - 1:
            name = STAGES[self.stage][0]
            print(f"[PROMOTE] stage '{name}' done at success={sr:.2f} "
                  f"({steps_in_stage} steps). Saving + filmstrip...", flush=True)
            self.model.save(os.path.join(HERE, "ppo_robot_lidar"))
            self.venv.save(os.path.join(HERE, "vecnormalize_robot_lidar.pkl"))
            save_filmstrip(self.model, self.venv, name)
            self.stage += 1
            self.stage_start_steps = self.num_timesteps
            cfg = STAGES[self.stage][1]
            self.venv.env_method("set_stage", cfg["n_obstacles"], cfg["moving"])
            print(f"[STAGE] -> '{STAGES[self.stage][0]}' {cfg}", flush=True)

    def _on_training_start(self):
        self.stage_start_steps = self.num_timesteps

    def on_rollout_end(self):
        # periodic 'latest' checkpoint (~once per rollout) so a live viewer can track training mid-run
        try:
            self.model.save(os.path.join(HERE, "ppo_robot_lidar_latest"))
            self.venv.save(os.path.join(HERE, "vecnormalize_robot_lidar_latest.pkl"))
        except Exception:
            pass
        self._rollout_end()

    def _on_step(self):
        return True


def main():
    args = sys.argv[1:]
    resume = "--resume" in args
    start_stage = int(args[args.index("--start-stage") + 1]) if "--start-stage" in args else 0
    nums = [a for a in args if a.isdigit()]
    total = int(nums[0]) if nums else 4_000_000
    n_envs = 8

    stage_cfg = STAGES[start_stage][1]
    venv = SubprocVecEnv([make_env(i, **stage_cfg) for i in range(n_envs)])
    mpath = os.path.join(HERE, "ppo_robot_lidar")
    vpath = os.path.join(HERE, "vecnormalize_robot_lidar.pkl")
    if resume and os.path.exists(mpath + ".zip") and os.path.exists(vpath):
        venv = VecNormalize.load(vpath, venv)
        venv.training = True; venv.norm_reward = True
        model = PPO.load(mpath, env=venv, device="cpu", tensorboard_log=LOGDIR)
        print(f"RESUME from {mpath} at stage {start_stage} ({STAGES[start_stage][0]})", flush=True)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO("MlpPolicy", venv, policy_kwargs=dict(net_arch=[128, 128]),
                    n_steps=1024, batch_size=2048, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                    learning_rate=3e-4, clip_range=0.2, ent_coef=0.005, device="cpu",
                    tensorboard_log=LOGDIR, verbose=1)
    cb = CurriculumCallback(venv)
    cb.stage = start_stage
    print(f"TRAIN_START curriculum total={total} start_stage={STAGES[start_stage][0]} resume={resume}", flush=True)
    # reset the step counter even on resume: keeps weights+optimizer, makes `total` mean
    # "this many MORE steps" and gives each remaining stage a fresh MAX_STEPS budget.
    model.learn(total_timesteps=total, callback=cb, progress_bar=False,
                reset_num_timesteps=True)

    model.save(os.path.join(HERE, "ppo_robot_lidar"))
    venv.save(os.path.join(HERE, "vecnormalize_robot_lidar.pkl"))
    # final filmstrip on the last stage
    save_filmstrip(model, venv, STAGES[cb.stage][0])
    print("TRAIN_DONE saved ppo_robot_lidar", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
