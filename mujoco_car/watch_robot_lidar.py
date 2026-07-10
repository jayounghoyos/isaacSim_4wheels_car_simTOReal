"""Watch the LIDAR/obstacle policy LIVE while it trains, with the LIDAR rays drawn.

Auto-reloads the newest checkpoint each episode, so as training progresses you see the
policy improve in real time. Run on your desktop:
    conda activate isaaclab
    cd ~/personalProjects/personalRobot
    PYTHONPATH=$PWD python mujoco_car/watch_robot_lidar.py [--stage 0..4] [--random] [--live]
--live (default when training) tracks the '_latest' checkpoint written every rollout.
"""
import os, sys, time
import numpy as np
import mujoco, mujoco.viewer
from mujoco_car.robot_env_lidar import RobotLidarEnv, _LIDAR_GROUP

HERE = os.path.dirname(__file__)
# stage index matches train_curriculum.STAGES: 0=warmup .. 4=moving
STAGES = [dict(n_obstacles=0, moving=False), dict(n_obstacles=1, moving=False),
          dict(n_obstacles=2, moving=False), dict(n_obstacles=3, moving=False),
          dict(n_obstacles=1, moving=True)]

stage = 3
if "--stage" in sys.argv:
    stage = int(sys.argv[sys.argv.index("--stage") + 1])
use_random = "--random" in sys.argv
live = ("--live" in sys.argv) or True   # default: follow the latest checkpoint

env = RobotLidarEnv(**STAGES[stage])

# checkpoint names: prefer the frequently-updated '_latest' (live), fall back to the promotion save
_MODEL_LIVE = os.path.join(HERE, "ppo_robot_lidar_latest")
_VEC_LIVE = os.path.join(HERE, "vecnormalize_robot_lidar_latest.pkl")
_MODEL_STABLE = os.path.join(HERE, "ppo_robot_lidar")
_VEC_STABLE = os.path.join(HERE, "vecnormalize_robot_lidar.pkl")

predict = None
_loaded = {"mtime": 0.0}
if not use_random:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

    def _paths():
        if live and os.path.exists(_MODEL_LIVE + ".zip"):
            return _MODEL_LIVE, _VEC_LIVE
        return _MODEL_STABLE, _VEC_STABLE

    def reload_policy():
        """Reload the newest checkpoint if it changed; returns a predict fn (or keeps the old one)."""
        global predict
        mpath, vpath = _paths()
        try:
            mtime = os.path.getmtime(mpath + ".zip")
            if mtime <= _loaded["mtime"]:
                return
            model = PPO.load(mpath, device="cpu")
            vec = VecNormalize.load(vpath, DummyVecEnv([lambda: RobotLidarEnv(**STAGES[stage])]))
            vec.training = False
            predict = lambda o: model.predict(vec.normalize_obs(o), deterministic=True)[0]
            _loaded["mtime"] = mtime
            print(f"[viewer] loaded checkpoint {os.path.basename(mpath)} (mtime {time.strftime('%H:%M:%S', time.localtime(mtime))})")
        except Exception as e:
            pass  # mid-write or missing; keep the current policy

    reload_policy()


def draw_rays(viewer):
    """Populate user scene geoms with one thin line per LIDAR ray (origin -> hit point)."""
    scn = viewer.user_scn
    origin = np.array(env.data.site_xpos[env._lidar_site], dtype=np.float64)
    yaw = env._car_yaw()
    n = 0
    for base in env._ray_base:
        if n >= scn.maxgeom:
            break
        ang = yaw + base
        vec = np.array([np.cos(ang), np.sin(ang), 0.0])
        dist = mujoco.mj_ray(env.model, env.data, origin, vec,
                             env._geomgroup, 1, env._robot_body, env._gid)
        d = dist if dist >= 0 else env.lidar_max
        end = origin + vec * d
        hit = dist >= 0
        g = scn.geoms[n]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                            np.zeros(3), np.zeros(9),
                            np.array([1, 0.3, 0.1, 1] if hit else [0.2, 0.9, 0.3, 0.5], np.float32))
        # MuJoCo 3.10: mjv_connector(geom, type, width, from_xyz, to_xyz)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 3.0,
                             origin.astype(np.float64), end.astype(np.float64))
        n += 1
    scn.ngeom = n


def main():
    obs, _ = env.reset()
    dt = env.frame_skip * env.model.opt.timestep
    ep = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.geomgroup[_LIDAR_GROUP] = 1   # show walls + obstacles
        while viewer.is_running():
            a = env.action_space.sample() if use_random else predict(obs)
            obs, r, term, trunc, info = env.step(a)
            draw_rays(viewer)
            viewer.sync()
            time.sleep(dt)
            if term or trunc:
                ep += 1
                tag = "REACHED" if info.get("is_success") else ("HIT" if info.get("collided") else "timeout")
                print(f"episode {ep}: {tag} (dist {info['dist']:.2f})")
                if not use_random:
                    reload_policy()   # pick up the newest training checkpoint between episodes
                obs, _ = env.reset(); time.sleep(0.4)


if __name__ == "__main__":
    main()
