"""Watch the pose-based navigation policy LIVE in the interactive MuJoCo viewer, with LIDAR rays.

Run on your desktop:
    PYTHONPATH=$PWD python mujoco_car/watch_nav.py [--stage 0..4] [--random]
- Orbit with the mouse. To see the ROBOT'S ONBOARD CAMERA view, cycle cameras in the viewer
  (press '[' or ']' , or use the Camera menu) until it shows "robot_cam".
- Auto-reloads the newest '_latest' checkpoint each episode so you watch it improve mid-training.
(For a guaranteed side-by-side third-person + camera-POV clip, use mujoco_car/record_nav.py.)
"""
import os, sys, time
import numpy as np
import mujoco, mujoco.viewer
from mujoco_car.robot_env_nav import RobotNavEnv, _LIDAR_GROUP

HERE = os.path.dirname(__file__)
STAGE_NOBS = [0, 1, 2, 3, 4]
stage = int(sys.argv[sys.argv.index("--stage") + 1]) if "--stage" in sys.argv else 3
use_random = "--random" in sys.argv

env = RobotNavEnv(n_obstacles=STAGE_NOBS[stage])
_M_LIVE, _V_LIVE = os.path.join(HERE, "ppo_robot_nav_latest"), os.path.join(HERE, "vecnormalize_robot_nav_latest.pkl")
_M_ST, _V_ST = os.path.join(HERE, "ppo_robot_nav"), os.path.join(HERE, "vecnormalize_robot_nav.pkl")
predict = None
_loaded = {"mtime": 0.0}

if not use_random:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

    def reload_policy():
        global predict
        mp, vp = (_M_LIVE, _V_LIVE) if os.path.exists(_M_LIVE + ".zip") else (_M_ST, _V_ST)
        try:
            mt = os.path.getmtime(mp + ".zip")
            if mt <= _loaded["mtime"]:
                return
            model = PPO.load(mp, device="cpu")
            vec = VecNormalize.load(vp, DummyVecEnv([lambda: RobotNavEnv(n_obstacles=STAGE_NOBS[stage])]))
            vec.training = False
            predict = lambda o: model.predict(vec.normalize_obs(o), deterministic=True)[0]
            _loaded["mtime"] = mt
            print(f"[viewer] loaded {os.path.basename(mp)}")
        except Exception:
            pass
    reload_policy()


def add_rays(scene):
    origin = np.array(env.data.site_xpos[env._lidar_site], dtype=np.float64)
    yaw = env._car_yaw()
    scene.ngeom = 0
    n = 0
    for base in env._ray_base:
        if n >= scene.maxgeom:
            break
        ang = yaw + base
        v = np.array([np.cos(ang), np.sin(ang), 0.0])
        dist = mujoco.mj_ray(env.model, env.data, origin, v, env._geomgroup, 1, env._robot_body, env._gid)
        d = dist if dist >= 0 else env.lidar_max
        g = scene.geoms[n]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array([1, 0.3, 0.1, 1] if dist >= 0 else [0.2, 0.9, 0.3, 0.6], np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 3.0, origin, (origin + v * d).astype(np.float64))
        n += 1
    scene.ngeom = n


def main():
    obs, _ = env.reset()
    dt = env.frame_skip * env.model.opt.timestep
    ep = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.geomgroup[_LIDAR_GROUP] = 1
        while viewer.is_running():
            a = env.action_space.sample() if use_random else predict(obs)
            obs, r, term, trunc, info = env.step(a)
            add_rays(viewer.user_scn)
            viewer.sync(); time.sleep(dt)
            if term or trunc:
                ep += 1
                tag = "REACHED" if info.get("is_success") else ("HIT" if info.get("collided") else "timeout")
                print(f"episode {ep}: {tag} (dist {info['dist']:.2f})")
                if not use_random:
                    reload_policy()
                obs, _ = env.reset(); time.sleep(0.3)


if __name__ == "__main__":
    main()
