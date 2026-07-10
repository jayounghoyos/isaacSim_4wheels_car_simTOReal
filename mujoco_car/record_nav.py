"""Record a side-by-side MP4: [third-person + LIDAR rays | robot camera POV]. Headless (EGL), no GUI.

Usage: PYTHONPATH=$PWD MUJOCO_GL=egl python mujoco_car/record_nav.py [out.mp4] [n_episodes] [--stage K] [--random]
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import imageio
import mujoco
from mujoco_car.robot_env_nav import RobotNavEnv, _LIDAR_GROUP

HERE = os.path.dirname(__file__)
STAGE_NOBS = [0, 1, 2, 3, 4]

out = next((a for a in sys.argv[1:] if a.endswith(".mp4")), "/tmp/nav.mp4")
n_ep = next((int(a) for a in sys.argv[1:] if a.isdigit()), 4)
stage = int(sys.argv[sys.argv.index("--stage") + 1]) if "--stage" in sys.argv else 3
use_random = "--random" in sys.argv

env = RobotNavEnv(render_mode="rgb_array", n_obstacles=STAGE_NOBS[stage])
predict = None
if not use_random:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    mp = os.path.join(HERE, "ppo_robot_nav_latest")
    vp = os.path.join(HERE, "vecnormalize_robot_nav_latest.pkl")
    if not os.path.exists(mp + ".zip"):
        mp, vp = os.path.join(HERE, "ppo_robot_nav"), os.path.join(HERE, "vecnormalize_robot_nav.pkl")
    model = PPO.load(mp, device="cpu")
    vec = VecNormalize.load(vp, DummyVecEnv([lambda: RobotNavEnv(n_obstacles=STAGE_NOBS[stage])]))
    vec.training = False
    predict = lambda o: model.predict(vec.normalize_obs(o), deterministic=True)[0]

H = 480
third = mujoco.Renderer(env.model, H, int(H * 4 / 3), max_geom=20000)
opt = mujoco.MjvOption(); opt.geomgroup[_LIDAR_GROUP] = 1


def third_with_rays():
    cam = mujoco.MjvCamera(); cx, cy = env._car_xy()
    cam.lookat[:] = [cx, cy, 0.2]; cam.distance = 11.0
    cam.azimuth = np.degrees(env._car_yaw()) + 180.0; cam.elevation = -32
    third.update_scene(env.data, camera=cam, scene_option=opt)
    scn = third.scene; n = scn.ngeom
    origin = np.array(env.data.site_xpos[env._lidar_site], np.float64); yaw = env._car_yaw()
    for base in env._ray_base:
        if n >= scn.maxgeom:
            break
        ang = yaw + base; v = np.array([np.cos(ang), np.sin(ang), 0.0])
        dist = mujoco.mj_ray(env.model, env.data, origin, v, env._geomgroup, 1, env._robot_body, env._gid)
        d = dist if dist >= 0 else env.lidar_max
        g = scn.geoms[n]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array([1, 0.3, 0.1, 1] if dist >= 0 else [0.2, 0.9, 0.3, 0.6], np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 3.0, origin, (origin + v * d).astype(np.float64))
        n += 1
    scn.ngeom = n
    return third.render()


frames, succ = [], 0
for ep in range(n_ep):
    obs, _ = env.reset(seed=1234 + ep)
    done = False
    while not done:
        a = env.action_space.sample() if use_random else predict(obs)
        obs, r, term, trunc, info = env.step(a)
        tp = third_with_rays()                      # H x (4H/3) x 3
        from PIL import Image
        pov_big = np.asarray(Image.fromarray(obs["img"]).resize((H, H), Image.NEAREST))  # exact H x H
        combo = np.concatenate([tp, np.full((H, 8, 3), 30, np.uint8), pov_big], axis=1)
        frames.append(combo)
        done = term or trunc
    succ += int(info.get("is_success", False))
    print(f"episode {ep}: {'REACHED' if info.get('is_success') else 'missed'} (dist {info['dist']:.2f})")

imageio.mimsave(out, frames, fps=20)
print(f"RECORD_DONE {succ}/{n_ep} -> {out} ({len(frames)} frames)")
env.close()
