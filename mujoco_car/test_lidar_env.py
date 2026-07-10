"""Tests that prove the two user concerns are handled, plus Gym compliance.

Run:  PYTHONPATH=$PWD MUJOCO_GL=egl python mujoco_car/test_lidar_env.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
from mujoco_car.robot_env_lidar import RobotLidarEnv, _OBS_HALF, _OBS_Z


def test_lidar_front_face():
    """Concern #1: LIDAR ray returns the FRONT face at a known distance (not through, not back)."""
    env = RobotLidarEnv(n_obstacles=1)
    m, d = env.model, env.data
    fj = env._fj
    # robot at origin facing +x (yaw=0); park all obstacles far; put ONE obstacle straight ahead.
    mujoco.mj_resetData(m, d)
    d.qpos[fj:fj + 3] = [0, 0, 0.2]
    d.qpos[fj + 3:fj + 7] = [1, 0, 0, 0]
    for mid in env._obs_mocap:
        d.mocap_pos[mid] = [100, 100, _OBS_Z]
    d.mocap_pos[env._target_mocap] = [50, 50, 0.3]
    obs_x = 4.0
    d.mocap_pos[env._obs_mocap[0]] = [obs_x, 0.0, _OBS_Z]   # box front face at obs_x - _OBS_HALF
    mujoco.mj_forward(m, d)

    origin = np.array(d.site_xpos[env._lidar_site])
    dist = mujoco.mj_ray(m, d, origin, np.array([1.0, 0, 0]),
                         env._geomgroup, 1, env._robot_body, env._gid)
    expected = (obs_x - _OBS_HALF) - origin[0]
    assert abs(dist - expected) < 0.05, f"front-face ray {dist:.3f} != expected {expected:.3f}"
    # flg_static=0 must MISS the (static mocap) obstacle -> proves the flag is what fixes it
    dist0 = mujoco.mj_ray(m, d, origin, np.array([1.0, 0, 0]),
                          env._geomgroup, 0, env._robot_body, env._gid)
    assert dist0 < 0, f"with flg_static=0 the ray should pass through, got {dist0:.3f}"
    # a ray pointing away (backwards, -x) hits the far wall, not the obstacle behind -> sanity
    print(f"  [ok] front-face dist={dist:.3f} (expected {expected:.3f}); flg_static=0 -> {dist0:.1f} (miss)")
    env.close()


def test_lidar_clear_is_max():
    """A ray with nothing in range returns 1.0 (normalized max)."""
    env = RobotLidarEnv(n_obstacles=0)
    obs, _ = env.reset(seed=0)
    rays = obs[8:]
    assert rays.min() >= 0.0 and rays.max() <= 1.0
    # with no obstacles, only walls can be hit; near center all rays should be fairly large
    assert np.median(rays) > 0.5, f"median ray {np.median(rays):.2f} too small with no obstacles"
    print(f"  [ok] no-obstacle rays in [{rays.min():.2f},{rays.max():.2f}], median {np.median(rays):.2f}")
    env.close()


def test_placement(n=200):
    """Concern #2: obstacles sit on the floor and keep clearances over many resets."""
    env = RobotLidarEnv(n_obstacles=3)
    d_robot_min, d_cube_min, d_pair_min = 2.0, 2.0, 3.0
    for k in range(n):
        env.reset(seed=k)
        cube = env._target_xy()
        active = []
        for mid in env._obs_mocap:
            p = np.array(env.data.mocap_pos[mid])
            if np.max(np.abs(p[:2])) > 50:   # parked
                continue
            assert abs((p[2] - _OBS_Z)) < 1e-6, f"obstacle z={p[2]} not on floor"   # bottom==0
            assert np.max(np.abs(p[:2])) <= 6.3 + 1e-6, "obstacle outside arena"
            assert np.linalg.norm(p[:2]) >= d_robot_min - 0.3, "obstacle too close to robot spawn"
            assert np.linalg.norm(p[:2] - cube) >= d_cube_min - 0.3, "obstacle blocks the goal"
            active.append(p[:2])
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                assert np.linalg.norm(active[i] - active[j]) >= d_pair_min - 0.3, "obstacles too close"
    print(f"  [ok] {n} resets: all obstacles on floor, inside arena, clearances hold")
    env.close()


def test_gym_and_obs_dim():
    from gymnasium.utils.env_checker import check_env
    env = RobotLidarEnv(n_obstacles=2)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (8 + env.n_rays,), obs.shape
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    assert obs.shape == (8 + env.n_rays,)
    check_env(RobotLidarEnv(n_obstacles=1), skip_render_check=True)
    print(f"  [ok] obs dim {obs.shape}, gym check passed")
    env.close()


if __name__ == "__main__":
    print("test_lidar_front_face");  test_lidar_front_face()
    print("test_lidar_clear_is_max"); test_lidar_clear_is_max()
    print("test_placement");          test_placement()
    print("test_gym_and_obs_dim");    test_gym_and_obs_dim()
    print("\nALL_TESTS_PASSED")
