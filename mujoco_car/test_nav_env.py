"""Tests for the pose-based navigation env (Dict obs + camera + A->B + 60/40 obstacles).

Run:  PYTHONPATH=$PWD MUJOCO_GL=egl python mujoco_car/test_nav_env.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
from mujoco_car.robot_env_nav import RobotNavEnv, _OBS_HALF, _OBS_Z, _ARENA


def test_dict_obs_and_camera():
    env = RobotNavEnv(n_obstacles=2)
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"vec", "img"}
    assert obs["vec"].shape == (8 + env.n_rays,), obs["vec"].shape
    assert obs["img"].shape == (64, 64, 3) and obs["img"].dtype == np.uint8
    assert obs["img"].mean() > 1.0, "camera image is blank"
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs["img"].mean() > 1.0
    print(f"  [ok] Dict obs vec{obs['vec'].shape} img{obs['img'].shape} camera-mean {obs['img'].mean():.1f}")
    env.close()


def test_lidar_front_face_raised():
    """LIDAR still hits the FRONT face at the raised mount; flg_static=0 misses (through-wall bug)."""
    env = RobotNavEnv(n_obstacles=1)
    m, d = env.model, env.data
    fj = env._fj
    mujoco.mj_resetData(m, d)
    d.qpos[fj:fj + 3] = [0, 0, 0.2]; d.qpos[fj + 3:fj + 7] = [1, 0, 0, 0]
    for mid in env._obs_mocap:
        d.mocap_pos[mid] = [200, 200, _OBS_Z]
    d.mocap_pos[env._target_mocap] = [80, 80, 0.02]
    obs_x = 4.0
    d.mocap_pos[env._obs_mocap[0]] = [obs_x, 0.0, _OBS_Z]
    mujoco.mj_forward(m, d)
    origin = np.array(d.site_xpos[env._lidar_site])
    dist = mujoco.mj_ray(m, d, origin, np.array([1.0, 0, 0]), env._geomgroup, 1, env._robot_body, env._gid)
    expected = (obs_x - _OBS_HALF) - origin[0]
    assert abs(dist - expected) < 0.05, f"front-face {dist:.3f} != {expected:.3f}"
    dist0 = mujoco.mj_ray(m, d, origin, np.array([1.0, 0, 0]), env._geomgroup, 0, env._robot_body, env._gid)
    assert dist0 < 0
    print(f"  [ok] raised-LIDAR front-face {dist:.3f} (exp {expected:.3f}); flg_static=0 -> miss; site z={origin[2]:.2f}")
    env.close()


def test_placement_6040(n=200):
    """Obstacles on floor, inside arena, clearances hold, and ~60% land near the goal."""
    env = RobotNavEnv(n_obstacles=4)
    near_goal, total = 0, 0
    for k in range(n):
        env.reset(seed=k)
        goal = env._target_xy(); robot = env._car_xy()
        active = []
        for mid in env._obs_mocap:
            p = np.array(env.data.mocap_pos[mid])
            if np.max(np.abs(p[:2])) > 100:
                continue
            assert abs(p[2] - _OBS_Z) < 1e-6, "obstacle not on floor"
            assert np.max(np.abs(p[:2])) <= _ARENA + 1e-6, "obstacle outside arena"
            assert np.linalg.norm(p[:2] - robot) >= 2.2 - 0.3, "too close to robot"
            assert np.linalg.norm(p[:2] - goal) >= 1.6 - 0.3, "on the goal"
            total += 1
            if np.linalg.norm(p[:2] - goal) <= 4.0:
                near_goal += 1
            active.append(p[:2])
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                assert np.linalg.norm(active[i] - active[j]) >= 2.8 - 0.3, "obstacles too close"
    frac = near_goal / max(total, 1)
    assert 0.45 <= frac <= 0.75, f"near-goal fraction {frac:.2f} not ~60%"
    print(f"  [ok] {n} resets: {total} obstacles, {frac:.0%} near goal (~60% target), all clearances hold")
    env.close()


def test_gym():
    from gymnasium.utils.env_checker import check_env
    check_env(RobotNavEnv(n_obstacles=1), skip_render_check=True)
    print("  [ok] gymnasium check passed (Dict obs)")


if __name__ == "__main__":
    print("test_dict_obs_and_camera");   test_dict_obs_and_camera()
    print("test_lidar_front_face_raised"); test_lidar_front_face_raised()
    print("test_placement_6040");         test_placement_6040()
    print("test_gym");                    test_gym()
    print("\nALL_NAV_TESTS_PASSED")
