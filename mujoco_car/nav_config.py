"""Single source of truth for the navigation env + training constants.

Change the world/curriculum/paths HERE (one place) instead of hunting through the env and trainer.
NOTE: the world dims below must match `robot/train_scene_nav.xml` — `test_nav_env.py` asserts this so
they can't silently drift.
"""
import os

HERE = os.path.dirname(__file__)
SCENE_XML = os.path.abspath(os.path.join(HERE, "..", "robot", "train_scene_nav.xml"))

# ---- world / arena (must match the wall + obstacle geoms in train_scene_nav.xml) ----
ARENA = 13.0            # robot/goal/obstacles kept within +/- this (walls inner face ~15)
WALL_INNER = 15.0       # inner face of the walls (for the scene-consistency test)
OBS_HALF = 0.3          # obstacle box half-size (x, y)
OBS_Z = 0.5             # obstacle center z (== half-height -> bottom on the floor)
PARK = (200.0, 200.0, 0.5)   # off-arena park position for unused obstacles
LIDAR_GROUP = 5         # geom group for LIDAR-sensable walls + obstacles
REST_Z = 0.20           # robot spawn height

# ---- robot / sensing / control ----
IMG = 64                # camera resolution (square)
N_RAYS = 24
LIDAR_MAX = 8.0
WHEEL_VEL_SCALE = 10.0  # ~1.8 m/s top speed
FRAME_SKIP = 25
MAX_STEPS = 400
REACH_TOL = 1.1
GOAL_LO, GOAL_HI = 2.5, 5.0   # default goal distance range (per-stage curriculum overrides this)

# ---- curriculum: (name, n_obstacles, goal_lo, goal_hi) — distance warmup then obstacles ----
STAGES = [
    ("warmup_near", 0, 2.5, 5.0),
    ("warmup_mid",  0, 4.0, 10.0),
    ("warmup_far",  0, 6.0, 18.0),
    ("1_obstacle",  1, 4.0, 14.0),
    ("2_obstacles", 2, 4.0, 14.0),
    ("3_obstacles", 3, 4.0, 14.0),
    ("4_obstacles", 4, 4.0, 14.0),
]

# ---- training ----
N_ENVS = 6
DEVICE = "cuda"
SUCCESS_THRESHOLD = 0.55
MIN_EPS_PER_STAGE = 50
MAX_STEPS_PER_STAGE = 1_200_000

# ---- output/model paths ----
MODEL = os.path.join(HERE, "ppo_robot_nav")
VEC = os.path.join(HERE, "vecnormalize_robot_nav.pkl")
LOGDIR = os.path.join(HERE, "runs_nav")
ONNX = os.path.join(HERE, "policy_nav.onnx")
NORM_JSON = os.path.join(HERE, "policy_nav_norm.json")
