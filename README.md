# personalRobot — Learning to Train a 4-Wheeled Robot (Sim → Real)

A learning project: take a 4-wheeled robot designed in **Onshape** and teach it, with
**reinforcement learning**, to **drive to a target cube** (later: using a camera + 2D LIDAR).
Training happens in **MuJoCo** (+ Gymnasium + Stable-Baselines3 PPO); the long-term goal is to
deploy the learned policy to the **real robot** (Jetson + ROS 2).

> Status: **pose-based navigation trained + sim-to-real interface built** (`mujoco_car/`, `ros2_nav/`).
> Full plan: `~/.claude/plans/i-have-a-robot-glistening-pearl.md`

## Sim-to-real: ONNX export + ROS 2 interface — `ros2_nav/`
The trained policy is exported to a **portable ONNX** file (`mujoco_car/policy_nav.onnx`, 1.9 MB) that
runs with only `onnxruntime` — no MuJoCo/PyTorch — the artifact you copy to the Jetson. Verified
bit-identical to the SB3 policy (`export_onnx.py`; max action diff 1e-6). ONNX-in-env matches training
(0/2/4 obstacles → 80/70/53%).

A **ROS 2 (Jazzy)** interface makes the SAME policy node run in sim and on the real robot — only the
topic *producers* change:
```
SIM:    mujoco_bridge  --/scan /odom /camera/image_raw /goal_pose-->  policy_node  --/cmd_vel-->  mujoco_bridge
ROBOT:  lidar+enc/IMU+cam drivers  --same topics-->  policy_node  --/cmd_vel-->  L298N motor node
```
- `ros2_nav/policy_node.py` — **hardware-agnostic**: builds the 32-vec obs from `/scan`+`/odom`+`/goal_pose`,
  the image from `/camera/image_raw`, runs `policy_nav.onnx`, publishes `/cmd_vel` (Twist). Event-driven
  on `/odom` for one-step latency.
- `ros2_nav/mujoco_bridge.py` — the sim stand-in: runs MuJoCo, publishes the sensor topics, subscribes
  `/cmd_vel`. Self-contained (mujoco+numpy only), mirroring the minimal on-robot deps.
- Run the full loop: `bash ros2_nav/run_sim.sh 120` (needs `/opt/ros/jazzy` + the `ros2_venv` py3.12 venv).
- Env: `ros2_venv/` = python3.12 venv (`--system-site-packages` for rclpy) + `mujoco` + `onnxruntime`.

> **⚠️ THE sim-to-real gap (`/odom`):** in sim `/odom` is perfect (from the physics engine). The real
> robot must PRODUCE it from **wheel encoders + an IMU** — the current TT-motors + L298N have neither,
> so 7 of the 32 observation values (pose + body velocities) can't be measured yet. Options: add
> encoders+IMU (~$10), or retrain a policy that uses only LIDAR+camera+last-action (no global pose).
> The ROS 2 structure makes this gap explicit and modular — the policy node doesn't change either way.


> **Why MuJoCo and not Isaac Sim?** We tried hard to use NVIDIA Isaac Lab (the Jetson-native path),
> but Isaac Sim's RTX renderer **crashes/hangs on this RTX 5060 Ti (Blackwell) with the 595 driver**
> — a confirmed NVIDIA bug. MuJoCo renders with plain OpenGL/EGL, works on the GPU today, installs in
> minutes, and teaches the identical RL concepts. Isaac/Jetson is parked for the future sim-to-real
> phase. See `~/.claude/.../memory/isaac-rl-project-setup.md` for the full story.

---

## Why this stack
- **Isaac Lab / Isaac Sim** — GPU-accelerated RL that matches the hardware (RTX 5060 Ti) and the
  end goal (Jetson + ROS 2). The most direct sim-to-real path.
- **Leatherback** — a community 4-wheeled RL example (waypoint navigation, PPO) we adapt instead of
  starting from scratch: https://github.com/MuammerBay/Leatherback
- **Wheeled Lab** — reference for sim2real techniques (domain randomization, etc.).

## Verified environment (this machine)
| Component | Value |
|---|---|
| GPU | NVIDIA RTX 5060 Ti, 16 GB VRAM (Blackwell / 50-series) |
| OS | Ubuntu 24.04.4 LTS |
| NVIDIA driver | **595.71.05** (`nvidia-driver-595-open`) |
| Simulator | **MuJoCo 3.10** (renders via EGL on the GPU) |
| RL | **Gymnasium 1.2** + **Stable-Baselines3 2.9** (PPO) |
| Python | **3.11** (conda env `isaaclab` — name kept; also holds the parked Isaac install) |
| PyTorch | **2.7.0** + CUDA **12.8** (`cu128`) |
### Reproduce the setup
```bash
conda activate isaaclab            # Python 3.11 env (torch 2.7.0+cu128 already present)
pip install mujoco "imageio[ffmpeg]"
```

### The working demo — `mujoco_car/`  ✅ trains to 100% success
A differential-drive car learns to drive to a randomly-placed cube (PPO, ~400k steps, a few minutes).
```bash
export PYTHONPATH=$PWD MUJOCO_GL=egl
python mujoco_car/render_car.py out.png            # see the car + cube (still image)
python mujoco_car/train.py 400000 8                # train PPO (8 parallel envs) -> saves model
python mujoco_car/record.py after.mp4 5            # record 5 episodes of the trained policy
python mujoco_car/record.py before.mp4 2 --random  # random baseline for comparison
tensorboard --logdir mujoco_car/runs               # reward curves
```
Files: `car.xml` (MJCF model), `env.py` (Gymnasium env: 8-d obs, 2-d action, progress reward),
`train.py`, `record.py`. Result: random policy ≈ −20 reward / 0% reached → trained ≈ +23 / **100%**.

### Your real robot — `robot/` + `mujoco_car/robot_env.py`
Exported from Onshape with `onshape-to-robot` → MJCF (`robot/robot.xml` + meshes in `robot/assets/`).
4 wheels detected as joints (`wheel_fl/fr/rl/rr`). Skid-steer: action[0]=left pair, action[1]=right pair.
```bash
# one-time export (creds from onshapeAPI.env -> ONSHAPE_* env vars):
export ONSHAPE_API=https://cad.onshape.com
export ONSHAPE_ACCESS_KEY=...  ONSHAPE_SECRET_KEY=...     # from onshapeAPI.env (strip quotes!)
onshape-to-robot robot/                                  # reads robot/config.json
# train / watch:
python mujoco_car/train_robot.py 600000 8
python mujoco_car/continue_train_robot.py 1000000 8      # keep improving from checkpoint
PYTHONPATH=$PWD python mujoco_car/watch_robot.py          # live viewer (run on your desktop)
```
**Export gotchas we hit:** (1) `onshape-to-robot` needs **`ONSHAPE_API`** set too, not just the keys.
(2) Keys in `onshapeAPI.env` were **quote-wrapped** → strip quotes or the HMAC signature 401s.
**Model fixes after export (all in `robot/robot.xml`, hand-edited):**
1. `position`→**`velocity`** actuators with capped `forcerange` (drive by speed, stable).
2. Onshape's default density made it **308 kg → rescaled to ~33 kg** (mass+inertia together).
3. Wheels penetrate the floor at the origin → **spawn at the captured rest height (z≈0.20)**.
4. **Wheel collision meshes → cylinders** (radius 0.184) — mesh hulls are blobby and *hop*; cylinders
   roll cleanly. Overdamped contact (`solref="0.06 2.5"`) kills residual bounce.
5. **Mirrored axles** — left wheels spin about −Y, right about +Y. To drive forward, the two sides
   need **opposite motor signs**; `robot_env.py` negates the left side so action=[+1,+1]=forward.
   (Diagnosed by printing each wheel's world spin axis — both ±Y, confirming mirroring.)

### Obstacle avoidance + 2D LIDAR — `robot/train_scene_lidar.xml` + `mujoco_car/robot_env_lidar.py`
Robot learns to reach the cube **inside a walled cage while avoiding obstacles**, sensing them with a
simulated **2D LIDAR** (24 rays). Difficulty **auto-advances**: 1 obstacle → 2 → 3 → 1 *moving*
obstacle (each between robot and cube). Cube position stays privileged (LIDAR = avoidance only;
camera-to-find-the-cube is the next round). Observation = base 8 + 24 normalized ray ranges = **32-d**.
```bash
export PYTHONPATH=$PWD MUJOCO_GL=egl
python mujoco_car/test_lidar_env.py                      # proves the two LIDAR/obstacle requirements
python mujoco_car/train_curriculum.py 6000000           # auto-advancing curriculum, filmstrip per stage
PYTHONPATH=$PWD python mujoco_car/watch_robot_lidar.py --stage 3   # live viewer WITH lidar rays drawn
```
**LIDAR done right — `mj_ray` (the two failure modes the user flagged):**
1. **Rays passing *through* walls / hitting the *back* face.** `mj_ray` returns the **nearest** surface,
   but **`flg_static=0` silently excludes static geoms** (walls are static) — *that* is the classic
   "ray goes through the wall" bug. Fixes: call with **`flg_static=1`**; a **group-5 geomgroup mask**
   so only walls+obstacles are sensed (floor/robot/goal ignored); `bodyexclude=robot`. Origin kept
   outside geoms (terminate on collision) so the near hit is always the **front** face. Proven in
   `test_lidar_env.py` (front-face distance exact; `flg_static=0` → −1 = through the wall).
2. **Obstacles spaced / not levitating.** Boxes at **z = half-height** (bottom on floor), kinematic
   **mocap** bodies (stay put, don't fall), placed by **rejection sampling** with enforced clearances
   from robot/cube/each-other/walls. Checked over 200 resets in the test.
**Render/viewer note:** walls+obstacles live in **geom group 5** (hidden by default) → set
`scene_option.geomgroup[5]=1` (renderer) / `viewer.opt.geomgroup[5]=1` (viewer) to see them.

**Results (v1, deterministic eval):** warmup 88%, 1-obstacle 75%, 2-obstacle 25%, 3-obstacle 37%.
The steep **1→2 obstacle cliff** exposed the real issue: penalizing only *contact* isn't enough —
detouring around an obstacle temporarily increases distance-to-goal (losing progress reward), so the
greedy policy drove straight into obstacles. **Fix = a LIDAR proximity penalty** (`robot_env_lidar.py`
`step()`): `prox_pen = 5.0 * max(0, 0.22 - min_lidar_range)` — a smooth, growing cost as it nears any
obstacle/wall *before* contact, giving a gradient to steer away early. Also: smaller obstacles (0.6 m),
wider placement gaps (3.0 m), 2 M steps/stage, promotion threshold 0.60 rolling (≈78% deterministic).
The goal is rendered as a **Jenga tower** (visual only — its shape isn't in the observation).
**Live-watch workflow:** training writes `ppo_robot_lidar_latest` every rollout; the viewer reloads it
each episode so you watch the policy improve mid-training. Resume a run:
`python mujoco_car/train_curriculum.py N --resume --start-stage K`.
**Viewer gotcha:** never stop it with `pkill -f watch_robot_lidar` — that command's own text matches
the pattern and kills the launcher. Close the window or `kill <PID>`.

---

## Pose-based navigation + LIDAR + CAMERA — `robot/train_scene_nav.xml` + `mujoco_car/robot_env_nav.py`
Autonomous-driving style: the robot is **commanded to a goal pose** (A→B, given in the observation like
a GPS waypoint — a legitimate destination, not privileged obstacle info) and drives there from **anywhere
in a 30 m arena**, avoiding obstacles. **LIDAR = primary obstacle sense; onboard camera = auxiliary sense.**
Obstacles (up to 4): **60% near the goal, 40% random**; no moving obstacles. Goal rendered as a Jenga tower.
```bash
export PYTHONPATH=$PWD MUJOCO_GL=egl
python mujoco_car/test_nav_env.py                       # Dict-obs, raised-LIDAR front-face, 60/40 placement
python mujoco_car/train_nav_curriculum.py 6000000       # curriculum 0→1→2→3→4 obstacles (CNN, ~1-2 h)
PYTHONPATH=$PWD python mujoco_car/watch_nav.py --stage 3 # live viewer + LIDAR rays (cycle camera to robot_cam for POV)
MUJOCO_GL=egl python mujoco_car/record_nav.py out.mp4 4  # side-by-side [third-person+rays | camera POV] video
```
- **Observation = Dict** `{"vec": 32 (goal-pose + state + 24 LIDAR), "img": 64×64×3 camera}`; SB3
  **`MultiInputPolicy`** (CNN for image + MLP for vector); `VecNormalize(norm_obs_keys=["vec"])` (images
  stay uint8, normalized by the CNN). Trains on **CUDA**.
- **Camera pipeline verified:** per-subprocess EGL `Renderer` works inside `SubprocVecEnv` (each of 6
  workers renders its own 64×64 image). Env step is cheap (~0.8 ms; camera +0.3 ms).
- **LIDAR raised** to `z=0.42` (site local) so rays clear the chassis despite the tiny residual bounce.
- **Headless cv2 caveat:** the env ships `opencv-python-headless` (`imshow` disabled), so the live
  viewer uses the MuJoCo window and camera-POV comes from `record_nav.py` (imageio, no GUI needed).

---

## The learning curriculum
| Phase | Goal (what you learn) | Deliverable |
|---|---|---|
| **0. Setup** | The Isaac toolchain; why drivers/versions matter for sim-to-real | A stock Isaac Lab example trains end-to-end on the GPU |
| **1. RL foundations** | Env / Obs / Action / Reward / episode loop; how PPO learns. Read & retrain **Leatherback** | Explain every field in an `*_env_cfg.py`; retrain Leatherback |
| **2. Your robot** | CAD → URDF → USD asset pipeline (`onshape-to-robot` → Isaac URDF importer) | Your robot's USD loads; wheels actuate in the GUI |
| **3. "Go to cube" (state-based)** | Reward shaping & observation design; train without pixels first (privileged relative-to-cube vector) | Robot reliably drives to the cube from state |
| **4. Real sensors** | Add **2D RTX LIDAR**, then **camera**; perception in the obs space and the cost of pixels | Robot reaches the cube from camera + LIDAR only |
| **5. Sim-to-real readiness** | Domain randomization, latency, control-rate; **ROS 2 (Jazzy)** bridge mirroring real topics | Robust policy + ROS 2 graph matching the future real robot |
| **6. Real robot** *(future)* | Jetson + ROS 2 deploy; export policy (ONNX) | Roadmap only — not built yet |

> ⚠️ **Known Blackwell risk (Phase 4):** `TiledCamera` may hang on RTX 50-series (Isaac Lab #4951).
> Fallback is the standard `Camera` (slice RGBA→RGB `[..., :3]`) with fewer parallel envs. This is
> why the curriculum trains **state-based first** and only adds the camera once tiled rendering is
> confirmed on this card.

## Repo layout
```
personalRobot/
  README.md            # this file — the learning hub
  robot/               # Onshape export config + generated URDF/meshes/USD
    urdf/  usd/
  isaac_ext/           # Isaac Lab external task project (go_to_cube)
  scripts/             # train / play / export / ros2-bridge helpers
  docs/                # per-phase learning notes
  onshapeAPI.env       # Onshape API keys (gitignored — never commit)
```
