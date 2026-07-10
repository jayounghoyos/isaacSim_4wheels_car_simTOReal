# Phase 1 — Anatomy of an Isaac Lab RL Task (Leatherback)

Study notes for `references/Leatherback/.../tasks/direct/leatherback/leatherback_env.py`.
This is a **Direct RL** environment: you implement the env logic directly in a class that
subclasses `DirectRLEnv`. (The other style, **Manager-based**, composes the env from reusable
"manager" terms — we'll meet it later; Direct is easier to read first.)

## The method contract (what every DirectRLEnv must implement)
The Isaac Lab engine calls these for you, every step, for all parallel envs at once (tensors of
shape `[num_envs, ...]`):

| Method | Called when | Job |
|---|---|---|
| `__init__` | once | allocate buffers, read config, find joint indices |
| `_setup_scene` | once | spawn ground, robot, lights, markers; clone N envs |
| `_pre_physics_step(actions)` | each RL step | turn raw policy actions → joint commands (scale/clip) |
| `_apply_action` | each physics substep | write the commands to the sim |
| `_get_observations` | each RL step | build the tensor the policy sees → `{"policy": obs}` |
| `_get_rewards` | each RL step | compute the scalar reward per env |
| `_get_dones` | each RL step | return `(terminated, truncated)` flags |
| `_reset_idx(env_ids)` | on done | re-randomize the robots/targets that finished |

The config class (`LeatherbackEnvCfg`, `@configclass`) declares the **contract numbers**:
`action_space=2`, `observation_space=8`, `decimation=4` (1 RL step = 4 physics steps),
`episode_length_s=20`, `scene.num_envs=4096` (4096 robots train in parallel on the GPU).

---

## Observations — `_get_observations` (lines 113–143)
Builds the **8-dim** vector (matches `observation_space=8`):

| # | Value | Code | Why |
|---|---|---|---|
| 1 | distance to target | `torch.norm(position_error_vector)` | "how far" |
| 2–3 | `cos`, `sin` of heading error | lines 129–130 | "which way to turn", **wrap-safe** |
| 4 | forward velocity (body x) | `root_lin_vel_b[:,0]` | self-relative speed |
| 5 | lateral velocity (body y) | `root_lin_vel_b[:,1]` | sideways drift |
| 6 | yaw rate | `root_ang_vel_w[:,2]` | how fast turning |
| 7 | last throttle | `_throttle_state[:,0]` | memory of own action |
| 8 | last steering | `_steering_state[:,0]` | memory of own action |

### Design principles to internalize
1. **Egocentric / relative, never absolute.** The obs is the vector *to the target* and velocities
   *in the robot's body frame* — never world XY. This is *why* the trained policy works for a target
   placed anywhere: it only ever learned "reduce this relative error."
2. **Encode angles as `(cos, sin)`, not the raw angle.** A raw angle jumps from +π to −π — a
   discontinuity a neural net handles badly. `(cos, sin)` is a smooth point on the unit circle.
3. **Body-frame velocities** (`_b` suffix) are orientation-invariant — "am I moving forward?" means
   the same thing regardless of which way the robot faces in the world.
4. **Include the last action** (7,8): gives the policy temporal context → smoother control.
5. **Guard against NaN** and return `{"policy": obs}` (the key the actor network reads; a `"critic"`
   key could provide privileged info in asymmetric actor-critic).

---

## Rewards — `_get_rewards` (lines 145–166)
`composite = progress·1.0 + heading·0.05 + goal_bonus·10.0`
- **progress** = `prev_distance − curr_distance` → dense, every-step "got closer" signal (the engine).
- **heading** = `exp(−|heading_err| / 0.25)` → bell curve, 1.0 when facing the target.
- **goal_bonus** = `+10` each time within 0.15 m of the current waypoint.
Tuning these three weights *is* reward shaping.

---

## Actions — `_pre_physics_step` / `_apply_action` (lines 95–111)
2 actions → Leatherback's **Ackermann** model:
- `actions[:,0]` = throttle → broadcast to **4 drive wheels**, scaled ×10, clipped to ±50, set as
  **joint velocity** target.
- `actions[:,1]` = steering → broadcast to **2 front knuckles**, scaled ×0.1, clipped ±0.75, set as
  **joint position** target.

> 🔧 **This is the main thing your differential-drive robot changes.** You have no steering joints.
> Instead map 2 actions → left-pair / right-pair **wheel velocities**:
> `v_left, v_right = f(actions)`; set velocity targets on the left wheels and right wheels.
> Everything else (observations, rewards, reset) transfers almost unchanged. Target = the **cube**.

---

## Reset / randomization — `_reset_idx` (lines 172–221)
On episode end: place the robot at its default pose + the env origin (with a small random lateral
offset and heading), then lay out the targets and recompute the initial errors. **Randomizing start
and target each episode is what forces the policy to *generalize* instead of memorizing one path.**
For go-to-cube: randomize a single cube position in a region around the robot.

## Running it (after install verifies)
```bash
conda activate isaaclab
# train (headless, fast):
python references/Leatherback/scripts/skrl/train.py --task <Leatherback-task-id> --headless
# watch a trained policy:
python references/Leatherback/scripts/skrl/play.py  --task <Leatherback-task-id>
```
(We'll confirm the exact task id from `scripts/list_envs.py` once Isaac Lab imports cleanly.)
