"""Export the trained SB3 nav policy to a portable ONNX file (runs on the Jetson, no MuJoCo/SB3).

Bakes the VecNormalize stats into the graph, so the robot feeds RAW observations:
  inputs:  img  (1,3,64,64) float32 in [0,255]   (camera, CHW)
           vec  (1,32)      float32 RAW (unnormalized: [dist,cos,sin,vx_b,vy_b,yaw,lastL,lastR]+24 lidar)
  output:  action (1,2) float32 in [-1,1]  (left_pair, right_pair wheel command)

Usage: PYTHONPATH=$PWD python mujoco_car/export_onnx.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from mujoco_car.robot_env_nav import RobotNavEnv

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "policy_nav.onnx")


class OnnxPolicy(nn.Module):
    """Raw obs -> normalize vec (baked stats) -> SB3 policy -> deterministic action, clipped [-1,1]."""
    def __init__(self, sb3_policy, vec_mean, vec_var, clip=10.0, eps=1e-8):
        super().__init__()
        self.policy = sb3_policy
        self.register_buffer("mean", torch.as_tensor(vec_mean, dtype=torch.float32))
        self.register_buffer("std", torch.sqrt(torch.as_tensor(vec_var, dtype=torch.float32) + eps))
        self.clip = clip

    def forward(self, img, vec):
        vec_n = torch.clamp((vec - self.mean) / self.std, -self.clip, self.clip)
        obs = {"img": img, "vec": vec_n}
        # deterministic action = distribution mean (DiagGaussian) = action_net(latent_pi)
        features = self.policy.extract_features(obs)
        latent_pi, _ = self.policy.mlp_extractor(features)
        mean_actions = self.policy.action_net(latent_pi)
        return torch.clamp(mean_actions, -1.0, 1.0)


def main():
    model = PPO.load(os.path.join(HERE, "ppo_robot_nav"), device="cpu")
    vec = VecNormalize.load(os.path.join(HERE, "vecnormalize_robot_nav.pkl"),
                            DummyVecEnv([lambda: RobotNavEnv(n_obstacles=0)]))
    rms = vec.obs_rms["vec"]
    wrapper = OnnxPolicy(model.policy, rms.mean, rms.var).eval()

    dummy_img = torch.zeros(1, 3, 64, 64, dtype=torch.float32)
    dummy_vec = torch.zeros(1, 32, dtype=torch.float32)

    torch.onnx.export(
        wrapper, (dummy_img, dummy_vec), OUT,
        input_names=["img", "vec"], output_names=["action"],
        dynamic_axes={"img": {0: "batch"}, "vec": {0: "batch"}, "action": {0: "batch"}},
        opset_version=13,   # opset 13: supported by onnxruntime 1.10 on the Jetson Nano (JetPack 4.6/py3.6)
    )
    print(f"ONNX_SAVED -> {OUT}")

    # --- verify: ONNX output matches SB3 deterministic action on real observations ---
    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    env = RobotNavEnv(n_obstacles=2)
    max_err = 0.0
    for ep in range(5):
        obs, _ = env.reset(seed=900 + ep)
        for _ in range(30):
            # SB3 path (normalizes internally via VecNormalize)
            sb3_a, _ = model.predict(vec.normalize_obs(obs), deterministic=True)
            # ONNX path (raw obs; wrapper normalizes)
            img = obs["img"].transpose(2, 0, 1)[None].astype(np.float32)   # HWC uint8 -> 1,C,H,W float
            v = obs["vec"][None].astype(np.float32)
            onnx_a = sess.run(["action"], {"img": img, "vec": v})[0][0]
            max_err = max(max_err, float(np.abs(sb3_a - onnx_a).max()))
            obs, r, term, trunc, info = env.step(sb3_a)
            if term or trunc:
                obs, _ = env.reset(seed=900 + ep)
    env.close()
    print(f"VERIFY max|SB3 - ONNX| action diff = {max_err:.6f}  ({'OK' if max_err < 1e-4 else 'MISMATCH'})")

    # also dump the normalization stats as JSON for the robot-side code reference
    import json
    with open(os.path.join(HERE, "policy_nav_norm.json"), "w") as f:
        json.dump({"vec_mean": rms.mean.tolist(), "vec_var": rms.var.tolist(),
                   "clip": 10.0, "wheel_vel_scale": RobotNavEnv().wheel_vel_scale}, f, indent=2)
    print("saved policy_nav_norm.json (normalization stats for reference)")


if __name__ == "__main__":
    main()
