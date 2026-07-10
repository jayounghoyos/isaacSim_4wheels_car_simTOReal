"""Minimal Isaac Sim / Isaac Lab smoke test.

Launches the kit app headless, creates a physics sim, steps it, and runs a GPU
tensor op inside the running app. If this prints SMOKE_OK, Isaac Sim 6.0.1 +
Isaac Lab + torch 2.7.0+cu128 are functionally working together on this machine,
regardless of pip's version-pin warnings.
"""

from isaaclab.app import AppLauncher

# Headless launch (no GUI). EULA must be accepted via OMNI_KIT_ACCEPT_EULA=YES.
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))
sim.reset()

for _ in range(30):
    sim.step()

# Prove torch GPU works inside the running app context too.
x = torch.randn(512, 512, device="cuda")
val = float((x @ x).sum().item())

print(f"SMOKE_OK stepped_sim=30 gpu_matmul_sum={val:.2f} torch={torch.__version__}")

simulation_app.close()
