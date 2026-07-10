"""GUI render test — does Isaac Sim's RTX renderer initialize on this Blackwell GPU?

Opens the Isaac Sim viewport (non-headless), builds a trivial scene (ground + light + cube),
resets (this is where the RTX renderer initializes — the step that hangs/crashes on 595+Blackwell),
and if it survives, renders for a while so a window is visible. Prints GUI_RENDERER_OK on success.
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)  # <-- GUI
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))

# Something to look at
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))
cube = sim_utils.CuboidCfg(
    size=(0.5, 0.5, 0.5),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.1)),
)
cube.func("/World/Cube", cube, translation=(0.0, 0.0, 0.5))

sim.reset()
print("GUI_RENDERER_OK: RTX renderer initialized and scene reset", flush=True)

# Render a while so the window is visible (and we can confirm it doesn't hang mid-run)
for i in range(1800):  # ~ up to a minute of frames
    sim.step()
    if i % 300 == 0:
        print(f"GUI_RENDER_FRAME {i}", flush=True)

print("GUI_RENDER_DONE", flush=True)
simulation_app.close()
