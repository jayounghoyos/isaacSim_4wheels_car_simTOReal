"""Render a single still of the car + target cube (offscreen, EGL/GPU). Saves a PNG."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
import mujoco
import imageio

xml_path = os.path.join(os.path.dirname(__file__), "car.xml")
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/car_render.png"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=720, width=1280)
cam = mujoco.MjvCamera()
cam.lookat[:] = [1.3, 0.0, 0.0]
cam.distance = 5.5
cam.azimuth = 110
cam.elevation = -25
renderer.update_scene(data, camera=cam)
img = renderer.render()
imageio.imwrite(out, img)
renderer.close()
print("CAR_RENDER_OK ->", out)
