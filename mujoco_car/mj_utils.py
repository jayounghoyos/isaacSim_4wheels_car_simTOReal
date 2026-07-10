"""Shared MuJoCo helpers for the navigation env (kept here so no active file depends on legacy modules)."""
import numpy as np


def _yaw_from_quat(q):
    """Yaw angle (rad) from a MuJoCo [w, x, y, z] quaternion."""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
