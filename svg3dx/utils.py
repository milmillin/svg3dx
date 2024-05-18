import numpy as np

from .typing import InVec3f, NpMatX3f

def rgb2hex(x: InVec3f) -> str:
    v = (np.asarray(x) * 255).round().astype(np.uint8)
    return f"#{v[0]:02x}{v[1]:02x}{v[2]:02x}"

def normalize(x: NpMatX3f) -> NpMatX3f:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)