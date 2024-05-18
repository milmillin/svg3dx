import math
import numpy as np

from svg3dx import Mesh, Camera, render, DiffuseShader

f = math.sqrt(2.0) / 2.0
verts = np.array([(0, -1, 0), (-f, 0, f), (f, 0, f), (f, 0, -f), (-f, 0, -f), (0, 1, 0)], dtype=np.float32) * 15
triangles = np.array(
    [
        (0, 2, 1),
        (0, 3, 2),
        (0, 4, 3),
        (0, 1, 4),
        (5, 1, 2),
        (5, 2, 3),
        (5, 3, 4),
        (5, 4, 1),
    ]
)

render(Mesh(verts, triangles), Camera.create_perspective(100, 100, eye=[50, 40, 120], near=10, far=200), shader=DiffuseShader()).save_svg(
    "test.svg"
)
render(Mesh(verts, triangles), Camera.create_orthogonal(100, 100, eye=[50, 40, 120], dist=20, near=10, far=2000), shader=DiffuseShader()).save_svg(
    "test_orth.svg"
)
