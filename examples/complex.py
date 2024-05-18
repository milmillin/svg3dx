import math
import numpy as np

from svg3dx import Mesh, Camera, render, DiffuseShader, order_faces


data = np.load("complex.npz")
V = data["V"]
F = data["F"]

V -= V.min(axis=0)
V *= 2 / V.max()
V -= V.max(axis=0) / 2

print(V.min(axis=0), V.max(axis=0))

order = np.arange(len(F.T))
cam = Camera.create_orthogonal(200, 200, eye=[1, 1, 1], dist=1)
mesh = Mesh(V, F.T)
render(mesh, cam, DiffuseShader(), face_order=order).save_svg("complex.svg")
