from beartype.claw import beartype_this_package

beartype_this_package()

from .core import Shader, ConstantShader, DiffuseShader, render, Camera, Mesh, order_faces
