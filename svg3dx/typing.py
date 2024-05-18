from typing_extensions import Union, Annotated
import numpy as np
from beartype.vale import Is
from beartype.typing import Tuple, List


IsNpFloat = Is[lambda x: np.issubdtype(x.dtype, np.floating)]
IsNpInt = Is[lambda x: np.issubdtype(x.dtype, np.integer)]

NpVec3f = Annotated[np.ndarray, Is[lambda x: x.shape == (3,)], IsNpFloat]
NpVec3i = Annotated[np.ndarray, Is[lambda x: x.shape == (3,)], IsNpInt]
NpVecXi = Annotated[np.ndarray, Is[lambda x: x.ndim == 1], IsNpInt]

NpMatX3f = Annotated[np.ndarray, Is[lambda x: x.ndim == 2 and x.shape[-1] == 3], IsNpFloat]
NpMatX3i = Annotated[np.ndarray, Is[lambda x: x.ndim == 2 and x.shape[-1] == 3], IsNpInt]
NpMatX2f = Annotated[np.ndarray, Is[lambda x: x.ndim == 2 and x.shape[-1] == 2], IsNpFloat]
NpMatX2i = Annotated[np.ndarray, Is[lambda x: x.ndim == 2 and x.shape[-1] == 2], IsNpInt]

NpMat44f = Annotated[np.ndarray, Is[lambda x: x.shape == (4, 4)], IsNpFloat]

Number = Union[int, float]

InVec3f = Union[Tuple[Number, Number, Number], List[Number], NpVec3f]
InVec3i = Union[Tuple[int, int, int], List[int], NpVec3i]
