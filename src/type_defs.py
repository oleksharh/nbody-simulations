import numpy as np
import numpy.typing as npt

type Float64Vec1D = npt.NDArray[np.float64]  # shape (N,)
type Float64Vec2D = npt.NDArray[np.float64]  # shape (N, dims)
type Float64Vec3D = npt.NDArray[np.float64]  # shape (N, 3)


type Int8Vec1D = npt.NDArray[np.int8]
type Int64Vec1D = npt.NDArray[np.int64]
type Int64Vec4D = npt.NDArray[np.int64]

type BoolVec1D = npt.NDArray[np.bool]