import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

from type_defs import Float64Vec1D, Float64Vec2D, Float64Vec3D

PosFunc = Callable[["Simulation"], Union[Float64Vec2D, Float64Vec3D]]
VelFunc = Callable[["Simulation"], Union[Float64Vec2D, Float64Vec3D]]
MassFunc = Callable[["Simulation"], Float64Vec1D]


class Simulation(ABC):
    """
    Parameters
    ----------
    N : int
    dt : float
    dims : int
    R : int
    pos : Union[Float64Vec2D, Float64Vec3D]
    vel : Union[Float64Vec2D, Float64Vec3D]
    acc : Union[Float64Vec2D, Float64Vec3D]
    masses : Float64Vec1D
    """

    def __init__(
        self,
        N: int,
        dt: float,
        dims: int,
        R: int,
        pos_func: Optional[PosFunc] = None,
        vel_func: Optional[VelFunc] = None,
        mass_func: Optional[MassFunc] = None,
    ) -> None:

        if dims not in (2, 3):
            raise ValueError("dims argument must be 2 or 3")
        self.N = N
        self.dt = dt
        self.dims = dims
        self.R = R

        self.pos: Union[Float64Vec2D, Float64Vec3D] = (
            pos_func(self) if pos_func else self._default_positions()
        )
        self.masses: Float64Vec1D = (
            mass_func(self) if mass_func else self._default_masses()
        )
        self.vel: Union[Float64Vec2D, Float64Vec3D] = (
            vel_func(self) if vel_func else self._default_velocities()
        )

        self.acc: Union[Float64Vec2D, Float64Vec3D] = np.zeros_like(
            self.pos
        )  # same as np.zeros((self.N, self.dims), dtype=np.float64)

    def _default_positions(self):
        if self.dims == 2:
            return self.circle_bounding()

        return self.sphere_bounding()

    def _default_velocities(self):
        if self.dims != 2:
            return np.zeros((self.N, self.dims), dtype=np.float64)

        x = self.pos[:, 0]
        y = self.pos[:, 1]
        radius = np.hypot(x, y)

        rotation_scale = 0.8

        # Approximation for a uniform surface-density disc.
        enclosed_mass = self.masses.sum() * (radius / self.R) ** 2

        safe_radius = np.maximum(radius, 1e-8)
        speed = rotation_scale * np.sqrt(enclosed_mass / safe_radius)

        vx = -y / safe_radius * speed
        vy = x / safe_radius * speed

        return np.column_stack((vx, vy)).astype(np.float64)

    def _default_masses(self):
        return np.full(shape=self.N, fill_value=1.0, dtype=np.float64)

    def circle_bounding(self):
        N = self.N
        theta = np.random.uniform(0, 2 * np.pi, size=N)

        r = self.R * np.sqrt(np.random.uniform(0, 1, size=N))

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return np.column_stack((x, y)).astype(np.float64)

    def sphere_bounding(self):
        N = self.N
        vec = np.random.randn(3, N)
        vec /= np.linalg.norm(vec, axis=0)  # fit to (0, 1) range

        u = np.random.uniform(0, 1, N)
        r = self.R * (u ** (1 / 3))

        point_pos = vec * r
        return point_pos.T.astype(np.float64)

    @abstractmethod
    def step(self) -> None:
        """
        Step method, updates self.pos and self.vel attributes
        every timestep.
        """
        pass
