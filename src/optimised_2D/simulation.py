from numba import jit
import numpy as np
from typing import Callable


@jit(nopython=True, cache=True)
def _calculate_acc(pos, acc, mass, R):
    acc[:] = 0.0

    for i in range(len(pos)):
        pos_i = pos[i]
        mass_i = mass[i]
        for j in range(i + 1, len(pos)):
            pos_j = pos[j]
            mass_j = mass[j]

            r_vec = pos_j - pos_i
            r_sq = np.dot(r_vec, r_vec)

            # if r_sq < 0.001:
            #     r_sq = 0.001

            # r = np.sqrt(r_sq)  # displacement distance


            epsilon = 0.5
            # factor = r_vec / (r_sq * r + epsilon)
            factor = r_vec / (r_sq + epsilon**2) ** 1.5

            acc[i] += mass_j * factor
            acc[j] -= mass_i * factor

    # Adding soft boundary
    for i in range(len(pos)):
        k = 1  # stifness coefficient in F_boundry = -k(r- R)*n_vector, where n_vector is pos_vector / r

        r_sq = np.dot(pos[i], pos[i])
        r = np.sqrt(r_sq)

        if r >= R:
            hat_n = pos[i] / r

            acc[i] -= k * (r - R) * hat_n
            # mass division is pointless, as heavier objects will
            # fly further out the boundary - which is not the behaviour I want


@jit(nopython=True, cache=True)
def _step(pos, vel, acc, mass, dt, R):

    # Calculating positons at t + dt
    for i in range(len(pos)):
        pos[i] += vel[i] * dt + 0.5 * acc[i] * dt**2

    # Velocitites at t + dt/2
    for i in range(len(pos)):
        vel[i] += 0.5 * acc[i] * dt

    # Calculating accelerations at t + dt
    _calculate_acc(pos, acc, mass, R)

    # Velocitites at t + dt/2
    for i in range(len(pos)):
        vel[i] += 0.5 * acc[i] * dt

    return pos, vel


class NBodySim:
    def __init__(
        self,
        N: int,
        dt: float,
        dims: int,
        R: int,
        pos_func=None,
        vel_func=None,
        mass_func=None,
    ) -> None:
        # R is a soft boundary radius
        self.N = N
        self.dt = dt
        self.dims = dims
        self.R = R

        if dims not in (2, 3):
            raise ValueError("dims argument must be 2 or 3")

        self.pos = pos_func(self) if pos_func else self._default_positions()
        self.mass = mass_func(self) if mass_func else self._default_masses()
        self.vel = vel_func(self) if vel_func else self._default_velocities()

        self.acc = np.zeros_like(self.pos) # same as np.zeros((self.N, self.dims), dtype=np.float64)

        _calculate_acc(self.pos, self.acc, self.mass, self.R)

    def _default_positions(self):
        if self.dims == 2:
            return self.circle_bounding()

        return self.sphere_bounding()

    def _default_velocities(self):
        if self.dims != 2:
            return np.zeros(
                (self.N, self.dims), dtype=np.float64
            )

        x = self.pos[:, 0]
        y = self.pos[:, 1]
        radius = np.hypot(x, y)

        rotation_scale = 0.8

        # Approximation for a uniform surface-density disc.
        enclosed_mass = self.mass.sum() * (radius / self.R) ** 2

        safe_radius = np.maximum(radius, 1e-8)
        speed = rotation_scale * np.sqrt(
            enclosed_mass / safe_radius
        )

        vx = -y / safe_radius * speed
        vy = x / safe_radius * speed

        return np.column_stack((vx, vy)).astype(np.float64)

    def _default_masses(self):
        return np.full(shape=self.N, fill_value=1.0)


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

    def step(self):
        self.pos, self.vel = _step(
            self.pos, self.vel, self.acc, self.mass, self.dt, self.R
        )


# TODO: 1) Add different interesting initialisation setups
# TODO: 2) decide between Plummer-style gravitational softening or clumping method that was before
# TODO: 3) Rewrite default initial setup, as this one is still chaotic
# TODO: 4) Add docstring for NBodySim init method
# TODO: 5) Implement Barnes Hut algorithm
