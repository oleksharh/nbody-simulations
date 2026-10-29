from numba import jit
import numpy as np

import matplotlib.pyplot as plt


@jit(nopython=True)
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


@jit(nopython=True)
def _calculate_acc(pos, acc, mass, R):
    acc[:] = 0.0

    epsilon = 0.1

    for i in range(len(pos)):
        pos_i = pos[i]
        mass_i = mass[i]
        for j in range(i + 1, len(pos)):
            pos_j = pos[j]
            mass_j = mass[j]

            r_vec = pos_j - pos_i
            r_sq = np.dot(r_vec, r_vec)

            if r_sq < 0.001:
                r_sq = 0.001

            r = np.sqrt(r_sq)  # displacement distance

            factor = r_vec / (r_sq * r + epsilon)

            acc[i] += mass_j * factor
            acc[j] -= mass_i * factor

    # Adding soft boundary
    for i in range(len(pos)):
        k = 1  # stifness coefficient in Fboundry = -k(r- R)*n_vector, where n_vector is pos_vector / r

        r_sq = np.dot(pos[i], pos[i])
        r = np.sqrt(r_sq)

        if r >= R:
            hat_n = pos[i] / r

            acc[i] -= k * (r - R) * hat_n


class NBodySim:
    def __init__(self, N: int, dt: float, dims: int, R: int) -> None:
        # R is a soft boundary radius

        self.dt = dt
        self.dims = dims
        self.R = R

        self._initialise_spread_objects(N)
        _calculate_acc(self.pos, self.acc, self.mass, self.R)

    def _initialise_spread_objects(self, N: int):
        # will be in (x, y) format within these vectors
        self.pos = np.random.uniform(-15.0, 15.0, size=(N, self.dims)).astype(
            np.float64
        )
        self.vel = np.random.uniform(0.0, 0.0, size=(N, self.dims)).astype(np.float64)
        self.acc = np.zeros((N, self.dims), dtype=np.float64)
        self.mass = np.random.uniform(10.0, 10.0, size=N).astype(np.float64)

        # rnd_black_hole = np.random.randint(0, 3)

        # self.pos[rnd_black_hole] = np.full(self.dims, 5.0)
        # self.mass[rnd_black_hole] = 10_000

    def step(self):
        self.pos, self.vel = _step(
            self.pos, self.vel, self.acc, self.mass, self.dt, self.R
        )


# TODO: 1) Add different interesting initialisation setups
