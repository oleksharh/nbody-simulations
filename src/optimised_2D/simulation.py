from numba import jit
import numpy as np

import matplotlib.pyplot as plt


@jit(nopython=True)
def _step(pos, vel, acc, mass, dt):

    for i in range(len(pos)):
        vel[i] += 0.5 * (acc[i] + acc[i]) * dt  # acc1

    acc[:] = 0.0

    for i in range(len(pos)):
        pos_i = pos[i]
        mass_i = mass[i]
        for j in range(i + 1, len(pos)):
            pos_j = pos[j]
            mass_j = mass[j]

            r_vec = pos_j - pos_i
            r_sq = r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1]

            if r_sq < 0.001:
                r_sq = 0.001

            r = np.sqrt(r_sq)  # displacement distance

            factor = r_vec / (r_sq * r)

            acc[i] += mass_j * factor * dt
            acc[j] -= mass_i * factor * dt

    for i in range(len(pos)):
        pos[i] += vel[i] * dt + 0.5 * acc[i] * dt**2

    for i in range(len(pos)):
        vel[i] += 0.5 * acc[i] * dt  # acc2

    return pos, vel


class Grid2D:
    def __init__(self, N: int, dt: float) -> None:
        self.dt = dt
        self._initialise_spread_objects(N)

    def _initialise_spread_objects(self, N: int):
        # will be in (x, y) format within these vectors
        self.pos = np.random.uniform(-15.0, 15.0, size=(N, 2)).astype(np.float64)
        self.vel = np.random.uniform(-10.0, 10.0, size=(N, 2)).astype(np.float64)
        self.acc = np.zeros((N, 2), dtype=np.float64)
        self.mass = np.random.uniform(100.0, 10000.0, size=N).astype(np.float64)

    def step(self):
        self.pos, self.vel = _step(self.pos, self.vel, self.acc, self.mass, self.dt)
