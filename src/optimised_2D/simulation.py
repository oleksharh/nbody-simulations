from numba import jit
import numpy as np

import matplotlib.pyplot as plt


@jit(nopython=True)
def _step(pos, vel, acc, mass, dt):

    # Calculating positons at t + dt
    for i in range(len(pos)):
        pos[i] += vel[i] * dt + 0.5 * acc[i] * dt**2

    # Velocitites at t + dt/2
    for i in range(len(pos)):
        vel[i] += 0.5 * acc[i] * dt

    # Calculating accelerations at t + dt
    _calculate_acc(pos, acc, mass)

    # Velocitites at t + dt/2
    for i in range(len(pos)):
        vel[i] += 0.5 * acc[i] * dt

    return pos, vel


@jit(nopython=True)
def _calculate_acc(pos, acc, mass):
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

            acc[i] += mass_j * factor
            acc[j] -= mass_i * factor


class Grid2D:
    def __init__(self, N: int, dt: float) -> None:
        self.dt = dt

        self._initialise_spread_objects(N)
        _calculate_acc(self.pos, self.acc, self.mass)

    def _initialise_spread_objects(self, N: int):
        # will be in (x, y) format within these vectors
        self.pos = np.random.uniform(-15.0, 15.0, size=(N, 2)).astype(np.float64)
        self.vel = np.random.uniform(-10.0, 10.0, size=(N, 2)).astype(np.float64)
        self.acc = np.zeros((N, 2), dtype=np.float64)
        self.mass = np.random.uniform(100.0, 500.0, size=N).astype(np.float64)

        rnd_black_hole = np.random.randint(0, N)

        self.pos[rnd_black_hole] = [110, 110]
        self.mass[rnd_black_hole] = 1_000_000



        # TEST N = 2 INITILISATION
        # self.pos = np.zeros((N, 2), dtype=np.float64)
        # self.vel = np.zeros((N, 2), dtype=np.float64)
        # self.acc = np.zeros((N, 2), dtype=np.float64)
        # self.mass = np.zeros(N, dtype=np.float64)

        # self.pos[0] = [-1, 0]
        # self.pos[1] = [1, 0]
        # self.vel[0] = [0, 11]
        # self.vel[1] = [0, -11]

        # self.mass[0], self.mass[1] = 500, 500

    def step(self):
        self.pos, self.vel = _step(self.pos, self.vel, self.acc, self.mass, self.dt)
