import numpy as np
from numba import jit
from simulation import Simulation


@jit(nopython=True)
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

        r = np.linalg.norm(pos[i])  # euclidian norm, distnace

        if r >= R:
            hat_n = pos[i] / r

            acc[i] -= k * (r - R) * hat_n
            # mass division is pointless, as heavier objects will
            # fly further out the boundary - which is not the behaviour I want


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


class VelVerletSim(Simulation):
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
        super().__init__(N, dt, dims, R, pos_func, vel_func, mass_func)

        _calculate_acc(self.pos, self.acc, self.masses, self.R)

    def step(self):
        self.pos, self.vel = _step(
            self.pos, self.vel, self.acc, self.masses, self.dt, self.R
        )


# TODO: 1) Add different interesting initialisation setups
# TODO: 2) decide between Plummer-style gravitational softening or clumping method that was before
# TODO: 3) Rewrite default initial setup, as this one is still chaotic
# TODO: 4) Add docstring for NBodySim init method
