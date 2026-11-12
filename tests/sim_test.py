import numpy as np
import pytest
from simulations.base_sim import Simulation
from simulations.vel_verlet import VelVerletSim
from simulations.bhut_flat import build_tree, calculate_acc, FlatBHut2DSim


@pytest.fixture(
    params=[
        lambda: VelVerletSim(N=100, dt=0.01, dims=2, R=5),
        lambda: VelVerletSim(N=100, dt=0.01, dims=3, R=5),
        lambda: FlatBHut2DSim(
            N=10000, dt=0.01, dims=2, R=3, theta=0.7, max_depth=10, soft_bounding=True
        ),
    ],
    ids=["vel_verlet_2D", "vel_verlet_3D", "bhut_flat"],
)
def sim(request):

    return request.param()


def test_sim_step(sim: Simulation):
    """
    Checks whether step function works correctly, doesn't account for overtime error
    """
    old_pos = sim.pos.copy()
    old_vel = sim.vel.copy()
    old_acc = sim.acc.copy()
    dt = sim.dt

    new_pos = old_pos + old_vel * dt + 0.5 * old_acc * dt**2

    sim.step()

    np.testing.assert_allclose(new_pos, sim.pos)


# TODO: Custom generated values here, constant with seed, and passed to bhut and vel verlet
# algorithm specific funcs to check for error overtime
