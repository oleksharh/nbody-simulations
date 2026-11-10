import os
os.environ["NUMBA_NUM_THREADS"] = "24"

from simulations.vel_verlet import VelVerletSim
from simulations.bhut import BarnesHutSim
from simulations.bhut_flat import FlatBHut2DSim
from renderer import run_nbody_simulation

# nbody_sim = VelVerletSim(N=100, dt=0.01, dims=3, R=10)

# nbody_sim = BarnesHutSim(N=50, dt=0.01, dims=2, R=50, max_depth=10)

nbody_sim = FlatBHut2DSim(N=5_000, dt=0.01, dims=2, R=3, theta=0.7, max_depth=10, soft_bounding=False)

run_nbody_simulation(
    nbody_sim,
    render_decoupl_rng=1,
    bounds=True,
    canvas_size=(800, 600),
)

# TODO: Need to add testing pipeline of some sort to check if Sim classes conform to the abstract class,
# whether renderer works and doesn't throw and error