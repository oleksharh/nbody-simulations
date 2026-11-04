import os
os.environ["NUMBA_NUM_THREADS"] = "24"

from vel_verlet import VelVerletSim
from barnes_hut import BarnesHutSim
from quad_tree import FlatBHutSim
from renderer import run_nbody_simulation

# nbody_sim = VelVerletSim(N=100, dt=0.01, dims=3, R=10)

# nbody_sim = BarnesHutSim(N=50, dt=0.01, dims=2, R=50, max_depth=10)

nbody_sim = FlatBHutSim(N=5_000, dt=0.01, dims=2, R=3, theta=0.7, max_depth=10, soft_bounding=False)

run_nbody_simulation(
    nbody_sim,
    render_decoupl_rng=1,
    bounds=True,
    canvas_size=(800, 600),
)
