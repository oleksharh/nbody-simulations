from simulation import NBodySim
from barnes_hut import BarnesHutSim
from renderer import run_nbody_simulation

nbody_sim = NBodySim(N=2000, dt=0.01, dims=3, R=10)

# nbody_sim = BarnesHutSim(N=50, dt=0.01, dims=2, R=50, max_depth=10)


run_nbody_simulation(
    nbody_sim,
    render_decoupl_rng=1,
    bounds=True,
    canvas_size=(800, 600),
)
