from simulation import NBodySim
from renderer_vispy import run_nbody_simulation

nbody_sim = NBodySim(N=200, dt=0.01, dims=2, R=50)

run_nbody_simulation(nbody_sim, render_decoupl_rng=7)

