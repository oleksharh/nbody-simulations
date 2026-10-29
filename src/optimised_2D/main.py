from simulation import NBodySim
from renderer_vispy import run_nbody_simulation

nbody_sim = NBodySim(N=100, dt=0.001, dims=2, R=50)

# nbody_sim.initialisation_setup(pos, vel, acc, mass) this will be added functionality

run_nbody_simulation(nbody_sim)

