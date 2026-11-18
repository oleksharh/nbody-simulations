import os

# Sets 75% of available threads to be utilised by the force solvers
cpu_count = os.cpu_count() or 1
worker_count = max(1, int(cpu_count * 0.75))

os.environ["NUMBA_NUM_THREADS"] = str(worker_count)


# os.environ["NUMBA_NUM_THREADS"] = "Set your own value instead"
#################################################################

from simulations import DirectSum, FlatBHut2DSim
from rendering import render

# sim = DirectSum(N=2000, dt=0.01, dims=3, R=100)

sim = FlatBHut2DSim(
    N=5_000,
    dt=0.001,
    dims=2,
    R=5,
    theta=0.7,
    max_depth=7,
    soft_bounding=False
)

render(sim)