from simulation import Simulation


class BarnesHutSim(Simulation):
    def __init__(self, N: int, dt: float, dims: int, R: int) -> None:
        super().__init__(N, dt, dims, R)

        