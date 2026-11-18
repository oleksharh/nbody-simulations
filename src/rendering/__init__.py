from .particle_state import ParticleState
from .particle_renderer import ParticleRenderer
from .renderer import SimWorker

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulations import Simulation


def render(sim: Simulation):
    state = ParticleState()

    state.publish(sim.pos, sim.masses, sim.vel)

    worker = SimWorker(sim, state)
    renderer = ParticleRenderer(state, dims=sim.dims)

    worker.start()

    try:
        renderer.run()
    finally:
        worker.stop()
        worker.join()
        renderer.close()


__all__ = ["ParticleState", "ParticleRenderer", "SimWorker", "render"]
