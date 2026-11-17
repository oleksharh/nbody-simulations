
import threading


class SimWorker(threading.Thread):

    def __init__(self, sim, state):
        super().__init__(daemon=True)

        self.sim = sim
        self.state = state
        self.running = threading.Event()
        self.running.set()

    def run(self):
        while self.running.is_set():
            self.sim.step()

            self.state.publish(
                self.sim.pos,
                self.sim.masses,
                self.sim.vel,
            )

    def stop(self):
        self.running.clear()
