import threading
import time


class SimWorker(threading.Thread):

    def __init__(self, sim, state, target_fps=120.0):
        super().__init__(daemon=True)

        self.sim = sim
        self.state = state
        self.target_fps = target_fps
        self.frame_duration = 1.0 / target_fps
        
        self.running = threading.Event()
        self.running.set()

    def run(self):
        while self.running.is_set():
            start_time = time.perf_counter()

            self.sim.step()
            self.state.publish(
                self.sim.pos,
                self.sim.masses,
                self.sim.vel,
            )

            elapsed = time.perf_counter() - start_time

            if elapsed < self.frame_duration:
                time.sleep(self.frame_duration - elapsed)

    def stop(self):
        self.running.clear()