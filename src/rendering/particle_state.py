import threading
import numpy as np


class ParticleState:
    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = None
        self.version = 0

    def publish(self, pos, mass, vel):
        
        snapshot = (
            np.array(pos, dtype=np.float32, copy=True),
            np.array(mass, dtype=np.float32, copy=True),
            np.array(vel, dtype=np.float32, copy=True),
        )

        with self._lock:
            self._snapshot = snapshot
            self.version += 1

    def read(self):
        with self._lock:
            return self._snapshot, self.version
