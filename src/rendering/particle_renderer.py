import time
import numpy as np
import vispy.scene
from vispy.scene import visuals
from vispy import app

class ParticleRenderer:
    def __init__(self, state, dims=2, canvas_size=(900, 700)):
        self.state = state
        self.dims = dims
        self.last_version = -1

        # Metrics tracking
        self.render_frame_count = 0
        self.sim_tick_count = 0
        self.fps_update_interval = 0.5  # Update title every 0.5 seconds
        self.last_metrics_time = time.time()
        
        self.current_fps = 0.0
        self.current_tps = 0.0

        self.canvas = vispy.scene.SceneCanvas(
            keys="interactive",
            show=True,
            size=canvas_size,
            bgcolor="black",
        )

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "panzoom" if dims == 2 else "turntable"

        self.scatter = visuals.Markers()
        self.view.add(self.scatter)

        self.timer = app.Timer(
            interval=0,
            connect=self.update,
            start=True,
        )

    def update(self, event=None):
        # 1. Count every single render pass for FPS
        self.render_frame_count += 1

        # 2. Try reading the latest state from the simulation worker
        snapshot, version = self.state.read()

        # Check if the simulation worker actually produced a NEW step
        if snapshot is not None and version != self.last_version:
            self.sim_tick_count += 1  # Count unique simulation updates
            self.last_version = version
            
            # Unpack snapshot data
            pos, mass, vel = snapshot

            # Compute colors and sizes
            speed = np.linalg.norm(vel, axis=1)
            max_speed = speed.max()
            if max_speed > 0:
                normalised = speed / max_speed
            else:
                normalised = np.zeros_like(speed)

            colors = np.empty((len(pos), 4), dtype=np.float32)
            colors[:, 0] = normalised
            colors[:, 1] = 0.3
            colors[:, 2] = 1.0 - normalised
            colors[:, 3] = 1.0

            log_mass = np.log10(np.maximum(mass, 1e-30))
            mass_range = np.ptp(log_mass)

            if mass_range > 0:
                sizes = 3.0 + 8.0 * ((log_mass - log_mass.min()) / mass_range)
            else:
                sizes = np.full(len(mass), 4.0)

            # Update geometry on screen only when new data arrives
            self.scatter.set_data(
                pos,
                size=sizes,
                face_color=colors,
                edge_width=0,
                symbol="o",
            )

        # 3. Periodically update the window title with true metrics
        now = time.time()
        elapsed = now - self.last_metrics_time
        if elapsed >= self.fps_update_interval:
            self.current_fps = self.render_frame_count / elapsed
            self.current_tps = self.sim_tick_count / elapsed
            
            # Reset counters
            self.render_frame_count = 0
            self.sim_tick_count = 0
            self.last_metrics_time = now
            
            # Display both Render FPS and Simulation TPS
            self.canvas.title = f"Render FPS: {self.current_fps:.1f} | Sim TPS: {self.current_tps:.1f}"

    def run(self):
        app.run()

    def close(self, worker=None):
        if worker is not None:
            worker.stop()
            worker.join()
        self.timer.stop()
        self.canvas.close()