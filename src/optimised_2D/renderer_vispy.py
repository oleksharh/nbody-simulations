import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import vispy.scene
from vispy.scene import visuals
from vispy import app
from simulation import NBodySim

CANVAS_SIZE = (512, 512)
DEFAULT_DURATION_SEC = 10
MASS_REGULAR_THRESHOLD = 500
MASS_MIN = 100
MASS_MAX = 10000


def create_mass_colormap(masses: np.ndarray):
    cmap_colors = [
        (0.00, "red"),
        (0.33, "yellow"),
        (0.66, "blue"),
        (1.00, "white"),
    ]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("mass_custom", cmap_colors)
    norm = mcolors.LogNorm(vmin=masses.min(), vmax=masses.max())
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap)
    return scalar_map.to_rgba(masses)


def calculate_marker_sizes(masses: np.ndarray) -> np.ndarray:
    log_masses = np.log10(masses)
    log_min = np.log10(MASS_MIN)
    log_mid = np.log10(MASS_REGULAR_THRESHOLD)
    log_max = np.log10(MASS_MAX)

    marker_sizes = np.empty_like(masses, dtype=float)
    regular_mask = masses <= MASS_REGULAR_THRESHOLD
    outlier_mask = masses > MASS_REGULAR_THRESHOLD

    if np.any(regular_mask):
        norm_reg = np.clip(
            (log_masses[regular_mask] - log_min) / (log_mid - log_min), 0.0, 1.0
        )
        marker_sizes[regular_mask] = 3.0 + norm_reg * (6.0 - 3.0)

    if np.any(outlier_mask):
        norm_out = np.clip(
            (log_masses[outlier_mask] - log_mid) / (log_max - log_mid), 0.0, 1.0
        )
        marker_sizes[outlier_mask] = 6.0 + norm_out * (25.0 - 6.0)

    return marker_sizes


def run_nbody_simulation(
    sim: NBodySim,
    fps: int = 60,
    save: bool = False,
    out_filename: str = "",
    duration: int = 60,
):
    """Entry point to running the N-body simulation with VisPy rendering.

     Parameters
    ----------
    sim: NBodySim
        Simulation object of the simulation object
    fps : int
        FPS rate for the animation 60 by default
    save : bool
        Render and save into out_filename, default is nbody_anim_{n}_{dt}_{dims}D.mp4
    out_filename : str
        Output file name for the save mode instead of real-time rendering,
        default nbody_anim_{n}_{dt}_{dims}D.mp4
    duration : int
        Duration of the rendered video in seconds, default 60 secs
    """

    # Simulation constants
    N = sim.N
    DT = sim.dt
    DIMS = sim.dims
    R = sim.R  # soft boundary

    if out_filename == "":
        out_filename = f"nbody_anim_{N}_{DT}_{DIMS}D.mp4"

    # Initialising Canvas and View
    canvas = vispy.scene.SceneCanvas(
        size=CANVAS_SIZE, show=not save, keys=None if save else "interactive"
    )
    view = canvas.central_widget.add_view()

    # World setup
    face_colors = create_mass_colormap(sim.mass)
    marker_sizes = calculate_marker_sizes(sim.mass)

    # Scatter plot visuals setup
    scatter = visuals.Markers()
    scatter.antialias = 0

    def update_scatter():
        scatter.set_data(
            sim.pos,
            edge_width=0,
            face_color=face_colors,
            size=marker_sizes,
            symbol="o",
        )

    update_scatter()
    view.add(scatter)

    # Camera Config
    view.camera = "panzoom" if DIMS == 2 else "arcball"
    view_rng = (-int(R + 0.1 * R), int(R + 0.1 * R))
    view.camera.set_range(x=view_rng, y=view_rng, margin=0)

    # XYZ axis toggle
    visuals.XYZAxis(parent=view.scene)

    # Circle / Sphere Boundary
    if DIMS == 2:
        boundary_circle = visuals.Ellipse(
            center=(0, 0),
            radius=(R, R),  # (radius_x, radius_y)
            color=(0, 0, 0, 0),  # Transparent fill
            border_color=(1, 1, 1, 0.5),  # Semi-transparent white border
            border_width=2.0,
            parent=view.scene,
        )
    else:
        boundary_sphere = visuals.Sphere(
            radius=R,
            method="ico",  # icosahedron mesh generation
            color=(1, 1, 1, 0.05),  # Extremely faint translucent fill
            edge_color=(1, 1, 1, 0.3),  # Subtle wireframe/edge outline
            parent=view.scene,
        )

    def update_frame(event=None):
        sim.step()

        scatter.set_data(
            sim.pos,
            edge_width=0,
            face_color=face_colors,
            size=marker_sizes,
            symbol="o",
        )

    if not save:
        print("Running live interactive simulation...")
        framde_dt = int(1 / fps)
        timer = app.Timer(interval=framde_dt, connect=update_frame, start=True)
        vispy.app.run()
    else:
        os.makedirs("static", exist_ok=True)
        filepath = os.path.join("static", out_filename)
        writer = imageio.get_writer(filepath, fps=fps)
        total_frames = int(fps * duration)

        print(f"Rendering {total_frames} frames offscreen to {filepath}...")
        for _ in range(total_frames):
            update_frame()
            writer.append_data(canvas.render())

        writer.close()
        canvas.close()
        print(f"Finished rendering, saved to {filepath}")


if __name__ == "__main__":
    import sys

    if sys.flags.interactive != 1:
        sim = NBodySim(N=100, dt=0.001, dims=2, R=50)
        run_nbody_simulation(sim=sim, save=False)
