import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import vispy.scene
from vispy.scene import visuals
from vispy import app
from simulation import NBodySim


def run_nbody_simulation(
    n: int,
    dt: float,
    dims: int = 2,
    frame_dt: float = 0.016,
    save: bool = False,
    **kwargs,
):
    """
    Entry point to running n_body simulation

    Parameters
    ----------
    n : float
        Number of object/bodies to be in the simulation
    dt : float
        Time step value for the simulation engine
    dims : int
        Choice of 2D or 3D simulation rendering
    frame_dt : float
        Time step for one frame rendering, 16ms by default to make it roughly 60FPS
    save : bool
        Render and save into out_filename, default is nbody_anim_{n}_{dt}_{dims}D.mp4
    out_filename : str
        Output file name for the save mode instead of real-time rendering,
        default nbody_anim_{n}_{dt}_{dims}D.mp4
    duration_sec : int
        Duration of the rendered video in seconds, default 10 secs


    """
    # Unpacking kwargs
    out_filename = f"nbody_anim_{n}_{dt}_{dims}D.mp4"
    duration_sec = 10
    if save:
        out_filename = kwargs.get("out_filename", out_filename)
        duration_sec = int(kwargs.get("duration_sec", duration_sec))

    # Initialising Canvas
    if save:
        canvas = vispy.scene.SceneCanvas(size=(512, 512), show=False)
    else:
        canvas = vispy.scene.SceneCanvas(keys="interactive", show=True)
    view = canvas.central_widget.add_view()

    # World setup
    nbody_sim = NBodySim(N=n, dt=dt, dims=dims)
    pos = nbody_sim.pos
    masses = nbody_sim.mass

    # Custom color map (mass-based from low to high)
    cmap_colors = [
        (0.00, "red"),
        (0.33, "yellow"),
        (0.66, "blue"),
        (1.00, "white"),
    ]

    custom_cmap = mcolors.LinearSegmentedColormap.from_list("mass_custom", cmap_colors)
    norm = mcolors.LogNorm(vmin=masses.min(), vmax=masses.max())
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap)
    face_colors = scalar_map.to_rgba(masses)  # to rgba acceptable arg for vispy scatter

    # Mass dependent sizing of the objects
    log_masses = np.log10(masses)

    log_min = np.log10(100)  # 2
    log_mid = np.log10(500)  # 2.7
    log_max = np.log10(10000)  # 4

    marker_sizes = np.empty_like(masses, dtype=float)

    regular_mask = masses <= 500
    outlier_mask = masses > 500

    # Map masses 100-500 to range [3.0, 6.0]
    if np.any(regular_mask):
        norm_reg = (log_masses[regular_mask] - log_min) / (log_mid - log_min)
        norm_reg = np.clip(norm_reg, 0.0, 1.0)  # anything out of mass-range (100, 500)
        marker_sizes[regular_mask] = 3.0 + norm_reg * (6.0 - 3.0)

    if np.any(outlier_mask):
        norm_out = (log_masses[outlier_mask] - log_mid) / (log_max - log_mid)
        norm_out = np.clip(norm_out, 0.0, 1.0)  # anythin out of mass-range (500, 10k)
        marker_sizes[outlier_mask] = 6.0 + norm_out * (25.0 - 6.0)

    # marker sizes are just px values for the scatter px size of the objects

    scatter = visuals.Markers()
    scatter.antialias = 0

    def update_scatter():
        scatter.set_data(
            pos, edge_width=0, face_color=face_colors, size=marker_sizes, symbol="o"
        )

    update_scatter()

    view.add(scatter)
    if dims == 2:
        view.camera = "panzoom"
    else:
        view.camera = "arcball"
    view.camera.set_range(x=(-30, 30), y=(-30, 30), margin=0)

    # turning the axis on
    axis = visuals.XYZAxis(parent=view.scene)

    def update(event=None):
        nbody_sim.step()

        # update positions after one time step
        update_scatter()

    if not save:
        print("here")
        timer = app.Timer(interval=frame_dt, connect=update, start=True)
        vispy.app.run()
    else:
        writer = imageio.get_writer(f"static/{out_filename}", fps=60)
        total_frames = int(duration_sec / frame_dt)

        print(f"Rendering {total_frames} frames offscreen...")

        for frame in range(total_frames):
            update()
            frame_img = canvas.render()

            writer.append_data(frame_img)

        writer.close()
        canvas.close()

        print(f"Finished Rendering, saved to static/{out_filename}")


if __name__ == "__main__":
    import sys

    if sys.flags.interactive != 1:
        run_nbody_simulation(100, 0.001, 2, save=True, duration_sec=30)
