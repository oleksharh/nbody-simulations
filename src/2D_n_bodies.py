from matplotlib import pyplot as plt
import math
import numpy as np

G = 10


class Grid2D:
    def __init__(self, n_bodies, dt):
        self.pos_x, self.pos_y, self.vel_x, self.vel_y, self.mass = (
            np.zeros(n_bodies),
            np.zeros(n_bodies),
            np.zeros(n_bodies),
            np.zeros(n_bodies),
            np.zeros(n_bodies),
        )

        self.dt = dt
        self.i_pointer = 0

    @property
    def radius(self):
        """Mapping mass range to a radius range of (1, 20)"""
        m = self.mass
        return ((m - 1) / (1000 - 1)) * (10 - 1) + 1

    def add_obj(self, x, y, vel_x, vel_y, mass):
        if mass < 0.1 or mass > 1000:
            raise ValueError(f"Mass {mass} must be in range (0.1, 500).")

        i = self.i_pointer
        self.pos_x[i] = x
        self.pos_y[i] = y
        self.vel_x[i] = vel_x
        self.vel_y[i] = vel_y
        self.mass[i] = mass

        self.i_pointer += 1

    def _get_pairwise_distances(self):

        # Following Euclidian Distance formula sqrt((x2-x1)^2 + (y2-y1)^2)
        # for every possible n body combination

        # creates 2d matrix N x N size, initially (1, N) and subtracts (N, 1)
        # to find every possible (points i and j) Xj-Xi and same for y elements
        dx = self.pos_x[np.newaxis, :] - self.pos_x[:, np.newaxis]
        dy = self.pos_y[np.newaxis, :] - self.pos_y[:, np.newaxis]

        r = np.sqrt(dx**2 + dy**2)

        # distances to itself ignored and to avoid 0 division error are set to inf
        np.fill_diagonal(r, np.inf)

        return dx, dy, r

    def compute_accelerations(self):
        dx, dy, r = self._get_pairwise_distances()

        mass_matrix = self.mass[
            np.newaxis, :
        ]  # creates convinient calculation shape (1, N)
        print("Mass", mass_matrix)
        # we consider vector formula a = G * mass/r^2 * unit_r
        # where unit_r = displacement vector/r = (dx, dy)/r
        # we have dx and dy, and substitute the unit_r in a formula
        # getting a = G * mass/r^2 * ((dx, dy)/r) = G * mass/r^3 * (dx, dy)
        factor = G * mass_matrix / (r**3)

        print(factor)

        # Sum forces from all other bodies (axis=1 sums across columns for each body i)
        acc_x = np.sum(factor * dx, axis=1)
        acc_y = np.sum(factor * dy, axis=1)

        return acc_x, acc_y

    def step(self):
        # Velocity Verlet integration
        ax1, ay1 = self.compute_accelerations()

        self.pos_x += self.vel_x * self.dt + 0.5 * ax1 * self.dt**2
        self.pos_y += self.vel_y * self.dt + 0.5 * ay1 * self.dt**2

        # with new x1 and x2 acc recalculations
        ax2, ay2 = self.compute_accelerations()

        self.vel_x += 0.5 * (ax1 + ax2) * self.dt
        self.vel_y += 0.5 * (ay1 + ay2) * self.dt







# --- Quick ChatGPT-made Stable Setup---
n = 4
grid = Grid2D(
    n_bodies=n, dt=0.001
)  # Slightly smaller dt for better precision with high speeds

# 1. Central Star (Massive)
grid.add_obj(x=0.0, y=0.0, vel_x=0.0, vel_y=0.0, mass=1000)

# 2. Planet 1 (Close orbit: r = 4.0)
# v = sqrt(G * M / r) = sqrt(10 * 1000 / 4) = sqrt(2500) = 50.0
grid.add_obj(x=4.0, y=0.0, vel_x=0.0, vel_y=50.0, mass=10.0)

# 3. Planet 2 (Medium orbit: r = 8.0)
# v = sqrt(10 * 1000 / 8) = sqrt(1250) ≈ 35.35
grid.add_obj(x=0.0, y=-8.0, vel_x=35.35, vel_y=0.0, mass=15.0)

# 4. Planet 3 (Far orbit: r = 13.0)
# v = sqrt(10 * 1000 / 13) = sqrt(769.23) ≈ 27.73
grid.add_obj(x=-13.0, y=0.0, vel_x=0.0, vel_y=-27.73, mass=20.0)

# --- Live Plotting Loop ---
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))

ax.set_facecolor("black")
ax.set_title("Stable N-Body Orbit Simulation", color="white")
fig.patch.set_facecolor("black")

scat = ax.scatter(
    grid.pos_x,
    grid.pos_y,
    s=grid.radius**2 * 10,
    c=grid.mass,
    cmap="plasma",
    edgecolors="white",
    linewidths=0.5,
)

tracked_body_idx = 0  # Tracking the central star
view_span = 20.0  # Camera view window

print("Running stable simulation... Close the plot window to exit.")

try:
    while plt.fignum_exists(fig.number):
        for _ in range(10):  # More substeps per frame for smooth physics
            grid.step()

        offsets = np.column_stack((grid.pos_x, grid.pos_y))
        scat.set_offsets(offsets)

        target_x = grid.pos_x[tracked_body_idx]
        target_y = grid.pos_y[tracked_body_idx]

        ax.set_xlim(target_x - view_span, target_x + view_span)
        ax.set_ylim(target_y - view_span, target_y + view_span)

        fig.canvas.draw()
        fig.canvas.flush_events()

        plt.pause(0.001)

except KeyboardInterrupt:
    print("Simulation stopped by user.")
