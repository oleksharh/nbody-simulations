import matplotlib.pyplot as plt

from simulation import Grid2D

space = Grid2D(N=100, dt=0.001)

plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))


scatter = ax.scatter(
    space.pos[:, 0], space.pos[:, 1], c=space.mass, cmap="viridis", s=20
)

ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
plt.show()


try:
    for frame in range(100000):

        space.step()

        scatter.set_offsets(space.pos)

        fig.canvas.draw()
        plt.pause(0.000001)


except KeyboardInterrupt:
    plt.ioff()
