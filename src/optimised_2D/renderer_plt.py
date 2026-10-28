import matplotlib.pyplot as plt

from simulation import NBodySim

space = NBodySim(N=100, dt=0.001, dims=2)

plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))


scatter = ax.scatter(
    space.pos[:, 0], space.pos[:, 1], c=space.mass, cmap="viridis", s=20
)

q = ax.quiver(space.pos[:, 0], space.pos[:, 1], space.vel[:, 0], space.vel[:, 1], angles="xy", scale_units="xy", scale=10)


ax.set_xlim(-2000, 2000)
ax.set_ylim(-2000, 2000)
plt.show()


try:
    for frame in range(100000):

        space.step()

        scatter.set_offsets(space.pos)
        q.set_offsets(space.pos)

        q.set_UVC(space.vel[:, 0], space.vel[:, 1])

        fig.canvas.draw()
        plt.pause(0.000001)


except KeyboardInterrupt:
    plt.ioff()
