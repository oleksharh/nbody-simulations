# Integrator and Force Solvers used

## All methods use the same integrator: Kick–Drift–Kick (KDK) Velocity Verlet

*Better Explained at:*

- [Verlet Integration – The n-Body Problem](https://gereshes.com/2018/07/09/verlet-integration-the-n-body-problem/).
- [Verlet Integration by James Schloss](https://www.algorithm-archive.org/contents/verlet_integration/verlet_integration.html).

---

The KDK integrator splits each timestep $(\Delta t)$ into three stages:

1. **First Kick (half-step velocity update):** Update velocity using the current acceleration over half a timestep.
2. **Drift (full-step position update):** Update position using the half-step velocity over the full timestep.
3. **Second Kick (half-step velocity update):** Recalculate forces and accelerations at the new position, then update velocity over the remaining half timestep.

In code:

```python
# Calculating positions at t + dt
for i in prange(len(pos)):
    pos[i] += vel[i] * dt + 0.5 * acc[i] * dt**2

# Velocities at t + dt/2
for i in prange(len(pos)):
    vel[i] += 0.5 * acc[i] * dt

# Calculating accelerations at t + dt
_calculate_acc(pos, acc, mass, R)

# Velocities at t + dt/2
for i in prange(len(pos)):
    vel[i] += 0.5 * acc[i] * dt
```

### Position update equivalence

The following expressions are mathematically equivalent:

$$
\mathbf{x}_{t+\Delta t} =
\mathbf{x}_t + \mathbf{v}_t\Delta t +
\frac{1}{2}\mathbf{a}_t\Delta t^2
$$

and

$$
\mathbf{x}_{t+\Delta t} =
\mathbf{x}_t +
\left(\mathbf{v}_t + \frac{1}{2}\mathbf{a}_t\Delta t\right)\Delta t
$$

Therefore, in code:

```python
pos[i] += vel[i] * dt + 0.5 * acc[i] * dt**2
```

## Direct Summation Method

The Direct Summation method iterates over all $(N)$ bodies, calculating the displacement vector between each body and every other body to determine their pairwise gravitational interactions.

For each body, it evaluates $(N-1)$ displacement vector pairs, resulting in:

$$
N(N-1) = N^2-N
$$

Therefore, the total number of pairwise interactions scales quadratically with the number of bodies.

**Time Complexity:**

$$
\boxed{O(N^2)}
$$

## Barnes Hut Method

*Better Explained at:*

- [The Barnes-Hut Approximation by Jeffrey Heer](https://jheer.github.io/barnes-hut/).
- [The Barnes-Hut Algorithm by Tom Ventimiglia & Kevin Wayne](https://arborjs.org/docs/barnes-hut).

---

Currently implements a **2D Barnes Hut method** using a QuadTree data structure. It significantly reduces the computational cost of the Direct Summation method, achieving an average time complexity of approximately $O(N\log N)$.

### QuadTree Construction

The algorithm recursively subdivides the simulation space into four quadrants, creating child nodes whenever the number of bodies exceeds the specified threshold.

- **Internal nodes:** Represent a region containing multiple bodies. They store the total mass and centre of mass (mass-weighted position) of that region.
- **External nodes (leaf nodes):** Represent individual bodies or regions that cannot be subdivided further.

A maximum tree depth is enforced to prevent infinite subdivision when bodies overlap or occupy identical positions. In such cases, the resulting leaf node accommodates multiple bodies, typically using a linked list rather than further subdividing the tree.

### Force Approximation

The algorithm uses the ratio between the node's spatial size ($s$) and its distance ($d$) from the body being evaluated to determine whether the node can be approximated as a single point mass.

$$
\frac{s}{d} < \theta
$$

Where:

- $s$ — Width of the node's spatial region.
- $d$ — Distance between the evaluated body and the node's centre of mass.
- $\theta$ — Opening-angle threshold, typically set to $0.5$.

Two possible outcomes:

1. **Approximation ($s/d < \theta$):** The node is sufficiently distant, so its entire mass is approximated as a single point mass located at its centre of mass. The algorithm calculates the gravitational force using this approximation without traversing its children.
2. **Subdivision ($s/d \geq \theta$):** The node is sufficiently close that the approximation may introduce excessive error. The algorithm recursively traverses its children, eventually calculating interactions with individual bodies in the region.

### Computational Complexity

The Barnes–Hut algorithm typically achieves:

$$
O(N\log N)
$$

However, its actual performance depends on the spatial distribution of bodies and the opening-angle threshold.

- **Smaller $\theta$:** More nodes are opened and traversed, increasing computational cost while generally improving approximation accuracy. As $\theta$ approaches zero, the algorithm approaches Direct Summation behaviour, with $O(N^2)$ complexity.
- **Larger $\theta$:** More nodes are approximated, reducing computational cost at the expense of accuracy.
- **Highly clustered or overlapping bodies:** Can result in deeper trees and increased traversal costs. If multiple bodies occupy identical positions, the depth limiter prevents endless subdivision, but interactions within the resulting multi-body leaf may require direct pairwise calculations, potentially reaching $O(N^2)$ complexity.
