import numpy as np
from typing import Optional, List
from simulation import MassFunc, PosFunc, VelFunc, Simulation



class Node:
    def __init__(self, centre_pos: np.ndarray, half: float, depth: int) -> None:
        self.centre_pos = centre_pos
        self.half = half
        self.depth = depth

        self.is_leaf = True
        self.body: Optional[int] = None
        self.children: List[Optional["Node"]] = [None] * 4  # SW, SE, NW, NE

        self.mass = 0.0
        self.com_pos = np.zeros(2)


class BarnesHutSim(Simulation):
    def __init__(
        self,
        N: int,
        dt: float,
        dims: int,
        R: int,
        max_depth: int,
        theta: float = 0.5,
        pos_func: Optional[PosFunc] = None,
        vel_func: Optional[VelFunc] = None,
        mass_func: Optional[MassFunc] = None,
    ) -> None:
        if dims != 2:
            raise ValueError("Only 2 dims for now")
        super().__init__(N, dt, dims, R, pos_func, vel_func, mass_func)
        self.max_depth = max_depth
        self.theta = theta
        self.root: Optional[Node] = None

        self._update_acc()

    def _get_root_square(self):
        mins = self.pos.min(axis=0)
        maxs = self.pos.max(axis=0)
        centre_pos = (mins + maxs) / 2
        half = (maxs - mins).max() / 2 * 1.01 + 1e-9
        return centre_pos, half

    @staticmethod
    def _quadrant_idx(pos, centre_pos) -> int:
        dpos = pos - centre_pos
        # SW, SE, NW, NE -> 0, 1, 2, 3
        return int(dpos[0] >= 0) | (int(dpos[1] >= 0) << 1)

    def _make_child(self, node: Node, c: int) -> Node:
        new_half = node.half / 2
        centre_pos = node.centre_pos + np.array(
            [
                new_half if c & 1 else -new_half,
                new_half if c & 2 else -new_half,
            ]
        )
        child = Node(centre_pos, new_half, node.depth + 1)
        node.children[c] = child
        return child

    def _insert(self, node: Node, i: int) -> None:
        if node.is_leaf and node.body is None:
            node.body = i
            node.mass = float(self.masses[i])
            node.com_pos = self.pos[i].copy()

            return

        if node.is_leaf and node.body is not None:
            if node.depth >= self.max_depth:
                om, opos = node.mass, node.com_pos
                node.mass += self.masses[i]
                node.com_pos = (opos * om + self.pos[i] * self.masses[i]) / node.mass

                return

            obody = node.body
            node.body = None
            node.is_leaf = False
            self._insert(node, obody)

        if not node.is_leaf:
            c_idx = self._quadrant_idx(self.pos[i], node.centre_pos)
            if node.children[c_idx] is None:
                self._make_child(node, c_idx)
            self._insert(node.children[c_idx], i)

    def create_tree(self) -> None:
        centre_pos, half = self._get_root_square()
        self.root = Node(centre_pos, half, depth=0)
        for i in range(self.N):
            self._insert(self.root, i)

    def _compute_mass(self, node: Node):
        if node.is_leaf:
            return

        total = 0.0
        weighted = np.zeros(2)
        for ch in node.children:
            if ch is None:
                continue
            self._compute_mass(ch)
            total += ch.mass
            weighted += ch.mass * ch.com_pos

        node.mass = total
        node.com_pos = weighted / total

    def _node_acc(self, node: Node, i: int):
        if node.mass == 0.0 or (node.is_leaf and node.body == i):
            return np.zeros(2)

        r_vec = node.com_pos - self.pos[i]
        r_sq = r_vec @ r_vec  # dot product

        # far enough for estimation
        epsilon = 0.5
        if node.is_leaf or (2 * node.half) ** 2 < self.theta**2 * r_sq:
            return node.mass * r_vec / (r_sq + epsilon**2) ** 1.5

        # too close for estimation
        total = np.zeros(2)
        for ch in node.children:
            if ch is not None:
                total += self._node_acc(ch, i)
        return total

    def _update_acc(self) -> None:
        self.create_tree()
        self._compute_mass(self.root)

        for i in range(self.N):
            self.acc[i] = self._node_acc(self.root, i)

        # soft boundary, same as in Direcut Sum implementation
        k = 1.0
        for i in range(self.N):
            r = np.linalg.norm(self.pos[i])
            if r >= self.R:
                hat_n = self.pos[i] / r

                self.acc[i] -= k * (r - self.R) * hat_n

    def step(self) -> None:
        dt = self.dt
        self.pos += self.vel * dt + 0.5 * self.acc * dt**2
        self.vel += 0.5 * self.acc * dt
        self._update_acc()
        self.vel += 0.5 * self.acc * dt
