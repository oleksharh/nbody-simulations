import numpy as np
from numpy import int8, int64, float64
from numba import njit, prange
from typing import NamedTuple, Optional

from type_defs import (
    Float64Vec1D,
    Float64Vec2D,
    Float64Vec3D,
    Int64Vec1D,
    Int8Vec1D,
    BoolVec1D,
    Int64Vec4D,
)


class QuadTree(NamedTuple):
    """
    Helper Typing class for QuadTree Structure of Arrays

    Parameters
    ----------
    size : Int64Vec1D
        shape (1,), int64 - nodes populated as ndarray only because int would be immutable once compiled
    half : Float64Vec1D
        (max_nodes,) float64
    center_pos : Float64Vec2D
        (max_nodes, 2) float64
    depth : Int8Vec1D
        (max_nodes,) int8
    mass : Float64Vec1D
        (max_nodes,) float64
    com_pos : Float64Vec2D
        (max_nodes, 2) float64
    is_leaf : BoolVec1D
        (max_nodes,) bool True by default
    body_idx : Int64Vec1D
        (max_nodes,) int64    -1 means no body index reference is present
    child_indices : Int64Vec4D
        (max_nodes, 4) int64  -1 means no body index reference is present
    next_idx : Int64Vec1D
        Array with indices pointing to original body_idx - implicit idx of body in the pos, masses, acc, vel arrays of Sim class
    """

    size: Int64Vec1D
    half: Float64Vec1D
    center_pos: Float64Vec2D
    depth: Int8Vec1D
    mass: Float64Vec1D
    com_pos: Float64Vec2D
    is_leaf: BoolVec1D
    body_idx: Int64Vec1D
    child_indices: Int64Vec4D

    next_idx: Int64Vec1D


def zeros_tree(max_nodes: int, n_count: int) -> QuadTree:
    """
    Initialises QuadTree SoA to initial values,
    essentially allocates arrays in memory and
    returns QuadTree object
    """
    return QuadTree(
        size=np.zeros(1, dtype=int64),  # currently existing nodes
        half=np.zeros(max_nodes, dtype=float64),
        center_pos=np.zeros((max_nodes, 2), dtype=float64),
        depth=np.zeros(max_nodes, dtype=int8),
        mass=np.zeros(max_nodes, dtype=float64),
        com_pos=np.zeros((max_nodes, 2), dtype=float64),
        is_leaf=np.full(max_nodes, fill_value=True),
        body_idx=np.full(max_nodes, fill_value=-1, dtype=int64),
        child_indices=np.full(
            (max_nodes, 4), fill_value=-1, dtype=int64
        ),  # implicit idx (this is the quadrant) of the index pointing (filled index) to the node
        next_idx=np.full(
            n_count, fill_value=-1, dtype=int64
        ),  # implicit original body_idx used to create an implicit linked list
    )


########################
# JIT Helper Functions #
########################


@njit
def get_square_values(positions: Float64Vec2D):
    """
    Find the longest possible distance vector of distnat objects and
    generates square center coordinates and it's half value of width
    """
    min_x = positions[:, 0].min()
    max_x = positions[:, 0].max()
    min_y = positions[:, 1].min()
    max_y = positions[:, 1].max()

    center_pos = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    half = max(max_x - min_x, max_y - min_y) / 2 * 1.01 + 1e-9
    return half, center_pos


@njit
def quadrant_idx(pos, center):
    """
    Returns Quadrant index of the object at position: pos

    SW, SE, NW, NE -> 0, 1, 2, 3
    """
    return int(pos[0] >= center[0]) | (int(pos[1] >= center[1]) << 1)


@njit
def allocate_node(qt: QuadTree, half: np.float64, center_pos, depth: int):
    """
    Allocates node by adding it's square valuse and depth, node still needs to be inserted

    Parameters
    ----------
    center_pos
        Takes one (x, y) coordinates of square's center position
    """
    i = qt.size[0]
    if i == -1:
        return -1

    if i >= qt.half.shape[0]:
        return -1

    qt.size[0] = i + 1
    qt.half[i] = half
    qt.center_pos[i] = center_pos
    qt.depth[i] = depth

    return i


@njit
def child_square_values(half: np.float64, center_pos, q_idx: int):
    """
    Finds and returns new square values created in a given quadrant
    """
    new_half = half / 2
    # q_idx = quadrant_idx(pos, center_pos)
    new_center_pos = center_pos + np.array(
        [
            new_half if q_idx & 1 else -new_half,  # X axis shifting
            new_half if q_idx & 2 else -new_half,  # Y axis shifting
        ]
    )

    return new_half, new_center_pos


@njit
def insert_node(
    qt: QuadTree,
    node: int,
    body: int,
    pos: Float64Vec2D,
    masses: Float64Vec1D,
    max_depth: int,
):
    """
    Allocates and inserts node's values:

    - body_idx
    - position
    - body's mass
    - and generates square values in the tree
    - and gives current depth to that node

    Parameters
    ----------
    node : int64
        node index, start from root - entry point
    body : int64
        body index - ith index of the body that's being inserted
    pos:
    """

    # 1) if node body is empty no body and is leaf
    if qt.is_leaf[node] and qt.body_idx[node] == -1:
        qt.body_idx[node] = body
        qt.mass[node] = masses[body]
        qt.com_pos[node] = pos[body]
        return True

    # 2) if node body exists but its external -
    # so it needs to fall through to the next
    # case 3 or be limited with max depth reached
    if qt.is_leaf[node]:
        if qt.depth[node] >= max_depth:
            m = qt.mass[node] + masses[body]
            qt.com_pos[node] = (
                qt.mass[node] * qt.com_pos[node] + masses[body] * pos[body]
            ) / m
            qt.mass[node] = m

            # bodies that fall into the same quadrant create an implicit linked list
            if qt.body_idx[node] != -1:
                qt.next_idx[body] = qt.body_idx[node]
                qt.body_idx[node] = (
                    body  # this will be a head ptr of the implicit linked list
                )
            return True

        old_body = qt.body_idx[node]
        qt.body_idx[node] = -1
        qt.is_leaf[node] = False
        if not insert_node(qt, node, old_body, pos, masses, max_depth):
            return False

    # 3) If node is not leaf and has body
    q_idx = quadrant_idx(pos[body], qt.center_pos[node])
    ch_node = qt.child_indices[node, q_idx]
    if ch_node == -1:
        ch_half, ch_center_pos = child_square_values(
            qt.half[node], qt.center_pos[node], q_idx
        )
        ch_node = allocate_node(qt, ch_half, ch_center_pos, qt.depth[node] + 1)
        if ch_node < 0:
            return False
        qt.child_indices[node, q_idx] = ch_node
    return insert_node(qt, ch_node, body, pos, masses, max_depth)


@njit
def compute_mass(qt: QuadTree, node_idx: int):
    if qt.is_leaf[node_idx] == True:
        return

    total = 0.0
    weighted = np.zeros(2)
    for ch_idx in qt.child_indices[node_idx]:
        if ch_idx == -1:
            continue
        compute_mass(qt, ch_idx)
        total += qt.mass[ch_idx]
        weighted += qt.mass[ch_idx] * qt.com_pos[ch_idx]

    qt.mass[node_idx] = total
    qt.com_pos[node_idx] = weighted / total


@njit
def populate_tree(
    qt: QuadTree, pos: Float64Vec2D, masses: Float64Vec1D, max_depth: int
):
    """
    Populates the tree with positions and masses
    """
    half, center_pos = get_square_values(pos)
    qt.size[0] = 0
    root_idx = allocate_node(qt, half, center_pos, depth=0)
    if root_idx < 0:
        return -1

    for i in range(pos.shape[0]):
        if not insert_node(qt, root_idx, i, pos, masses, max_depth):
            return -1  # overflow signal

    compute_mass(qt, root_idx)

    return qt.size[0]


@njit
def node_acc(qt: QuadTree, node_idx, body_idx, pos, masses, theta):
    if qt.mass[node_idx] == 0.0:
        return np.zeros(2)

    epsilon = 0.5

    if qt.is_leaf[node_idx]:
        total = np.zeros(2)

        j = qt.body_idx[
            node_idx
        ]  # iterating over the linked list to see if is body_idx in there to be ignored
        # ignoring aggregate mass in case there is body_idx in there
        while j != -1:
            if j != body_idx:
                r_vec = pos[j] - pos[body_idx]
                r_sq = r_vec @ r_vec

                total += masses[j] * r_vec / (r_sq + epsilon**2) ** 1.5

            j = qt.next_idx[j]

        return total

    r_vec = qt.com_pos[node_idx] - pos[body_idx]
    r_sq = r_vec @ r_vec
    if (2 * qt.half[node_idx]) ** 2 < theta**2 * r_sq:
        return qt.mass[node_idx] * r_vec / (r_sq + epsilon**2) ** 1.5

    total = np.zeros(2)
    for ch_idx in qt.child_indices[node_idx]:
        if ch_idx >= 0:
            total += node_acc(qt, ch_idx, body_idx, pos, masses, theta)
    return total


@njit(parallel=True)
def calculate_acc(qt, root_idx, acc, pos, masses, theta, R, soft_bounding):
    # updates acc array in place no need to return
    N = acc.shape[0]
    for i in prange(N):
        acc[i] = node_acc(qt, root_idx, i, pos, masses, theta)

    if soft_bounding:
        k = 1.0
        for i in prange(N):
            r = np.linalg.norm(pos[i])
            if r >= R:
                hat_n = pos[i] / r

                acc[i] -= k * (r - R) * hat_n


def build_tree(pos: Float64Vec2D, masses: Float64Vec1D, max_nodes: int, max_depth: int):
    """
    Initialisation of tree SoA and its population
    """
    while True:
        qt = zeros_tree(max_nodes, len(pos))
        used = populate_tree(qt, pos, masses, max_depth)
        if used >= 0:
            return qt, max_nodes
        max_nodes *= 2


#########################
# Simulation Exec Class #
#########################
from simulations.base_sim import MassFunc, PosFunc, Simulation, VelFunc


class FlatBHut2DSim(Simulation):
    """
    Barnes-Hut implementation using structure of arrays (SoA)

    It utilizes numba jit-supported functions to support faster calculations
    and paralellisation of the calculations.
    """

    def __init__(
        self,
        N: int,
        dt: float,
        dims: int,
        R: int,
        theta: float,
        max_depth: int,
        soft_bounding: bool = True,
        pos_func: Optional[PosFunc] = None,
        vel_func: Optional[VelFunc] = None,
        mass_func: Optional[MassFunc] = None,
    ) -> None:
        super().__init__(N, dt, dims, R, pos_func, vel_func, mass_func)

        self.theta = theta
        self.max_depth = max_depth
        self.soft_bounding = soft_bounding

        temp_qt, self.max_nodes = build_tree(
            self.pos, self.masses, self.N * 4, self.max_depth
        )
        calculate_acc(
            temp_qt,
            0,
            self.acc,
            self.pos,
            self.masses,
            self.theta,
            self.R,
            self.soft_bounding,
        )
        del temp_qt

    def step(self) -> None:
        dt = self.dt
        self.pos += self.vel * dt + 0.5 * self.acc * dt**2
        self.vel += 0.5 * self.acc * dt
        qt, self.max_nodes = build_tree(
            self.pos, self.masses, self.N * 4, self.max_depth
        )
        calculate_acc(
            qt,
            0,
            self.acc,
            self.pos,
            self.masses,
            self.theta,
            self.R,
            self.soft_bounding,
        )
        self.vel += 0.5 * self.acc * dt
