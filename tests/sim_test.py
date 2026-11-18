import numpy as np
import pytest
import math
from simulations import Simulation, DirectSum, FlatBHut2DSim, build_tree, calculate_acc

@pytest.fixture(
    params=[
        lambda: DirectSum(N=100, dt=0.01, dims=2, R=5),
        lambda: DirectSum(N=100, dt=0.01, dims=3, R=5),
        lambda: FlatBHut2DSim(
            N=10000, dt=0.01, dims=2, R=3, theta=0.7, max_depth=10, soft_bounding=True
        ),
    ],
    ids=["vel_verlet_2D", "vel_verlet_3D", "bhut_flat"],
)
def sim(request):

    return request.param()


def test_sim_step(sim: Simulation):
    """
    Checks whether step function works correctly, doesn't account for overtime error
    """
    old_pos = sim.pos.copy()
    old_vel = sim.vel.copy()
    old_acc = sim.acc.copy()
    dt = sim.dt

    new_pos = old_pos + old_vel * dt + 0.5 * old_acc * dt**2

    sim.step()

    np.testing.assert_allclose(new_pos, sim.pos)


EPS = 0.5


def make_bodies(n, seed=0, spread=1.0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=spread, size=(n, 2)).astype(np.float64)
    masses = rng.uniform(0.5, 2.0, size=n).astype(np.float64)
    return pos, masses


def clustered_bodies():
    """Clustered and nearly identical pos bodies to see
    if tree is constructed correctly and creates linked list"""
    pos = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1e-12, 0.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
            [0.5, 0.5],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    masses = np.arange(1, len(pos) + 1, dtype=np.float64)
    return pos, masses


def brute_force_acc(pos, masses):
    """Copy of Direct Sum func in Vel Verlet module"""
    n = len(pos)
    acc = np.zeros((n, 2))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # no self-force
            r_vec = pos[j] - pos[i]
            r_sq = r_vec @ r_vec
            acc[i] += masses[j] * r_vec / (r_sq + EPS**2) ** 1.5
    return acc


def tree_acc(pos, masses, theta, max_depth, max_nodes=None):
    if max_nodes is None:
        max_nodes = len(pos) * 4
    qt, _ = build_tree(pos, masses, max_nodes, max_depth)
    acc = np.zeros_like(pos)
    # R huge + soft_bounding False so the boundary force never kicks in
    calculate_acc(qt, 0, acc, pos, masses, theta, 1e12, False)
    return acc


def leaf_chains(qt):
    """
    Returns {leaf_node_idx: [body indices in its chain]} and fails on a cycle.
    """
    n_bodies = len(qt.next_idx)
    chains = {}
    for node in range(qt.size[0]):
        if not qt.is_leaf[node] or qt.body_idx[node] == -1:
            continue
        chain = []
        j = qt.body_idx[node]
        while j != -1:
            chain.append(int(j))
            assert len(chain) <= n_bodies, f"cycle in chain at leaf {node}"
            j = qt.next_idx[j]
        chains[node] = chain
    return chains


def all_chained_bodies(qt):
    """
    Returns all found bodies in the chains
    This should match the N number of bodies
    """
    found = []
    chains = leaf_chains(qt)
    for node in chains:
        for body in chains[node]:
            found.append(body)
    return found


def mass_and_com(indices, pos, masses):
    total = 0.0
    weighted = np.zeros(2)
    for j in indices:
        total += masses[j]
        weighted += masses[j] * pos[j]

    return total, weighted / total


def net_force(masses, acc):
    """
    Sum of m_i * a_i over all bodies, returned as (fx, fy, scale),
    where scale is the sum of absolute contributions (to compare against).
    """
    fx = 0.0
    fy = 0.0
    scale = 0.0
    for i in range(len(masses)):
        contribution_x = masses[i] * acc[i][0]
        contribution_y = masses[i] * acc[i][1]
        fx += contribution_x
        fy += contribution_y
        scale += abs(contribution_x) + abs(contribution_y)
    return fx, fy, scale


def relative_error(acc, ref):
    """sqrt( sum (acc - ref)^2 / sum ref^2 ) over every body and coordinate."""
    diff_sq = 0.0
    ref_sq = 0.0
    for i in range(len(ref)):
        for k in range(2):
            d = acc[i][k] - ref[i][k]
            diff_sq += d * d
            ref_sq += ref[i][k] * ref[i][k]
    return math.sqrt(diff_sq / ref_sq)


########################
# 1. Tree structure    #
########################


@pytest.mark.parametrize("max_depth", [2, 3, 5, 20])
def test_every_body_in_exactly_one_chain(max_depth):
    """Every body index 0..N-1 appears once across all leaf chains."""
    pos, masses = clustered_bodies()
    qt, _ = build_tree(pos, masses, len(pos) * 4, max_depth)

    found = all_chained_bodies(qt)
    assert sorted(found) == list(range(len(pos)))


def test_leaf_mass_and_com_match_chain():
    pos, masses = clustered_bodies()
    qt, _ = build_tree(pos, masses, len(pos) * 4, 3)

    chains = leaf_chains(qt)
    for node in chains:
        expected_mass, expected_com = mass_and_com(chains[node], pos, masses)
        np.testing.assert_allclose(qt.mass[node], expected_mass)
        np.testing.assert_allclose(qt.com_pos[node], expected_com, atol=1e-12)


def test_root_mass_and_com():
    pos, masses = make_bodies(500)
    qt, _ = build_tree(pos, masses, len(pos) * 4, 10)

    expected_mass, expected_com = mass_and_com(range(len(pos)), pos, masses)
    np.testing.assert_allclose(qt.mass[0], expected_mass)
    np.testing.assert_allclose(qt.com_pos[0], expected_com, atol=1e-10)


def test_rebuild_is_deterministic():
    pos, masses = clustered_bodies()
    qt1, _ = build_tree(pos, masses, 40, 3)
    qt2, _ = build_tree(pos, masses, 40, 3)

    np.testing.assert_array_equal(qt1.next_idx, qt2.next_idx)
    np.testing.assert_array_equal(qt1.body_idx, qt2.body_idx)
    np.testing.assert_array_equal(qt1.child_indices, qt2.child_indices)


def test_rebuild_after_moving_bodies():
    pos, masses = clustered_bodies()
    build_tree(pos, masses, 40, 3)

    rng = np.random.default_rng(1)
    moved = pos.copy()
    for i in range(len(moved)):
        for k in range(2):
            moved[i][k] += rng.normal(scale=0.3)

    qt, _ = build_tree(moved, masses, 40, 3)
    found = all_chained_bodies(qt)
    assert sorted(found) == list(range(len(pos)))


########################
# 2. Force accuracy    #
########################


@pytest.mark.parametrize("max_depth", [2, 3, 10])
def test_theta_zero_matches_brute_force_clustered(max_depth):
    """Strongest test of the chain logic: no approximation, so must be exact."""
    pos, masses = clustered_bodies()
    acc = tree_acc(pos, masses, theta=0.0, max_depth=max_depth)
    np.testing.assert_allclose(acc, brute_force_acc(pos, masses), rtol=1e-9, atol=1e-12)


def test_theta_zero_matches_brute_force_random():
    pos, masses = make_bodies(300, seed=3)
    acc = tree_acc(pos, masses, theta=0.0, max_depth=3)  # low depth forces buckets
    np.testing.assert_allclose(acc, brute_force_acc(pos, masses), rtol=1e-9, atol=1e-12)


def test_error_small_and_grows_with_theta():
    pos, masses = make_bodies(500, seed=4)
    ref = brute_force_acc(pos, masses)

    errors = {}
    for theta in [0.3, 0.7, 1.5]:
        acc = tree_acc(pos, masses, theta=theta, max_depth=15)
        errors[theta] = relative_error(acc, ref)

    assert errors[0.3] < 0.02
    assert errors[0.3] <= errors[0.7] <= errors[1.5]


########################
# 3. Edge cases        #
########################


def test_all_bodies_same_position():
    n = 20
    pos = np.full((n, 2), 0.3)
    masses = np.ones(n)
    acc = tree_acc(pos, masses, theta=0.5, max_depth=5)

    assert np.all(np.isfinite(acc))
    np.testing.assert_allclose(acc, 0.0, atol=1e-12)  # all r_vec are zero


def test_single_body_zero_acc():
    pos = np.array([[1.0, 2.0]])
    masses = np.array([3.0])
    acc = tree_acc(pos, masses, theta=0.5, max_depth=5)

    assert np.all(np.isfinite(acc))
    np.testing.assert_allclose(acc, 0.0)


def test_two_bodies_equal_and_opposite():
    pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    masses = np.array([2.0, 2.0])
    acc = tree_acc(pos, masses, theta=0.0, max_depth=5)

    np.testing.assert_allclose(acc[0], -acc[1])
    assert acc[0][0] > 0  # body 0 pulled toward +x


def test_bodies_on_quadrant_boundary():
    # center of bounding box is (0, 0); bodies sit exactly on the axes
    pos = np.array(
        [[-1.0, -1.0], [1.0, 1.0], [0.0, 0.5], [0.5, 0.0], [0.0, 0.0], [0.0, -0.5]]
    )
    masses = np.ones(len(pos))
    acc = tree_acc(pos, masses, theta=0.0, max_depth=6)
    np.testing.assert_allclose(acc, brute_force_acc(pos, masses), rtol=1e-9, atol=1e-12)


def test_node_overflow_retry_gives_same_result():
    pos, masses = make_bodies(200, seed=5)
    small = tree_acc(pos, masses, theta=0.5, max_depth=10, max_nodes=1)
    large = tree_acc(pos, masses, theta=0.5, max_depth=10, max_nodes=10_000)
    np.testing.assert_allclose(small, large)


########################
# 4. Invariants        #
########################


def test_net_force_zero_at_theta_zero():
    pos, masses = make_bodies(300, seed=6)
    acc = tree_acc(pos, masses, theta=0.0, max_depth=3)

    fx, fy, scale = net_force(masses, acc)
    assert math.sqrt(fx * fx + fy * fy) < 1e-9 * scale


def test_net_force_small_with_chains_clustered():
    pos, masses = clustered_bodies()
    acc = tree_acc(pos, masses, theta=0.0, max_depth=2)

    fx, fy, _ = net_force(masses, acc)
    assert abs(fx) < 1e-9
    assert abs(fy) < 1e-9
