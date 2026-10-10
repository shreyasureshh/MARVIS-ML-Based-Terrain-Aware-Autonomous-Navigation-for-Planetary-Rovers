import math

import numpy as np

from marvis.planner import astar, evaluate_path, nearest_free, path_length, plan
from marvis.traversability import CostMap


def test_straight_line_shortest():
    blocked = np.zeros((10, 10), bool)
    p = astar(blocked, None, (0, 0), (0, 9))
    assert p[0] == (0, 0) and p[-1] == (0, 9)
    assert math.isclose(path_length(p), 9.0)


def test_diagonal_uses_8_connectivity():
    p = astar(np.zeros((5, 5), bool), None, (0, 0), (4, 4))
    assert math.isclose(path_length(p), 4 * math.sqrt(2))


def test_wall_forces_detour_and_no_corner_cutting():
    blocked = np.zeros((7, 7), bool)
    blocked[1:7, 3] = True  # wall with gap only at row 0
    p = astar(blocked, None, (6, 0), (6, 6))
    assert p is not None
    assert all(not blocked[c] for c in p)
    assert (0, 3) in p
    for a, b in zip(p, p[1:]):  # no diagonal squeezing past a blocked corner
        if a[0] != b[0] and a[1] != b[1]:
            assert not blocked[a[0], b[1]] and not blocked[b[0], a[1]]


def test_unreachable_returns_none():
    blocked = np.zeros((5, 5), bool)
    blocked[:, 2] = True
    assert astar(blocked, None, (0, 0), (0, 4)) is None


def test_terrain_planner_avoids_costly_band():
    blocked = np.zeros((9, 9), bool)
    cost = np.zeros((9, 9))
    cost[2:7, 2:7] = 1.0  # expensive patch in the middle
    short = astar(blocked, None, (4, 0), (4, 8))
    safe = astar(blocked, cost, (4, 0), (4, 8), epsilon=0.05)
    sum_cost = lambda p: sum(cost[c] for c in p)
    assert sum_cost(safe) < sum_cost(short)
    assert path_length(safe) > path_length(short)


def test_astar_is_optimal_vs_bruteforce_dijkstra():
    rng = np.random.default_rng(0)
    blocked = rng.random((12, 12)) < 0.2
    blocked[0, 0] = blocked[11, 11] = False
    cost = rng.random((12, 12))
    p = astar(blocked, cost, (0, 0), (11, 11), epsilon=0.1)
    # Dijkstra == A* with zero heuristic: compare against path cost via scipy-free Bellman-Ford
    import heapq
    dist = {(0, 0): 0.0}
    pq = [(0.0, (0, 0))]
    from marvis.planner import MOVES
    while pq:
        d, (r, c) = heapq.heappop(pq)
        if d > dist.get((r, c), np.inf):
            continue
        for dr, dc, L in MOVES:
            n = (r + dr, c + dc)
            if not (0 <= n[0] < 12 and 0 <= n[1] < 12) or blocked[n]:
                continue
            if dr and dc and (blocked[r + dr, c] or blocked[r, c + dc]):
                continue
            nd = d + L * (0.1 + cost[n])
            if nd < dist.get(n, np.inf):
                dist[n] = nd
                heapq.heappush(pq, (nd, n))
    if p is None:
        assert (11, 11) not in dist
    else:
        got = sum(math.hypot(a[0] - b[0], a[1] - b[1]) * (0.1 + cost[b]) for a, b in zip(p, p[1:]))
        assert math.isclose(got, dist[(11, 11)], rel_tol=1e-9)


def test_nearest_free_snaps():
    blocked = np.zeros((5, 5), bool)
    blocked[4, 2] = True
    assert nearest_free(blocked, (4, 2)) != (4, 2)
    assert not blocked[nearest_free(blocked, (4, 2))]


def _cm(unc_value):
    g = 16
    return CostMap(np.full((g, g), 0.1), np.zeros((g, g), bool), np.zeros((g, g), bool), np.full((g, g), unc_value))


def test_behaviour_thresholds():
    assert plan(_cm(0.1), "uncertainty", (15, 8), (0, 8)).behaviour == "proceed"
    assert plan(_cm(0.3), "uncertainty", (15, 8), (0, 8)).behaviour == "slow"
    r = plan(_cm(0.9), "uncertainty", (15, 8), (0, 8), max_replans=2, replan_multiplier=3.0)
    assert r.behaviour == "stop" and r.replans == 2 and math.isclose(r.uncertainty_weight_used, 9.0)


def test_evaluate_path_counts_gt_obstacles():
    g = 4
    frac = np.zeros((4, g, g)); frac[0] = 1.0  # all soil
    frac[:, 1, 1] = 0; frac[3, 1, 1] = 1.0      # one big-rock cell
    labelled = np.ones((g, g))
    costs = np.array([0.1, 0.3, 0.75, 1.0])
    hr = np.array([False, False, True, True]); imp = np.array([False, False, False, True])
    good = evaluate_path([(0, 0), (0, 1), (0, 2)], frac, labelled, costs, hr, imp, np.zeros((g, g)))
    bad = evaluate_path([(0, 0), (1, 1), (2, 2)], frac, labelled, costs, hr, imp, np.zeros((g, g)))
    assert good["success"] and good["gt_high_risk_frac"] == 0
    assert not bad["success"] and bad["gt_impassable_cells"] == 1
    assert evaluate_path(None, frac, labelled, costs, hr, imp, np.zeros((g, g)))["success"] is False


def test_invalid_grids_raise():
    import pytest
    from marvis.planner import InvalidGridError
    b = np.zeros((5, 5), bool)
    with pytest.raises(InvalidGridError):
        astar(b, None, (0, 0), (9, 9))                       # goal out of bounds
    with pytest.raises(InvalidGridError):
        astar(b, np.full((4, 4), 0.1), (0, 0), (4, 4))       # shape mismatch
    c = np.zeros((5, 5)); c[2, 2] = np.nan
    with pytest.raises(InvalidGridError):
        astar(b, c, (0, 0), (4, 4))                          # NaN cost
    with pytest.raises(InvalidGridError):
        astar(b, -np.ones((5, 5)), (0, 0), (4, 4))           # negative cost


def test_blocked_start_or_goal_returns_none_and_expansions_counted():
    b = np.zeros((6, 6), bool); b[0, 0] = True
    assert astar(b, None, (0, 0), (5, 5)) is None
    st = {}
    p = astar(np.zeros((6, 6), bool), None, (0, 0), (5, 5), stats=st)
    assert p and 0 < st["expanded"] <= 36


def test_fully_blocked_grid_uncertainty_planner_stops():
    g = 8
    cm = CostMap(np.zeros((g, g)), np.ones((g, g), bool), np.ones((g, g), bool), np.zeros((g, g)))
    r = plan(cm, "uncertainty", (7, 4), (0, 4))
    assert r.path is None and r.behaviour == "stop"
    assert plan(cm, "baseline", (7, 4), (0, 4)).path is None


def test_uncertainty_planner_avoids_uncertain_region():
    g = 9
    unc = np.zeros((g, g)); unc[2:7, 3:6] = 1.0
    cm = CostMap(np.full((g, g), 0.1), np.zeros((g, g), bool), np.zeros((g, g), bool), unc)
    t = plan(cm, "terrain", (8, 4), (0, 4))
    u = plan(cm, "uncertainty", (8, 4), (0, 4), uncertainty_weight=2.0, proceed_below=0.9, stop_above=0.95)
    mean_u = lambda p: np.mean([unc[c] for c in p])
    assert mean_u(u.path) < mean_u(t.path)
