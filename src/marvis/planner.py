"""8-connected A* on a cost grid, three planner variants, uncertainty-driven behaviour, path metrics."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

SQRT2 = math.sqrt(2.0)
MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]

Cell = tuple[int, int]


def octile(a: Cell, b: Cell) -> float:
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dr + dc) + (SQRT2 - 2) * min(dr, dc)


def astar(blocked: np.ndarray, cell_cost: np.ndarray | None, start: Cell, goal: Cell,
          epsilon: float = 1.0) -> list[Cell] | None:
    """Minimise sum over steps of  step_length * (epsilon + cell_cost[next_cell]).

    cell_cost=None with epsilon=1 gives the plain shortest path. Diagonal moves may not
    cut the corner of a blocked cell. Returns the list of cells start..goal, or None.
    """
    h_, w_ = blocked.shape
    if blocked[start] or blocked[goal]:
        return None
    cost = np.zeros(blocked.shape) if cell_cost is None else np.asarray(cell_cost, dtype=np.float64)
    assert (cost >= 0).all(), "cell costs must be non-negative"
    unit = epsilon + float(cost[~blocked].min()) if (~blocked).any() else epsilon  # admissible heuristic scale
    g = np.full(blocked.shape, np.inf)
    parent = -np.ones((h_, w_, 2), dtype=np.int64)
    g[start] = 0.0
    heap = [(unit * octile(start, goal), 0.0, start)]
    closed = np.zeros(blocked.shape, dtype=bool)
    while heap:
        _, gc, cur = heapq.heappop(heap)
        if closed[cur]:
            continue
        if cur == goal:
            break
        closed[cur] = True
        r, c = cur
        for dr, dc, d in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h_ and 0 <= nc < w_) or blocked[nr, nc] or closed[nr, nc]:
                continue
            if dr and dc and (blocked[r + dr, c] or blocked[r, c + dc]):
                continue
            ng = gc + d * (epsilon + cost[nr, nc])
            if ng < g[nr, nc]:
                g[nr, nc] = ng
                parent[nr, nc] = (r, c)
                heapq.heappush(heap, (ng + unit * octile((nr, nc), goal), ng, (nr, nc)))
    if not np.isfinite(g[goal]):
        return None
    path = [goal]
    while path[-1] != start:
        pr, pc = parent[path[-1]]
        path.append((int(pr), int(pc)))
    return path[::-1]


def path_length(path: list[Cell]) -> float:
    return float(sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path, path[1:])))


def path_accumulated(path: list[Cell], grid: np.ndarray) -> float:
    """Length-weighted sum of a per-cell quantity along the path (same weighting the planner uses)."""
    return float(sum(math.hypot(a[0] - b[0], a[1] - b[1]) * grid[b] for a, b in zip(path, path[1:])))


def path_mean(path: list[Cell], grid: np.ndarray) -> float:
    return float(np.mean([grid[p] for p in path]))


def nearest_free(blocked: np.ndarray, cell: Cell) -> Cell | None:
    """Snap a requested start/goal to the nearest unblocked cell (Euclidean)."""
    if not blocked[cell]:
        return cell
    free = np.argwhere(~blocked)
    if len(free) == 0:
        return None
    d = ((free - np.array(cell)) ** 2).sum(1)
    r, c = free[int(d.argmin())]
    return int(r), int(c)


# ----------------------------------------------------------------- planner variants
PLANNERS = ("baseline", "terrain", "uncertainty")


@dataclass
class PlanResult:
    planner: str
    path: list[Cell] | None
    behaviour: str | None = None          # proceed / slow / stop->replanned / stop (only for uncertainty planner)
    uncertainty_weight_used: float | None = None
    replans: int = 0
    log: list[str] = field(default_factory=list)


def behaviour_from_uncertainty(mean_u: float, proceed_below: float, stop_above: float) -> str:
    if mean_u < proceed_below:
        return "proceed"
    if mean_u <= stop_above:
        return "slow"
    return "stop"


def plan(costmap, planner: str, start: Cell, goal: Cell, *, epsilon: float = 0.05,
         uncertainty_weight: float = 1.0, proceed_below: float = 0.25, stop_above: float = 0.45,
         replan_multiplier: float = 3.0, max_replans: int = 2) -> PlanResult:
    """costmap: traversability.CostMap. Start/goal are snapped to the nearest unblocked cell."""
    blocked = costmap.blocked
    s, g = nearest_free(blocked, start), nearest_free(blocked, goal)
    if s is None or g is None:
        return PlanResult(planner, None, log=["no free cell for start/goal"])
    if planner == "baseline":
        return PlanResult(planner, astar(blocked, None, s, g, epsilon=1.0))
    if planner == "terrain":
        return PlanResult(planner, astar(blocked, costmap.terrain_cost, s, g, epsilon=epsilon))
    if planner != "uncertainty":
        raise ValueError(planner)
    if costmap.uncertainty is None:
        raise ValueError("uncertainty planner needs costmap.uncertainty")
    w = uncertainty_weight
    res = PlanResult(planner, None, uncertainty_weight_used=w)
    for k in range(max_replans + 1):
        path = astar(blocked, costmap.total_cost(w), s, g, epsilon=epsilon)
        if path is None:
            res.log.append(f"w={w:.2f}: no path")
            res.path, res.behaviour = None, "stop"
            return res
        mu = path_mean(path, costmap.uncertainty)
        b = behaviour_from_uncertainty(mu, proceed_below, stop_above)
        res.log.append(f"w={w:.2f}: mean path uncertainty {mu:.3f} -> {b}")
        res.path, res.uncertainty_weight_used, res.replans = path, w, k
        if b != "stop":
            res.behaviour = b if k == 0 else f"{b} (after {k} replan{'s' * (k > 1)})"
            return res
        w *= replan_multiplier
    res.behaviour = "stop"  # still too uncertain after max replans: rover halts and waits for new data
    return res


# ------------------------------------------------------------------ GT evaluation
def evaluate_path(path: list[Cell] | None, gt_frac: np.ndarray, labelled: np.ndarray, class_costs: np.ndarray,
                  high_risk: np.ndarray, impassable: np.ndarray, pred_terrain_cost: np.ndarray,
                  min_labelled: float = 0.5) -> dict:
    """Score a path against ground truth (gt_frac: per-cell class fractions (C, G, G)).

    Cells with < min_labelled labelled pixels are excluded from GT ratios ("unknown").
    success = a path exists AND it never enters a cell whose labelled pixels are majority impassable.
    """
    if path is None:
        return {"found": False, "success": False, "length": None, "pred_cost": None, "gt_cost": None,
                "gt_high_risk_frac": None, "gt_impassable_cells": None, "gt_known_frac": None}
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = gt_frac / labelled[None]
    known = np.array([labelled[p] >= min_labelled for p in path])
    risk = np.array([norm[high_risk][:, p[0], p[1]].sum() > 0.5 if k else False for p, k in zip(path, known)])
    imp = np.array([norm[impassable][:, p[0], p[1]].sum() > 0.5 if k else False for p, k in zip(path, known)])
    gt_cost_grid = np.nan_to_num(np.tensordot(class_costs, norm, axes=(0, 0)), nan=0.0)
    return {
        "found": True,
        "success": bool(not imp.any()),
        "length": path_length(path),
        "pred_cost": path_accumulated(path, pred_terrain_cost),
        "gt_cost": path_accumulated(path, gt_cost_grid),
        "gt_high_risk_frac": float(risk[known].mean()) if known.any() else None,
        "gt_impassable_cells": int(imp.sum()),
        "gt_known_frac": float(known.mean()),
    }
