"""Class probabilities -> traversability cost grid.

IMAGE-SPACE APPROXIMATION: no ground-plane projection is done, so one grid cell
covers very different ground areas near vs. far from the rover. Terrain class is
a proxy for physical safety, and the class costs in configs/traversability.yaml
are assumptions, not measurements.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import CONFIGS, ClassInfo, load_yaml


@dataclass
class TraversabilityConfig:
    class_costs: np.ndarray        # (C,)
    impassable: np.ndarray         # (C,) bool
    impassable_prob_threshold: float
    grid_size: int
    footprint_radius_cells: int
    high_risk: np.ndarray          # (C,) bool
    raw: dict

    @classmethod
    def from_yaml(cls, info: ClassInfo, path=CONFIGS / "traversability.yaml", **overrides) -> "TraversabilityConfig":
        raw = load_yaml(path)
        raw.update(overrides)
        costs = np.array([float(raw["class_costs"][n]) for n in info.names])
        imp = np.array([n in raw["impassable_classes"] for n in info.names])
        risk = np.array([n in raw["high_risk_classes"] for n in info.names])
        return cls(costs, imp, float(raw["impassable_prob_threshold"]), int(raw["grid_size"]),
                   int(raw["footprint_radius_cells"]), risk, raw)


def expected_cost(probs: np.ndarray, class_costs: np.ndarray) -> np.ndarray:
    """probs (C, H, W) -> per-pixel expected cost E_c[cost] (H, W)."""
    return np.tensordot(class_costs, probs, axes=(0, 0))


def block_mean(x: np.ndarray, grid: int) -> np.ndarray:
    """Average-pool a (..., H, W) array to (..., grid, grid). H, W must be multiples of grid."""
    *lead, h, w = x.shape
    assert h % grid == 0 and w % grid == 0, f"{h}x{w} not divisible by {grid}"
    return x.reshape(*lead, grid, h // grid, grid, w // grid).mean(axis=(-3, -1))


def disk(radius: int) -> np.ndarray:
    r = np.arange(-radius, radius + 1)
    return (r[:, None] ** 2 + r[None, :] ** 2) <= radius ** 2


def inflate(obstacles: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return obstacles.copy()
    return ndimage.binary_dilation(obstacles, structure=disk(radius))


@dataclass
class CostMap:
    terrain_cost: np.ndarray      # (G, G) expected terrain cost in [0, 1]
    obstacles: np.ndarray         # (G, G) bool, raw (before inflation)
    blocked: np.ndarray           # (G, G) bool, after footprint inflation
    uncertainty: np.ndarray | None = None  # (G, G) normalized entropy in [0, 1]

    def total_cost(self, uncertainty_weight: float = 0.0) -> np.ndarray:
        c = self.terrain_cost.copy()
        if uncertainty_weight and self.uncertainty is not None:
            c = c + uncertainty_weight * self.uncertainty
        return c


def build_cost_map(probs: np.ndarray, cfg: TraversabilityConfig, uncertainty: np.ndarray | None = None,
                   valid: np.ndarray | None = None) -> CostMap:
    """probs (C, H, W); uncertainty (H, W) normalized entropy; valid (H, W) bool pixels usable for planning
    (e.g. not the rover body). Invalid cells are treated as blocked."""
    g = cfg.grid_size
    cost_px = expected_cost(probs, cfg.class_costs)
    terrain = block_mean(cost_px, g)
    p_imp = block_mean(probs[cfg.impassable].sum(0), g) if cfg.impassable.any() else np.zeros((g, g))
    obstacles = p_imp >= cfg.impassable_prob_threshold
    if valid is not None:
        obstacles |= block_mean(valid.astype(np.float64), g) < 0.5
    blocked = inflate(obstacles, cfg.footprint_radius_cells)
    unc = block_mean(uncertainty, g) if uncertainty is not None else None
    return CostMap(terrain, obstacles, blocked, unc)


def gt_grid(mask: np.ndarray, num_classes: int, ignore_index: int, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """GT label mask (H, W) -> per-cell class fractions (C, G, G) and labelled fraction (G, G)."""
    onehot = np.stack([(mask == c) for c in range(num_classes)]).astype(np.float64)
    frac = block_mean(onehot, grid)
    labelled = block_mean((mask != ignore_index).astype(np.float64), grid)
    return frac, labelled
