import numpy as np
import pytest

from marvis.config import load_classes
from marvis.traversability import TraversabilityConfig, block_mean, build_cost_map, expected_cost, inflate


@pytest.fixture
def cfg():
    return TraversabilityConfig.from_yaml(load_classes(), grid_size=4, footprint_radius_cells=1)


def test_expected_cost_matches_onehot_table(cfg):
    for c in range(4):
        p = np.zeros((4, 2, 2)); p[c] = 1
        assert np.allclose(expected_cost(p, cfg.class_costs), cfg.class_costs[c])


def test_expected_cost_is_probability_weighted(cfg):
    p = np.zeros((4, 1, 1)); p[0] = 0.5; p[2] = 0.5  # soil/sand
    assert np.isclose(expected_cost(p, cfg.class_costs)[0, 0], 0.5 * (cfg.class_costs[0] + cfg.class_costs[2]))


def test_sand_costs_more_than_soil_and_rock_is_impassable(cfg):
    names = load_classes().names
    c = dict(zip(names, cfg.class_costs))
    assert c["soil"] < c["bedrock"] < c["sand"] < c["big_rock"]
    assert cfg.impassable[names.index("big_rock")] and cfg.impassable.sum() == 1


def test_block_mean():
    x = np.arange(16, dtype=float).reshape(4, 4)
    assert np.allclose(block_mean(x, 2), [[2.5, 4.5], [10.5, 12.5]])


def test_inflation_radius():
    o = np.zeros((7, 7), bool); o[3, 3] = True
    assert inflate(o, 0).sum() == 1
    assert inflate(o, 1).sum() == 5   # plus-shaped disk
    assert inflate(o, 2).sum() == 13


def test_cost_map_marks_rock_and_inflates(cfg):
    p = np.zeros((4, 16, 16)); p[0] = 1  # soil everywhere
    p[:, 0:4, 0:4] = 0; p[3, 0:4, 0:4] = 1  # big rock in top-left grid cell
    cm = build_cost_map(p, cfg)
    assert cm.obstacles[0, 0] and cm.obstacles.sum() == 1
    assert cm.blocked[0, 1] and cm.blocked[1, 0] and not cm.blocked[1, 1]
    assert np.isclose(cm.terrain_cost[3, 3], cfg.class_costs[0])


def test_uncertainty_term(cfg):
    p = np.full((4, 16, 16), 0.25)
    cm = build_cost_map(p, cfg, uncertainty=np.ones((16, 16)))
    assert np.allclose(cm.total_cost(2.0) - cm.total_cost(0.0), 2.0)
