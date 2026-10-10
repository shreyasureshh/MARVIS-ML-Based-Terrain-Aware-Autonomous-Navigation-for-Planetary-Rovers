"""ML-to-navigation evaluation:  python -m marvis.evaluate_planning <experiment_name> [--n 40]

Uses the cached MC Dropout test predictions (python -m marvis.evaluate <exp> --save-preds),
plans with the baseline / terrain-aware / uncertainty-aware planners on fixed start/goal
pairs, and scores every path against the GROUND-TRUTH labels.
Terrain costs are assumptions (configs/traversability.yaml); only the GT class
fractions along each path are measured.
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import OUTPUTS, load_classes
from .dataset import SegDataset
from .planner import PLANNERS, evaluate_path, plan
from .traversability import TraversabilityConfig, build_cost_map, gt_grid


def start_goal_pairs(g: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Rover sits at the bottom of the NavCam image (near field); goals are in the mid field.
    Fixed for every image (snapped to the nearest free cell if blocked)."""
    s = (g - 4, g // 2)
    r = int(g * 0.4)
    return [(s, (r, g // 6)), (s, (r, g // 2)), (s, (r, 5 * g // 6))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--n", type=int, default=40, help="number of test images")
    args = ap.parse_args(argv)
    info = load_classes()
    tcfg = TraversabilityConfig.from_yaml(info)
    u = tcfg.raw["uncertainty"]
    eps = tcfg.raw["planner"]["step_epsilon"]
    exp = OUTPUTS / "experiments" / args.experiment
    pred_dir = exp / "test_preds"
    ds = SegDataset("test", None, "none")
    rows = []
    n_used = 0
    for i in range(len(ds)):
        if n_used >= args.n:
            break
        f = pred_dir / f"{ds.item_id(i)}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        probs, ent = z["probs"].astype(np.float64), z["entropy"].astype(np.float64)
        size = probs.shape[-1]
        gt = ds.load_mask(i, size)
        valid = ds.load_valid(i, size)
        cm = build_cost_map(probs, tcfg, uncertainty=ent, valid=valid)
        frac, labelled = gt_grid(gt, info.num_classes, info.ignore_index, tcfg.grid_size)
        n_used += 1
        for k, (s, g) in enumerate(start_goal_pairs(tcfg.grid_size)):
            for name in PLANNERS:
                r = plan(cm, name, s, g, epsilon=eps, uncertainty_weight=u["weight"],
                         proceed_below=u["proceed_below"], stop_above=u["stop_above"],
                         replan_multiplier=u["replan_weight_multiplier"])
                ev = evaluate_path(r.path, frac, labelled, tcfg.class_costs, tcfg.high_risk, tcfg.impassable,
                                   cm.terrain_cost)
                rows.append({"image": ds.item_id(i), "pair": k, "planner": name, "behaviour": r.behaviour,
                             "replans": r.replans,
                             "mean_path_uncertainty": float(np.mean([cm.uncertainty[p] for p in r.path])) if r.path else None,
                             **ev})
    df = pd.DataFrame(rows)
    df.to_csv(exp / "planning_results.csv", index=False)
    agg = df.groupby("planner").agg(
        trials=("found", "size"), found_rate=("found", "mean"), success_rate=("success", "mean"),
        length=("length", "mean"), pred_cost=("pred_cost", "mean"), gt_cost=("gt_cost", "mean"),
        gt_high_risk_frac=("gt_high_risk_frac", "mean"), gt_impassable_cells=("gt_impassable_cells", "mean"),
        mean_path_uncertainty=("mean_path_uncertainty", "mean")).reindex(list(PLANNERS))
    beh = df[df.planner == "uncertainty"].behaviour.fillna("none").value_counts().to_dict()
    summary = {"experiment": args.experiment, "n_images": n_used, "pairs_per_image": len(start_goal_pairs(tcfg.grid_size)),
               "table": agg.reset_index().to_dict(orient="records"), "uncertainty_behaviour_counts": beh,
               "config": {k: tcfg.raw[k] for k in ("class_costs", "grid_size", "footprint_radius_cells", "uncertainty")}}
    (exp / "planning_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(agg.round(4).to_string())
    print("behaviour:", beh)
    cols = [("length", "path length (cells)"), ("gt_cost", "accumulated GT terrain cost"),
            ("gt_high_risk_frac", "fraction on GT high-risk terrain"), ("success_rate", "success rate")]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (c, t) in zip(axs, cols):
        ax.bar(agg.index, agg[c], color=["#7f7f7f", "#1f77b4", "#d62728"])
        ax.set_title(t, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"Planner comparison on {n_used} test images x {summary['pairs_per_image']} start/goal pairs")
    fig.tight_layout(); fig.savefig(exp / "planner_comparison.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
