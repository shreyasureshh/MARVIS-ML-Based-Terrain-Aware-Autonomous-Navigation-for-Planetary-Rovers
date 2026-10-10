"""ML-to-navigation evaluation.

  # 1) calibrate proceed/slow/stop thresholds on VALIDATION predictions
  python -m marvis.evaluate <run> --split val --save-preds
  python -m marvis.evaluate_planning <run> --split val --calibrate
  # 2) evaluate on TEST with the frozen thresholds
  python -m marvis.evaluate <run> --split test --save-preds
  python -m marvis.evaluate_planning <run> --split test --n 40

Every path is scored against GROUND-TRUTH labels. Terrain costs, grid geometry,
footprint and thresholds are assumptions (configs/traversability.yaml), not measurements.
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
from .planner import PLANNERS, astar, evaluate_path, nearest_free, path_mean, plan
from .traversability import TraversabilityConfig, build_cost_map, gt_grid

MIN_LABELLED_PATH_FRAC = 0.5


def start_goal_pairs(g: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Image-space convention: the rover is at the bottom-centre of the NavCam image (near field),
    goals are in the mid field. Same pairs for every image; snapped to the nearest free cell."""
    s = (g - 4, g // 2)
    r = int(g * 0.4)
    return [(s, (r, g // 6)), (s, (r, g // 2)), (s, (r, 5 * g // 6))]


def _load(ds, i, pred_dir, info, tcfg):
    f = pred_dir / f"{ds.item_id(i)}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    probs, ent = z["probs"].astype(np.float64), z["entropy"].astype(np.float64)
    size = probs.shape[-1]
    gt = ds.load_mask(i, size)
    valid = ds.load_valid(i, size)
    cm = build_cost_map(probs, tcfg, uncertainty=ent, valid=valid)
    frac, labelled = gt_grid(gt, info.num_classes, info.ignore_index, tcfg.grid_size)
    return cm, frac, labelled


def calibrate(ds, pred_dir, info, tcfg, eps, q_a, q_b) -> dict:
    """A/B = quantiles of the mean uncertainty along terrain-aware paths on validation images."""
    vals = []
    for i in range(len(ds)):
        z = _load(ds, i, pred_dir, info, tcfg)
        if z is None:
            continue
        cm = z[0]
        for s, g in start_goal_pairs(tcfg.grid_size):
            s2, g2 = nearest_free(cm.blocked, s), nearest_free(cm.blocked, g)
            if s2 is None or g2 is None:
                continue
            p = astar(cm.blocked, cm.terrain_cost, s2, g2, epsilon=eps)
            if p:
                vals.append(path_mean(p, cm.uncertainty))
    if len(vals) < 10:
        raise RuntimeError(f"only {len(vals)} validation paths - not enough to calibrate thresholds")
    a, b = np.quantile(vals, [q_a, q_b])
    return {"proceed_below": float(a), "stop_above": float(b), "quantiles": [q_a, q_b], "n_paths": len(vals),
            "source": "validation split, terrain-aware paths"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=40, help="max number of images")
    ap.add_argument("--calibrate", action="store_true", help="fit A/B thresholds on this split (use val)")
    args = ap.parse_args(argv)
    info = load_classes()
    tcfg = TraversabilityConfig.from_yaml(info)
    u = tcfg.raw["uncertainty"]
    eps = tcfg.raw["planner"]["step_epsilon"]
    exp = OUTPUTS / "experiments" / args.experiment
    pred_dir = exp / f"{args.split}_preds"
    if not pred_dir.exists():
        raise FileNotFoundError(f"{pred_dir} missing - run `python -m marvis.evaluate {args.experiment} "
                                f"--split {args.split} --save-preds` first")
    ds = SegDataset(args.split, None, "none")
    thr_file = exp / "behaviour_thresholds.json"
    if args.calibrate:
        if args.split.startswith("test"):
            raise SystemExit("refusing to calibrate thresholds on a test split")
        thr = calibrate(ds, pred_dir, info, tcfg, eps, u.get("calibration_quantiles", [0.5, 0.9])[0],
                        u.get("calibration_quantiles", [0.5, 0.9])[1])
        thr_file.write_text(json.dumps(thr, indent=2))
        print(json.dumps(thr, indent=2))
        return
    if thr_file.exists():
        thr = json.loads(thr_file.read_text())
    else:
        thr = {"proceed_below": u["proceed_below"], "stop_above": u["stop_above"],
               "source": "configs/traversability.yaml (NOT calibrated)"}
    rows, n_used, skipped = [], 0, []
    for i in range(len(ds)):
        if n_used >= args.n:
            break
        z = _load(ds, i, pred_dir, info, tcfg)
        if z is None:
            continue
        cm, frac, labelled = z
        if cm.blocked.all() or labelled.mean() < 0.2:
            skipped.append(ds.item_id(i))
            continue
        n_used += 1
        for k, (s, g) in enumerate(start_goal_pairs(tcfg.grid_size)):
            for name in PLANNERS:
                r = plan(cm, name, s, g, epsilon=eps, uncertainty_weight=u["weight"],
                         proceed_below=thr["proceed_below"], stop_above=thr["stop_above"],
                         replan_multiplier=u["replan_weight_multiplier"])
                ev = evaluate_path(r.path, frac, labelled, tcfg.class_costs, tcfg.high_risk, tcfg.impassable,
                                   cm.terrain_cost)
                rows.append({"image": ds.item_id(i), "pair": k, "planner": name, "behaviour": r.behaviour,
                             "halted": r.behaviour == "stop", "replans": r.replans, "expanded_nodes": r.expanded,
                             "mean_path_uncertainty": path_mean(r.path, cm.uncertainty) if r.path else None,
                             "gt_evaluable": bool(ev["gt_known_frac"] is not None
                                                  and ev["gt_known_frac"] >= MIN_LABELLED_PATH_FRAC), **ev})
    if not rows:
        raise RuntimeError("no evaluable images")
    df = pd.DataFrame(rows)
    df.to_csv(exp / f"planning_results_{args.split}.csv", index=False)
    agg = df.groupby("planner").agg(
        trials=("found", "size"), found_rate=("found", "mean"), success_rate=("success", "mean"),
        halted_rate=("halted", "mean"), length=("length", "mean"), pred_cost=("pred_cost", "mean"),
        gt_cost=("gt_cost", "mean"), gt_high_risk_frac=("gt_high_risk_frac", "mean"),
        gt_impassable_cells=("gt_impassable_cells", "mean"), expanded_nodes=("expanded_nodes", "mean"),
        mean_path_uncertainty=("mean_path_uncertainty", "mean"), gt_known_frac=("gt_known_frac", "mean"),
    ).reindex(list(PLANNERS))
    beh = df[df.planner == "uncertainty"].behaviour.fillna("none").value_counts().to_dict()
    summary = {"experiment": args.experiment, "split": args.split, "n_images": n_used, "skipped_images": skipped,
               "pairs_per_image": len(start_goal_pairs(tcfg.grid_size)), "thresholds": thr,
               "table": agg.reset_index().to_dict(orient="records"), "uncertainty_behaviour_counts": beh,
               "sufficient_images": n_used >= 30,
               "config": {k: tcfg.raw[k] for k in ("class_costs", "grid_size", "footprint_radius_cells", "uncertainty")},
               "definitions": {
                   "success": "path found AND no cell on it is majority ground-truth impassable (big rock)",
                   "gt_high_risk_frac": "fraction of labelled path cells whose GT is majority sand/big rock",
                   "gt_cost": "length-weighted assumed cost of the GT terrain along the path",
                   "pred_cost": "length-weighted predicted expected cost along the path",
                   "halted": "uncertainty-aware rover stopped (path still too uncertain after max replans)",
                   "length": "image-grid cells (NOT metres)"}}
    (exp / f"planning_summary_{args.split}.json").write_text(json.dumps(summary, indent=2, default=float))
    print(agg.round(4).to_string())
    print("behaviour:", beh, "| thresholds:", thr["source"])
    if n_used < 30:
        print(f"WARNING: only {n_used} evaluable images (< 30) - results are preliminary")
    cols = [("length", "path length (grid cells)"), ("gt_cost", "accumulated GT terrain cost"),
            ("gt_high_risk_frac", "fraction on GT high-risk terrain"), ("success_rate", "success rate"),
            ("expanded_nodes", "A* expanded nodes")]
    fig, axs = plt.subplots(1, len(cols), figsize=(17, 3.4))
    for ax, (c, t) in zip(axs, cols):
        ax.bar(agg.index, agg[c], color=["#7f7f7f", "#1f77b4", "#2ca02c"])
        ax.set_title(t, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"Planner comparison on {n_used} {args.split} images × {summary['pairs_per_image']} start/goal pairs "
                 f"(costs assumed; risk measured vs GT labels)", fontsize=10)
    fig.tight_layout(); fig.savefig(exp / f"planner_comparison_{args.split}.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
