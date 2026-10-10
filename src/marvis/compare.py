"""Aggregate experiment runs:  python -m marvis.compare

Groups runs outputs/experiments/<name>_s<seed>[_tag] by experiment name, reports mean ± std
over seeds (or marks a single-seed result as PRELIMINARY), and selects the demo model by
best VALIDATION mIoU (never by test). Writes outputs/experiments_summary.json and
outputs/selected_model.json. Runs tagged (e.g. "_smoke") are excluded.
"""
from __future__ import annotations

import json
import re

import numpy as np

from .config import OUTPUTS

RUN_RE = re.compile(r"^(?P<name>.+)_s(?P<seed>\d+)$")


def collect() -> dict:
    exps: dict = {}
    root = OUTPUTS / "experiments"
    if not root.exists():
        return {}
    for d in sorted(root.iterdir()):
        m = RUN_RE.match(d.name)
        if not m or not (d / "checkpoints" / "best.pt").exists():
            continue
        ts = json.loads((d / "train_summary.json").read_text()) if (d / "train_summary.json").exists() else {}
        mt = json.loads((d / "metrics_test.json").read_text()) if (d / "metrics_test.json").exists() else None
        run = {"run": d.name, "seed": int(m["seed"]), "best_val_mIoU": ts.get("best_val_mIoU"),
               "train_seconds": ts.get("train_seconds"), "epochs_completed": ts.get("epochs_completed")}
        if mt:
            s = mt["segmentation"]["deterministic"]
            u = mt["uncertainty"].get("mc_predictive_entropy", {})
            run.update({"test_mIoU": s["mIoU"], "test_mIoU_mc": mt["segmentation"]["mc"]["mIoU"],
                        "test_pixel_accuracy": s["pixel_accuracy"], "test_mean_dice": s["mean_dice"],
                        "test_IoU": {k: v["IoU"] for k, v in s["per_class"].items()},
                        "error_AUROC_mc_entropy": u.get("error_detection_AUROC"), "ECE_mc": u.get("ECE"),
                        "AURC_mc": u.get("AURC"),
                        "latency_s_mc": mt["efficiency"]["latency_s_mc_median"],
                        "latency_s_det": mt["efficiency"]["latency_s_deterministic_median"],
                        "n_params": mt["efficiency"]["n_params"]})
        exps.setdefault(m["name"], []).append(run)
    return exps


def _ms(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else None, "n": len(v)}


def main():
    exps = collect()
    summary = {}
    for name, runs in exps.items():
        keys = ["best_val_mIoU", "test_mIoU", "test_mIoU_mc", "test_pixel_accuracy", "test_mean_dice",
                "error_AUROC_mc_entropy", "ECE_mc", "AURC_mc", "train_seconds"]
        summary[name] = {"seeds": sorted(r["seed"] for r in runs), "runs": runs,
                         "status": "preliminary (single seed)" if len(runs) < 2 else f"{len(runs)} seeds",
                         **{k: _ms([r.get(k) for r in runs]) for k in keys}}
    (OUTPUTS / "experiments_summary.json").write_text(json.dumps(summary, indent=2))
    best = None
    for name, runs in exps.items():
        for r in runs:
            if r["best_val_mIoU"] is not None and (best is None or r["best_val_mIoU"] > best["best_val_mIoU"]):
                best = r
    if best:
        (OUTPUTS / "selected_model.json").write_text(json.dumps(
            {"experiment": best["run"], "criterion": "highest best_val_mIoU (validation split)",
             "best_val_mIoU": best["best_val_mIoU"]}, indent=2))
    for name, s in summary.items():
        t = s["test_mIoU"]
        print(f"{name:32s} {s['status']:26s} val mIoU {s['best_val_mIoU']['mean']:.4f}  "
              f"test mIoU {'%.4f' % t['mean'] if t else 'not evaluated'}")
    print("selected:", best["run"] if best else None)


if __name__ == "__main__":
    main()
