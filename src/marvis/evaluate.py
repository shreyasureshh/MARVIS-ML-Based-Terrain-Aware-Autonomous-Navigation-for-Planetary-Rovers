"""Evaluate an experiment on the TEST split:  python -m marvis.evaluate <experiment_name> [--mc 20]

Writes outputs/experiments/<name>/{metrics.json, *.png, test_preds/*.npz}.
Deterministic (dropout off) and MC Dropout predictions are both scored so the
uncertainty comparison is controlled (same weights, same pixels).
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import OUTPUTS, colorize, load_classes
from .dataset import SegDataset
from .metrics import (confusion_matrix, entropy_error_summary, expected_calibration_error, risk_coverage,
                      segmentation_scores)
from .model import load_checkpoint
from .uncertainty import predict

MAX_UNC_PIXELS = 2_000_000
MIN_UNC_PIXELS = 10_000
MIN_PER_OUTCOME = 100
CONVENTIONS = {
    "pixels": "Only labelled pixels (GT != ignore_index; null, rover-body and beyond-range pixels excluded).",
    "segmentation_aggregation": "Confusion matrix summed over all test pixels, then per-class IoU/Dice/precision/"
                                "recall; mIoU = mean over classes present in GT (dataset-level, not per-image mean).",
    "uncertainty_score": "Higher = less certain. Predictive entropy H[E_t p_t] and mutual information "
                         "H[E_t p_t] - E_t H[p_t], both divided by log(C) to lie in [0, 1].",
    "confidence": "max_c of the (MC-mean) predictive probability; used for ECE (15 equal-width bins).",
    "error_detection_AUROC": "Positive = misclassified pixel; score = uncertainty. 0.5 = chance.",
    "AURC": "Mean error rate over all coverage levels when keeping the most certain pixels first (lower is better).",
    "sampling": f"Up to {MAX_UNC_PIXELS:,} labelled pixels sampled uniformly (seed 0) for uncertainty metrics.",
}


def uncertainty_evaluable(err: np.ndarray) -> tuple[bool, str]:
    if len(err) < MIN_UNC_PIXELS:
        return False, f"only {len(err)} labelled pixels (< {MIN_UNC_PIXELS})"
    n_err = int(err.sum())
    if n_err < MIN_PER_OUTCOME or len(err) - n_err < MIN_PER_OUTCOME:
        return False, f"{n_err} errors / {len(err) - n_err} correct: too few of one outcome for AUROC/ECE"
    return True, "ok"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--mc", type=int, default=20)
    ap.add_argument("--split", default="test")
    ap.add_argument("--save-preds", action="store_true", help="cache per-image probs/entropy (for planner eval)")
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    info = load_classes()
    C, IGN = info.num_classes, info.ignore_index
    exp = OUTPUTS / "experiments" / args.experiment
    model, ck = load_checkpoint(exp / "checkpoints" / "best.pt")
    size = ck["config"]["image_size"]
    ds = SegDataset(args.split, size, "none")
    dl = DataLoader(ds, batch_size=1, num_workers=2)

    cm = {"deterministic": np.zeros((C, C), np.int64), "mc": np.zeros((C, C), np.int64)}
    group_cm: dict = {}
    pix = {k: [] for k in ("conf_det", "conf_mc", "ent_det", "ent_mc", "mi_mc", "err_det", "err_mc")}
    lat = {"deterministic": [], "mc": []}
    pred_dir = exp / f"{args.split}_preds"
    if args.save_preds:
        pred_dir.mkdir(exist_ok=True)
    keep_frac = min(1.0, MAX_UNC_PIXELS / max(len(ds) * size * size * 0.6, 1))
    qual, per_image = [], []
    for i, (x, y) in enumerate(dl):
        t0 = time.perf_counter(); det = predict(model, x, 0); lat["deterministic"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); mc = predict(model, x, args.mc); lat["mc"].append(time.perf_counter() - t0)
        gt = y[0].numpy()
        valid = gt != IGN
        pd_, pm = det["probs"][0].argmax(0), mc["probs"][0].argmax(0)
        cm["deterministic"] += confusion_matrix(gt, pd_, C, IGN)
        cm["mc"] += confusion_matrix(gt, pm, C, IGN)
        img_cm = confusion_matrix(gt, pm, C, IGN)
        per_image.append((ds.item_id(i), float(x.mean()), img_cm))
        g = ds.group_of(i)
        if g is not None:
            group_cm.setdefault(g, np.zeros((C, C), np.int64))
            group_cm[g] += img_cm
        sel = valid & (rng.random(gt.shape) < keep_frac)
        pix["conf_det"].append(det["probs"][0].max(0)[sel]); pix["conf_mc"].append(mc["probs"][0].max(0)[sel])
        pix["ent_det"].append(det["entropy"][0][sel]); pix["ent_mc"].append(mc["entropy"][0][sel])
        pix["mi_mc"].append(mc["mutual_information"][0][sel])
        pix["err_det"].append((pd_ != gt)[sel]); pix["err_mc"].append((pm != gt)[sel])
        if args.save_preds:
            np.savez_compressed(pred_dir / f"{ds.item_id(i)}.npz", probs=mc["probs"][0].astype(np.float16),
                                entropy=mc["entropy"][0].astype(np.float16),
                                mi=mc["mutual_information"][0].astype(np.float16),
                                det_probs=det["probs"][0].astype(np.float16))
        if len(qual) < 8 and i % max(1, len(ds) // 8) == 0:
            qual.append((x[0].permute(1, 2, 0).numpy(), gt, pm, mc["entropy"][0]))
    P = {k: np.concatenate(v) for k, v in pix.items()}

    seg = {k: segmentation_scores(v, info.names) for k, v in cm.items()}
    ok, why = uncertainty_evaluable(P["err_mc"])
    ok_d, why_d = uncertainty_evaluable(P["err_det"])
    unc, rc, rel = {}, {}, {}
    if ok_d:
        unc["softmax_entropy (deterministic)"] = {**entropy_error_summary(P["ent_det"], P["err_det"]),
                                                  "ECE": expected_calibration_error(P["conf_det"], ~P["err_det"])["ECE"]}
        rc["deterministic"] = risk_coverage(P["ent_det"], P["err_det"])
        unc["softmax_entropy (deterministic)"]["AURC"] = rc["deterministic"]["AURC"]
        rel["deterministic"] = expected_calibration_error(P["conf_det"], ~P["err_det"])
    if ok:
        unc["mc_predictive_entropy"] = {**entropy_error_summary(P["ent_mc"], P["err_mc"]),
                                        "ECE": expected_calibration_error(P["conf_mc"], ~P["err_mc"])["ECE"]}
        unc["mc_mutual_information"] = entropy_error_summary(P["mi_mc"], P["err_mc"])
        rc["mc"] = risk_coverage(P["ent_mc"], P["err_mc"])
        unc["mc_predictive_entropy"]["AURC"] = rc["mc"]["AURC"]
        rel["mc"] = expected_calibration_error(P["conf_mc"], ~P["err_mc"])
    mc_check = {"mean_mutual_information": float(P["mi_mc"].mean()) if len(P["mi_mc"]) else None,
                "fraction_pixels_MI_gt_1e-4": float((P["mi_mc"] > 1e-4).mean()) if len(P["mi_mc"]) else None}
    mc_check["stochastic"] = bool(mc_check["fraction_pixels_MI_gt_1e-4"] and mc_check["fraction_pixels_MI_gt_1e-4"] > 0)
    n_params = sum(p.numel() for p in model.parameters())
    metrics = {
        "experiment": args.experiment, "split": args.split, "n_images": len(ds), "image_size": size,
        "mc_samples": args.mc, "segmentation": seg, "uncertainty": unc, "risk_coverage": rc, "reliability": rel,
        "by_group": {g: segmentation_scores(v, info.names) for g, v in sorted(group_cm.items())},
        "by_brightness_tertile": brightness_breakdown(per_image, info.names),
        "efficiency": {"n_params": n_params, "size_MB_fp32": n_params * 4 / 1e6,
                       "latency_s_deterministic_median": float(np.median(lat["deterministic"])),
                       "latency_s_mc_median": float(np.median(lat["mc"])), "device": "cpu (4 threads)"},
        "uncertainty_pixels_used": int(len(P["err_mc"])),
        "uncertainty_evaluable": {"mc": [ok, why], "deterministic": [ok_d, why_d]},
        "mc_dropout_variation_check": mc_check,
        "conventions": CONVENTIONS,
    }
    (exp / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2))
    plots(exp, metrics, info, qual, suffix="" if args.split == "test" else f"_{args.split}")
    s = seg["mc"]
    msg = f"[{args.experiment}] {args.split}: mIoU det {seg['deterministic']['mIoU']:.4f} | MC {s['mIoU']:.4f} | " \
          f"acc {s['pixel_accuracy']:.4f}"
    if "mc_predictive_entropy" in unc:
        u = unc["mc_predictive_entropy"]
        msg += f" | entropy correct {u['mean_entropy_correct']:.3f} vs wrong {u['mean_entropy_incorrect']:.3f}" \
               f" | error AUROC {u['error_detection_AUROC']:.3f} | ECE {u['ECE']:.4f}"
    else:
        msg += f" | uncertainty metrics not reported: {why}"
    print(msg + f" | MC varies: {mc_check['stochastic']}")


def brightness_breakdown(per_image, names) -> dict:
    """Image-condition generalisation: split test images into tertiles of mean intensity (descriptive only)."""
    if len(per_image) < 9:
        return {}
    b = np.array([p[1] for p in per_image])
    q = np.quantile(b, [1 / 3, 2 / 3])
    out = {}
    for name, sel in (("dark", b <= q[0]), ("medium", (b > q[0]) & (b <= q[1])), ("bright", b > q[1])):
        cm = sum((p[2] for p, k in zip(per_image, sel) if k), np.zeros_like(per_image[0][2]))
        s = segmentation_scores(cm, names)
        out[name] = {"n_images": int(sel.sum()), "mean_intensity_range": [float(b[sel].min()), float(b[sel].max())],
                     "mIoU": s["mIoU"], "pixel_accuracy": s["pixel_accuracy"],
                     "per_class_IoU": {n: s["per_class"][n]["IoU"] for n in names}}
    return out


def plots(exp, m, info, qual, suffix=""):
    names = info.names
    colors = info.colors / 255.0
    # confusion matrix (row-normalised)
    cmx = np.array(m["segmentation"]["mc"]["confusion_matrix"], dtype=float)
    cmn = cmx / np.maximum(cmx.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for r in range(len(names)):
        for c in range(len(names)):
            ax.text(c, r, f"{cmn[r, c]:.2f}", ha="center", va="center", color="white" if cmn[r, c] > 0.5 else "black")
    ax.set_xticks(range(len(names)), names, rotation=30); ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth"); ax.set_title("Confusion matrix (row-normalised)")
    fig.tight_layout(); fig.savefig(exp / f"confusion_matrix{suffix}.png", dpi=130); plt.close(fig)
    # reliability diagram
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    for k, st in (("deterministic", "o-"), ("mc", "s-")):
        if k not in m["reliability"]:
            continue
        r = m["reliability"][k]
        xs = [c for c in r["bin_confidence"] if c is not None]
        ys = [a for a, c in zip(r["bin_accuracy"], r["bin_confidence"]) if c is not None]
        ax.plot(xs, ys, st, ms=4, label=f"{k} (ECE {r['ECE']:.3f})")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title("Reliability diagram"); ax.legend()
    fig.tight_layout(); fig.savefig(exp / f"reliability{suffix}.png", dpi=130); plt.close(fig)
    # risk-coverage
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    for k in ("deterministic", "mc"):
        if k not in m["risk_coverage"]:
            continue
        r = m["risk_coverage"][k]
        ax.plot(r["coverage"], r["risk"], label=f"{k} (AURC {r['AURC']:.3f})")
    ax.set_xlabel("coverage (most-certain fraction kept)"); ax.set_ylabel("error rate on kept pixels")
    ax.set_title("Risk vs coverage"); ax.legend()
    fig.tight_layout(); fig.savefig(exp / f"risk_coverage{suffix}.png", dpi=130); plt.close(fig)
    # qualitative grid
    if qual:
        fig, axs = plt.subplots(len(qual), 4, figsize=(10, 2.5 * len(qual)))
        axs = np.atleast_2d(axs)
        for r, (img, gt, pr, ent) in enumerate(qual):
            for c, (im, t) in enumerate(((img, "image"), (colorize(gt, info), "ground truth"),
                                         (colorize(pr, info), "prediction (MC mean)"), (ent, "entropy"))):
                axs[r, c].imshow(im, cmap="magma" if c == 3 else None, vmin=0 if c == 3 else None,
                                 vmax=1 if c == 3 else None)
                axs[r, c].axis("off")
                if r == 0:
                    axs[r, c].set_title(t)
        fig.tight_layout(); fig.savefig(exp / f"qualitative{suffix}.png", dpi=90); plt.close(fig)
    # training curves
    hist_p = exp / "history.json"
    if hist_p.exists():
        h = json.loads(hist_p.read_text())
        ep = [r["epoch"] for r in h]
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
        axs[0].plot(ep, [r["train_loss"] for r in h], label="train"); axs[0].plot(ep, [r["val_loss"] for r in h], label="val")
        axs[0].set_title("loss"); axs[0].set_xlabel("epoch"); axs[0].legend()
        axs[1].plot(ep, [r["val_mIoU"] for r in h], label="val mIoU")
        for i, n in enumerate(names):
            axs[1].plot(ep, [r["val_per_class_IoU"][n] or 0 for r in h], ls=":", c=colors[i], label=n)
        axs[1].set_title("validation IoU"); axs[1].set_xlabel("epoch"); axs[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(exp / "training_curves.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
