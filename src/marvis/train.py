"""Train one experiment:  python -m marvis.train configs/experiments/<name>.yaml"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import OUTPUTS, load_classes, load_yaml
from .dataset import SegDataset, class_pixel_counts
from .losses import SegLoss, class_weights_from_counts
from .metrics import confusion_matrix, segmentation_scores
from .model import MarvisSegNet, save_checkpoint


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def run_eval(model, loader, criterion, info):
    model.eval()
    cm = np.zeros((info.num_classes, info.num_classes), dtype=np.int64)
    tot, n = 0.0, 0
    for x, y in loader:
        logits = model(x)
        tot += criterion(logits, y).item() * len(x)
        n += len(x)
        cm += confusion_matrix(y.numpy(), logits.argmax(1).numpy(), info.num_classes, info.ignore_index)
    s = segmentation_scores(cm, info.names)
    return tot / max(n, 1), s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--epochs", type=int, help="override")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    if args.epochs:
        cfg["epochs"] = args.epochs
    out = OUTPUTS / "experiments" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    torch.set_num_threads(cfg.get("threads", 4))
    seed_all(cfg.get("seed", 0))
    info = load_classes()

    size = cfg["image_size"]
    train_ds = SegDataset("train", size, cfg["aug"], max_items=cfg.get("max_train_images"))
    val_ds = SegDataset("val", size, "none")
    tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 2),
                    drop_last=True, persistent_workers=cfg.get("num_workers", 2) > 0)
    vl = DataLoader(val_ds, batch_size=cfg["batch_size"], num_workers=cfg.get("num_workers", 2))

    weights = None
    if cfg["loss"] in ("wce", "ce_dice") and cfg.get("class_weight_power", 0.5) > 0:
        weights = class_weights_from_counts(class_pixel_counts("train"), cfg.get("class_weight_power", 0.5))
    criterion = SegLoss(cfg["loss"], info.ignore_index, weights, cfg.get("dice_weight", 1.0))
    model = MarvisSegNet(cfg["arch"], cfg["encoder"], info.num_classes, cfg.get("dropout", 0.2))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    steps = cfg["epochs"] * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)

    hist, best = [], -1.0
    print(f"[{cfg['name']}] train={len(train_ds)} val={len(val_ds)} batches/epoch={len(tl)} "
          f"class_weights={None if weights is None else weights.numpy().round(3).tolist()}", flush=True)
    t_start = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(tl):
            loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
            n += len(x)
            if i % 20 == 0:
                print(f"  ep{ep} it{i}/{len(tl)} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        vloss, vs = run_eval(model, vl, criterion, info)
        rec = {"epoch": ep, "train_loss": tot / n, "val_loss": vloss, "val_mIoU": vs["mIoU"],
               "val_pixel_accuracy": vs["pixel_accuracy"], "epoch_seconds": time.time() - t0,
               "val_per_class_IoU": {k: v["IoU"] for k, v in vs["per_class"].items()}}
        hist.append(rec)
        print(f"[{cfg['name']}] epoch {ep}: train {rec['train_loss']:.4f} val {vloss:.4f} "
              f"mIoU {vs['mIoU']:.4f} acc {vs['pixel_accuracy']:.4f} ({rec['epoch_seconds']:.0f}s)", flush=True)
        if vs["mIoU"] > best:
            best = vs["mIoU"]
            save_checkpoint(model, out / "checkpoints" / "best.pt", {"epoch": ep, "val_mIoU": best, "config": cfg})
        (out / "history.json").write_text(json.dumps(hist, indent=2))
    summary = {"best_val_mIoU": best, "train_seconds": time.time() - t_start, "epochs": cfg["epochs"],
               "n_train": len(train_ds), "n_val": len(val_ds),
               "n_params": sum(p.numel() for p in model.parameters())}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
