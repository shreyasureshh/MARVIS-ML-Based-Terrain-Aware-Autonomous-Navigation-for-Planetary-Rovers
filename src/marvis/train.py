"""Train one experiment:

  python -m marvis.train configs/experiments/E1_unet_r34_wce_light.yaml [--seed 0] [--epochs N]
                                                                        [--max-train-images N] [--resume]

Model selection uses the VALIDATION split only (best val mIoU); the test split is never
touched here. Outputs go to outputs/experiments/<name>_s<seed>/.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .config import CONFIGS, MANIFESTS, OUTPUTS, ROOT, load_classes, load_yaml
from .dataset import SegDataset, class_pixel_counts
from .losses import SegLoss, class_weights_from_counts
from .metrics import confusion_matrix, segmentation_scores
from .model import MarvisSegNet, save_checkpoint

REQUIRED = ("name", "arch", "encoder", "loss", "aug", "image_size", "batch_size", "lr", "epochs")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _worker_init(base_seed: int, worker_id: int):
    np.random.seed(base_seed * 100 + worker_id)
    random.seed(base_seed * 100 + worker_id)


def validate_config(cfg: dict):
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    grid = load_yaml(CONFIGS / "traversability.yaml")["grid_size"]
    if cfg["image_size"] % grid:
        raise ValueError(f"image_size {cfg['image_size']} must be a multiple of the planning grid ({grid})")
    if cfg["loss"] not in ("ce", "wce", "ce_dice"):
        raise ValueError(f"unknown loss {cfg['loss']}")
    if cfg["aug"] not in ("none", "light", "strong"):
        raise ValueError(f"unknown aug {cfg['aug']}")


def run_metadata(cfg: dict) -> dict:
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:  # noqa: BLE001
            return None
    man = {}
    for s in ("train", "val", "test"):
        p = MANIFESTS / f"{s}.csv"
        man[s] = hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None
    import segmentation_models_pytorch as smp
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "python": sys.version.split()[0], "torch": torch.__version__, "smp": smp.__version__,
            "numpy": np.__version__, "platform": platform.platform(), "manifest_md5": man,
            "seed": cfg["seed"], "started": time.strftime("%Y-%m-%d %H:%M:%S")}


@torch.no_grad()
def run_eval(model, loader, criterion, info, device):
    model.eval()
    cm = np.zeros((info.num_classes, info.num_classes), dtype=np.int64)
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        tot += criterion(logits, y).item() * len(x)
        n += len(x)
        cm += confusion_matrix(y.cpu().numpy(), logits.argmax(1).cpu().numpy(), info.num_classes, info.ignore_index)
    return tot / max(n, 1), segmentation_scores(cm, info.names)


class Logger:
    def __init__(self, path: Path):
        self.f = open(path, "a")

    def __call__(self, msg: str):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-train-images", type=int)
    ap.add_argument("--device", help="cpu | cuda | auto (default from config, else auto)")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/last.pt")
    ap.add_argument("--tag", default="", help="suffix for the run directory (e.g. smoke)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any config key (YAML value)")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    for k in ("seed", "epochs", "max_train_images", "device"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    cfg.setdefault("seed", 0)
    validate_config(cfg)
    run_name = f"{cfg['name']}_s{cfg['seed']}" + (f"_{args.tag}" if args.tag else "")
    out = OUTPUTS / "experiments" / run_name
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    log = Logger(out / "train.log")
    dev = cfg.get("device", "auto")
    device = torch.device("cuda" if dev == "auto" and torch.cuda.is_available() else ("cpu" if dev == "auto" else dev))
    torch.set_num_threads(cfg.get("threads", 4))
    seed_all(cfg["seed"])
    info = load_classes()
    meta = run_metadata(cfg)
    (out / "config.json").write_text(json.dumps({**cfg, "run_name": run_name, "meta": meta}, indent=2))

    size = cfg["image_size"]
    train_ds = SegDataset("train", size, cfg["aug"], max_items=cfg.get("max_train_images"))
    val_ds = SegDataset("val", size, "none", max_items=cfg.get("max_val_images"))
    nw = cfg.get("num_workers", 2)
    g = torch.Generator().manual_seed(cfg["seed"])
    tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw, drop_last=True,
                    persistent_workers=nw > 0, generator=g,
                    worker_init_fn=functools.partial(_worker_init, cfg["seed"]))
    vl = DataLoader(val_ds, batch_size=cfg["batch_size"], num_workers=nw)
    if len(tl) == 0:
        raise ValueError(f"train split has {len(train_ds)} images < batch_size {cfg['batch_size']}")

    weights = None
    if cfg["loss"] in ("wce", "ce_dice") and cfg.get("class_weight_power", 0.5) > 0:
        weights = class_weights_from_counts(class_pixel_counts("train"), cfg.get("class_weight_power", 0.5))
    criterion = SegLoss(cfg["loss"], info.ignore_index, None if weights is None else weights.to(device),
                        cfg.get("dice_weight", 1.0))
    model = MarvisSegNet(cfg["arch"], cfg["encoder"], info.num_classes, cfg.get("dropout", 0.2),
                         pretrained=cfg.get("pretrained", True)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    steps = cfg["epochs"] * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)

    hist, best, start_ep, bad_epochs = [], -1.0, 1, 0
    patience = cfg.get("early_stopping_patience")
    last = out / "checkpoints" / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        hist, best, bad_epochs, start_ep = ck["history"], ck["best"], ck["bad_epochs"], ck["epoch"] + 1
        torch.set_rng_state(ck["rng_torch"]); np.random.set_state(ck["rng_numpy"]); random.setstate(ck["rng_py"])
        log(f"resumed from epoch {ck['epoch']} (best val mIoU {best:.4f})")
    log(f"[{run_name}] device={device} train={len(train_ds)} val={len(val_ds)} batches/epoch={len(tl)} "
        f"class_weights={None if weights is None else weights.numpy().round(3).tolist()}")
    t_start = time.time()
    stopped_early = False
    for ep in range(start_ep, cfg["epochs"] + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(tl):
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep} iter {i}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
            n += len(x)
            if i % 20 == 0:
                log(f"  ep{ep} it{i}/{len(tl)} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
        vloss, vs = run_eval(model, vl, criterion, info, device)
        rec = {"epoch": ep, "train_loss": tot / n, "val_loss": vloss, "val_mIoU": vs["mIoU"],
               "val_pixel_accuracy": vs["pixel_accuracy"], "epoch_seconds": time.time() - t0,
               "val_per_class_IoU": {k: v["IoU"] for k, v in vs["per_class"].items()}}
        hist.append(rec)
        log(f"[{run_name}] epoch {ep}: train {rec['train_loss']:.4f} val {vloss:.4f} "
            f"mIoU {vs['mIoU']:.4f} acc {vs['pixel_accuracy']:.4f} ({rec['epoch_seconds']:.0f}s)")
        if vs["mIoU"] > best:
            best, bad_epochs = vs["mIoU"], 0
            save_checkpoint(model, out / "checkpoints" / "best.pt",
                            {"epoch": ep, "val_mIoU": best, "config": cfg, "run_meta": meta})
        else:
            bad_epochs += 1
        torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": ep, "history": hist, "best": best, "bad_epochs": bad_epochs,
                    "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(), "rng_py": random.getstate()},
                   last)
        (out / "history.json").write_text(json.dumps(hist, indent=2))
        if patience and bad_epochs >= patience:
            log(f"early stopping: no val mIoU improvement for {patience} epochs")
            stopped_early = True
            break
    summary = {"run_name": run_name, "best_val_mIoU": best, "train_seconds": time.time() - t_start + sum(
        r["epoch_seconds"] for r in hist[: start_ep - 1]), "epochs_completed": len(hist), "epochs_planned": cfg["epochs"],
        "stopped_early": stopped_early, "n_train": len(train_ds), "n_val": len(val_ds),
        "n_params": sum(p.numel() for p in model.parameters()), "device": str(device)}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary))


if __name__ == "__main__":
    main()
