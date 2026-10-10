"""Dataset inspection and split building.

  python -m marvis.prepare inspect [--raw-root PATH] [--limit N]
  python -m marvis.prepare split   [--raw-root PATH]

`inspect` discovers every file, validates every image/label/mask, and writes
outputs/dataset/inspection_report.json, outputs/dataset/sample_overlays.png and
data/manifests/index.csv. `split` reads index.csv and writes group-aware,
leakage-checked manifests (train/val/test[/test_<variant>]) plus class statistics.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import MANIFESTS, OUTPUTS, colorize, load_classes, load_dataset_config
from .dataset import build_lut, load_image, load_label, read_manifest, write_manifest
from .dataset_index import Record, discover, is_usable, validate_record

DATASET_OUT = OUTPUTS / "dataset"


# ------------------------------------------------------------------ inspect
def inspect(cfg: dict, limit: int | None = None, out_dir: Path = DATASET_OUT, manifest_dir: Path = MANIFESTS) -> dict:
    disc = discover(cfg["raw_root"], cfg)
    recs = disc.records[:limit] if limit else disc.records
    value_counter: dict[str, Counter] = {"train": Counter(), "test": Counter()}
    mask_values: dict = {}
    sizes, modes = Counter(), Counter()
    for r in recs:
        facts = validate_record(r, cfg, value_counter[r.source], mask_values)
        if "image_size" in facts:
            sizes[f"image {facts['image_size'][0]}x{facts['image_size'][1]}"] += 1
            modes[f"image {facts['image_mode']}"] += 1
        if "label_size" in facts:
            sizes[f"label {facts['label_size'][0]}x{facts['label_size'][1]}"] += 1
            modes[f"label {facts['label_mode']}"] += 1
    issues = Counter(i.split(":")[0] for r in recs for i in r.issues)
    usable = [r for r in recs if is_usable(r)]
    info = load_classes()
    known = set(cfg["label_map"]) | set(cfg["null_values"])
    report = {
        "raw_root": cfg["raw_root"],
        "inspected_label_files": len(recs), "limit": limit, "usable_pairs": len(usable),
        "discovery": disc.counts,
        "unpaired_images": len(disc.unpaired_images), "unpaired_image_examples": disc.unpaired_images[:5],
        "duplicate_image_keys": len(disc.duplicate_image_keys),
        "duplicate_label_keys": disc.duplicate_label_keys[:20],
        "unclassified_file_examples": disc.unclassified[:10],
        "issue_counts": dict(issues),
        "issue_examples": {k: next(r.label for r in recs if any(i.startswith(k) for i in r.issues)) for k in issues},
        "sizes": dict(sizes), "modes": dict(modes),
        "raw_label_values": {s: {str(k): int(v) for k, v in sorted(c.items())} for s, c in value_counter.items()},
        "raw_values_not_in_config": sorted({int(k) for c in value_counter.values() for k in c} - known),
        "mask_unique_values": {k: sorted(int(x) for x in v) for k, v in mask_values.items()},
        "config_label_map": {str(k): info.names[v] for k, v in cfg["label_map"].items()},
        "config_null_values": cfg["null_values"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inspection_report.json").write_text(json.dumps(report, indent=2))
    write_manifest([r.as_row() for r in recs], manifest_dir / "index.csv", cfg["raw_root"])
    _overlays(usable, cfg, out_dir / "sample_overlays.png")
    return report


def _overlays(recs: list[Record], cfg: dict, path: Path, n: int = 6):
    pool = [r for r in recs if r.source == "train"] or recs
    if not pool:
        return
    info = load_classes()
    lut = build_lut(cfg["label_map"], info.ignore_index)
    pick = [pool[int(i)] for i in np.linspace(0, len(pool) - 1, min(n, len(pool)))]
    fig, axs = plt.subplots(2, len(pick), figsize=(3 * len(pick), 6.2), squeeze=False)
    for c, r in enumerate(pick):
        row = r.as_row()
        img = load_image(row, 512)
        lab = load_label(row, lut, info.ignore_index, 512)
        ov = (img * 0.5 + colorize(lab, info) * 0.5).astype(np.uint8)
        ov[lab == info.ignore_index] = (img[lab == info.ignore_index] * 0.35).astype(np.uint8)
        axs[0, c].imshow(img); axs[1, c].imshow(ov)
        axs[0, c].set_title(r.key[:22], fontsize=7)
        for a in axs[:, c]:
            a.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=info.colors[i] / 255) for i in range(info.num_classes)]
    fig.legend(handles + [plt.Rectangle((0, 0), 1, 1, color="k")], info.names + ["ignored (null / rover / range)"],
               loc="lower center", ncol=info.num_classes + 1, fontsize=8)
    fig.suptitle("AI4MARS samples: image (top) / label overlay (bottom)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ split
def _group(r: dict) -> str:
    return r["group"] or f"key:{r['key']}"


def _take_groups(rows: list[dict], cap: int | None, rng: np.random.Generator) -> list[dict]:
    """Shuffle rows group-wise and keep whole groups until `cap` rows (None = all)."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(_group(r), []).append(r)
    order = list(groups)
    rng.shuffle(order)
    out = []
    for g in order:
        if cap is not None and len(out) >= cap:
            break
        out.extend(groups[g])
    return out


def split(cfg: dict, manifest_dir: Path = MANIFESTS, out_dir: Path = DATASET_OUT) -> dict:
    idx = manifest_dir / "index.csv"
    if not idx.exists():
        raise FileNotFoundError("run `python -m marvis.prepare inspect` first")
    rows = read_manifest("index", manifest_dir, cfg["raw_root"])
    usable = [r for r in rows if not any(i.startswith(("missing_image", "corrupt_", "size_mismatch_image_label",
                                                         "ambiguous_image", "duplicate_label"))
                                         for i in (r["issues"] or "").split(";") if i)]
    rng = np.random.default_rng(cfg["split_seed"])
    notes = []
    train_pool = [r for r in usable if r["source"] == "train"]
    test_pool = [r for r in usable if r["source"] == "test"]
    # de-duplicate identical image content inside the training pool
    seen_md5, dedup = set(), []
    for r in train_pool:
        if r["md5"] and r["md5"] in seen_md5:
            continue
        seen_md5.add(r["md5"]); dedup.append(r)
    n_dup_train = len(train_pool) - len(dedup)
    train_pool = dedup
    leak = {"train_duplicate_images_removed": n_dup_train}
    if test_pool:
        test_md5 = {r["md5"] for r in test_pool if r["md5"]}
        test_keys = {r["key"] for r in test_pool}
        test_groups = {r["group"] for r in test_pool if r["group"]}
        before = len(train_pool)
        train_pool = [r for r in train_pool if r["key"] not in test_keys and r["md5"] not in test_md5]
        leak["train_images_also_in_test_removed"] = before - len(train_pool)
        before = len(train_pool)
        if cfg.get("drop_test_group_overlap", True):
            train_pool = [r for r in train_pool if r["group"] not in test_groups or r["group"] is None]
        leak["train_images_sharing_test_group_removed"] = before - len(train_pool)
        variants = sorted({r["variant"] for r in test_pool})
        primary = next((v for v in variants if cfg["primary_test_variant"] in v), variants[0])
        tests = {}
        for v in variants:
            vr = [dict(r, subset=v) for r in test_pool if r["variant"] == v]
            tests["test" if v == primary else f"test_{v}"] = _take_groups(vr, cfg.get("max_test_images"),
                                                                          np.random.default_rng(cfg["split_seed"]))
        notes.append(f"Official expert-labelled test set kept intact; primary variant '{primary}', "
                     f"other variants saved as separate evaluation manifests: {[v for v in variants if v != primary]}.")
    else:
        notes.append("No official test labels found: test split carved (group-wise) from the crowd-sourced labels.")
        shuffled = _take_groups(train_pool, None, rng)
        n_test = int(round(len(shuffled) * cfg["val_fraction"]))
        test_rows, cut = [], 0
        while cut < len(shuffled) and len(test_rows) < n_test:
            g = _group(shuffled[cut])
            while cut < len(shuffled) and _group(shuffled[cut]) == g:
                test_rows.append(shuffled[cut]); cut += 1
        train_pool = shuffled[cut:]
        tests = {"test": [dict(r, subset="crowd") for r in test_rows]}
    shuffled = _take_groups(train_pool, None, rng)
    n_val = int(round(len(shuffled) * cfg["val_fraction"]))
    val_rows, cut = [], 0
    while cut < len(shuffled) and len(val_rows) < n_val:
        g = _group(shuffled[cut])
        while cut < len(shuffled) and _group(shuffled[cut]) == g:
            val_rows.append(shuffled[cut]); cut += 1
    splits = {"train": _take_groups(shuffled[cut:], cfg.get("max_train_images"), rng),
              "val": _take_groups(val_rows, cfg.get("max_val_images"), rng), **tests}
    # final leakage assertions
    for a in splits:
        for b in splits:
            if a < b and not (a.startswith("test") and b.startswith("test")):
                ka = {r["key"] for r in splits[a]}; kb = {r["key"] for r in splits[b]}
                ga = {_group(r) for r in splits[a]}; gb = {_group(r) for r in splits[b]}
                assert not ka & kb, f"image overlap {a}/{b}"
                if not (a.startswith("test") or b.startswith("test")):
                    assert not ga & gb, f"group overlap {a}/{b}"
    for name, rs in splits.items():
        write_manifest(rs, manifest_dir / f"{name}.csv", cfg["raw_root"])
    info = load_classes()
    lut = build_lut(cfg["label_map"], info.ignore_index)
    counts = {}
    for name, rs in splits.items():
        c = np.zeros(info.num_classes + 1, dtype=np.int64)
        for r in rs:
            m = load_label(r, lut, info.ignore_index)
            c += np.bincount(np.where(m == info.ignore_index, info.num_classes, m).ravel(),
                             minlength=info.num_classes + 1)
        counts[name] = {**{n: int(c[i]) for i, n in enumerate(info.names)}, "ignored": int(c[-1])}
    (manifest_dir / "class_counts.json").write_text(json.dumps(counts, indent=2))
    sizes = {k: len(v) for k, v in splits.items()}
    groups = {k: len({_group(r) for r in v}) for k, v in splits.items()}
    frac = {k: {n: v[n] / max(sum(v[m] for m in info.names), 1) for n in info.names} for k, v in counts.items()}
    stats = {"split_sizes": sizes, "split_groups": groups, "class_counts": counts, "class_fraction": frac,
             "leakage": leak, "notes": notes, "split_seed": cfg["split_seed"],
             "group_regex": cfg.get("group_regex"), "groups_parsed_fraction":
                 {k: float(np.mean([r["group"] is not None for r in v])) if v else None for k, v in splits.items()}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "split_report.json").write_text(json.dumps(stats, indent=2))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    _class_plot(frac, info, out_dir / "class_distribution.png")
    return stats


def _class_plot(frac: dict, info, path: Path):
    names = list(frac)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    w = 0.8 / max(len(names), 1)
    for j, s in enumerate(names):
        ax.bar(np.arange(info.num_classes) + j * w, [frac[s][n] for n in info.names], w, label=s)
    ax.set_xticks(np.arange(info.num_classes) + w * (len(names) - 1) / 2, info.names)
    ax.set_ylabel("fraction of labelled pixels"); ax.set_title("Class distribution per split"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inspect", "split"])
    ap.add_argument("--raw-root")
    ap.add_argument("--limit", type=int, help="inspect only the first N label files")
    a = ap.parse_args(argv)
    cfg = load_dataset_config(raw_root=a.raw_root)
    if a.cmd == "inspect" and not Path(cfg["raw_root"]).is_dir():
        raise SystemExit(
            f"AI4MARS not found at {cfg['raw_root']}.\n"
            "Provide it with one of:\n"
            "  python -m marvis.fetch_subset --zenodo-record 15995036 --n-train 1000 --out data/raw/ai4mars\n"
            "  export MARVIS_RAW_ROOT=/path/to/extracted/ai4mars   (or --raw-root PATH)\n"
            "See README section 'Dataset setup' for the expected layout.")
    if a.cmd == "inspect":
        rep = inspect(cfg, a.limit)
        print(json.dumps({k: rep[k] for k in ("raw_root", "inspected_label_files", "usable_pairs", "discovery",
                                              "issue_counts", "raw_label_values", "raw_values_not_in_config",
                                              "mask_unique_values", "sizes")}, indent=2))
        print(f"report: {DATASET_OUT / 'inspection_report.json'}")
    else:
        st = split(cfg)
        print(json.dumps({k: st[k] for k in ("split_sizes", "split_groups", "leakage", "notes")}, indent=2))


if __name__ == "__main__":
    main()
