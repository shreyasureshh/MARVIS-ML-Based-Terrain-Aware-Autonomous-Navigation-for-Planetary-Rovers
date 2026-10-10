"""Manifest-backed AI4MARS dataset.

Preprocessing contract (applied exactly once, here):
  image -> RGB uint8 (grayscale NavCam replicated to 3 channels) -> resize (bilinear) -> float [0, 1]
  label -> raw values remapped by configs/dataset.yaml `label_map`; null values, unknown values,
           rover-body pixels and beyond-range pixels -> ignore_index; resize with NEAREST.
ImageNet normalisation is NOT done here; it lives inside MarvisSegNet so training,
evaluation and the dashboard cannot disagree.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augment import augment
from .config import MANIFESTS, load_classes, load_dataset_config

MANIFEST_FIELDS = ["key", "image", "label", "rover_mask", "range_mask", "source", "variant", "group", "md5", "subset",
                   "key_rule", "issues"]


PATH_FIELDS = ("image", "label", "rover_mask", "range_mask")


def read_manifest(split: str, manifest_dir: Path | None = None, raw_root: str | Path | None = None) -> list[dict]:
    """Paths in manifests are relative to the dataset root, so manifests are portable across machines."""
    p = (manifest_dir or MANIFESTS) / f"{split}.csv"
    if not p.exists():
        raise FileNotFoundError(f"manifest {p} not found - run `python -m marvis.prepare split` first")
    root = Path(raw_root or load_dataset_config()["raw_root"])
    with open(p, newline="") as f:
        rows = [{k: (v or None) for k, v in row.items()} for row in csv.DictReader(f)]
    for r in rows:
        for k in PATH_FIELDS:
            if r.get(k) and not Path(r[k]).is_absolute():
                r[k] = str(root / r[k])
    return rows


def write_manifest(rows: list[dict], path: Path, raw_root: str | Path | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(raw_root or load_dataset_config()["raw_root"])
    out = []
    for r in rows:
        r = dict(r)
        for k in PATH_FIELDS:
            if r.get(k) and Path(r[k]).is_relative_to(root):
                r[k] = str(Path(r[k]).relative_to(root))
        out.append(r)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)


def build_lut(label_map: dict[int, int], ignore_index: int, max_value: int = 65535) -> np.ndarray:
    lut = np.full(max_value + 1, ignore_index, dtype=np.uint8)
    for raw, cls in label_map.items():
        lut[raw] = cls
    return lut


def map_labels(raw: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return lut[raw.astype(np.int64)]


def load_label(row: dict, lut: np.ndarray, ignore_index: int, size: int | None = None) -> np.ndarray:
    lab = Image.open(row["label"])
    if lab.mode not in ("L", "P", "I", "I;16"):
        lab = lab.convert("L")
    m = map_labels(np.asarray(lab), lut)
    invalid = ~valid_mask(row, m.shape)
    m[invalid] = ignore_index
    if size is not None and m.shape != (size, size):
        m = np.asarray(Image.fromarray(m).resize((size, size), Image.NEAREST))
    return m


def valid_mask(row: dict, shape: tuple[int, int], size: int | None = None) -> np.ndarray:
    """True where the pixel is real terrain the rover could drive on (not rover body, not beyond range).
    Mask convention (VERIFY with the inspector): non-zero = excluded."""
    v = np.ones(shape, dtype=bool)
    for k in ("rover_mask", "range_mask"):
        if row.get(k):
            m = np.asarray(Image.open(row[k]).convert("L"))
            if m.shape == shape:
                v &= m == 0
    if size is not None and v.shape != (size, size):
        v = np.asarray(Image.fromarray(v.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 0
    return v


def load_image(row: dict, size: int | None = None) -> np.ndarray:
    im = Image.open(row["image"]).convert("RGB")
    if size is not None and im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im)


class SegDataset(Dataset):
    def __init__(self, split: str, size: int | None, aug: str = "none", max_items: int | None = None,
                 manifest_dir: Path | None = None, dataset_cfg: dict | None = None):
        self.split, self.size, self.aug = split, size, aug
        cfg = dataset_cfg or load_dataset_config()
        self.rows = read_manifest(split, manifest_dir, cfg["raw_root"])
        if max_items:
            self.rows = self.rows[:max_items]  # manifests are already shuffled deterministically
        info = load_classes()
        self.ignore_index = info.ignore_index
        self.lut = build_lut(cfg["label_map"], self.ignore_index)

    def __len__(self):
        return len(self.rows)

    def item_id(self, i: int) -> str:
        return self.rows[i]["key"]

    def group_of(self, i: int) -> str | None:
        """Evaluation subset name (e.g. test-label agreement level), None if not applicable."""
        return self.rows[i].get("subset")

    def image_bytes(self, i: int) -> bytes:
        return Path(self.rows[i]["image"]).read_bytes()

    def load_image(self, i: int, size: int | None = None) -> np.ndarray:
        return load_image(self.rows[i], size or self.size)

    def load_mask(self, i: int, size: int | None = None) -> np.ndarray:
        return load_label(self.rows[i], self.lut, self.ignore_index, size or self.size)

    def load_valid(self, i: int, size: int | None = None) -> np.ndarray:
        row = self.rows[i]
        shape = Image.open(row["label"]).size[::-1]
        return valid_mask(row, shape, size or self.size)

    def __getitem__(self, i):
        img = torch.from_numpy(self.load_image(i).copy()).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(self.load_mask(i).astype(np.int64))
        img, mask = augment(img, mask, self.aug, self.ignore_index)
        return img.contiguous(), mask.contiguous()


def class_pixel_counts(split: str = "train", manifest_dir: Path | None = None) -> np.ndarray:
    """Labelled-pixel counts per class at native resolution, cached by `prepare split`."""
    p = (manifest_dir or MANIFESTS) / "class_counts.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `python -m marvis.prepare split`")
    d = json.loads(p.read_text())
    names = load_classes().names
    return np.array([d[split][n] for n in names], dtype=np.float64)
