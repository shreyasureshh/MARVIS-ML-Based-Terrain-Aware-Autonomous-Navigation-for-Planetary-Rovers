"""Config loading and project paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    if not v:
        return default
    p = Path(v)
    return p if p.is_absolute() else ROOT / p


CONFIGS = _env_path("MARVIS_CONFIGS", ROOT / "configs")
DATA = _env_path("MARVIS_DATA", ROOT / "data")
OUTPUTS = _env_path("MARVIS_OUTPUTS", ROOT / "outputs")
MANIFESTS = DATA / "manifests"


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class ClassInfo:
    names: list[str]
    colors: np.ndarray  # (C, 3) uint8
    ignore_index: int

    @property
    def num_classes(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)


def load_classes(path: str | Path | None = None) -> ClassInfo:
    cfg = load_yaml(path or CONFIGS / "classes.yaml")
    cls = sorted(cfg["classes"], key=lambda c: c["id"])
    assert [c["id"] for c in cls] == list(range(len(cls))), "class ids must be 0..C-1"
    return ClassInfo(
        names=[c["name"] for c in cls],
        colors=np.array([c["color"] for c in cls], dtype=np.uint8),
        ignore_index=int(cfg["ignore_index"]),
    )


def colorize(mask: np.ndarray, info: ClassInfo, ignore_color=(0, 0, 0)) -> np.ndarray:
    """Label mask (H, W) -> RGB (H, W, 3)."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[:] = ignore_color
    for i in range(info.num_classes):
        out[mask == i] = info.colors[i]
    return out


def load_dataset_config(path: str | Path | None = None, raw_root: str | Path | None = None) -> dict:
    cfg = load_yaml(path or CONFIGS / "dataset.yaml")
    cfg["raw_root"] = str(resolve(raw_root or os.environ.get("MARVIS_RAW_ROOT") or cfg["raw_root"]))
    cfg["label_map"] = {int(k): int(v) for k, v in cfg["label_map"].items()}
    cfg["null_values"] = [int(v) for v in cfg.get("null_values", [])]
    return cfg
