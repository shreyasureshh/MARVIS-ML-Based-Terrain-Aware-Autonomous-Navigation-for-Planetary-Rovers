"""Config loading and project paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"


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


def load_classes(path: str | Path = CONFIGS / "classes.yaml") -> ClassInfo:
    cfg = load_yaml(path)
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
