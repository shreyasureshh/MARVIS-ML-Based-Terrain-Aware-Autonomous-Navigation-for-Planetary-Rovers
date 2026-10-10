"""Losses: (weighted) cross-entropy and CE + soft Dice, both honouring ignore_index."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights_from_counts(counts: np.ndarray, power: float = 0.5) -> torch.Tensor:
    """Inverse-frequency weights ** power (0.5 = 'sqrt' damping), normalised to mean 1."""
    freq = counts / counts.sum()
    w = (1.0 / np.maximum(freq, 1e-6)) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def soft_dice_loss(logits, target, ignore_index: int, eps: float = 1.0):
    c = logits.shape[1]
    valid = (target != ignore_index)
    t = target.clone()
    t[~valid] = 0
    p = logits.softmax(1) * valid[:, None]
    oh = F.one_hot(t, c).permute(0, 3, 1, 2).float() * valid[:, None]
    inter = (p * oh).sum((0, 2, 3))
    denom = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    present = oh.sum((0, 2, 3)) > 0
    return 1 - dice[present].mean() if present.any() else logits.sum() * 0


class SegLoss(nn.Module):
    def __init__(self, kind: str, ignore_index: int, weights: torch.Tensor | None = None, dice_weight: float = 1.0):
        super().__init__()
        self.kind, self.ignore_index, self.dice_weight = kind, ignore_index, dice_weight
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)

    def forward(self, logits, target):
        loss = self.ce(logits, target)
        if self.kind == "ce_dice":
            loss = loss + self.dice_weight * soft_dice_loss(logits, target, self.ignore_index)
        elif self.kind not in ("ce", "wce"):
            raise ValueError(self.kind)
        return loss
