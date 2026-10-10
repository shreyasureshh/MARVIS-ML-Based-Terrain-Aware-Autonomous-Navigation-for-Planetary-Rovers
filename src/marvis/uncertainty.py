"""MC Dropout inference and predictive-uncertainty measures."""
from __future__ import annotations

import math

import numpy as np
import torch


def entropy(p: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    return -(p * (p + eps).log()).sum(dim)


@torch.no_grad()
def predict(model, x: torch.Tensor, mc_samples: int = 0) -> dict:
    """x: (B, 3, H, W). Returns numpy arrays:
    probs (B, C, H, W) mean predictive probs, entropy (B, H, W) predictive entropy normalized to [0, 1],
    and for MC: mutual_information (B, H, W) (epistemic part, also normalized by log C).

    mc_samples=0 -> deterministic softmax (dropout off).
    """
    model.eval()
    if mc_samples <= 0:
        p = model(x).softmax(1)
        c = p.shape[1]
        return {"probs": p.cpu().numpy(), "entropy": (entropy(p) / math.log(c)).cpu().numpy()}
    samples = model.mc_forward(x, mc_samples).softmax(2)  # (T, B, C, H, W)
    p = samples.mean(0)
    c = p.shape[1]
    h = entropy(p)
    expected_h = entropy(samples, dim=2).mean(0)
    return {
        "probs": p.cpu().numpy(),
        "entropy": (h / math.log(c)).cpu().numpy(),
        "mutual_information": ((h - expected_h).clamp_min(0) / math.log(c)).cpu().numpy(),
    }


def normalized_entropy_np(p: np.ndarray, axis: int = 0) -> np.ndarray:
    c = p.shape[axis]
    return -(p * np.log(p + 1e-12)).sum(axis) / math.log(c)
