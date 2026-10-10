"""Paired image/mask augmentations on tensors. img: (3, H, W) float [0, 1]; mask: (H, W) long."""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def hflip(img, mask):
    return img.flip(-1), mask.flip(-1)


def random_resized_crop(img, mask, scale=(0.6, 1.0), ignore_index=255):
    _, h, w = img.shape
    s = random.uniform(*scale)
    ch, cw = int(h * s ** 0.5), int(w * s ** 0.5)
    y, x = random.randint(0, h - ch), random.randint(0, w - cw)
    img = F.interpolate(img[None, :, y:y + ch, x:x + cw], size=(h, w), mode="bilinear", align_corners=False)[0]
    mask = F.interpolate(mask[None, None, y:y + ch, x:x + cw].float(), size=(h, w), mode="nearest")[0, 0].long()
    return img, mask


def photometric(img, strength=0.2):
    b = 1 + random.uniform(-strength, strength)
    c = 1 + random.uniform(-strength, strength)
    g = 1 + random.uniform(-strength, strength)
    m = img.mean()
    img = ((img - m) * c + m) * b
    return img.clamp(0, 1) ** g


def noise_blur(img, sigma=0.03):
    if random.random() < 0.5:
        img = img + torch.randn_like(img) * random.uniform(0, sigma)
    if random.random() < 0.3:
        k = torch.tensor([1.0, 2.0, 1.0])
        k = (k[:, None] * k[None, :]) / 16
        img = F.conv2d(img[:, None], k[None, None], padding=1)[:, 0]
    return img.clamp(0, 1)


def augment(img, mask, level: str, ignore_index=255):
    """level: none | light | strong."""
    if level == "none":
        return img, mask
    if random.random() < 0.5:
        img, mask = hflip(img, mask)
    if level == "light":
        return photometric(img, 0.1), mask
    if level == "strong":
        if random.random() < 0.7:
            img, mask = random_resized_crop(img, mask, ignore_index=ignore_index)
        return noise_blur(photometric(img, 0.3)), mask
    raise ValueError(level)
