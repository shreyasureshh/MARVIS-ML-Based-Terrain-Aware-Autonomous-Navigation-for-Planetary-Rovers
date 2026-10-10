"""Segmentation models (segmentation_models_pytorch) with dropout for MC Dropout.

Dropout sits on the deepest encoder feature map and before the segmentation head.
For MC Dropout the (deterministic) encoder runs once and only the decoder + head are
sampled T times as one batched call, which keeps CPU inference cheap.
"""
from __future__ import annotations

import inspect
import urllib.request
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from .config import DATA

# torchvision / HuggingFace hosts are blocked in the build environment; timm's GitHub
# release hosts the same ImageNet ResNet weights in torchvision key layout.
PRETRAINED_URLS = {
    "resnet34": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth",
    "resnet18": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet18-5c21b8e5.pth",
}
ARCHS = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}


def fetch_pretrained(encoder: str) -> Path:
    dst = DATA / "weights" / f"{encoder}_imagenet.pth"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PRETRAINED_URLS[encoder], dst)
    return dst


class MarvisSegNet(nn.Module):
    def __init__(self, arch: str = "unet", encoder: str = "resnet34", num_classes: int = 4,
                 dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        net = ARCHS[arch](encoder_name=encoder, encoder_weights=None, classes=num_classes)
        if pretrained:
            sd = torch.load(fetch_pretrained(encoder), map_location="cpu", weights_only=True)
            sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
            net.encoder.load_state_dict(sd, strict=True)  # strict: every encoder weight must match
        self.encoder, self.decoder, self.head = net.encoder, net.decoder, net.segmentation_head
        self.feat_drop = nn.Dropout2d(dropout)
        self.head_drop = nn.Dropout2d(dropout)
        # smp changed decoder.forward(*features) -> forward(features) between versions
        params = list(inspect.signature(self.decoder.forward).parameters.values())
        self._decoder_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.meta = {"arch": arch, "encoder": encoder, "num_classes": num_classes, "dropout": dropout}

    def _decode(self, feats: list[torch.Tensor]) -> torch.Tensor:
        feats = list(feats)
        feats[-1] = self.feat_drop(feats[-1])
        d = self.decoder(*feats) if self._decoder_varargs else self.decoder(feats)
        return self.head(self.head_drop(d))

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: RGB in [0, 1]; ImageNet normalisation happens here."""
        return self.encoder((x - self.mean) / self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._decode(self._encode(x))

    @torch.no_grad()
    def mc_forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """Returns logits (T, B, C, H, W) with dropout active (BatchNorm stays in eval mode)."""
        was_training = self.training
        self.eval()
        feats = self._encode(x)
        for m in (self.feat_drop, self.head_drop):
            m.train()
        b = x.shape[0]
        rep = [f.repeat(T, *[1] * (f.dim() - 1)) for f in feats]
        out = self._decode(rep)
        for m in (self.feat_drop, self.head_drop):
            m.eval()
        self.train(was_training)
        return out.view(T, b, *out.shape[1:])


def save_checkpoint(model: MarvisSegNet, path: Path, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": model.meta, **(extra or {})}, path)


def load_checkpoint(path: Path, map_location="cpu") -> tuple[MarvisSegNet, dict]:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    m = ck["meta"]
    model = MarvisSegNet(m["arch"], m["encoder"], m["num_classes"], m["dropout"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
