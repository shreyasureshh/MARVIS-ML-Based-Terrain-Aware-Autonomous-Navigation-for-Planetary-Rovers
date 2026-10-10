"""Model / loss / augmentation / checkpoint tests. Random tensors here are SOFTWARE SMOKE inputs only."""
import random

import numpy as np
import pytest
import torch

from marvis.augment import augment
from marvis.losses import SegLoss, class_weights_from_counts, soft_dice_loss
from marvis.model import MarvisSegNet, load_checkpoint, save_checkpoint
from marvis.uncertainty import predict


@pytest.fixture(scope="module", params=["unet", "deeplabv3plus"])
def model(request):
    torch.manual_seed(0)
    return MarvisSegNet(request.param, "resnet34", 4, 0.3, pretrained=False)


def test_output_shape(model):
    model.eval()
    assert model(torch.rand(2, 3, 64, 64)).shape == (2, 4, 64, 64)


def test_mc_dropout_varies_and_bn_frozen(model):
    model.eval()
    bn = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    rm = bn.running_mean.clone()
    out = model.mc_forward(torch.rand(1, 3, 64, 64), 6)
    assert out.shape == (6, 1, 4, 64, 64)
    assert out.std(0).mean() > 0                                 # samples differ
    assert torch.equal(bn.running_mean, rm) and not bn.training  # BN untouched, eval mode
    assert not model.feat_drop.training and not model.head_drop.training  # dropout restored


def test_deterministic_predict_is_repeatable(model):
    x = torch.rand(1, 3, 64, 64)
    a, b = predict(model, x, 0), predict(model, x, 0)
    assert np.array_equal(a["probs"], b["probs"])
    assert "mutual_information" not in a


def test_predict_mc_outputs(model):
    o = predict(model, torch.rand(1, 3, 64, 64), 5)
    assert np.allclose(o["probs"].sum(1), 1, atol=1e-5)
    assert (o["entropy"] >= 0).all() and (o["entropy"] <= 1 + 1e-6).all()
    assert (o["mutual_information"] >= 0).all() and (o["mutual_information"] <= o["entropy"] + 1e-6).all()


def test_checkpoint_roundtrip(tmp_path, model):
    p = tmp_path / "ck.pt"
    save_checkpoint(model, p, {"config": {"image_size": 64}})
    m2, ck = load_checkpoint(p)
    x = torch.rand(1, 3, 64, 64)
    model.eval()
    assert torch.allclose(model(x), m2(x), atol=1e-6)
    assert ck["meta"]["arch"] == model.meta["arch"] and ck["config"]["image_size"] == 64


@pytest.mark.parametrize("kind", ["ce", "wce", "ce_dice"])
def test_loss_ignores_void_and_backprops(kind):
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    y = torch.randint(0, 4, (2, 8, 8))
    y[:, :4] = 255
    w = class_weights_from_counts(np.array([100, 50, 10, 1.0])) if kind != "ce" else None
    loss = SegLoss(kind, 255, w)(logits, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[:, :, :4].abs().sum() == 0      # ignored pixels get no gradient
    assert logits.grad[:, :, 4:].abs().sum() > 0


def test_all_ignored_batch_dice_is_finite():
    logits = torch.randn(1, 4, 4, 4, requires_grad=True)
    y = torch.full((1, 4, 4), 255)
    d = soft_dice_loss(logits, y, 255)
    assert torch.isfinite(d) and d.item() == 0


def test_class_weights_rare_class_heavier():
    w = class_weights_from_counts(np.array([1000, 100, 10, 1.0]))
    assert torch.all(w[1:] > w[:-1]) and np.isclose(w.mean().item(), 1.0)


@pytest.mark.parametrize("level", ["none", "light", "strong"])
def test_augment_preserves_label_integrity(level):
    random.seed(1); torch.manual_seed(1)
    img = torch.rand(3, 32, 32)
    mask = torch.randint(0, 4, (32, 32)); mask[:5] = 255
    for _ in range(20):
        i2, m2 = augment(img, mask, level, 255)
        assert i2.shape == img.shape and m2.shape == mask.shape and m2.dtype == torch.long
        assert set(m2.unique().tolist()) <= {0, 1, 2, 3, 255}   # nearest-neighbour: no new label values
        assert 0 <= i2.min() and i2.max() <= 1
