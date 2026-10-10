"""Data-pipeline tests on a tiny SOFTWARE FIXTURE (generated 32x32 PNGs, not AI4MARS data).

The fixture layout mimics the MSL naming scheme only to exercise discovery / pairing /
validation / leakage logic; it says nothing about the real archive, which must be
checked with `python -m marvis.prepare inspect`.
"""
import json

import numpy as np
import pytest
from PIL import Image

from marvis.config import load_dataset_config
from marvis.dataset import SegDataset, build_lut, class_pixel_counts, map_labels
from marvis.dataset_index import canonical_key, discover, is_usable
from marvis.prepare import inspect, split

S = 32


def _name(sclk, site, drive, product="EDR"):
    return f"NLA_{sclk:09d}{product}_F{site:03d}{drive:04d}AUT_04096M1"


def _png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "ai4mars" / "msl"
    rng = np.random.default_rng(0)
    items = []
    # 12 training labels across 6 site/drive groups (2 images per group = stereo-like pairs)
    for k in range(12):
        sclk, site, drive = 400000000 + k, 1 + k // 2, 0
        items.append(("train", sclk, site, drive))
    # 3 official test labels in two variants, in new groups
    for k in range(3):
        items.append(("test", 500000000 + k, 50 + k, 0))
    for src, sclk, site, drive in items:
        n = _name(sclk, site, drive)
        _png(root / "images" / "edr" / f"{n}.JPG", rng.integers(0, 255, (S, S)))
        _png(root / "images" / "mxy" / f"{_name(sclk, site, drive, 'MXY')}.png", np.pad(np.ones((4, S)), ((S - 4, 0), (0, 0))))
        _png(root / "images" / "rng-30m" / f"{_name(sclk, site, drive, 'RNG')}.png", np.pad(np.ones((4, S)), ((0, S - 4), (0, 0))))
        lab = rng.integers(0, 4, (S, S))
        lab[10:12] = 255
        if src == "train":
            _png(root / "labels" / "train" / f"{n}.png", lab)
        else:
            for v in ("masked-gold-min1-100agree", "masked-gold-min3-100agree"):
                _png(root / "labels" / "test" / v / f"{n}_merged.png", lab)
    # broken cases
    _png(root / "labels" / "train" / f"{_name(410000000, 90, 0)}.png", np.zeros((S, S)))         # no image
    bad = _name(420000000, 91, 0)
    _png(root / "images" / "edr" / f"{bad}.JPG", np.zeros((S, S)))
    _png(root / "labels" / "train" / f"{bad}.png", np.full((S, S), 7))                           # unknown value
    mis = _name(430000000, 92, 0)
    _png(root / "images" / "edr" / f"{mis}.JPG", np.zeros((S * 2, S * 2)))
    _png(root / "labels" / "train" / f"{mis}.png", np.zeros((S, S)))                             # size mismatch
    cor = _name(440000000, 93, 0)
    (root / "images" / "edr" / f"{cor}.JPG").write_bytes(b"not a jpeg")
    _png(root / "labels" / "train" / f"{cor}.png", np.zeros((S, S)))                             # corrupt image
    cfg = load_dataset_config(raw_root=root)
    return cfg, tmp_path


def test_canonical_key_strips_product_and_suffix():
    k1, r1 = canonical_key("NLA_397681339EDR_F0020000AUT_04096M1")
    k2, _ = canonical_key("NLA_397681339MXY_F0020000AUT_04096M1")
    k3, _ = canonical_key("NLA_397681339EDR_F0020000AUT_04096M1_merged")
    assert k1 == k2 == k3 and r1 == "msl_product_name"
    assert canonical_key("random_name")[1] == "plain_stem"


def test_discovery_pairs_and_masks(tree):
    cfg, _ = tree
    d = discover(cfg["raw_root"], cfg)
    assert d.counts["labels_by_source"] == {"train": 16, "test": 6}
    assert d.counts["test_variants"] == {"masked-gold-min1-100agree": 3, "masked-gold-min3-100agree": 3}
    ok = [r for r in d.records if r.key.startswith("NLA_4000")]
    assert all(r.image and r.rover_mask and r.range_mask for r in ok)
    assert ok[0].group == "0010000"
    missing = [r for r in d.records if "missing_image" in r.issues]
    assert len(missing) == 1


def test_inspect_flags_invalid_and_split_has_no_leakage(tree):
    cfg, tmp = tree
    man, out = tmp / "manifests", tmp / "out"
    rep = inspect(cfg, manifest_dir=man, out_dir=out)
    ic = rep["issue_counts"]
    assert ic["missing_image"] == 1 and ic["unknown_label_values"] == 1
    assert ic["size_mismatch_image_label"] == 1 and ic["corrupt_image"] == 1
    assert rep["raw_values_not_in_config"] == [7]
    assert rep["usable_pairs"] == 12 + 6 + 1  # unknown values are not fatal (mapped to ignore)
    assert (out / "sample_overlays.png").exists()
    st = split(cfg, manifest_dir=man, out_dir=out)
    assert set(st["split_sizes"]) == {"train", "val", "test", "test_masked-gold-min1-100agree"}
    assert st["split_sizes"]["test"] == 3
    tr = SegDataset("train", 16, manifest_dir=man, dataset_cfg=cfg)
    va = SegDataset("val", 16, manifest_dir=man, dataset_cfg=cfg)
    te = SegDataset("test", 16, manifest_dir=man, dataset_cfg=cfg)
    g = lambda ds: {r["group"] for r in ds.rows}
    assert not g(tr) & g(va) and not g(tr) & g(te) and not g(va) & g(te)
    assert te.group_of(0) == "masked-gold-min3-100agree"
    counts = class_pixel_counts("train", man)
    assert counts.shape == (4,) and counts.sum() > 0
    assert json.loads((man / "class_counts.json").read_text())["train"]["ignored"] > 0


def test_dataset_item_shapes_ignore_and_masks(tree):
    cfg, tmp = tree
    man, out = tmp / "manifests", tmp / "out"
    inspect(cfg, manifest_dir=man, out_dir=out)
    split(cfg, manifest_dir=man, out_dir=out)
    ds = SegDataset("train", 32, manifest_dir=man, dataset_cfg=cfg)
    x, y = ds[0]
    assert x.shape == (3, 32, 32) and 0 <= x.min() and x.max() <= 1
    assert y.shape == (32, 32) and y.dtype.is_floating_point is False
    vals = set(np.unique(y.numpy()).tolist())
    assert vals <= {0, 1, 2, 3, 255}
    yy = y.numpy()
    assert (yy[10:12] == 255).all()          # null rows
    assert (yy[-4:] == 255).all()            # rover-mask rows -> ignore
    assert (yy[:4] == 255).all()             # range-mask rows -> ignore
    v = ds.load_valid(0)
    assert not v[-4:].any() and not v[:4].any() and v[16].all()


def test_label_lut_unknown_to_ignore():
    lut = build_lut({0: 0, 1: 1, 2: 2, 3: 3}, 255)
    out = map_labels(np.array([[0, 3, 7, 255]]), lut)
    assert out.tolist() == [[0, 3, 255, 255]]


def test_is_usable():
    from marvis.dataset_index import Record
    assert is_usable(Record("k", "l", issues=["unknown_label_values:7"]))
    assert not is_usable(Record("k", "l", issues=["missing_image"]))


def test_manifests_store_relative_paths(tree):
    cfg, tmp = tree
    man, out = tmp / "manifests", tmp / "out"
    inspect(cfg, manifest_dir=man, out_dir=out)
    split(cfg, manifest_dir=man, out_dir=out)
    text = (man / "train.csv").read_text()
    assert cfg["raw_root"] not in text and "images/edr/" in text
    ds = SegDataset("train", 16, manifest_dir=man, dataset_cfg=cfg)
    assert ds[0][0].shape == (3, 16, 16)
