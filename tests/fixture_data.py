"""SOFTWARE FIXTURE generator: tiny AI4MARS-shaped tree of generated PNGs.

Used only to test code paths. Nothing produced from it is an ML result.
"""
import numpy as np
from PIL import Image


def name(sclk, site, drive, product="EDR"):
    return f"NLA_{sclk:09d}{product}_F{site:03d}{drive:04d}AUT_04096M1"


def png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def make_tree(root, n_train=40, n_test=12, size=64, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    for k in range(n_train + n_test):
        src = "train" if k < n_train else "test"
        sclk, site = 400000000 + k, 1 + k // 2
        n = name(sclk, site, 0)
        lab = np.zeros((size, size), np.uint8)              # soil background
        lab[yy > size * 0.6] = 1                            # bedrock band
        cx, cy, r = rng.integers(8, size - 8, 3)
        lab[(xx - cx) ** 2 + (yy - cy) ** 2 < (r % 8 + 3) ** 2] = 3   # rock blob
        lab[: size // 8] = 2                                # sand far field
        img = (lab * 50 + rng.integers(0, 40, lab.shape)).astype(np.uint8)
        png(root / "images" / "edr" / f"{n}.JPG", img)
        rover = np.zeros((size, size)); rover[-size // 16:] = 1
        png(root / "images" / "mxy" / f"{name(sclk, site, 0, 'MXY')}.png", rover)
        lab_out = lab.copy(); lab_out[rng.random(lab.shape) < 0.05] = 255
        if src == "train":
            png(root / "labels" / "train" / f"{n}.png", lab_out)
        else:
            png(root / "labels" / "test" / "masked-gold-min3-100agree" / f"{n}_merged.png", lab_out)
    return root
