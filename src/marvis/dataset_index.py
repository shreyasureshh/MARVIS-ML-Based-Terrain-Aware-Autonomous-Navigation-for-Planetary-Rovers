"""Content-based discovery and validation of AI4MARS image / label / mask files.

Nothing about directory names is hard-coded beyond the tokens in configs/dataset.yaml.
Every file under raw_root is classified by the tokens in its *relative* path:

  path contains `label_dir_token`            -> label PNG  (+ `test_dir_token` -> official test label;
                                                the label's parent directory name is its "variant")
  path contains `rover_mask_dir_token`*      -> rover-body mask
  path contains `range_mask_dir_token`*      -> range mask
  anything else with an image extension      -> camera image

(* = path component starts with the token, e.g. "rng-30m".)

Files are paired by a canonical key. MSL product names differ only in the 3-letter
product type after the spacecraft clock (EDR image, MXY rover mask, RNG range mask)
and test labels carry a "_merged" suffix, so the key is the stem with that product
type and suffix removed. Names that do not match fall back to the plain stem; the
inspector reports how many pairs each rule produced so a wrong assumption is visible.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MSL_KEY = re.compile(r"^(N[LR][AB]_\d{9})[A-Z]{3}(_.+)$")
SUFFIXES = ("_merged",)


def canonical_key(stem: str) -> tuple[str, str]:
    """Return (key, rule) for a file stem."""
    s = stem
    for suf in SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
    m = MSL_KEY.match(s)
    if m:
        return m.group(1) + m.group(2), "msl_product_name"
    return s, "plain_stem"


def _has_part(rel: Path, token: str, prefix: bool) -> bool:
    t = token.lower()
    return any((p.lower().startswith(t) if prefix else p.lower() == t) for p in rel.parts[:-1])


@dataclass
class Record:
    key: str
    label: str
    image: str | None = None
    rover_mask: str | None = None
    range_mask: str | None = None
    source: str = "train"          # "train" (crowd-sourced) or "test" (official expert test set)
    variant: str | None = None     # test-label variant (directory name), None for train labels
    group: str | None = None       # leakage group (rover site/drive) parsed from the image name
    key_rule: str = ""
    md5: str | None = None
    issues: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        d = asdict(self)
        d["issues"] = ";".join(self.issues)
        return d


@dataclass
class Discovery:
    records: list[Record]
    images: dict[str, str]
    unpaired_images: list[str]
    duplicate_image_keys: dict[str, list[str]]
    duplicate_label_keys: list[str]
    unclassified: list[str]
    counts: dict


def discover(raw_root: str | Path, cfg: dict) -> Discovery:
    root = Path(raw_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")
    images: dict[str, list[str]] = {}
    rover: dict[str, str] = {}
    rng: dict[str, str] = {}
    labels: list[tuple[str, Path]] = []
    unclassified = []
    rules = Counter()
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if p.suffix.lower() not in IMAGE_EXTS:
            unclassified.append(str(rel))
            continue
        key, rule = canonical_key(p.stem)
        if _has_part(rel, cfg["label_dir_token"], prefix=False):
            labels.append((key, p))
            rules[f"label:{rule}"] += 1
        elif _has_part(rel, cfg["rover_mask_dir_token"], prefix=True):
            rover[key] = str(p)
        elif _has_part(rel, cfg["range_mask_dir_token"], prefix=True):
            rng[key] = str(p)
        else:
            images.setdefault(key, []).append(str(p))
            rules[f"image:{rule}"] += 1
    group_re = re.compile(cfg["group_regex"]) if cfg.get("group_regex") else None
    records, seen = [], Counter()
    for key, lp in labels:
        rel = lp.relative_to(root)
        is_test = _has_part(rel, cfg["test_dir_token"], prefix=False)
        r = Record(key=key, label=str(lp), source="test" if is_test else "train",
                   variant=lp.parent.name if is_test else None, key_rule=canonical_key(lp.stem)[1])
        seen[(key, r.variant)] += 1
        imgs = images.get(key)
        if not imgs:
            r.issues.append("missing_image")
        else:
            r.image = imgs[0]
            if len(imgs) > 1:
                r.issues.append("ambiguous_image")
            if group_re:
                m = group_re.match(Path(r.image).name)
                r.group = "".join(m.groups()) if m and m.groups() else None
        r.rover_mask, r.range_mask = rover.get(key), rng.get(key)
        records.append(r)
    for r in records:
        if seen[(r.key, r.variant)] > 1 and "duplicate_label" not in r.issues:
            r.issues.append("duplicate_label")
    label_keys = {k for k, _ in labels}
    counts = {
        "image_files": sum(len(v) for v in images.values()), "label_files": len(labels),
        "rover_masks": len(rover), "range_masks": len(rng), "unclassified_files": len(unclassified),
        "key_rules": dict(rules),
        "labels_by_source": dict(Counter(r.source for r in records)),
        "test_variants": dict(Counter(r.variant for r in records if r.variant)),
        "labels_with_rover_mask": sum(r.rover_mask is not None for r in records),
        "labels_with_range_mask": sum(r.range_mask is not None for r in records),
        "group_parsed": sum(r.group is not None for r in records),
    }
    return Discovery(
        records=records,
        images={k: v[0] for k, v in images.items()},
        unpaired_images=sorted(v[0] for k, v in images.items() if k not in label_keys),
        duplicate_image_keys={k: v for k, v in images.items() if len(v) > 1},
        duplicate_label_keys=sorted({f"{k}|{v}" for (k, v), n in seen.items() if n > 1}),
        unclassified=unclassified,
        counts=counts,
    )


def md5_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _open(path: str) -> Image.Image:
    with Image.open(path) as im:
        im.verify()  # integrity check (truncated / corrupt files)
    im = Image.open(path)
    im.load()
    return im


def validate_record(r: Record, cfg: dict, value_counter: Counter | None = None,
                    mask_values: dict | None = None) -> dict:
    """Open and check every file of one record. Appends to r.issues; returns per-file facts."""
    facts: dict = {}
    allowed = set(cfg["label_map"]) | set(cfg["null_values"])
    if r.image:
        try:
            im = _open(r.image)
            facts["image_size"], facts["image_mode"] = im.size, im.mode
            r.md5 = md5_file(r.image)
        except Exception as e:  # noqa: BLE001 - any decode error is a data issue
            r.issues.append(f"corrupt_image:{type(e).__name__}")
    try:
        lab = _open(r.label)
        facts["label_size"], facts["label_mode"] = lab.size, lab.mode
        if lab.mode not in ("L", "P", "I", "I;16"):
            r.issues.append(f"label_mode_{lab.mode}")
            lab = lab.convert("L")
        arr = np.asarray(lab)
        vals, cnt = np.unique(arr, return_counts=True)
        if value_counter is not None:
            value_counter.update(dict(zip(vals.tolist(), cnt.tolist())))
        bad = sorted(set(vals.tolist()) - allowed)
        if bad:
            r.issues.append("unknown_label_values:" + ",".join(map(str, bad[:10])))
        if "image_size" in facts and facts["image_size"] != lab.size:
            r.issues.append("size_mismatch_image_label")
    except Exception as e:  # noqa: BLE001
        r.issues.append(f"corrupt_label:{type(e).__name__}")
    for kind, path in (("rover_mask", r.rover_mask), ("range_mask", r.range_mask)):
        if not path:
            continue
        try:
            m = _open(path)
            if "label_size" in facts and m.size != facts["label_size"]:
                r.issues.append(f"size_mismatch_{kind}")
            if mask_values is not None:
                mask_values.setdefault(kind, Counter()).update(np.unique(np.asarray(m.convert("L"))).tolist())
        except Exception as e:  # noqa: BLE001
            r.issues.append(f"corrupt_{kind}:{type(e).__name__}")
    return facts


FATAL_PREFIXES = ("missing_image", "corrupt_", "size_mismatch_image_label", "ambiguous_image", "duplicate_label")


def is_usable(r: Record) -> bool:
    """Unknown label values are not fatal (they are mapped to ignore); missing/corrupt files are."""
    return not any(i.startswith(FATAL_PREFIXES) for i in r.issues)
