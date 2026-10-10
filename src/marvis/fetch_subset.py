"""Fetch a SUBSET of the AI4MARS zip archive without downloading all of it.

Uses HTTP Range requests: the zip central directory (at the end of the file) is read
first, then only the members you select are downloaded and decompressed.

  # 1) look at the real archive structure first (downloads only the directory, a few MB)
  python -m marvis.fetch_subset --zenodo-record 15995036 --list-only
  # 2) fetch ~1000 crowd-labelled pairs + the official test labels (with their images/masks)
  python -m marvis.fetch_subset --zenodo-record 15995036 --n-train 1000 --n-test 300 --out data/raw/ai4mars

  (or pass --url <direct .zip URL> instead of --zenodo-record)

Requires the server to support Range requests (Zenodo does). Nothing is downloaded
beyond the selected members; the script prints the byte count before and after.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path, PurePosixPath

from .config import ROOT, resolve
from .dataset_index import IMAGE_EXTS, canonical_key


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests with a small block cache."""

    def __init__(self, url: str, block: int = 1 << 20, max_blocks: int = 64, timeout: int = 60):
        self.url, self.block, self.max_blocks, self.timeout = url, block, max_blocks, timeout
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            self.size = int(r.headers["Content-Length"])
            if r.headers.get("Accept-Ranges", "").lower() != "bytes":
                # some servers omit the header but still honour ranges; probe one byte
                self._fetch(0, 0)
        self.pos = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.bytes_downloaded = 0

    def _fetch(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if r.status != 206:
                raise IOError(f"server ignored Range request (HTTP {r.status}); refusing to download the full file")
            data = r.read()
        if hasattr(self, "bytes_downloaded"):
            self.bytes_downloaded += len(data)
        return data

    def _block(self, i: int) -> bytes:
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        start = i * self.block
        data = self._fetch(start, min(start + self.block, self.size) - 1)
        self.cache[i] = data
        if len(self.cache) > self.max_blocks:
            self.cache.popitem(last=False)
        return data

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: off, io.SEEK_CUR: self.pos + off, io.SEEK_END: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            i, o = divmod(self.pos, self.block)
            chunk = self._block(i)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def zenodo_zip_url(record: str) -> str:
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60) as r:
        meta = json.load(r)
    files = [f for f in meta.get("files", []) if f["key"].lower().endswith(".zip")]
    if not files:
        raise RuntimeError(f"no .zip file in Zenodo record {record}: {[f['key'] for f in meta.get('files', [])]}")
    f = max(files, key=lambda f: f.get("size", 0))
    print(f"Zenodo record {record}: {f['key']} ({f.get('size', 0) / 1e9:.2f} GB)")
    return f["links"]["self"]


def safe_member(name: str) -> bool:
    p = PurePosixPath(name)
    return not p.is_absolute() and ".." not in p.parts and not name.startswith(("/", "\\"))


def classify(name: str, tokens: dict) -> str | None:
    parts = [p.lower() for p in PurePosixPath(name).parts[:-1]]
    if PurePosixPath(name).suffix.lower() not in IMAGE_EXTS:
        return None
    if tokens["label"] in parts:
        return "test_label" if tokens["test"] in parts else "train_label"
    if any(p.startswith(tokens["rover"]) for p in parts):
        return "rover"
    if any(p.startswith(tokens["range"]) for p in parts):
        return "range"
    return "image"


def select(names: list[str], n_train: int, n_test: int | None, seed: int, tokens: dict, mission: str | None) -> list[str]:
    by_kind: dict[str, dict[str, list[str]]] = {}
    for nm in names:
        if mission and mission.lower() not in [p.lower() for p in PurePosixPath(nm).parts]:
            continue
        k = classify(nm, tokens)
        if k:
            by_kind.setdefault(k, {}).setdefault(canonical_key(PurePosixPath(nm).stem)[0], []).append(nm)
    rng = random.Random(seed)
    train_keys = sorted(by_kind.get("train_label", {}))
    test_keys = sorted(by_kind.get("test_label", {}))
    imgs = by_kind.get("image", {})
    train_keys = [k for k in train_keys if k in imgs]
    test_keys = [k for k in test_keys if k in imgs]
    rng.shuffle(train_keys)
    rng.shuffle(test_keys)
    keys = set(train_keys[:n_train]) | set(test_keys if n_test is None else test_keys[:n_test])
    chosen = []
    for kind in ("image", "train_label", "test_label", "rover", "range"):
        for k, ms in by_kind.get(kind, {}).items():
            if k in keys:
                chosen.extend(ms)
    return sorted(set(chosen))


def main(argv=None):
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url")
    src.add_argument("--zenodo-record")
    ap.add_argument("--list-only", action="store_true", help="write the archive listing and stop")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=None, help="default: all official test labels")
    ap.add_argument("--mission", default="msl", help="only members under a directory with this name ('' = any)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/raw/ai4mars")
    ap.add_argument("--max-gb", type=float, default=2.0, help="refuse if the selection exceeds this (compressed)")
    ap.add_argument("--block-kb", type=int, default=1024, help="HTTP range read size")
    a = ap.parse_args(argv)
    url = a.url or zenodo_zip_url(a.zenodo_record)
    f = HttpRangeFile(url, block=a.block_kb * 1024)
    print(f"archive size {f.size / 1e9:.2f} GB (not downloaded)")
    zf = zipfile.ZipFile(f)
    infos = zf.infolist()
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    listing = out / "_archive_listing.json"
    dirs: dict[str, int] = {}
    for zi in infos:
        d = str(PurePosixPath(zi.filename).parent)
        dirs[d] = dirs.get(d, 0) + 1
    listing.write_text(json.dumps({"url": url, "n_members": len(infos), "dirs": dirs,
                                   "examples": [zi.filename for zi in infos[:50]]}, indent=2))
    print(f"{len(infos)} members in {len(dirs)} directories; listing -> {listing.relative_to(ROOT) if listing.is_relative_to(ROOT) else listing}"
          f" ({f.bytes_downloaded / 1e6:.1f} MB fetched so far)")
    if a.list_only:
        for d, n in sorted(dirs.items())[:60]:
            print(f"  {n:7d}  {d}")
        return
    tokens = {"label": "labels", "test": "test", "rover": "mxy", "range": "rng"}
    chosen = select([zi.filename for zi in infos], a.n_train, a.n_test, a.seed, tokens, a.mission or None)
    by_name = {zi.filename: zi for zi in infos}
    total = sum(by_name[n].compress_size for n in chosen)
    print(f"selected {len(chosen)} members, {total / 1e6:.1f} MB compressed")
    if total > a.max_gb * 1e9:
        raise SystemExit(f"selection {total / 1e9:.2f} GB exceeds --max-gb {a.max_gb}")
    for i, n in enumerate(chosen):
        if not safe_member(n):
            print(f"  skipping unsafe member name {n!r}")
            continue
        dst = out / n
        if dst.exists() and dst.stat().st_size == by_name[n].file_size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(by_name[n]) as src_f, open(dst, "wb") as dst_f:
            shutil.copyfileobj(src_f, dst_f)
        if i % 200 == 0:
            print(f"  {i}/{len(chosen)} ({f.bytes_downloaded / 1e6:.0f} MB downloaded)", flush=True)
    print(f"done: {len(chosen)} files under {out}; {f.bytes_downloaded / 1e6:.1f} MB downloaded in total")
    return {"archive_bytes": f.size, "downloaded_bytes": f.bytes_downloaded, "files": len(chosen)}


if __name__ == "__main__":
    main()
