"""Partial-download tool against a LOCAL range-capable HTTP server serving a zipped SOFTWARE FIXTURE."""
import http.server
import os
import threading
import zipfile

import numpy as np
import pytest

from fixture_data import make_tree
from marvis import fetch_subset as fs


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    honour_range = True

    def log_message(self, *a):
        pass

    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404); return None
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        f = open(path, "rb")
        if rng and self.honour_range:
            a, b = rng.split("=")[1].split("-")
            a, b = int(a), min(int(b), size - 1)
            f.seek(a)
            self._len = b - a + 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        else:
            self._len = size
            self.send_response(200)
        self.send_header("Content-Length", str(self._len))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return f

    def copyfile(self, src, dst):
        dst.write(src.read(self._len))


@pytest.fixture
def server(tmp_path):
    raw = make_tree(tmp_path / "tree" / "msl", n_train=30, n_test=6, size=64)
    rng = np.random.default_rng(0)
    zpath = tmp_path / "srv" / "ai4mars.zip"
    zpath.parent.mkdir()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(raw.rglob("*")):
            if p.is_file():
                z.write(p, "ai4mars-dataset/msl/" + str(p.relative_to(raw)))
        z.writestr("ai4mars-dataset/msl/padding.bin", rng.bytes(3_000_000))  # make the archive "large"
        z.writestr("../evil.png", b"x")
    handler = lambda *a, **k: RangeHandler(*a, directory=str(zpath.parent), **k)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_port}/ai4mars.zip", tmp_path, zpath
    httpd.shutdown()


def test_fetch_subset_downloads_only_selection(server, monkeypatch):
    url, tmp, zpath = server
    monkeypatch.delenv("HTTPS_PROXY", raising=False); monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1"); monkeypatch.setenv("no_proxy", "127.0.0.1")
    out = tmp / "out"
    st = fs.main(["--url", url, "--n-train", "5", "--n-test", "2", "--out", str(out), "--max-gb", "1", "--block-kb", "16"])
    assert st["downloaded_bytes"] < 0.5 * st["archive_bytes"]
    labels = list(out.rglob("labels/train/*.png"))
    tests = list(out.rglob("labels/test/*/*.png"))
    images = list(out.rglob("images/edr/*.JPG"))
    assert len(labels) == 5 and len(tests) == 2 and len(images) == 7
    assert len(list(out.rglob("images/mxy/*.png"))) == 7
    assert not (tmp / "evil.png").exists()
    assert (out / "_archive_listing.json").exists()


def test_refuses_server_without_range(server, monkeypatch):
    url, tmp, _ = server
    monkeypatch.setenv("NO_PROXY", "127.0.0.1"); monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setattr(RangeHandler, "honour_range", False)
    with pytest.raises(IOError):
        fs.HttpRangeFile(url).read(10)


def test_safe_member():
    assert fs.safe_member("a/b.png") and not fs.safe_member("../x.png") and not fs.safe_member("/etc/x")
