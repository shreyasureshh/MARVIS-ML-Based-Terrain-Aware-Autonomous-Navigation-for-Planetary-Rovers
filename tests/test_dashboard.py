"""Dashboard renders every page without exceptions and shows 'Not run yet' when artifacts are missing."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(env_extra):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **env_extra}
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "dashboard_driver.py")], env=env, capture_output=True,
                       text=True, timeout=600, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_dashboard_with_no_artifacts(tmp_path):
    rep = render({"MARVIS_OUTPUTS": str(tmp_path / "out"), "MARVIS_DATA": str(tmp_path / "data")})
    assert len(rep) == 8
    for page, r in rep.items():
        assert not r["exceptions"], (page, r["exceptions"])
    for page in ("2 · Dataset", "3 · Segmentation vs GT", "4 · Model comparison", "5 · Uncertainty & calibration",
                 "6 · Traversability & planning"):
        assert any("Not run yet" in w for w in rep[page]["warnings"]), (page, rep[page])
