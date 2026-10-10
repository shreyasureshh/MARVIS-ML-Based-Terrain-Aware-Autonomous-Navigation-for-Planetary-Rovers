"""End-to-end SOFTWARE SMOKE TEST on a generated fixture (not AI4MARS data).

Runs the real CLIs (prepare -> train -> evaluate -> calibrate -> plan -> compare) in an
isolated MARVIS_DATA / MARVIS_OUTPUTS so nothing touches the project's outputs. It checks
integration (shapes, files, keys), never metric values.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fixture_data import make_tree

ROOT = Path(__file__).resolve().parents[1]


def run(env, *args):
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"{args}\nSTDOUT:\n{r.stdout[-3000:]}\nSTDERR:\n{r.stderr[-3000:]}"
    return r.stdout


@pytest.mark.slow
def test_full_pipeline_on_fixture(tmp_path):
    raw = make_tree(tmp_path / "raw" / "msl")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "MARVIS_DATA": str(tmp_path / "data"),
           "MARVIS_OUTPUTS": str(tmp_path / "out"), "MARVIS_RAW_ROOT": str(raw)}
    run(env, "marvis.prepare", "inspect")
    run(env, "marvis.prepare", "split")
    out = run(env, "marvis.train", "configs/experiments/E1_unet_r34_wce_light.yaml", "--epochs", "2",
              "--set", "image_size=64", "batch_size=4", "num_workers=0", "pretrained=false",
              "early_stopping_patience=5")
    exp = tmp_path / "out" / "experiments" / "E1_unet_r34_wce_light_s0"
    assert (exp / "checkpoints" / "best.pt").exists() and (exp / "checkpoints" / "last.pt").exists()
    hist = json.loads((exp / "history.json").read_text())
    assert len(hist) == 2
    # resume: already finished -> no extra epochs, summary rewritten
    run(env, "marvis.train", "configs/experiments/E1_unet_r34_wce_light.yaml", "--epochs", "2", "--resume",
        "--set", "image_size=64", "batch_size=4", "num_workers=0", "pretrained=false")
    assert len(json.loads((exp / "history.json").read_text())) == 2
    cfg = json.loads((exp / "config.json").read_text())
    assert cfg["meta"]["torch"] and "manifest_md5" in cfg["meta"]
    run(env, "marvis.evaluate", exp.name, "--split", "val", "--mc", "3", "--save-preds")
    run(env, "marvis.evaluate", exp.name, "--split", "test", "--mc", "3", "--save-preds")
    m = json.loads((exp / "metrics_test.json").read_text())
    assert set(m["segmentation"]) == {"deterministic", "mc"}
    assert m["mc_dropout_variation_check"]["stochastic"] is True
    assert len(m["segmentation"]["mc"]["confusion_matrix"]) == 4
    run(env, "marvis.evaluate_planning", exp.name, "--split", "val", "--calibrate")
    thr = json.loads((exp / "behaviour_thresholds.json").read_text())
    assert thr["proceed_below"] <= thr["stop_above"]
    run(env, "marvis.evaluate_planning", exp.name, "--split", "test", "--n", "12")
    ps = json.loads((exp / "planning_summary_test.json").read_text())
    assert {r["planner"] for r in ps["table"]} == {"baseline", "terrain", "uncertainty"}
    assert ps["thresholds"]["source"].startswith("validation")
    run(env, "marvis.compare")
    sel = json.loads((tmp_path / "out" / "selected_model.json").read_text())
    assert sel["experiment"] == exp.name
    with pytest.raises(AssertionError):
        run(env, "marvis.evaluate_planning", exp.name, "--split", "test", "--calibrate")
    # dashboard renders every page against these artifacts without exceptions
    from test_dashboard import render
    rep = render({k: env[k] for k in ("MARVIS_DATA", "MARVIS_OUTPUTS", "MARVIS_RAW_ROOT")})
    for page, r in rep.items():
        assert not r["exceptions"], (page, r["exceptions"])
    assert rep["4 · Model comparison"]["n_dataframes"] >= 2
    assert rep["6 · Traversability & planning"]["n_dataframes"] >= 1
