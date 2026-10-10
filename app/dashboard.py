"""MARVIS research dashboard.   streamlit run app/dashboard.py

Every number, table and chart is read from files written by the pipeline (outputs/...)
or computed live by the trained model. Panels whose artifacts do not exist show a
"Not run yet" notice with the command that produces them; nothing is invented.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marvis.config import CONFIGS, MANIFESTS, OUTPUTS, colorize, load_classes, load_yaml  # noqa: E402
from marvis.planner import PLANNERS, behaviour_from_uncertainty, evaluate_path, path_length, path_mean, plan  # noqa: E402
from marvis.traversability import TraversabilityConfig, build_cost_map, gt_grid  # noqa: E402

st.set_page_config(page_title="MARVIS", page_icon="🛰️", layout="wide")
INFO = load_classes()
EXP_DIR = OUTPUTS / "experiments"
DATASET_DIR = OUTPUTS / "dataset"
PLANNER_COLORS = {"baseline": (170, 170, 170), "terrain": (30, 144, 255), "uncertainty": (0, 230, 118)}
PLANNER_LABELS = {"baseline": "Baseline (uniform cost)", "terrain": "Terrain-aware", "uncertainty": "Uncertainty-aware"}


# ---------------------------------------------------------------- helpers
def read_json(p: Path):
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except json.JSONDecodeError:
        st.error(f"Corrupt JSON: `{p}`")
        return None


def runs() -> list[str]:
    if not EXP_DIR.exists():
        return []
    return sorted(p.name for p in EXP_DIR.iterdir() if (p / "checkpoints" / "best.pt").exists())


def selected_run() -> str | None:
    rs = runs()
    if not rs:
        return None
    sel = read_json(OUTPUTS / "selected_model.json")
    default = sel["experiment"] if sel and sel["experiment"] in rs else rs[0]
    return st.sidebar.selectbox("Model run", rs, index=rs.index(default),
                                help="Default = highest validation mIoU (python -m marvis.compare)")


def not_run(what: str, cmd: str | None = None):
    st.warning(f"**Not run yet:** {what}" + (f"\n\n`{cmd}`" if cmd else ""), icon="⏳")


def show_img(p: Path, caption: str | None = None, cmd: str | None = None):
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        not_run(f"`{p.relative_to(OUTPUTS.parent) if p.is_relative_to(OUTPUTS.parent) else p}` missing", cmd)


def cmap(x: np.ndarray, name: str) -> np.ndarray:
    import matplotlib
    return (matplotlib.colormaps[name](np.clip(x, 0, 1))[..., :3] * 255).astype(np.uint8)


def overlay(img: np.ndarray, mask: np.ndarray, alpha=0.5) -> np.ndarray:
    out = (img * (1 - alpha) + colorize(mask, INFO).astype(np.float32) * alpha).astype(np.uint8)
    ign = mask == INFO.ignore_index
    out[ign] = (img[ign] * 0.35).astype(np.uint8)
    return out


def legend_html() -> str:
    chip = "<span style='display:inline-block;margin:2px 12px 2px 0'><span style='display:inline-block;width:14px;" \
           "height:14px;background:{c};border:1px solid #444;vertical-align:middle'></span> {n}</span>"
    return "".join(chip.format(c=f"rgb{tuple(int(v) for v in INFO.colors[i])}", n=n) for i, n in enumerate(INFO.names)) \
        + chip.format(c="#111", n="ignored (null / rover / beyond range)")


@st.cache_resource(show_spinner="Loading model…")
def get_model(run: str):
    import torch

    from marvis.model import load_checkpoint
    torch.set_num_threads(4)
    model, ck = load_checkpoint(EXP_DIR / run / "checkpoints" / "best.pt")
    return model, int(ck["config"]["image_size"])


@st.cache_data(show_spinner="Running inference…", max_entries=32)
def infer(run: str, img_bytes: bytes, mc: int):
    import torch

    from marvis.uncertainty import predict
    model, size = get_model(run)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    t0 = time.perf_counter()
    out = predict(model, x, mc)
    return np.asarray(img), out["probs"][0], out["entropy"][0], out.get("mutual_information", [None])[0], \
        time.perf_counter() - t0


@st.cache_resource
def test_dataset():
    try:
        from marvis.dataset import SegDataset
        return SegDataset("test", None, "none")
    except Exception as e:  # noqa: BLE001 - shown to the user
        return e


def pick_image(run: str, mc: int):
    """Shared image picker. Returns dict or None."""
    ds = test_dataset()
    srcs = ["Test image", "Upload"] if not isinstance(ds, Exception) else ["Upload"]
    src = st.radio("Image source", srcs, horizontal=True, key="src")
    if src == "Test image":
        ids = [ds.item_id(i) for i in range(len(ds))]
        idx = st.selectbox(f"Test image ({len(ids)} in held-out split)", range(len(ids)), format_func=lambda i: ids[i],
                           key="img_idx")
        img_bytes = ds.image_bytes(idx)
    else:
        if isinstance(ds, Exception):
            st.caption(f"Test split unavailable: {ds}")
        up = st.file_uploader("NavCam-like image", type=["png", "jpg", "jpeg"])
        if not up:
            st.info("Upload an image to run the model.")
            return None
        img_bytes, idx = up.getvalue(), None
    img, probs, ent, mi, dt = infer(run, img_bytes, mc)
    size = img.shape[0]
    gt = valid = None
    if idx is not None:
        gt, valid = ds.load_mask(idx, size), ds.load_valid(idx, size)
    return {"img": img, "probs": probs, "ent": ent, "mi": mi, "dt": dt, "gt": gt, "valid": valid, "mc": mc}


# ---------------------------------------------------------------- pages
def page_overview():
    st.title("🛰️ MARVIS")
    st.subheader("ML-based terrain perception, uncertainty estimation and traversability prediction for planetary rovers")
    st.markdown("""
**Research questions**
1. **Perception**: how accurately can a deep model segment Mars terrain (AI4MARS, MSL NavCam)?
2. **Generalisation**: how does performance hold up on held-out sites, expert labels and image conditions?
3. **Uncertainty**: does MC Dropout uncertainty identify the pixels where the model is wrong?
4. **Downstream impact**: does uncertainty-aware traversability change route decisions, scored against ground truth?
""")
    st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica", fontsize=11];
  img [label="NavCam image"]; seg [label="U-Net / DeepLabV3+\\nResNet34 + MC Dropout (T passes)"];
  prob [label="mean class\\nprobabilities"]; unc [label="predictive entropy /\\nmutual information"];
  cost [label="expected terrain cost\\n(assumed YAML table)"]; grid [label="64×64 image-space grid\\n+ footprint inflation"];
  plan [label="A*: baseline / terrain /\\nuncertainty-aware"]; beh [label="proceed / slow /\\nstop + replan"];
  ev [label="evaluation vs\\nground-truth labels", fillcolor="#fff4e0"];
  img -> seg -> prob -> cost -> grid -> plan -> beh; seg -> unc -> grid; unc -> beh; plan -> ev; }
""")
    st.markdown("**Terrain classes**<br>" + legend_html(), unsafe_allow_html=True)
    st.markdown("#### Experiment status")
    stage = [("Dataset inspected", DATASET_DIR / "inspection_report.json", "python -m marvis.prepare inspect"),
             ("Splits built", MANIFESTS / "split_report.json", "python -m marvis.prepare split")]
    rows = [{"stage": n, "status": "✅ done" if p.exists() else "⏳ not run", "command": c} for n, p, c in stage]
    for r in runs():
        d = EXP_DIR / r
        rows.append({"stage": f"run {r}", "status": " · ".join(
            f"{'✅' if (d / f).exists() else '⏳'} {lbl}" for f, lbl in
            (("train_summary.json", "trained"), ("metrics_test.json", "test-evaluated"),
             ("behaviour_thresholds.json", "thresholds (val)"), ("planning_summary_test.json", "planning"))),
            "command": ""})
    if not runs():
        rows.append({"stage": "Model training", "status": "⏳ not run",
                     "command": "python -m marvis.train configs/experiments/E1_unet_r34_wce_light.yaml"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.info("Navigation is used to *evaluate* the ML model's downstream value. Terrain costs, grid geometry, rover "
            "footprint and thresholds are assumptions; terrain class is a proxy, not proof of traversability.")


def page_dataset():
    st.title("Dataset & class distribution")
    rep = read_json(DATASET_DIR / "inspection_report.json")
    stats = read_json(DATASET_DIR / "stats.json")
    if not rep:
        not_run("dataset inspection", "python -m marvis.prepare inspect --raw-root <path-to-ai4mars>")
        return
    c = st.columns(4)
    c[0].metric("Label files inspected", rep["inspected_label_files"])
    c[1].metric("Usable image–label pairs", rep["usable_pairs"])
    c[2].metric("Unpaired images", rep["unpaired_images"])
    c[3].metric("Issue types", len(rep["issue_counts"]))
    with st.expander("Inspection details (raw label values, masks, issues)"):
        st.json({k: rep[k] for k in ("raw_root", "discovery", "issue_counts", "raw_label_values",
                                     "raw_values_not_in_config", "mask_unique_values", "sizes", "modes",
                                     "config_label_map", "config_null_values")})
    if stats:
        l, r = st.columns(2)
        with l:
            st.markdown("**Split sizes (images / leakage groups)**")
            st.dataframe(pd.DataFrame({"images": stats["split_sizes"], "groups": stats["split_groups"]}),
                         use_container_width=True)
            st.markdown("**Labelled-pixel class fraction**")
            st.dataframe(pd.DataFrame(stats["class_fraction"]).T.style.format("{:.3f}"), use_container_width=True)
            st.markdown("**Leakage handling**")
            st.json(stats["leakage"])
            for n in stats["notes"]:
                st.caption("• " + n)
        with r:
            show_img(DATASET_DIR / "class_distribution.png")
    else:
        not_run("split building", "python -m marvis.prepare split")
    st.markdown("**Sample image / label overlays**")
    show_img(DATASET_DIR / "sample_overlays.png")


def page_segmentation():
    st.title("Segmentation: prediction vs ground truth")
    run = selected_run()
    if not run:
        not_run("no trained model", "python -m marvis.train configs/experiments/E1_unet_r34_wce_light.yaml")
        return
    mc = st.sidebar.slider("MC Dropout samples (0 = deterministic)", 0, 30, 20)
    d = pick_image(run, mc)
    if d is None:
        return
    img, probs, ent, gt = d["img"], d["probs"], d["ent"], d["gt"]
    pred = probs.argmax(0)
    st.markdown(legend_html(), unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].image(img, caption="Input", use_container_width=True)
    cols[1].image(overlay(img, pred), caption="Prediction" + (" (MC mean)" if mc else ""), use_container_width=True)
    if gt is not None:
        cols[2].image(overlay(img, gt), caption="Ground truth (dark = ignored)", use_container_width=True)
        err = (pred != gt) & (gt != INFO.ignore_index)
        cols[3].image(cmap(err.astype(float), "gray"), caption="Errors on labelled pixels (white)",
                      use_container_width=True)
    else:
        cols[2].info("No ground truth for uploaded images.")
    c2 = st.columns(4)
    c2[0].image(cmap(ent, "magma"), caption="Predictive entropy (0–1)", use_container_width=True)
    if d["mi"] is not None:
        c2[1].image(cmap(d["mi"] / max(float(d["mi"].max()), 1e-6), "magma"),
                    caption="Mutual information (rescaled to max)", use_container_width=True)
    c2[2].image(cmap(probs.max(0), "viridis"), caption="Confidence max p", use_container_width=True)
    st.caption(f"CPU inference {d['dt']:.2f}s with {mc} MC samples · run `{run}`")
    if gt is not None:
        lab = gt != INFO.ignore_index
        if lab.sum():
            from marvis.metrics import confusion_matrix, segmentation_scores
            s = segmentation_scores(confusion_matrix(gt, pred, INFO.num_classes, INFO.ignore_index), INFO.names)
            e = (pred != gt)[lab]
            st.markdown(f"**This image:** pixel accuracy {s['pixel_accuracy']:.3f} · mIoU {s['mIoU']:.3f} · "
                        f"mean entropy on correct pixels {ent[lab][~e].mean() if (~e).any() else float('nan'):.3f} "
                        f"vs wrong {ent[lab][e].mean() if e.any() else float('nan'):.3f}")


def page_models():
    st.title("Model comparison & training curves")
    rs = runs()
    if not rs:
        not_run("no experiments trained", "python -m marvis.train configs/experiments/E1_unet_r34_wce_light.yaml")
        return
    summ = read_json(OUTPUTS / "experiments_summary.json")
    if summ:
        st.markdown("**Experiments (mean over seeds; single seed = preliminary)**")
        rows = []
        for name, s in summ.items():
            f = lambda k: None if not s.get(k) else (f"{s[k]['mean']:.4f}" + (f" ± {s[k]['std']:.4f}" if s[k]["std"] else ""))
            rows.append({"experiment": name, "status": s["status"], "val mIoU": f("best_val_mIoU"),
                         "test mIoU": f("test_mIoU"), "test mIoU (MC)": f("test_mIoU_mc"),
                         "mean Dice": f("test_mean_dice"), "error AUROC": f("error_AUROC_mc_entropy"),
                         "ECE": f("ECE_mc")})
        st.dataframe(pd.DataFrame(rows).set_index("experiment"), use_container_width=True)
        st.caption("Differences smaller than the seed-to-seed spread are not evidence that one method is better.")
    else:
        not_run("experiment aggregation", "python -m marvis.compare")
    rows = []
    for r in rs:
        m = read_json(EXP_DIR / r / "metrics_test.json")
        ts = read_json(EXP_DIR / r / "train_summary.json") or {}
        cfg = read_json(EXP_DIR / r / "config.json") or {}
        row = {"run": r, "arch": cfg.get("arch"), "loss": cfg.get("loss"), "aug": cfg.get("aug"),
               "epochs": ts.get("epochs_completed"), "train min": round((ts.get("train_seconds") or 0) / 60, 1),
               "best val mIoU": ts.get("best_val_mIoU")}
        if m:
            s = m["segmentation"]["deterministic"]
            row.update({"test mIoU": s["mIoU"], "pixel acc": s["pixel_accuracy"],
                        **{f"IoU {n}": s["per_class"][n]["IoU"] for n in INFO.names},
                        "latency det (s)": m["efficiency"]["latency_s_deterministic_median"],
                        "latency MC (s)": m["efficiency"]["latency_s_mc_median"],
                        "params (M)": m["efficiency"]["n_params"] / 1e6})
        rows.append(row)
    st.markdown("**Individual runs**")
    st.dataframe(pd.DataFrame(rows).set_index("run"), use_container_width=True)
    r = st.selectbox("Inspect run", rs)
    m = read_json(EXP_DIR / r / "metrics_test.json")
    c1, c2 = st.columns(2)
    with c1:
        show_img(EXP_DIR / r / "training_curves.png", "Training / validation curves", f"python -m marvis.evaluate {r}")
    with c2:
        show_img(EXP_DIR / r / "confusion_matrix.png", "Test confusion matrix (MC mean)", f"python -m marvis.evaluate {r}")
    if m:
        st.markdown("**Per-class test metrics (deterministic)**")
        st.dataframe(pd.DataFrame(m["segmentation"]["deterministic"]["per_class"]).T, use_container_width=True)
        if m.get("by_brightness_tertile"):
            st.markdown("**Generalisation: test mIoU by image brightness tertile**")
            st.dataframe(pd.DataFrame(m["by_brightness_tertile"]).T.drop(columns=["per_class_IoU"]),
                         use_container_width=True)
        others = [p for p in (EXP_DIR / r).glob("metrics_test_*.json")]
        if others:
            st.markdown("**Generalisation: other expert-label variants**")
            st.dataframe(pd.DataFrame({p.stem.replace("metrics_", ""): {
                "mIoU": read_json(p)["segmentation"]["deterministic"]["mIoU"],
                "pixel acc": read_json(p)["segmentation"]["deterministic"]["pixel_accuracy"],
                "images": read_json(p)["n_images"]} for p in others}).T, use_container_width=True)
    show_img(EXP_DIR / r / "qualitative.png", "Qualitative test predictions: image / GT / prediction / entropy")


def page_uncertainty():
    st.title("Uncertainty, calibration & error detection")
    run = selected_run()
    m = read_json(EXP_DIR / run / "metrics_test.json") if run else None
    if not m:
        not_run("test evaluation", f"python -m marvis.evaluate {run or '<run>'} --split test --save-preds")
        return
    st.markdown(f"Run `{run}` · {m['n_images']} test images · {m['uncertainty_pixels_used']:,} labelled pixels "
                f"sampled · T = {m['mc_samples']} MC samples · MC samples vary: "
                f"**{m['mc_dropout_variation_check']['stochastic']}**")
    for k, (ok, why) in m["uncertainty_evaluable"].items():
        if not ok:
            st.error(f"{k}: uncertainty metrics not reported — {why}")
    if m["uncertainty"]:
        st.dataframe(pd.DataFrame(m["uncertainty"]).T, use_container_width=True)
    with st.expander("Metric conventions"):
        st.json(m["conventions"])
    c1, c2 = st.columns(2)
    with c1:
        show_img(EXP_DIR / run / "reliability.png", "Reliability diagram")
    with c2:
        show_img(EXP_DIR / run / "risk_coverage.png", "Risk vs coverage")


def page_planning():
    st.title("Traversability maps & planner comparison")
    run = selected_run()
    tab_live, tab_agg = st.tabs(["Interactive (one image)", "Held-out evaluation"])
    with tab_agg:
        s = read_json(EXP_DIR / run / "planning_summary_test.json") if run else None
        if not s:
            not_run("planning evaluation", f"python -m marvis.evaluate_planning {run or '<run>'} --split test")
        else:
            if not s["sufficient_images"]:
                st.warning(f"Only {s['n_images']} evaluable images (< 30): preliminary.")
            st.markdown(f"{s['n_images']} test images × {s['pairs_per_image']} fixed start/goal pairs. "
                        f"Thresholds: *{s['thresholds']['source']}* "
                        f"(A = {s['thresholds']['proceed_below']:.3f}, B = {s['thresholds']['stop_above']:.3f}).")
            st.dataframe(pd.DataFrame(s["table"]).set_index("planner"), use_container_width=True)
            st.markdown(f"**Uncertainty-aware behaviour counts:** {s['uncertainty_behaviour_counts']}")
            st.json(s["definitions"])
            show_img(EXP_DIR / run / "planner_comparison_test.png")
            csv = EXP_DIR / run / "planning_results_test.csv"
            if csv.exists():
                with st.expander("Per-trial results"):
                    st.dataframe(pd.read_csv(csv), use_container_width=True)
    with tab_live:
        if not run:
            not_run("no trained model")
            return
        raw = load_yaml(CONFIGS / "traversability.yaml")
        thr = read_json(EXP_DIR / run / "behaviour_thresholds.json") or {
            "proceed_below": raw["uncertainty"]["proceed_below"], "stop_above": raw["uncertainty"]["stop_above"],
            "source": "YAML fallback (not calibrated)"}
        with st.sidebar:
            st.markdown("### Assumed terrain costs")
            costs = {n: st.slider(n, 0.0, 1.0, float(raw["class_costs"][n]), 0.05, key=f"c_{n}") for n in INFO.names}
            uw = st.slider("Uncertainty weight", 0.0, 5.0, float(raw["uncertainty"]["weight"]), 0.1)
            radius = st.slider("Footprint radius (cells)", 0, 5, int(raw["footprint_radius_cells"]))
            a = st.slider("Proceed below (A)", 0.0, 1.0, float(thr["proceed_below"]), 0.01)
            b = st.slider("Stop above (B)", 0.0, 1.0, max(float(thr["stop_above"]), a), 0.01)
            st.caption(f"Default A/B source: {thr['source']}")
        d = pick_image(run, int(raw["uncertainty"]["mc_samples"]))
        if d is None:
            return
        tcfg = TraversabilityConfig.from_yaml(INFO, class_costs=costs, footprint_radius_cells=radius)
        cm = build_cost_map(d["probs"], tcfg, uncertainty=d["ent"], valid=d["valid"])
        g, size = tcfg.grid_size, d["img"].shape[0]
        c = st.columns(3)
        c[0].image(Image.fromarray(cmap(cm.terrain_cost, "RdYlGn_r")).resize((size, size), Image.NEAREST),
                   caption="Expected terrain cost (assumed costs)", use_container_width=True)
        c[1].image(Image.fromarray(cmap(cm.uncertainty, "magma")).resize((size, size), Image.NEAREST),
                   caption="Grid uncertainty (entropy)", use_container_width=True)
        c[2].image(Image.fromarray((cm.blocked * 255).astype(np.uint8)).resize((size, size), Image.NEAREST),
                   caption="Blocked (obstacle + footprint + invalid)", use_container_width=True)
        if "sg" not in st.session_state:
            st.session_state.sg = [(g - 4, g // 2), (int(g * 0.4), g // 2)]
        (sr, sc), (gr, gc) = st.session_state.sg
        cc = st.columns(4)
        s_ = (cc[0].number_input("start row", 0, g - 1, sr), cc[1].number_input("start col", 0, g - 1, sc))
        g_ = (cc[2].number_input("goal row", 0, g - 1, gr), cc[3].number_input("goal col", 0, g - 1, gc))
        res = {n: plan(cm, n, s_, g_, epsilon=raw["planner"]["step_epsilon"], uncertainty_weight=uw,
                       proceed_below=a, stop_above=b, replan_multiplier=raw["uncertainty"]["replan_weight_multiplier"])
               for n in PLANNERS}
        base = overlay(d["img"], d["probs"].argmax(0), 0.35)
        bm = np.kron(cm.blocked, np.ones((size // g, size // g), bool))
        base[bm] = (base[bm] * 0.4).astype(np.uint8)
        canvas = draw_paths(base, g, {n: r.path for n, r in res.items()}, s_, g_)
        left, right = st.columns([1.1, 1])
        with left:
            try:
                from streamlit_image_coordinates import streamlit_image_coordinates
                st.caption("Click to set start, click again to set goal.")
                click = streamlit_image_coordinates(canvas, key="map")
                if click and click != st.session_state.get("last_click"):
                    st.session_state.last_click = click
                    k = st.session_state.get("next_pt", 0)
                    st.session_state.sg[k] = (int(click["y"] / 512 * g), int(click["x"] / 512 * g))
                    st.session_state.next_pt = 1 - k
                    st.rerun()
            except ImportError:
                st.image(canvas)
            st.markdown(" ".join(f"<span style='color:rgb{PLANNER_COLORS[n]};font-weight:700'>━ {PLANNER_LABELS[n]}</span>"
                                 for n in PLANNERS), unsafe_allow_html=True)
        with right:
            frac = labelled = None
            if d["gt"] is not None:
                frac, labelled = gt_grid(d["gt"], INFO.num_classes, INFO.ignore_index, g)
            rows = []
            for n, r in res.items():
                row = {"planner": n, "found": r.path is not None, "expanded": r.expanded, "behaviour": r.behaviour or "—"}
                if r.path:
                    row.update({"length (cells)": round(path_length(r.path), 1),
                                "mean uncertainty": round(path_mean(r.path, cm.uncertainty), 3)})
                    if frac is not None:
                        ev = evaluate_path(r.path, frac, labelled, tcfg.class_costs, tcfg.high_risk, tcfg.impassable,
                                           cm.terrain_cost)
                        row.update({"GT cost": round(ev["gt_cost"], 2), "GT high-risk frac": ev["gt_high_risk_frac"],
                                    "GT rock cells": ev["gt_impassable_cells"]})
                rows.append(row)
            st.dataframe(pd.DataFrame(rows).set_index("planner"), use_container_width=True)
            for line in res["uncertainty"].log:
                st.code(line, language=None)
            if d["gt"] is None:
                st.caption("Uploaded image: no GT metrics and no rover-body mask.")
        st.session_state.live = {"img": base, "probs": d["probs"], "cm": cm, "res": res, "grid": g, "thr": (a, b)}


def draw_paths(base: np.ndarray, grid: int, paths: dict, start=None, goal=None, rover=None) -> Image.Image:
    im = Image.fromarray(base).convert("RGB").resize((512, 512), Image.NEAREST)
    d = ImageDraw.Draw(im)
    s = 512 / grid
    c = lambda p: (p[1] * s + s / 2, p[0] * s + s / 2)
    for name, path in paths.items():
        if path and len(path) > 1:
            d.line([c(p) for p in path], fill=PLANNER_COLORS[name], width=4)
    for p, col in ((start, (255, 255, 255)), (goal, (255, 215, 0))):
        if p is not None:
            x, y = c(p); d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col, outline=(0, 0, 0), width=2)
    if rover is not None:
        x, y = c(rover); d.rectangle([x - 8, y - 8, x + 8, y + 8], fill=(255, 0, 255), outline=(0, 0, 0), width=2)
    return im


def page_rover():
    st.title("Rover run (illustrative)")
    live = st.session_state.get("live")
    if not live:
        st.info("Open **Traversability & planning → Interactive** first to choose an image and plan.")
        return
    import plotly.graph_objects as go
    choice = st.radio("Path", list(PLANNERS), horizontal=True, index=2, format_func=PLANNER_LABELS.get)
    path = live["res"][choice].path
    if not path:
        st.error("This planner found no path.")
        return
    delay = st.slider("Animation delay (s/step)", 0.01, 0.3, 0.05)
    cm, g = live["cm"], live["grid"]
    size = live["img"].shape[0]
    pg = live["probs"].reshape(INFO.num_classes, g, size // g, g, size // g).mean((2, 4))
    a, b = live["thr"]
    if st.button("▶ Drive", type="primary"):
        c1, c2 = st.columns(2)
        ph_map, ph_stat, ph_plot = c1.empty(), c2.empty(), c2.empty()
        cum, xs = [0.0], [0.0]
        for k, p in enumerate(path):
            if k:
                dd = float(np.hypot(p[0] - path[k - 1][0], p[1] - path[k - 1][1]))
                cum.append(cum[-1] + dd * cm.terrain_cost[p]); xs.append(xs[-1] + dd)
            u = float(cm.uncertainty[p])
            state = behaviour_from_uncertainty(u, a, b)
            ph_map.image(draw_paths(live["img"], g, {choice: path}, path[0], path[-1], rover=p))
            ci = int(pg[:, p[0], p[1]].argmax())
            ph_stat.markdown(f"**Step {k + 1}/{len(path)}** · cell {p}  \nPredicted terrain: **{INFO.names[ci]}** "
                             f"(p = {pg[ci, p[0], p[1]]:.2f})  \nLocal uncertainty **{u:.3f}** → "
                             f"{ {'proceed': '🟢', 'slow': '🟡', 'stop': '🔴'}[state]} **{state.upper()}**  \n"
                             f"Accumulated predicted cost **{cum[-1]:.2f}**")
            fig = go.Figure(go.Scatter(x=xs, y=cum, mode="lines"))
            fig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10), title="running predicted cost",
                              xaxis_title="distance (grid cells)")
            ph_plot.plotly_chart(fig, use_container_width=True, key=f"rc{k}")
            time.sleep(delay * {"proceed": 1, "slow": 2, "stop": 4}[state])
        st.success(f"Path complete. Planner-level decision: {live['res'][choice].behaviour or 'n/a'}")
    st.caption("Speed states use per-cell uncertainty with the same A/B thresholds; no vehicle dynamics are modelled.")


def page_limitations():
    st.title("Limitations")
    p = ROOT / "LIMITATIONS.md"
    st.markdown(p.read_text() if p.exists() else "See README.")


PAGES = {"1 · Overview & status": page_overview, "2 · Dataset": page_dataset, "3 · Segmentation vs GT": page_segmentation,
         "4 · Model comparison": page_models, "5 · Uncertainty & calibration": page_uncertainty,
         "6 · Traversability & planning": page_planning, "Rover run": page_rover, "Limitations": page_limitations}
choice = st.sidebar.radio("MARVIS", list(PAGES), key="page")
PAGES[choice]()
