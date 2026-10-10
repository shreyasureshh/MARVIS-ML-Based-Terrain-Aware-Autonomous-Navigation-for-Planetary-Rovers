"""MARVIS research dashboard.   streamlit run app/dashboard.py

Every number shown here is read from files written by the pipeline scripts
(outputs/...). Panels whose artifacts do not exist say so instead of showing anything.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marvis.config import OUTPUTS, colorize, load_classes, load_yaml  # noqa: E402
from marvis.planner import PLANNERS, behaviour_from_uncertainty, evaluate_path, plan  # noqa: E402
from marvis.traversability import TraversabilityConfig, build_cost_map, gt_grid  # noqa: E402

st.set_page_config(page_title="MARVIS", page_icon="🛰️", layout="wide")
INFO = load_classes()
EXP_DIR = OUTPUTS / "experiments"
DATASET_DIR = OUTPUTS / "dataset"
PLANNER_COLORS = {"baseline": (160, 160, 160), "terrain": (30, 144, 255), "uncertainty": (0, 230, 118)}
PLANNER_LABELS = {"baseline": "Baseline (shortest, avoid obstacles)", "terrain": "Terrain-aware",
                  "uncertainty": "Uncertainty-aware"}


# ---------------------------------------------------------------- helpers
def read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def experiments() -> list[str]:
    return sorted(p.name for p in EXP_DIR.glob("*") if (p / "checkpoints" / "best.pt").exists()) if EXP_DIR.exists() else []


def selected_experiment() -> str | None:
    sel = read_json(OUTPUTS / "selected_model.json")
    exps = experiments()
    if sel and sel["experiment"] in exps:
        return sel["experiment"]
    return exps[0] if exps else None


def not_run(msg: str):
    st.warning(f"**Not run yet.** {msg}", icon="⏳")


def show_img(p: Path, caption: str | None = None):
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        not_run(f"`{p.relative_to(ROOT)}` does not exist.")


@st.cache_resource(show_spinner="Loading model…")
def get_model(exp: str):
    import torch

    from marvis.model import load_checkpoint
    torch.set_num_threads(4)
    model, ck = load_checkpoint(EXP_DIR / exp / "checkpoints" / "best.pt")
    return model, ck["config"]["image_size"]


@st.cache_data(show_spinner="Running MC Dropout inference…", max_entries=32)
def infer(exp: str, img_bytes: bytes, mc: int):
    import io

    import torch

    from marvis.uncertainty import predict
    model, size = get_model(exp)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    t0 = time.perf_counter()
    out = predict(model, x, mc)
    dt = time.perf_counter() - t0
    return np.asarray(img), out["probs"][0], out["entropy"][0], dt


@st.cache_resource
def test_dataset():
    try:
        from marvis.dataset import SegDataset
        return SegDataset("test", None, "none")
    except Exception as e:  # dataset not prepared
        return e


def overlay(img: np.ndarray, mask: np.ndarray, alpha=0.5) -> np.ndarray:
    col = colorize(mask, INFO).astype(np.float32)
    return (img * (1 - alpha) + col * alpha).astype(np.uint8)


def heat(x: np.ndarray, cmap: str) -> np.ndarray:
    import matplotlib.cm as cm
    return (cm.get_cmap(cmap)(np.clip(x, 0, 1))[..., :3] * 255).astype(np.uint8)


def draw_paths(base: np.ndarray, grid: int, paths: dict, start=None, goal=None, rover=None) -> Image.Image:
    im = Image.fromarray(base).convert("RGB").resize((512, 512), Image.NEAREST)
    d = ImageDraw.Draw(im)
    s = 512 / grid
    c = lambda p: (p[1] * s + s / 2, p[0] * s + s / 2)
    for name, path in paths.items():
        if path:
            d.line([c(p) for p in path], fill=PLANNER_COLORS[name], width=4)
    for p, col in ((start, (255, 255, 255)), (goal, (255, 215, 0))):
        if p is not None:
            x, y = c(p); d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col, outline=(0, 0, 0), width=2)
    if rover is not None:
        x, y = c(rover); d.rectangle([x - 8, y - 8, x + 8, y + 8], fill=(255, 0, 255), outline=(0, 0, 0), width=2)
    return im


def legend_html():
    chips = "".join(
        f"<span style='display:inline-block;margin:2px 10px 2px 0'><span style='display:inline-block;width:14px;height:14px;"
        f"background:rgb{tuple(int(v) for v in INFO.colors[i])};border:1px solid #444;vertical-align:middle'></span> {n}</span>"
        for i, n in enumerate(INFO.names))
    return chips + "<span style='display:inline-block;margin-left:4px'><span style='display:inline-block;width:14px;" \
                   "height:14px;background:#000;border:1px solid #444;vertical-align:middle'></span> unlabelled / ignored</span>"


# ---------------------------------------------------------------- pages
def page_overview():
    st.title("🛰️ MARVIS")
    st.subheader("ML-based terrain perception, uncertainty estimation and traversability prediction for planetary rovers")
    st.markdown("""
**Research questions**
1. **Perception**: how accurately can a deep model segment Mars terrain (AI4MARS, MSL NavCam)?
2. **Generalisation**: how does performance hold up on held-out images and harder subsets?
3. **Uncertainty**: does MC Dropout uncertainty flag the pixels where the model is wrong?
4. **Downstream impact**: does uncertainty-aware traversability change route decisions measurably (scored against ground-truth labels)?
""")
    st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];
  img [label="NavCam image"]; seg [label="U-Net / DeepLabV3+\\n(ResNet34, MC Dropout ×T)"];
  prob [label="class probabilities"]; unc [label="predictive entropy"];
  cost [label="expected terrain cost\\n(YAML cost table)"]; grid [label="64×64 grid +\\nfootprint inflation"];
  plan [label="A* planners\\nbaseline / terrain / uncertainty"]; beh [label="behaviour\\nproceed / slow / stop+replan"];
  img -> seg -> prob -> cost -> grid -> plan -> beh; seg -> unc -> grid; unc -> beh; }
""")
    st.markdown("**Terrain classes**<br>" + legend_html(), unsafe_allow_html=True)
    st.info("Navigation is an *evaluation* of the ML model's downstream value. Terrain costs are assumed, "
            "not measured; the planning grid is in image space (no ground-plane projection). See **Limitations**.")
    sel = selected_experiment()
    st.markdown(f"**Model used in the demo:** `{sel}`" if sel else "")
    if not sel:
        not_run("No trained model found in `outputs/experiments/*/checkpoints/best.pt`.")


def page_dataset():
    st.title("Dataset analysis")
    s = read_json(DATASET_DIR / "stats.json")
    if not s:
        not_run("Run `python scripts/inspect_dataset.py` and `python scripts/make_splits.py`.")
        return
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Split sizes**")
        st.table(pd.DataFrame([s["split_sizes"]]))
        st.markdown("**Class pixel distribution (labelled pixels, per split)**")
        df = pd.DataFrame(s["class_fraction"]).T
        st.dataframe(df.style.format("{:.3f}"), use_container_width=True)
        if "notes" in s:
            for n in s["notes"]:
                st.caption("• " + n)
    with c2:
        show_img(DATASET_DIR / "class_distribution.png")
    st.markdown("**Sample image / label overlays**")
    show_img(DATASET_DIR / "sample_overlays.png")


def page_live():
    st.title("Live demo")
    exp = selected_experiment()
    if not exp:
        not_run("Train a model first (see README, Stage 2).")
        return
    info_ds = test_dataset()
    tcfg_raw = load_yaml(ROOT / "configs" / "traversability.yaml")
    with st.sidebar:
        st.markdown("### Input")
        src = st.radio("Image source", ["Test image", "Upload"], horizontal=True)
        mc = st.slider("MC Dropout samples", 0, 30, int(tcfg_raw["uncertainty"]["mc_samples"]),
                       help="0 = deterministic softmax")
        st.markdown("### Assumed terrain costs")
        costs = {n: st.slider(n, 0.0, 1.0, float(tcfg_raw["class_costs"][n]), 0.05) for n in INFO.names}
        st.markdown("### Planner")
        uw = st.slider("Uncertainty weight", 0.0, 5.0, float(tcfg_raw["uncertainty"]["weight"]), 0.1)
        radius = st.slider("Footprint radius (cells)", 0, 5, int(tcfg_raw["footprint_radius_cells"]))
        thr_a = st.slider("Proceed below (A)", 0.0, 1.0, float(tcfg_raw["uncertainty"]["proceed_below"]), 0.01)
        thr_b = st.slider("Stop above (B)", 0.0, 1.0, float(tcfg_raw["uncertainty"]["stop_above"]), 0.01)
    gt = valid = None
    if src == "Test image":
        if isinstance(info_ds, Exception):
            not_run(f"Test split unavailable ({info_ds}). Upload an image instead.")
            return
        ids = [info_ds.item_id(i) for i in range(len(info_ds))]
        idx = st.selectbox("Test image", range(len(ids)), format_func=lambda i: ids[i])
        img_bytes = info_ds.image_bytes(idx)
    else:
        up = st.file_uploader("NavCam-like grayscale/RGB image", type=["png", "jpg", "jpeg"])
        if not up:
            st.info("Upload an image to run the pipeline.")
            return
        img_bytes = up.getvalue()
    img, probs, ent, dt = infer(exp, img_bytes, mc)
    size = img.shape[0]
    if src == "Test image":
        gt = info_ds.load_mask(idx, size)
        valid = info_ds.load_valid(idx, size)
    tcfg = TraversabilityConfig.from_yaml(INFO, class_costs=costs, footprint_radius_cells=radius)
    cm = build_cost_map(probs, tcfg, uncertainty=ent, valid=valid)
    g = tcfg.grid_size
    pred = probs.argmax(0)
    cols = st.columns(4)
    cols[0].image(img, caption="Input", use_container_width=True)
    cols[1].image(overlay(img, pred), caption="Segmentation (MC mean)", use_container_width=True)
    cols[2].image(Image.fromarray(heat(cm.total_cost(0), "RdYlGn_r")).resize((size, size), Image.NEAREST),
                  caption="Expected terrain cost (64×64)", use_container_width=True)
    cols[3].image(heat(ent, "magma"), caption="Predictive entropy (normalised)", use_container_width=True)
    st.caption(f"Inference: {dt:.2f}s on CPU with {mc} MC samples · model `{exp}`")
    if gt is not None:
        c = st.columns(4)
        c[1].image(overlay(img, gt), caption="Ground truth (black = unlabelled)", use_container_width=True)
        err = (pred != gt) & (gt != INFO.ignore_index)
        c[3].image(heat(err.astype(float), "gray"), caption="Errors vs GT (white)", use_container_width=True)

    st.markdown("#### Plan")
    st.caption("Click the map to set start then goal (or use the number boxes). Grid row 0 = top of image.")
    if "sg" not in st.session_state:
        st.session_state.sg = [(g - 4, g // 2), (int(g * 0.4), g // 2)]
    s_, g_ = st.session_state.sg
    cc = st.columns(4)
    s_ = (cc[0].number_input("start row", 0, g - 1, s_[0]), cc[1].number_input("start col", 0, g - 1, s_[1]))
    g_ = (cc[2].number_input("goal row", 0, g - 1, g_[0]), cc[3].number_input("goal col", 0, g - 1, g_[1]))
    results = {n: plan(cm, n, s_, g_, epsilon=tcfg_raw["planner"]["step_epsilon"], uncertainty_weight=uw,
                       proceed_below=thr_a, stop_above=thr_b,
                       replan_multiplier=tcfg_raw["uncertainty"]["replan_weight_multiplier"]) for n in PLANNERS}
    base = overlay(img, pred, 0.35)
    blocked_rgb = base.copy()
    bmask = np.kron(cm.blocked, np.ones((size // g, size // g), bool))
    blocked_rgb[bmask] = (blocked_rgb[bmask] * 0.4).astype(np.uint8)
    canvas = draw_paths(blocked_rgb, g, {n: r.path for n, r in results.items()}, s_, g_)
    left, right = st.columns([1.1, 1])
    with left:
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates
            click = streamlit_image_coordinates(canvas, key="map")
            if click and click != st.session_state.get("last_click"):
                st.session_state.last_click = click
                cell = (int(click["y"] / 512 * g), int(click["x"] / 512 * g))
                k = st.session_state.get("next_pt", 0)
                st.session_state.sg[k] = cell
                st.session_state.next_pt = 1 - k
                st.rerun()
        except ImportError:
            st.image(canvas)
        st.markdown(" ".join(f"<span style='color:rgb{PLANNER_COLORS[n]};font-weight:700'>━ {PLANNER_LABELS[n]}</span>"
                             for n in PLANNERS) + " &nbsp; dark = blocked (obstacle + footprint)", unsafe_allow_html=True)
    with right:
        rows = []
        frac = labelled = None
        if gt is not None:
            frac, labelled = gt_grid(gt, INFO.num_classes, INFO.ignore_index, g)
        for n, r in results.items():
            row = {"planner": n, "found": r.path is not None}
            if r.path:
                row["length"] = round(sum(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(r.path, r.path[1:])), 1)
                row["pred. terrain cost"] = round(sum(np.hypot(a[0] - b[0], a[1] - b[1]) * cm.terrain_cost[b]
                                                      for a, b in zip(r.path, r.path[1:])), 2)
                row["mean uncertainty"] = round(float(np.mean([cm.uncertainty[p] for p in r.path])), 3)
                if frac is not None:
                    ev = evaluate_path(r.path, frac, labelled, tcfg.class_costs, tcfg.high_risk, tcfg.impassable,
                                       cm.terrain_cost)
                    row["GT high-risk frac"] = None if ev["gt_high_risk_frac"] is None else round(ev["gt_high_risk_frac"], 3)
                    row["GT rock cells"] = ev["gt_impassable_cells"]
            row["behaviour"] = r.behaviour or "—"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("planner"), use_container_width=True)
        ur = results["uncertainty"]
        if ur.log:
            st.markdown("**Uncertainty-aware decision log**")
            for line in ur.log:
                st.code(line, language=None)
        if gt is None:
            st.caption("Uploaded image: no ground truth, so GT path metrics are not available. "
                       "No rover-body mask is applied.")
    st.session_state.live = {"img": base, "probs": probs, "ent": ent, "cm": cm, "results": results, "grid": g,
                             "gt": gt, "thr": (thr_a, thr_b), "tcfg": tcfg}


def page_rover():
    st.title("Rover run")
    live = st.session_state.get("live")
    if not live:
        st.info("Open **Live demo** first to choose an image and plan paths.")
        return
    choice = st.radio("Path to drive", list(PLANNERS), horizontal=True, index=2, format_func=PLANNER_LABELS.get)
    path = live["results"][choice].path
    if not path:
        st.error("This planner found no path.")
        return
    speed = st.slider("Animation delay (s/step)", 0.01, 0.3, 0.06)
    cm, g, ent = live["cm"], live["grid"], live["ent"]
    size = live["img"].shape[0]
    pred_grid = np.stack([cm.terrain_cost]).squeeze()
    probs_g = live["probs"].reshape(INFO.num_classes, g, size // g, g, size // g).mean((2, 4))
    a, b = live["thr"]
    if st.button("▶ Drive", type="primary"):
        c1, c2 = st.columns([1, 1])
        ph_map, ph_stat, ph_plot = c1.empty(), c2.empty(), c2.empty()
        cum, xs = [0.0], [0.0]
        for k, p in enumerate(path):
            if k:
                d = float(np.hypot(p[0] - path[k - 1][0], p[1] - path[k - 1][1]))
                cum.append(cum[-1] + d * pred_grid[p]); xs.append(xs[-1] + d)
            u = float(cm.uncertainty[p])
            state = behaviour_from_uncertainty(u, a, b)
            ph_map.image(draw_paths(live["img"], g, {choice: path}, path[0], path[-1], rover=p))
            cls = INFO.names[int(probs_g[:, p[0], p[1]].argmax())]
            icon = {"proceed": "🟢", "slow": "🟡", "stop": "🔴"}[state]
            ph_stat.markdown(f"**Step {k + 1}/{len(path)}** · cell {p}  \n"
                             f"Predicted terrain: **{cls}** (P={probs_g[:, p[0], p[1]].max():.2f})  \n"
                             f"Local uncertainty: **{u:.3f}** → {icon} **{state.upper()}**  \n"
                             f"Accumulated predicted cost: **{cum[-1]:.2f}**")
            fig = go.Figure(go.Scatter(x=xs, y=cum, mode="lines"))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="running cost",
                              xaxis_title="distance (cells)")
            ph_plot.plotly_chart(fig, use_container_width=True, key=f"rc{k}")
            time.sleep(speed * (2 if state == "slow" else 4 if state == "stop" else 1))
        st.success(f"Reached goal. Planner-level decision: {live['results'][choice].behaviour or 'n/a'}")
    st.caption("Per-step speed state uses the *local* cell uncertainty with the same thresholds A/B as the "
               "path-level decision. Speeds are illustrative (no dynamics model).")


def page_models():
    st.title("Model performance")
    exps = experiments()
    if not exps:
        not_run("No experiments trained.")
        return
    rows = []
    for e in exps:
        m = read_json(EXP_DIR / e / "metrics_test.json")
        ts = read_json(EXP_DIR / e / "train_summary.json") or {}
        cfg = read_json(EXP_DIR / e / "config.json") or {}
        r = {"experiment": e, "arch": cfg.get("arch"), "loss": cfg.get("loss"), "aug": cfg.get("aug"),
             "epochs": ts.get("epochs"), "train min": round(ts.get("train_seconds", 0) / 60, 1),
             "best val mIoU": ts.get("best_val_mIoU")}
        if m:
            s = m["segmentation"]["deterministic"]
            r.update({"test mIoU": s["mIoU"], "test mIoU (MC)": m["segmentation"]["mc"]["mIoU"],
                      "pixel acc": s["pixel_accuracy"], "mean Dice": s["mean_dice"],
                      **{f"IoU {n}": s["per_class"][n]["IoU"] for n in INFO.names},
                      "latency s (det)": m["efficiency"]["latency_s_deterministic_median"],
                      "latency s (MC)": m["efficiency"]["latency_s_mc_median"],
                      "params M": m["efficiency"]["n_params"] / 1e6})
        else:
            r["test mIoU"] = "not evaluated"
        rows.append(r)
    st.markdown("**Experiment comparison (test split, deterministic unless marked MC)**")
    st.dataframe(pd.DataFrame(rows).set_index("experiment"), use_container_width=True)
    e = st.selectbox("Inspect experiment", exps, index=exps.index(selected_experiment()) if selected_experiment() in exps else 0)
    m = read_json(EXP_DIR / e / "metrics_test.json")
    c1, c2 = st.columns(2)
    with c1:
        show_img(EXP_DIR / e / "training_curves.png", "Training curves")
    with c2:
        show_img(EXP_DIR / e / "confusion_matrix.png", "Confusion matrix (test, MC mean)")
    if m:
        pc = m["segmentation"]["deterministic"]["per_class"]
        st.dataframe(pd.DataFrame(pc).T, use_container_width=True)
        if m.get("by_group"):
            st.markdown("**Generalisation: test mIoU by subset**")
            st.dataframe(pd.DataFrame({g: {"mIoU": v["mIoU"], "pixel acc": v["pixel_accuracy"],
                                           **{n: v["per_class"][n]["IoU"] for n in INFO.names}}
                                       for g, v in m["by_group"].items()}).T, use_container_width=True)
    show_img(EXP_DIR / e / "qualitative.png", "Qualitative test predictions: image / GT / prediction / entropy")


def page_uncertainty():
    st.title("Uncertainty quality")
    e = selected_experiment()
    m = read_json(EXP_DIR / e / "metrics_test.json") if e else None
    if not m:
        not_run("Run `python -m marvis.evaluate <experiment>`.")
        return
    st.markdown(f"Experiment `{e}` · {m['n_images']} test images · {m['uncertainty_pixels_used']:,} labelled pixels sampled · "
                f"T = {m['mc_samples']} MC samples")
    df = pd.DataFrame(m["uncertainty"]).T
    st.dataframe(df, use_container_width=True)
    st.caption("error_detection_AUROC: how well the uncertainty score ranks wrong pixels above correct ones "
               "(0.5 = useless, 1 = perfect). AURC: area under risk–coverage (lower is better). ECE: calibration error.")
    c1, c2 = st.columns(2)
    with c1:
        show_img(EXP_DIR / e / "reliability.png")
    with c2:
        show_img(EXP_DIR / e / "risk_coverage.png")


def page_planner():
    st.title("Planner comparison (ML → navigation)")
    e = selected_experiment()
    s = read_json(EXP_DIR / e / "planning_summary.json") if e else None
    if not s:
        not_run("Run `python -m marvis.evaluate <exp> --save-preds` then `python -m marvis.evaluate_planning <exp>`.")
        return
    st.markdown(f"`{e}` · {s['n_images']} test images × {s['pairs_per_image']} fixed start/goal pairs. "
                "Path risk and success are measured against **ground-truth labels**; costs use the assumed YAML table.")
    st.dataframe(pd.DataFrame(s["table"]).set_index("planner"), use_container_width=True)
    st.markdown(f"**Uncertainty-aware behaviour counts:** {s['uncertainty_behaviour_counts']}")
    show_img(EXP_DIR / e / "planner_comparison.png")
    csv = EXP_DIR / e / "planning_results.csv"
    if csv.exists():
        with st.expander("Per-trial results"):
            st.dataframe(pd.read_csv(csv), use_container_width=True)


def page_limitations():
    st.title("Limitations")
    p = ROOT / "LIMITATIONS.md"
    st.markdown(p.read_text() if p.exists() else "See README.")


PAGES = {"Overview": page_overview, "Dataset": page_dataset, "Live demo": page_live, "Rover run": page_rover,
         "Model performance": page_models, "Uncertainty": page_uncertainty, "Planner comparison": page_planner,
         "Limitations": page_limitations}
choice = st.sidebar.radio("MARVIS", list(PAGES))
PAGES[choice]()
