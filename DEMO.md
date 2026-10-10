# MARVIS: 5-minute demo script

**Prerequisite:** the pipeline has been run on real AI4MARS data (`scripts/run.sh data`, `all`, `compare`).
If a page shows **"Not run yet"**, say so; do not fill the gap verbally with numbers.
Launch with `scripts/run.sh dashboard`.

| Time | Click | Say |
|---|---|---|
| 0:00 | **1 · Overview** | "MARVIS asks four ML questions: how well can we segment Mars terrain, does it generalise, does the model know when it is wrong, and does that change where the rover drives?" Point at the pipeline diagram and the experiment-status table: "every row here comes from a file the pipeline wrote." |
| 0:40 | **2 · Dataset** | Usable pairs, raw label values, class-fraction table: "big rock is rare, which is why we test class weighting and Dice." Overlays: "black is ignored: null labels, rover body, and pixels beyond 30 m." Leakage panel: "train and val are split by rover site, and the expert test set is never mixed in." |
| 1:30 | **3 · Segmentation vs GT** | Pick a test image. "Prediction next to ground truth; white pixels are errors." Point at the entropy map: "uncertainty is high on class boundaries and in the far field." Read the per-image line comparing entropy on correct vs wrong pixels. |
| 2:20 | **4 · Model comparison** | Experiment table: compare E1 with E2–E5 and read the ± spread. "Only differences larger than the seed spread count; single-seed rows are labelled preliminary." Then show per-class IoU, the brightness tertiles and the expert-agreement variants for generalisation. |
| 3:00 | **5 · Uncertainty** | AUROC row: "0.5 would mean uncertainty is useless for spotting errors." Risk–coverage: "keeping only the most certain X % of pixels lowers the error to Y %." Reliability diagram: "is the model over-confident?" |
| 3:40 | **6 · Traversability & planning → Interactive** | Click a start and a goal. "Grey is shortest-path, blue minimises assumed terrain cost, green also avoids uncertain regions." Move the uncertainty-weight slider and show the decision log (proceed / slow / stop + replan). Then the **Held-out evaluation** tab: "scored against ground-truth labels on ≥ 30 test images: high-risk fraction, success, expanded nodes." |
| 4:30 | **Rover run** → ▶ Drive | "Speed state comes from local uncertainty; the running cost plot shows accumulated predicted cost." |
| 4:45 | **Limitations** | "Terrain class is a proxy, not traction. The grid is in image space and the costs are assumptions. Next step: ground-plane projection and a ROS 2 / Nav2 costmap layer." |
