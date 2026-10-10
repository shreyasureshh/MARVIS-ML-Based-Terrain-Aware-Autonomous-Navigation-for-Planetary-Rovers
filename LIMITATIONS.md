### What is measured vs. assumed

| Measured on held-out data | Assumed / approximated |
|---|---|
| Segmentation quality (IoU, Dice, precision, recall, confusion matrix) | Terrain → cost table (`configs/traversability.yaml`) |
| Uncertainty quality (error-detection AUROC, AURC, ECE) | Rover footprint radius in grid cells |
| Fraction of each planned path on ground-truth high-risk terrain | Which classes count as "high risk" / "impassable" |
| A* expanded nodes, path length in grid cells | Thresholds A/B: *calibrated* as validation quantiles, but the quantile choice (50 % / 90 %) is an assumption |
| Whether a path enters a ground-truth big-rock cell | That "big rock" pixels are physically impassable |

### Status

- **No real-data results exist yet.** The AI4MARS archive could not be reached from the build environment
  (network policy). All code was tested on generated software fixtures only.
- **Label and mask encodings are not yet verified.** The values 0–3 / 255 and "non-zero mask = excluded" are
  configured from the dataset documentation. `python -m marvis.prepare inspect` reports the actual values,
  and they must be checked before training.

### Known limitations

- **Terrain class is a proxy, not traversability.** AI4MARS labels terrain *type*. It does not measure slip,
  sinkage, slope, rock height or traction. "Sand = high cost" is an engineering prior, not a learned fact.
- **Image-space planning.** The planning grid is the camera image downsampled to 64×64. There is no
  ground-plane / stereo projection, so a cell near the horizon covers far more ground than one near the
  rover, and path "length" is in image cells, not metres.
- **Planner evaluation uses ground-truth labels as the oracle.** The test labels are expert labels, but
  pixels without agreement (or beyond range) are unlabelled. Cells that are less than 50 % labelled count as
  "unknown" and are excluded from the risk ratios.
- **MC Dropout is an approximation** to Bayesian uncertainty. Dropout is applied only at the encoder
  bottleneck and before the head. It captures model uncertainty poorly far from the training data.
- **CPU-scale training.** The defaults target a 4-core CPU and a subset of AI4MARS, so absolute scores
  will not be comparable to published full-dataset results.
- **Seeds.** Results from one seed are labelled *preliminary*. A method is only called better when the
  difference exceeds the seed-to-seed spread.
- **Fixed start/goal pairs** in image coordinates: the rover is assumed at the bottom centre of every
  image. Real goals would come from a mission planner.
- **Label noise.** Training labels are crowd-sourced. Expert test labels have agreement variants, which we
  report separately.
- **Uploaded images** have no ground truth or rover-body mask, so only predictions are shown for them.
- **Rover run animation** is illustrative: speed states come from per-cell uncertainty. There is no
  vehicle dynamics model.
- **ROS 2 / Gazebo integration is not implemented** (design only; see README "Future work").
