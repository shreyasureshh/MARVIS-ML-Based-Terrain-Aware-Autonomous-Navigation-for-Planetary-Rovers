### What is measured vs. assumed

| Measured on held-out data | Assumed / approximated |
|---|---|
| Segmentation quality (IoU, Dice, precision, recall, confusion matrix) | Terrain → cost table (`configs/traversability.yaml`) |
| Uncertainty quality (error-detection AUROC, AURC, ECE) | Rover footprint radius in grid cells |
| Fraction of each planned path on ground-truth high-risk terrain | Uncertainty thresholds A/B for proceed / slow / stop |
| Whether a path enters a ground-truth big-rock cell | That "big rock" pixels are physically impassable |

### Known limitations

- **Terrain class is a proxy, not traversability.** AI4MARS labels terrain *type*. It does not measure slip,
  sinkage, slope, rock height or traction. "Sand = high cost" is an engineering prior, not a learned fact.
- **Image-space planning.** The planning grid is the camera image downsampled to 64×64. There is no
  ground-plane / stereo projection, so a cell near the horizon covers far more ground than one near the
  rover, and path "length" is in image cells, not metres.
- **Planner evaluation uses ground-truth labels as the oracle.** These are crowd-sourced and partly
  unlabelled. Unlabelled cells are counted as "unknown" and excluded from the risk ratios.
- **MC Dropout is an approximation** to Bayesian uncertainty. Dropout is applied only at the encoder
  bottleneck and before the head. It captures model uncertainty poorly far from the training data.
- **Small-scale training.** Training ran on a CPU-only machine with a subset of AI4MARS (see README for
  the exact numbers). Absolute scores are not comparable to published full-dataset results.
- **Single seed per experiment.** Differences between experiments smaller than run-to-run noise are not
  meaningful. We did not have the compute to run multiple seeds.
- **Uploaded images** have no ground truth or rover-body mask, so only predictions are shown for them.
- **Rover run animation** is illustrative: speed states come from per-cell uncertainty. There is no
  vehicle dynamics model.
- **ROS 2 / Gazebo integration is not implemented** (design only; see README "Future work").
