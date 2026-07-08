# Baseline Evaluation: Generalization Across Unseen Conditions

This document reports the first (baseline) evaluation of the semantic
segmentation model. The model was trained on a **single map and a single
weather condition** (Town03, ClearNoon), then evaluated on the training
condition and on three **unseen** conditions to measure how well it
generalizes.

## Setup

- **Training data:** Town03 / ClearNoon, 500 frames
- **Model:** DeepLabV3 (ResNet-50 backbone), fine-tuned to 3 classes
- **Classes:** `0 = drivable road`, `1 = lane marking`, `2 = background`
- **Input resolution:** 512 x 256
- **Metric:** per-class IoU and mean IoU (mIoU), accumulated over a
  pixel-level confusion matrix across all evaluation frames
- **Evaluation frames:** 500 for the training condition, 200 for each
  unseen condition

## Results

| Condition | Type | Road IoU | Lane IoU | Background IoU | mIoU |
|---|---|---|---|---|---|
| Town03 / ClearNoon | seen (baseline) | 0.960 | **0.389** | 0.998 | 0.782 |
| Town05 / ClearNoon | unseen map | 0.778 | **0.100** | 0.913 | 0.597 |
| Town03 / HardRain  | unseen weather | 0.887 | **0.130** | 0.957 | 0.658 |
| Town03 / Sunset    | unseen weather | 0.887 | **0.183** | 0.986 | 0.685 |

(Lane IoU is highlighted because lane marking is the target class of this
project and the hardest to segment: it occupies only ~2% of pixels.)

## Findings

**1. Lane segmentation collapses under every unseen condition.**
The lane IoU drops from a baseline of 0.389 to 0.100–0.183 whenever the
map or weather changes. Road and background stay comparatively high, so
the model generalizes for large regions but fails on the thin, fine-grained
lane structure once conditions shift.

**2. Map change hurts more than weather change.**
The unseen *map* (Town05) produces the lowest lane IoU (0.100) — lower than
either unseen *weather* condition (0.130 rain, 0.183 sunset). The same
ordering holds for road IoU: the unseen map drops road IoU to 0.778, while
the unseen weather conditions keep it at 0.887. This indicates the model
overfit to the road geometry and lane layout of the single training map,
and that map diversity — not weather diversity — is the primary
generalization bottleneck.

**3. Interpretation.**
Training on one map, one weather is the direct cause. The confusion-matrix
based per-class IoU makes the weakness measurable rather than anecdotal,
and establishes a baseline against which future improvements can be
quantified.

## Next step (hypothesis to test)

Retrain with **multiple maps** (e.g. Town01, Town02, Town03) mixed into the
training set, keep the evaluation protocol identical, and re-measure lane
IoU on the unseen map (Town05). If the map-diversity hypothesis is correct,
the unseen-map lane IoU should rise from its current 0.100. This closes one
full loop of *hypothesis -> experiment -> verification*.
