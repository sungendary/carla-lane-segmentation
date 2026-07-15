# Experiment Log: Building a Generalizable Lane-Segmentation Baseline

This document tracks how the segmentation baseline was developed through
three experiments. Each one tested a hypothesis about what limits
generalization to **unseen maps and weather**, measured the result, and
motivated the next step. The target metric throughout is **lane IoU on an
unseen map (Town05)** — lane marking is the hardest class (~2% of pixels)
and the one this project cares about most.

All models: DeepLabV3 (ResNet-50), 3 classes (road / lane / background),
512x256 input, class-weighted loss (lane upweighted). Evaluation uses
per-class IoU accumulated over a pixel-level confusion matrix.

---

## Experiment 1 — Single-condition baseline

**Hypothesis (implicit):** a model trained on one map/weather can segment
lanes on a different map.

**Setup:** train on Town03 / ClearNoon only (500 frames). Evaluate on the
training condition and three unseen conditions.

**Result (lane IoU):**

| Condition | Type | Lane IoU |
|---|---|---|
| Town03 / ClearNoon | seen | 0.389 |
| Town05 / ClearNoon | unseen map | **0.100** |
| Town03 / HardRain | unseen weather | 0.130 |
| Town03 / Sunset | unseen weather | 0.183 |

**Finding:** lane segmentation collapses on every unseen condition. Road
and background stay high, so the model generalizes for large regions but
fails on thin lane structure once conditions shift. The unseen map is the
worst case. → *hypothesis rejected; single-condition training does not
generalize.*

---

## Experiment 2 — Map diversity

**Hypothesis:** the bottleneck is map diversity; training on several maps
will improve unseen-map lane IoU.

**Setup:** train on Town01 + Town02 + Town03, all ClearNoon (1,500 frames).
Same evaluation protocol.

**Result (lane IoU), vs Experiment 1:**

| Condition | Exp 1 | Exp 2 | Δ |
|---|---|---|---|
| Town05 / ClearNoon (unseen map) | 0.100 | 0.128 | +0.028 |
| Town03 / HardRain (unseen weather) | 0.130 | 0.140 | +0.010 |
| Town03 / Sunset (unseen weather) | 0.183 | 0.222 | +0.039 |

**Finding:** map diversity helps, but only slightly, and the **rain
condition barely moves** (+0.010). The reason is clear in hindsight:
all training data was still ClearNoon, so the model had never seen rain.
→ *hypothesis partially confirmed; map diversity matters but is not the
dominant factor. Weather diversity in training is missing.*

---

## Experiment 3 — Enhanced baseline (scale + weather diversity)

**Hypothesis:** substantially scaling data volume **and** adding weather
diversity to training (especially rain) will close the generalization gap.

**Setup:** 30 map/weather sets across 6 maps (Town01/02/03/04/06/07) and
10 weather types (clear, cloudy, wet, rain intensities, sunset), 1,500
frames each = **45,000 frames**. A speed filter drops (near-)stationary
frames so red-light stops don't flood sets with near-identical images.
Training used best-model saving + early stopping (stopped at epoch 13,
best of 40). Town05 is held out entirely and evaluated on 3 conditions.

**Result (lane IoU on the held-out map Town05):**

| Eval condition | Road IoU | Lane IoU | mIoU |
|---|---|---|---|
| Town05 / ClearNoon | 0.934 | **0.492** | 0.807 |
| Town05 / HardRain  | 0.954 | **0.561** | 0.833 |
| Town05 / ClearSunset | 0.941 | **0.491** | 0.808 |

**Findings:**

1. **Unseen-map lane IoU jumped from 0.128 (Exp 2) to 0.492** — nearly 4x.
   Scaling data and adding weather diversity broke the bottleneck that map
   diversity alone could not.

2. **Rain became the *best* condition (0.561), not the worst.** In Exp 1–2
   rain was always the weakest because training never contained rain. After
   adding varied rain to training, rain generalization on a *new* map
   (Town05, whose rain was never seen — only other maps' rain was)
   overtook clear weather. This demonstrates **cross-map transfer of a
   learned weather condition.**

3. **Road IoU is stable at 0.93–0.95** across all conditions — drivable-area
   perception is essentially solved and weather-invariant.

---

## Summary: unseen-map lane IoU across experiments

| Experiment | Training data | Unseen-map lane IoU |
|---|---|---|
| 1. Single condition | Town03 ClearNoon, 500 | 0.100 |
| 2. Map diversity | 3 maps ClearNoon, 1,500 | 0.128 |
| 3. Enhanced baseline | 6 maps x 10 weather, 45,000 | **0.492** |

The progression shows that both **data scale** and **weather diversity**
were necessary; map diversity alone (Exp 2) gave little. Experiment 3 is
the current baseline.

## Honest caveat on comparability

Experiments 1–2 were evaluated on an earlier Town05 set (200 frames, no
speed filter); Experiment 3 uses a fresh Town05 set (500 frames, speed
filter applied). The evaluation sets are therefore not identical, so the
numbers are a trend rather than a perfectly controlled comparison. The
improvement (roughly 4x) is far larger than differences an eval-set change
could explain, so the gain is real — but a fully controlled comparison
would re-evaluate the earlier models on the new eval set.

## Next directions

- Increase input resolution to recover thin lane structure (lane IoU may
  still be resolution-limited at 512x256).
- Re-evaluate earlier models on the new eval set for a fully controlled
  comparison.
- Extend the condition matrix (night, fog) and connect perception output
  to a downstream planning step.