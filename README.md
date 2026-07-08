# CARLA Lane Segmentation

Semantic segmentation for autonomous-driving perception in the
[CARLA](https://carla.org/) simulator. The model segments each camera
pixel into **drivable road**, **lane marking**, or **background**, and the
project measures how well that perception **generalizes to unseen maps and
weather** — not just whether it works on the training condition.

This is a personal research project built to study the perception stage of
an autonomous-driving pipeline end to end: synchronized data collection,
automatic ground-truth labeling, model training, and quantitative
per-class evaluation.

> **Status:** active. The current milestone establishes a baseline model
> and a generalization benchmark. Subsequent work improves generalization
> and is tracked through commit history.

---

## Highlights

- **Fully automatic labeling.** Ground-truth segmentation masks come
  directly from CARLA's semantic-segmentation camera — no manual
  annotation. Data is collected in **synchronous mode**, so every RGB frame
  and its label mask are guaranteed to correspond to the same simulation
  tick.
- **Generalization is measured, not assumed.** The model is evaluated on
  an unseen map and unseen weather conditions, with per-class IoU, to
  expose where perception breaks down.
- **Reproducible pipeline.** Datasets and weights are not committed (they
  are large and regenerable); the scripts below reproduce them from
  scratch.

---

## Key result

The model was trained on a **single condition** (Town03 / ClearNoon) and
evaluated on the training condition plus three unseen ones. Lane marking is
the target class and the hardest to segment (~2% of pixels).

| Condition | Type | Road IoU | Lane IoU | mIoU |
|---|---|---|---|---|
| Town03 / ClearNoon | seen (baseline) | 0.960 | **0.389** | 0.782 |
| Town05 / ClearNoon | unseen map | 0.778 | **0.100** | 0.597 |
| Town03 / HardRain  | unseen weather | 0.887 | **0.130** | 0.658 |
| Town03 / Sunset    | unseen weather | 0.887 | **0.183** | 0.685 |

**Finding:** lane segmentation collapses under every unseen condition, and
an **unseen map degrades performance more than unseen weather** — the model
overfit to the geometry of its single training map. This points to map
diversity as the primary generalization bottleneck. Full analysis in
[`results/baseline_comparison.md`](results/baseline_comparison.md).

### Prediction example

Left to right: input RGB, ground-truth mask, model prediction
(gray = road, yellow = lane, black = background).

![prediction result](results/prediction_result.png)

---

## Pipeline

```
collect_data.py   ->   train.py   ->   evaluate.py / predict.py
 (RGB + labels)        (DeepLabV3)      (per-class IoU / visualization)
```

| Script | Role |
|---|---|
| `src/collect_data.py` | Drives an autopilot vehicle in synchronous mode and saves paired RGB + semantic-label frames for a chosen map/weather. |
| `src/dataset.py` | PyTorch `Dataset`: loads RGB/label pairs and remaps CARLA's raw class IDs to the 3 project classes. |
| `src/model.py` | Builds a DeepLabV3 (ResNet-50) with its output layer replaced for 3 classes. |
| `src/train.py` | Training loop with a class-weighted loss (lanes upweighted to counter class imbalance). |
| `src/evaluate.py` | Confusion-matrix based per-class IoU / mIoU over a dataset. |
| `src/predict.py` | Runs inference on one sample and saves an RGB / ground-truth / prediction comparison image. |
| `tools/check_label.py` | Inspects the raw class IDs present in a label file. |
| `learning/mission*.py` | Step-by-step scripts written while learning the CARLA API (spawn, camera, synchronous mode). |

---

## Setup

**Requirements**

- CARLA 0.9.16
- Python 3.10, PyTorch (CUDA build)
- An NVIDIA GPU (developed on an RTX 4080, 16 GB)

```bash
# in your CARLA python environment
pip install -r requirements.txt
```

The CARLA server must be running before any data-collection script:

```bash
cd /path/to/CARLA_0.9.16
./CarlaUE4.sh
```

## Reproduce

Datasets and the trained model are not stored in this repo. Regenerate them:

**1. Collect data.** Set `MAP_NAME`, `WEATHER`, and `SAVE_SUBDIR` at the top
of `src/collect_data.py`, then run it once per condition. For the baseline:

```bash
python src/collect_data.py     # e.g. Town03 / ClearNoon -> dataset/town03_clear
```

**2. Train** on the collected condition:

```bash
python src/train.py            # writes seg_model.pth
```

**3. Evaluate** on any condition by setting `EVAL_DIR` in `src/evaluate.py`:

```bash
python src/evaluate.py         # prints per-class IoU and mIoU
```

**4. Visualize** a prediction:

```bash
python src/predict.py          # writes a comparison image
```

---

## Roadmap

- [ ] Retrain with multiple maps to test the map-diversity hypothesis and
      raise unseen-map lane IoU above the 0.100 baseline.
- [ ] Increase input resolution to recover thin lane structure.
- [ ] Expand the condition matrix (more maps, night, fog) into a fuller
      generalization benchmark.
- [ ] Connect perception output to a downstream planning step.

## Notes

Class remapping (CARLA raw ID -> project class): `Road (1) -> 0`,
`RoadLine (24) -> 1`, everything else `-> 2`. Labels are resized with
nearest-neighbor interpolation to avoid creating invalid intermediate
class IDs.
