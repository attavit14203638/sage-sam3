# Locked evaluation protocol

This document defines how every model in this repository is evaluated. All results in
[RESULTS.md](RESULTS.md) follow these rules.

## 1. Dataset and annotation semantics

- Dataset: OAM-TCD (`restor/tcd`), native 2048 × 2048 RGB tiles, nominal 0.10 m/pixel.
- Training: all 4,169 TRAIN images; no validation carve.
- Reporting: 439 TEST images, 30,646 annotations.
- Original labels: category 2 = individual tree (25,695 annotations); category 1 = unresolved
  canopy group (4,951 annotations).
- The reporting export collapses `category_id` to 1 (`tree`) while preserving
  `original_category_id`.
- Primary `--class-split tree`: non-tree annotations are set to `iscrowd=1`, preserving
  existing crowd flags. COCO matching decides which detections are ignored.
- `--class-split none` is a separate collapsed diagnostic view. Never difference a tree-only
  number against a collapsed number.

`Core/oam_tcd_protocol.py` validates image/annotation counts, native dimensions, source
split, collapsed reporting categories, and annotation/image-ID hashes.

## 2. Model and checkpoint protocol

- Unified-Prompt SAM3 run (internal A0): `sam3_crop_frts`. Thirty epochs, seed 42, two NVIDIA A40 GPUs, bf16, effective
  image batch 16, crop from 896 to 1152, resized to 1008, D4/photometric augmentation, full-resolution
  semantic auxiliary supervision, native O2M, final-layer sampled focal/Dice masks.
- Granularity-aware run: `sam3_prompt_granularity`. Identical recipe and initialization
  (original pre-trained `sam3.pt`); training supervises two category-preserving text queries
  (`tree`, `tree canopy`) per image with a 1/√2 outer image-weighting correction, and
  inference uses the single `tree` prompt.
- Report the **last-epoch exported weights**, not the optional best checkpoint.
- The authoritative checkpoint for any prediction dump is recorded in that dump's
  `inference_config.json`.

## 3. Locked inference

| Setting | Value |
|---|---|
| SAM3 prompt at inference | `tree` |
| Tile size | 1024 px |
| Overlap | 256 px |
| SAM3 model resolution | 1008 px |
| Merge | mask-NMS |
| Mask-NMS IoU threshold | 0.30 |
| Detection score threshold | 0.0 |
| Minimum mask area | 4 px |
| Inference / evaluation maxDets | 512 |
| Test-time augmentation | None |

`Core/project_workflows.py` supplies these settings explicitly, and
`Core/device_policy.py::check_dumps_comparable` verifies two dumps agree on every
prediction-affecting field before any delta is read.

## 4. Metrics and boundary bands

Report Mask AP (IoU 0.50:0.05:0.95), AP50, AP75, size-binned AP, and AR@512. AP is a
score-ordered COCO evaluation, not the Hungarian assignment used during training.

Boundary AP uses the same COCO machinery with pairwise similarity
`min(Mask IoU, Boundary IoU)`. Keep the three band conventions distinct:

| Evaluation JSON key | Band | Role |
|---|---|---|
| `boundary_ap_instance` | k = 0.19 × each mask's equivalent diameter | Instance-relative band; primary comparison |
| `boundary_ap_fine` | uniform 8 source pixels | Uniform 8-pixel band; compatibility key |
| `boundary_ap` | 0.02 times image diagonal (approximately 58 px) | Image-relative band; literature comparison |

For a binary mask M, the instance-relative implementation uses:

- `D_eq = 2 * sqrt(count_nonzero(M) / pi)`, using raster mask area, not bbox/COCO metadata area.
- `d(M) = max(1, round(0.19 * D_eq))`, independently for prediction and GT.
- `B(M) = M minus erode(M, d(M))`, with a 3×3 all-ones structuring element.
- Background padding makes an image-edge truncation part of the contour.

This is Chebyshev erosion, not Euclidean. The band is broad and the score is a
boundary-sensitive overlap relative to instance scale, not a fixed physical edge-error
allowance. k = 0.19 was fixed before the final model comparison and remains locked. In the
matched final individual-tree evaluations, every size-bin retention ratio remains inside the
declared usable window for Unified-Prompt SAM3 and the Mask R-CNN reference. Exact preliminary
calibration estimates are not reported because that sweep combined different reporting populations.

## 5. Size bins and non-reporting diagnostics

Use the `eval_predictions.py`/COCO size-bin convention throughout a comparison; do not
difference size bins across tools with different binning metadata. Semantic-union Boundary
IoU, cgF1, and best-F1 threshold sweeps do not implement the crowd-ignore reporting contract
and are diagnostics only. AP is reported from the locked score-range predictions, not from a
test-selected operating threshold.

## 6. Efficacy criteria for the granularity-aware arm

Declared before training and bound to the frozen Unified-Prompt SAM3 evaluation
(`Context/active/prompt_granularity_efficacy.json`):

- Primary effect: Mask AP delta ≥ +0.010.
- Non-regression: instance Boundary AP delta ≥ −0.003 and Mask AP_small delta ≥ −0.003.
- The seed-42 outcome is descriptive. No standard deviation or training-seed variability estimate is reported.
