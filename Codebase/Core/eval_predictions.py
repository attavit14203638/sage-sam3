#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone evaluation of dumped COCO-format predictions.

Reported metrics (see Context/active/evaluation_protocol.md section 1):
  - Mask AP / AP50 / AP75 / size bins / AR@maxDets (official pycocotools COCOeval).
    AP50 is primary: it is the only metric the OAM-TCD dataset paper reports.
  - Boundary AP (Cheng et al., CVPR 2021), primary for boundary fidelity. Requires
    the optional boundary-iou-api; skipped with a hint if absent.

Internal diagnostics, computed but NOT for publication:
  - semantic-union Boundary IoU (BoundaryIoUEvaluator) -- a union metric, so it
    cannot express a per-instance claim and ignores iscrowd. Retained because it is
    the most sensitive detector of a refinement head degenerating into dilation.
  - cgF1 (appendix) and best instance-F1 over a score-threshold sweep (appendix).

Protocols: --class-split tree is the primary reporting protocol and marks canopy
GT iscrowd=1 (COCO ignore semantics); collapsed (--class-split none) is retained
for continuity. Only Mask AP and Boundary AP honour ignore regions, so the other
metrics are skipped rather than silently mis-reported when a split is active.

Usage:
  python eval_predictions.py \
      --pred /path/to/coco_predictions_segm.json \
      --gt   /path/to/test_annotations.coco.json \
      --out  /path/to/eval_results.json \
      --protocol oam_tcd_test --class-split tree
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from atomic_io import atomic_write_text

CORE_DIR = Path(__file__).resolve().parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _prepare_valid_region_predictions(pred_path: str, gt_path: str, out_path: str) -> tuple[str, dict[str, object]]:
    gt_data = json.loads(Path(gt_path).read_text())
    regions = gt_data.get("ignore_regions") or []
    if not regions:
        return pred_path, {"enabled": False, "input_predictions": None, "kept_predictions": None}

    from pycocotools import mask as mask_utils

    images = {int(image["id"]): image for image in gt_data.get("images", [])}
    regions_by_image: dict[int, list[dict]] = {}
    for region in regions:
        image_id = int(region["image_id"])
        image = images[image_id]
        height = int(image["height"])
        width = int(image["width"])
        region_segmentation = region["segmentation"]
        if isinstance(region_segmentation, dict):
            region_rle = dict(region_segmentation)
            if isinstance(region_rle.get("counts"), str):
                region_rle["counts"] = region_rle["counts"].encode("ascii")
        else:
            region_rles = mask_utils.frPyObjects(region_segmentation, height, width)
            region_rle = mask_utils.merge(region_rles)
        regions_by_image.setdefault(image_id, []).append(region_rle)

    predictions = json.loads(Path(pred_path).read_text())
    filtered: list[dict] = []
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        image_regions = regions_by_image.get(image_id)
        if not image_regions:
            continue
        segmentation = prediction.get("segmentation")
        if not isinstance(segmentation, dict):
            raise ValueError("Valid-region filtering requires RLE prediction segmentations.")
        pred_rle = dict(segmentation)
        if isinstance(pred_rle.get("counts"), str):
            pred_rle["counts"] = pred_rle["counts"].encode("ascii")
        clipped = pred_rle
        for region_rle in image_regions:
            clipped = mask_utils.merge([clipped, region_rle], intersect=True)
        area = int(mask_utils.area(clipped))
        if area <= 0:
            continue
        kept = dict(prediction)
        clipped_json = dict(clipped)
        if isinstance(clipped_json.get("counts"), bytes):
            clipped_json["counts"] = clipped_json["counts"].decode("ascii")
        kept["segmentation"] = clipped_json
        kept["bbox"] = [float(value) for value in mask_utils.toBbox(clipped).tolist()]
        kept["area"] = area
        filtered.append(kept)

    filtered_path = Path(out_path).with_name(Path(out_path).stem + "_valid_region_predictions.json")
    atomic_write_text(filtered_path, json.dumps(filtered))
    return str(filtered_path), {
        "enabled": True,
        "input_predictions": len(predictions),
        "kept_predictions": len(filtered),
        "regions": len(regions),
        "filtered_predictions_path": str(filtered_path),
    }


# OAM-TCD original category IDs, preserved per-annotation by dataset_adapter.py
# while category_id is collapsed to 1 for SAM3's single text prompt. The order is
# the reverse of the dataset documentation's listing, so it must not be guessed:
# cat 2 is the individual tree (83.8% of TEST, median 1,765px^2) and cat 1 is the
# closed-canopy group (16.2%, median 15,311px^2).
TREE_ORIGINAL_CATEGORY_ID = 2
CANOPY_ORIGINAL_CATEGORY_ID = 1
CLASS_SPLIT_KEEP_ID = {
    "tree": TREE_ORIGINAL_CATEGORY_ID,
    "canopy": CANOPY_ORIGINAL_CATEGORY_ID,
}


def _prepare_class_split_gt(gt_path: str, class_split: str, out_path: str) -> tuple[str, dict[str, object]]:
    """Derive a GT file that scores only one OAM-TCD class, ignoring the other.

    The detector is class-agnostic (categories are collapsed to a single SAM3 text
    prompt), so a per-class report cannot come from category filtering. Instead the
    other class is marked `iscrowd=1`, which is COCO's ignore semantics: a detection
    matching an ignored GT is neither a true positive nor a false positive, and
    ignored GT do not enter the recall denominator. That isolates mask quality on the
    target class without penalising the model for finding objects this protocol
    declines to score.

    Deleting the other class instead would turn every canopy-matched prediction into
    a false positive, conflating mask quality with a detection penalty. That
    alternative is deliberately not implemented.
    """
    if class_split == "none":
        return gt_path, {"enabled": False, "class_split": "none"}

    keep_id = CLASS_SPLIT_KEEP_ID.get(class_split)
    if keep_id is None:
        raise ValueError(f"Unknown class split {class_split!r}; expected one of {sorted(CLASS_SPLIT_KEEP_ID)} or 'none'.")

    gt_data = json.loads(Path(gt_path).read_text())
    annotations = gt_data.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError(f"{gt_path} has no annotations list; cannot apply a class split.")

    without_field = sum(1 for annotation in annotations if "original_category_id" not in annotation)
    if without_field:
        # Failing loudly matters: silently treating a missing field as "not the
        # target class" would ignore the entire dataset and report AP on nothing.
        raise ValueError(
            f"{without_field} of {len(annotations)} annotations in {gt_path} lack "
            "'original_category_id', so a tree/canopy split cannot be derived. This "
            "export predates the field, or was not produced by dataset_adapter.py."
        )

    scored = 0
    ignored = 0
    already_crowd = 0
    for annotation in annotations:
        if int(annotation["original_category_id"]) == keep_id:
            if int(annotation.get("iscrowd", 0)) == 1:
                already_crowd += 1
            scored += 1
        else:
            annotation["iscrowd"] = 1
            ignored += 1

    if scored == 0:
        raise ValueError(
            f"class split {class_split!r} left zero scored annotations in {gt_path}; "
            f"present original_category_ids are "
            f"{sorted({int(a['original_category_id']) for a in annotations})}."
        )

    split_path = Path(out_path).with_name(Path(out_path).stem + f"_gt_{class_split}.json")
    atomic_write_text(split_path, json.dumps(gt_data))
    return str(split_path), {
        "enabled": True,
        "class_split": class_split,
        "kept_original_category_id": keep_id,
        "scored_annotations": scored,
        "ignored_annotations": ignored,
        "scored_already_iscrowd": already_crowd,
        "split_gt_path": str(split_path),
        "mechanism": "iscrowd=1 on the non-target class (COCO ignore semantics)",
    }


def _ap_at_max_dets(evaluator, max_dets: int) -> float:
    max_dets_index = evaluator.params.maxDets.index(max_dets)
    precision = evaluator.eval["precision"][:, :, :, 0, max_dets_index]
    valid_precision = precision[precision > -1]
    return float(valid_precision.mean()) if valid_precision.size else -1.0


def run_coco_eval(pred_path: str, gt_path: str, max_dets: int = 512) -> dict[str, float]:
    """Runs pycocotools COCOeval with params.maxDets set to [1, 10, max_dets].

    pycocotools defaults to maxDets=[1, 10, 100], which caps the top-scored
    detections considered per image at 100. Tiled inference merges predictions
    from multiple overlapping tiles per image (2x2 -> up to ~4x a single
    tile's query cap before mask-NMS dedup), so per-image counts routinely
    exceed 100. `max_dets=512` is the project's canonical cap, matching the
    OAM-TCD dataset paper's DETECTIONS_PER_IMAGE=512 setting for fair
    comparison. Using the pycocotools default of 100 would silently
    truncate and understate both precision and recall.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)
    e = COCOeval(coco_gt, coco_dt, "segm")
    e.params.maxDets = [1, 10, max_dets]
    e.evaluate()
    e.accumulate()
    e.summarize()
    stats = e.stats
    return {
        "AP": _ap_at_max_dets(e, max_dets),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR@1": float(stats[6]),
        "AR@10": float(stats[7]),
        f"AR@{max_dets}": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }


def run_boundary_iou(pred_path: str, gt_path: str) -> dict[str, float]:
    """Semantic-union boundary IoU. INTERNAL DIAGNOSTIC ONLY -- do not report.

    This merges every GT instance into one mask per image and every prediction into
    another, so it cannot express a per-instance claim and it ignores `iscrowd`
    (making it a silent no-op under --class-split). It is retained because a union
    metric is unusually sensitive to a global area bias, which makes it the best
    available detector for a refinement head degenerating into learned dilation.
    The reportable boundary metric is `run_boundary_ap`. See
    Context/active/evaluation_protocol.md section 1.1.
    """
    from eval_overrides import BoundaryIoUEvaluator

    evaluator = BoundaryIoUEvaluator(gt_path, iou_type="segm")
    return evaluator.evaluate(pred_path)


BOUNDARY_AP_INSTALL_HINT = (
    "pip install git+https://github.com/bowenc0221/boundary-iou-api.git"
)

# Upstream default. On COCO, where an object fills much of the frame, 2% of the image diagonal is a thin rind. On a 2048px aerial tile it is 57px -- wider than most crowns -- which makes Boundary IoU collapse into Mask IoU. See Context/active/evaluation_protocol.md section 1.3.

BOUNDARY_AP_LITERATURE_RATIO = 0.02

# Fine comparison band in source pixels, fixed from the baseline saturation sweep:
#   - 16px is provably degenerate on the target population -- Boundary AP_small equals Mask AP_small to three decimals (0.2374) and AR_small pins at 0.389.
#   - 8px is the largest non-degenerate band: AP50 0.388 vs Mask 0.712.
#   - 4px and 2px are floored, AP75 0.001 and 0.000, so they cannot resolve an improvement at the current baseline.

BOUNDARY_AP_FINE_BAND_PX = 8.0

# Bands for the one-off saturation sweep that locates the collapse empirically rather than by the erosion argument alone.
BOUNDARY_AP_SWEEP_BANDS_PX = (2.0, 4.0, 8.0, 16.0, 32.0, 57.0)


def dataset_image_diagonal(gt_path: str) -> tuple[float, tuple[int, int]]:
    """Image diagonal in pixels, requiring a uniform tile size.

    The upstream API expresses the boundary band as a fraction of the image
    diagonal, so converting a band in source pixels into that fraction is only
    well defined when every image is the same size. Raising here is deliberate:
    on a mixed-size dataset one ratio would silently mean a different physical
    band per image, which is the very failure this conversion exists to remove.
    """
    with open(gt_path) as handle:
        images = json.load(handle)["images"]
    if not images:
        raise ValueError(f"{gt_path} declares no images")
    sizes = {(int(i["height"]), int(i["width"])) for i in images}
    if len(sizes) != 1:
        raise ValueError(
            f"{gt_path} mixes image sizes {sorted(sizes)}; a single dilation_ratio "
            "would mean a different boundary band per image. Evaluate uniform "
            "subsets separately."
        )
    height, width = sizes.pop()
    return float(np.hypot(height, width)), (height, width)


def boundary_band_to_ratio(band_px: float, diagonal: float) -> float:
    """Convert a band in source pixels to the upstream image-relative ratio."""
    if band_px <= 0:
        raise ValueError(f"band_px must be positive, got {band_px}")
    return float(band_px) / float(diagonal)


def run_boundary_ap(
    pred_path: str,
    gt_path: str,
    max_dets: int = 512,
    dilation_ratio: float = 0.02,
    band_mode: str = "image_relative",
    band_k: float | None = None,
) -> dict[str, Any]:
    """Boundary AP (Cheng et al., CVPR 2021), the reportable boundary metric.

    Per-instance, evaluated inside the COCO AP protocol, so it inherits maxDets and
    correct `iscrowd` ignore semantics and therefore supports the tree-only protocol
    with no matching or clipping workaround. The upstream API scores with
    min(Mask IoU, Boundary IoU) per the paper's supplement, which prevents a ring
    mask from scoring perfectly against a disc.

    `dilation_ratio` is kept at the upstream default of 0.02, matching
    BoundaryIoUEvaluator, so the two boundary measurements use the same band width
    and a disagreement between them is attributable to population rather than scale.

    Returns {"available": False, ...} rather than raising if the optional dependency
    is absent, so a missing package degrades one metric instead of voiding an
    otherwise complete evaluation.
    """
    try:
        from boundary_iou.coco_instance_api.coco import COCO as BoundaryCOCO
        from boundary_iou.coco_instance_api.cocoeval import COCOeval as BoundaryCOCOeval
    except ImportError as exc:
        logging.warning(
            "Boundary AP skipped: %s. Install with:\n  %s",
            exc,
            BOUNDARY_AP_INSTALL_HINT,
        )
        return {
            "available": False,
            "reason": f"import failed: {exc}",
            "install_hint": BOUNDARY_AP_INSTALL_HINT,
        }

    # band_mode selects HOW the band width is chosen, not what is computed with it.
    #   image_relative    -- upstream: dilation_ratio * image diagonal, one width for every
    #                        instance in the image. Degenerate on this domain (see 1.3, 0.7).
    #   instance_relative -- band_k * equivalent_diameter, per instance (Core/boundary_band).
    # Both are reported; the literature band must always accompany the instance-relative
    # figure so comparisons to published Boundary IoU numbers are not silently broken.
    if band_mode not in ("image_relative", "instance_relative"):
        raise ValueError(
            f"band_mode must be 'image_relative' or 'instance_relative', got {band_mode!r}"
        )
    if band_mode == "instance_relative":
        from boundary_band import BOUNDARY_BAND_K, patched_instance_relative_bands

        effective_k = BOUNDARY_BAND_K if band_k is None else float(band_k)
        band_context = patched_instance_relative_bands(effective_k)
    else:
        if band_k is not None:
            raise ValueError(
                "band_k is only meaningful for band_mode='instance_relative'; the "
                "image-relative band is set by dilation_ratio."
            )
        effective_k = None
        band_context = contextlib.nullcontext()

    try:
        with band_context:
            coco_gt = BoundaryCOCO(gt_path, get_boundary=True, dilation_ratio=dilation_ratio)
            coco_dt = coco_gt.loadRes(pred_path)
            evaluator = BoundaryCOCOeval(
                coco_gt, coco_dt, iouType="boundary", dilation_ratio=dilation_ratio
            )
            evaluator.params.maxDets = [1, 10, max_dets]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
    except Exception as exc:
        # The upstream package targets NumPy < 1.20 and uses removed aliases such as
        # `np.float`. A runtime failure here must not discard an otherwise complete
        # evaluation: Mask AP has already been computed and costs minutes to redo.
        logging.exception("Boundary AP failed at runtime; reporting it as unavailable.")
        return {
            "available": False,
            "reason": f"runtime failure: {type(exc).__name__}: {exc}",
            "install_hint": BOUNDARY_AP_INSTALL_HINT,
        }

    stats = evaluator.stats
    return {
        "available": True,
        "band_mode": band_mode,
        "band_k": effective_k,
        "dilation_ratio": dilation_ratio,
        "iou_type": "boundary",
        "reference": "Cheng et al., Boundary IoU, CVPR 2021",
        "Boundary_AP": _ap_at_max_dets(evaluator, max_dets),
        "Boundary_AP50": float(stats[1]),
        "Boundary_AP75": float(stats[2]),
        "Boundary_AP_small": float(stats[3]),
        "Boundary_AP_medium": float(stats[4]),
        "Boundary_AP_large": float(stats[5]),
        f"Boundary_AR@{max_dets}": float(stats[8]),
    }


def boundary_gate_quantities(coco_eval: dict[str, Any], boundary_fine: dict[str, Any]) -> dict[str, Any]:
    """Boundary-to-mask ratios, i.e. the quantities the arm gate is defined on.

    A raw boundary number cannot separate "better boundaries" from "better masks",
    because the metric scores ``min(Mask IoU, Boundary IoU)`` and is therefore
    bounded above by its mask counterpart -- a system that merely produces better
    masks scores better Boundary AP. Normalising by the mask counterpart is what
    isolates the boundary component, so ``ratio_ap`` is the binding direction check
    for a boundary contribution: it must rise, not merely Boundary AP.

    The ratio across IoU thresholds is also a diagnostic of increasingly strict boundary
    agreement.

    Returns ``{"available": False}`` when Boundary AP could not be computed, so a
    missing optional dependency degrades this row instead of voiding the run.
    """
    if not boundary_fine.get("available"):
        return {"available": False, "reason": boundary_fine.get("reason", "boundary AP unavailable")}

    def ratio(boundary_key: str, mask_key: str) -> float:
        mask_value = float(coco_eval.get(mask_key, 0.0))
        if mask_value <= 0:
            return float("nan")
        return float(boundary_fine[boundary_key]) / mask_value

    return {
        "available": True,
        "band_px": boundary_fine.get("band_px"),
        "note": "ratio_ap is the binding direction check; it must rise versus the reference dump.",
        "ratio_ap": ratio("Boundary_AP", "AP"),
        "ratio_ap50": ratio("Boundary_AP50", "AP50"),
        "ratio_ap75": ratio("Boundary_AP75", "AP75"),
        "ratio_ap_small": ratio("Boundary_AP_small", "AP_small"),
        "ratio_ap_medium": ratio("Boundary_AP_medium", "AP_medium"),
        "ratio_ap_large": ratio("Boundary_AP_large", "AP_large"),
    }


def run_boundary_ap_sweep(
    pred_path: str,
    gt_path: str,
    mask_ap: float,
    bands_px: tuple[float, ...] = BOUNDARY_AP_SWEEP_BANDS_PX,
    max_dets: int = 512,
) -> dict[str, Any]:
    """Boundary AP across boundary band widths, to locate where it saturates.

    Boundary IoU only differs from Mask IoU while the eroded mask is non-empty.
    Once the band exceeds an instance's inradius, `mask XOR erosion(mask) == mask`
    and the metric is *identically* Mask IoU -- so Boundary AP silently stops
    measuring boundaries. This sweep shows at which band that happens on this
    dataset instead of relying on the erosion argument, which makes the reported
    fine band an evidence-based choice rather than one that could be accused of
    having been picked to flatter a result.

    Intended to be run ONCE on the baseline, not per arm. `mask_ap` is the Mask AP
    the same dump scores, which is the value each band is compared against.
    """
    diagonal, (height, width) = dataset_image_diagonal(gt_path)
    bands: dict[str, Any] = {}
    for band_px in bands_px:
        ratio = boundary_band_to_ratio(band_px, diagonal)
        print(f"\n--- Boundary AP at {band_px:g}px band (ratio {ratio:.6f}) ---")
        result = run_boundary_ap(pred_path, gt_path, max_dets=max_dets, dilation_ratio=ratio)
        if result.get("available"):
            boundary_ap = float(result["Boundary_AP"])
            result["band_px"] = float(band_px)
            # 1.0 means the band is wide enough that the metric has degenerated
            # into Mask AP and carries no boundary information at all.
            result["ratio_to_mask_ap"] = (
                boundary_ap / mask_ap if mask_ap > 0 else float("nan")
            )
        bands[f"{band_px:g}px"] = result
    return {
        "image_size": [height, width],
        "image_diagonal": diagonal,
        "mask_ap": mask_ap,
        "bands": bands,
        "note": (
            "ratio_to_mask_ap near 1.0 means the boundary band exceeds the crowns, "
            "erosion empties them, and Boundary IoU is identically Mask IoU. Choose "
            "the reported fine band from the largest width that is still clearly "
            "below 1.0."
        ),
    }


def run_cgf1(pred_path: str, gt_path: str, iou_type: str = "segm") -> dict[str, float]:
    """Official SAM3 cgF1 metric, using the project's eval_overrides patches
    (COCOCustom tolerant of plain COCO json, CGF1Eval threshold fixed to
    config.EVAL_CGF1_SCORE_THRESHOLD)."""
    import eval_overrides  # noqa: F401  (applies monkeypatches on import)
    from sam3.eval.cgf1_eval import CGF1Evaluator

    evaluator = CGF1Evaluator(gt_path=gt_path, iou_type=iou_type, verbose=True)
    return evaluator.evaluate(pred_path)


def run_f1_sweep(pred_path: str, gt_path: str, match_iou: float = 0.5) -> dict[str, float]:
    """Greedy score-sorted IoU>=match_iou matching, done once per image, then
    swept over score thresholds via prefix cutoffs (matching is prefix-stable
    since predictions are processed in descending score order)."""
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_utils

    coco_gt = COCO(gt_path)
    with open(pred_path) as f:
        preds = json.load(f)

    img_ids = sorted(coco_gt.imgs.keys())
    img_to_preds: dict[int, list[dict]] = {img_id: [] for img_id in img_ids}
    for p in preds:
        if p["image_id"] in img_to_preds:
            img_to_preds[p["image_id"]].append(p)

    total_gt = 0
    # Per image: sorted descending scores + cumulative TP count at each rank.
    per_image_scores: list[np.ndarray] = []
    per_image_cum_tp: list[np.ndarray] = []

    for img_id in img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_rles = [coco_gt.annToRLE(a) for a in anns]
        total_gt += len(anns)

        img_preds = sorted(img_to_preds[img_id], key=lambda p: p["score"], reverse=True)
        n_pred = len(img_preds)
        scores = np.array([p["score"] for p in img_preds], dtype=np.float64)
        cum_tp = np.zeros(n_pred, dtype=np.int64)

        if n_pred == 0 or len(gt_rles) == 0:
            per_image_scores.append(scores)
            per_image_cum_tp.append(cum_tp)
            continue

        pred_rles = [p["segmentation"] for p in img_preds]
        iou = np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)), dtype=np.float64)
        matched_gt = np.zeros(len(gt_rles), dtype=bool)
        running_tp = 0
        for r in range(n_pred):
            row = np.where(matched_gt, -1.0, iou[r])
            j = int(np.argmax(row))
            if row[j] >= match_iou:
                matched_gt[j] = True
                running_tp += 1
            cum_tp[r] = running_tp

        per_image_scores.append(scores)
        per_image_cum_tp.append(cum_tp)

    thresholds = np.arange(0.0, 1.0, 0.005)
    best_f1 = 0.0
    best_thr = 0.0
    best_tp = best_fp = best_fn = 0

    for thr in thresholds:
        tp = fp = 0
        for scores, cum_tp in zip(per_image_scores, per_image_cum_tp):
            # scores sorted descending -> prefix length with score >= thr.
            n_kept = int(np.searchsorted(-scores, -thr, side="right")) if len(scores) else 0
            img_tp = int(cum_tp[n_kept - 1]) if n_kept > 0 else 0
            tp += img_tp
            fp += n_kept - img_tp

        fn = total_gt - tp

        precision = tp / max(tp + fp, 1)
        recall = tp / max(total_gt, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_tp = tp
            best_fp = fp
            best_fn = fn

    return {
        "best_f1": best_f1,
        "threshold": best_thr,
        "precision": best_tp / max(best_tp + best_fp, 1),
        "recall": best_tp / max(total_gt, 1),
        "tp": best_tp,
        "fp": best_fp,
        "fn": best_fn,
        "total_gt": total_gt,
    }


SMALL_MAX_AREA = 32.0 * 32.0     # < 1024 px^2
MEDIUM_MAX_AREA = 96.0 * 96.0    # < 9216 px^2


def _load_coco_gt(gt_path: Path):
    from pycocotools.coco import COCO

    return COCO(str(gt_path))


def _mask_utils():
    from pycocotools import mask as mask_utils

    return mask_utils


def _build_image_index(coco_gt):
    """Return per-image GT records and pre-encoded GT segm RLEs."""
    img_to_gts: dict[int, list[dict[str, Any]]] = {}
    gt_rles_by_img: dict[int, list[dict[str, Any]]] = {}
    for img_id in coco_gt.imgs:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        img_to_gts[img_id] = anns
        rles = [coco_gt.annToRLE(ann) for ann in anns]
        for r in rles:
            if isinstance(r.get("counts"), str):
                r["counts"] = r["counts"].encode("ascii")
        gt_rles_by_img[img_id] = rles
    return img_to_gts, gt_rles_by_img


def _group_preds_by_image(pred_data: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for pred in pred_data:
        grouped.setdefault(pred["image_id"], []).append(pred)
    for img_id in grouped:
        grouped[img_id].sort(key=lambda p: p["score"], reverse=True)
    return grouped


def _precompute_ious(
    img_to_gts: dict[int, list[dict[str, Any]]],
    sorted_preds_by_img: dict[int, list[dict[str, Any]]],
    gt_rles_by_img: dict[int, list[dict[str, Any]]],
) -> dict[int, np.ndarray]:
    """Precompute the (n_pred x n_gt) segm-IoU matrix per image once."""
    mask_utils = _mask_utils()
    iou_by_img: dict[int, np.ndarray] = {}
    for img_id, preds in sorted_preds_by_img.items():
        gts = img_to_gts.get(img_id, [])
        if not preds or not gts:
            iou_by_img[img_id] = np.zeros((len(preds), len(gts)), dtype=np.float32)
            continue
        iscrowd = [int(g.get("iscrowd", 0)) for g in gts]
        pred_rles = [p["segmentation"] for p in preds]
        gt_rles = gt_rles_by_img[img_id]
        iou = mask_utils.iou(pred_rles, gt_rles, iscrowd)
        iou_arr = np.asarray(iou, dtype=np.float32)
        if iou_arr.ndim != 2:
            iou_arr = iou_arr.reshape(len(preds), len(gts))
        iou_by_img[img_id] = iou_arr
    return iou_by_img


def _greedy_match(iou_keep: np.ndarray, match_iou: float) -> np.ndarray:
    """Greedy one-to-one matching over score-sorted predictions (rows).

    Returns a boolean array of length n_gt indicating which GT got matched.
    Mirrors the matching used by eval_predictions.run_f1_sweep.
    """
    n_gt = iou_keep.shape[1]
    matched = np.zeros(n_gt, dtype=bool)
    for row in iou_keep:
        best_iou = -1.0
        best_idx = -1
        for j in range(n_gt):
            if matched[j]:
                continue
            v = float(row[j])
            if v > best_iou:
                best_iou = v
                best_idx = j
        if best_idx >= 0 and best_iou >= match_iou:
            matched[best_idx] = True
    return matched


def _size_bin(area: float) -> str:
    if area < SMALL_MAX_AREA:
        return "small"
    if area < MEDIUM_MAX_AREA:
        return "medium"
    return "large"


def _gt_count_bin(n_gt: int, budget: int) -> str:
    if n_gt == 0:
        return "empty"
    if n_gt < budget:
        return f"under_budget(<{budget})"
    return f"at_or_over_budget(>={budget})"


def _empty_size_acc() -> dict[str, dict[str, int]]:
    return {k: {"n_gt": 0, "matched": 0, "covered": 0} for k in ("small", "medium", "large")}


def diagnose(
    img_to_gts: dict[int, list[dict[str, Any]]],
    sorted_preds_by_img: dict[int, list[dict[str, Any]]],
    iou_by_img: dict[int, np.ndarray],
    match_iou: float,
    threshold: float,
    budget: int,
) -> dict[str, Any]:
    """Measure recall diagnostics at one score threshold: matched and covered ground-truth recall overall, by ground-truth count bin and by size, the redundancy gap, predictions per image against the budget, and duplication per covered ground truth."""
    n_images = len(img_to_gts)
    total_gt = 0
    total_matched = 0
    total_covered = 0
    total_pred_kept = 0
    preds_per_image: list[int] = []

    # duplication: #preds at IoU>=match per GT (only over covered GT)
    dup_counts: list[int] = []

    size_acc = _empty_size_acc()

    gt_count_acc: dict[str, dict[str, int]] = {}

    for img_id, gts in img_to_gts.items():
        n_gt = len(gts)
        total_gt += n_gt
        preds = sorted_preds_by_img.get(img_id, [])
        preds_per_image.append(len(preds))

        gbin = _gt_count_bin(n_gt, budget)
        gacc = gt_count_acc.setdefault(
            gbin, {"n_images": 0, "n_gt": 0, "matched": 0, "covered": 0, "n_pred_kept": 0}
        )
        gacc["n_images"] += 1
        gacc["n_gt"] += n_gt

        if n_gt == 0:
            continue

        if not preds:
            # no predictions: nothing matched/covered
            for g in gts:
                size_acc[_size_bin(float(g.get("area", 0.0)))]["n_gt"] += 1
            continue

        scores = np.asarray([p["score"] for p in preds], dtype=np.float32)
        keep_mask = scores >= threshold
        n_keep = int(keep_mask.sum())
        total_pred_kept += n_keep
        gacc["n_pred_kept"] += n_keep

        iou = iou_by_img[img_id]
        iou_keep = iou[keep_mask]  # (n_keep, n_gt), rows already score-sorted

        if n_keep == 0:
            matched = np.zeros(n_gt, dtype=bool)
            covered = np.zeros(n_gt, dtype=bool)
            per_gt_hits = np.zeros(n_gt, dtype=np.int64)
        else:
            matched = _greedy_match(iou_keep, match_iou)
            hit_matrix = iou_keep >= match_iou           # (n_keep, n_gt)
            per_gt_hits = hit_matrix.sum(axis=0)          # preds per GT at IoU>=match
            covered = per_gt_hits > 0

        total_matched += int(matched.sum())
        total_covered += int(covered.sum())
        gacc["matched"] += int(matched.sum())
        gacc["covered"] += int(covered.sum())

        for j, g in enumerate(gts):
            sb = _size_bin(float(g.get("area", 0.0)))
            size_acc[sb]["n_gt"] += 1
            if matched[j]:
                size_acc[sb]["matched"] += 1
            if covered[j]:
                size_acc[sb]["covered"] += 1
                dup_counts.append(int(per_gt_hits[j]))

    def _recall(num: int, den: int) -> float:
        return (num / den) if den > 0 else 0.0

    preds_arr = np.asarray(preds_per_image, dtype=np.int64) if preds_per_image else np.zeros(1, np.int64)
    dup_arr = np.asarray(dup_counts, dtype=np.int64) if dup_counts else np.zeros(1, np.int64)

    result: dict[str, Any] = {
        "threshold": threshold,
        "match_iou": match_iou,
        "budget": budget,
        "n_images": n_images,
        "total_gt": total_gt,
        "total_pred_kept": total_pred_kept,
        "preds_per_image": {
            "min": int(preds_arr.min()),
            "mean": float(preds_arr.mean()),
            "max": int(preds_arr.max()),
            "saturated_at_budget_frac": float(np.mean(preds_arr >= budget)),
        },
        "overall": {
            "matched_recall": _recall(total_matched, total_gt),
            "coverage_recall": _recall(total_covered, total_gt),
            "matched_gt": total_matched,
            "covered_gt": total_covered,
            "redundancy_gap": _recall(total_covered - total_matched, total_gt),
        },
        "by_gt_count_bin": {
            b: {
                "n_images": v["n_images"],
                "n_gt": v["n_gt"],
                "matched_recall": _recall(v["matched"], v["n_gt"]),
                "coverage_recall": _recall(v["covered"], v["n_gt"]),
                "mean_pred_kept_per_image": (v["n_pred_kept"] / v["n_images"]) if v["n_images"] else 0.0,
            }
            for b, v in sorted(gt_count_acc.items())
        },
        "by_gt_size": {
            sb: {
                "n_gt": v["n_gt"],
                "matched_recall": _recall(v["matched"], v["n_gt"]),
                "coverage_recall": _recall(v["covered"], v["n_gt"]),
            }
            for sb, v in size_acc.items()
        },
        "duplication": {
            "covered_gt": int(dup_arr.size if dup_counts else 0),
            "mean_preds_per_covered_gt": float(dup_arr.mean()) if dup_counts else 0.0,
            "median_preds_per_covered_gt": float(np.median(dup_arr)) if dup_counts else 0.0,
            "max_preds_per_covered_gt": int(dup_arr.max()) if dup_counts else 0,
        },
    }
    return result


def _print_report(r: dict[str, Any]) -> None:
    print("\n================ RECALL-CEILING ATTRIBUTION ================")
    print(f"images={r['n_images']}  total_gt={r['total_gt']}  "
          f"threshold={r['threshold']}  match_iou={r['match_iou']}  budget={r['budget']}")
    ppi = r["preds_per_image"]
    print(f"\npreds/image: min={ppi['min']} mean={ppi['mean']:.1f} max={ppi['max']} "
          f"saturated@budget={ppi['saturated_at_budget_frac']*100:.1f}%")

    ov = r["overall"]
    print("\n-- Overall (at threshold -> {:.3g}) --".format(r["threshold"]))
    print(f"  matched recall (1-to-1)   = {ov['matched_recall']:.4f}  ({ov['matched_gt']} GT)")
    print(f"  coverage recall (any pred) = {ov['coverage_recall']:.4f}  ({ov['covered_gt']} GT)")
    print(f"  redundancy gap            = {ov['redundancy_gap']:.4f}  "
          f"(coverage - matched; budget wasted on duplicates)")
    print(f"  STRUCTURAL MISS           = {1.0 - ov['coverage_recall']:.4f}  "
          f"(GT never localized by ANY query -> assignment failure)")

    print("\n-- By image GT-count bin (query-cap isolation) --")
    print(f"  {'bin':<26} {'imgs':>5} {'n_gt':>7} {'matchR':>8} {'covR':>8} {'mPred':>7}")
    for b, v in r["by_gt_count_bin"].items():
        print(f"  {b:<26} {v['n_images']:>5} {v['n_gt']:>7} "
              f"{v['matched_recall']:>8.4f} {v['coverage_recall']:>8.4f} "
              f"{v['mean_pred_kept_per_image']:>7.1f}")

    print("\n-- By GT size (COCO small/medium/large) --")
    print(f"  {'size':<8} {'n_gt':>7} {'matchR':>8} {'covR':>8}")
    for sb, v in r["by_gt_size"].items():
        print(f"  {sb:<8} {v['n_gt']:>7} {v['matched_recall']:>8.4f} {v['coverage_recall']:>8.4f}")

    d = r["duplication"]
    print("\n-- Duplication on covered GT --")
    print(f"  mean preds/covered GT   = {d['mean_preds_per_covered_gt']:.2f}")
    print(f"  median preds/covered GT = {d['median_preds_per_covered_gt']:.1f}")
    print(f"  max preds/covered GT    = {d['max_preds_per_covered_gt']}")
    print("\nInterpretation:")
    print("  * coverage_recall is the TRUE structural ceiling (threshold/duplication removed).")
    print("  * If under_budget images still show low coverage_recall, the 200-query cap is")
    print("    exonerated and the loss is assignment/mislocalization.")
    print("  * High redundancy_gap + high duplication => queries cluster on easy crowns")
    print("    instead of covering distinct (esp. small/grouped) crowns.")
    print("===========================================================\n")


def run_recall_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    """Runs the diagnostic against already-parsed args (or an equivalent
    argparse.Namespace built programmatically, e.g. from a notebook) and
    writes the result JSON to args.out. Returns the saved result dict."""
    gt_path = Path(args.gt)
    pred_path = Path(args.pred)
    out_path = Path(args.out)
    if not gt_path.exists():
        print(f"Error: GT file not found: {gt_path}")
        sys.exit(1)
    if not pred_path.exists():
        print(f"Error: predictions file not found: {pred_path}")
        sys.exit(1)

    print(f"Loading GT from: {gt_path}")
    coco_gt = _load_coco_gt(gt_path)
    img_to_gts, gt_rles_by_img = _build_image_index(coco_gt)

    print(f"Loading predictions from: {pred_path}")
    with pred_path.open("r") as f:
        pred_data = json.load(f)
    sorted_preds_by_img = _group_preds_by_image(pred_data)

    print(f"Precomputing per-image segm IoU matrices "
          f"(images={len(img_to_gts)}, preds={len(pred_data)})...")
    iou_by_img = _precompute_ious(img_to_gts, sorted_preds_by_img, gt_rles_by_img)

    iou_list = sorted({float(x) for x in args.iou.split(",") if x.strip()})
    if not iou_list:
        print("Error: no valid --iou values parsed.")
        sys.exit(1)
    primary_iou = iou_list[-1]

    by_match_iou: dict[str, Any] = {}
    primary_result: dict[str, Any] = {}
    for match_iou in iou_list:
        res = diagnose(
            img_to_gts=img_to_gts,
            sorted_preds_by_img=sorted_preds_by_img,
            iou_by_img=iou_by_img,
            match_iou=match_iou,
            threshold=args.threshold,
            budget=args.budget,
        )
        by_match_iou[f"{match_iou:g}"] = res
        if match_iou == primary_iou:
            primary_result = res
        _print_report(res)

    # Localization-vs-proposal split: coverage at the lowest vs primary IoU.
    lo_iou = iou_list[0]
    lo = by_match_iou[f"{lo_iou:g}"]["overall"]["coverage_recall"]
    hi = by_match_iou[f"{primary_iou:g}"]["overall"]["coverage_recall"]
    localization_split = {
        "low_iou": lo_iou,
        "primary_iou": primary_iou,
        "coverage_recall_low_iou": lo,
        "coverage_recall_primary_iou": hi,
        "localization_loss": lo - hi,
        "non_proposal_floor": 1.0 - lo,
    }
    print("\n================ LOCALIZATION vs NON-PROPOSAL ================")
    print(f"  coverage @ IoU {lo_iou:g} = {lo:.4f}   (a prediction exists near the GT)")
    print(f"  coverage @ IoU {primary_iou:g} = {hi:.4f}   (prediction is precise enough)")
    print(f"  localization loss   = {lo - hi:.4f}   (exists but imprecise -> resolution/boundary)")
    print(f"  non-proposal floor  = {1.0 - lo:.4f}   (never proposed at all -> budget/encoder)")
    print("=============================================================\n")

    result = dict(primary_result)
    result["by_match_iou"] = by_match_iou
    result["localization_split"] = localization_split
    result["gt_path"] = str(gt_path)
    result["pred_path"] = str(pred_path)
    result["num_predictions"] = len(pred_data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, json.dumps(result, indent=2))
    print(f"Saved diagnostic: {out_path}")
    return result


def main() -> None:
    """Command-line entry point: evaluate dumped predictions with COCO AP, Boundary AP bands, cgF1, an F1 sweep, and recall diagnostics."""
    ap = argparse.ArgumentParser(description="Standalone evaluation of dumped predictions.")
    ap.add_argument("--pred", required=True, type=str, help="COCO-format predictions JSON.")
    ap.add_argument("--gt", required=True, type=str, help="COCO ground-truth annotations JSON.")
    ap.add_argument("--out", required=True, type=str, help="Output JSON path.")
    ap.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold for F1 sweep.")
    ap.add_argument("--max-dets", type=int, default=512, help="COCOeval maxDets cap (default 512; matches OAM-TCD paper).")
    ap.add_argument(
        "--boundary-fine-band-px", type=float, default=BOUNDARY_AP_FINE_BAND_PX,
        help=(
            "Boundary AP band in SOURCE pixels for the reported fine measurement. "
            "The upstream 0.02 ratio is 57px on a 2048px tile, wider than most "
            "crowns, which silently reduces Boundary IoU to Mask IoU."
        ),
    )
    ap.add_argument(
        "--boundary-band-sweep", action="store_true",
        help=(
            "Also sweep Boundary AP across band widths to locate where it "
            "saturates into Mask AP. Intended to be run once on the baseline to "
            "justify the fine band, not per arm; it costs one extra COCOeval per "
            "band."
        ),
    )
    ap.add_argument(
        "--band-k", type=float, default=None,
        help="Also compute Boundary AP with an INSTANCE-RELATIVE band of k * "
             "equivalent_diameter (Core/boundary_band.py). Pass 0 to skip. Omit to use the "
             "pre-registered BOUNDARY_BAND_K. Reported as `boundary_ap_instance` ALONGSIDE the "
             "literature and fixed-px bands, never replacing them -- dropping the literature "
             "band would silently break comparison with published Boundary IoU numbers.",
    )
    ap.add_argument("--protocol", choices=("none", "oam_tcd_test"), default="none", help="Optional annotation protocol validator to record in the result.")
    ap.add_argument(
        "--class-split",
        choices=("none", "tree", "canopy"),
        default="none",
        help=(
            "Score only one OAM-TCD class by marking the other iscrowd=1 (COCO ignore "
            "semantics). 'tree' is the primary reporting protocol. Only Mask AP and "
            "Boundary AP honour ignore regions, so the union B-IoU, cgF1 and F1 sweep "
            "are skipped when this is not 'none'."
        ),
    )
    args = ap.parse_args()

    evaluation_pred_path, valid_region_info = _prepare_valid_region_predictions(args.pred, args.gt, args.out)
    if valid_region_info.get("enabled"):
        print(
            f"Valid-region filtering: {valid_region_info['input_predictions']} -> "
            f"{valid_region_info['kept_predictions']} predictions"
        )

    evaluation_gt_path, class_split_info = _prepare_class_split_gt(args.gt, args.class_split, args.out)
    if class_split_info.get("enabled"):
        print(
            f"Class split '{args.class_split}': scoring "
            f"{class_split_info['scored_annotations']} annotations, ignoring "
            f"{class_split_info['ignored_annotations']} via iscrowd=1"
        )

    results: dict[str, dict] = {
        "evaluation_protocol": {
            "maxDets": args.max_dets,
            "evaluation_max_dets": args.max_dets,
            "inference_maxdets": 0,
            "class_split": args.class_split,
        },
        "valid_region_filter": valid_region_info,
        "class_split": class_split_info,
    }

    # Validated against the caller's GT, not the derived split file: the protocol
    # check exists to prove the export is the canonical one, and the split is a
    # derived artifact of this run.
    if args.protocol == "oam_tcd_test":
        from oam_tcd_protocol import validate_oam_tcd_test
        results["dataset_protocol"] = validate_oam_tcd_test(args.gt)

    print("=== COCO segm evaluation (Mask AP) ===")
    results["coco_eval"] = run_coco_eval(evaluation_pred_path, evaluation_gt_path, max_dets=args.max_dets)

    # Two bands, deliberately. The literature ratio keeps the row comparable to
    # published Boundary AP, but on 2048px tiles it is a 57px band that exceeds
    # most crowns, so Boundary IoU degenerates into Mask IoU and the number
    # carries almost no boundary information. The fine band is the reported
    # boundary-fidelity metric. See evaluation_protocol.md section 1.3.
    print(f"\n=== Boundary AP, literature band "
          f"(ratio {BOUNDARY_AP_LITERATURE_RATIO}) ===")
    results["boundary_ap"] = run_boundary_ap(
        evaluation_pred_path, evaluation_gt_path, max_dets=args.max_dets,
        dilation_ratio=BOUNDARY_AP_LITERATURE_RATIO,
    )

    diagonal, image_size = dataset_image_diagonal(evaluation_gt_path)
    fine_ratio = boundary_band_to_ratio(args.boundary_fine_band_px, diagonal)
    print(f"\n=== Boundary AP, fine band ({args.boundary_fine_band_px:g}px, "
          f"ratio {fine_ratio:.6f}) ===")
    results["boundary_ap_fine"] = run_boundary_ap(
        evaluation_pred_path, evaluation_gt_path, max_dets=args.max_dets,
        dilation_ratio=fine_ratio,
    )
    results["boundary_ap_fine"]["band_px"] = float(args.boundary_fine_band_px)
    results["boundary_ap_fine"]["image_size"] = list(image_size)

    # Instance-relative band width tracks each crown rather than the image diagonal. The
    # uniform 8-pixel comparison is close to Mask AP for small crowns and floors for large
    # crowns on this dataset. Both image-scale comparison bands remain in the output.
    if args.band_k is None or args.band_k > 0:
        from boundary_band import BOUNDARY_BAND_K

        band_k = BOUNDARY_BAND_K if args.band_k is None else float(args.band_k)
        print(f"\n=== Boundary AP, instance-relative band (k = {band_k}) ===")
        results["boundary_ap_instance"] = run_boundary_ap(
            evaluation_pred_path, evaluation_gt_path, max_dets=args.max_dets,
            band_mode="instance_relative", band_k=band_k,
        )
        results["boundary_ap_instance"]["image_size"] = list(image_size)

    if args.boundary_band_sweep:
        print("\n=== Boundary AP band saturation sweep ===")
        results["boundary_ap_sweep"] = run_boundary_ap_sweep(
            evaluation_pred_path, evaluation_gt_path,
            mask_ap=results["coco_eval"]["AP"], max_dets=args.max_dets,
        )

    # These three cannot express an ignore region: BoundaryIoUEvaluator merges all
    # instances into a semantic union and calls getAnnIds unfiltered, and the cgF1 /
    # F1-sweep matchers have no iscrowd branch. Running them on a split GT would
    # silently report collapsed-protocol numbers under a tree-only label.
    skip_reason = (
        f"skipped under --class-split {args.class_split}: this metric has no iscrowd "
        "ignore semantics, so it would report the collapsed population under a "
        "class-split label"
    )
    if args.class_split == "none":
        print("\n=== Boundary IoU (semantic union; internal diagnostic) ===")
        results["boundary_iou"] = run_boundary_iou(evaluation_pred_path, evaluation_gt_path)

        print("\n=== cgF1 (official SAM3 metric) ===")
        results["cgf1"] = run_cgf1(evaluation_pred_path, evaluation_gt_path)

        print("\n=== F1 sweep (greedy IoU>=0.5) ===")
        results["f1_sweep"] = run_f1_sweep(evaluation_pred_path, evaluation_gt_path, args.match_iou)
    else:
        for key in ("boundary_iou", "cgf1", "f1_sweep"):
            results[key] = {"skipped": True, "reason": skip_reason}
        print(f"\nSkipped boundary_iou / cgf1 / f1_sweep: {skip_reason}")

    results["boundary_gate"] = boundary_gate_quantities(
        results["coco_eval"], results.get("boundary_ap_fine", {})
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, json.dumps(results, indent=2))

    print("\n=== Summary ===")
    ce = results["coco_eval"]
    bap = results["boundary_ap"]
    ar_key = next(k for k in ce if k.startswith("AR@") and k not in ("AR@1", "AR@10"))
    print(f"protocol  = {args.class_split}")
    print(f"AP50      = {ce['AP50']:.4f}   <- primary, external comparison")
    bap_fine = results["boundary_ap_fine"]
    if bap_fine.get("available"):
        print(f"Bnd AP    = {bap_fine['Boundary_AP']:.4f}   <- primary, boundary "
              f"fidelity ({bap_fine['band_px']:g}px band)")
        print(f"Bnd AP50  = {bap_fine['Boundary_AP50']:.4f}   <- reported headline, "
              f"OAM-TCD's AP50 currency")
        print(f"Bnd AP75  = {bap_fine['Boundary_AP75']:.4f}   (diagnostic; near the floor)")
        print(f"Bnd AP_lg = {bap_fine['Boundary_AP_large']:.4f}")
        gate = results["boundary_gate"]
        print(f"Bnd/Mask  = {gate['ratio_ap']:.4f}   <- binding direction check: "
              f"must RISE vs the reference")
        print(f"            profile across IoU: AP50 {gate['ratio_ap50']:.4f} -> "
              f"AP {gate['ratio_ap']:.4f} -> AP75 {gate['ratio_ap75']:.4f}")
    else:
        print(f"Bnd AP    = UNAVAILABLE ({bap_fine.get('reason')})")
    if bap.get("available"):
        collapse = bap["Boundary_AP"] / ce["AP"] if ce["AP"] > 0 else float("nan")
        print(f"Bnd AP@lit= {bap['Boundary_AP']:.4f}   <- literature ratio "
              f"{BOUNDARY_AP_LITERATURE_RATIO}, {collapse:.3f}x Mask AP")
        if collapse > 0.95:
            print("            NOTE: within 5% of Mask AP -- at this band the crowns "
                  "are smaller than the boundary rind, so Boundary IoU has "
                  "degenerated into Mask IoU. Do not read it as boundary quality.")
    print(f"AP        = {ce['AP']:.4f}")
    print(f"AP75      = {ce['AP75']:.4f}")
    print(f"AP_small  = {ce['AP_small']:.4f}")
    print(f"{ar_key}  = {ce[ar_key]:.4f}")
    if args.class_split == "none":
        bi = results["boundary_iou"]
        cg = results["cgf1"]
        f1 = results["f1_sweep"]
        print(f"[diagnostic, not reported] semantic union B-IoU = "
              f"{bi.get('coco_eval_segm_boundary_iou', 0.0):.6f}")
        print(f"cgF1@0.50 = {cg.get('cgF1_eval_segm_cgF1@0.5', 0.0):.4f}  "
              f"F1@0.50 = {cg.get('cgF1_eval_segm_F1@0.5', 0.0):.4f}")
        print(f"Best F1 = {f1['best_f1']:.4f} @ thr={f1['threshold']:.3f} "
              f"(P={f1['precision']:.4f}, R={f1['recall']:.4f})")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
