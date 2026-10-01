"""Empirical resolution of OAM-TCD's reported 43.22 holdout mAP50.

Why this exists:
The OAM-TCD paper (Table 4, Section 'Instance segmentation', p. 7) reports:
  "mAP50=41.79+-1.38 [cross-validation], and mAP50=43.22 on the holdout set
   using the COCO API segm task."
The paper releases the model as `restor/tcd-mask-rcnn-r50` on HuggingFace Hub,
which is a 2-class Mask R-CNN (tree=2, canopy=1). The paper does not state
whether 43.22 was computed over both categories, tree-only, or collapsed.

This script:
  1. Downloads `restor/tcd-mask-rcnn-r50` (`config.yaml` + `model.pth`) via
     HuggingFace Hub if not already present.
  2. Runs whole-image inference (the paper's test protocol: "we do not perform
     tiled inference when testing on the holdout set", maxdets=512).
  3. Evaluates:
     - `none`   : collapsed (all annotations scored as 1 class)
     - `tree`   : tree-only (25,671 annotations scored, canopy ignored via iscrowd=1)
     - `canopy` : canopy-only (4,975 annotations scored, tree ignored via iscrowd=1)
  4. Prints the comparison table against the paper's 43.22 to resolve the
     exact protocol used by the original authors.

Usage:
    python Core/eval_official_maskrcnn.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent
CODEBASE_DIR = CORE_DIR.parent
PY = sys.executable

ANN = CODEBASE_DIR / "runtime" / "data" / "oam_tcd_instance_coco" / "annotations" / "test_annotations.coco.json"
IMAGE_ROOT = CODEBASE_DIR / "runtime" / "data" / "oam_tcd_instance_coco"
CHECKPOINT_DIR = CODEBASE_DIR / "runtime" / "checkpoints" / "maskrcnn_r50"
OUT_DIR = CODEBASE_DIR / "runtime" / "inference_output" / "oam_tcd_official_maskrcnn_r50_whole-image"
PRED_PATH = OUT_DIR / "predictions" / "coco_predictions_segm_nms.json"

PAPER_REPORTED_AP50 = 43.22  # holdout set mAP50 percentage (0.4322)


def ensure_official_checkpoint(checkpoint_dir: Path = CHECKPOINT_DIR) -> tuple[Path, Path]:
    """Download the released `restor/tcd-mask-rcnn-r50` configuration and weights unless they are already present."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / "config.yaml"
    model_path = checkpoint_dir / "model.pth"

    if config_path.is_file() and model_path.is_file():
        print(f"Official checkpoint ready at: {checkpoint_dir}")
        return config_path, model_path

    print(f"Downloading official Mask R-CNN from HuggingFace Hub: restor/tcd-mask-rcnn-r50 ...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("huggingface_hub is required to download the official checkpoint. Install via pip.")

    hf_hub_download(repo_id="restor/tcd-mask-rcnn-r50", filename="config.yaml", local_dir=str(checkpoint_dir))
    hf_hub_download(repo_id="restor/tcd-mask-rcnn-r50", filename="model.pth", local_dir=str(checkpoint_dir))
    print(f"Downloaded official checkpoint to: {checkpoint_dir}")
    return config_path, model_path


def run_inference(config_path: Path, model_path: Path, *, force: bool = False) -> Path:
    """Run the released model over the test images and return the prediction path; an existing dump is reused unless `force` is set."""
    if PRED_PATH.is_file() and not force:
        print(f"Inference dump exists, skipping: {PRED_PATH}")
        return PRED_PATH

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, str(CORE_DIR / "tiled_inference.py"),
        "--backend", "detectron2-maskrcnn",
        "--detectron-config", str(config_path),
        "--detectron-weights", str(model_path),
        "--device", "cuda",
        "--ann", str(ANN),
        "--image-root", str(IMAGE_ROOT),
        "--run-dir", str(OUT_DIR),
        "--out", str(PRED_PATH),
        "--tile-policy", "whole-image",
        "--resolution", "1024",
        "--merge-method", "mask-nms",
        "--dedup-iou", "0.30",
        "--maxdets", "512",
        "--score-thresh", "0.0",
        "--min-area", "4",
    ]
    print("\nRunning official Mask R-CNN whole-image inference:")
    print(" ".join(cmd))
    start = time.time()
    subprocess.run(cmd, cwd=CODEBASE_DIR, check=True)
    print(f"Inference completed in {(time.time() - start) / 60:.1f} min -> {PRED_PATH}")
    return PRED_PATH


def evaluate_split(pred_path: Path, split: str) -> dict:
    """Evaluate the predictions for one class-split mode with `eval_predictions.py` and return its metrics."""
    out_path = OUT_DIR / ("eval.json" if split == "none" else f"eval_{split}.json")
    cmd = [
        PY, str(CORE_DIR / "eval_predictions.py"),
        "--pred", str(pred_path),
        "--gt", str(ANN),
        "--out", str(out_path),
        "--max-dets", "512",
        "--protocol", "oam_tcd_test",
        "--boundary-fine-band-px", "8",
        "--class-split", split,
    ]
    print(f"\nEvaluating split '{split}':")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=CODEBASE_DIR, check=True)
    return json.loads(out_path.read_text())


def main() -> None:
    """Evaluate the released Mask R-CNN on the test set and print its metrics beside the 43.22 AP50 reported for OAM-TCD."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force-inference", action="store_true", help="Redo inference even if predictions exist.")
    args = parser.parse_args()

    config_path, model_path = ensure_official_checkpoint()
    pred_path = run_inference(config_path, model_path, force=args.force_inference)

    results = {}
    for split in ("none", "tree", "canopy"):
        results[split] = evaluate_split(pred_path, split)

    print("\n" + "=" * 80)
    print("EMPIRICAL COMPARISON: Official restor/tcd-mask-rcnn-r50 vs Paper 43.22")
    print("=" * 80)
    print(f"{'Split':12s} {'Mask AP50':10s} {'Mask AP':10s} {'Bnd AP@8':10s} {'Bnd AP50':10s} {'Delta vs 43.22':15s}")
    print("-" * 80)

    for split, label in [("none", "all (collapsed)"), ("tree", "tree-only"), ("canopy", "canopy-only")]:
        payload = results[split]
        ce = payload.get("coco_eval", {})
        fine = payload.get("boundary_ap_fine", {})
        ap50 = ce.get("AP50", 0.0)
        ap = ce.get("AP", 0.0)
        bnd_ap = fine.get("Boundary_AP", 0.0)
        bnd_ap50 = fine.get("Boundary_AP50", 0.0)
        delta = (ap50 * 100.0) - PAPER_REPORTED_AP50
        print(f"{label:15s} {ap50:.4f}     {ap:.4f}     {bnd_ap:.4f}     {bnd_ap50:.4f}     {delta:+.2f}%")

    print("=" * 80)
    print(f"Paper reported: holdout mAP50 = {PAPER_REPORTED_AP50:.2f}% (0.4322)")
    print("The split whose Mask AP50 is closest to 0.4322 is the one the paper reported.")


if __name__ == "__main__":
    main()
