#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Whole-image and fixed tiled inference for the active OAM-TCD baseline.

The selected backend predicts masks for either one complete source extent or
fixed sliding windows. `inference_engine.py` maps each mask to source
coordinates, serializes COCO RLE, and performs backend-neutral cross-window
NMS. The standalone `whole-image` policy performs exactly one source pass and
is never merged with tiled predictions. The fixed policy preserves the
canonical OAM-TCD 1024-pixel tile and 256-pixel overlap control.

SAM3 and optional Detectron2 Mask R-CNN backends use their own model-specific
preparation and prediction paths; training behavior is not changed.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import inference_engine
from atomic_io import atomic_write_text


def _image_index(coco: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(img["id"]): img for img in coco["images"]}


def _raw_instances_from_file(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data.get("instances", [])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported raw dump structure in {path}")


def _tile_starts(dim: int, tile: int, overlap: int) -> list[int]:
    if tile >= dim:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, dim - tile + 1, step))
    if starts[-1] != dim - tile:
        starts.append(dim - tile)
    return starts


def _merge_for_image(
    instances: list[dict[str, Any]],
    *,
    image_id: int,
    category_id: int,
    H: int,
    W: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    return inference_engine.merge_instances(
        instances,
        image_id=image_id,
        category_id=category_id,
        dedup_iou=args.dedup_iou,
        maxdets=args.maxdets,
        merge_method=args.merge_method,
    )


def _raw_out_path(args: argparse.Namespace, out_path: Path) -> Path | None:
    if args.raw_out:
        return Path(args.raw_out)
    if args.dump_raw:
        return out_path.with_name(out_path.stem + "_raw_instances.json")
    return None


def run(args: argparse.Namespace) -> None:
    """Run whole-image or fixed tiled inference over a COCO image list, merge each image's window predictions, and write COCO predictions with their provenance."""
    ann_path = Path(args.ann)
    image_root = Path(args.image_root) if args.image_root else None
    run_root = Path(getattr(args, "run_dir", "")).resolve() if getattr(args, "run_dir", "") else None
    if run_root is not None:
        run_root.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    elif run_root is not None:
        out_path = run_root / "predictions" / "coco_predictions_segm_nms.json"
    else:
        raise ValueError("--out or --run-dir is required")
    if not ann_path.exists():
        print(f"Error: annotation file not found: {ann_path}")
        sys.exit(1)

    with ann_path.open("r") as f:
        coco = json.load(f)
    images = coco["images"]
    image_by_id = _image_index(coco)
    category_id = coco["categories"][0]["id"] if coco.get("categories") else 1
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    results: list[dict[str, Any]] = []
    all_raw_instances: list[dict[str, Any]] = []
    tile_decisions: dict[int, dict[str, Any]] = {}
    raw_out_path = _raw_out_path(args, out_path)
    t_start = time.time()

    if args.raw_in:
        raw_instances = _raw_instances_from_file(Path(args.raw_in))
        allowed_ids = {int(img["id"]) for img in images}
        raw_by_image: dict[int, list[dict[str, Any]]] = {img_id: [] for img_id in allowed_ids}
        for inst in raw_instances:
            img_id = int(inst["image_id"])
            if img_id in allowed_ids:
                raw_by_image.setdefault(img_id, []).append(inst)
        for idx, img_meta in enumerate(images):
            image_id = int(img_meta["id"])
            H, W = int(img_meta["height"]), int(img_meta["width"])
            instances = raw_by_image.get(image_id, [])
            kept = _merge_for_image(
                instances,
                image_id=image_id,
                category_id=category_id,
                H=H,
                W=W,
                args=args,
            )
            results.extend(kept)
            if (idx + 1) % 10 == 0 or idx + 1 == len(images):
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
                print(f"[{idx + 1}/{len(images)}] raw={len(instances)} kept={len(kept)} "
                      f"({rate:.2f} img/s, {len(results)} preds total)")
    else:
        if image_root is None:
            raise ValueError("--image-root is required unless --raw-in is provided")
        backend = inference_engine.build_backend(args)

        for idx, img_meta in enumerate(images):
            image_id = int(img_meta["id"])
            H, W = int(img_meta["height"]), int(img_meta["width"])
            img_path = image_root / img_meta["file_name"]
            image = np.asarray(Image.open(img_path).convert("RGB"))
            if args.tile_policy == "whole-image":
                ys, xs = [0], [0]
                pixels = image
                instances = inference_engine.instances_from_window(
                    backend, pixels,
                    source_x0=0, source_y0=0, source_width=W, source_height=H,
                    image_height=H, image_width=W, image_id=image_id, category_id=category_id,
                    window_id=f"{image_id}:whole-image", window_row=0, window_col=0,
                    min_area=args.min_area, edge_margin=args.edge_margin,
                    resize_to_model=True, is_whole_image=True,
                )
                tile_decisions[image_id] = {
                    "policy": "whole-image",
                    "tile": None,
                    "overlap": None,
                    "resolution": args.resolution,
                    "raster_height_px": H,
                    "raster_width_px": W,
                    "n_windows": 1,
                }
            else:
                tile = min(args.tile, H, W)
                ys = _tile_starts(H, tile, args.overlap)
                xs = _tile_starts(W, tile, args.overlap)
                tile_decisions[image_id] = {
                    "policy": "fixed",
                    "tile": tile,
                    "overlap": args.overlap,
                    "resolution": args.resolution,
                    "raster_height_px": H,
                    "raster_width_px": W,
                }
                instances = []
                for tile_row, y0 in enumerate(ys):
                    for tile_col, x0 in enumerate(xs):
                        width = min(tile, W - x0)
                        height = min(tile, H - y0)
                        crop = image[y0:y0 + height, x0:x0 + width]
                        tile_started = time.time()
                        tile_instances = inference_engine.instances_from_window(
                            backend, crop,
                            source_x0=x0, source_y0=y0, source_width=width, source_height=height,
                            image_height=H, image_width=W, image_id=image_id, category_id=category_id,
                            window_id=f"{image_id}:r{tile_row}:c{tile_col}",
                            window_row=tile_row, window_col=tile_col,
                            min_area=args.min_area, edge_margin=args.edge_margin,
                            resize_to_model=False, is_whole_image=False,
                        )
                        instances.extend(tile_instances)
                        tile_number = tile_row * len(xs) + tile_col + 1
                        print(
                            f"[image {idx + 1}/{len(images)}] tile {tile_number}/{len(ys) * len(xs)} "
                            f"offset=({x0},{y0}) raw={len(tile_instances)} "
                            f"elapsed={time.time() - tile_started:.1f}s",
                            flush=True,
                        )

            if raw_out_path is not None:
                all_raw_instances.extend(instances)

            kept = _merge_for_image(
                instances,
                image_id=image_id,
                category_id=category_id,
                H=H,
                W=W,
                args=args,
            )
            results.extend(kept)

            if (idx + 1) % 10 == 0 or idx + 1 == len(images):
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
                remaining = (len(images) - idx - 1) / rate if rate > 0 else 0.0
                print(f"[{idx + 1}/{len(images)}] tiles={len(ys) * len(xs)} "
                      f"raw={len(instances)} kept={len(kept)} "
                      f"({rate:.2f} img/s, {len(results)} preds total, "
                      f"eta {remaining / 3600:.1f}h)")

            # A single write at the very end means a kill or a crash at image
            # 4000 destroys days of GPU time. Periodic snapshots make a long
            # dump harvestable early, which matters when the dump only needs to
            # be large enough to train on rather than complete.
            if (
                args.checkpoint_every
                and (idx + 1) % args.checkpoint_every == 0
                and idx + 1 < len(images)
            ):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                partial_path = out_path.with_suffix(".partial.json")
                tmp_path = partial_path.with_suffix(".tmp")
                tmp_path.write_text(json.dumps(results))
                tmp_path.replace(partial_path)
                print(f"[checkpoint] {len(results)} preds from {idx + 1} images "
                      f"-> {partial_path}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, json.dumps(results))
    print(f"\nSaved {len(results)} predictions for {len(images)} images -> {out_path}")
    _write_inference_config(args, out_path, run_root, ann_path, image_root, len(images), len(results))
    if raw_out_path is not None and not args.raw_in:
        raw_out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_payload = {
            "metadata": {
                "ann": str(ann_path),
                "image_root": str(image_root) if image_root is not None else None,
                "tile": args.tile,
                "overlap": args.overlap,
                "tile_policy": args.tile_policy,
                "score_thresh": args.score_thresh,
                "resolution": args.resolution,
                "min_area": args.min_area,
                "edge_margin": args.edge_margin,
                "backend": args.backend,
                "image_count": len(images),
            },
            "images": [image_by_id[int(img["id"])] for img in images],
            "instances": all_raw_instances,
        }
        atomic_write_text(raw_out_path, json.dumps(raw_payload))
        print(f"Saved {len(all_raw_instances)} raw pre-dedup instances -> {raw_out_path}")
    if tile_decisions:
        tile_policy_path = out_path.with_name(out_path.stem + "_tile_policy.json")
        tile_policy_payload = {
            "policy": args.tile_policy,
            "tiled_tile_px": args.tile,
            "tiled_overlap_px": args.overlap,
            "resolution": args.resolution,
            "images": [
                {
                    "image_id": image_id,
                    "file_name": image_by_id[image_id].get("file_name"),
                    **decision,
                }
                for image_id, decision in tile_decisions.items()
            ],
        }
        atomic_write_text(tile_policy_path, json.dumps(tile_policy_payload))
        print(f"Saved tile-policy metadata for {len(tile_decisions)} images -> {tile_policy_path}")
    print("Next: run eval_predictions.py recall diagnostics on <this file> and compare "
          "non_proposal_floor / small-crown coverage against the baseline dump.")


def _checkpoint_identity(path: Path) -> dict[str, Any]:
    """Size, mtime and a content hash of the weights actually used.

    The path alone is not enough: `model_final.pth` and `model_best.pth` differ,
    checkpoints get overwritten, and a dump whose provenance is "some checkpoint in
    that directory" cannot be compared to anything. Hashing 64 MB rather than the
    whole file keeps this to well under a second on a network filesystem while
    still distinguishing two checkpoints of the same model.
    """
    if not path.is_file():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as handle:
        while read < 64 * 1024 * 1024:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            read += len(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": stat.st_size,
        "mtime_utc": datetime.datetime.fromtimestamp(stat.st_mtime, datetime.timezone.utc).isoformat(),
        "sha256_first_64mib": digest.hexdigest(),
        "hashed_bytes": read,
    }


def _describe_compute() -> dict[str, Any]:
    try:
        from device_policy import describe_device, resolve_autocast_dtype

        return describe_device(amp_dtype=resolve_autocast_dtype("bfloat16"))
    except Exception as exc:  # never let provenance capture break a finished dump
        return {"error": f"{type(exc).__name__}: {exc}"}


def _write_inference_config(
    args: argparse.Namespace,
    out_path: Path,
    run_root: Path | None,
    ann_path: Path,
    image_root: Path | None,
    n_images: int,
    n_predictions: int,
) -> None:
    """Record checkpoint, compute, data, tiling, merge, and filtering provenance."""
    weights = args.checkpoint if args.backend == "sam3" else args.detectron_weights
    payload = {
        "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "backend": args.backend,
        "weights": _checkpoint_identity(Path(weights)) if weights else None,
        "detectron_config": args.detectron_config or None,
        "prompt": args.prompt if args.backend == "sam3" else None,
        "resolution": args.resolution,
        "tile_policy": args.tile_policy,
        "tile_px": args.tile,
        "overlap_px": args.overlap,
        "merge_method": args.merge_method,
        # Named after the CLI flag (--dedup-iou -> args.dedup_iou) so the record always
        # holds the value that was actually used.
        "dedup_iou": args.dedup_iou,
        "maxdets": args.maxdets,
        "score_thresh": args.score_thresh,
        "min_area": args.min_area,
        "device": getattr(args, "device", None),
        # Which GPU and which autocast dtype actually ran. Two dumps compared to
        # each other must agree on both: the V100 node has no bf16 and falls back
        # to float16, and this project has already been bitten by cross-architecture
        # numerics differing in the fourth decimal. See Core/device_policy.py.
        "compute": _describe_compute(),
        # TTA options are absent from the CLI, so these values are structurally false.
        "test_time_augmentation": {"hflip": False, "multi_scale": False, "removed_from_cli": True},
        "annotations": str(ann_path),
        "image_root": str(image_root) if image_root is not None else None,
        "images": n_images,
        "predictions": n_predictions,
        "predictions_path": str(out_path),
        "raw_in": args.raw_in or None,
        "limit": args.limit or None,
    }
    target_dir = run_root if run_root is not None else out_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    config_path = target_dir / "inference_config.json"
    atomic_write_text(config_path, json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved inference configuration -> {config_path}")


def main() -> None:
    """Command-line entry point for tiled inference."""
    p = argparse.ArgumentParser(description="Tiled SAM3 inference for recall-ceiling testing.")
    p.add_argument("--checkpoint", type=str, default="", help="SAM3 checkpoint .pt; required for --backend sam3 unless --raw-in is used.")
    p.add_argument("--backend", choices=["", "sam3", "detectron2-maskrcnn"], default="", help="Inference backend; defaults to SAM3.")
    p.add_argument("--detectron-config", type=str, default="", help="Detectron2 model config; required for --backend detectron2-maskrcnn.")
    p.add_argument("--detectron-weights", type=str, default="", help="Detectron2 weights; required for --backend detectron2-maskrcnn.")
    p.add_argument("--device", type=str, default="cuda", help="Backend device for Detectron2 (default cuda).")
    p.add_argument("--ann", required=True, type=str, help="COCO test annotations JSON.")
    p.add_argument("--image-root", type=str, default="",
                   help="Root joined with COCO file_name; required unless --raw-in is used.")
    p.add_argument("--out", required=False, type=str, default="", help="Output predictions JSON; defaults inside --run-dir.")
    p.add_argument("--run-dir", type=str, default="", help="Self-contained configuration output directory.")
    p.add_argument("--raw-in", type=str, default="", help="Existing raw pre-dedup instances JSON; skips model inference.")
    p.add_argument("--dump-raw", action="store_true", help="Write pre-dedup per-tile instances next to --out.")
    p.add_argument("--raw-out", type=str, default="", help="Explicit raw pre-dedup instances JSON path.")
    p.add_argument("--merge-method", choices=["mask-nms", "box-nms"], default="mask-nms",
                   help="Cross-tile merge method (default mask-NMS).")
    p.add_argument("--tile", type=int, default=1024, help="Square tile crop size in native px (default 1024 -> 2x2 on 2048).")
    p.add_argument("--overlap", type=int, default=256, help="Overlap between adjacent crops in px (default 256). Used as-is in --tile-policy tiled.")
    p.add_argument("--tile-policy", choices=["whole-image", "tiled"], default="tiled",
                   help="'whole-image' runs one complete source pass; 'tiled' runs overlapping fixed-size crops.")
    p.add_argument("--score-thresh", type=float, default=0.0,
                   help="Confidence threshold; 0.0 measures the recall ceiling (default 0.0).")
    p.add_argument("--dedup-iou", type=float, default=0.5,
                   help="IoU threshold for box/mask NMS controls (default 0.5).")
    p.add_argument("--maxdets", type=int, default=0,
                   help="Cap on kept detections per image after merge (0 = unlimited).")
    p.add_argument("--min-area", type=int, default=4, help="Discard masks smaller than this (px).")
    p.add_argument("--edge-margin", type=int, default=1, help="Tile-border margin for edge-touch flagging in raw dumps.")
    p.add_argument("--prompt", type=str, default="tree", help="Text prompt (default 'tree').")
    p.add_argument("--resolution", type=int, default=1008, help="Model input resolution (default 1008).")
    p.add_argument("--limit", type=int, default=0, help="Process only the first N images (smoke test).")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="Write a harvestable <out>.partial.json every N images (0 = only at the end). "
                        "Use this for long dumps so the run can be stopped early without losing it.")
    args = p.parse_args()
    if not args.backend:
        args.backend = "sam3"
    run(args)


if __name__ == "__main__":
    main()
