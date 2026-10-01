"""Backend-neutral instance segmentation inference for OAM-TCD.

Defines the `InstanceSegmentationBackend` protocol with SAM3 and Detectron2 Mask R-CNN
implementations, lifts each window's predicted masks into full-image COCO instances encoded as
run-length encoding (RLE), and merges the overlapping predictions of neighbouring tiles with greedy
non-maximum suppression (NMS) on masks or boxes.
"""

from __future__ import annotations

import io
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from project_paths import add_support_to_pythonpath

add_support_to_pythonpath()

from pycocotools import mask as mask_utils  # noqa: E402


@dataclass(frozen=True)
class BackendPrediction:
    """Masks shaped `(N, H, W)` with one score per mask, as returned by a backend call."""
    masks: np.ndarray
    scores: np.ndarray


class InstanceSegmentationBackend(Protocol):
    """Interface that every inference backend implements."""
    def predict(self, image: np.ndarray, *, resize_to_model: bool) -> BackendPrediction:
        """Return masks and scores for one RGB image."""


def read_rgb_window(
    path: str | Path,
    xoff: int,
    yoff: int,
    width: int,
    height: int,
    output_width: int | None = None,
    output_height: int | None = None,
) -> np.ndarray:
    """Read an RGB window from a raster with `gdal_translate`, optionally resampled to the requested output size."""
    if min(int(xoff), int(yoff), int(width), int(height)) < 0:
        raise ValueError("Raster window offsets and sizes must be non-negative.")
    if (output_width is None) != (output_height is None):
        raise ValueError("output_width and output_height must be provided together.")
    if output_width is not None and min(int(output_width), int(output_height)) <= 0:
        raise ValueError("Output dimensions must be positive.")
    command = [
        "gdal_translate", "-q", "-of", "PNG", "-b", "1", "-b", "2", "-b", "3", "-srcwin",
        str(int(xoff)), str(int(yoff)), str(int(width)), str(int(height)),
    ]
    if output_width is not None:
        command.extend(["-outsize", str(int(output_width)), str(int(output_height))])
    command.extend([str(path), "/vsistdout/"])
    result = subprocess.run(command, check=True, capture_output=True)
    with Image.open(io.BytesIO(result.stdout)) as image:
        return np.asarray(image.convert("RGB"))


def _resize_square(image: np.ndarray, resolution: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(image.astype(np.uint8)).resize(
            (resolution, resolution), Image.Resampling.BILINEAR
        )
    )


class SAM3Backend:
    """Backend that prompts a fine-tuned SAM3 model with a text prompt."""
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        score_threshold: float,
        resolution: int,
        prompt: str,
    ) -> None:
        from visualization import load_finetuned_processor

        # Locked inference uses tiling without test-time augmentation. Averaging augmented
        # probability fields can alter the contours measured by the boundary metrics.
        self.resolution = resolution
        self.processor = load_finetuned_processor(
            checkpoint_path=checkpoint_path,
            confidence_threshold=score_threshold,
            resolution=resolution,
        )
        self.prompt = prompt

    def predict(self, image: np.ndarray, *, resize_to_model: bool) -> BackendPrediction:
        """Predict masks for one image, resized to the model resolution when `resize_to_model` is set."""
        from visualization import predict_image

        model_image = _resize_square(image, self.resolution) if resize_to_model else image
        prediction = predict_image(self.processor, model_image, prompt=self.prompt)
        return BackendPrediction(
            masks=np.asarray(prediction.masks, dtype=bool),
            scores=np.asarray(prediction.scores, dtype=float),
        )


class Detectron2MaskRCNNBackend:
    """Backend that wraps a Detectron2 Mask R-CNN predictor."""
    def __init__(self, predictor: Any, *, resolution: int) -> None:
        self.predictor = predictor
        self.resolution = resolution

    @classmethod
    def from_config(
        cls,
        *,
        config_path: str | Path,
        weights_path: str | Path,
        score_threshold: float,
        resolution: int,
        device: str,
    ) -> "Detectron2MaskRCNNBackend":
        """Build the predictor from a Detectron2 configuration file and a weights file."""
        from config import MASKRCNN_MAX_SIZE_TEST, MASKRCNN_MIN_SIZE_TEST

        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
        except ImportError as exc:
            raise RuntimeError(
                "Detectron2 is required for --backend detectron2-maskrcnn. "
                "Install the project's pinned Detectron2 environment before using this backend."
            ) from exc
        # Single scale, no multi-scale TTA: see the note in SAM3Backend.
        cfg = get_cfg()
        cfg.INPUT.MIN_SIZE_TRAIN = [800]
        cfg.merge_from_file(str(config_path))
        cfg.MODEL.WEIGHTS = str(weights_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_threshold)
        cfg.MODEL.DEVICE = device
        cfg.INPUT.MIN_SIZE_TEST = MASKRCNN_MIN_SIZE_TEST
        cfg.INPUT.MAX_SIZE_TEST = MASKRCNN_MAX_SIZE_TEST
        return cls(DefaultPredictor(cfg), resolution=resolution)

    def predict(self, image: np.ndarray, *, resize_to_model: bool) -> BackendPrediction:
        """Predict masks for one RGB image (converted to BGR for Detectron2)."""
        output = self.predictor(image[:, :, ::-1])
        instances = output["instances"].to("cpu")
        return BackendPrediction(
            masks=instances.pred_masks.numpy().astype(bool),
            scores=instances.scores.numpy().astype(float),
        )


def build_backend(args: Any) -> InstanceSegmentationBackend:
    """Build the backend named by `args.backend` (`sam3` by default, or `detectron2-maskrcnn`)."""
    backend_name = getattr(args, "backend", "") or "sam3"
    if backend_name == "sam3":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --backend sam3")
        return SAM3Backend(
            checkpoint_path=args.checkpoint,
            score_threshold=args.score_thresh,
            resolution=args.resolution,
            prompt=args.prompt,
        )
    if backend_name == "detectron2-maskrcnn":
        if not args.detectron_config or not args.detectron_weights:
            raise ValueError(
                "--detectron-config and --detectron-weights are required for "
                "--backend detectron2-maskrcnn"
            )
        return Detectron2MaskRCNNBackend.from_config(
            config_path=args.detectron_config,
            weights_path=args.detectron_weights,
            score_threshold=args.score_thresh,
            resolution=args.resolution,
            device=args.device,
        )
    raise ValueError(f"Unknown backend: {backend_name!r}")


def _standardize_rle(rle: dict[str, Any]) -> dict[str, Any]:
    out = {"size": list(rle["size"]), "counts": rle["counts"]}
    if isinstance(out["counts"], bytes):
        out["counts"] = out["counts"].decode("ascii")
    return out


def _mask_nms(masks: np.ndarray, scores: np.ndarray, *, iou_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if len(scores) == 0:
        return masks, scores
    rles = [mask_utils.encode(np.asfortranarray(m)) for m in masks]
    order = scores.argsort()[::-1]
    keep: list[int] = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(int(i))
        for j in order:
            if j == i or j in suppressed:
                continue
            iou = mask_utils.iou([rles[i]], [rles[j]], [0])[0]
            if iou > iou_threshold:
                suppressed.add(int(j))
    keep_arr = np.array(keep, dtype=int)
    return masks[keep_arr], scores[keep_arr]


def _bbox_grid_keys(
    bbox: list[float], *, cell_size: int = 1024, max_cells: int = 256
) -> list[tuple[int, int]] | None:
    x0, y0, width, height = [float(value) for value in bbox]
    x1 = max(x0, x0 + width)
    y1 = max(y0, y0 + height)
    ix0 = int(np.floor(x0 / cell_size))
    iy0 = int(np.floor(y0 / cell_size))
    ix1 = int(np.floor(max(x0, x1 - 1e-6) / cell_size))
    iy1 = int(np.floor(max(y0, y1 - 1e-6) / cell_size))
    if (ix1 - ix0 + 1) * (iy1 - iy0 + 1) > max_cells:
        return None
    return [(ix, iy) for ix in range(ix0, ix1 + 1) for iy in range(iy0, iy1 + 1)]


def _embed_mask_rle(mask: np.ndarray, x0: int, y0: int, height: int, width: int) -> dict[str, Any]:
    mask_height, mask_width = mask.shape
    if x0 < 0 or y0 < 0 or x0 + mask_width > width or y0 + mask_height > height:
        raise ValueError(f"Window mask {(mask_height, mask_width)} at {(x0, y0)} exceeds image {(height, width)}")
    counts: list[int] = []
    run_value = 0
    run_length = 0

    def append_run(value: int, length: int) -> None:
        nonlocal run_value, run_length
        if length <= 0:
            return
        if value == run_value:
            run_length += length
        else:
            counts.append(run_length)
            run_value = value
            run_length = length

    append_run(0, x0 * height)
    for col in range(mask_width):
        append_run(0, y0)
        column = np.asarray(mask[:, col], dtype=np.uint8)
        boundaries = np.flatnonzero(column[1:] != column[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [mask_height]))
        for start, end in zip(starts, ends):
            append_run(int(column[start]), int(end - start))
        append_run(0, height - y0 - mask_height)
    append_run(0, (width - x0 - mask_width) * height)
    counts.append(run_length)
    rle = mask_utils.frPyObjects({"size": [height, width], "counts": counts}, height, width)
    return _standardize_rle(rle)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST),
        dtype=bool,
    )


def instances_from_window(
    backend: InstanceSegmentationBackend,
    pixels: np.ndarray,
    *,
    source_x0: int,
    source_y0: int,
    source_width: int,
    source_height: int,
    image_height: int,
    image_width: int,
    image_id: int,
    category_id: int,
    window_id: str,
    window_row: int,
    window_col: int,
    min_area: int,
    edge_margin: int,
    resize_to_model: bool,
    is_whole_image: bool,
) -> list[dict[str, Any]]:
    """Turn one window's predicted masks into full-image COCO instances: scale each mask to source pixels, drop masks below `min_area`, embed the RLE at the window offset, and record tile provenance and whether the mask touches the window edge."""
    prediction = backend.predict(pixels, resize_to_model=resize_to_model)
    masks = np.asarray(prediction.masks, dtype=bool)
    scores = np.asarray(prediction.scores, dtype=float)
    if masks.size == 0:
        return []
    if masks.ndim != 3 or masks.shape[0] != scores.shape[0]:
        raise ValueError("Backend must return masks shaped (N,H,W) and one score per mask.")
    mask_height, mask_width = masks.shape[1:]
    scale_x = source_width / mask_width
    scale_y = source_height / mask_height
    out: list[dict[str, Any]] = []
    for prediction_index, (mask, score) in enumerate(zip(masks, scores)):
        ys, xs = np.nonzero(mask)
        if not xs.size:
            continue
        model_x0, model_y0 = int(xs.min()), int(ys.min())
        model_x1, model_y1 = int(xs.max()) + 1, int(ys.max()) + 1
        local_x0 = max(0, int(np.floor(model_x0 * scale_x)))
        local_y0 = max(0, int(np.floor(model_y0 * scale_y)))
        local_x1 = min(source_width, int(np.ceil(model_x1 * scale_x)))
        local_y1 = min(source_height, int(np.ceil(model_y1 * scale_y)))
        if local_x1 <= local_x0 or local_y1 <= local_y0:
            continue
        tight_mask = _resize_mask(
            mask[model_y0:model_y1, model_x0:model_x1],
            local_x1 - local_x0,
            local_y1 - local_y0,
        )
        area = int(tight_mask.sum())
        if area < min_area:
            continue
        local_bbox = [
            float(local_x0),
            float(local_y0),
            float(local_x1 - local_x0),
            float(local_y1 - local_y0),
        ]
        global_x0, global_y0 = source_x0 + local_x0, source_y0 + local_y0
        out.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "score": float(score),
                "segmentation": _embed_mask_rle(tight_mask, global_x0, global_y0, image_height, image_width),
                "bbox": [float(global_x0), float(global_y0), local_bbox[2], local_bbox[3]],
                "area": area,
                "tile_id": window_id,
                "tile_row": int(window_row),
                "tile_col": int(window_col),
                "tile_offset": [int(source_x0), int(source_y0)],
                "tile_bbox": [int(source_x0), int(source_y0), int(source_width), int(source_height)],
                "local_bbox": local_bbox,
                "edge_touch": (
                    local_x0 <= edge_margin
                    or local_y0 <= edge_margin
                    or local_x1 >= source_width - edge_margin
                    or local_y1 >= source_height - edge_margin
                ),
                "is_full_pass": bool(is_whole_image),
                "tile_pred_index": int(prediction_index),
            }
        )
    return out


def greedy_nms(
    instances: list[dict[str, Any]], *, image_id: int, category_id: int, iou_threshold: float,
    maxdets: int = 0, mode: str = "mask",
) -> list[dict[str, Any]]:
    """Greedy non-maximum suppression by descending score on mask or box overlap, keeping at most `maxdets` instances when that is positive."""
    if not instances:
        return []
    ordered = sorted(instances, key=lambda item: float(item["score"]), reverse=True)
    items = [item["segmentation"] if mode == "mask" else item["bbox"] for item in ordered]
    grid: dict[tuple[int, int], list[int]] = {}
    overflow: list[int] = []
    keep: list[int] = []
    for index, instance in enumerate(ordered):
        if maxdets > 0 and len(keep) >= maxdets:
            break
        keys = _bbox_grid_keys(instance["bbox"])
        candidates: set[int] = set(overflow)
        if keys is None:
            for bucket in grid.values():
                candidates.update(bucket)
        else:
            for key in keys:
                candidates.update(grid.get(key, []))
        if candidates:
            candidate_indexes = sorted(candidates)
            ious = np.asarray(
                mask_utils.iou(
                    [items[index]],
                    [items[candidate] for candidate in candidate_indexes],
                    [0] * len(candidate_indexes),
                ),
                dtype=np.float32,
            ).reshape(-1)
            if np.any(ious >= iou_threshold):
                continue
        keep.append(index)
        if keys is None:
            overflow.append(index)
        else:
            for key in keys:
                grid.setdefault(key, []).append(index)
    return [
        {
            "image_id": int(image_id),
            "category_id": int(category_id),
            "score": float(ordered[index]["score"]),
            "segmentation": _standardize_rle(ordered[index]["segmentation"]),
            "bbox": [float(value) for value in ordered[index]["bbox"]],
        }
        for index in keep
    ]


def merge_instances(instances: list[dict[str, Any]], *, image_id: int, category_id: int, dedup_iou: float,
                    maxdets: int, merge_method: str) -> list[dict[str, Any]]:
    """Merge one image's window instances with mask or box non-maximum suppression, as selected by `merge_method`."""
    mode = "box" if merge_method == "box-nms" else "mask"
    return greedy_nms(
        instances,
        image_id=image_id,
        category_id=category_id,
        iou_threshold=dedup_iou,
        maxdets=maxdets,
        mode=mode,
    )
