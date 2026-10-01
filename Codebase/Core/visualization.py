"""Visualization utilities for fine-tuned SAM3 OAM-TCD predictions.

Loads a fine-tuned SAM3 checkpoint, runs prediction on an image (with an
optional ground-truth COCO overlay), and renders the result with matplotlib.

Typical usage from a notebook:

    from visualization import (
        load_finetuned_processor,
        predict_image,
        visualize_prediction,
        visualize_sample_from_coco,
    )

    processor = load_finetuned_processor(
        checkpoint_path="/.../experiments/<run>/checkpoints/checkpoint.pt",
    )
    fig = visualize_sample_from_coco(
        processor,
        coco_json="/.../runtime/data/oam_tcd_instance_coco/annotations/test_annotations.coco.json",
        image_root="/.../runtime/data/oam_tcd_instance_coco",
        image_id=42,
    )
    fig.savefig("pred_42.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from project_paths import CHECKPOINT_DIR, add_support_to_pythonpath

add_support_to_pythonpath()

from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402


# ── Configuration constants ─────────────────────────────────────────────────
DEFAULT_PROMPT = "tree"
DEFAULT_CONFIDENCE = 0.5
DEFAULT_RESOLUTION = 1008
VISUALIZATION_MAX_IMAGE_SIZE = 2048


@dataclass
class Prediction:
    """Container for SAM3 prediction outputs on a single image."""

    image: np.ndarray            # (H, W, 3) uint8
    masks: np.ndarray            # (N, H, W) bool
    boxes: np.ndarray            # (N, 4) xyxy in original-image coords
    scores: np.ndarray           # (N,) float


def load_finetuned_processor(
    checkpoint_path: str | Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    resolution: int = DEFAULT_RESOLUTION,
    strict: bool = False,
) -> Sam3Processor:
    """Build SAM3 image model and load fine-tuned weights on top.

    Args:
        checkpoint_path: path to a trainer-saved ``.pt`` file (dict with key
            ``"model"``) **or** to a raw state dict ``.pt`` file. The base
            SAM3 weights from ``runtime/checkpoints/sam3/sam3.pt`` are loaded
            first by ``build_sam3_image_model``; the fine-tuned weights are
            then merged in.
        device: torch device to place the model on.
        confidence_threshold: SAM3 prediction score threshold.
        resolution: model input resolution (must match training, 1008).
        strict: if True, fail on any missing/unexpected keys when loading
            the fine-tuned state dict.

    Returns:
        A ready-to-use ``Sam3Processor``.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    base_checkpoint_path = CHECKPOINT_DIR / "sam3" / "sam3.pt"
    if not base_checkpoint_path.exists():
        raise FileNotFoundError(f"Base SAM3 checkpoint not found: {base_checkpoint_path}")
    started = time.perf_counter()
    print(f"[load_finetuned_processor] building base model from {base_checkpoint_path}", flush=True)
    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=str(base_checkpoint_path),
        load_from_HF=False,
    )
    print(
        f"[load_finetuned_processor] base model ready in {time.perf_counter() - started:.1f}s",
        flush=True,
    )

    started = time.perf_counter()
    print(f"[load_finetuned_processor] reading fine-tuned checkpoint {checkpoint_path}", flush=True)
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    print(
        f"[load_finetuned_processor] fine-tuned checkpoint ready in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )

    started = time.perf_counter()
    print("[load_finetuned_processor] applying fine-tuned model state", flush=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(
        f"[load_finetuned_processor] fine-tuned model state applied in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )
    if missing:
        print(f"[load_finetuned_processor] {len(missing)} missing keys "
              f"(showing first 5): {missing[:5]}")
    if unexpected:
        print(f"[load_finetuned_processor] {len(unexpected)} unexpected keys "
              f"(showing first 5): {unexpected[:5]}")

    model.eval()
    processor = Sam3Processor(
        model,
        resolution=resolution,
        device=device,
        confidence_threshold=confidence_threshold,
    )

    print("[load_finetuned_processor] processor ready", flush=True)
    return processor


def predict_image(
    processor: Sam3Processor,
    image: str | Path | Image.Image | np.ndarray,
    prompt: str = DEFAULT_PROMPT,
) -> Prediction:
    """Run a text-prompted prediction on a single image."""
    if isinstance(image, (str, Path)):
        pil = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        pil = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    device_type = "cuda" if "cuda" in str(processor.device) else "cpu"
    enabled = (device_type == "cuda")

    # bf16 was hardcoded here, which silently breaks on any pre-Ampere GPU: the
    # V100 node has no bf16, and a dump produced in a different precision is not
    # comparable to one produced in bf16. device_policy picks the best dtype the
    # device supports and logs any fallback once, and the choice is recorded in
    # each run's inference_config.json. See Core/device_policy.py.
    from device_policy import resolve_autocast_dtype

    amp_dtype = resolve_autocast_dtype("bfloat16", device_type=device_type) if enabled else None
    with torch.autocast(device_type=device_type, dtype=amp_dtype or torch.float32, enabled=enabled and amp_dtype is not None):
        state = processor.set_image(pil)
        out = processor.set_text_prompt(prompt=prompt, state=state)
    masks = _to_numpy(out["masks"]).astype(bool)
    boxes = _to_numpy(out["boxes"]).astype(float)
    scores = _to_numpy(out["scores"]).astype(float)

    if masks.ndim == 4:  # (N, 1, H, W) → (N, H, W)
        masks = masks[:, 0]

    return Prediction(image=np.array(pil), masks=masks, boxes=boxes, scores=scores)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.float()
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ── Plotting ────────────────────────────────────────────────────────────────

def visualize_prediction(
    pred: Prediction,
    gt_masks: Optional[np.ndarray] = None,
    gt_boxes: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    score_threshold: float = DEFAULT_CONFIDENCE,
    max_instances: Optional[int] = None,
    show_scores: bool = False,
    show_boxes: bool = False,
    show_gt_boxes: bool = False,
    figsize: tuple[float, float] = (18, 6),
    gt_cmap: str = "tab20",
    pred_cmap: str = "tab20",
):
    """Render a 3-panel figure: RGB | GT overlay | Prediction overlay.

    The middle GT panel is omitted (collapses to 2 panels) if ``gt_masks`` is None.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    keep = pred.scores >= score_threshold
    masks = pred.masks[keep]
    boxes = pred.boxes[keep]
    scores = pred.scores[keep]
    if max_instances is not None and len(masks) > max_instances:
        order = np.argsort(-scores)[:max_instances]
        masks, boxes, scores = masks[order], boxes[order], scores[order]

    n_panels = 3 if gt_masks is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 2:
        axes = [axes[0], None, axes[1]]

    axes[0].imshow(pred.image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    if gt_masks is not None:
        axes[1].set_facecolor("black")
        axes[1].imshow(pred.image, alpha=0.35)
        _overlay_masks(
            axes[1], gt_masks, alpha=0.78, cmap_name=gt_cmap,
            edge_color=(1.0, 1.0, 1.0), edge_alpha=1.0,
        )
        if gt_boxes is not None and show_gt_boxes:
            for box in gt_boxes:
                x1, y1, x2, y2 = box
                axes[1].add_patch(Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor="lime", linewidth=1.0,
                ))
        axes[1].set_title(f"Ground truth (n={len(gt_masks)})")
        axes[1].axis("off")

    axes[2].imshow(pred.image)
    _overlay_masks(axes[2], masks, alpha=0.5, cmap_name=pred_cmap)
    if show_boxes:
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            axes[2].add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="red", linewidth=1.0,
            ))
            if show_scores:
                axes[2].text(x1, max(0, y1 - 2), f"{score:.2f}",
                             fontsize=6, color="yellow",
                             bbox=dict(facecolor="black", alpha=0.5, pad=0.5))
    
    pred_title = f"Prediction (n={len(masks)})"
    if gt_masks is not None:
        biou_val = _calculate_biou_for_vis(masks, gt_masks, pred.image.shape[0], pred.image.shape[1])
        pred_title += f" | B-IoU: {biou_val:.3f}"
    axes[2].set_title(pred_title)
    axes[2].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def visualize_boundary_iou_per_instance(
    image: np.ndarray,
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    *,
    k: float = None,
    cmap: str = "RdYlGn",
    band_alpha: float = 0.95,
    show_misses: bool = True,
    title: str = "",
    ax=None,
):
    """Draw each instance's boundary band, coloured by its own Boundary IoU.

    The figure a Boundary-AP paper actually needs: a thin per-instance contour (the same
    band the metric scores) with a colour that says how well that crown's boundary was
    recovered. Red = poor, green = good, on the pre-registered instance-relative band.

    False positives (matched no GT) and misses (no prediction) carry no Boundary IoU, so
    they are drawn distinctly rather than given a misleading colour.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    from boundary_band import (BOUNDARY_BAND_K, make_instance_relative_mask_to_boundary,
                               per_instance_boundary_iou)

    if k is None:
        k = BOUNDARY_BAND_K
    bious, matches, missed = per_instance_boundary_iou(pred_masks, gt_masks, k=k)
    to_boundary = make_instance_relative_mask_to_boundary(k)

    preds = np.asarray(pred_masks, dtype=bool)
    gts = np.asarray(gt_masks, dtype=bool)
    if preds.ndim == 2:
        preds = preds[None]
    if gts.ndim == 2:
        gts = gts[None]

    if ax is None:
        _fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

    mapper = cm.ScalarMappable(norm=colors.Normalize(0.0, 1.0), cmap=cmap)
    h, w = image.shape[:2]

    # Misses first, so predicted bands draw over them.
    if show_misses:
        for gj in missed:
            band = to_boundary(gts[gj])
            rgba = np.zeros((h, w, 4), dtype=float)
            rgba[band] = (0.35, 0.35, 0.95, 0.85)      # blue = crown found by nobody
            ax.imshow(rgba)

    for pi in range(len(preds)):
        band = to_boundary(preds[pi])
        if not band.any():
            continue
        rgba = np.zeros((h, w, 4), dtype=float)
        if bious[pi] is None:
            rgba[band] = (0.55, 0.55, 0.55, 0.75)      # grey = false positive, no B-IoU
        else:
            r, g, b, _ = mapper.to_rgba(float(bious[pi]))
            rgba[band] = (r, g, b, band_alpha)
        ax.imshow(rgba)

    scored = [v for v in bious if v is not None]
    caption = (f"Boundary IoU, instance-relative band (k = {k:g})   "
               f"n={len(scored)} matched")
    if scored:
        caption += f", mean {np.mean(scored):.3f}"
    if any(v is None for v in bious):
        caption += f"   grey: {sum(v is None for v in bious)} FP"
    if missed:
        caption += f"   blue: {len(missed)} missed"
    ax.set_title(title or caption, fontsize=10)

    cb = plt.colorbar(mapper, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(f"per-instance Boundary IoU (band = {k:g} x equivalent diameter)", fontsize=9)
    return ax.figure


def _calculate_biou_for_vis(pred_masks: np.ndarray, gt_masks: np.ndarray, h: int, w: int, dilation_ratio: float = 0.02) -> float:
    # LEGACY, and not suitable for a reported figure: this is a UNION Boundary IoU over all
    # masks at the LITERATURE band (0.02 x image diagonal), which Claim 1 shows collapses to
    # Mask IoU for 84% of OAM-TCD crowns. Retained only so historical per-image numbers on
    # existing visualisations stay reproducible. For figures use
    # `visualize_boundary_iou_per_instance`, which is per instance and uses the
    # pre-registered instance-relative band.
    # Boundary IoU per Cheng et al. 2021 (inner-band variant). The band geometry
    # is kept identical to `eval_overrides.BoundaryIoUEvaluator`, which is itself
    # aligned with the `boundary-iou-api` reference: d = max(1, round(...)), a
    # full 3x3 structuring element (Chebyshev distance), and border_value=0 so
    # the image edge counts as contour. See that class and
    # Context/active/evaluation_protocol.md section 1.3.0. Changing one of these
    # two without the other silently makes the per-image number on a
    # visualization disagree with the diagnostic.
    from scipy.ndimage import binary_erosion

    pred_array = np.asarray(pred_masks, dtype=bool)
    gt_array = np.asarray(gt_masks, dtype=bool)
    pred_mask = pred_array.any(axis=0) if pred_array.ndim == 3 else pred_array
    gt_mask = gt_array.any(axis=0) if gt_array.ndim == 3 else gt_array
    if pred_mask.size == 0:
        pred_mask = np.zeros((h, w), dtype=bool)
    if gt_mask.size == 0:
        gt_mask = np.zeros((h, w), dtype=bool)

    pred_any = pred_mask.any()
    gt_any = gt_mask.any()
    if not gt_any and not pred_any:
        return 1.0

    dilation_pixels = max(1, int(round(float(np.hypot(h, w)) * dilation_ratio)))
    struct = np.ones((3, 3), dtype=bool)

    if gt_any:
        gt_eroded = binary_erosion(
            gt_mask, structure=struct, iterations=dilation_pixels, border_value=0
        )
        gt_boundary = gt_mask & ~gt_eroded
    else:
        gt_boundary = np.zeros_like(gt_mask, dtype=bool)

    if pred_any:
        pred_eroded = binary_erosion(
            pred_mask, structure=struct, iterations=dilation_pixels, border_value=0
        )
        pred_boundary = pred_mask & ~pred_eroded
    else:
        pred_boundary = np.zeros_like(pred_mask, dtype=bool)

    intersection = np.sum(gt_boundary & pred_boundary)
    union = np.sum(gt_boundary | pred_boundary)

    if union == 0:
        return 1.0 if np.all(gt_mask == pred_mask) else 0.0
    return float(intersection / union)


def _overlay_masks(
    ax,
    masks: np.ndarray,
    alpha: float,
    cmap_name: str,
    edge_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    edge_alpha: float = 0.95,
) -> None:
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_erosion

    if len(masks) == 0:
        return
    
    # Support high-contrast categorical colormaps
    qualitative_cmaps = {"tab10", "tab20", "tab20b", "tab20c", "Set1", "Set2", "Set3", "Pastel1", "Pastel2", "Paired", "Dark2", "Accent"}
    if cmap_name in qualitative_cmaps:
        cmap = plt.get_cmap(cmap_name)
        n_colors = 10 if cmap_name == "tab10" else (8 if cmap_name in {"Set2", "Dark2", "Accent"} else (9 if cmap_name in {"Set1", "Pastel1"} else 20))
        if len(masks) <= n_colors:
            colors = [cmap(i) for i in range(len(masks))]
        else:
            colors = [plt.get_cmap("turbo")(i / len(masks)) for i in range(len(masks))]
    else:
        try:
            cmap = plt.get_cmap(cmap_name)
            colors = cmap(np.linspace(0.1, 0.9, len(masks)))
        except ValueError:
            # Fallback to tab20 if colormap not found
            cmap = plt.get_cmap("tab20")
            colors = [cmap(i % 20) for i in range(len(masks))]

    H, W = masks.shape[-2:]
    overlay = np.zeros((H, W, 4), dtype=float)
    boundary = np.zeros((H, W), dtype=bool)
    for mask, color in zip(masks, colors):
        m = mask.astype(bool)
        overlay[m] = color
        # Track each instance's own boundary (mask minus its 1px-eroded interior)
        # so tightly packed/overlapping instances stay visually separable even
        # when the colormap wraps and repeats colors.
        eroded = binary_erosion(m, iterations=1, border_value=0)
        boundary |= m & ~eroded
    overlay[..., 3] = (overlay[..., :3].sum(-1) > 0).astype(float) * alpha
    # Draw crisp, high-opacity outlines on top of the (semi-transparent) fill.
    overlay[boundary, :3] = edge_color
    overlay[boundary, 3] = edge_alpha
    ax.imshow(overlay)


# ── COCO helpers ────────────────────────────────────────────────────────────

def visualize_coco_ground_truth_sample(
    coco_json: str | Path,
    image_root: str | Path,
    image_id: int | None = None,
    max_instances: int = 200,
    figsize: tuple[float, float] = (18, 6),
):
    """Render one COCO sample as RGB | instance GT | RGB+GT overlay.

    If ``image_id`` is ``None``, the first image with at least one annotation is used.
    This helper is intended as a lightweight export sanity check before training.
    """
    import matplotlib.pyplot as plt

    image_root = Path(image_root)
    with open(coco_json, "r") as f:
        coco = json.load(f)

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    if image_id is None:
        img_meta = next((img for img in coco["images"] if anns_by_image.get(int(img["id"]))), None)
        if img_meta is None:
            raise ValueError(f"No annotated images found in {coco_json}")
    else:
        img_meta = next((img for img in coco["images"] if int(img["id"]) == int(image_id)), None)
        if img_meta is None:
            raise KeyError(f"image_id {image_id} not found in {coco_json}")

    sample_anns = anns_by_image.get(int(img_meta["id"]), [])
    if not sample_anns:
        raise ValueError(f"image_id={img_meta['id']} has no annotations")

    image_path = image_root / img_meta["file_name"]
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    rng = random.Random(int(img_meta["id"]))
    gt_canvas = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    overlay_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    gt_draw = ImageDraw.Draw(gt_canvas)
    overlay_draw = ImageDraw.Draw(overlay_layer)
    for ann in sample_anns[:max_instances]:
        color = tuple(rng.randint(40, 255) for _ in range(3))
        _draw_coco_polygons(
            gt_draw,
            ann.get("segmentation"),
            fill=(*color, 210),
            outline=(255, 255, 255, 255),
        )
        _draw_coco_polygons(
            overlay_draw,
            ann.get("segmentation"),
            fill=(*color, 90),
            outline=(*color, 230),
        )

    image_overlay = Image.alpha_composite(image.convert("RGBA"), overlay_layer)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(image)
    axes[0].set_title(f"Image\n{img_meta['file_name']}")
    axes[1].imshow(gt_canvas)
    axes[1].set_title(
        f"Instance GT (shown={min(len(sample_anns), max_instances)}, total={len(sample_anns)})"
    )
    axes[2].imshow(image_overlay)
    axes[2].set_title("Image + GT overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    print("image_id:", img_meta["id"])
    print("image_path:", image_path)
    print("annotations:", len(sample_anns))
    return fig


def _draw_coco_polygons(draw, segmentation, *, fill, outline) -> None:
    if not isinstance(segmentation, list):
        return
    polygons = [segmentation] if segmentation and all(isinstance(v, (int, float)) for v in segmentation) else segmentation
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue
        points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
        draw.polygon(points, fill=fill, outline=outline)


def visualize_sample_from_coco(
    processor: Sam3Processor,
    coco_json: str | Path,
    image_root: str | Path,
    image_id: int,
    prompt: str = DEFAULT_PROMPT,
    **vis_kwargs,
):
    """Pick an image by COCO ``image_id``, run prediction, plot vs. GT."""
    image_root = Path(image_root)
    with open(coco_json, "r") as f:
        coco = json.load(f)

    img_meta = next((im for im in coco["images"] if im["id"] == image_id), None)
    if img_meta is None:
        raise KeyError(f"image_id {image_id} not found in {coco_json}")

    image_path = image_root / img_meta["file_name"]
    pred = predict_image(processor, image_path, prompt=prompt)

    H, W = img_meta["height"], img_meta["width"]
    anns = [a for a in coco["annotations"] if a["image_id"] == image_id]
    gt_masks = _coco_anns_to_masks(anns, H, W)
    gt_boxes = np.array([_xywh_to_xyxy(a["bbox"]) for a in anns]) if anns else None

    title = vis_kwargs.pop("title", f"image_id={image_id}  ({img_meta['file_name']})")
    return visualize_prediction(
        pred, gt_masks=gt_masks, gt_boxes=gt_boxes, title=title, **vis_kwargs
    )


def _xywh_to_xyxy(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def _display_shape(H: int, W: int, max_size: int = VISUALIZATION_MAX_IMAGE_SIZE) -> tuple[int, int]:
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    scale = min(1.0, float(max_size) / max(int(H), int(W)))
    return max(1, round(int(H) * scale)), max(1, round(int(W) * scale))


def _decode_coco_rle_counts(counts) -> list[int]:
    if isinstance(counts, list):
        return [int(value) for value in counts]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    decoded: list[int] = []
    index = 0
    while index < len(counts):
        value = 0
        shift = 0
        while True:
            code = ord(counts[index]) - 48
            index += 1
            value |= (code & 0x1F) << shift
            shift += 5
            if not code & 0x20:
                if code & 0x10:
                    value |= -1 << shift
                break
        if len(decoded) > 2:
            value += decoded[-2]
        decoded.append(value)
    return decoded


def _rle_to_display_mask(seg: dict, H: int, W: int, display_h: int, display_w: int) -> np.ndarray:
    source_h, source_w = (int(value) for value in seg.get("size", [H, W]))
    counts = _decode_coco_rle_counts(seg.get("counts", []))
    mask = np.zeros((display_h, display_w), dtype=bool)
    cursor = 0
    for run_index, run_length in enumerate(counts):
        run_length = max(0, int(run_length))
        end = min(source_h * source_w, cursor + run_length)
        if run_index % 2 == 1 and end > cursor:
            first_column, last_column = cursor // source_h, (end - 1) // source_h
            for column in range(first_column, last_column + 1):
                start = max(cursor, column * source_h) - column * source_h
                stop = min(end, (column + 1) * source_h) - 1 - column * source_h
                out_column = min(display_w - 1, column * display_w // source_w)
                out_start = min(display_h - 1, start * display_h // source_h)
                out_stop = min(display_h - 1, stop * display_h // source_h)
                mask[out_start:out_stop + 1, out_column] = True
        cursor = end
    return mask


def _segmentation_to_display_mask(
    segmentation, H: int, W: int, display_h: int, display_w: int,
) -> Optional[np.ndarray]:
    if segmentation is None:
        return None
    if isinstance(segmentation, list):
        from pycocotools import mask as mask_utils
        polygons = segmentation if segmentation and isinstance(segmentation[0], list) else [segmentation]
        scaled_polygons = []
        for polygon in polygons:
            scaled = list(polygon)
            scaled[0::2] = [float(value) * display_w / W for value in scaled[0::2]]
            scaled[1::2] = [float(value) * display_h / H for value in scaled[1::2]]
            scaled_polygons.append(scaled)
        rles = mask_utils.frPyObjects(scaled_polygons, display_h, display_w)
        return np.asarray(mask_utils.decode(mask_utils.merge(rles))).astype(bool)
    if isinstance(segmentation, dict):
        return _rle_to_display_mask(segmentation, H, W, display_h, display_w)
    return None


def _paint_rle_on_target(
    segmentation: dict,
    H: int,
    W: int,
    display_h: int,
    display_w: int,
    target: np.ndarray,
    color: np.ndarray | None = None,
) -> None:
    source_h, source_w = (int(value) for value in segmentation.get("size", [H, W]))
    counts = _decode_coco_rle_counts(segmentation.get("counts", []))
    cursor = 0
    for run_index, run_length in enumerate(counts):
        run_length = max(0, int(run_length))
        end = min(source_h * source_w, cursor + run_length)
        if run_index % 2 == 1 and end > cursor:
            first_column, last_column = cursor // source_h, (end - 1) // source_h
            for column in range(first_column, last_column + 1):
                start = max(cursor, column * source_h) - column * source_h
                stop = min(end, (column + 1) * source_h) - 1 - column * source_h
                out_column = min(display_w - 1, column * display_w // source_w)
                out_start = min(display_h - 1, start * display_h // source_h)
                out_stop = min(display_h - 1, stop * display_h // source_h)
                if color is None:
                    target[out_start:out_stop + 1, out_column] = True
                else:
                    target[out_start:out_stop + 1, out_column, :3] = color[:3]
        cursor = end


def _paint_segmentation_on_target(
    segmentation,
    H: int,
    W: int,
    display_h: int,
    display_w: int,
    target: np.ndarray,
    color: np.ndarray | None = None,
) -> None:
    if isinstance(segmentation, dict):
        _paint_rle_on_target(segmentation, H, W, display_h, display_w, target, color)
        return
    if not isinstance(segmentation, list):
        return
    from pycocotools import mask as mask_utils
    polygons = segmentation if segmentation and isinstance(segmentation[0], list) else [segmentation]
    scaled_polygons = []
    for polygon in polygons:
        if len(polygon) < 6:
            continue
        scaled = list(polygon)
        scaled[0::2] = [float(value) * display_w / W for value in scaled[0::2]]
        scaled[1::2] = [float(value) * display_h / H for value in scaled[1::2]]
        scaled_polygons.append(scaled)
    if not scaled_polygons:
        return
    x_values = [value for polygon in scaled_polygons for value in polygon[0::2]]
    y_values = [value for polygon in scaled_polygons for value in polygon[1::2]]
    x0, x1 = max(0, int(np.floor(min(x_values)))), min(display_w, int(np.ceil(max(x_values))) + 1)
    y0, y1 = max(0, int(np.floor(min(y_values)))), min(display_h, int(np.ceil(max(y_values))) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    local_polygons = []
    for polygon in scaled_polygons:
        local = list(polygon)
        local[0::2] = [value - x0 for value in local[0::2]]
        local[1::2] = [value - y0 for value in local[1::2]]
        local_polygons.append(local)
    rles = mask_utils.frPyObjects(local_polygons, y1 - y0, x1 - x0)
    patch = np.asarray(mask_utils.decode(mask_utils.merge(rles)))
    if patch.ndim == 3:
        patch = patch.any(axis=2)
    if color is None:
        target[y0:y1, x0:x1] |= patch.astype(bool)
    else:
        target_patch = target[y0:y1, x0:x1]
        target_patch[patch.astype(bool), :3] = color[:3]


def _overlay_segmentation_collection(
    ax,
    items: list[dict],
    H: int,
    W: int,
    display_h: int,
    display_w: int,
    alpha: float,
    fill_color: tuple[float, float, float],
    edge_color: tuple[float, float, float],
    edge_alpha: float,
) -> None:
    from scipy.ndimage import binary_erosion
    if not items:
        return
    overlay = np.zeros((display_h, display_w, 4), dtype=float)
    boundary = np.zeros((display_h, display_w), dtype=bool)
    for item in items:
        mask = _segmentation_to_display_mask(
            item.get("segmentation"), H, W, display_h, display_w,
        )
        if mask is None or not mask.any():
            continue
        overlay[mask, :3] = fill_color
        eroded = binary_erosion(mask, iterations=1, border_value=0)
        boundary |= mask & ~eroded
    overlay[..., 3] = (overlay[..., :3].sum(-1) > 0).astype(float) * alpha
    overlay[boundary, :3] = edge_color
    overlay[boundary, 3] = edge_alpha
    ax.imshow(overlay)


def _coco_anns_to_masks(
    anns, H: int, W: int, max_instances: Optional[int] = None,
    output_shape: tuple[int, int] | None = None,
) -> Optional[np.ndarray]:
    if max_instances is not None:
        anns = anns[:max_instances]
    if not anns:
        return None
    display_h, display_w = output_shape or (H, W)
    masks = []
    for ann in anns:
        mask = _segmentation_to_display_mask(ann.get("segmentation"), H, W, display_h, display_w)
        if mask is not None:
            masks.append(mask)
    if not masks:
        return None
    return np.stack(masks, axis=0)


def list_coco_image_ids(coco_json: str | Path, n: int = 20) -> list[int]:
    """Return the first ``n`` image ids in a COCO json (handy for notebooks)."""
    with open(coco_json, "r") as f:
        coco = json.load(f)
    return [im["id"] for im in coco["images"][:n]]


def pick_dense_coco_image_ids(
    coco_json: str | Path,
    n: int,
    min_gt: int = 5,
) -> list[int]:
    """Return the deterministic top-``n`` images by ground-truth instance count."""
    with open(coco_json, "r") as f:
        coco = json.load(f)

    counts: dict[int, int] = {}
    for annotation in coco["annotations"]:
        image_id = int(annotation["image_id"])
        counts[image_id] = counts.get(image_id, 0) + 1
    valid_ids = {int(image["id"]) for image in coco["images"]}
    ranked = sorted(
        ((count, image_id) for image_id, count in counts.items() if image_id in valid_ids and count >= min_gt),
        key=lambda item: (-item[0], item[1]),
    )
    return [image_id for _, image_id in ranked[:n]]


def pick_best_biou_image_ids(
    coco_json: str | Path,
    pred_json: str | Path,
    n: int,
    score_threshold: float = DEFAULT_CONFIDENCE,
    min_gt: int = 5,
    candidate_pool: int = 64,
) -> list[int]:
    """Return the best-B-IoU images from a bounded dense-scene candidate pool."""
    with open(coco_json, "r") as f:
        coco = json.load(f)

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        anns_by_image.setdefault(image_id, []).append(ann)
    meta_by_id = {int(im["id"]): im for im in coco["images"]}
    candidates = sorted(
        (
            (len(anns), image_id)
            for image_id, anns in anns_by_image.items()
            if len(anns) >= min_gt and image_id in meta_by_id
        ),
        key=lambda item: (-item[0], item[1]),
    )[:candidate_pool]
    candidate_ids = {image_id for _, image_id in candidates}

    scored: list[tuple[int, float]] = []
    for image_id, preds in _iter_dumped_prediction_groups(pred_json):
        if image_id not in candidate_ids or not preds:
            continue
        img_meta = meta_by_id[image_id]
        H, W = int(img_meta["height"]), int(img_meta["width"])
        display_shape = _display_shape(H, W)
        gt_mask = _coco_anns_to_union_mask(
            anns_by_image[image_id], H, W, output_shape=display_shape,
        )
        pred_mask = _preds_to_union_mask(
            preds, H, W, score_threshold, output_shape=display_shape,
        )
        if not pred_mask.any():
            continue
        scored.append((image_id, _calculate_biou_for_vis(
            pred_mask, gt_mask, display_shape[0], display_shape[1],
        )))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return [image_id for image_id, _ in scored[:n]]


def pick_dense_biou_image_ids(
    coco_json: str | Path,
    pred_json: str | Path,
    buckets: list[tuple[int, int | None]],
    per_bucket: int = 1,
    score_threshold: float = DEFAULT_CONFIDENCE,
) -> dict[str, list[int]]:
    """Pick images with highest B-IoU from GT-density buckets.

    Args:
        coco_json: COCO ground-truth annotations.
        pred_json: COCO-format predictions (e.g. tiled + mask-NMS output).
        buckets: list of (min_gt, max_gt) tuples; max_gt=None means no upper bound.
            e.g. [(50, 100), (100, 300), (300, 500), (500, None)]
        per_bucket: number of top-B-IoU images to pick per bucket.
        score_threshold: score threshold for predictions.

    Returns:
        Dict mapping bucket label (e.g. "50-100") to list of image_ids.
    """
    with open(coco_json, "r") as f:
        coco = json.load(f)

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        anns_by_image.setdefault(image_id, []).append(ann)
    meta_by_id = {int(im["id"]): im for im in coco["images"]}

    # Pre-load all predictions grouped by image_id
    preds_by_image = load_dumped_predictions(pred_json)

    result: dict[str, list[int]] = {}
    for min_gt, max_gt in buckets:
        label = f"{min_gt}-{max_gt if max_gt is not None else '+'}"
        candidates = []
        for image_id, anns in anns_by_image.items():
            n_gt = len(anns)
            if n_gt < min_gt:
                continue
            if max_gt is not None and n_gt >= max_gt:
                continue
            if image_id not in meta_by_id:
                continue
            preds = preds_by_image.get(image_id, [])
            if not preds:
                continue
            img_meta = meta_by_id[image_id]
            H, W = int(img_meta["height"]), int(img_meta["width"])
            display_shape = _display_shape(H, W)
            gt_mask = _coco_anns_to_union_mask(
                anns, H, W, output_shape=display_shape,
            )
            pred_mask = _preds_to_union_mask(
                preds, H, W, score_threshold, output_shape=display_shape,
            )
            if not pred_mask.any():
                continue
            biou = _calculate_biou_for_vis(
                pred_mask, gt_mask, display_shape[0], display_shape[1],
            )
            candidates.append((biou, image_id, n_gt))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        result[label] = [image_id for _, image_id, _ in candidates[:per_bucket]]

    return result


def _iter_json_array_objects(path: str | Path, chunk_size: int = 1 << 20) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    with Path(path).open("r") as f:
        buffer = f.read(chunk_size)
        position = 0
        started = False
        need_value = True
        while True:
            while position >= len(buffer):
                more = f.read(chunk_size)
                if not more:
                    return
                buffer = more
                position = 0
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position >= len(buffer):
                continue

            if not started:
                if buffer[position] != "[":
                    raise ValueError(f"Expected a JSON array in {path}")
                started = True
                position += 1
                continue

            if not need_value:
                if buffer[position] == "]":
                    return
                if buffer[position] != ",":
                    raise ValueError(f"Expected a comma between JSON array elements in {path}")
                position += 1
                need_value = True
                continue

            if buffer[position] == "]":
                return
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    more = f.read(chunk_size)
                    if not more:
                        raise
                    buffer = buffer[position:] + more
                    position = 0
                    continue
                yield value
                position = end
                need_value = False
                break


def _iter_dumped_prediction_groups(pred_json: str | Path) -> Iterator[tuple[int, list[dict]]]:
    current_image_id: int | None = None
    current: list[dict] = []
    for prediction in _iter_json_array_objects(pred_json):
        image_id = int(prediction["image_id"])
        if current_image_id is None:
            current_image_id = image_id
        elif image_id != current_image_id:
            yield current_image_id, current
            current_image_id = image_id
            current = []
        current.append(prediction)
    if current_image_id is not None:
        yield current_image_id, current


def load_dumped_predictions(
    pred_json: str | Path,
    image_ids: set[int] | None = None,
) -> dict[int, list[dict]]:
    """Stream and group selected dumped COCO-format predictions by image_id."""
    by_image: dict[int, list[dict]] = {}
    for prediction in _iter_json_array_objects(pred_json):
        image_id = int(prediction["image_id"])
        if image_ids is None or image_id in image_ids:
            by_image.setdefault(image_id, []).append(prediction)
    return by_image


def _load_dumped_preds(
    pred_json: str | Path,
    image_ids: set[int] | None = None,
) -> dict[int, list[dict]]:
    return load_dumped_predictions(pred_json, image_ids)


def _preds_to_masks_boxes_scores(
    preds: list[dict], H: int, W: int, score_threshold: float = 0.0,
    max_instances: Optional[int] = None,
    output_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kept = [p for p in preds if p.get("score", 1.0) >= score_threshold]
    kept.sort(key=lambda p: float(p.get("score", 1.0)), reverse=True)
    if max_instances is not None:
        kept = kept[:max_instances]
    display_h, display_w = output_shape or (H, W)
    if not kept:
        return (np.zeros((0, display_h, display_w), dtype=bool), np.zeros((0, 4)), np.zeros((0,)))

    decoded = []
    decoded_predictions = []
    for prediction in kept:
        mask = _segmentation_to_display_mask(
            prediction.get("segmentation"), H, W, display_h, display_w,
        )
        if mask is not None:
            decoded.append(mask)
            decoded_predictions.append(prediction)
    if not decoded:
        return (np.zeros((0, display_h, display_w), dtype=bool), np.zeros((0, 4)), np.zeros((0,)))
    boxes = np.array([
        tuple(coordinate * scale for coordinate, scale in zip(
            _xywh_to_xyxy(prediction["bbox"]),
            (display_w / W, display_h / H, display_w / W, display_h / H),
        ))
        for prediction in decoded_predictions
    ])
    scores = np.array([float(p.get("score", 1.0)) for p in decoded_predictions])
    return np.stack(decoded, axis=0), boxes, scores


def _coco_anns_to_union_mask(
    anns, H: int, W: int, output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    display_h, display_w = output_shape or (H, W)
    union = np.zeros((display_h, display_w), dtype=bool)
    for ann in anns:
        mask = _segmentation_to_display_mask(
            ann.get("segmentation"), H, W, display_h, display_w,
        )
        if mask is not None:
            union |= mask
    return union


def _preds_to_union_mask(
    preds: list[dict], H: int, W: int, score_threshold: float = 0.0,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    display_h, display_w = output_shape or (H, W)
    union = np.zeros((display_h, display_w), dtype=bool)
    for prediction in preds:
        if prediction.get("score", 1.0) < score_threshold:
            continue
        mask = _segmentation_to_display_mask(
            prediction.get("segmentation"), H, W, display_h, display_w,
        )
        if mask is not None:
            union |= mask
    return union


def visualize_dumped_prediction(
    coco_json: str | Path,
    image_root: str | Path | None,
    image_id: int,
    pred_json: str | Path,
    label: str = "Prediction",
    score_threshold: float = DEFAULT_CONFIDENCE,
    max_instances: Optional[int] = None,
    figsize: tuple[float, float] = (18, 6),
    gt_cmap: str = "tab20",
    pred_cmap: str = "tab20",
    max_gt_instances: Optional[int] = None,
    predictions: list[dict] | None = None,
):
    """Render Image | GT | Prediction for one image_id from a single dumped
    COCO-format predictions json (e.g. tiled + mask-NMS output), at a fixed
    score threshold. Use this to inspect the tuned operating point without
    re-running the model.
    """
    import matplotlib.pyplot as plt

    image_root_path = Path(image_root) if image_root is not None else None
    with open(coco_json, "r") as f:
        coco = json.load(f)

    img_meta = next((im for im in coco["images"] if int(im["id"]) == int(image_id)), None)
    if img_meta is None:
        raise KeyError(f"image_id {image_id} not found in {coco_json}")

    H, W = int(img_meta["height"]), int(img_meta["width"])
    display_h, display_w = _display_shape(H, W)
    if image_root_path is None:
        raise ValueError("image_root is required for OAM-TCD visualization")
    with Image.open(image_root_path / img_meta["file_name"]) as source_image:
        source_image = source_image.convert("RGB")
        source_image.thumbnail((display_w, display_h), Image.Resampling.LANCZOS)
        image = np.asarray(source_image)

    anns = [a for a in coco["annotations"] if a["image_id"] == image_id]
    display_shape = (display_h, display_w)
    gt_union = _coco_anns_to_union_mask(anns, H, W, output_shape=display_shape)
    preds = predictions if predictions is not None else _load_dumped_preds(pred_json, {int(image_id)}).get(int(image_id), [])
    visible_preds = [p for p in preds if p.get("score", 1.0) >= score_threshold]
    visible_preds.sort(key=lambda p: float(p.get("score", 1.0)), reverse=True)
    if max_instances is not None:
        visible_preds = visible_preds[:max_instances]
    visible_anns = anns if max_gt_instances is None else anns[:max_gt_instances]
    pred_union = _preds_to_union_mask(
        preds, H, W, score_threshold, output_shape=display_shape,
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title(f"Image\n{img_meta['file_name']}")
    axes[0].axis("off")

    axes[1].set_facecolor("black")
    axes[1].imshow(image)
    _overlay_segmentation_collection(
        axes[1], visible_anns, H, W, display_h, display_w,
        alpha=0.42, fill_color=(0.05, 0.62, 0.58),
        edge_color=(0.02, 0.16, 0.16), edge_alpha=0.90,
    )
    gt_total = len(anns)
    axes[1].set_title(f"Ground truth (shown={len(visible_anns)}/{gt_total})")
    axes[1].axis("off")

    axes[2].imshow(image)
    _overlay_segmentation_collection(
        axes[2], visible_preds, H, W, display_h, display_w,
        alpha=0.42, fill_color=(0.95, 0.42, 0.10),
        edge_color=(0.25, 0.05, 0.01), edge_alpha=0.90,
    )
    pred_total = len(visible_preds)
    biou_val = _calculate_biou_for_vis(pred_union, gt_union, display_h, display_w)
    axes[2].set_title(f"{label} (shown={pred_total}; thr={score_threshold:.2f})\nB-IoU: {biou_val:.3f}")
    axes[2].axis("off")

    fig.suptitle(f"image_id={image_id}  GT n={gt_total}")
    fig.tight_layout()
    return fig


def _decode_rle_crop(segmentation: dict[str, Any], x0: int, y0: int, width: int, height: int) -> np.ndarray:
    full_height, full_width = (int(value) for value in segmentation["size"])
    counts = segmentation["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    if isinstance(counts, str):
        values: list[int] = []
        index = 0
        while index < len(counts):
            value = 0
            shift = 0
            while True:
                byte = ord(counts[index]) - 48
                index += 1
                value |= (byte & 0x1F) << shift
                shift += 5
                if not byte & 0x20:
                    break
            if byte & 0x10:
                value |= -1 << shift
            if len(values) > 2:
                value += values[-2]
            values.append(value)
        counts = values
    crop = np.zeros((height, width), dtype=bool)
    position = 0
    value = 0
    for count in counts:
        end = position + int(count)
        if value and count:
            first_col = position // full_height
            last_col = (end - 1) // full_height
            for col in range(max(first_col, x0), min(last_col + 1, x0 + width)):
                local_start = position - col * full_height if col == first_col else 0
                local_end = end - col * full_height if col == last_col else full_height
                start_row = max(local_start, y0)
                end_row = min(local_end, y0 + height)
                if end_row > start_row:
                    crop[start_row - y0:end_row - y0, col - x0] = True
        position = end
        value = 1 - value
    if position != full_height * full_width:
        raise ValueError("RLE counts do not cover the declared mask size.")
    return crop


def _bbox_intersects_crop(
    bbox: Sequence[float], x0: int, y0: int, width: int, height: int,
) -> bool:
    x, y, box_width, box_height = bbox
    return x < x0 + width and x + box_width > x0 and y < y0 + height and y + box_height > y0


def _crop_segmentation_to_display_mask(
    segmentation,
    x0: int,
    y0: int,
    width: int,
    height: int,
    display_h: int,
    display_w: int,
    full_h: int,
    full_w: int,
) -> Optional[np.ndarray]:
    if isinstance(segmentation, dict):
        native = _decode_rle_crop(segmentation, x0, y0, width, height)
        return np.asarray(
            Image.fromarray(native.astype(np.uint8)).resize(
                (display_w, display_h), Image.Resampling.NEAREST,
            )
        ).astype(bool)
    if not isinstance(segmentation, list):
        return None
    from pycocotools import mask as mask_utils
    polygons = segmentation if segmentation and isinstance(segmentation[0], list) else [segmentation]
    scaled_polygons = []
    for polygon in polygons:
        if len(polygon) < 6:
            continue
        scaled = list(polygon)
        scaled[0::2] = [(float(value) - x0) * display_w / width for value in scaled[0::2]]
        scaled[1::2] = [(float(value) - y0) * display_h / height for value in scaled[1::2]]
        scaled_polygons.append(scaled)
    if not scaled_polygons:
        return None
    rles = mask_utils.frPyObjects(scaled_polygons, display_h, display_w)
    mask = np.asarray(mask_utils.decode(mask_utils.merge(rles))).astype(bool)
    return mask.any(axis=2) if mask.ndim == 3 else mask


def _overlay_crop_segmentation_collection(
    ax,
    items: list[dict],
    x0: int,
    y0: int,
    width: int,
    height: int,
    display_h: int,
    display_w: int,
    full_h: int,
    full_w: int,
    fill_color: tuple[float, float, float],
    edge_color: tuple[float, float, float],
    alpha: float,
) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    overlay = np.zeros((display_h, display_w, 4), dtype=float)
    union = np.zeros((display_h, display_w), dtype=bool)
    boundary = np.zeros((display_h, display_w), dtype=bool)
    for item in items:
        segmentation = item.get("segmentation")
        mask = _crop_segmentation_to_display_mask(
            segmentation, x0, y0, width, height, display_h, display_w, full_h, full_w,
        )
        if mask is None or not mask.any():
            continue
        union |= mask
        overlay[mask, :3] = fill_color
        eroded = binary_erosion(mask, iterations=1, border_value=0)
        boundary |= mask & ~eroded
    overlay[..., 3] = (overlay[..., :3].sum(-1) > 0).astype(float) * alpha
    overlay[boundary, :3] = edge_color
    overlay[boundary, 3] = 0.90
    ax.imshow(overlay)
    return union


def visualize_dumped_prediction_tile(
    coco_json: str | Path,
    image_root: str | Path | None,
    image_id: int,
    pred_json: str | Path,
    x0: int,
    y0: int,
    width: int,
    height: int,
    label: str = "Tiled + mask-NMS",
    score_threshold: float = DEFAULT_CONFIDENCE,
    predictions: list[dict] | None = None,
    display_size: int = 2048,
):
    """Draw one window of a dumped prediction set over its image with ground truth, at the given score threshold."""
    import matplotlib.pyplot as plt

    with open(coco_json, "r") as f:
        coco = json.load(f)
    img_meta = next((im for im in coco["images"] if int(im["id"]) == int(image_id)), None)
    if img_meta is None:
        raise KeyError(f"image_id {image_id} not found in {coco_json}")
    full_h, full_w = int(img_meta["height"]), int(img_meta["width"])
    if min(x0, y0, width, height) < 0 or x0 + width > full_w or y0 + height > full_h:
        raise ValueError("Tile crop must lie inside the source image.")
    scale = min(1.0, float(display_size) / max(width, height))
    display_h, display_w = max(1, round(height * scale)), max(1, round(width * scale))

    if image_root is None:
        raise ValueError("image_root is required for OAM-TCD visualization")
    with Image.open(Path(image_root) / img_meta["file_name"]) as source_image:
        image = np.asarray(source_image.convert("RGB").crop(
            (x0, y0, x0 + width, y0 + height)
        ).resize((display_w, display_h), Image.Resampling.LANCZOS))

    annotations = [
        annotation for annotation in coco["annotations"]
        if int(annotation["image_id"]) == int(image_id)
        and annotation.get("bbox")
        and _bbox_intersects_crop(annotation["bbox"], x0, y0, width, height)
    ]
    all_predictions = predictions if predictions is not None else _load_dumped_preds(pred_json, {int(image_id)}).get(int(image_id), [])
    visible_predictions = [
        prediction for prediction in all_predictions
        if prediction.get("score", 1.0) >= score_threshold
        and prediction.get("bbox")
        and _bbox_intersects_crop(prediction["bbox"], x0, y0, width, height)
    ]
    visible_predictions.sort(key=lambda prediction: float(prediction.get("score", 1.0)), reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=180)
    axes[0].imshow(image)
    axes[0].set_title(f"Tile RGB\n(x={x0}, y={y0}, w={width}, h={height})")
    axes[0].axis("off")
    axes[1].imshow(image)
    gt_union = _overlay_crop_segmentation_collection(
        axes[1], annotations, x0, y0, width, height, display_h, display_w,
        full_h, full_w, (0.05, 0.62, 0.58), (0.02, 0.16, 0.16), 0.42,
    )
    axes[1].set_title(f"Ground truth instances\nvisible={len(annotations)}")
    axes[1].axis("off")
    axes[2].imshow(image)
    pred_union = _overlay_crop_segmentation_collection(
        axes[2], visible_predictions, x0, y0, width, height, display_h, display_w,
        full_h, full_w, (0.95, 0.42, 0.10), (0.25, 0.05, 0.01), 0.42,
    )
    biou = _calculate_biou_for_vis(pred_union, gt_union, display_h, display_w)
    axes[2].set_title(
        f"{label}\nvisible={len(visible_predictions)}, threshold={score_threshold:.2f}, B-IoU={biou:.3f}"
    )
    axes[2].axis("off")
    fig.tight_layout()
    return fig


def visualize_prediction_comparison(
    coco_json: str | Path,
    image_root: str | Path,
    image_id: int,
    pred_json_a: str | Path,
    pred_json_b: str | Path,
    label_a: str = "Whole-image",
    label_b: str = "Tiled + mask-NMS",
    score_threshold_a: float = DEFAULT_CONFIDENCE,
    score_threshold_b: float = DEFAULT_CONFIDENCE,
    max_instances: Optional[int] = None,
    figsize: tuple[float, float] = (24, 6),
    gt_cmap: str = "tab20",
    pred_cmap: str = "tab20",
):
    """Render Image | GT | Prediction A | Prediction B for one image_id.

    Both prediction sets must be dumped COCO-format prediction jsons (list of
    dicts with image_id/segmentation(RLE)/bbox/score), e.g.
    ``coco_predictions_segm.json`` (whole-image) and
    ``coco_predictions_segm_nms.json`` (tiled + mask-NMS). B-IoU is computed
    against GT for each prediction panel using the same formula as
    ``eval_overrides.BoundaryIoUEvaluator``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image_root = Path(image_root)
    with open(coco_json, "r") as f:
        coco = json.load(f)

    img_meta = next((im for im in coco["images"] if int(im["id"]) == int(image_id)), None)
    if img_meta is None:
        raise KeyError(f"image_id {image_id} not found in {coco_json}")

    H, W = img_meta["height"], img_meta["width"]
    image = np.asarray(Image.open(image_root / img_meta["file_name"]).convert("RGB"))

    anns = [a for a in coco["annotations"] if a["image_id"] == image_id]
    gt_masks = _coco_anns_to_masks(anns, H, W)
    if gt_masks is None:
        gt_masks = np.zeros((0, H, W), dtype=bool)

    preds_a = _load_dumped_preds(pred_json_a).get(int(image_id), [])
    preds_b = _load_dumped_preds(pred_json_b).get(int(image_id), [])
    masks_a, boxes_a, scores_a = _preds_to_masks_boxes_scores(preds_a, H, W, score_threshold_a)
    masks_b, boxes_b, scores_b = _preds_to_masks_boxes_scores(preds_b, H, W, score_threshold_b)

    if max_instances is not None:
        def _cap(masks, boxes, scores):
            if len(masks) <= max_instances:
                return masks, boxes, scores
            order = np.argsort(-scores)[:max_instances]
            return masks[order], boxes[order], scores[order]

        masks_a, boxes_a, scores_a = _cap(masks_a, boxes_a, scores_a)
        masks_b, boxes_b, scores_b = _cap(masks_b, boxes_b, scores_b)

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title(f"Image\n{img_meta['file_name']}")
    axes[0].axis("off")

    axes[1].set_facecolor("black")
    axes[1].imshow(image, alpha=0.35)
    _overlay_masks(
        axes[1], gt_masks, alpha=0.78, cmap_name=gt_cmap,
        edge_color=(1.0, 1.0, 1.0), edge_alpha=1.0,
    )
    axes[1].set_title(f"Ground truth (n={len(gt_masks)})")
    axes[1].axis("off")

    for ax, masks, label, thr in (
        (axes[2], masks_a, label_a, score_threshold_a),
        (axes[3], masks_b, label_b, score_threshold_b),
    ):
        ax.imshow(image)
        _overlay_masks(ax, masks, alpha=0.5, cmap_name=pred_cmap)
        biou_val = _calculate_biou_for_vis(masks, gt_masks, H, W)
        ax.set_title(f"{label} (n={len(masks)}, thr={thr:.2f})\nB-IoU: {biou_val:.3f}")
        ax.axis("off")

    fig.suptitle(f"image_id={image_id}  GT n={len(gt_masks)}")
    fig.tight_layout()
    return fig


# ── CLI: whole-image vs tiled+NMS comparison ─────────────────────────────────
#
# Usage:
#   python Core/visualization.py \
#       --gt runtime/data/oam_tcd_instance_coco/annotations/test_annotations.coco.json \
#       --image-root runtime/data/oam_tcd_instance_coco \
#       --pred-a experiments/sam3_crop_frts/predictions/oam_tcd/coco_predictions_segm.json \
#       --pred-b runtime/predictions/sam3_crop_frts/coco_predictions_segm_nms.json \
#       --image-ids 12 34 56 \
#       --out-dir runtime/results/vis_tiled_comparison
#
# If --image-ids is omitted, picks the N images with the most GT instances
# (small/dense crowns are where tiling should help most).

def _pick_dense_image_ids(coco_json: str, n: int) -> list[int]:
    with open(coco_json) as f:
        coco = json.load(f)
    counts: dict[int, int] = {}
    for ann in coco["annotations"]:
        counts[ann["image_id"]] = counts.get(ann["image_id"], 0) + 1
    return [img_id for img_id, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]


def _cli_main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Visualize whole-image vs tiled+NMS predictions.")
    ap.add_argument("--gt", required=True, type=str)
    ap.add_argument("--image-root", required=True, type=str)
    ap.add_argument("--pred-a", required=True, type=str, help="Whole-image predictions COCO json.")
    ap.add_argument("--pred-b", required=True, type=str, help="Tiled+NMS predictions COCO json.")
    ap.add_argument("--label-a", type=str, default="Whole-image")
    ap.add_argument("--label-b", type=str, default="Tiled + mask-NMS")
    ap.add_argument("--score-thr-a", type=float, default=DEFAULT_CONFIDENCE)
    ap.add_argument("--score-thr-b", type=float, default=DEFAULT_CONFIDENCE)
    ap.add_argument("--image-ids", type=int, nargs="*", default=None)
    ap.add_argument("--n-auto", type=int, default=4, help="If --image-ids omitted, how many to auto-pick.")
    ap.add_argument("--out-dir", required=True, type=str)
    args = ap.parse_args()

    image_ids = args.image_ids or _pick_dense_image_ids(args.gt, args.n_auto)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        print(f"Rendering image_id={image_id} ...")
        fig = visualize_prediction_comparison(
            coco_json=args.gt,
            image_root=args.image_root,
            image_id=image_id,
            pred_json_a=args.pred_a,
            pred_json_b=args.pred_b,
            label_a=args.label_a,
            label_b=args.label_b,
            score_threshold_a=args.score_thr_a,
            score_threshold_b=args.score_thr_b,
        )
        out_path = out_dir / f"compare_{image_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    _cli_main()
