#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""D4 augmentation transforms and configuration-aligned inspection utilities.

`RandomVerticalFlip` and `RandomRot90` mirror SAM3's datapoint-aware horizontal flip for
images, masks, boxes, semantic targets, and geometric prompts. Rotations use exact 90-degree
multiples, avoiding interpolation and preserving area. The inspection utilities instantiate
the configured training transforms and verify geometry on OAM-TCD samples without training.
Heavy dependencies used only for inspection are imported lazily.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image as PILImage

from project_paths import DATA_DIR, add_support_to_pythonpath

# Mirror the training resolution / dataset knobs from the single source of truth.
try:
    from config import MAX_ANN_PER_IMG, MAX_TRAIN_QUERIES, RESOLUTION
except Exception:  # pragma: no cover - config import is light, but stay defensive
    RESOLUTION, MAX_ANN_PER_IMG, MAX_TRAIN_QUERIES = 1008, 1000, 10

# Augmentation defaults retained for configuration-aligned inspection.
CROP_MIN_SIZE = 896
CROP_MAX_SIZE = 1152
HFLIP_P = 0.5
COLOR_P = 0.5
COLOR_JITTER = dict(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
CONSISTENT_TRANSFORM = False

try:  # Pillow >= 9.1 exposes the Transpose enum; older versions use module attrs
    _PIL_ROTATE = {
        1: PILImage.Transpose.ROTATE_90,
        2: PILImage.Transpose.ROTATE_180,
        3: PILImage.Transpose.ROTATE_270,
    }
except AttributeError:  # pragma: no cover - legacy Pillow
    _PIL_ROTATE = {
        1: PILImage.ROTATE_90,
        2: PILImage.ROTATE_180,
        3: PILImage.ROTATE_270,
    }


# ── geometric helpers (datapoint-level) ──────────────────────────────────────

def vflip(datapoint, index):
    """Vertical flip of image ``index`` and all its targets (mirror of ``hflip``)."""
    data = datapoint.images[index].data
    datapoint.images[index].data = F.vflip(data)

    if torch.is_tensor(data):
        h = int(data.shape[-2])
    else:
        _, h = data.size  # PIL: (width, height)

    flip_y = torch.as_tensor([1, -1, 1, -1], dtype=torch.float32)
    shift_y = torch.as_tensor([0, h, 0, h], dtype=torch.float32)
    for obj in datapoint.images[index].objects:
        boxes = obj.bbox.view(1, 4)
        obj.bbox = boxes[:, [0, 3, 2, 1]] * flip_y + shift_y
        if obj.segment is not None:
            obj.segment = F.vflip(obj.segment)

    for query in datapoint.find_queries:
        if query.semantic_target is not None:
            query.semantic_target = F.vflip(query.semantic_target)
        if query.image_id == index and query.input_bbox is not None:
            boxes = query.input_bbox
            query.input_bbox = boxes[:, [0, 3, 2, 1]] * flip_y + shift_y
        if query.image_id == index and query.input_points is not None:
            points = query.input_points
            query.input_points = points * torch.as_tensor(
                [1, -1, 1], dtype=torch.float32
            ) + torch.as_tensor([0, h, 0], dtype=torch.float32)
    return datapoint


def _rot90_box_xyxy(boxes: torch.Tensor, k: int, h: int, w: int) -> torch.Tensor:
    """Remap XYXY-abs boxes under ``k`` CCW quarter-turns. ``h,w`` are pre-rotation."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if k == 1:
        return torch.stack([y1, w - x2, y2, w - x1], dim=1)
    if k == 2:
        return torch.stack([w - x2, h - y2, w - x1, h - y1], dim=1)
    return torch.stack([h - y2, x1, h - y1, x2], dim=1)  # k == 3


def _rot90_points(points: torch.Tensor, k: int, h: int, w: int) -> torch.Tensor:
    """Remap (x, y, label) points under ``k`` CCW quarter-turns. ``h,w`` pre-rotation."""
    x = points[..., 0].clone()
    y = points[..., 1].clone()
    out = points.clone()
    if k == 1:
        out[..., 0], out[..., 1] = y, w - x
    elif k == 2:
        out[..., 0], out[..., 1] = w - x, h - y
    else:  # k == 3
        out[..., 0], out[..., 1] = h - y, x
    return out


def rot90(datapoint, index, k: int):
    """Rotate image ``index`` and all its targets by ``k`` CCW quarter-turns."""
    k = k % 4
    if k == 0:
        return datapoint

    data = datapoint.images[index].data
    if torch.is_tensor(data):
        h, w = int(data.shape[-2]), int(data.shape[-1])
        datapoint.images[index].data = torch.rot90(data, k, dims=(-2, -1))
    else:
        w, h = data.size  # PIL: (width, height)
        datapoint.images[index].data = data.transpose(_PIL_ROTATE[k])

    for obj in datapoint.images[index].objects:
        obj.bbox = _rot90_box_xyxy(obj.bbox.view(1, 4), k, h, w)
        if obj.segment is not None:
            obj.segment = torch.rot90(obj.segment, k, dims=(-2, -1))

    for query in datapoint.find_queries:
        if query.semantic_target is not None:
            query.semantic_target = torch.rot90(query.semantic_target, k, dims=(-2, -1))
        if query.image_id == index and query.input_bbox is not None:
            query.input_bbox = _rot90_box_xyxy(query.input_bbox, k, h, w)
        if query.image_id == index and query.input_points is not None:
            query.input_points = _rot90_points(query.input_points, k, h, w)

    # k=1/3 swap height<->width; the datapoint stores size as (h, w).
    datapoint.images[index].size = (w, h) if k in (1, 3) else (h, w)
    return datapoint


# ── transform classes (mirror RandomHorizontalFlip) ──────────────────────────

class RandomVerticalFlip:
    """Flip each image, with its boxes, masks, and prompts, vertically with probability `p`."""
    def __init__(self, consistent_transform, p=0.5) -> None:
        self.p = p
        self.consistent_transform = consistent_transform

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for i in range(len(datapoint.images)):
                    datapoint = vflip(datapoint, i)
            return datapoint
        for i in range(len(datapoint.images)):
            if random.random() < self.p:
                datapoint = vflip(datapoint, i)
        return datapoint


class RandomRot90:
    """Random exact 90/180/270-degree rotation (subset of the D4 group)."""

    def __init__(self, consistent_transform, p=0.75) -> None:
        self.p = p
        self.consistent_transform = consistent_transform

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                k = random.choice((1, 2, 3))
                for i in range(len(datapoint.images)):
                    datapoint = rot90(datapoint, i, k)
            return datapoint
        for i in range(len(datapoint.images)):
            if random.random() < self.p:
                k = random.choice((1, 2, 3))
                datapoint = rot90(datapoint, i, k)
        return datapoint


# ── instantiation helpers ────────────────────────────────────────────────────

def _instantiate(cfg: dict):
    """Instantiate a single `_target_` config exactly like the SAM3 trainer."""
    add_support_to_pythonpath()
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    return instantiate(OmegaConf.create(cfg), _convert_="all")


def _decode_only_transforms() -> list:
    """Transform list that only decodes RLE masks (no geometry), instantiated.

    DecodeRle turns each `obj.segment` RLE into a uint8 (H, W) mask tensor and
    must precede any geometric op. We deliberately skip resize/pad/tensor/normalize
    so the returned datapoint keeps a PIL image and absolute-XYXY boxes.
    """
    return [_instantiate({"_target_": "sam3.train.transforms.segmentation.DecodeRle"})]


def configured_geometric_ops(
    *,
    respect_boxes: bool = True,
    force: bool = False,
    include_resize: bool = True,
    resolution: int = RESOLUTION,
    crop_min: int = CROP_MIN_SIZE,
    crop_max: int = CROP_MAX_SIZE,
) -> list[tuple[str, Any]]:
    """Ordered (label, transform) list inserted AFTER DecodeRle, BEFORE resize.

    Phase 1 covers the shipped datapoint-aware ops only (crop, hflip, colorjitter).
    `force=True` sets flip/colorjitter probabilities to 1.0 so the geometric effect
    is always visible for the per-op alignment check; `force=False` uses the real
    p=0.5 schedule for a realistic gallery.
    """
    hflip_p = 1.0 if force else HFLIP_P
    color_p = 1.0 if force else COLOR_P
    ops: list[tuple[str, Any]] = []

    ops.append((
        f"crop[{crop_min},{crop_max}] respect_boxes={respect_boxes}",
        _instantiate({
            "_target_": "sam3.train.transforms.basic_for_api.RandomSizeCropAPI",
            "min_size": crop_min,
            "max_size": crop_max,
            "respect_boxes": respect_boxes,
            "consistent_transform": CONSISTENT_TRANSFORM,
        }),
    ))
    ops.append((
        f"hflip(p={hflip_p:g})",
        _instantiate({
            "_target_": "sam3.train.transforms.basic_for_api.RandomHorizontalFlip",
            "p": hflip_p,
            "consistent_transform": CONSISTENT_TRANSFORM,
        }),
    ))
    ops.append((
        f"colorjitter(p={color_p:g})",
        _instantiate({
            "_target_": "sam3.train.transforms.basic_for_api.RandomSelectAPI",
            "p": color_p,
            "transforms1": {
                "_target_": "sam3.train.transforms.basic_for_api.ColorJitter",
                "consistent_transform": CONSISTENT_TRANSFORM,
                **COLOR_JITTER,
            },
            "transforms2": {"_target_": "sam3.train.transforms.basic_for_api.IdentityAPI"},
        }),
    ))
    if include_resize:
        ops.append((
            f"resize->{resolution} (square)",
            _instantiate({
                "_target_": "sam3.train.transforms.basic_for_api.RandomResizeAPI",
                "sizes": resolution,
                "square": True,
                "consistent_transform": CONSISTENT_TRANSFORM,
            }),
        ))
    return ops


# ── dataset construction ─────────────────────────────────────────────────────

def build_decoded_dataset(data_root: str | Path, split: str = "train"):
    """Build a `Sam3ImageDataset` whose transforms only decode RLE masks.

    Calling `dataset[i]` returns a `Datapoint` with a PIL image, uint8 mask
    tensors, and absolute-XYXY boxes — the clean starting point for applying the
    candidate geometric ops with per-op snapshots.
    """
    add_support_to_pythonpath()
    from sam3.train.data.sam3_image_dataset import Sam3ImageDataset

    data_root = Path(data_root)
    ann_name = "train_annotations.coco.json" if split == "train" else "test_annotations.coco.json"
    ann_file = data_root / "annotations" / ann_name
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_file}")

    return Sam3ImageDataset(
        img_folder=str(data_root),
        ann_file=str(ann_file),
        transforms=_decode_only_transforms(),
        load_segmentation=True,
        max_ann_per_img=MAX_ANN_PER_IMG,
        multiplier=1,
        max_train_queries=MAX_TRAIN_QUERIES,
        max_val_queries=MAX_TRAIN_QUERIES,
        training=(split == "train"),
        use_caching=False,
    )


# ── datapoint -> displayable arrays ──────────────────────────────────────────

def _image_to_uint8(data) -> np.ndarray:
    """Return an (H, W, 3) uint8 array from a PIL image or a CHW tensor."""
    from PIL import Image as PILImage

    if isinstance(data, PILImage.Image):
        return np.asarray(data.convert("RGB"))
    import torch

    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().float()
        if arr.ndim == 3:
            arr = arr.permute(1, 2, 0)
        arr = arr.numpy()
        if arr.max() <= 1.5:  # likely normalized/0-1 float
            arr = (arr - arr.min()) / (np.ptp(arr) + 1e-6)
            arr = arr * 255.0
        return arr.clip(0, 255).astype(np.uint8)
    return np.asarray(data)


def datapoint_to_arrays(datapoint, query_index: int = 0) -> dict[str, Any]:
    """Extract image, per-instance masks, abs-XYXY boxes, and point prompts.

    Only objects with a non-empty mask are returned (so out-of-crop objects with
    zeroed masks/boxes are dropped from the overlay, matching what downstream
    FilterEmptyTargets removes).
    """
    import torch

    image = datapoint.images[0]
    img = _image_to_uint8(image.data)
    H, W = img.shape[:2]

    masks: list[np.ndarray] = []
    boxes: list[list[float]] = []
    for obj in image.objects:
        seg = obj.segment
        if seg is None:
            continue
        if isinstance(seg, torch.Tensor):
            m = seg.detach().cpu().numpy()
        else:
            m = np.asarray(seg)
        m = m.astype(bool)
        if m.ndim != 2 or m.sum() == 0:
            continue
        masks.append(m)
        bb = obj.bbox
        bb = bb.detach().cpu().view(-1).tolist() if isinstance(bb, torch.Tensor) else list(bb)
        boxes.append([float(v) for v in bb[:4]])

    points = None
    if datapoint.find_queries and query_index < len(datapoint.find_queries):
        q = datapoint.find_queries[query_index]
        if getattr(q, "input_points", None) is not None:
            pts = q.input_points
            pts = pts.detach().cpu().view(-1, 3).numpy() if isinstance(pts, torch.Tensor) else np.asarray(pts).reshape(-1, 3)
            points = pts

    masks_arr = np.stack(masks, axis=0) if masks else np.zeros((0, H, W), dtype=bool)
    boxes_arr = np.asarray(boxes, dtype=float) if boxes else np.zeros((0, 4), dtype=float)
    return {"image": img, "masks": masks_arr, "boxes": boxes_arr, "points": points, "size": (H, W)}


# ── apply ops with per-op snapshots ──────────────────────────────────────────

def apply_with_snapshots(datapoint, ops, *, seed: int = 0, query_index: int = 0):
    """Apply `ops` sequentially (as training does), snapshotting after each op.

    Returns a list of (label, arrays) where the first entry is the decoded input.
    Pixel data in each snapshot is copied, so later in-place mutations don't alias.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    stages = [("decoded (2048 input)", datapoint_to_arrays(datapoint, query_index))]
    dp = datapoint
    for label, op in ops:
        dp = op(dp, epoch=0)
        stages.append((label, datapoint_to_arrays(dp, query_index)))
    return stages


# ── plotting ─────────────────────────────────────────────────────────────────

def _overlay_masks(ax, masks: np.ndarray, alpha: float = 0.5) -> None:
    import matplotlib.pyplot as plt

    if masks.shape[0] == 0:
        return
    cmap = plt.get_cmap("tab20")
    H, W = masks.shape[-2:]
    overlay = np.zeros((H, W, 4), dtype=float)
    for i, m in enumerate(masks):
        overlay[m] = cmap(i % 20)
    overlay[..., 3] = (overlay[..., :3].sum(-1) > 0).astype(float) * alpha
    ax.imshow(overlay)


def plot_stages(stages, *, suptitle: str = "", show_boxes: bool = True, show_points: bool = True):
    """Render one panel per op with mask + box (+ point) overlays for alignment."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (label, arr) in zip(axes, stages):
        ax.imshow(arr["image"])
        _overlay_masks(ax, arr["masks"], alpha=0.5)
        if show_boxes:
            for x1, y1, x2, y2 in arr["boxes"]:
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor="red", linewidth=0.8))
        if show_points and arr["points"] is not None and len(arr["points"]):
            ax.scatter(arr["points"][:, 0], arr["points"][:, 1], s=18, c="yellow",
                       edgecolors="black", linewidths=0.5, zorder=5)
        H, W = arr["size"]
        ax.set_title(f"{label}\n{W}x{H}px  n_obj={arr['masks'].shape[0]}", fontsize=9)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


# ── quantitative checks ──────────────────────────────────────────────────────

def crop_scale_report(dataset, *, n_samples: int = 12, seed: int = 0,
                      resolution: int = RESOLUTION) -> dict[str, Any]:
    """Compare realized crop sizes for respect_boxes True vs False over N samples.

    For each sample we apply ONLY the crop op to the decoded datapoint and record
    the resulting (h, w). This empirically reveals whether `respect_boxes=True`
    forces near-full-image crops (which would re-introduce the 2048->1008
    downsampling and defeat the resolution-recovery goal).
    """
    rng = random.Random(seed)
    n = len(dataset)
    idxs = rng.sample(range(n), min(n_samples, n))

    out: dict[str, Any] = {"resolution": resolution, "n_samples": len(idxs), "variants": {}}
    for respect in (True, False):
        sizes_wh: list[tuple[int, int]] = []
        n_obj_after: list[int] = []
        crop_op = _instantiate({
            "_target_": "sam3.train.transforms.basic_for_api.RandomSizeCropAPI",
            "min_size": CROP_MIN_SIZE,
            "max_size": CROP_MAX_SIZE,
            "respect_boxes": respect,
            "consistent_transform": CONSISTENT_TRANSFORM,
        })
        for idx in idxs:
            dp = dataset[idx]
            dp = crop_op(dp, epoch=0)
            h, w = dp.images[0].size
            sizes_wh.append((int(w), int(h)))
            n_obj_after.append(sum(
                1 for obj in dp.images[0].objects
                if obj.segment is not None and int(np.asarray(_as_np(obj.segment)).sum()) > 0
            ))
        ws = np.array([s[0] for s in sizes_wh], float)
        hs = np.array([s[1] for s in sizes_wh], float)
        sides = np.concatenate([ws, hs])
        out["variants"][f"respect_boxes={respect}"] = {
            "crop_w": _stats(ws),
            "crop_h": _stats(hs),
            "scale_factor_to_resolution": _stats(resolution / sides),
            "mean_n_obj_after_crop": float(np.mean(n_obj_after)),
            "full_image_crop_frac": float(np.mean(sides >= 2048 - 1)),
        }
    out["baseline_scale_factor_2048_to_res"] = round(resolution / 2048.0, 4)
    return out


def small_crown_pixel_gain(dataset, *, n_samples: int = 12, seed: int = 0,
                           resolution: int = RESOLUTION, small_area_px: float = 32 * 32) -> dict[str, Any]:
    """Mean mask-area ratio (crop-path 1008 vs whole-image 1008) for small crowns.

    Path A (baseline): resize the whole 2048 image to `resolution` (square) and
    measure each small object's mask area. Path B (configured crop): crop then resize to
    `resolution`. Ratio > 1 means small crowns occupy more pixels under the crop
    recipe (the mechanism that should lift small-crown recall).
    """
    rng = random.Random(seed)
    n = len(dataset)
    idxs = rng.sample(range(n), min(n_samples, n))

    resize_full = configured_geometric_ops(include_resize=True)[-1][1]
    ratios: list[float] = []
    for idx in idxs:
        base = dataset[idx]
        # native-frame small-object areas (decoded, 2048 frame)
        native_small = _small_object_areas(base, small_area_px)
        if not native_small:
            continue
        # Path A: whole-image resize to resolution.
        dp_a = resize_full(copy.deepcopy(base), epoch=0)
        a_areas = _object_area_by_index(dp_a)
        # Path B: crop (respect_boxes=False so we actually downscale less) + resize.
        ops_b = configured_geometric_ops(respect_boxes=False, force=False, include_resize=True)
        dp_b = copy.deepcopy(base)
        for _, op in ops_b:
            dp_b = op(dp_b, epoch=0)
        b_areas = _object_area_by_index(dp_b)
        for oi in native_small:
            a = a_areas.get(oi, 0.0)
            b = b_areas.get(oi, 0.0)
            if a > 0 and b > 0:
                ratios.append(b / a)
    return {
        "n_samples": len(idxs),
        "n_small_objects_compared": len(ratios),
        "mean_area_ratio_cropB_over_baselineA": float(np.mean(ratios)) if ratios else float("nan"),
        "median_area_ratio": float(np.median(ratios)) if ratios else float("nan"),
        "note": "ratio>1 => small crowns gain pixels under the crop recipe",
    }


# ── D4 single-image stub (vflip + rot90) ──────────────────────────────────────

def vflip_box_xyxy(boxes: np.ndarray, h: int) -> np.ndarray:
    """Vertical flip remap (mirror of shipped hflip): [x1,y1,x2,y2]->[x1,h-y2,x2,h-y1]."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    return np.stack([b[:, 0], h - b[:, 3], b[:, 2], h - b[:, 1]], axis=1)


def vflip_points_xy(points: np.ndarray, h: int) -> np.ndarray:
    """Vertical flip point remap: (x,y)->(x,h-y) (label preserved)."""
    p = np.asarray(points, float).reshape(-1, 3).copy()
    p[:, 1] = h - p[:, 1]
    return p


def rot90_box_xyxy(boxes: np.ndarray, k: int, h: int, w: int) -> tuple[np.ndarray, int, int]:
    """Remap XYXY boxes under torch.rot90(k) (CCW). Returns (boxes, new_h, new_w).

    k in {1,2,3}. 90/270 swap H<->W. Continuous-coordinate maps (mirror hflip's
    use of the full extent, not extent-1) so a k=4 round-trip is exact:
      k=1 (CCW): (x,y)->(y, W-x)     box->[y1, W-x2, y2, W-x1]   out (W,H)
      k=2 (180): (x,y)->(W-x, H-y)   box->[W-x2, H-y2, W-x1, H-y1] out (H,W)
      k=3 (CW):  (x,y)->(H-y, x)     box->[H-y2, x1, H-y1, x2]   out (W,H)
    """
    b = np.asarray(boxes, float).reshape(-1, 4)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    k = k % 4
    if k == 0:
        return b.copy(), h, w
    if k == 1:
        return np.stack([y1, w - x2, y2, w - x1], axis=1), w, h
    if k == 2:
        return np.stack([w - x2, h - y2, w - x1, h - y1], axis=1), h, w
    return np.stack([h - y2, x1, h - y1, x2], axis=1), w, h


def rot90_points_xy(points: np.ndarray, k: int, h: int, w: int) -> tuple[np.ndarray, int, int]:
    """Remap (x,y,label) points under torch.rot90(k) (CCW). Returns (points,new_h,new_w)."""
    p = np.asarray(points, float).reshape(-1, 3).copy()
    x, y = p[:, 0].copy(), p[:, 1].copy()
    k = k % 4
    if k == 0:
        return p, h, w
    if k == 1:
        p[:, 0], p[:, 1] = y, w - x
        return p, w, h
    if k == 2:
        p[:, 0], p[:, 1] = w - x, h - y
        return p, h, w
    p[:, 0], p[:, 1] = h - y, x
    return p, w, h


def stub_d4_single_image(image: np.ndarray, box: Optional[list[float]] = None,
                         point: Optional[list[float]] = None):
    """Apply vflip + rot90(k=1,2,3) to ONE image (torchvision/torch.rot90) with a
    synthetic box+point, using the exact remap math the datapoint-aware ops above
    implement.

    This is a concept stub for the geometric correctness of the D4 ops; the
    datapoint-aware (mask/box/point) versions are ``vflip``/``rot90`` above.
    """
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.patches import Rectangle

    img = np.asarray(image)
    H, W = img.shape[:2]
    if box is None:
        box = [W * 0.12, H * 0.10, W * 0.42, H * 0.30]  # asymmetric so rotation is obvious
    if point is None:
        point = [W * 0.27, H * 0.20, 1.0]

    t = torch.from_numpy(img).permute(2, 0, 1)  # CHW

    panels: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    panels.append(("original", img, np.asarray(box, float).reshape(1, 4), np.asarray(point, float).reshape(1, 3)))

    # vertical flip
    v_img = torch.flip(t, dims=[1]).permute(1, 2, 0).numpy()
    panels.append(("vflip", v_img, vflip_box_xyxy([box], H), vflip_points_xy([point], H)))

    # rot90 k=1,2,3 (CCW)
    for k in (1, 2, 3):
        r_img = torch.rot90(t, k, dims=(1, 2)).permute(1, 2, 0).numpy()
        rb, _, _ = rot90_box_xyxy([box], k, H, W)
        rp, _, _ = rot90_points_xy([point], k, H, W)
        panels.append((f"rot90 k={k}", r_img, rb, rp))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (label, im, bx, pt) in zip(axes, panels):
        ax.imshow(im)
        x1, y1, x2, y2 = bx[0]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=1.5))
        ax.scatter(pt[:, 0], pt[:, 1], s=40, c="yellow", edgecolors="black", zorder=5)
        ax.set_title(f"{label}\n{im.shape[1]}x{im.shape[0]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("D4 stub (single image): box/point follow the image under vflip + rot90", fontsize=11)
    fig.tight_layout()
    return fig


# ── orchestrator ─────────────────────────────────────────────────────────────

def phase1_report(data_root: str | Path | None = None, *, split: str = "train",
                  n_samples: int = 10, seed: int = 0):
    """Run the full Phase-1 inspection and return figures + a printable summary.

    Shows BOTH respect_boxes settings (False = recommended ~1024 crops vs
    True = recorded decision) for the per-op alignment and the gallery, so a
    single run is decisive. Returns a dict with keys 'crop_scale',
    'small_crown_gain', 'figures' (a list of (title, matplotlib Figure)).
    Nothing is trained.
    """
    if data_root is None:
        data_root = DATA_DIR / "oam_tcd_instance_coco"
    dataset = build_decoded_dataset(data_root, split=split)

    figures: list[tuple[str, Any]] = []
    rng = random.Random(seed)
    gallery_first_img = None

    # (1) Per-op alignment on ONE sample, for both respect_boxes settings, ops FORCED on.
    align_idx = rng.sample(range(len(dataset)), 1)[0]
    for respect in (False, True):
        dp = dataset[align_idx]
        ops = configured_geometric_ops(respect_boxes=respect, force=True, include_resize=True)
        stages = apply_with_snapshots(dp, ops, seed=seed)
        figures.append((
            f"per-op alignment respect_boxes={respect} (idx={align_idx})",
            plot_stages(stages, suptitle=(
                f"Per-op alignment, idx={align_idx}, respect_boxes={respect} "
                f"(ops forced on) -- masks/boxes/points must track the image")),
        ))

    # (2) Realistic gallery (final 1008px frame, p=0.5) for both respect_boxes settings.
    gal_idxs = rng.sample(range(len(dataset)), min(n_samples, len(dataset)))
    for respect in (False, True):
        items = []
        for j, idx in enumerate(gal_idxs):
            dp = dataset[idx]
            ops = configured_geometric_ops(respect_boxes=respect, force=False, include_resize=True)
            stages = apply_with_snapshots(dp, ops, seed=seed + j)
            items.append((f"idx={idx}", stages[-1][1]))
        if gallery_first_img is None and items:
            gallery_first_img = items[0][1]["image"]
        figures.append((f"gallery respect_boxes={respect}", _plot_gallery(items, respect)))

    # (3) Quantitative checks.
    crop_scale = crop_scale_report(dataset, n_samples=max(n_samples, 12), seed=seed)
    gain = small_crown_pixel_gain(dataset, n_samples=max(n_samples, 12), seed=seed)

    # (4) D4 stub on a real gallery image.
    if gallery_first_img is None:
        gallery_first_img = np.zeros((1008, 1008, 3), np.uint8)
    figures.append(("D4 stub (vflip + rot90)", stub_d4_single_image(gallery_first_img)))

    _print_summary(crop_scale, gain)
    return {"crop_scale": crop_scale, "small_crown_gain": gain, "figures": figures}


# ── Phase 2: verify the REAL config pipeline (incl. project-owned D4) ─────────

_AUG_TARGETS = (
    "RandomSizeCropAPI", "RandomHorizontalFlip",
    "RandomVerticalFlip", "RandomRot90", "RandomSelectAPI",
)


def _label_for(short: str, d: dict) -> str:
    if short == "RandomSizeCropAPI":
        return (f"crop[{d.get('min_size')},{d.get('max_size')}] "
                f"respect={d.get('respect_boxes')}/{d.get('respect_input_boxes', True)}")
    if short in ("RandomHorizontalFlip", "RandomVerticalFlip", "RandomRot90"):
        name = {"RandomHorizontalFlip": "hflip", "RandomVerticalFlip": "vflip",
                "RandomRot90": "rot90"}[short]
        return f"{name}(p={float(d.get('p', 0.5)):g})"
    if short == "RandomSelectAPI":
        inner = d.get("transforms1", {})
        itgt = inner.get("_target_", "?").rsplit(".", 1)[-1] if isinstance(inner, dict) else "?"
        return f"{itgt.lower()}(p={float(d.get('p', 0.5)):g})"
    if short == "RandomResizeAPI":
        return f"resize->{RESOLUTION}"
    return short


def config_train_ops(*, force: bool = False, include_resize: bool = False) -> list[tuple[str, Any]]:
    """Build labeled (label, transform) pairs from the REAL `Core/config.py`
    train pipeline (`config._train_transforms()`), so the Phase-1 snapshot
    machinery can verify the configured augmentations -- including the
    project-owned D4 ``augmentation.RandomVerticalFlip`` / ``RandomRot90`` --
    IN-PIPELINE. This is the single source of truth: identical ``_target_`` +
    kwargs to what training instantiates; only transform PROBABILITIES are
    overridden to 1.0 when ``force=True`` so each geometric effect is visible.
    """
    import sys

    from project_paths import CORE_DIR
    if str(CORE_DIR) not in sys.path:  # so `augmentation.*` _target_ resolves
        sys.path.insert(0, str(CORE_DIR))

    import config as cfg
    compose = cfg._train_transforms()[0]
    op_dicts = compose["transforms"] if isinstance(compose, dict) else []

    ops: list[tuple[str, Any]] = []
    for d in op_dicts:
        if not isinstance(d, dict) or "_target_" not in d:
            continue
        short = str(d["_target_"]).rsplit(".", 1)[-1]
        keep = short in _AUG_TARGETS or (include_resize and short == "RandomResizeAPI")
        if not keep:
            continue
        d = copy.deepcopy(d)
        if force and "p" in d:
            d["p"] = 1.0
        ops.append((_label_for(short, d), _instantiate(d)))
    return ops


def d4_invariants(dataset, *, n_samples: int = 6, seed: int = 0) -> dict[str, Any]:
    """Decisive correctness check for the project-owned D4 transforms.

    vflip and rot90(k) permute pixels, so they must preserve EACH object's mask
    pixel count exactly and its bbox area (rot90 swaps w<->h). We apply the
    low-level ``vflip``/``rot90`` ops directly (the exact math training uses,
    defined earlier in this module) and report the worst-case deltas vs the
    decoded datapoint.
    """
    def _areas(dp):
        out = []
        for obj in dp.images[0].objects:
            if obj.segment is None:
                continue
            m = _as_np(obj.segment).astype(bool)
            if m.ndim != 2 or m.sum() == 0:
                continue
            bb = _as_np(obj.bbox).reshape(-1)[:4]
            out.append((float(m.sum()), float(abs(bb[2] - bb[0]) * abs(bb[3] - bb[1]))))
        return out

    rng = random.Random(seed)
    idxs = rng.sample(range(len(dataset)), min(n_samples, len(dataset)))
    cases = {
        "vflip": lambda dp: vflip(dp, 0),
        "rot90_k1": lambda dp: rot90(dp, 0, 1),
        "rot90_k2": lambda dp: rot90(dp, 0, 2),
        "rot90_k3": lambda dp: rot90(dp, 0, 3),
    }
    report: dict[str, Any] = {}
    for name, fn in cases.items():
        max_mask_delta = 0.0
        max_boxarea_reldelta = 0.0
        n_obj = 0
        for idx in idxs:
            base = _areas(dataset[idx])
            after = _areas(fn(copy.deepcopy(dataset[idx])))
            if len(base) != len(after):  # geometry must not drop/add objects
                report[name] = {"ok": False, "reason": "object count changed",
                                "n_before": len(base), "n_after": len(after)}
                break
            for (a0, ba0), (a1, ba1) in zip(base, after):
                n_obj += 1
                max_mask_delta = max(max_mask_delta, abs(a0 - a1))
                if ba0 > 0:
                    max_boxarea_reldelta = max(max_boxarea_reldelta, abs(ba0 - ba1) / ba0)
        else:
            report[name] = {
                "ok": bool(max_mask_delta == 0.0 and max_boxarea_reldelta < 1e-3),
                "n_obj": n_obj,
                "max_mask_area_delta_px": max_mask_delta,
                "max_boxarea_rel_delta": max_boxarea_reldelta,
            }
    report["all_ok"] = all(v.get("ok") for v in report.values() if isinstance(v, dict))
    return report


def phase2_verify(data_root: str | Path | None = None, *, split: str = "train",
                  n_samples: int = 6, seed: int = 0):
    """Re-run the inspection against the REAL `config._train_transforms()`.

    Confirms the configured pipeline -- crop(respect_boxes/input=False) + full D4
    (hflip/vflip/rot90) + p-gated ColorJitter -- aligns masks/boxes/points on
    real OAM-TCD datapoints, with the project-owned D4 ops exercised in-pipeline.
    Returns {'figures', 'op_labels', 'd4_invariants'}. Nothing is trained.
    """
    if data_root is None:
        data_root = DATA_DIR / "oam_tcd_instance_coco"
    dataset = build_decoded_dataset(data_root, split=split)

    figures: list[tuple[str, Any]] = []
    rng = random.Random(seed)

    # (1) Per-op alignment on ONE sample, REAL config ops forced on (D4 visible).
    align_idx = rng.sample(range(len(dataset)), 1)[0]
    ops = config_train_ops(force=True, include_resize=True)
    stages = apply_with_snapshots(dataset[align_idx], ops, seed=seed)
    figures.append((
        f"phase2 real-config per-op alignment (idx={align_idx})",
        plot_stages(stages, suptitle=(
            f"REAL config _train_transforms, idx={align_idx} (ops forced on) -- "
            "masks/boxes/points must track image through crop+D4+colorjitter")),
    ))

    # (2) Realistic gallery under the real p-schedule (final 1008px frame).
    gal_idxs = rng.sample(range(len(dataset)), min(n_samples, len(dataset)))
    items = []
    for j, idx in enumerate(gal_idxs):
        ops = config_train_ops(force=False, include_resize=True)
        stages = apply_with_snapshots(dataset[idx], ops, seed=seed + j)
        items.append((f"idx={idx}", stages[-1][1]))
    figures.append(("phase2 real-config gallery", _plot_gallery(items)))

    # (3) D4 invariants (decisive area-preservation check on real datapoints).
    inv = d4_invariants(dataset, n_samples=max(n_samples, 6), seed=seed)
    chain = [lbl for lbl, _ in config_train_ops(force=False, include_resize=True)]
    _print_phase2(chain, inv)
    return {"figures": figures, "op_labels": chain, "d4_invariants": inv}


def _print_phase2(op_labels: list[str], inv: dict) -> None:
    print("\n================ PHASE 2: REAL CONFIG PIPELINE ==================")
    print("train _train_transforms() augmentation chain (post-DecodeRle):")
    for i, lbl in enumerate(op_labels, 1):
        print(f"  {i}. {lbl}")
    print("\n[D4 invariants: vflip/rot90 must preserve mask area & box area]")
    for name in ("vflip", "rot90_k1", "rot90_k2", "rot90_k3"):
        v = inv.get(name)
        if not isinstance(v, dict):
            continue
        if "reason" in v:
            print(f"  {name:9s}: FAIL ({v['reason']}: {v.get('n_before')}->{v.get('n_after')})")
            continue
        flag = "OK  " if v["ok"] else "FAIL"
        print(f"  {name:9s}: {flag}  n_obj={v['n_obj']}  "
              f"max_mask_area_delta={v['max_mask_area_delta_px']:.0f}px  "
              f"max_boxarea_rel_delta={v['max_boxarea_rel_delta']:.2e}")
    print(f"\n  ALL D4 INVARIANTS OK: {inv.get('all_ok')}")
    print("==================================================================\n")


def _plot_gallery(items, respect_boxes: bool | None = None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(items)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (label, arr) in zip(axes, items):
        ax.imshow(arr["image"])
        _overlay_masks(ax, arr["masks"], alpha=0.5)
        for x1, y1, x2, y2 in arr["boxes"]:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=0.6))
        H, W = arr["size"]
        ax.set_title(f"{label}  {W}x{H}  n={arr['masks'].shape[0]}", fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    suffix = "" if respect_boxes is None else f"  [respect_boxes={respect_boxes}]"
    fig.suptitle(f"Configured augmentation, final 1008-pixel frame (p=0.5 schedule){suffix}", fontsize=11)
    fig.tight_layout()
    return fig


# ── small utilities ──────────────────────────────────────────────────────────

def _as_np(seg):
    import torch

    return seg.detach().cpu().numpy() if isinstance(seg, torch.Tensor) else np.asarray(seg)


def _stats(a: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, float)
    return {"min": float(a.min()), "mean": float(a.mean()), "max": float(a.max())}


def _small_object_areas(datapoint, small_area_px: float) -> list[int]:
    """Indices of objects whose decoded mask area is below `small_area_px`."""
    out = []
    for i, obj in enumerate(datapoint.images[0].objects):
        if obj.segment is None:
            continue
        area = float(_as_np(obj.segment).sum())
        if 0 < area < small_area_px:
            out.append(i)
    return out


def _object_area_by_index(datapoint) -> dict[int, float]:
    out: dict[int, float] = {}
    for i, obj in enumerate(datapoint.images[0].objects):
        if obj.segment is None:
            continue
        out[i] = float(_as_np(obj.segment).sum())
    return out


def _print_summary(crop_scale: dict, gain: dict) -> None:
    print("\n================ PHASE 1: AUGMENTATION INSPECTION ================")
    print(f"baseline scale factor (2048 -> {crop_scale['resolution']}): "
          f"{crop_scale['baseline_scale_factor_2048_to_res']}  (configured target is approximately 1.0)")
    for variant, v in crop_scale["variants"].items():
        sf = v["scale_factor_to_resolution"]
        print(f"\n[{variant}]  (n={crop_scale['n_samples']})")
        print(f"  crop_w: {v['crop_w']}")
        print(f"  crop_h: {v['crop_h']}")
        print(f"  scale_factor_to_resolution: min={sf['min']:.3f} mean={sf['mean']:.3f} max={sf['max']:.3f}")
        print(f"  mean_n_obj_after_crop: {v['mean_n_obj_after_crop']:.1f}")
        print(f"  full_image_crop_frac (crop hit 2048): {v['full_image_crop_frac']*100:.1f}%")
    print("\n[small-crown pixel gain]")
    print(f"  small objects compared: {gain['n_small_objects_compared']}")
    print(f"  mean area ratio (cropB / baselineA): {gain['mean_area_ratio_cropB_over_baselineA']:.3f}")
    print(f"  ({gain['note']})")
    print("==================================================================\n")
