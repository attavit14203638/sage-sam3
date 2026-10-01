#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Instance-relative boundary bands for Boundary IoU / Boundary AP.

Boundary IoU (Cheng et al., CVPR 2021) scores the agreement of two masks' inner
boundary bands, where the band width is ``dilation_ratio * image_diagonal`` -- a
single width for every instance in an image. On COCO that is reasonable, because
objects occupy a large and fairly uniform fraction of the frame. On aerial tree
crowns it is not: a 2048px tile holds crowns from ~16px to ~200px across, so one
width cannot be a thin rind for all of them.

In the matched final OAM-TCD individual-tree evaluation, the image-relative band's
Boundary AP / Mask AP retention is 1.00, 1.00, and 0.99 for small, medium, and large
crowns. A uniform 8-pixel band instead retains 0.91, 0.25, and 0.09. One image-scale
width therefore cannot remain boundary-sensitive across the evaluated crown sizes.

This module makes the band **instance-relative** instead:

    band_width = k * equivalent_diameter,   equivalent_diameter = 2 * sqrt(area / pi)

`k = 0.19` was fixed before the final model comparison. At that value, the matched
final retention ratios are 0.35, 0.49, and 0.67 for Unified-Prompt SAM3 and remain
inside the declared usable window for Mask R-CNN as well. The preliminary calibration
sweep used an unsplit boundary population with tree-split Mask AP denominators, so its
exact half-retention estimates are not publication claims.

**This is a reparameterisation, not a new metric.** The band construction, the
`min(MaskIoU, BoundaryIoU)` scoring and the COCO AP protocol are all unchanged; only
the rule for choosing the width differs. Papers reporting this must also report the
literature band alongside, so no comparison to published numbers is silently broken.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import numpy as np

__all__ = [
    "BOUNDARY_BAND_K",
    "equivalent_diameter",
    "instance_band_px",
    "instance_relative_dilation",
    "make_instance_relative_mask_to_boundary",
    "patched_instance_relative_bands",
    "per_instance_boundary_iou",
]

# Fixed at 0.19 before the final model comparison. Matched final individual-tree evaluations keep every size-bin retention ratio inside the declared usable window for Unified-Prompt SAM3 and Mask R-CNN; exact preliminary calibration estimates are not reported because the sweep combined different populations.
BOUNDARY_BAND_K = 0.19


def equivalent_diameter(area: float) -> float:
    """Diameter of a disc with the same area, in the units `area` is measured in.

    The scale a band is proportional to must be a property of the instance and not of
    its bounding box, because crowns are irregular and a box diagonal would make the
    band depend on rotation.
    """
    return 2.0 * math.sqrt(max(float(area), 0.0) / math.pi)


def instance_band_px(area: float, k: float = BOUNDARY_BAND_K) -> float:
    """Band width in pixels for one instance. Continuous; not yet quantised."""
    return k * equivalent_diameter(area)


def instance_relative_dilation(mask: np.ndarray, k: float = BOUNDARY_BAND_K) -> int:
    """Erosion iterations for `mask`'s own scale, floored at 1.

    Upstream's `mask_to_boundary` erodes with a 3x3 kernel `dilation` times, so the
    band width is an integer number of pixels and the value must be at least 1 -- a
    zero-width band would make Boundary IoU identically 1.0 for every prediction and
    silently report a perfect score.
    """
    return max(1, int(round(instance_band_px(float(np.count_nonzero(mask)), k))))


def make_instance_relative_mask_to_boundary(
    k: float = BOUNDARY_BAND_K,
) -> Callable[..., np.ndarray]:
    """Build a drop-in replacement for `boundary_iou`'s `mask_to_boundary`.

    Deliberately reimplements the upstream body rather than calling it with a
    per-instance ratio. Upstream derives `dilation` from the *image* diagonal, so
    passing a per-instance ratio would require inverting that derivation for every
    mask and would silently break the moment upstream changed how the diagonal is
    computed. Reimplementing is ~10 lines and the erosion semantics are then explicit.

    Signature matches upstream, including the ignored `dilation_ratio`, so it can be
    substituted without touching call sites. `dilation_ratio` is accepted and
    discarded: the whole point is that the width no longer comes from the image.
    """

    def instance_relative_mask_to_boundary(
        mask: np.ndarray, dilation_ratio: float = 0.02, **_: Any
    ) -> np.ndarray:
        mask = np.ascontiguousarray(mask).astype(np.uint8)
        height, width = mask.shape[:2]
        dilation = instance_relative_dilation(mask, k)

        # Border padding matches upstream: without it, a crown clipped by the tile edge
        # would have its clipped side counted as interior rather than boundary, which
        # matters here because tiled inference puts many crowns against a tile edge.
        padded = np.pad(mask, 1, mode="constant", constant_values=0)
        eroded = _erode(padded, iterations=dilation)[1 : height + 1, 1 : width + 1]
        return (mask - eroded).astype(np.uint8)

    instance_relative_mask_to_boundary.band_k = float(k)  # type: ignore[attr-defined]
    return instance_relative_mask_to_boundary


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    """3x3 binary erosion, `iterations` times. cv2 if present, else scipy.

    cv2 is what upstream uses and is faster, but it is not a declared dependency of
    this project, so scipy -- which is pinned in requirements.txt -- is the fallback.
    Both are exact 8-connected erosions, so the result is identical either way.
    """
    try:
        import cv2

        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.erode(mask, kernel, iterations=iterations)
    except ImportError:
        from scipy.ndimage import binary_erosion

        structure = np.ones((3, 3), dtype=bool)
        return binary_erosion(
            mask.astype(bool), structure=structure, iterations=iterations, border_value=0
        ).astype(np.uint8)


@contextmanager
def patched_instance_relative_bands(k: float = BOUNDARY_BAND_K) -> Iterator[None]:
    """Temporarily make `boundary_iou` use instance-relative bands.

    Patches BOTH the defining module attribute and every already-imported binding of
    `mask_to_boundary`. Patching only the module is the failure this project has already
    hit once: a module that did `from ... import mask_to_boundary` at import time keeps
    its own reference and would silently continue using the image-relative width, which
    is exactly the kind of half-applied patch that produces a plausible but wrong number.
    See the `point_sample` patch in `eval_overrides.py` for the same pattern.

    Restores every binding on exit, including on exception, so a failed evaluation
    cannot leave the process computing a different metric than it reports.
    """
    import importlib

    replacement = make_instance_relative_mask_to_boundary(k)

    # Modules that define or re-export the symbol. Missing ones are skipped rather than
    # raising: the package layout has changed between upstream versions, and a patch
    # that covers three of four locations is more dangerous than one that fails loudly.
    candidates = (
        "boundary_iou.utils.boundary_utils",
        "boundary_iou.coco_instance_api.coco",
        "boundary_iou.coco_instance_api.cocoeval",
    )
    patched: list[tuple[Any, str, Any]] = []
    found_definition = False

    for name in candidates:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if not hasattr(module, "mask_to_boundary"):
            continue
        if name.endswith("boundary_utils"):
            found_definition = True
        patched.append((module, "mask_to_boundary", module.mask_to_boundary))
        setattr(module, "mask_to_boundary", replacement)

    if not found_definition:
        for module, attr, original in patched:
            setattr(module, attr, original)
        raise RuntimeError(
            "boundary_iou.utils.boundary_utils.mask_to_boundary was not found, so the "
            "instance-relative band could not be installed. Refusing to continue: the "
            "evaluation would silently run with the image-relative band and report it "
            "as instance-relative."
        )

    try:
        yield
    finally:
        for module, attr, original in patched:
            setattr(module, attr, original)


def per_instance_boundary_iou(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    *,
    k: float = None,
    match_iou: float = 0.5,
) -> tuple[list[float | None], dict[int, int], list[int]]:
    """Per-instance Boundary IoU under the INSTANCE-RELATIVE band, for figures.

    This is the quantity Boundary AP aggregates. Boundary AP itself is a dataset-level
    average over recall thresholds and cannot be drawn on one image; what a figure can show
    honestly is each instance's Boundary IoU, which is what the AP matching consumes. Caption
    figures accordingly.

    Distinct from `_calculate_biou_for_vis` in two ways that both matter:
      * band width is `k * equivalent_diameter` PER INSTANCE (Claim 1), not
        `0.02 * image_diagonal` shared by every instance -- the literature band collapses to
        Mask IoU for 84% of OAM-TCD crowns, so figures drawn with it are not showing
        boundary quality at all;
      * it is per instance, not a union over all masks. The union figure cannot colour
        instances individually and hides which crowns are actually poor.

    Returns `(bious, matches, unmatched_gt)`:
      * `bious[i]` is prediction i's Boundary IoU, or None if it matched no GT;
      * `matches[i] = j` for the GT index prediction i matched;
      * `unmatched_gt` lists GT indices no prediction claimed (misses).
    """
    if k is None:
        k = BOUNDARY_BAND_K
    to_boundary = make_instance_relative_mask_to_boundary(k)

    preds = np.asarray(pred_masks, dtype=bool)
    gts = np.asarray(gt_masks, dtype=bool)
    if preds.ndim == 2:
        preds = preds[None]
    if gts.ndim == 2:
        gts = gts[None]

    n_p, n_g = len(preds), len(gts)
    bious: list[float | None] = [None] * n_p
    matches: dict[int, int] = {}
    if n_p == 0 or n_g == 0:
        return bious, matches, list(range(n_g))

    # Greedy mask-IoU assignment, the same rule the oracle tools use, so a figure and the
    # tables agree on which prediction owns which crown.
    # Cast to float BEFORE the contraction: np.einsum on boolean arrays accumulates in
    # bool, so every overlapping pair would score an intersection of exactly 1 (True) and
    # nothing would ever match.
    pf = preds.reshape(n_p, -1).astype(np.float32)
    gf = gts.reshape(n_g, -1).astype(np.float32)
    inter = (pf @ gf.T).astype(np.float64)
    areas_p = pf.sum(1).astype(np.float64)[:, None]
    areas_g = gf.sum(1).astype(np.float64)[None, :]
    union = areas_p + areas_g - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    taken_p, taken_g = set(), set()
    for pi, gj in order:
        if iou[pi, gj] < match_iou:
            break
        if pi in taken_p or gj in taken_g:
            continue
        taken_p.add(int(pi))
        taken_g.add(int(gj))
        matches[int(pi)] = int(gj)

    for pi, gj in matches.items():
        pb = to_boundary(preds[pi])
        gb = to_boundary(gts[gj])
        i = float(np.logical_and(pb, gb).sum())
        u = float(np.logical_or(pb, gb).sum())
        bious[pi] = (i / u) if u > 0 else 1.0

    return bious, matches, [j for j in range(n_g) if j not in taken_g]
