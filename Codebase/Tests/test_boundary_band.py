#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for Core/boundary_band.py -- the instance-relative Boundary IoU band.

The band rule is a metric contribution, so the failure modes that matter are the
silent ones: a band width that is subtly wrong, a patch that covers three of four
import sites, or a reimplementation that has drifted from upstream's erosion
semantics. Each of those produces a plausible number rather than an error.

Run: python Tests/run_all.py boundary_band
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# Tests live in Codebase/Tests/; the modules under test live in Codebase/Core/.
CORE_DIR = Path(__file__).resolve().parent.parent / "Core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from boundary_band import (  # noqa: E402
    BOUNDARY_BAND_K,
    equivalent_diameter,
    instance_band_px,
    instance_relative_dilation,
    make_instance_relative_mask_to_boundary,
    patched_instance_relative_bands,
)

CHECKS = 0
FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"  ok  {label}")
    else:
        FAILURES.append(f"{label}{(' -- ' + detail) if detail else ''}")
        print(f"  FAIL {label}{(' -- ' + detail) if detail else ''}")


def section(title: str) -> None:
    print(f"\n{title}")


def disc(diameter: float, canvas: int = 256) -> np.ndarray:
    """Filled disc of the given diameter, centred on a square canvas."""
    yy, xx = np.mgrid[0:canvas, 0:canvas]
    c = (canvas - 1) / 2.0
    return (((yy - c) ** 2 + (xx - c) ** 2) <= (diameter / 2.0) ** 2).astype(np.uint8)


# ── band width arithmetic ────────────────────────────────────────────────────

def test_equivalent_diameter() -> None:
    section("equivalent_diameter: area -> diameter of an equal-area disc")
    for d in (16.0, 22.76, 65.78, 156.95):
        area = math.pi * (d / 2.0) ** 2
        check(f"round-trips at d={d}", abs(equivalent_diameter(area) - d) < 1e-9)
    check("zero area is zero, not NaN", equivalent_diameter(0) == 0.0)
    check("negative area clamps to zero rather than raising on sqrt",
          equivalent_diameter(-5) == 0.0)


def test_band_scales_with_instance() -> None:
    section("instance_band_px: width is proportional to crown size")
    areas = [math.pi * (d / 2) ** 2 for d in (20.0, 40.0, 80.0)]
    widths = [instance_band_px(a) for a in areas]
    check("doubling diameter doubles the band",
          abs(widths[1] / widths[0] - 2.0) < 1e-9 and abs(widths[2] / widths[1] - 2.0) < 1e-9,
          f"{widths}")
    # The band must stay a fixed FRACTION of the crown -- that is the whole claim.
    for d, w in zip((20.0, 40.0, 80.0), widths):
        check(f"band is k of diameter at d={d:g}", abs(w / d - BOUNDARY_BAND_K) < 1e-9)


def test_measured_k_reproduces_d50() -> None:
    """k must reproduce the measured d50 bands (0.7), or it was mistyped."""
    section("k = 0.19 against the measured d50 operating points")
    for bin_name, diam, d50 in (("medium", 65.78, 12.5), ("large", 156.95, 30.0)):
        predicted = instance_band_px(math.pi * (diam / 2) ** 2)
        rel = abs(predicted - d50) / d50
        check(f"{bin_name}: k*d = {predicted:.1f}px vs measured d50 {d50}px (within 10%)",
              rel < 0.10, f"relative error {rel:.1%}")
    # Small is the declared non-uniformity: k=0.19 gives a stricter band than its own
    # d50 of 5.5px. Asserted so the limitation cannot be quietly lost.
    small = instance_band_px(math.pi * (22.76 / 2) ** 2)
    check("small: k*d is NARROWER than its own d50, the declared limitation",
          small < 5.5, f"{small:.2f}px vs 5.5px")


def test_dilation_never_zero() -> None:
    section("instance_relative_dilation: floors at 1")
    tiny = np.zeros((16, 16), dtype=np.uint8)
    tiny[7:9, 7:9] = 1                      # 4 px, k*d ~ 0.43 -> rounds to 0
    check("a 4px instance still gets dilation >= 1",
          instance_relative_dilation(tiny) >= 1,
          "a zero-width band scores every prediction as perfect")
    check("an empty mask does not raise and still floors at 1",
          instance_relative_dilation(np.zeros((16, 16), dtype=np.uint8)) == 1)


# ── the band itself ──────────────────────────────────────────────────────────

def test_band_is_a_rind_at_every_scale() -> None:
    """The point of the whole exercise: the band is a fixed fraction of any crown."""
    section("band area fraction is scale-invariant (the contribution, in one test)")
    fn = make_instance_relative_mask_to_boundary()
    fractions = []
    for d in (40.0, 80.0, 160.0):
        m = disc(d, canvas=256)
        band = fn(m)
        fractions.append(band.sum() / max(m.sum(), 1))
    spread = max(fractions) - min(fractions)
    check("band/area fraction varies < 0.10 across a 4x size range",
          spread < 0.10, f"fractions {[round(f, 3) for f in fractions]}")

    # Contrast with the fixed band, which is what this replaces: a fixed 8px erosion
    # takes ~60% of a 40px crown and ~19% of a 160px one.
    fixed = []
    for d in (40.0, 80.0, 160.0):
        m = disc(d, canvas=256)
        from boundary_band import _erode
        padded = np.pad(m, 1, mode="constant", constant_values=0)
        er = _erode(padded, iterations=8)[1:m.shape[0] + 1, 1:m.shape[1] + 1]
        fixed.append((m - er).sum() / max(m.sum(), 1))
    check("a FIXED 8px band varies much more over the same range",
          (max(fixed) - min(fixed)) > spread,
          f"fixed spread {max(fixed) - min(fixed):.3f} vs instance-relative {spread:.3f}")


def test_band_is_inner_and_nonempty() -> None:
    section("band geometry: inner, non-empty, contained in the mask")
    fn = make_instance_relative_mask_to_boundary()
    m = disc(80.0)
    band = fn(m)
    check("band is a subset of the mask (inner band, per the paper)",
          bool(np.all(band[m == 0] == 0)))
    check("band is non-empty", band.sum() > 0)
    check("band is strictly smaller than the mask", band.sum() < m.sum())
    check("band is binary", set(np.unique(band)).issubset({0, 1}))


def test_matches_upstream_when_widths_agree() -> None:
    """Reimplementation must agree with upstream where both use the same width.

    This is the drift check. `make_instance_relative_mask_to_boundary` reimplements
    upstream's body; if the padding or erosion semantics differ, every Boundary AP we
    report is subtly wrong in a way no other test would catch.
    """
    section("agreement with upstream mask_to_boundary at a matched width")
    try:
        from boundary_iou.utils.boundary_utils import mask_to_boundary
    except ImportError as exc:
        print(f"  SKIP boundary_iou not installed ({exc}); run on the cluster")
        return

    for d in (40.0, 80.0, 160.0):
        m = disc(d, canvas=256)
        h, w = m.shape
        dilation = instance_relative_dilation(m)
        # Feed upstream the image-relative ratio that yields the SAME integer dilation.
        ratio = dilation / math.sqrt(h ** 2 + w ** 2)
        ours = make_instance_relative_mask_to_boundary()(m)
        theirs = mask_to_boundary(m, dilation_ratio=ratio)
        check(f"byte-identical to upstream at d={d:g} (dilation {dilation})",
              np.array_equal(ours, theirs),
              f"{int(np.abs(ours.astype(int) - theirs.astype(int)).sum())} px differ")


# ── the patch ────────────────────────────────────────────────────────────────

def test_patch_covers_every_binding() -> None:
    section("patched_instance_relative_bands: all bindings, and restored on exit")
    try:
        import boundary_iou.utils.boundary_utils as bu
    except ImportError as exc:
        print(f"  SKIP boundary_iou not installed ({exc}); run on the cluster")
        return

    import importlib

    names = ("boundary_iou.utils.boundary_utils",
             "boundary_iou.coco_instance_api.coco",
             "boundary_iou.coco_instance_api.cocoeval")
    before = {}
    for n in names:
        try:
            mod = importlib.import_module(n)
        except ImportError:
            continue
        if hasattr(mod, "mask_to_boundary"):
            before[n] = mod.mask_to_boundary

    with patched_instance_relative_bands():
        for n, original in before.items():
            mod = importlib.import_module(n)
            check(f"{n} is patched",
                  mod.mask_to_boundary is not original,
                  "an unpatched binding silently keeps the image-relative width")

    for n, original in before.items():
        mod = importlib.import_module(n)
        check(f"{n} is restored", mod.mask_to_boundary is original)


def test_patch_restores_on_exception() -> None:
    section("patch is restored even when the body raises")
    try:
        import boundary_iou.utils.boundary_utils as bu
    except ImportError as exc:
        print(f"  SKIP boundary_iou not installed ({exc}); run on the cluster")
        return

    original = bu.mask_to_boundary
    try:
        with patched_instance_relative_bands():
            raise RuntimeError("simulated evaluation failure")
    except RuntimeError:
        pass
    check("restored after an exception",
          bu.mask_to_boundary is original,
          "a leaked patch makes later runs report a different metric than they claim")


def test_per_instance_boundary_iou() -> None:
    """The per-instance quantity figures colour by, and that Boundary AP aggregates.

    A first implementation used `np.einsum` on boolean arrays for the IoU matching; einsum
    accumulates in bool, so every overlapping pair scored an intersection of exactly 1 and
    nothing ever matched. Pinned here because the failure was silent -- it returned a valid
    shape with every entry None.
    """
    section("per-instance Boundary IoU: matching, FP, misses, ordering")
    from boundary_band import per_instance_boundary_iou

    def disc(cy, cx, r, size=200):
        yy, xx = np.mgrid[0:size, 0:size]
        return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r

    gt = np.stack([disc(50, 50, 20), disc(50, 140, 20), disc(150, 50, 20)])
    pred = np.stack([disc(50, 50, 20), disc(52, 142, 20), disc(150, 150, 15)])
    bious, matches, missed = per_instance_boundary_iou(pred, gt)

    check("an exact match scores ~1", bious[0] is not None and bious[0] > 0.95,
          f"{bious[0]}")
    check("a shifted prediction scores lower than an exact one",
          bious[1] is not None and bious[1] < bious[0], f"{bious[1]} vs {bious[0]}")
    check("a false positive has no Boundary IoU", bious[2] is None)
    check("the unclaimed GT is reported as missed", missed == [2], f"{missed}")
    check("matching is one-to-one", matches == {0: 0, 1: 1}, f"{matches}")

    # Degenerate inputs must not raise: figures are generated over whole test sets.
    empty = np.zeros((0, 200, 200), dtype=bool)
    b2, m2, miss2 = per_instance_boundary_iou(empty, gt)
    check("no predictions -> every GT missed, no crash", b2 == [] and miss2 == [0, 1, 2])
    b3, m3, miss3 = per_instance_boundary_iou(pred, empty)
    check("no GT -> every prediction unscored, no crash",
          b3 == [None, None, None] and miss3 == [])


def main() -> int:
    print("=" * 70)
    print("boundary_band: instance-relative Boundary IoU bands")
    print("=" * 70)
    test_equivalent_diameter()
    test_band_scales_with_instance()
    test_measured_k_reproduces_d50()
    test_dilation_never_zero()
    test_band_is_a_rind_at_every_scale()
    test_band_is_inner_and_nonempty()
    test_matches_upstream_when_widths_agree()
    test_patch_covers_every_binding()
    test_patch_restores_on_exception()
    test_per_instance_boundary_iou()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S) of {CHECKS} checks")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"ALL CHECKS PASSED ({CHECKS} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
