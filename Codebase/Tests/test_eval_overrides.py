"""Tests for boundary geometry and mechanism-neutral SAM3 loss safety wrappers.

Tier 1 checks Boundary IoU construction and evaluator-to-visualisation agreement. Tier 2
uses the real SAM3 loss stack when available and verifies that `ProjectSampledMasks` remains
numerically identical to the native objective while handling empty and oversized batches.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

# torch and the system libomp both ship an OpenMP runtime on macOS; without this
# the interpreter aborts on import before a single check runs. KMP_DUPLICATE_LIB_OK
# only downgrades that abort to a warning, and this suite also imports scipy, so
# two OpenMP runtimes end up live in one process and a later multithreaded torch
# kernel segfaults. Single-threading both runtimes removes the collision. Nothing
# here is performance-sensitive -- the largest tensor is a few hundred KB.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Tests live in Codebase/Tests/; the modules under test live in Codebase/Core/.
CORE_DIR = Path(__file__).resolve().parent.parent / "Core"
SAM3_PKG = CORE_DIR.parent / "Support" / "sam3" / "sam3"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

FAILURES: list[str] = []
SKIPPED: list[str] = []

import numpy as np
import torch
import torch.nn.functional as F


torch.set_num_threads(1)


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok  {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL {label}" + (f" -- {detail}" if detail else ""))


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


# ═══════════════════════════════════════════════════════════════════════════════
# sam3 surface for tier 2
# ═══════════════════════════════════════════════════════════════════════════════


def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def _try_install_sam3() -> bool:
    """Make the real sam3 loss modules importable, or return False.

    Three obstacles on a laptop, each handled as narrowly as possible so the code
    actually under test stays real:

      * `sam3/__init__.py` imports huggingface_hub, so `sam3` is registered as a
        namespace package pointing at the real source and its `__init__` never
        runs. Every `sam3.train.loss.*` module then resolves to the real file.
      * `sigmoid_focal_loss.py` imports triton, which does not exist on macOS. It
        is replaced with the pure-torch equivalent, so `loss_mask` is still a real
        number and the inertness comparison is still meaningful (both sides go
        through the same function). **triton itself is never stubbed** -- torch's
        dynamo imports it for real and a stub breaks torch.
      * `eval_overrides.py` imports pycocotools and iopath at module scope for evaluator
        and prediction-dumper classes that are not exercised by the loss tests.

    Everything else is left unchanged. Module-scope patches log warnings when optional
    targets are unavailable.
    """
    if not SAM3_PKG.is_dir():
        return False

    pkg = types.ModuleType("sam3")
    pkg.__path__ = [str(SAM3_PKG)]
    sys.modules.setdefault("sam3", pkg)

    if "torchmetrics" not in sys.modules:
        _stub("torchmetrics", functional=types.SimpleNamespace(f1_score=lambda *a, **k: torch.zeros(())))

    name = "sam3.train.loss.sigmoid_focal_loss"
    if name not in sys.modules:
        for parent in ("sam3.train", "sam3.train.loss"):
            if parent not in sys.modules:
                mod = types.ModuleType(parent)
                mod.__path__ = [str(SAM3_PKG / Path(*parent.split(".")[1:]))]
                sys.modules[parent] = mod

        def _focal(inputs, targets, alpha, gamma):
            prob = inputs.sigmoid()
            ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            p_t = prob * targets + (1 - prob) * (1 - targets)
            loss = ce * ((1 - p_t) ** gamma)
            if alpha >= 0:
                loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
            return loss

        _stub(
            name,
            triton_sigmoid_focal_loss=_focal,
            triton_sigmoid_focal_loss_reduce=lambda i, t, a, g: _focal(i, t, a, g).sum(),
        )

    if "pycocotools" not in sys.modules:
        _stub("pycocotools")
        _stub("pycocotools.mask")
        _stub("pycocotools.coco", COCO=object)
        _stub("pycocotools.cocoeval", COCOeval=object)
    if "iopath" not in sys.modules:
        _stub("iopath")
        _stub("iopath.common")
        _stub("iopath.common.file_io", g_pathmgr=types.SimpleNamespace(open=open, exists=lambda p: False))

    try:
        import sam3.train.loss.loss_fns  # noqa: F401
        import sam3.train.loss.mask_sampling  # noqa: F401
        import eval_overrides  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"  (sam3 loss stack unavailable: {type(exc).__name__}: {exc})")
        return False
    return True


HAVE_SAM3 = _try_install_sam3()


def _psm(**kw):
    """A ProjectSampledMasks with small, test-sized parameters (no mechanism kwargs)."""
    from eval_overrides import ProjectSampledMasks

    params = dict(
        weight_dict={"loss_mask": 1.0, "loss_dice": 1.0},
        num_sample_points=64,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    )
    params.update(kw)
    return ProjectSampledMasks(**params)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1a: Boundary-IoU band geometry (reference-aligned)
# ═══════════════════════════════════════════════════════════════════════════════


def _ref_band(mask: np.ndarray, d: int) -> np.ndarray:
    """The band as eval_overrides/visualization now build it."""
    from scipy.ndimage import binary_erosion

    struct = np.ones((3, 3), dtype=bool)
    eroded = binary_erosion(mask, structure=struct, iterations=d, border_value=0)
    return mask & ~eroded


def test_biou_band_matches_reference_geometry() -> None:
    section("Boundary-IoU band geometry (bowenc0221/boundary-iou-api)")
    from scipy.ndimage import generate_binary_structure, binary_erosion

    m = np.zeros((12, 12), dtype=bool)
    m[0:6, 0:6] = True
    band = _ref_band(m, 2)
    check(
        "the array border counts as contour (border_value=0)",
        bool(band[0, 0] and band[0, 5] and band[5, 0]),
        "an instance truncated by the raster edge must carry a band there",
    )

    full = np.ones((12, 12), dtype=bool)
    band_full = _ref_band(full, 2)
    check(
        "an all-foreground mask yields the d-wide frame, not an empty band",
        int(band_full.sum()) == 12 * 12 - 8 * 8,
        f"{int(band_full.sum())} px",
    )

    square = np.zeros((40, 40), dtype=bool)
    square[10:30, 10:30] = True
    interior = square & ~_ref_band(square, 3)
    check(
        "Chebyshev erosion: a 20x20 square at d=3 leaves a 14x14 interior",
        int(interior.sum()) == 14 * 14,
        f"{int(interior.sum())} px",
    )

    yy, xx = np.mgrid[0:60, 0:60]
    disc = ((yy - 30) ** 2 + (xx - 30) ** 2) < 20**2
    l_inf = _ref_band(disc, 3)
    l1 = disc & ~binary_erosion(
        disc, structure=generate_binary_structure(2, 1), iterations=3, border_value=0
    )
    check(
        "L-inf band differs from the 4-connected L1 band (a different band shape)",
        not np.array_equal(l_inf, l1) and int(l_inf.sum()) > int(l1.sum()),
        f"L-inf {int(l_inf.sum())} px vs L1 {int(l1.sum())} px",
    )

    diag = float(np.hypot(2048, 2048))
    check(
        "d rounds, not truncates: ratio 0.02 on a 2048px tile gives 58",
        max(1, int(round(diag * 0.02))) == 58 and int(max(1, diag * 0.02)) == 57,
        "the old truncation gave 57",
    )
    check(
        "the 8px fine band is unaffected by round-vs-truncate",
        max(1, int(round(diag * (8.0 / diag)))) == 8,
    )


def test_visualization_band_matches_evaluator() -> None:
    section("visualization.py B-IoU stays identical to BoundaryIoUEvaluator")
    src_eval = (CORE_DIR / "eval_overrides.py").read_text()
    src_vis = (CORE_DIR / "visualization.py").read_text()
    seg_eval = src_eval[src_eval.index("class BoundaryIoUEvaluator") : src_eval.index("class ProjectSemanticSegCriterion")]
    seg_vis = src_vis[src_vis.index("def _calculate_biou_for_vis") :][:4000]
    for token in ("int(round(", "np.ones((3, 3), dtype=bool)", "border_value=0"):
        check(
            f"both use {token!r}",
            token in seg_eval and token in seg_vis,
            "visualization.py documents itself as identical; they must move together",
        )
    for stale in ("generate_binary_structure(2, 1)", "^ gt_eroded", ".all()"):
        code_eval = "\n".join(l for l in seg_eval.splitlines() if not l.lstrip().startswith("#"))
        code_vis = "\n".join(l for l in seg_vis.splitlines() if not l.lstrip().startswith("#"))
        check(f"neither retains {stale!r}", stale not in code_eval and stale not in code_vis)


def test_metric_constant_agreement() -> None:
    section("config constants agree with the metric they are scored against")
    try:
        import config
        from eval_predictions import BOUNDARY_AP_FINE_BAND_PX
    except Exception as exc:
        SKIPPED.append(f"constant agreement ({type(exc).__name__}: {exc})")
        print(f"  skip  config/eval_predictions unavailable: {type(exc).__name__}")
        return
    check(
        "BOUNDARY_FIXED_BAND_PX == BOUNDARY_AP_FINE_BAND_PX (one declaration of the "
        "fixed 8px comparison row)",
        config.BOUNDARY_FIXED_BAND_PX == BOUNDARY_AP_FINE_BAND_PX,
        f"{config.BOUNDARY_FIXED_BAND_PX} vs {BOUNDARY_AP_FINE_BAND_PX}",
    )
    try:
        from boundary_band import BOUNDARY_BAND_K as METRIC_K
        check(
            "config.BOUNDARY_BAND_K == boundary_band.BOUNDARY_BAND_K (Claim 1's "
            "pre-registered constant, declared once per module, must agree)",
            config.BOUNDARY_BAND_K == METRIC_K, f"{config.BOUNDARY_BAND_K} vs {METRIC_K}",
        )
    except Exception as exc:
        print(f"  skip  boundary_band unavailable: {type(exc).__name__}")
    check(
        "no separate boundary loss term is wired into the objective",
        "loss_boundary" not in config._loss_config()["loss_fns_find"][-1].get("weight_dict", {})
        if isinstance(config._loss_config(), dict) and "loss_fns_find" in config._loss_config()
        else True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2: inertness and weight_dict completeness (needs the sam3 loss stack)
# ═══════════════════════════════════════════════════════════════════════════════


def _fixture(n: int = 3, grid: int = 16, gt_hw: int = 64, batch: int = 2, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randn(n, grid, grid, generator=g)
    gt = torch.zeros(n, gt_hw, gt_hw)
    for i in range(n):
        gt[i, 10 + 4 * i : 40 + 4 * i, 8 + 6 * i : 44 + 6 * i] = 1.0
    return src, gt


def test_inertness_matches_stock_masks() -> None:
    section("ProjectSampledMasks is bit-identical to stock Masks (fixes, not objective)")
    if not HAVE_SAM3:
        SKIPPED.append("inertness (sam3 loss stack unavailable)")
        print("  skip  needs the real sam3 loss stack")
        return
    from sam3.train.loss.loss_fns import Masks
    from eval_overrides import ProjectSampledMasks

    src, gt = _fixture()
    kw = dict(
        weight_dict={"loss_mask": 1.0, "loss_dice": 1.0},
        num_sample_points=64,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    )
    stock, proj = Masks(**kw), ProjectSampledMasks(**kw)

    torch.manual_seed(7)
    a = stock._sampled_loss(src.clone(), gt.clone(), torch.tensor(3.0))
    torch.manual_seed(7)
    b = proj._sampled_loss(src.clone(), gt.clone(), torch.tensor(3.0))

    check("the same keys are returned", set(a) == set(b) == {"loss_mask", "loss_dice"})
    for k in a:
        check(f"{k} is bit-identical", torch.equal(a[k], b[k]), f"{a[k].item()} vs {b[k].item()}")
    check("no extra key is added", "loss_boundary" not in b)


def test_zero_instance_batch_emits_every_key() -> None:
    section("A zero-instance batch does not raise and emits every weight_dict key")
    if not HAVE_SAM3:
        SKIPPED.append("zero-instance (sam3 loss stack unavailable)")
        print("  skip  needs the real sam3 loss stack")
        return
    on = _psm()
    empty_src, empty_gt = torch.zeros(0, 16, 16), torch.zeros(0, 64, 64)
    try:
        losses = on._sampled_loss(empty_src, empty_gt, torch.tensor(1.0))
        raised = None
    except Exception as exc:
        losses, raised = None, exc
    check("no exception", raised is None, repr(raised))
    if losses is not None:
        check("both keys present", set(losses) == {"loss_mask", "loss_dice"})
        check("all zero", all(float(v) == 0.0 for v in losses.values()))
        check("reduce_loss accepts the dict", _reduces(on, losses))


def _reduces(module, losses) -> bool:
    try:
        module.reduce_loss(losses)
        return True
    except Exception as exc:
        print(f"        reduce_loss raised: {type(exc).__name__}: {exc}")
        return False


def test_chunked_point_sample_patch() -> None:
    """Require bit-identical chunking before `grid_sample` exceeds 32-bit indexing."""
    section("point_sample patch: bit-identical, size-triggered, RNG-neutral")
    if not HAVE_SAM3:
        SKIPPED.append("point_sample patch (sam3 loss stack unavailable)")
        print("  skip  needs the real sam3 loss stack")
        return

    import eval_overrides
    import sam3.train.loss.loss_fns as loss_fns
    import sam3.train.loss.mask_sampling as mask_sampling

    check("mask_sampling.point_sample is patched",
          mask_sampling.point_sample is eval_overrides._chunked_point_sample)
    check("loss_fns' import-time binding is patched too (the loss reads this binding)",
          loss_fns.point_sample is eval_overrides._chunked_point_sample)

    original = eval_overrides._original_point_sample
    patched = eval_overrides._chunked_point_sample

    # Small input: must pass straight through, same object semantics.
    masks = torch.randn(40, 1, 32, 32)
    coords = torch.rand(40, 17, 2)
    check("small input is bit-identical (passes through unchunked)",
          torch.equal(original(masks, coords, align_corners=False),
                      patched(masks, coords, align_corners=False)))

    # Force the chunked branch by lowering the threshold, then compare bitwise.
    real_limit = eval_overrides._POINT_SAMPLE_SAFE_ELEMENTS
    real_chunk = eval_overrides._POINT_SAMPLE_MAX_CHUNK
    try:
        eval_overrides._POINT_SAMPLE_SAFE_ELEMENTS = 1000  # 40*1*32*32 = 40960 > 1000
        eval_overrides._POINT_SAMPLE_MAX_CHUNK = 7         # deliberately not a divisor of 40
        chunked = patched(masks, coords, align_corners=False)
        check("chunked branch is bit-identical to one unchunked call",
              torch.equal(original(masks, coords, align_corners=False), chunked),
              f"max abs diff {(original(masks, coords, align_corners=False) - chunked).abs().max().item()}")
        check("shape is preserved across a non-divisor chunk size",
              tuple(chunked.shape) == tuple(original(masks, coords, align_corners=False).shape))
    finally:
        eval_overrides._POINT_SAMPLE_SAFE_ELEMENTS = real_limit
        eval_overrides._POINT_SAMPLE_MAX_CHUNK = real_chunk

    # The arithmetic the diagnosis rests on.
    per_mask = 1008 * 1008
    n_limit = (2**31 - 1) // per_mask
    check("the real 1008x1008 overflow threshold is N=2113 (fix triggers below it)",
          n_limit == 2113 and real_limit // per_mask < n_limit,
          f"n_limit={n_limit}, patch triggers at N>{real_limit // per_mask}")

    # Chunking must not touch the global RNG: point_sample draws nothing.
    torch.manual_seed(0)
    _ = patched(masks, coords, align_corners=False)
    after_patched = torch.rand(3)
    torch.manual_seed(0)
    _ = original(masks, coords, align_corners=False)
    after_original = torch.rand(3)
    check("point_sample consumes no RNG, chunked or not (augmentation stream intact)",
          torch.equal(after_patched, after_original))


def test_patch_registry_reports_every_patch() -> None:
    """Every patch is accounted for; in a GPU environment every patch must have applied."""
    section("patch registry: every patch reports its outcome")
    if not HAVE_SAM3:
        SKIPPED.append("patch registry (sam3 loss stack unavailable)")
        print("  skip  needs the real sam3 loss stack")
        return

    import eval_overrides

    status = eval_overrides.patch_status()
    check("one entry per named patch, in order",
          tuple(status) == eval_overrides.PATCH_NAMES and len(status) == 8, str(tuple(status)))
    check("each entry is applied or carries a reason",
          all(value == "applied" or value.startswith("failed: ") for value in status.values()), str(status))
    if torch.cuda.is_available():
        check("a GPU environment applies every patch",
              all(value == "applied" for value in status.values()), str(status))
    else:
        applied = [name for name, value in status.items() if value == "applied"]
        print(f"  info  no CUDA device: {len(applied)} of {len(status)} patches applied here: {applied}")


def main() -> None:
    print("=" * 62)
    print("eval_overrides self-tests")
    print(f"sam3 loss stack: {'available' if HAVE_SAM3 else 'UNAVAILABLE (tier 2 skipped)'}")
    print("=" * 62)

    test_biou_band_matches_reference_geometry()
    test_visualization_band_matches_evaluator()
    test_metric_constant_agreement()

    test_inertness_matches_stock_masks()
    test_zero_instance_batch_emits_every_key()
    test_chunked_point_sample_patch()
    test_patch_registry_reports_every_patch()

    print()
    print("=" * 62)
    if SKIPPED:
        print(f"{len(SKIPPED)} CHECK GROUP(S) SKIPPED")
        for name in SKIPPED:
            print(f"  - {name}")
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED")
        for name in FAILURES:
            print(f"  - {name}")
        print("=" * 62)
        raise SystemExit(1)
    print("ALL CHECKS PASSED")
    print("=" * 62)


if __name__ == "__main__":
    main()
