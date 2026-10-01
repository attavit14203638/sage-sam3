#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Validation of the class-split and Boundary AP additions to eval_predictions.py.

The two things pinned here are the ones that would fail silently rather than
loudly, and both would corrupt a reported number rather than crash a run:

  1. `iscrowd` ignore semantics actually take effect. A prediction landing on the
     non-target class must be neither a true positive nor a false positive. The
     discriminating fixture is a *perfect* prediction on an ignored crown: under
     working ignore semantics AP stays 1.0, whereas the naive alternative of
     deleting the other class's GT turns that prediction into a false positive and
     drives AP down. A test that only counted annotations could not tell these
     apart.
  2. A GT export lacking `original_category_id` raises instead of scoring nothing.
     Treating a missing field as "not the target class" would ignore the whole
     dataset and report AP over an empty population, which reads as a legitimate
     number.

Needs only numpy + pycocotools; no GPU, no checkpoint, no real dataset.

Run:  python test_eval_predictions.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

# Tests live in Codebase/Tests/; the modules under test live in Codebase/Core/.
CORE_DIR = Path(__file__).resolve().parent.parent / "Core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from pycocotools import mask as mask_utils  # noqa: E402

import eval_predictions as ep  # noqa: E402

try:
    import pytest

    @pytest.fixture
    def tmp(tmp_path):  # alias so `def test_...(tmp)` collects under pytest
        return tmp_path
except ImportError:  # standalone `python test_eval_predictions.py` runs without pytest
    pass

SIZE = 128
# Two well-separated squares so neither prediction can accidentally match the
# other's crown: any cross-match would make the ignore-semantics test pass for the
# wrong reason.
TREE_BOX = (20, 20, 44, 44)      # y0, x0, y1, x1
CANOPY_BOX = (80, 80, 120, 120)


def _rle(y0: int, x0: int, y1: int, x1: int) -> dict:
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    encoded = mask_utils.encode(np.asfortranarray(mask))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def _area(y0: int, x0: int, y1: int, x1: int) -> int:
    return (y1 - y0) * (x1 - x0)


def _write_fixture(tmp: Path, n_images: int = 2) -> tuple[Path, Path]:
    """One tree crown and one canopy crown per image, each perfectly predicted."""
    images, annotations, predictions = [], [], []
    annotation_id = 1
    for image_id in range(n_images):
        images.append(
            {"id": image_id, "file_name": f"img_{image_id}.png", "height": SIZE, "width": SIZE}
        )
        for original_category_id, box in (
            (ep.TREE_ORIGINAL_CATEGORY_ID, TREE_BOX),
            (ep.CANOPY_ORIGINAL_CATEGORY_ID, CANOPY_BOX),
        ):
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # collapsed, as dataset_adapter.py emits
                    "original_category_id": original_category_id,
                    "segmentation": _rle(*box),
                    "area": _area(*box),
                    "bbox": [box[1], box[0], box[3] - box[1], box[2] - box[0]],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": _rle(*box),
                    "score": 0.9,
                }
            )

    gt = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "tree"}],
    }
    gt_path = tmp / "gt.json"
    pred_path = tmp / "pred.json"
    gt_path.write_text(json.dumps(gt))
    pred_path.write_text(json.dumps(predictions))
    return gt_path, pred_path


def test_none_is_a_passthrough(tmp: Path) -> None:
    gt_path, _ = _write_fixture(tmp)
    out_path, info = ep._prepare_class_split_gt(str(gt_path), "none", str(tmp / "out.json"))
    assert out_path == str(gt_path), "class-split none must not rewrite the GT"
    assert info["enabled"] is False
    print("  ok: --class-split none is a pass-through")


def test_tree_split_marks_canopy_and_leaves_input_untouched(tmp: Path) -> None:
    gt_path, _ = _write_fixture(tmp, n_images=3)
    original = gt_path.read_text()

    split_path, info = ep._prepare_class_split_gt(str(gt_path), "tree", str(tmp / "out.json"))
    assert info["scored_annotations"] == 3, info
    assert info["ignored_annotations"] == 3, info
    assert info["kept_original_category_id"] == ep.TREE_ORIGINAL_CATEGORY_ID

    split = json.loads(Path(split_path).read_text())
    for annotation in split["annotations"]:
        is_tree = annotation["original_category_id"] == ep.TREE_ORIGINAL_CATEGORY_ID
        assert int(annotation["iscrowd"]) == (0 if is_tree else 1), annotation

    assert gt_path.read_text() == original, "the caller's GT file must not be mutated"
    print("  ok: tree split marks canopy iscrowd=1 without mutating the input")


def test_canopy_split_is_the_complement(tmp: Path) -> None:
    gt_path, _ = _write_fixture(tmp, n_images=2)
    _, tree_info = ep._prepare_class_split_gt(str(gt_path), "tree", str(tmp / "t.json"))
    _, canopy_info = ep._prepare_class_split_gt(str(gt_path), "canopy", str(tmp / "c.json"))
    assert tree_info["scored_annotations"] == canopy_info["ignored_annotations"]
    assert tree_info["ignored_annotations"] == canopy_info["scored_annotations"]
    print("  ok: tree and canopy splits partition the annotations")


def test_missing_original_category_id_raises(tmp: Path) -> None:
    gt_path, _ = _write_fixture(tmp)
    gt = json.loads(gt_path.read_text())
    del gt["annotations"][0]["original_category_id"]
    stripped = tmp / "gt_stripped.json"
    stripped.write_text(json.dumps(gt))

    try:
        ep._prepare_class_split_gt(str(stripped), "tree", str(tmp / "out.json"))
    except ValueError as exc:
        assert "original_category_id" in str(exc)
        print("  ok: missing original_category_id raises instead of scoring nothing")
        return
    raise AssertionError("expected ValueError for GT lacking original_category_id")


def test_absent_target_class_raises(tmp: Path) -> None:
    gt_path, _ = _write_fixture(tmp)
    gt = json.loads(gt_path.read_text())
    for annotation in gt["annotations"]:
        annotation["original_category_id"] = ep.CANOPY_ORIGINAL_CATEGORY_ID
    canopy_only = tmp / "gt_canopy_only.json"
    canopy_only.write_text(json.dumps(gt))

    try:
        ep._prepare_class_split_gt(str(canopy_only), "tree", str(tmp / "out.json"))
    except ValueError as exc:
        assert "zero scored annotations" in str(exc)
        print("  ok: a split that would score nothing raises")
        return
    raise AssertionError("expected ValueError when the target class is absent")


def test_unknown_split_raises(tmp: Path) -> None:
    gt_path, _ = _write_fixture(tmp)
    try:
        ep._prepare_class_split_gt(str(gt_path), "shrub", str(tmp / "out.json"))
    except ValueError as exc:
        assert "Unknown class split" in str(exc)
        print("  ok: an unknown split name raises")
        return
    raise AssertionError("expected ValueError for an unknown class split")


def test_ignore_semantics_do_not_penalise_the_other_class(tmp: Path) -> None:
    """The load-bearing test: a perfect prediction on an ignored crown is not a FP.

    Both crowns are predicted perfectly, so collapsed AP is 1.0. Under the tree
    split the canopy GT becomes an ignore region and its prediction must be
    discarded rather than counted against precision, leaving AP at 1.0. If the
    implementation deleted canopy GT instead of ignoring it, that prediction would
    become an unmatched false positive on every image and AP would fall well below
    1.0 -- which is precisely the alternative this test exists to rule out.
    """
    gt_path, pred_path = _write_fixture(tmp, n_images=2)

    collapsed = ep.run_coco_eval(str(pred_path), str(gt_path), max_dets=512)
    assert collapsed["AP"] > 0.99, f"fixture should be perfect collapsed, got {collapsed['AP']}"

    split_path, info = ep._prepare_class_split_gt(str(gt_path), "tree", str(tmp / "out.json"))
    assert info["ignored_annotations"] == 2
    tree_only = ep.run_coco_eval(str(pred_path), split_path, max_dets=512)

    assert tree_only["AP"] > 0.99, (
        "the canopy prediction was counted against precision, so iscrowd ignore "
        f"semantics are not in effect (tree-only AP {tree_only['AP']:.4f})"
    )
    print(f"  ok: ignored-class predictions are not false positives "
          f"(collapsed AP {collapsed['AP']:.4f}, tree-only AP {tree_only['AP']:.4f})")


def test_ignoring_a_class_changes_the_scored_population(tmp: Path) -> None:
    """Ignore regions must actually remove crowns from the recall denominator.

    Complement to the test above: that one proves ignoring does not *penalise*,
    this one proves it is not a no-op. The canopy prediction is deleted while its
    GT remains, so collapsed recall must suffer a genuine miss, and the tree split
    must not.
    """
    gt_path, pred_path = _write_fixture(tmp, n_images=2)
    predictions = json.loads(pred_path.read_text())
    # Drop every prediction on the canopy crown, keeping the tree predictions.
    canopy_rle = _rle(*CANOPY_BOX)
    kept = [p for p in predictions if p["segmentation"]["counts"] != canopy_rle["counts"]]
    assert len(kept) == 2, f"expected 2 tree predictions to survive, got {len(kept)}"
    partial_pred = tmp / "pred_no_canopy.json"
    partial_pred.write_text(json.dumps(kept))

    collapsed = ep.run_coco_eval(str(partial_pred), str(gt_path), max_dets=512)
    split_path, _ = ep._prepare_class_split_gt(str(gt_path), "tree", str(tmp / "out2.json"))
    tree_only = ep.run_coco_eval(str(partial_pred), split_path, max_dets=512)

    assert collapsed["AP"] < 0.99, (
        f"collapsed AP should register the missed canopy crowns, got {collapsed['AP']:.4f}"
    )
    assert tree_only["AP"] > collapsed["AP"] + 0.05, (
        "ignoring canopy did not remove it from the recall denominator: "
        f"collapsed {collapsed['AP']:.4f} vs tree-only {tree_only['AP']:.4f}"
    )
    print(f"  ok: ignoring a class removes it from recall "
          f"(collapsed AP {collapsed['AP']:.4f} -> tree-only AP {tree_only['AP']:.4f})")


def test_boundary_ap_is_optional_and_reports_its_absence(tmp: Path) -> None:
    """Boundary AP must degrade to a recorded gap, never a crash or a silent zero."""
    gt_path, pred_path = _write_fixture(tmp)
    result = ep.run_boundary_ap(str(pred_path), str(gt_path), max_dets=512)

    if result.get("available"):
        assert "Boundary_AP" in result and "Boundary_AP75" in result, result
        assert result["dilation_ratio"] == 0.02, result
        assert result["Boundary_AP"] > 0.99, (
            f"a pixel-exact fixture should score ~1.0 Boundary AP, got {result['Boundary_AP']}"
        )
        print(f"  ok: Boundary AP available and exact on a perfect fixture "
              f"({result['Boundary_AP']:.4f})")
    else:
        assert "reason" in result and "install_hint" in result, result
        assert result["install_hint"] == ep.BOUNDARY_AP_INSTALL_HINT
        print("  ok: Boundary AP unavailable, reported as a recorded gap "
              f"({result['reason']})")


def test_boundary_band_converts_to_a_ratio_that_round_trips(tmp: Path) -> None:
    """A band requested in source pixels must come back out as that many pixels.

    The upstream API only accepts a fraction of the image diagonal, so every
    reported Boundary AP depends on this one conversion. An error here would not
    crash: it would silently widen or narrow the rind and change the metric.
    """
    gt_path, _ = _write_fixture(tmp)
    diagonal, (height, width) = ep.dataset_image_diagonal(str(gt_path))

    assert abs(diagonal - math.hypot(height, width)) < 1e-6, diagonal

    for band_px in (2.0, 4.0, 8.0, 57.0):
        ratio = ep.boundary_band_to_ratio(band_px, diagonal)
        realized = int(max(1, ratio * diagonal))
        assert realized == int(band_px), (
            f"{band_px}px -> ratio {ratio} -> {realized}px, expected {int(band_px)}px"
        )
    print(f"  ok: pixel bands round trip through the ratio on a "
          f"{height}x{width} image")

    # The degeneracy the fine band exists to avoid: at the literature ratio the
    # band must be provably wider than a typical crown.
    lit_band = ep.BOUNDARY_AP_LITERATURE_RATIO * math.hypot(2048, 2048)
    assert lit_band > 50.0, lit_band
    print(f"  ok: literature ratio is a {lit_band:.0f}px band on a 2048px tile, "
          "wider than most crowns")

    try:
        ep.boundary_band_to_ratio(0.0, diagonal)
    except ValueError:
        print("  ok: a non-positive band is refused")
    else:
        raise AssertionError("a non-positive band should raise")


def test_mixed_image_sizes_are_refused(tmp: Path) -> None:
    """One ratio across mixed image sizes would mean a different band per image."""
    gt_path, _ = _write_fixture(tmp)
    gt = json.loads(gt_path.read_text())
    gt["images"].append({
        "id": 9999, "file_name": "odd.jpg",
        "height": gt["images"][0]["height"] * 2,
        "width": gt["images"][0]["width"],
    })
    mixed = tmp / "gt_mixed_sizes.json"
    mixed.write_text(json.dumps(gt))

    try:
        ep.dataset_image_diagonal(str(mixed))
    except ValueError as error:
        assert "mixes image sizes" in str(error), str(error)
        print("  ok: a mixed-size GT is refused rather than silently averaged")
    else:
        raise AssertionError("mixed image sizes should raise")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="eval_pred_test_") as tmp_name:
        tmp = Path(tmp_name)
        test_none_is_a_passthrough(tmp)
        test_tree_split_marks_canopy_and_leaves_input_untouched(tmp)
        test_canopy_split_is_the_complement(tmp)
        test_missing_original_category_id_raises(tmp)
        test_absent_target_class_raises(tmp)
        test_unknown_split_raises(tmp)
        test_ignore_semantics_do_not_penalise_the_other_class(tmp)
        test_ignoring_a_class_changes_the_scored_population(tmp)
        test_boundary_ap_is_optional_and_reports_its_absence(tmp)
        test_boundary_band_converts_to_a_ratio_that_round_trips(tmp)
        test_mixed_image_sizes_are_refused(tmp)
    print("\nall eval_predictions class-split / Boundary AP tests passed")


if __name__ == "__main__":
    main()
