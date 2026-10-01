"""OAM-TCD dataset loading, validation, and COCO instance export.

Loads the Hugging Face `restor/tcd` dataset, validates each sample's annotations (categories, boxes,
polygon and run-length-encoded masks, crowd flags), and writes the COCO instance export that
training and evaluation read. Each exported annotation keeps its original category ID, and the
exported category list is collapsed to one `tree` concept. The command-line entry point carves a
validation split from the TRAIN export; the reported runs do not use one.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
from PIL import Image
from atomic_io import atomic_write_text


OAM_TCD_IMAGE_SIZE = 2048
OAM_TCD_GSD_M_PER_PX = 0.10
SAM3_INPUT_SIZE = 1008
COCO_ANNOTATION_KEY = "coco_annotations"
DEFAULT_CATEGORY_NAME = "tree"
SAM3_TREE_CATEGORY = {"id": 1, "name": DEFAULT_CATEGORY_NAME}


# ── HF dataset loading + image/mask conversion ───────────────────────────────

@dataclass
class TreeSample:
    """One OAM-TCD image with its optional semantic tree mask and the raw dataset row."""
    image: Image.Image
    semantic_mask: np.ndarray | None
    source_index: int
    raw: dict[str, Any]


def load_dataset_dict(dataset_name: str = "restor/tcd", cache_dir: str | Path | None = None):
    """Load the Hugging Face OAM-TCD dataset, optionally from a local cache directory."""
    from datasets import load_dataset

    return load_dataset(dataset_name, cache_dir=str(cache_dir) if cache_dir else None)


def get_split(dataset_dict, split: str = "train"):
    """Return one split from a dataset dictionary, or raise a `KeyError` that lists the available splits."""
    if split in dataset_dict:
        return dataset_dict[split]
    available = ", ".join(dataset_dict.keys())
    raise KeyError(f"Split {split!r} not found. Available splits: {available}")


def load_tree_sample(dataset, index: int = 0, mask_key: str = "annotation") -> TreeSample:
    """Load one sample as a `TreeSample`, converting the image to RGB and the mask to binary."""
    sample = dataset[index]
    image = to_rgb_image(sample["image"])
    mask = to_binary_mask(sample.get(mask_key)) if mask_key in sample else None
    return TreeSample(image=image, semantic_mask=mask, source_index=index, raw=dict(sample))


def to_rgb_image(value: Any) -> Image.Image:
    """Convert an image, path, or array to an RGB PIL image."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    array = np.asarray(value)
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8)).convert("RGB")
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    return Image.fromarray(array.astype(np.uint8)).convert("RGB")


def to_binary_mask(value: Any) -> np.ndarray:
    """Convert a mask image, path, or array to a 0 or 1 array, taking the maximum over channels."""
    if isinstance(value, Image.Image):
        array = np.asarray(value)
    elif isinstance(value, (str, Path)):
        array = np.asarray(Image.open(value))
    else:
        array = np.asarray(value)
    if array.ndim == 3:
        array = array.max(axis=-1)
    return (array > 0).astype(np.uint8)


def semantic_to_instances(mask: np.ndarray, min_area: int = 25) -> list[np.ndarray]:
    """Split a binary semantic mask into connected-component instance masks of at least `min_area` pixels."""
    mask = (mask > 0).astype(np.uint8)
    try:
        from scipy import ndimage

        labels, n_labels = ndimage.label(mask)
        instances = []
        for label_id in range(1, n_labels + 1):
            inst = labels == label_id
            if int(inst.sum()) >= min_area:
                instances.append(inst.astype(np.uint8))
        return instances
    except Exception:
        if int(mask.sum()) >= min_area:
            return [mask]
        return []


def save_sample_preview(sample: TreeSample, output_dir: str | Path) -> None:
    """Write a sample's image and, when present, its semantic mask as PNG previews."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample.image.save(output_dir / f"sample_{sample.source_index:05d}_image.png")
    if sample.semantic_mask is not None:
        Image.fromarray(sample.semantic_mask * 255).save(
            output_dir / f"sample_{sample.source_index:05d}_mask.png"
        )


# ── COCO parsing, validation, cleaning, and export ───────────────────────────

def parse_coco_annotations(value: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse OAM-TCD coco_annotations.

    The HF field is a string containing JSON, not an already-decoded dict.
    This parser accepts decoded values too so saved reports and small tests can
    reuse the same validation path.
    """
    if value is None:
        return [], []
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "null":
            return [], []
        parsed = json.loads(text)
    else:
        parsed = value

    if isinstance(parsed, dict):
        categories = _as_dict_list(parsed.get("categories", []), field_name="categories")
        if "annotations" in parsed:
            return _as_dict_list(parsed["annotations"], field_name="annotations"), categories
        if COCO_ANNOTATION_KEY in parsed:
            return parse_coco_annotations(parsed[COCO_ANNOTATION_KEY])
        if _looks_like_annotation(parsed):
            return [parsed], categories
        raise ValueError(f"Unsupported coco_annotations object keys: {sorted(parsed.keys())}")

    if isinstance(parsed, list):
        return _as_dict_list(parsed, field_name="annotations"), []

    raise TypeError(f"Unsupported coco_annotations type: {type(parsed).__name__}")


def validate_sample(
    sample: dict[str, Any],
    sample_index: int,
    split: str,
    annotation_key: str = COCO_ANNOTATION_KEY,
    default_size: int = OAM_TCD_IMAGE_SIZE,
) -> dict[str, Any]:
    """Validate one sample's annotations and return its report (size, issues, and counts)."""
    width, height = sample_image_size(sample, default_size=default_size)
    report: dict[str, Any] = {
        "split": split,
        "sample_index": sample_index,
        "width": width,
        "height": height,
        "annotation_count": 0,
        "valid_annotation_count": 0,
        "invalid_annotation_count": 0,
        "category_histogram": {},
        "crowd_flags": {},
        "bbox": {"valid": 0, "invalid": 0},
        "segmentation": {"valid": 0, "invalid": 0},
        "issues": [],
    }

    try:
        annotations, _ = parse_coco_annotations(sample.get(annotation_key))
    except Exception as error:
        report["issues"].append(
            {
                "code": "parse_error",
                "message": str(error),
                "annotation_id": None,
            }
        )
        report["invalid_annotation_count"] = 1
        report["is_empty"] = True
        return report

    category_histogram: Counter[str] = Counter()
    crowd_flags: Counter[str] = Counter()
    for local_index, annotation in enumerate(annotations):
        report["annotation_count"] += 1
        category_histogram[str(annotation.get("category_id", "missing"))] += 1
        crowd_flags[str(normalize_iscrowd(annotation))] += 1

        annotation_issues = validate_annotation(annotation, width=width, height=height)
        bbox_is_valid = not any(issue["code"].startswith("bbox_") for issue in annotation_issues)
        seg_is_valid = not any(issue["code"].startswith("segmentation_") for issue in annotation_issues)
        report["bbox"]["valid" if bbox_is_valid else "invalid"] += 1
        report["segmentation"]["valid" if seg_is_valid else "invalid"] += 1

        if annotation_issues:
            report["invalid_annotation_count"] += 1
            for issue in annotation_issues:
                issue["annotation_id"] = annotation.get("id", local_index)
                report["issues"].append(issue)
        else:
            report["valid_annotation_count"] += 1

    report["category_histogram"] = dict(sorted(category_histogram.items()))
    report["crowd_flags"] = dict(sorted(crowd_flags.items()))
    report["is_empty"] = report["annotation_count"] == 0
    return report


def validate_annotation(
    annotation: dict[str, Any],
    width: int = OAM_TCD_IMAGE_SIZE,
    height: int = OAM_TCD_IMAGE_SIZE,
) -> list[dict[str, Any]]:
    """Return the issues found in one COCO annotation (category, box, segmentation, and crowd flag)."""
    issues: list[dict[str, Any]] = []
    if "category_id" not in annotation:
        issues.append({"code": "missing_category_id", "message": "annotation has no category_id"})

    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        issues.append({"code": "bbox_missing_or_malformed", "message": "bbox must be [x, y, width, height]"})
    else:
        issues.extend(validate_bbox(bbox, width=width, height=height))

    segmentation = annotation.get("segmentation")
    issues.extend(validate_segmentation(segmentation, width=width, height=height))

    area = annotation.get("area")
    if area is not None and (not _is_number(area) or float(area) <= 0):
        issues.append({"code": "area_non_positive", "message": "area must be positive when present"})

    return issues


def validate_bbox(bbox: list[Any], width: int, height: int) -> list[dict[str, Any]]:
    """Return the issues found in an `xywh` box, such as non-numeric values or a degenerate or out-of-image extent."""
    issues: list[dict[str, Any]] = []
    if not all(_is_number(value) for value in bbox):
        return [{"code": "bbox_non_numeric", "message": "bbox contains non-numeric values"}]

    x, y, box_width, box_height = [float(value) for value in bbox]
    if box_width <= 0 or box_height <= 0:
        issues.append({"code": "bbox_non_positive_size", "message": "bbox width and height must be positive"})
    if x < 0 or y < 0:
        issues.append({"code": "bbox_negative_origin", "message": "bbox origin must be non-negative"})
    if x + box_width > width + 1e-6 or y + box_height > height + 1e-6:
        issues.append({"code": "bbox_out_of_bounds", "message": "bbox extends outside image bounds"})
    return issues


def validate_segmentation(segmentation: Any, width: int, height: int) -> list[dict[str, Any]]:
    """Return the issues found in a polygon or run-length-encoded segmentation."""
    if segmentation is None:
        return [{"code": "segmentation_missing", "message": "segmentation is missing"}]
    if isinstance(segmentation, str) and not segmentation.strip():
        return [{"code": "segmentation_missing", "message": "segmentation is missing"}]
    if isinstance(segmentation, list) and not segmentation:
        return [{"code": "segmentation_missing", "message": "segmentation is missing"}]

    if isinstance(segmentation, dict):
        return validate_rle_segmentation(segmentation, width=width, height=height)

    if not isinstance(segmentation, list):
        return [{"code": "segmentation_malformed", "message": "segmentation must be polygons or RLE"}]

    issues: list[dict[str, Any]] = []
    polygons = list(iter_polygons(segmentation))
    if not polygons:
        return [{"code": "segmentation_empty_polygon", "message": "segmentation has no polygon coordinates"}]

    for polygon_index, polygon in enumerate(polygons):
        if len(polygon) < 6:
            issues.append(
                {
                    "code": "segmentation_polygon_too_short",
                    "message": f"polygon {polygon_index} has fewer than 3 points",
                }
            )
            continue
        if len(polygon) % 2 != 0:
            issues.append(
                {
                    "code": "segmentation_polygon_odd_length",
                    "message": f"polygon {polygon_index} has an odd coordinate count",
                }
            )
            continue
        if not all(_is_number(value) for value in polygon):
            issues.append(
                {
                    "code": "segmentation_polygon_non_numeric",
                    "message": f"polygon {polygon_index} contains non-numeric values",
                }
            )
            continue

        xs = [float(value) for value in polygon[0::2]]
        ys = [float(value) for value in polygon[1::2]]
        if min(xs) < 0 or min(ys) < 0 or max(xs) > width + 1e-6 or max(ys) > height + 1e-6:
            issues.append(
                {
                    "code": "segmentation_polygon_out_of_bounds",
                    "message": f"polygon {polygon_index} extends outside image bounds",
                }
            )
            
        area = polygon_area(xs, ys)
        if area < 1.0:
            issues.append(
                {
                    "code": "segmentation_polygon_empty",
                    "message": f"polygon {polygon_index} has near-zero area ({area:.2f})",
                }
            )
    return issues


def validate_rle_segmentation(segmentation: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    """Return the issues found in an RLE segmentation, such as missing counts or a size that differs from the image."""
    issues: list[dict[str, Any]] = []
    if "counts" not in segmentation:
        issues.append({"code": "segmentation_rle_missing_counts", "message": "RLE segmentation has no counts"})
    size = segmentation.get("size")
    if size is not None and list(size) != [height, width]:
        issues.append(
            {
                "code": "segmentation_rle_size_mismatch",
                "message": f"RLE size {size} does not match image size {[height, width]}",
            }
        )
    return issues


def validate_dataset_split(
    dataset: Any,
    split: str,
    max_samples: int | None = None,
    bad_example_limit: int = 20,
    annotation_key: str = COCO_ANNOTATION_KEY,
) -> dict[str, Any]:
    """Validate every sample of a split (or the first `max_samples`) and return the summarised report."""
    total_available = len(dataset)
    total_checked = min(total_available, max_samples) if max_samples is not None else total_available
    sample_reports = [
        validate_sample(
            dict(dataset[index]),
            sample_index=index,
            split=split,
            annotation_key=annotation_key,
        )
        for index in range(total_checked)
    ]
    return summarize_sample_reports(
        sample_reports,
        split=split,
        total_available=total_available,
        bad_example_limit=bad_example_limit,
    )


def summarize_sample_reports(
    sample_reports: list[dict[str, Any]],
    split: str,
    total_available: int,
    bad_example_limit: int = 20,
) -> dict[str, Any]:
    """Aggregate per-sample reports into category, crowd-flag, issue, and annotation-count summaries with a bounded list of bad examples."""
    category_histogram: Counter[str] = Counter()
    crowd_flags: Counter[str] = Counter()
    issue_histogram: Counter[str] = Counter()
    annotation_counts: list[int] = []
    bad_examples: list[dict[str, Any]] = []
    empty_samples: list[int] = []

    totals = {
        "annotations": 0,
        "valid_annotations": 0,
        "invalid_annotations": 0,
        "bbox_valid": 0,
        "bbox_invalid": 0,
        "segmentation_valid": 0,
        "segmentation_invalid": 0,
    }

    for report in sample_reports:
        annotation_counts.append(int(report["annotation_count"]))
        totals["annotations"] += int(report["annotation_count"])
        totals["valid_annotations"] += int(report["valid_annotation_count"])
        totals["invalid_annotations"] += int(report["invalid_annotation_count"])
        totals["bbox_valid"] += int(report["bbox"]["valid"])
        totals["bbox_invalid"] += int(report["bbox"]["invalid"])
        totals["segmentation_valid"] += int(report["segmentation"]["valid"])
        totals["segmentation_invalid"] += int(report["segmentation"]["invalid"])
        category_histogram.update(report["category_histogram"])
        crowd_flags.update(report["crowd_flags"])
        if report["is_empty"]:
            empty_samples.append(int(report["sample_index"]))

        for issue in report["issues"]:
            issue_histogram[issue["code"]] += 1
        if report["issues"] and len(bad_examples) < bad_example_limit:
            bad_examples.append(
                {
                    "split": split,
                    "sample_index": report["sample_index"],
                    "annotation_count": report["annotation_count"],
                    "issues": report["issues"][:5],
                }
            )

    return {
        "split": split,
        "total_available": total_available,
        "samples_checked": len(sample_reports),
        "annotation_counts": {
            "total": totals["annotations"],
            "valid": totals["valid_annotations"],
            "invalid": totals["invalid_annotations"],
            "per_sample_min": min(annotation_counts) if annotation_counts else 0,
            "per_sample_median": float(median(annotation_counts)) if annotation_counts else 0.0,
            "per_sample_max": max(annotation_counts) if annotation_counts else 0,
        },
        "category_histogram": dict(sorted(category_histogram.items())),
        "bbox_validity": {"valid": totals["bbox_valid"], "invalid": totals["bbox_invalid"]},
        "segmentation_validity": {
            "valid": totals["segmentation_valid"],
            "invalid": totals["segmentation_invalid"],
        },
        "crowd_flags": dict(sorted(crowd_flags.items())),
        "empty_samples": {
            "count": len(empty_samples),
            "examples": empty_samples[:bad_example_limit],
        },
        "issue_histogram": dict(sorted(issue_histogram.items())),
        "bad_sample_examples": bad_examples,
    }


def sample_to_coco_records(
    sample: dict[str, Any],
    sample_index: int,
    split: str,
    file_name: str,
    image_id: int | None = None,
    annotation_id_start: int = 1,
    used_annotation_ids: set[int] | None = None,
    annotation_key: str = COCO_ANNOTATION_KEY,
    collapse_categories: bool = True,
    clip_to_image: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], int, dict[str, int]]:
    """Convert one dataset sample into COCO image and annotation records, with drop statistics."""
    image_id = sample_index if image_id is None else image_id
    used_annotation_ids = used_annotation_ids if used_annotation_ids is not None else set()
    width, height = sample_image_size(sample)
    annotations, categories = parse_coco_annotations(sample.get(annotation_key))
    image_record = {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "source_split": split,
        "source_index": sample_index,
        "sam3_input_size": SAM3_INPUT_SIZE,
    }

    next_annotation_id = annotation_id_start
    output_annotations: list[dict[str, Any]] = []
    stats = {
        "raw_annotation_count": len(annotations),
        "dropped_annotations_empty_mask": 0,
    }
    for local_index, annotation in enumerate(annotations):
        raw_id = annotation.get("id", local_index)
        annotation_id = _coerce_int(raw_id, default=next_annotation_id)
        if annotation_id in used_annotation_ids:
            annotation_id = next_annotation_id
        used_annotation_ids.add(annotation_id)
        next_annotation_id = max(next_annotation_id + 1, annotation_id + 1)

        output, drop_reason = canonicalize_annotation_with_drop_reason(
            annotation,
            width=width,
            height=height,
            collapse_categories=collapse_categories,
            clip_to_image=clip_to_image,
        )
        if output is None:
            if drop_reason == "empty_mask":
                stats["dropped_annotations_empty_mask"] += 1
            continue
        output["id"] = annotation_id
        output["image_id"] = image_id
        output["source_split"] = split
        output["source_index"] = sample_index
        output["source_annotation_id"] = raw_id
        output_annotations.append(output)

    return image_record, output_annotations, categories, next_annotation_id, stats


def write_dataset_coco_split(
    dataset: Any,
    output_dir: str | Path,
    split: str,
    max_samples: int | None = None,
    image_format: str = "png",
    annotation_key: str = COCO_ANNOTATION_KEY,
    collapse_categories: bool = True,
    clip_to_image: bool = True,
    skip_empty_images: bool = False,
    return_stats: bool = False,
) -> Path | tuple[Path, dict[str, int]]:
    """Export a split as images and a COCO annotation file under `output_dir`, returning the annotation path (and statistics on request)."""
    output_dir = Path(output_dir)
    image_dir = output_dir / "images" / split
    annotation_dir = output_dir / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    total = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
    stale_images_removed = 0
    for stale_image in image_dir.glob(f"oam_tcd_{split}_*.{image_format}"):
        stale_image.unlink()
        stale_images_removed += 1

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    category_by_id: dict[int, dict[str, Any]] = {SAM3_TREE_CATEGORY["id"]: dict(SAM3_TREE_CATEGORY)}
    used_annotation_ids: set[int] = set()
    next_annotation_id = 1
    export_stats = {
        "images_total": total,
        "images_with_no_raw_annotations": 0,
        "images_empty_after_cleaning": 0,
        "images_skipped_no_raw_annotations": 0,
        "images_skipped_empty_after_cleaning": 0,
        "images_retained_empty": 0,
        "images_exported": 0,
        "stale_images_removed": stale_images_removed,
        "dropped_annotations_empty_mask": 0,
    }

    for sample_index in range(total):
        sample = dict(dataset[sample_index])
        file_name = f"images/{split}/oam_tcd_{split}_{sample_index:06d}.{image_format}"
        image_record, annotation_records, category_records, next_annotation_id, sample_stats = sample_to_coco_records(
            sample=sample,
            sample_index=sample_index,
            split=split,
            file_name=file_name,
            image_id=sample_index,
            annotation_id_start=next_annotation_id,
            used_annotation_ids=used_annotation_ids,
            annotation_key=annotation_key,
            collapse_categories=collapse_categories,
            clip_to_image=clip_to_image,
        )
        export_stats["dropped_annotations_empty_mask"] += sample_stats["dropped_annotations_empty_mask"]
        if sample_stats["raw_annotation_count"] == 0:
            export_stats["images_with_no_raw_annotations"] += 1
            if skip_empty_images:
                export_stats["images_skipped_no_raw_annotations"] += 1
                continue
        elif not annotation_records:
            export_stats["images_empty_after_cleaning"] += 1
            if skip_empty_images:
                export_stats["images_skipped_empty_after_cleaning"] += 1
                continue
        if not annotation_records:
            export_stats["images_retained_empty"] += 1

        image = to_rgb_image(sample["image"])
        image.save(output_dir / file_name)
        images.append(image_record)
        annotations.extend(annotation_records)
        export_stats["images_exported"] += 1
        if not collapse_categories:
            for category in category_records:
                if "id" in category:
                    category_by_id[int(category["id"])] = {
                        "id": int(category["id"]),
                        "name": str(category.get("name", DEFAULT_CATEGORY_NAME)),
                    }
        for annotation in annotation_records:
            category_id = int(annotation["category_id"])
            category_by_id.setdefault(category_id, {"id": category_id, "name": _default_category_name(category_id)})

    coco = {
        "info": {
            "description": "OAM-TCD COCO export for SAM3 fine-tuning",
            "source": "restor/tcd",
            "dataset_id": "restor/tcd",
            "dataset_adapter": {"name": "oam_tcd", "version": "1"},
            "dataset_gsd_m_per_px": OAM_TCD_GSD_M_PER_PX,
            "gsd_units": "m_per_pixel",
            "gsd_documentation": "https://huggingface.co/datasets/restor/tcd",
            "coordinate_policy": "Original 2048x2048 coordinate frame preserved; tiny border overshoots clipped to image bounds.",
            "empty_image_policy": (
                "Images with no raw annotations, or no annotations after cleaning, are skipped."
                if skip_empty_images
                else "Images with no raw annotations, or no annotations after cleaning, are retained as zero-annotation true-negative images."
            ),
            "category_policy": "OAM-TCD category IDs are collapsed to one SAM3 text prompt category named tree; original_category_id is preserved per annotation.",
            "sam3_input_size": SAM3_INPUT_SIZE,
        },
        "images": images,
        "annotations": annotations,
        "categories": [category_by_id[key] for key in sorted(category_by_id)],
    }
    output_path = annotation_dir / f"{split}_annotations.coco.json"
    atomic_write_text(output_path, json.dumps(coco, indent=2))
    if return_stats:
        return output_path, export_stats
    return output_path


def canonicalize_annotation(
    annotation: dict[str, Any],
    width: int = OAM_TCD_IMAGE_SIZE,
    height: int = OAM_TCD_IMAGE_SIZE,
    collapse_categories: bool = True,
    clip_to_image: bool = True,
) -> dict[str, Any] | None:
    """Return a cleaned COCO annotation, or `None` when it cannot be kept."""
    output, _ = canonicalize_annotation_with_drop_reason(
        annotation,
        width=width,
        height=height,
        collapse_categories=collapse_categories,
        clip_to_image=clip_to_image,
    )
    return output


def canonicalize_annotation_with_drop_reason(
    annotation: dict[str, Any],
    width: int = OAM_TCD_IMAGE_SIZE,
    height: int = OAM_TCD_IMAGE_SIZE,
    collapse_categories: bool = True,
    clip_to_image: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    """Clean one annotation (category collapse, clipped box, normalised segmentation) and return it with the reason it was dropped, if any."""
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(_is_number(value) for value in bbox):
        return None, "invalid_bbox"

    segmentation = canonicalize_segmentation(
        annotation.get("segmentation"),
        width=width,
        height=height,
        clip_to_image=clip_to_image,
    )
    if segmentation is None:
        return None, "invalid_segmentation"
    if segmentation == []:
        drop_reason = "empty_mask" if segmentation_has_empty_mask(
            annotation.get("segmentation"),
            width=width,
            height=height,
            clip_to_image=clip_to_image,
        ) else "invalid_segmentation"
        return None, drop_reason

    bbox = clip_bbox_xywh([float(value) for value in bbox], width=width, height=height)
    if bbox is None:
        return None, "invalid_bbox"

    area = bbox[2] * bbox[3]
    if area <= 0:
        return None, "invalid_bbox"

    original_category_id = _coerce_int(annotation.get("category_id"), default=1)
    category_id = SAM3_TREE_CATEGORY["id"] if collapse_categories else original_category_id
    return {
        "category_id": category_id,
        "original_category_id": original_category_id,
        "bbox": bbox,
        "segmentation": segmentation,
        "area": float(area),
        "iscrowd": normalize_iscrowd(annotation),
    }, None


def canonicalize_segmentation(
    segmentation: Any,
    width: int,
    height: int,
    clip_to_image: bool = True,
) -> list[list[float]] | dict[str, Any] | None:
    """Normalise a segmentation to clipped polygons or a copied RLE dictionary, or return `None` when it is unusable."""
    if isinstance(segmentation, dict):
        return deepcopy(segmentation)
    if not isinstance(segmentation, list):
        return None

    polygons: list[list[float]] = []
    for polygon in iter_polygons(segmentation):
        if len(polygon) < 6 or len(polygon) % 2 != 0:
            continue
        if not all(_is_number(value) for value in polygon):
            continue
        output_polygon = []
        for coord_index, value in enumerate(polygon):
            upper = width if coord_index % 2 == 0 else height
            coord = float(value)
            if clip_to_image:
                coord = min(max(coord, 0.0), float(upper))
            output_polygon.append(coord)
        xs = output_polygon[0::2]
        ys = output_polygon[1::2]
        if max(xs) <= min(xs) or max(ys) <= min(ys):
            continue
            
        area = polygon_area(xs, ys)
        if area < 1.0:
            continue
            
        polygons.append(output_polygon)
    return polygons


def segmentation_has_empty_mask(
    segmentation: Any,
    width: int,
    height: int,
    clip_to_image: bool = True,
) -> bool:
    """Return whether a segmentation rasterises to an empty mask, after optional clipping to the image."""
    if isinstance(segmentation, list) and not segmentation:
        return True
    if not isinstance(segmentation, list):
        return False

    saw_empty_polygon = False
    saw_non_empty_polygon = False
    for polygon in iter_polygons(segmentation):
        if len(polygon) < 6 or len(polygon) % 2 != 0:
            continue
        if not all(_is_number(value) for value in polygon):
            continue

        output_polygon = []
        for coord_index, value in enumerate(polygon):
            upper = width if coord_index % 2 == 0 else height
            coord = float(value)
            if clip_to_image:
                coord = min(max(coord, 0.0), float(upper))
            output_polygon.append(coord)

        xs = output_polygon[0::2]
        ys = output_polygon[1::2]
        if max(xs) <= min(xs) or max(ys) <= min(ys):
            saw_empty_polygon = True
            continue

        area = polygon_area(xs, ys)
        if area < 1.0:
            saw_empty_polygon = True
        else:
            saw_non_empty_polygon = True

    return saw_empty_polygon and not saw_non_empty_polygon


def polygon_area(xs: list[float], ys: list[float]) -> float:
    """Return the area of a polygon from its x and y coordinate lists (shoelace formula)."""
    return 0.5 * abs(
        sum(
            x0 * y1 - x1 * y0
            for x0, y0, x1, y1 in zip(
                xs,
                ys,
                xs[1:] + xs[:1],
                ys[1:] + ys[:1],
            )
        )
    )


def clip_bbox_xywh(bbox: list[float], width: int, height: int) -> list[float] | None:
    """Clip an `xywh` box to the image and return it, or `None` when nothing remains."""
    x, y, box_width, box_height = bbox
    x1 = min(max(x, 0.0), float(width))
    y1 = min(max(y, 0.0), float(height))
    x2 = min(max(x + box_width, 0.0), float(width))
    y2 = min(max(y + box_height, 0.0), float(height))
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None
    return [x1, y1, clipped_width, clipped_height]


def sample_image_size(sample: dict[str, Any], default_size: int = OAM_TCD_IMAGE_SIZE) -> tuple[int, int]:
    """Return the sample's `(width, height)`, falling back to the native OAM-TCD size."""
    image = sample.get("image")
    if hasattr(image, "size") and len(image.size) == 2:
        width, height = image.size
        return int(width), int(height)
    return default_size, default_size


def normalize_iscrowd(annotation: dict[str, Any]) -> int:
    """Return the crowd flag as an integer, reading `iscrowd` or `is_crowd` and defaulting to 0."""
    value = annotation.get("iscrowd", annotation.get("is_crowd", 0))
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def iter_polygons(segmentation: list[Any]) -> Iterable[list[Any]]:
    """Yield each polygon from a flat or nested polygon list."""
    if not segmentation:
        return
    if all(_is_number(value) for value in segmentation):
        yield segmentation
        return
    for polygon in segmentation:
        if isinstance(polygon, list):
            yield polygon


def _as_dict_list(value: Any, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    if not all(isinstance(item, dict) for item in value):
        raise TypeError(f"{field_name} must contain only objects")
    return value


def _looks_like_annotation(value: dict[str, Any]) -> bool:
    return any(key in value for key in ("bbox", "segmentation", "category_id"))


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _default_category_name(category_id: int) -> str:
    if category_id == 1:
        return DEFAULT_CATEGORY_NAME
    return f"category_{category_id}"


# ── VAL split carving ─────────────────────────────────────────────────────────
#
# OAM-TCD (`restor/tcd`) ships a `validation_fold` column on every TRAIN-split
# row: an integer in [0, 4] assigning each image to one of 5 biome-stratified
# folds (tiles from the same source OAM image are isolated to a single fold,
# so there is no leakage). `main()` below:
#
# 1. Reads the already-exported `train_annotations.coco.json` (built by
#    `write_dataset_coco_split` from HF split="train").
# 2. Loads only the `validation_fold` column from the HF dataset (cheap: no
#    image decoding) and maps it onto each COCO image via `source_index`,
#    which `sample_to_coco_records` sets to the HF row index.
# 3. Partitions images/annotations into:
#    - `val_annotations.coco.json`   (validation_fold == --val-fold)
#    - `train_split_annotations.coco.json`  (validation_fold != --val-fold)
#    The original `train_annotations.coco.json` (all folds) is left untouched
#    as a full-train reference.
#
# TEST (`test_annotations.coco.json`) is never touched by this and remains
# reserved for the final headline numbers.
#
# Usage (run wherever the full train export + HF cache live, e.g. HPC):
#
#     python Core/dataset_adapter.py --data-root runtime/data/oam_tcd_instance_coco \
#         --cache-dir runtime/data/hf_cache --val-fold 0

def _load_validation_folds(dataset_name: str, cache_dir: str | None) -> list[int]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    if "validation_fold" not in ds.column_names:
        raise KeyError(
            f"'validation_fold' column not found on {dataset_name}[train]; "
            f"available columns: {ds.column_names}"
        )
    return list(ds["validation_fold"])


def _split_coco(
    coco: dict[str, Any], folds_by_source_index: list[int], val_fold: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    val_image_ids: set[int] = set()
    train_image_ids: set[int] = set()
    skipped_no_fold = 0

    for img in coco["images"]:
        src_idx = img["source_index"]
        if src_idx < 0 or src_idx >= len(folds_by_source_index):
            skipped_no_fold += 1
            continue
        fold = folds_by_source_index[src_idx]
        if fold < 0:
            # -1 = holdout marker in the raw dataset; should not appear in
            # the train split, but guard defensively and exclude from both.
            skipped_no_fold += 1
            continue
        if fold == val_fold:
            val_image_ids.add(img["id"])
        else:
            train_image_ids.add(img["id"])

    val_images = [img for img in coco["images"] if img["id"] in val_image_ids]
    train_images = [img for img in coco["images"] if img["id"] in train_image_ids]
    val_anns = [ann for ann in coco["annotations"] if ann["image_id"] in val_image_ids]
    train_anns = [ann for ann in coco["annotations"] if ann["image_id"] in train_image_ids]

    base_info = dict(coco.get("info", {}))
    val_coco = {
        "info": {**base_info, "description": base_info.get("description", "") + " (VAL fold split)"},
        "images": val_images,
        "annotations": val_anns,
        "categories": coco["categories"],
    }
    train_coco = {
        "info": {**base_info, "description": base_info.get("description", "") + " (TRAIN minus VAL fold)"},
        "images": train_images,
        "annotations": train_anns,
        "categories": coco["categories"],
    }
    stats = {
        "val_fold": val_fold,
        "val_images": len(val_images),
        "val_annotations": len(val_anns),
        "train_images": len(train_images),
        "train_annotations": len(train_anns),
        "skipped_no_fold": skipped_no_fold,
    }
    return val_coco, train_coco, stats


def main() -> None:
    """Carve a leakage-safe validation split from the exported TRAIN COCO file (command-line entry point)."""
    p = argparse.ArgumentParser(
        description="Carve an official, leakage-safe VAL split out of the exported TRAIN COCO file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-root", required=True, type=str, help="oam_tcd_instance_coco export dir.")
    p.add_argument("--dataset-name", default="restor/tcd", type=str)
    p.add_argument("--cache-dir", default="", type=str, help="HF datasets cache dir (reuse existing download).")
    p.add_argument("--val-fold", type=int, default=0, help="Which validation_fold [0-4] to hold out as VAL.")
    args = p.parse_args()

    data_root = Path(args.data_root)
    train_ann_path = data_root / "annotations" / "train_annotations.coco.json"
    if not train_ann_path.exists():
        raise FileNotFoundError(f"Expected exported train annotations at {train_ann_path}")

    print(f"Loading {train_ann_path} ...")
    coco = json.loads(train_ann_path.read_text())
    print(f"  {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    print(f"Loading '{args.dataset_name}'[train]['validation_fold'] "
          f"(cache_dir={args.cache_dir or 'default'}) ...")
    folds = _load_validation_folds(args.dataset_name, args.cache_dir or None)
    fold_counts = Counter(folds)
    print(f"  fold distribution: {dict(sorted(fold_counts.items()))}")

    val_coco, train_coco, stats = _split_coco(coco, folds, args.val_fold)
    print(f"\nSplit result (val_fold={args.val_fold}):")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if stats["val_images"] == 0 or stats["train_images"] == 0:
        raise RuntimeError("Split produced an empty VAL or TRAIN set; check --val-fold and source_index alignment.")

    val_out = data_root / "annotations" / "val_annotations.coco.json"
    train_out = data_root / "annotations" / "train_split_annotations.coco.json"
    atomic_write_text(val_out, json.dumps(val_coco))
    atomic_write_text(train_out, json.dumps(train_coco))
    print(f"\nWrote {val_out}")
    print(f"Wrote {train_out}")
    print("\nNext: point Core/config.py train_ann_file -> train_split_annotations.coco.json "
          "and val_ann_file -> val_annotations.coco.json (leave test_annotations.coco.json untouched).")


if __name__ == "__main__":
    main()
