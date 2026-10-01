"""Validation of the OAM-TCD TEST export against the locked reporting protocol.

Checks the image and annotation counts, the native 2048 pixel frame, the collapsed category list,
and the source split, then returns the protocol record (including the annotation file hash) that
evaluation results embed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from atomic_io import atomic_write_text


OAM_TCD_TEST_IMAGE_COUNT = 439
OAM_TCD_TEST_ANNOTATION_COUNT = 30_646
OAM_TCD_IMAGE_SIZE = 2048
OAM_TCD_CATEGORIES = [{"id": 1, "name": "tree"}]
OAM_TCD_PROTOCOL_NAME = "oam_tcd_project_test_export_v1"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_oam_tcd_test(annotation_path: str | Path) -> dict[str, Any]:
    """Validate the TEST COCO export (439 images, 30,646 annotations, native 2048 pixel frame, collapsed category) and return its protocol record with file hashes."""
    path = Path(annotation_path)
    with path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images = coco.get("images")
    annotations = coco.get("annotations")
    categories = coco.get("categories")
    if not isinstance(images, list) or not isinstance(annotations, list) or not isinstance(categories, list):
        raise ValueError("OAM-TCD annotation file must contain COCO images, annotations, and categories lists.")
    if len(images) != OAM_TCD_TEST_IMAGE_COUNT:
        raise ValueError(
            f"Expected the OAM-TCD project test export to contain {OAM_TCD_TEST_IMAGE_COUNT} images; "
            f"found {len(images)} in {path}."
        )
    if len(annotations) != OAM_TCD_TEST_ANNOTATION_COUNT:
        raise ValueError(
            f"Expected the OAM-TCD project test export to contain {OAM_TCD_TEST_ANNOTATION_COUNT} "
            f"annotations; found {len(annotations)} in {path}."
        )
    if categories != OAM_TCD_CATEGORIES:
        raise ValueError(f"Expected collapsed SAM3 category mapping {OAM_TCD_CATEGORIES}; found {categories}.")

    image_ids = [int(image["id"]) for image in images]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("OAM-TCD test export contains duplicate image IDs.")
    if any((image.get("width"), image.get("height")) != (OAM_TCD_IMAGE_SIZE, OAM_TCD_IMAGE_SIZE) for image in images):
        raise ValueError("Every OAM-TCD test image must remain in the native 2048x2048 coordinate frame.")
    split_values = {image.get("source_split") for image in images}
    if split_values != {"test"}:
        raise ValueError(f"Expected every OAM-TCD test image to have source_split='test'; found {split_values}.")

    image_id_set = set(image_ids)
    annotation_image_ids = {int(annotation["image_id"]) for annotation in annotations}
    if not annotation_image_ids <= image_id_set:
        raise ValueError("OAM-TCD test export contains annotations for images outside its image list.")
    if any(int(annotation.get("category_id", -1)) != 1 for annotation in annotations):
        raise ValueError("OAM-TCD project annotations must use the collapsed category_id=1.")
    original_category_ids = sorted({int(annotation["original_category_id"]) for annotation in annotations if "original_category_id" in annotation})

    info = coco.get("info", {})
    return {
        "protocol_name": OAM_TCD_PROTOCOL_NAME,
        "annotation_path": str(path.resolve()),
        "annotation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "image_ids_sha256": _sha256_json(sorted(image_ids)),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "image_size": [OAM_TCD_IMAGE_SIZE, OAM_TCD_IMAGE_SIZE],
        "source_splits": sorted(split_values),
        "categories": categories,
        "category_policy": info.get("category_policy"),
        "original_category_ids_present": original_category_ids,
        "images_with_annotations": len(annotation_image_ids),
        "zero_annotation_images": len(image_id_set - annotation_image_ids),
    }


def main() -> None:
    """Validate an annotation file and write the protocol record as JSON."""
    parser = argparse.ArgumentParser(description="Validate the local OAM-TCD COCO test protocol.")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = validate_oam_tcd_test(args.annotations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.out, json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
