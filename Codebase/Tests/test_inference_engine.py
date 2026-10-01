from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

# Tests live in Codebase/Tests/; the modules under test live in Codebase/Core/.
# Core is put on the path explicitly so the suite does not depend on the working directory.
CORE_DIR = Path(__file__).resolve().parent.parent / "Core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import inference_engine


class StaticBackend:
    def __init__(self, masks: np.ndarray, scores: np.ndarray) -> None:
        self.masks = masks
        self.scores = scores
        self.resize_flags: list[bool] = []

    def predict(self, image: np.ndarray, *, resize_to_model: bool) -> inference_engine.BackendPrediction:
        self.resize_flags.append(resize_to_model)
        return inference_engine.BackendPrediction(masks=self.masks, scores=self.scores)


def test_whole_image_lifts_model_coordinates_to_source_extent():
    backend = StaticBackend(
        masks=np.array([[[0, 1], [0, 1]]], dtype=bool),
        scores=np.array([0.9]),
    )
    instances = inference_engine.instances_from_window(
        backend,
        np.zeros((2, 2, 3), dtype=np.uint8),
        source_x0=0,
        source_y0=0,
        source_width=8,
        source_height=4,
        image_height=4,
        image_width=8,
        image_id=7,
        category_id=1,
        window_id="7:whole-image",
        window_row=0,
        window_col=0,
        min_area=1,
        edge_margin=1,
        resize_to_model=False,
        is_whole_image=True,
    )
    assert backend.resize_flags == [False]
    assert len(instances) == 1
    instance = instances[0]
    assert instance["bbox"] == [4.0, 0.0, 4.0, 4.0]
    assert instance["is_full_pass"] is True
    assert int(mask_utils.area(instance["segmentation"])) == 16


def test_window_lifts_offsets_and_preserves_backend_resize_request():
    backend = StaticBackend(
        masks=np.array([[[1, 1], [1, 1]]], dtype=bool),
        scores=np.array([0.8]),
    )
    instances = inference_engine.instances_from_window(
        backend,
        np.zeros((2, 2, 3), dtype=np.uint8),
        source_x0=3,
        source_y0=2,
        source_width=4,
        source_height=4,
        image_height=12,
        image_width=12,
        image_id=8,
        category_id=1,
        window_id="8:r0:c0",
        window_row=0,
        window_col=0,
        min_area=1,
        edge_margin=1,
        resize_to_model=True,
        is_whole_image=False,
    )
    assert backend.resize_flags == [True]
    assert instances[0]["bbox"] == [3.0, 2.0, 4.0, 4.0]
    assert instances[0]["tile_bbox"] == [3, 2, 4, 4]


def test_greedy_merge_keeps_highest_scoring_duplicate():
    mask = np.ones((4, 4), dtype=np.uint8, order="F")
    rle = mask_utils.encode(mask)
    instances = [
        {"score": 0.8, "segmentation": rle, "bbox": [0.0, 0.0, 4.0, 4.0]},
        {"score": 0.9, "segmentation": rle, "bbox": [0.0, 0.0, 4.0, 4.0]},
    ]
    merged = inference_engine.merge_instances(
        instances,
        image_id=1,
        category_id=1,
        dedup_iou=0.5,
        maxdets=0,
        merge_method="mask-nms",
    )
    assert len(merged) == 1
    assert merged[0]["score"] == 0.9
