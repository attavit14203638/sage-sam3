#!/usr/bin/env python3
"""Detectron2 Mask R-CNN R50 fine-tuning on OAM-TCD.

Training constants are sourced from Core/config.py (MASKRCNN_* block).
The pipeline:
  1. Registers OAM-TCD train/val COCO datasets with Detectron2.
  2. Builds a Detectron2 config from the COCO-pretrained R50-FPN baseline.
  3. Trains with DefaultTrainer + COCOEvaluator on the val split.
  4. Saves the final model and a config.yaml to the experiment directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from config import (
    MASKRCNN_BASE_CONFIG,
    MASKRCNN_BASE_LR,
    MASKRCNN_CHECKPOINT_PERIOD,
    MASKRCNN_COCO_WEIGHTS,
    MASKRCNN_CROP_SIZE,
    MASKRCNN_DETECTIONS_PER_IMAGE,
    MASKRCNN_MAX_SIZE_TRAIN,
    MASKRCNN_MIN_SIZE_TRAIN,
    MASKRCNN_GAMMA,
    MASKRCNN_IMS_PER_BATCH,
    MASKRCNN_MAX_ITER,
    MASKRCNN_NUM_CLASSES,
    MASKRCNN_NUM_WORKERS,
    MASKRCNN_RUN_NAME,
    MASKRCNN_SCORE_THRESH_TEST,
    MASKRCNN_SEED,
    MASKRCNN_STEPS,
    MASKRCNN_TEST_PERIOD,
    VAL_ANN_FILENAME,
)
from config import MASKRCNN_TRAIN_ANN_FILENAME
from project_paths import CODEBASE_DIR, EXPERIMENTS_DIR, ensure_runtime_dirs

DATASET_TRAIN = "oam_tcd_train"
DATASET_VAL = "oam_tcd_val"


def register_datasets(data_root: Path) -> tuple[str, str]:
    """Register the TRAIN and validation COCO files with Detectron2 and return their dataset names."""
    from detectron2.data import MetadataCatalog
    from detectron2.data.datasets import register_coco_instances

    train_json = str(data_root / "annotations" / MASKRCNN_TRAIN_ANN_FILENAME)
    val_json = str(data_root / "annotations" / VAL_ANN_FILENAME)
    image_root = str(data_root)

    for name, json_path in ((DATASET_TRAIN, train_json), (DATASET_VAL, val_json)):
        if name in MetadataCatalog.list():
            MetadataCatalog.remove(name)
    register_coco_instances(DATASET_TRAIN, {}, train_json, image_root)
    register_coco_instances(DATASET_VAL, {}, val_json, image_root)
    return train_json, val_json


def build_cfg(
    data_root: Path,
    output_dir: Path,
    *,
    resume_from: Path | None = None,
) -> Any:
    """Build the Detectron2 Mask R-CNN configuration for a training run."""
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MASKRCNN_BASE_CONFIG))

    cfg.DATASETS.TRAIN = (DATASET_TRAIN,)
    cfg.DATASETS.TEST = (DATASET_VAL,)
    cfg.DATALOADER.NUM_WORKERS = MASKRCNN_NUM_WORKERS

    cfg.MODEL.WEIGHTS = MASKRCNN_COCO_WEIGHTS
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = MASKRCNN_NUM_CLASSES

    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.SIZE = MASKRCNN_CROP_SIZE
    cfg.INPUT.CROP.TYPE = "absolute"
    cfg.INPUT.MIN_SIZE_TRAIN = list(MASKRCNN_MIN_SIZE_TRAIN)
    cfg.INPUT.MAX_SIZE_TRAIN = MASKRCNN_MAX_SIZE_TRAIN

    cfg.SOLVER.IMS_PER_BATCH = MASKRCNN_IMS_PER_BATCH
    cfg.SOLVER.BASE_LR = MASKRCNN_BASE_LR
    cfg.SOLVER.MAX_ITER = MASKRCNN_MAX_ITER
    cfg.SOLVER.STEPS = list(MASKRCNN_STEPS)
    cfg.SOLVER.GAMMA = MASKRCNN_GAMMA
    cfg.SOLVER.CHECKPOINT_PERIOD = MASKRCNN_CHECKPOINT_PERIOD

    cfg.TEST.EVAL_PERIOD = MASKRCNN_TEST_PERIOD
    cfg.TEST.DETECTIONS_PER_IMAGE = MASKRCNN_DETECTIONS_PER_IMAGE
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = MASKRCNN_SCORE_THRESH_TEST

    cfg.OUTPUT_DIR = str(output_dir)
    cfg.SEED = MASKRCNN_SEED

    if resume_from is not None:
        cfg.MODEL.WEIGHTS = str(resume_from)

    return cfg


def setup_trainer(cfg: Any) -> Any:
    """Build the Detectron2 trainer with the project's evaluator and best-checkpoint hook."""
    from detectron2.engine import BestCheckpointer, DefaultTrainer

    class OAMTCDTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name):
            from detectron2.evaluation import COCOEvaluator

            output_dir = os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, output_dir=output_dir)

    trainer = OAMTCDTrainer(cfg)
    trainer.register_hooks(
        [
            BestCheckpointer(
                cfg.TEST.EVAL_PERIOD,
                trainer.checkpointer,
                "segm/AP50",
                "max",
                file_prefix="model_best",
            )
        ]
    )
    return trainer


def snapshot_config_module(log_dir: Path) -> Path | None:
    """Copy `config.py` into the run directory as `config_snapshot.py` and return its path, or `None` when the source is missing."""
    import config as config_module

    source = Path(config_module.__file__).resolve()
    if not source.exists():
        return None
    target = log_dir / "config_snapshot.py"
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def main() -> None:
    """Command-line entry point for the Mask R-CNN fine-tune."""
    parser = argparse.ArgumentParser(description="Detectron2 Mask R-CNN fine-tuning on OAM-TCD.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--experiment-log-dir", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    args = parser.parse_args()

    ensure_runtime_dirs()

    data_root = (args.data_root or CODEBASE_DIR / "runtime" / "data" / "oam_tcd_instance_coco").resolve()
    output_dir = (args.experiment_log_dir or EXPERIMENTS_DIR / MASKRCNN_RUN_NAME).resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    for ann_file in (MASKRCNN_TRAIN_ANN_FILENAME, VAL_ANN_FILENAME):
        if not (data_root / "annotations" / ann_file).is_file():
            raise FileNotFoundError(f"Missing annotation file: {data_root / 'annotations' / ann_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    register_datasets(data_root)
    cfg = build_cfg(data_root, output_dir, resume_from=args.resume_from)

    cfg_path = output_dir / "config.yaml"
    with cfg_path.open("w") as f:
        f.write(cfg.dump())
    print("Detectron2 config saved:", cfg_path)

    snapshot_path = snapshot_config_module(output_dir)
    print("Config snapshot:", snapshot_path)
    print("Data root:", data_root)
    print("Output dir:", output_dir)
    print("Max iters:", MASKRCNN_MAX_ITER)
    print("Base LR:", MASKRCNN_BASE_LR)
    print("Batch size:", MASKRCNN_IMS_PER_BATCH)
    print("Num classes:", MASKRCNN_NUM_CLASSES)
    print("Train annotations:", MASKRCNN_TRAIN_ANN_FILENAME)
    print("Min size train:", MASKRCNN_MIN_SIZE_TRAIN)
    print("Max size train:", MASKRCNN_MAX_SIZE_TRAIN)
    print("Crop size:", MASKRCNN_CROP_SIZE)
    print("Detections per image:", MASKRCNN_DETECTIONS_PER_IMAGE)

    trainer = setup_trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    final_weights = output_dir / "model_final.pth"
    print("Training complete. Final weights:", final_weights)


if __name__ == "__main__":
    main()
