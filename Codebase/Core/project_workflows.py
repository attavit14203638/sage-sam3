"""Notebook-facing workflows for assets, training launches, inference, and evaluation.

`AssetWorkflow` prepares the pretrained checkpoints and the OAM-TCD COCO export. `ExperimentWorkflow`
validates a run, launches training, and inspects its logs. Module-level helpers run tiled inference
and the locked evaluation, read metric rows, and support the explicit resume of an incomplete run
(checkpoint staging, backups, and a resume ledger). Existing run artefacts are never overwritten.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from atomic_io import atomic_write_text


def _utc_now() -> str:
    """ISO-8601 UTC timestamp for ledger rows."""
    return datetime.now(timezone.utc).isoformat()


def resume_paths(codebase: Path, run_name: str) -> dict[str, Path]:
    """The three recovery paths for a run, all outside the experiment dir.

    Derived from `run_name` so arms cannot collide, and kept in one place so the
    resume notebook and any script agree on where a run's recovery state lives.
    All three must sit outside `experiments/<run>/`, which the launcher wipes.
    """
    runtime = Path(codebase) / "runtime"
    return {
        "resume_from": runtime / f"{run_name}_resume.pt",
        "backup_dir": runtime / f"{run_name}_crash_backup",
        "ledger": runtime / f"{run_name}_resume_ledger.jsonl",
    }

from config import BOUNDARY_BAND_K, FULL_FT_MAX_DATA_EPOCHS, FULL_FT_RUN_NAME, FULL_FT_VAL_EPOCH_FREQ, HPC_LOAD_MODULES, MASKRCNN_RUN_NAME, MODEL_BACKEND
from project_paths import CHECKPOINT_DIR, CODEBASE_DIR, DATA_DIR, EXPERIMENTS_DIR, add_support_to_pythonpath, assert_experiment_directory_available, ensure_runtime_dirs


OAM_DATASET_NAME = "restor/tcd"
OAM_SPLITS = ("train", "test")


@dataclass(frozen=True)
class AssetWorkflow:
    """Prepare the pretrained checkpoints and the OAM-TCD COCO export, skipping steps whose outputs already exist."""
    force_download_assets: bool
    force_export_full: bool
    force_val_split: bool
    val_fold: int
    dataset_name: str
    export_dir: Path
    cache_dir: Path
    checkpoint_dir: Path
    asset_manifest: Path
    export_report: Path

    def print_summary(self) -> None:
        """Print the paths this workflow uses."""
        print("CODEBASE =", CODEBASE_DIR)
        print("Python =", sys.executable)
        print("HF cache =", self.cache_dir)
        print("SAM3 checkpoint dir =", self.checkpoint_dir)
        print("production export =", self.export_dir)
        print("FORCE_DOWNLOAD_ASSETS =", self.force_download_assets)
        print("FORCE_EXPORT_FULL =", self.force_export_full)
        print("FORCE_VAL_SPLIT =", self.force_val_split)
        print("VAL_FOLD =", self.val_fold)

    def ensure_checkpoint(self) -> None:
        """Download the SAM3 checkpoint unless it and its manifest already exist."""
        ready = (self.checkpoint_dir / "sam3.pt").is_file() and (self.checkpoint_dir / "config.json").is_file()
        if ready and self.asset_manifest.is_file() and not self.force_download_assets:
            print("Asset download skipped; checkpoint files and manifest already exist.")
            print(self.asset_manifest.read_text()[:4000])
            return
        download_sam3_checkpoint(self.checkpoint_dir)

    def ensure_maskrcnn_checkpoint(self) -> None:
        """Download the released Mask R-CNN checkpoint unless it and its manifest already exist."""
        maskrcnn_dir = CHECKPOINT_DIR / "maskrcnn_r50"
        ready = (maskrcnn_dir / "model.pth").is_file() and (maskrcnn_dir / "config.yaml").is_file()
        manifest_path = CHECKPOINT_DIR / "maskrcnn_asset_manifest.json"
        if ready and manifest_path.is_file() and not self.force_download_assets:
            print("Mask R-CNN checkpoint download skipped; files and manifest already exist.")
            print(manifest_path.read_text()[:4000])
            return
        download_maskrcnn_checkpoint(maskrcnn_dir)

    def ensure_export(self) -> dict[str, Any]:
        """Export the OAM-TCD splits as COCO, or validate and reuse an existing export."""
        if self._export_ready() and not self.force_export_full:
            print("Production COCO export skipped; existing export is ready.")
            if self._export_report_policy_ready():
                report = json.loads(self.export_report.read_text())
                print("Existing export report is policy-ready:", self.export_report)
            else:
                report = self._report_from_existing_export()
                self.export_report.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(self.export_report, json.dumps(report, indent=2))
                print("Existing export report was missing or stale; refreshed report only:", self.export_report)
        else:
            from dataset_adapter import (
                COCO_ANNOTATION_KEY,
                OAM_TCD_IMAGE_SIZE,
                SAM3_INPUT_SIZE,
                get_split,
                load_dataset_dict,
                validate_dataset_split,
                write_dataset_coco_split,
            )

            dataset_dict = load_dataset_dict(self.dataset_name, cache_dir=self.cache_dir)
            report = {
                "dataset_name": self.dataset_name,
                "cache_dir": str(self.cache_dir.resolve()),
                "annotation_key": COCO_ANNOTATION_KEY,
                "coordinate_policy": {
                    "oam_tcd_image_size": OAM_TCD_IMAGE_SIZE,
                    "sam3_input_size": SAM3_INPUT_SIZE,
                    "note": "Export keeps original OAM-TCD coordinates; SAM3 transforms resize inputs.",
                    "export_cleaning": "Default export clips tiny border overshoots, drops malformed/zero-area annotations, retains images with no usable annotations as zero-annotation true negatives, and collapses categories to tree.",
                },
                "splits": {},
                "exports": {},
            }
            for split in OAM_SPLITS:
                dataset = get_split(dataset_dict, split)
                report["splits"][split] = validate_dataset_split(dataset, split=split)
                export_path, export_stats = write_dataset_coco_split(
                    dataset,
                    output_dir=self.export_dir,
                    split=split,
                    annotation_key=COCO_ANNOTATION_KEY,
                    collapse_categories=True,
                    clip_to_image=True,
                    skip_empty_images=False,
                    return_stats=True,
                )
                report["exports"][split] = {"annotation_path": str(export_path.resolve()), **export_stats}
            self.export_report.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.export_report, json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
            print("Saved export report:", self.export_report)
        summary = self.export_summary()
        self._validate_export_summary(summary, report)
        print(json.dumps(summary, indent=2))
        return report

    def ensure_validation_split(self) -> tuple[Path, Path]:
        """Carve the validation split from TRAIN unless the split files already exist."""
        val_path = self.export_dir / "annotations" / "val_annotations.coco.json"
        train_path = self.export_dir / "annotations" / "train_split_annotations.coco.json"
        if val_path.is_file() and train_path.is_file() and not self.force_val_split:
            print("VAL split skipped; existing files already present.")
        else:
            self._run_command([
                sys.executable,
                CODEBASE_DIR / "Core" / "dataset_adapter.py",
                "--data-root",
                self.export_dir,
                "--cache-dir",
                self.cache_dir,
                "--val-fold",
                self.val_fold,
            ])
        if not val_path.is_file() or not train_path.is_file():
            raise FileNotFoundError("VAL split creation did not produce both annotation files.")
        print("VAL split ready:", val_path)
        print("TRAIN-split (minus VAL) ready:", train_path)
        return val_path, train_path

    def asset_record(self) -> dict[str, Any]:
        """Return a record of which checkpoint and cache assets exist."""
        return {
            "asset_manifest_exists": self.asset_manifest.is_file(),
            "sam3_checkpoint_exists": (self.checkpoint_dir / "sam3.pt").is_file(),
            "sam3_config_exists": (self.checkpoint_dir / "config.json").is_file(),
            "hf_cache_dir": str(self.cache_dir),
            "export_report_exists": self.export_report.is_file(),
            "export_report": str(self.export_report),
            "production_export_ready": self._export_ready(),
            "production_export_dir": str(self.export_dir),
        }

    def export_summary(self) -> dict[str, dict[str, Any]]:
        """Summarise each exported split (counts, categories, and missing images)."""
        summary: dict[str, dict[str, Any]] = {}
        for split in OAM_SPLITS:
            annotation_path = self.export_dir / "annotations" / f"{split}_annotations.coco.json"
            data = json.loads(annotation_path.read_text())
            missing = [image["file_name"] for image in data["images"] if not (self.export_dir / image["file_name"]).is_file()]
            image_ids_with_annotations = {annotation["image_id"] for annotation in data["annotations"]}
            summary[split] = {
                "annotation_path": str(annotation_path),
                "images": len(data["images"]),
                "annotations": len(data["annotations"]),
                "categories": data["categories"],
                "missing_image_count": len(missing),
                "missing_image_examples": missing[:5],
                "empty_images": sum(1 for image in data["images"] if image["id"] not in image_ids_with_annotations),
            }
        return summary

    def write_gt_count_distribution(self, buckets: tuple[int, ...] = (0, 1, 10, 50, 100, 150, 200, 300, 500, 1000)) -> Path:
        """Write the per-image ground-truth count distribution of each split and return the report path."""
        report: dict[str, Any] = {"buckets": list(buckets), "splits": {}}
        for split in OAM_SPLITS:
            annotation_path = self.export_dir / "annotations" / f"{split}_annotations.coco.json"
            coco = json.loads(annotation_path.read_text())
            per_image = Counter(annotation["image_id"] for annotation in coco["annotations"])
            counts = [per_image.get(image["id"], 0) for image in coco["images"]]
            report["splits"][split] = {
                "annotation_path": str(annotation_path),
                "summary": _summarize_counts(counts),
                "histogram_buckets": _bucketize(counts, buckets),
            }
        output_path = self.export_dir / "reports" / "gt_count_distribution.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(output_path, json.dumps(report, indent=2))
        print("Saved GT count distribution report:", output_path)
        print(json.dumps(report, indent=2))
        return output_path

    def _split_ready(self, split: str) -> bool:
        annotation_path = self.export_dir / "annotations" / f"{split}_annotations.coco.json"
        if not annotation_path.is_file():
            return False
        data = json.loads(annotation_path.read_text())
        return bool(data.get("images")) and (self.export_dir / data["images"][0]["file_name"]).is_file()

    def _export_ready(self) -> bool:
        return all(self._split_ready(split) for split in OAM_SPLITS)

    def _export_report_policy_ready(self) -> bool:
        if not self.export_report.is_file():
            return False
        try:
            report = json.loads(self.export_report.read_text())
        except json.JSONDecodeError:
            return False
        required = {
            "images_exported",
            "images_skipped_no_raw_annotations",
            "images_skipped_empty_after_cleaning",
            "images_retained_empty",
        }
        return all(required <= set(report.get("exports", {}).get(split, {})) for split in OAM_SPLITS)

    def _report_from_existing_export(self) -> dict[str, Any]:
        from dataset_adapter import COCO_ANNOTATION_KEY, OAM_TCD_IMAGE_SIZE, SAM3_INPUT_SIZE

        summary = self.export_summary()
        return {
            "dataset_name": self.dataset_name,
            "cache_dir": str(self.cache_dir.resolve()),
            "annotation_key": COCO_ANNOTATION_KEY,
            "coordinate_policy": {
                "oam_tcd_image_size": OAM_TCD_IMAGE_SIZE,
                "sam3_input_size": SAM3_INPUT_SIZE,
                "note": "Export keeps original OAM-TCD coordinates; SAM3 transforms resize inputs.",
                "export_cleaning": "Default export clips tiny border overshoots, drops malformed/zero-area annotations, retains images with no usable annotations as zero-annotation true negatives, and collapses categories to tree.",
            },
            "splits": {},
            "exports": {
                split: {
                    "annotation_path": str((self.export_dir / "annotations" / f"{split}_annotations.coco.json").resolve()),
                    "images_exported": summary[split]["images"],
                    "annotations_exported": summary[split]["annotations"],
                    "images_skipped_no_raw_annotations": 0,
                    "images_skipped_empty_after_cleaning": 0,
                    "images_retained_empty": summary[split]["empty_images"],
                }
                for split in OAM_SPLITS
            },
        }

    @staticmethod
    def _validate_export_summary(summary: dict[str, dict[str, Any]], report: dict[str, Any]) -> None:
        for split in OAM_SPLITS:
            if summary[split]["categories"] != [{"id": 1, "name": "tree"}]:
                raise ValueError(f"{split} export categories are not the canonical tree category.")
            if summary[split]["missing_image_count"] != 0 or summary[split]["images"] <= 0:
                raise ValueError(f"{split} export has missing images or no images.")
            if summary[split]["empty_images"] != report["exports"][split]["images_retained_empty"]:
                raise ValueError(f"{split} export report does not match the COCO export.")

    @staticmethod
    def _run_command(command: list[str | Path]) -> None:
        print("Running:")
        print(" ".join(str(part) for part in command))
        result = subprocess.run([str(part) for part in command], cwd=CODEBASE_DIR, env={**os.environ, "PYTHONUNBUFFERED": "1"}, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {result.returncode}")


def download_sam3_checkpoint(checkpoint_dir: Path = CHECKPOINT_DIR / "sam3") -> dict[str, Any]:
    """Download the SAM3 checkpoint files from the Hugging Face Hub and return a record of them."""
    ensure_runtime_dirs()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    downloaded = [
        hf_hub_download(repo_id="facebook/sam3", filename=filename, local_dir=str(checkpoint_dir))
        for filename in ("config.json", "sam3.pt")
    ]
    manifest = {
        "sam3": {
            "repo_id": "facebook/sam3",
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "files": downloaded,
        }
    }
    manifest_path = checkpoint_dir.parent / "asset_manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"Saved manifest: {manifest_path}")
    return manifest


MASKRCNN_REPO_ID = "restor/tcd-mask-rcnn-r50"
MASKRCNN_FILES = ("config.yaml", "model.pth")


def download_maskrcnn_checkpoint(checkpoint_dir: Path = CHECKPOINT_DIR / "maskrcnn_r50") -> dict[str, Any]:
    """Download the released Mask R-CNN configuration and weights from the Hugging Face Hub and return a record of them."""
    ensure_runtime_dirs()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    downloaded = [
        hf_hub_download(repo_id=MASKRCNN_REPO_ID, filename=filename, local_dir=str(checkpoint_dir))
        for filename in MASKRCNN_FILES
    ]
    manifest = {
        "maskrcnn_r50": {
            "repo_id": MASKRCNN_REPO_ID,
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "files": downloaded,
            "architecture": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x",
            "num_classes": 2,
            "score_thresh_test": 0.2,
            "detections_per_image": 512,
        }
    }
    manifest_path = checkpoint_dir.parent / "maskrcnn_asset_manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"Saved manifest: {manifest_path}")
    return manifest


def create_asset_workflow(
    *,
    force_download_assets: bool = False,
    force_export_full: bool = False,
    force_val_split: bool = False,
    val_fold: int = 0,
) -> AssetWorkflow:
    """Create an `AssetWorkflow` for the standard export directory."""
    export_dir = DATA_DIR / "oam_tcd_instance_coco"
    return AssetWorkflow(
        force_download_assets=force_download_assets,
        force_export_full=force_export_full,
        force_val_split=force_val_split,
        val_fold=val_fold,
        dataset_name=OAM_DATASET_NAME,
        export_dir=export_dir,
        cache_dir=DATA_DIR / "hf_cache",
        checkpoint_dir=CHECKPOINT_DIR / "sam3",
        asset_manifest=CHECKPOINT_DIR / "asset_manifest.json",
        export_report=export_dir / "reports" / "export_report.json",
    )


@dataclass(frozen=True)
class ExperimentWorkflow:
    """Validate, launch, and inspect one training run from a notebook."""
    export_dir: Path
    experiment_dir: Path
    run_name: str
    model_backend: str = MODEL_BACKEND

    def validate_environment(self) -> None:
        """Print and check the Python, PyTorch, and CUDA environment."""
        import torch

        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("cuda device:", torch.cuda.get_device_name(0))
        print("model_backend:", self.model_backend)
        if self.model_backend == "sam3":
            add_support_to_pythonpath()
            import sam3
            print("sam3 import ok")
        elif self.model_backend == "maskrcnn":
            import detectron2
            print("detectron2:", detectron2.__version__)
        else:
            raise ValueError(f"Unknown model_backend: {self.model_backend}")

    def validate_assets(self) -> tuple[Path, Path]:
        """Check that the COCO export and checkpoint exist and return the TRAIN and TEST annotation paths."""
        train = self.export_dir / "annotations" / "train_annotations.coco.json"
        test = self.export_dir / "annotations" / "test_annotations.coco.json"
        if not train.is_file() or not test.is_file():
            raise FileNotFoundError("Production COCO export is incomplete. Run Notebook/assets_preparation.ipynb first.")
        if self.model_backend == "sam3":
            assets = create_asset_workflow()
            if not assets.asset_manifest.is_file():
                raise FileNotFoundError("No asset manifest found. Run Notebook/assets_preparation.ipynb first.")
            sam3_checkpoint = assets.checkpoint_dir / "sam3.pt"
            sam3_config = assets.checkpoint_dir / "config.json"
            if not sam3_checkpoint.is_file() or not sam3_config.is_file():
                raise FileNotFoundError("SAM3 checkpoint assets are incomplete. Run Notebook/assets_preparation.ipynb first.")
            print(assets.asset_manifest.read_text()[:2000])
        elif self.model_backend == "maskrcnn":
            print("Mask R-CNN uses COCO-pretrained weights; no project checkpoint download required for training.")
            print("COCO weights are fetched automatically by Detectron2 model_zoo.")
        print("full export dir:", self.export_dir)
        return train, test

    def validate_export(self) -> dict[str, Any]:
        """Check that every exported split uses the canonical category list and return its summary."""
        assets = create_asset_workflow()
        summary = assets.export_summary()
        for split in OAM_SPLITS:
            if summary[split]["categories"] != [{"id": 1, "name": "tree"}]:
                raise ValueError(f"{split} categories are not canonical.")
            if summary[split]["missing_image_count"] != 0 or summary[split]["images"] <= 0:
                raise ValueError(f"{split} export is incomplete.")
        if summary["train"]["annotations"] > 0 and max(_annotation_counts(self.export_dir / "annotations" / "train_annotations.coco.json")) > 1000:
            raise ValueError("Train export exceeds the configured 1000-annotation cap.")
        print(json.dumps(summary, indent=2))
        return summary

    def augmentation_summary(self) -> list[str]:
        """List the configured training augmentations and check that the required ones are present."""
        if self.model_backend == "sam3":
            from augmentation import config_train_ops

            labels = [label for label, _ in config_train_ops(include_resize=True)]
            required = ["crop[", "hflip", "vflip", "rot90", "colorjitter", "resize->"]
            if not all(fragment in " ".join(labels) for fragment in required):
                raise ValueError("Configured augmentation chain is missing required operations.")
            print("augmentation chain:", " -> ".join(labels))
            return labels
        if self.model_backend == "maskrcnn":
            from config import (
                MASKRCNN_MIN_SIZE_TRAIN,
                MASKRCNN_MAX_SIZE_TRAIN,
                MASKRCNN_CROP_SIZE,
            )

            print("Mask R-CNN augmentation (Detectron2):")
            print(f"  MIN_SIZE_TRAIN: {MASKRCNN_MIN_SIZE_TRAIN}")
            print(f"  MAX_SIZE_TRAIN: {MASKRCNN_MAX_SIZE_TRAIN}")
            print(f"  CROP.ENABLED: True")
            print(f"  CROP.TYPE: absolute")
            print(f"  CROP.SIZE: {MASKRCNN_CROP_SIZE}")
            return [
                f"ResizeShortestEdge({MASKRCNN_MIN_SIZE_TRAIN}, max={MASKRCNN_MAX_SIZE_TRAIN})",
                f"AbsoluteCrop({MASKRCNN_CROP_SIZE})",
            ]
        print(f"Augmentation check skipped for model_backend={self.model_backend}")
        return []

    def launch_training(
        self,
        launch: bool,
        *,
        allow_precision_fallback: bool = False,
        resume_from: Path | None = None,
        ledger_path: Path | None = None,
        require_node_headroom: bool = True,
        worker_threads: int = 1,
        arm: str | None = None,
        train_workers: int | None = None,
        val_workers: int | None = None,
        micro_batch: int | None = None,
        checkpoint_backbone: bool = False,
        smoke_report: Path | None = None,
        efficacy_spec: Path | None = None,
    ) -> Path:
        """Launch one protected training workflow.

        `allow_precision_fallback` permits a non-reportable development run on hardware that
        cannot honour the configured bf16 recipe. Reported runs must keep it disabled.

        `resume_from` must point outside `experiment_dir` because launch preparation clears
        transient run contents. The recovery notebook stages checkpoints externally and
        preserves the append-only training log.

        `require_node_headroom` protects an unscheduled shared node from CPU saturation.
        Disabling it waives only the CPU-load threshold; GPU-memory requirements remain hard.

        `arm` selects a registered objective and enables it through a launch-scoped environment
        variable. The committed configuration remains inert, and arm, run name, backend, and
        output directory must agree.

        `micro_batch` may reduce per-rank activation memory while accumulation preserves the
        configured effective image batch. `train_workers`, `val_workers`, and `worker_threads`
        are throughput controls; they do not change seeded batch composition.
        """
        from prompt_granularity import ARM as PG_ARM, RUN_NAME as PG_RUN_NAME
        if self.run_name == PG_RUN_NAME and arm != PG_ARM:
            raise ValueError("The prompt-granularity directory requires its explicit arm switch.")
        if arm == PG_ARM and (self.run_name != PG_RUN_NAME or self.model_backend != "sam3"):
            raise ValueError("Prompt-granularity arm, run directory, and SAM3 backend must agree.")
        command = [
            sys.executable,
            str(CODEBASE_DIR / "Core" / "training_pipeline.py"),
            "--data-root",
            str(self.export_dir),
            "--experiment-log-dir",
            str(self.experiment_dir),
        ]
        if self.model_backend == "sam3":
            command.insert(2, "--config")
            command.insert(3, "full_ft")
        if allow_precision_fallback:
            command.append("--allow-precision-fallback")
        if resume_from is not None:
            resume_from = Path(resume_from).resolve()
            if not resume_from.is_file():
                raise FileNotFoundError(f"resume_from checkpoint does not exist: {resume_from}")
            if resume_from.is_relative_to(Path(self.experiment_dir).resolve()):
                raise ValueError(
                    "resume_from must not point inside experiment_dir: the launcher wipes "
                    "that directory (except training_log.md) before training. Copy the "
                    "checkpoint out first -- the resume notebook's staging cell does this."
                )
            command += ["--resume-from", str(resume_from)]
        log_path = self.experiment_dir / "training_log.md"
        print("model_backend:", self.model_backend)
        print("Run name from Core/config.py:", self.run_name)
        if self.model_backend == "sam3":
            print("Epochs from Core/config.py:", FULL_FT_MAX_DATA_EPOCHS)
            print("Validation frequency from Core/config.py:", FULL_FT_VAL_EPOCH_FREQ)
        elif self.model_backend == "maskrcnn":
            from config import MASKRCNN_MAX_ITER, MASKRCNN_BASE_LR, MASKRCNN_IMS_PER_BATCH
            print("Max iters from Core/config.py:", MASKRCNN_MAX_ITER)
            print("Base LR from Core/config.py:", MASKRCNN_BASE_LR)
            print("Batch size from Core/config.py:", MASKRCNN_IMS_PER_BATCH)
        print("Launch command:", " ".join(command))
        print("Experiment dir:", self.experiment_dir)
        print("Full markdown log:", log_path)
        print("allow_precision_fallback:", allow_precision_fallback)
        if not launch:
            print("Launch skipped. Set LAUNCH = True to start training.")
            return log_path

        assert_experiment_directory_available(self.experiment_dir, resume=resume_from is not None)
        if arm == PG_ARM:
            if (micro_batch, checkpoint_backbone, train_workers, val_workers) != (2, True, 4, 2):
                raise ValueError("Prompt-granularity is pinned to microbatch 2, backbone checkpointing, and 4/2 TRAIN/VAL workers.")
            if (
                allow_precision_fallback
                or smoke_report is None
                or efficacy_spec is None
                or not Path(smoke_report).is_file()
                or not Path(efficacy_spec).is_file()
            ):
                raise ValueError("Prompt-granularity requires bf16, a successful smoke report, and an approved efficacy declaration before full training.")
            smoke_report = Path(smoke_report).resolve()
            efficacy_spec = Path(efficacy_spec).resolve()
            command += [
                "--prompt-granularity-smoke", str(smoke_report),
                "--prompt-granularity-efficacy", str(efficacy_spec),
            ]

        # Checked AFTER the dry-run return above, so inspecting the command never
        # requires a healthy node -- only an actual launch does.
        from device_policy import (LOAD_PER_CORE_BLOCK,
                                   assert_node_headroom, shared_node_env)

        # gpus_per_node follows the visible device count, so ask for exactly as many
        # cards as this launch will actually try to use.
        from device_policy import visible_gpu_count

        # The two checks are relaxed SEPARATELY, because they fail differently. CPU load is a
        # fairness and throughput concern -- overriding it gives a slow run, which can be a
        # legitimate choice on a shared node. Insufficient GPU memory is an OOM partway into a
        # multi-day run, which never is. So `require_node_headroom=False` lifts the load
        # threshold only; the GPU requirement stays hard.
        # Required memory is anchored on observed full-fine-tuning peaks rather than linear
        # batch-size extrapolation because fixed model, optimizer, and context costs dominate.
        # The additional margin protects against allocator fragmentation and co-tenant drift.
        _micro = micro_batch or 8
        _need = int(23_770 + (_micro - 4) * 1000 + 2_000) if _micro >= 2 else 12000
        _need = min(max(12000, _need), 32000)
        _visible_gpus = visible_gpu_count()
        if arm == PG_ARM:
            from prompt_granularity import smoke_peak_reserved_mib
            if _visible_gpus != 2:
                raise RuntimeError(f"Prompt-granularity requires exactly two visible A40 GPUs, found {_visible_gpus}.")
            _need = max(_need, int(smoke_peak_reserved_mib(smoke_report) + 4096))
        node_report = assert_node_headroom(
            strict=True,
            require_gpu_mib=_need,
            require_gpus=2 if arm == PG_ARM else max(1, _visible_gpus),
            block_above=float("inf") if not require_node_headroom else LOAD_PER_CORE_BLOCK,
        )
        if not require_node_headroom:
            print(f"Node-load guard waived by request (load "
                  f"{node_report['load_per_core']:.2f}/core). The GPU-memory guard is still "
                  f"enforced. Expect roughly {max(1.0, node_report['load_per_core'] / 0.6):.1f}x "
                  f"the uncontended wall clock.")
        print(f"Node: {node_report['n_cores']} cores, 5-min load "
              f"{node_report['load5']:.1f} ({node_report['load_per_core']:.2f}/core)")
        for gpu in node_report.get("gpus", []):
            print(f"  GPU {gpu['index']}: {gpu['free_mib']} MiB free of "
                  f"{gpu['total_mib']}, {gpu['util_pct']}% util")
        launch_env = shared_node_env(worker_threads)
        if micro_batch is not None:
            launch_env["TRAIN_BATCH_SIZE_OVERRIDE"] = str(micro_batch)
            _accum = 8 // micro_batch
            print(f"Microbatch reduced to {micro_batch} (accumulation {_accum}); effective "
                  f"batch {micro_batch * _accum * max(1, visible_gpu_count())} -- unchanged "
                  f"from the recipe")
            # Expandable segments reduce allocator fragmentation at the smaller micro-batch.
            # training_pipeline.py sets this with setdefault, so a value passed here wins.
            launch_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if checkpoint_backbone:
            launch_env["ACT_CKPT_VISION_BACKBONE_OVERRIDE"] = "1"
            print("Whole-backbone activation checkpointing: ON (numerically exact; "
                  "activation memory of the trunk traded for ~30% step time)")
        if train_workers is not None:
            launch_env["FULL_FT_NUM_TRAIN_WORKERS_OVERRIDE"] = str(train_workers)
        if val_workers is not None:
            launch_env["FULL_FT_NUM_VAL_WORKERS_OVERRIDE"] = str(val_workers)
        if train_workers is not None or val_workers is not None:
            print(f"Dataloader workers reduced for this launch: train={train_workers}, "
                  f"val={val_workers} (throughput only; batch composition unchanged)")
        # Enable the mechanism at the launch boundary rather than in committed configuration.
        # This keeps the default path inert and binds activation to the selected run identity.
        if arm is not None:
            from config import ARM_RUN_NAMES
            expected = ARM_RUN_NAMES.get(arm)
            if expected is None:
                raise ValueError(f"unknown arm {arm!r}; expected one of "
                                 f"{sorted(ARM_RUN_NAMES)}")
            if self.run_name != expected:
                raise RuntimeError(
                    f"This workflow is for run {self.run_name!r}, but arm={arm!r} "
                    f"belongs in {expected!r}. Training would write the mechanism into "
                    f"another run's directory, after wiping it.\n"
                    f"  Build the workflow for the arm: "
                    f"create_experiment_workflow(arm={arm!r})"
                )
            launch_env["FULL_FT_RUN_NAME_OVERRIDE"] = expected
            launch_env[f"{arm.upper()}_ENABLED_OVERRIDE"] = "1"
            print(f"Mechanism arm: {arm} (enabled for this launch only)")
        if arm == PG_ARM:
            launch_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            launch_env["ACT_CKPT_VISION_BACKBONE_OVERRIDE"] = "1" if checkpoint_backbone else "0"
            launch_env["TRAIN_BATCH_SIZE_OVERRIDE"] = str(_micro)
            preflight = [
                sys.executable, str(CODEBASE_DIR / "Core/prompt_granularity.py"), "preflight",
                "--data-root", str(self.export_dir), "--report", str(smoke_report),
                "--efficacy-spec", str(efficacy_spec), "--micro-batch", str(_micro),
                "--run-dir", str(self.experiment_dir),
            ]
            if checkpoint_backbone:
                preflight.append("--checkpoint-backbone")
            if resume_from is not None:
                preflight += ["--resume-from", str(resume_from)]
            subprocess.run(preflight, cwd=CODEBASE_DIR, env=launch_env, check=True)

        # Refuse to wipe a finished run. `prepare_experiment_log_dir` deletes everything in
        # experiments/<run>/ except training_log.md, so relaunching onto a completed run
        # destroys its checkpoints and final_model. A resume is exempt: continuing a crashed
        # run is exactly what it is for.
        if resume_from is None:
            finished = [d.name for d in (self.experiment_dir / "final_model",
                                         self.experiment_dir / "checkpoints") if d.exists()]
            if finished:
                raise RuntimeError(
                    f"{self.experiment_dir} already holds a run ({', '.join(finished)}). "
                    f"Launching would delete it -- the launcher wipes this directory.\n"
                    f"  If this is a NEW arm, give it its own run name (see ARM_RUN_NAMES "
                    f"in Core/config.py); `arm=` sets it for you.\n"
                    f"  If you are continuing a crashed run, pass resume_from=.\n"
                    f"  If you really mean to discard it, move the directory aside first."
                )
        print(f"Thread caps for this launch: {launch_env['OMP_NUM_THREADS']} per process")
        if ledger_path is not None:
            # Recorded before the run starts, so an attempt that dies mid-epoch
            # still leaves evidence of which checkpoint it began from.
            from device_policy import describe_device

            append_resume_ledger(
                ledger_path,
                action="resume_launch",
                run=self.run_name,
                from_epoch=(read_checkpoint_epoch(resume_from) if resume_from is not None else None),
                resume_from=(str(resume_from) if resume_from is not None else None),
                device=describe_device().get("device_name"),
                allow_precision_fallback=allow_precision_fallback,
            )
        return_code = _run_with_markdown_log(
            command,
            cwd=CODEBASE_DIR,
            log_path=log_path,
            run_name=self.run_name,
            append=resume_from is not None,
            extra_env=launch_env,
        )
        if return_code != 0:
            raise RuntimeError(f"Training failed with exit code {return_code}")
        return log_path

    def inspect_logs(self) -> dict[str, Any]:
        """Summarise the run's logs for the selected backend."""
        if self.model_backend == "maskrcnn":
            return self._inspect_maskrcnn_logs()
        return self._inspect_sam3_logs()

    def _inspect_sam3_logs(self) -> dict[str, Any]:
        logs_dir = self.experiment_dir / "logs" / "oam_tcd"
        report = {
            "experiment_dir": str(self.experiment_dir),
            "train_stats": _read_jsonl(logs_dir / "train_stats.json"),
            "val_stats": _read_jsonl(logs_dir / "val_stats.json"),
            "best_stats": _read_jsonl(logs_dir / "best_stats.json"),
            "checkpoint_exists": (self.experiment_dir / "checkpoints" / "checkpoint.pt").is_file(),
            "generated_config_exists": (self.experiment_dir / "config" / "generated_sam3_config.yaml").is_file(),
        }
        print("experiment dir:", self.experiment_dir)
        for key in ("train_stats", "val_stats", "best_stats"):
            print(f"{key.replace('_', ' ')} rows:", len(report[key]))
        if report["train_stats"]:
            print("latest train:", json.dumps(report["train_stats"][-1], indent=2)[:4000])
        if report["val_stats"]:
            print("latest val:", json.dumps(report["val_stats"][-1], indent=2)[:4000])
        print("checkpoint exists:", report["checkpoint_exists"])
        print("run generated config exists:", report["generated_config_exists"])
        return report

    def _inspect_maskrcnn_logs(self) -> dict[str, Any]:
        report = {
            "experiment_dir": str(self.experiment_dir),
            "config_exists": (self.experiment_dir / "config.yaml").is_file(),
            "final_weights_exists": (self.experiment_dir / "model_final.pth").is_file(),
            "last_checkpoint_exists": (self.experiment_dir / "model_*.pth").is_file() if list(self.experiment_dir.glob("model_*.pth")) else False,
            "inference_dir_exists": (self.experiment_dir / "inference").is_dir(),
        }
        print("experiment dir:", self.experiment_dir)
        print("config exists:", report["config_exists"])
        print("final weights exist:", report["final_weights_exists"])
        checkpoints = sorted(self.experiment_dir.glob("model_*.pth"))
        if checkpoints:
            print("checkpoints:", [p.name for p in checkpoints])
        if report["inference_dir_exists"]:
            metrics = list((self.experiment_dir / "inference").glob("*.json"))
            print("inference metrics files:", [p.name for p in metrics])
        return report


def create_experiment_workflow(run_name: str | None = None, *,
                               arm: str | None = None) -> ExperimentWorkflow:
    """Create a workflow with one consistent run identity.

    Passing the arm here keeps the experiment directory, training log, inference output, and
    evaluation paths bound to the registered run name before any launch operation.
    """
    if arm is not None:
        from config import ARM_RUN_NAMES
        if arm not in ARM_RUN_NAMES:
            raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARM_RUN_NAMES)}")
        if run_name is not None and run_name != ARM_RUN_NAMES[arm]:
            raise ValueError(f"run_name {run_name!r} contradicts arm {arm!r} "
                             f"(expected {ARM_RUN_NAMES[arm]!r}). Pass one or the other.")
        run_name = ARM_RUN_NAMES[arm]
    if run_name is None:
        run_name = MASKRCNN_RUN_NAME if MODEL_BACKEND == "maskrcnn" else FULL_FT_RUN_NAME
    return ExperimentWorkflow(
        export_dir=DATA_DIR / "oam_tcd_instance_coco",
        experiment_dir=EXPERIMENTS_DIR / run_name,
        run_name=run_name,
        model_backend=MODEL_BACKEND,
    )


def _annotation_counts(annotation_path: Path) -> list[int]:
    coco = json.loads(annotation_path.read_text())
    counts = Counter(annotation["image_id"] for annotation in coco["annotations"])
    return [counts.get(image["id"], 0) for image in coco["images"]]


def _bucketize(counts: list[int], buckets: tuple[int, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, low in enumerate(buckets):
        high = buckets[index + 1] if index + 1 < len(buckets) else None
        label = f">={low}" if high is None else f"{low}-{high - 1}"
        if high is None:
            result[label] = int(sum(1 for count in counts if count >= low))
        else:
            result[label] = int(sum(1 for count in counts if low <= count < high))
    return result


def _summarize_counts(counts: list[int]) -> dict[str, float | int]:
    if not counts:
        return {"images": 0}
    import numpy as np

    values = np.asarray(counts, dtype=np.int64)
    return {
        "images": int(values.size),
        "total_annotations": int(values.sum()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": int(values.min()),
        "max": int(values.max()),
        **{f"p{int(percentile)}": float(np.percentile(values, percentile)) for percentile in (50, 75, 90, 95, 99, 100)},
        "images_with_zero_gt": int((values == 0).sum()),
        "images_above_100": int((values > 100).sum()),
        "images_above_200": int((values > 200).sum()),
        "images_above_500": int((values > 500).sum()),
        "fraction_above_200": float((values > 200).mean()),
    }


# =============================================================================
# Locked measurement protocol
#
# These values define a comparable inference dump. They are centralised here so every
# notebook and command uses the same tiling, merge, filtering, and evaluation settings.
# =============================================================================
INFERENCE_TILE = 1024
INFERENCE_OVERLAP = 256
INFERENCE_DEDUP_IOU = 0.30
INFERENCE_SCORE_THRESH = 0.0
INFERENCE_MERGE_METHOD = "mask-nms"
INFERENCE_MIN_AREA = 4
MAX_DETS = 512              # OAM-TCD paper DETECTIONS_PER_IMAGE
BOUNDARY_FINE_BAND_PX = 8   # Fixed-width comparison; BOUNDARY_BAND_K is the primary instance band.
SAM3_PROMPT = "tree"
SAM3_RESOLUTION = 1008

# Every protocol value above is passed to tiled_inference.py EXPLICITLY, never
# left to a default, because two of the defaults disagree with this protocol:
# `--dedup-iou` defaults to 0.5 (we require 0.30) and `--maxdets` defaults to 0 /
# unlimited (we require 512). A dump produced by omitting those flags would look
# valid, would record its own metadata faithfully, and would not be comparable
# with A0. `--merge-method` and `--min-area` currently match their defaults, and
# are still passed explicitly so a future change to a default cannot silently
# alter an arm.


def run_tiled_inference(
    *,
    out_path: Path,
    ann: Path,
    image_root: Path,
    checkpoint: Path | None = None,
    tile_policy: str = "tiled",
    backend: str = "sam3",
    detectron_config: Path | None = None,
    detectron_weights: Path | None = None,
    force: bool = False,
) -> Path:
    """Run the locked tiled-inference protocol. Single definition for every arm.

    Skips work when `out_path` already exists unless `force=True`, so a notebook
    can be re-run end to end without silently regenerating a dump that other
    results were computed against.
    """
    out_path, ann, image_root = Path(out_path), Path(ann), Path(image_root)
    if out_path.exists() and not force:
        print(f"Prediction dump exists, skipping inference:\n  {out_path}")
        print("Pass force=True to regenerate it.")
        return out_path
    if not ann.is_file():
        raise FileNotFoundError(f"annotations missing: {ann}. Run assets_preparation.ipynb first.")

    command = [sys.executable, str(CODEBASE_DIR / "Core" / "tiled_inference.py")]
    if backend == "sam3":
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError(
                f"SAM3 weights missing: {checkpoint}. Training exports final_model/ only on a "
                "complete run; do not substitute a mid-run checkpoint."
            )
        command += [
            "--checkpoint", str(checkpoint),
            "--prompt", SAM3_PROMPT,
            "--resolution", str(SAM3_RESOLUTION),
        ]
    elif backend == "maskrcnn":
        command += [
            "--backend", "detectron2-maskrcnn",
            "--detectron-config", str(detectron_config),
            "--detectron-weights", str(detectron_weights),
            "--resolution", "1024",
            "--device", "cuda",
        ]
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    command += [
        "--ann", str(ann),
        "--image-root", str(image_root),
        "--out", str(out_path),
        # --run-dir decides where inference_config.json lands: tiled_inference.py
        # writes it to run_root if given, else to out_path.parent. Omitting it put
        # the provenance file under predictions/ instead of the run dir, which is
        # where check_dumps_comparable looks -- so the gate failed on a dump whose
        # protocol was actually correct. The run dir is out_path's grandparent
        # (<run>/predictions/<file>.json).
        "--run-dir", str(out_path.parent.parent),
        "--tile-policy", tile_policy,
        "--tile", str(INFERENCE_TILE),
        "--overlap", str(INFERENCE_OVERLAP),
        "--score-thresh", str(INFERENCE_SCORE_THRESH),
        "--dedup-iou", str(INFERENCE_DEDUP_IOU),
        "--maxdets", str(MAX_DETS),
        "--merge-method", INFERENCE_MERGE_METHOD,
        "--min-area", str(INFERENCE_MIN_AREA),
    ]
    print("Inference command:", " ".join(command))
    started = time.time()
    subprocess.run(command, cwd=CODEBASE_DIR, check=True)
    print(f"inference finished in {(time.time() - started) / 60:.1f} min")
    return out_path


def require_boundary_iou() -> None:
    """Fail before evaluating if Boundary AP cannot be computed.

    The locked protocol includes instance-relative Boundary AP at k=0.19, so a
    missing `boundary_iou` package must stop the run rather than yield a
    Mask-AP-only result that looks complete.
    """
    try:
        from boundary_iou.coco_instance_api.coco import COCO as _BoundaryCOCO  # noqa: F401
    except ImportError as exc:
        from eval_predictions import BOUNDARY_AP_INSTALL_HINT

        raise RuntimeError(
            f"boundary_iou is not importable ({exc}). Boundary AP is the primary metric "
            f"for this arm, so fix this before evaluating:\n  {BOUNDARY_AP_INSTALL_HINT}"
        ) from exc
    print("boundary_iou: OK (Boundary AP will be computed)")


def _validate_evaluation_payload(payload: dict[str, Any], class_split: str | None = None) -> None:
    instance = payload.get("boundary_ap_instance", {})
    if not instance.get("available") or instance.get("Boundary_AP") is None:
        raise ValueError("Instance-relative Boundary AP is missing or unavailable; do not substitute the fixed-8px result.")
    if instance.get("band_mode") != "instance_relative" or instance.get("band_k") != BOUNDARY_BAND_K:
        raise ValueError(f"Expected instance-relative Boundary AP at k={BOUNDARY_BAND_K}.")
    for name, value in (("Mask AP", payload.get("coco_eval", {}).get("AP")), ("Boundary AP", instance["Boundary_AP"])):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite value in [0, 1], got {value!r}.")
    protocol = payload.get("evaluation_protocol", {})
    if protocol.get("maxDets") != MAX_DETS:
        raise ValueError(f"Expected evaluation maxDets={MAX_DETS}.")
    if class_split is not None and protocol.get("class_split") != class_split:
        raise ValueError(f"Expected class_split={class_split!r}, got {protocol.get('class_split')!r}.")


def run_evaluation(
    *,
    pred: Path,
    gt: Path,
    out_path: Path,
    class_split: str,
    band_sweep: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run the locked evaluation protocol and return the parsed payload."""
    pred, gt, out_path = Path(pred), Path(gt), Path(out_path)
    if out_path.is_file() and not force:
        payload = json.loads(out_path.read_text())
        _validate_evaluation_payload(payload, class_split)
        if any(source.is_file() and source.stat().st_mtime > out_path.stat().st_mtime for source in (pred, gt)):
            raise RuntimeError(f"Evaluation predates its inputs: {out_path}. Verify provenance before explicitly recomputing it.")
        if band_sweep and "boundary_ap_sweep" not in payload:
            raise ValueError("The cached evaluation has no band sweep; use a new output path or explicitly request force=True.")
        print("Existing evaluation reused without modification:", out_path)
        return payload
    command = [
        sys.executable, str(CODEBASE_DIR / "Core" / "eval_predictions.py"),
        "--pred", str(pred), "--gt", str(gt), "--out", str(out_path),
        "--max-dets", str(MAX_DETS),
        "--protocol", "oam_tcd_test",
        "--boundary-fine-band-px", str(BOUNDARY_FINE_BAND_PX),
        "--band-k", str(BOUNDARY_BAND_K),
        "--class-split", class_split,
    ]
    if band_sweep:
        command.append("--boundary-band-sweep")
    print("Eval command:", " ".join(command))
    subprocess.run(command, cwd=CODEBASE_DIR, check=True)
    payload = json.loads(out_path.read_text())
    _validate_evaluation_payload(payload, class_split)
    return payload


def read_metric_row(label: str, eval_path: Path, *, class_split: str | None = None) -> dict[str, Any] | None:
    """One row of the comparison table, or None when the dump is absent."""
    eval_path = Path(eval_path)
    if not eval_path.exists():
        print(f"missing, skipped: {eval_path}")
        return None
    payload = json.loads(eval_path.read_text())
    _validate_evaluation_payload(payload, class_split)
    coco = payload.get("coco_eval", {})
    instance = payload["boundary_ap_instance"]
    fine = payload.get("boundary_ap_fine", {})
    lit = payload.get("boundary_ap", {})
    return {
        "run": label,
        "Mask AP": coco.get("AP"),
        "Mask AP50": coco.get("AP50"),
        "Mask AP75": coco.get("AP75"),
        "Mask AP_small": coco.get("AP_small"),
        "Mask AP_medium": coco.get("AP_medium"),
        "Mask AP_large": coco.get("AP_large"),
        "Mask AR@512": coco.get("AR@512"),
        "Bnd AP instance": instance.get("Boundary_AP"),
        "Bnd AP50 instance": instance.get("Boundary_AP50"),
        "Bnd AP75 instance": instance.get("Boundary_AP75"),
        "Bnd AP_small": instance.get("Boundary_AP_small"),
        "Bnd AP_medium": instance.get("Boundary_AP_medium"),
        "Bnd AP_large": instance.get("Boundary_AP_large"),
        "Bnd AP 8px": fine.get("Boundary_AP"),
        "Bnd AP lit": lit.get("Boundary_AP"),
    }


def read_checkpoint_epoch(checkpoint: Path) -> int:
    """Return a checkpoint's `epoch` field, i.e. the number of epochs completed.

    `Trainer.save_checkpoint` writes `epoch = epoch + 1` as soon as an epoch's
    training loop finishes and *before* the validation pass, so this is the
    authoritative resume position. The markdown log is a broader event record and does not
    replace checkpoint state.

    Tries `mmap=True` first, which lazily maps tensor storages instead of reading
    them, so pulling one int out of a 10 GB file costs milliseconds rather than a
    minute. That matters because the staging watcher calls this once per epoch.
    Falls back to a full load for checkpoints that cannot be mapped (legacy
    non-zipfile format, older torch).
    """
    import torch

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "epoch" not in payload:
        raise KeyError(
            f"{checkpoint} has no 'epoch' field (keys: {sorted(payload)}). Refusing to "
            "guess how far this run got."
        )
    return int(payload["epoch"])


def append_resume_ledger(ledger_path: Path, **fields: Any) -> dict[str, Any]:
    """Append one timestamped JSONL row to a run's resume ledger.

    The ledger is the durable record of a run's recovery history. Every other
    source is lossy: `prepare_experiment_log_dir` wipes the experiment dir on
    each launch, and `training_log.md` interleaves several attempts without
    recording which checkpoint each one started from. The ledger lives in
    `runtime/`, outside the wipe, and is append-only, so it survives every
    relaunch and answers "what did we resume from, when, and on what device?".
    """
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": _utc_now(), **fields}
    with ledger_path.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")
    return entry


def read_resume_ledger(ledger_path: Path) -> list[dict[str, Any]]:
    """Read a resume ledger, skipping any row truncated by an interrupted write."""
    ledger_path = Path(ledger_path)
    if not ledger_path.is_file():
        return []
    rows = []
    for line in ledger_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  (skipping malformed ledger row: {line[:60]}...)")
    return rows


def summarize_resume_ledger(ledger_path: Path) -> list[dict[str, Any]]:
    """Print a run's recovery history: launches, stagings, and any staging errors."""
    rows = read_resume_ledger(ledger_path)
    if not rows:
        print(f"resume ledger is empty or absent: {ledger_path}")
        return rows

    print(f"resume ledger: {ledger_path}  ({len(rows)} entries)")
    for row in rows:
        ts = str(row.get("ts", "?"))[:19].replace("T", " ")
        action = row.get("action", "?")
        if action == "resume_launch":
            dev = row.get("device", "?")
            print(f"  {ts}  LAUNCH   resume from epoch {row.get('from_epoch')} on {dev}")
        elif action == "staged":
            prev = row.get("previous_epoch")
            prev_txt = f" (was {prev})" if prev is not None else ""
            print(f"  {ts}  staged   epoch {row.get('epoch')}{prev_txt}")
        elif action == "error":
            print(f"  {ts}  ERROR    {row.get('detail')}")
        else:
            print(f"  {ts}  {action}  {row}")

    staged = [r.get("epoch") for r in rows if r.get("action") == "staged" and r.get("epoch") is not None]
    if staged:
        print(f"\n  epochs staged: {sorted(set(staged))}  (highest {max(staged)})")
    errors = [r for r in rows if r.get("action") == "error"]
    if errors:
        print(f"  {len(errors)} staging error(s) recorded -- read them before trusting the staged file")
    return rows


class CheckpointStager:
    """Background watcher that re-stages the resume checkpoint after every epoch.

    Without this, `resume_from` is only as fresh as the last manual copy and later progress
    can be lost. The launch cell blocks during training, so refresh must run alongside it.

    Use as a context manager around `launch_training`:

        with CheckpointStager(EXPERIMENT_DIR, RESUME_FROM, ledger_path=LEDGER):
            experiment.launch_training(True, resume_from=RESUME_FROM)

    Safety properties:

    * **Only ever moves forward.** Stages solely when the live checkpoint's epoch
      is strictly greater than the staged one, so it cannot discard progress.
      This also means it stays away from `dest` until an epoch completes, which
      is why it cannot disturb the trainer's own long startup read of `dest`.
    * **Never writes a partial file.** Copies to `.partial` and renames. On POSIX
      the rename does not invalidate a reader's open descriptor, so even a
      concurrent read of `dest` is safe.
    * **Waits for the write to settle.** Requires (mtime, size) to be unchanged
      across two polls before reading, so a checkpoint still being written is
      never staged.
    * **Cannot kill the run.** Any staging failure is logged to the ledger and
      printed; the watcher keeps going and training is unaffected.
    """

    def __init__(
        self,
        experiment_dir: Path,
        dest: Path,
        *,
        ledger_path: Path | None = None,
        interval_seconds: float = 120.0,
        verbose: bool = True,
    ) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.dest = Path(dest)
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self.interval = float(interval_seconds)
        self.verbose = verbose
        self.staged_epochs: list[int] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sig: tuple[int, int] | None = None
        self._handled_sig: tuple[int, int] | None = None
        self._staged_epoch: int | None = None

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> CheckpointStager:
        try:
            self._staged_epoch = read_checkpoint_epoch(self.dest) if self.dest.is_file() else None
        except Exception as exc:
            self._staged_epoch = None
            self._record_error(f"could not read staged checkpoint at start: {exc}")
        self._thread = threading.Thread(target=self._loop, name="checkpoint-stager", daemon=True)
        self._thread.start()
        if self.verbose:
            start = self._staged_epoch if self._staged_epoch is not None else "none"
            print(f"[stager] watching {self.experiment_dir.name} every {self.interval:.0f}s "
                  f"(staged epoch at start: {start})")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 60)
        # One last pass: an epoch may have completed between the final poll and
        # the process exiting, and that is exactly the epoch a crash would lose.
        self._poll_once(final=True)
        if self.verbose:
            print(f"[stager] stopped. epochs staged this session: {self.staged_epochs or 'none'}")
            if self.errors:
                print(f"[stager] {len(self.errors)} error(s): {self.errors}")

    # -- internals ---------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._poll_once()

    def _record_error(self, detail: str) -> None:
        self.errors.append(detail)
        print(f"[stager] ERROR {detail}")
        if self.ledger_path is not None:
            try:
                append_resume_ledger(self.ledger_path, action="error", detail=detail)
            except Exception:
                pass

    def _poll_once(self, final: bool = False) -> None:
        live = self.experiment_dir / "checkpoints" / "checkpoint.pt"
        try:
            if not live.is_file():
                return
            stat = live.stat()
            sig = (stat.st_mtime_ns, stat.st_size)
            if sig == self._handled_sig:
                return
            # Require stability across two observations so a checkpoint still
            # being written is never read. On the final pass, take it as-is:
            # training has exited, so nothing can still be writing.
            if not final and sig != self._last_sig:
                self._last_sig = sig
                return

            live_epoch = read_checkpoint_epoch(live)
            if self._staged_epoch is not None and live_epoch <= self._staged_epoch:
                self._handled_sig = sig
                return

            result = stage_resume_checkpoint(self.experiment_dir, self.dest, quiet=True)
            previous, self._staged_epoch = self._staged_epoch, result["staged_epoch"]
            self._handled_sig = sig
            self.staged_epochs.append(result["staged_epoch"])
            print(f"[stager] staged epoch {result['staged_epoch']} -> {self.dest.name}"
                  f"{'' if previous is None else f' (was {previous})'}")
            if self.ledger_path is not None:
                append_resume_ledger(
                    self.ledger_path,
                    action="staged",
                    epoch=result["staged_epoch"],
                    previous_epoch=previous,
                    size_bytes=self.dest.stat().st_size,
                )
        except Exception as exc:  # never let the watcher take training down
            self._record_error(f"{type(exc).__name__}: {exc}")
            self._handled_sig = self._last_sig


def backup_training_artifacts(experiment_dir: Path, backup_dir: Path) -> dict[str, Any]:
    """Snapshot a run's per-epoch records before a relaunch wipes them.

    `prepare_experiment_log_dir` clears transient experiment contents except
    `training_log.md` at launch. Copy recoverable state out before any resume operation.

    Idempotent by content: a file already present in the backup with the same
    size is left alone, so repeated calls do not churn a 10 GB checkpoint.
    """
    experiment_dir, backup_dir = Path(experiment_dir), Path(backup_dir)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"experiment dir does not exist: {experiment_dir}")

    copied, skipped = [], []
    # `final_model/` holds the exported last-epoch weights that offline inference
    # reads, and the launcher wipes it like everything else. It was omitted here
    # once and a mis-targeted relaunch deleted a completed run's exported weights
    # before they had been copied anywhere -- recoverable only because that run
    # re-exported them. Back it up.
    for rel in ("logs", "tensorboard", "checkpoints/checkpoint.pt", "final_model", "training_log.md"):
        src = experiment_dir / rel
        if not src.exists():
            continue
        dest = backup_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            for item in src.rglob("*"):
                if not item.is_file():
                    continue
                target = dest / item.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.stat().st_size == item.stat().st_size:
                    skipped.append(str(item.relative_to(experiment_dir)))
                    continue
                shutil.copy2(item, target)
                copied.append(str(item.relative_to(experiment_dir)))
        else:
            if dest.exists() and dest.stat().st_size == src.stat().st_size:
                skipped.append(rel)
            else:
                shutil.copy2(src, dest)
                copied.append(rel)

    print(f"backup -> {backup_dir}")
    print(f"  {len(copied)} file(s) copied, {len(skipped)} already present and identical in size")
    live = experiment_dir / "checkpoints" / "checkpoint.pt"
    epoch = read_checkpoint_epoch(live) if live.is_file() else None
    if epoch is not None:
        print(f"  checkpoint.pt is at epoch {epoch}")
    return {"backup_dir": str(backup_dir), "copied": copied, "skipped": skipped, "checkpoint_epoch": epoch}


def stage_resume_checkpoint(
    experiment_dir: Path,
    dest: Path,
    *,
    allow_downgrade: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Copy a run's live checkpoint out of the experiment dir so it survives relaunch.

    `resume_from` must point outside `experiment_dir` because the launcher wipes
    that directory first, so this copy is a hard requirement of resuming, not a
    convenience.

    The copy is guarded by the checkpoint's own `epoch` field. Staging refuses to replace a
    checkpoint with an older one. Pass `allow_downgrade=True` only for an explicit rewind.
    """
    experiment_dir, dest = Path(experiment_dir), Path(dest)
    live = experiment_dir / "checkpoints" / "checkpoint.pt"
    if not live.is_file():
        raise FileNotFoundError(
            f"no live checkpoint at {live}. Training writes it after each epoch's "
            "training loop; if it is absent the run never completed an epoch."
        )
    if dest.resolve().is_relative_to(experiment_dir.resolve()):
        raise ValueError(
            f"staging destination {dest} is inside {experiment_dir}, which the launcher "
            "wipes before training. Stage to runtime/ instead."
        )

    live_epoch = read_checkpoint_epoch(live)
    staged_epoch = read_checkpoint_epoch(dest) if dest.is_file() else None

    if staged_epoch is not None:
        if staged_epoch == live_epoch:
            if not quiet:
                print(f"already staged at epoch {live_epoch}: {dest}\nno copy needed.")
            return {"staged_epoch": live_epoch, "live_epoch": live_epoch, "copied": False}
        if staged_epoch > live_epoch and not allow_downgrade:
            raise RuntimeError(
                f"REFUSING to stage: {dest} is at epoch {staged_epoch} but the live "
                f"checkpoint is at epoch {live_epoch}. Overwriting would throw away "
                f"{staged_epoch - live_epoch} completed epoch(s). This usually means the "
                "experiment dir was wiped by a relaunch and the staged copy is the only "
                "surviving record of the newer state -- resume from it as-is. Pass "
                "allow_downgrade=True only if you intend to rewind."
            )
        if not quiet:
            print(f"staged copy is at epoch {staged_epoch}, live is at epoch {live_epoch}", end="")
            print(" (downgrade permitted)" if staged_epoch > live_epoch else " -> updating")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    shutil.copy2(live, tmp)          # copy via .partial so an interrupted copy
    tmp.replace(dest)                # cannot leave a truncated checkpoint staged
    verified = read_checkpoint_epoch(dest)
    if verified != live_epoch:
        raise RuntimeError(f"staged checkpoint reads epoch {verified}, expected {live_epoch}")
    if not quiet:
        print(f"staged epoch {verified} -> {dest} ({dest.stat().st_size / 1e9:.2f} GB, verified)")
        print(f"resume will land on epoch {verified + 1}")
    return {"staged_epoch": verified, "live_epoch": live_epoch, "copied": True}


def reunify_training_records(experiment_dir: Path, *backup_dirs: Path) -> dict[str, Any]:
    """Merge a crashed run's stats and TensorBoard events back into the resumed run.

    `prepare_experiment_log_dir` in `training_pipeline.py` wipes everything in the
    experiment dir except `training_log.md` on every launch, so a resume starts
    with empty `logs/oam_tcd/` and `tensorboard/` and the pre-crash epochs' rows
    only survive in a backup. This function restores them.

    Rows are deduplicated by their `Trainer/epoch` field (present in every row of
    train/best/val stats), so **re-running this is safe** — a second call changes
    nothing. Backup dirs are read first and the current run last, so on a
    duplicate epoch (an epoch completed in two attempts) the surviving run's row
    wins; conflicts are reported, not hidden. TensorBoard event files are copied
    only when the filename is absent (event filenames carry a timestamp, so no
    two writers produce the same name).

    Returns a report dict; every merge is printed as it happens.
    """
    experiment_dir = Path(experiment_dir)
    report: dict[str, Any] = {"merged": {}, "conflicts": [], "tensorboard_copied": []}

    logs_dir = experiment_dir / "logs" / "oam_tcd"
    for name in ("train_stats.json", "best_stats.json", "val_stats.json"):
        by_epoch: dict[float, dict] = {}
        epoch_of_conflict: set[float] = set()
        # Backups first (older attempts), the current run last so its rows win.
        sources = [b / "logs" / "oam_tcd" / name for b in backup_dirs] + [logs_dir / name]
        found_any = False
        for src in sources:
            if not src.is_file():
                continue
            found_any = True
            for line in src.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                epoch = row.get("Trainer/epoch")
                if epoch is None:
                    raise ValueError(f"{src} has a row without Trainer/epoch; refusing to guess at dedupe.")
                key = float(epoch)
                # A re-run over an already-merged file sees every row twice with
                # identical content; only a genuine difference is a conflict.
                if key in by_epoch and by_epoch[key] != row:
                    epoch_of_conflict.add(key)
                by_epoch[key] = row
        if not found_any:
            continue
        merged = [by_epoch[e] for e in sorted(by_epoch)]
        logs_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(logs_dir / name, "".join(json.dumps(r) + "\n" for r in merged))
        report["merged"][name] = len(merged)
        for e in sorted(epoch_of_conflict):
            report["conflicts"].append(f"{name}: epoch {e:g} completed in two attempts; kept the surviving run's row")
        print(f"{name}: {len(merged)} epochs merged (coverage {merged[0]['Trainer/epoch']:g}..{merged[-1]['Trainer/epoch']:g})")

    tb_dir = experiment_dir / "tensorboard"
    for backup in backup_dirs:
        backup_tb = Path(backup) / "tensorboard"
        if not backup_tb.is_dir():
            continue
        tb_dir.mkdir(exist_ok=True)
        for event_file in sorted(backup_tb.glob("events.*")):
            dest = tb_dir / event_file.name
            if not dest.exists():
                shutil.copy2(event_file, dest)
                report["tensorboard_copied"].append(event_file.name)
    if report["tensorboard_copied"]:
        print(f"tensorboard: {len(report['tensorboard_copied'])} event file(s) restored from backup")
    for conflict in report["conflicts"]:
        print("NOTE:", conflict)
    return report


def _run_with_markdown_log(command: list[str], *, cwd: Path, log_path: Path, run_name: str, append: bool = False, extra_env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # If HPC modules are configured, wrap the command in a login shell that loads
    # them first. This lets notebooks launch training without requiring the user
    # to manually module-load before starting Jupyter.
    if HPC_LOAD_MODULES:
        module_cmd = "module load " + " ".join(shlex.quote(m) for m in HPC_LOAD_MODULES)
        inner = " ".join(shlex.quote(arg) for arg in command)
        wrapped_command = ["/bin/bash", "-lc", f"{module_cmd} && {inner}"]
    else:
        wrapped_command = command

    # append=True (resume launches): keep the crashed attempt's output in place
    # and continue the same file, so the training log remains one continuous
    # record of the run. The default "w" truncates, which would erase the crash
    # evidence the resume exists to learn from.
    # PYTHONFAULTHANDLER=1 makes every Python process in the tree (spawned ranks
    # included) dump all thread stacks to stderr on a fatal signal -- this is
    # what turns a bare SIGSEGV into an attributable faulting op. Harmless when
    # nothing crashes.
    mode = "a" if append else "w"
    with log_path.open(mode) as log_file:
        if append:
            log_file.write(f"\n\n---\n\n## Resumed launch ({time.strftime('%Y-%m-%d %H:%M:%S')})\n\n```bash\n{' '.join(wrapped_command)}\n```\n\n```text\n")
        else:
            log_file.write(f"# Training log: {run_name}\n\n## Launch command\n\n```bash\n{' '.join(wrapped_command)}\n```\n\n## Output\n\n```text\n")
        # extra_env carries the BLAS/OpenCV thread caps from `shared_node_env`. They
        # have to be applied here, at the subprocess boundary, rather than in the
        # calling kernel: the DataLoader forks on Linux and a forked child inherits
        # the parent's already-initialised OpenMP pool, so a late os.environ change
        # would not reach the workers that actually spawn the threads.
        process = subprocess.Popen(wrapped_command, cwd=cwd, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1", **(extra_env or {})}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
        log_file.write(f"\n```\n\nReturn code: `{return_code}`\n")
    return return_code


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
