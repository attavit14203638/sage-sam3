"""Shared path constants and overwrite guards for the Codebase layout.

Paths are derived from this file's location, so the layout works unchanged on any machine. The
guards refuse to overwrite an existing run directory unless the run is being explicitly resumed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


CODEBASE_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = CODEBASE_DIR / "Core"
SUPPORT_DIR = CODEBASE_DIR / "Support"
RUNTIME_DIR = CODEBASE_DIR / "runtime"
EXPERIMENTS_DIR = CODEBASE_DIR / "experiments"
DATA_DIR = RUNTIME_DIR / "data"
CHECKPOINT_DIR = RUNTIME_DIR / "checkpoints"
INFERENCE_OUTPUT_DIR = RUNTIME_DIR / "inference_output"
A0_RUN_NAME = "sam3_crop_frts"
A0_INFERENCE_DIR = INFERENCE_OUTPUT_DIR / f"oam_tcd_{A0_RUN_NAME}_tiled"


def assert_experiment_directory_available(
    path: Path, *, resume: bool = False, allow_training_log: bool = False
) -> None:
    """Raise unless `path` is free for a fresh run or, with `resume`, holds an incomplete run; completed runs can never be resumed or overwritten."""
    path = Path(path)
    if not path.exists():
        return
    if not path.is_dir():
        raise NotADirectoryError(path)
    if resume:
        if (path / "final_model" / "pytorch_model.bin").is_file() or (path / "model_final.pth").is_file():
            raise RuntimeError(f"Completed run cannot be resumed or overwritten: {path}")
        return
    existing = sorted(
        item.name for item in path.iterdir()
        if not item.name.startswith(".nfs")
        and not (allow_training_log and item.name == "training_log.md")
    )
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing run {path}: {', '.join(existing[:8])}. "
            "Use a new run name, or explicitly resume an incomplete run from a staged checkpoint."
        )


def ensure_runtime_dirs() -> None:
    """Create the data, checkpoint, experiments, and inference-output directories."""
    for path in (DATA_DIR, CHECKPOINT_DIR, EXPERIMENTS_DIR, INFERENCE_OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def run_dir(run_name: str) -> Path:
    """Returns (and creates) the single consolidated output folder for one
    configuration/run: INFERENCE_OUTPUT_DIR / run_name. All inference-time artifacts for
    that run (predictions, eval results, visualizations, diagnostics) should
    nest under this folder so a configuration is never split across
    top-level runtime dirs."""
    d = INFERENCE_OUTPUT_DIR / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def inference_run_name(dataset_id: str, model_profile: str, policy: str) -> str:
    """Build the canonical inference configuration name.

    The three segments identify the evaluation dataset/split, model profile,
    and inference policy without relying on mutable implementation qualifiers.
    """
    raw_parts = (dataset_id, model_profile, policy)
    if any(not part or part.strip() != part or "/" in part or "\\" in part for part in raw_parts):
        raise ValueError(
            "Inference run-name segments must be non-empty, trimmed, and free of path separators: "
            f"got dataset_id={dataset_id!r}, model_profile={model_profile!r}, policy={policy!r}."
        )
    parts = tuple(re.sub(r"-+", "_", part) for part in raw_parts)
    if any(not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", part) for part in parts):
        raise ValueError(
            "Inference run-name segments must contain only lowercase letters, digits, and single underscores: "
            f"got dataset_id={dataset_id!r}, model_profile={model_profile!r}, policy={policy!r}."
        )
    return "_".join(parts)


def inference_run_dir(run_name: str) -> Path:
    """Return the inference output directory for `run_name`, creating it."""
    return run_dir(run_name)


def add_support_to_pythonpath() -> None:
    """Put the vendored SAM3 source on `sys.path` when it is present."""
    sam3_path = SUPPORT_DIR / "sam3"
    sam3_path_str = str(sam3_path)
    if sam3_path.exists() and sam3_path_str not in sys.path:
        sys.path.insert(0, sam3_path_str)
