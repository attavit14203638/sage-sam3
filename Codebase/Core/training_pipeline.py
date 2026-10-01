"""Training entry point for the SAM3 fine-tune.

Builds the configuration from `config.py`, writes it into the run directory, checks the COCO
export, and launches the SAM3 trainer. When `MODEL_BACKEND` is `maskrcnn` it hands off to
`train_maskrcnn.py` instead. A fresh launch refuses to reuse an existing run directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Cap caching-allocator block size because per-batch instance counts produce variable dense
# mask allocations. Uniform reusable blocks reduce fragmentation during long training runs.
# Launch-specific configuration may override this value after measuring the target hardware.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

from config import (
    FULL_FT_RUN_NAME,
    MODEL_BACKEND,
    PROMPT_GRANULARITY_ENABLED,
    TEST_ANN_FILENAME,
    TRAIN_ANN_FILENAME,
    VAL_ANN_FILENAME,
    build_sam3_train_config,
    get_training_preset,
)
from project_paths import CODEBASE_DIR, EXPERIMENTS_DIR, SUPPORT_DIR, assert_experiment_directory_available, ensure_runtime_dirs
from atomic_io import atomic_write_text


CONFIG_NAMES = ("full_ft",)


def build_finetuned_sam3_image_model(**kwargs):
    """Model builder wired into `config.py`'s Hydra `_target_` for the SAM3 Trainer.

    Imports `eval_overrides` for its side effects: installing module-level
    monkey-patches (addmm_act grad guard, matcher NaN/Inf cost-matrix guard)
    that must be in place before the model is constructed and before the
    first training step.
    """
    import eval_overrides  # noqa: F401
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(**kwargs)
    from config import ACT_CKPT_VISION_BACKBONE

    if ACT_CKPT_VISION_BACKBONE:
        # Plain attribute, read at call time (vl_combiner.forward_image); set here, in every
        # rank, before DDP. Recomputation is numerically identical (non-reentrant checkpoint
        # restores the forward's RNG state), so this does not touch the recipe's numbers.
        model.backbone.act_ckpt_whole_vision_backbone = True
        print("[memory] whole-vision-backbone activation checkpointing ON "
              "(recipe numbers unchanged; step time +~30%)")
    return model


def _load_trainer_checkpoint(model, checkpoint_path):
    """Load a trainer-saved checkpoint that lacks the 'detector.' prefix."""
    import torch
    from iopath.common.file_io import g_pathmgr

    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    has_detector_prefix = any("detector." in k for k in ckpt)
    if has_detector_prefix:
        state_dict = {
            k.replace("detector.", ""): v
            for k, v in ckpt.items()
            if "detector" in k
        }
    else:
        state_dict = ckpt

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(
            f"[checkpoint] WARNING: {len(missing_keys)} missing keys "
            f"(first 5): {missing_keys[:5]}"
        )
    else:
        print(
            f"[checkpoint] Loaded {len(state_dict)} keys. "
            f"{len(unexpected_keys)} unexpected."
        )


def parse_args() -> argparse.Namespace:
    """Parse the command-line options of the training entry point."""
    parser = argparse.ArgumentParser(description="Project-owned training pipeline entry point.")
    parser.add_argument("--config", choices=CONFIG_NAMES, default="full_ft")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--experiment-log-dir", type=Path, default=None)
    parser.add_argument("--experiments-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--prompt-granularity-smoke", type=Path, default=None)
    parser.add_argument("--prompt-granularity-efficacy", type=Path, default=None)
    parser.add_argument(
        "--allow-precision-fallback",
        action="store_true",
        help="Permit training on a GPU that cannot run the recipe's AMP dtype (e.g. the V100 "
             "node, which has no bf16). For smoke tests and calibration only -- such a run is "
             "NOT the locked recipe and its numbers must not be reported.",
    )
    return parser.parse_args()


def check_coco_export(data_root: Path, strict: bool = True) -> list[str]:
    """Check the TRAIN, validation, and TEST annotation files, the category list, and the first exported image; raise on the first problem when `strict`, otherwise return the problems as messages."""
    messages: list[str] = []
    train_ann = data_root / "annotations" / TRAIN_ANN_FILENAME
    val_ann = data_root / "annotations" / VAL_ANN_FILENAME
    test_ann = data_root / "annotations" / TEST_ANN_FILENAME
    missing = [path for path in (train_ann, val_ann, test_ann) if not path.exists()]
    if missing:
        message = "Missing COCO annotation file(s): " + ", ".join(str(path) for path in missing)
        if train_ann in missing or val_ann in missing:
            message += (
                ". Run `python Core/dataset_adapter.py --data-root <data_root>` to export "
                f"{TRAIN_ANN_FILENAME} / {TEST_ANN_FILENAME}."
            )
        if strict:
            raise FileNotFoundError(message)
        return [message]

    with train_ann.open("r") as handle:
        train_data = json.load(handle)
    images = train_data.get("images", [])
    annotations = train_data.get("annotations", [])
    categories = train_data.get("categories", [])
    from prompt_granularity import CATEGORIES
    expected_categories = CATEGORIES if PROMPT_GRANULARITY_ENABLED else [{"id": 1, "name": "tree"}]
    if categories != expected_categories:
        message = f"Expected categories {expected_categories}, got: {categories}"
        if strict:
            raise ValueError(message)
        messages.append(message)
    if not images:
        message = f"No images found in {train_ann}"
        if strict:
            raise ValueError(message)
        messages.append(message)
    if not annotations:
        message = f"No annotations found in {train_ann}"
        if strict:
            raise ValueError(message)
        messages.append(message)

    if images:
        first_image = data_root / images[0]["file_name"]
        if not first_image.exists():
            message = f"First exported image is missing: {first_image}"
            if strict:
                raise FileNotFoundError(message)
            messages.append(message)
    return messages


def write_generated_config(config: dict[str, Any], path: Path) -> None:
    """Write the generated SAM3 configuration to `path` as JSON, which is valid YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML and keeps this generation path free of extra local dependencies.
    atomic_write_text(path, json.dumps(config, indent=2))


def snapshot_config_module(log_dir: Path) -> Path | None:
    """Copy the live Core/config.py into the run dir as config_snapshot.py.

    This preserves the exact centralized hyperparameter choices for the run in
    the same human-readable format as config.py, which is far easier to read
    than the verbose generated config.yaml / config_resolved.yaml.
    """
    import config

    source = Path(config.__file__).resolve()
    if not source.exists():
        return None
    target = log_dir / "config_snapshot.py"
    shutil.copy2(source, target)
    return target


def default_run_name(config_name: str) -> str:
    """Return the default run name for a configuration name."""
    if config_name == "full_ft":
        return FULL_FT_RUN_NAME
    return config_name


def resolve_experiment_log_dir(args: argparse.Namespace) -> Path:
    """Resolve the run directory from the command-line options."""
    if args.experiment_log_dir is not None:
        return args.experiment_log_dir.resolve()
    root = args.experiments_root.resolve() if args.experiments_root else EXPERIMENTS_DIR
    run_name = args.run_name or default_run_name(args.config)
    
    # If the user explicitly passed an absolute path as run_name, use it directly
    if Path(run_name).is_absolute():
        return Path(run_name).resolve()
        
    return (root / run_name).resolve()


def prepare_experiment_log_dir(log_dir: Path, *, resume: bool = False) -> str:
    """Create the run directory, or clear one that holds only a launcher log or an incomplete run being resumed (keeping `training_log.md`); returns `created` or `overwritten`."""
    assert_experiment_directory_available(log_dir, resume=resume, allow_training_log=True)
    if log_dir.exists():
        # Do not blindly call shutil.rmtree(log_dir) because training_log.md
        # might already be open by the launching notebook process.
        # Instead, clean up all files in log_dir except training_log.md and .nfs files.
        for item in log_dir.iterdir():
            if item.name.startswith(".nfs") or item.name == "training_log.md":
                continue
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            except OSError as exc:
                logging.warning("Could not remove %s while preparing %s: %s", item, log_dir, exc)
        return "overwritten"
    log_dir.mkdir(parents=True, exist_ok=True)
    return "created"


def run_generated_config_path(log_dir: Path) -> Path:
    """Return the path of the generated configuration inside a run directory."""
    return log_dir / "config" / "generated_sam3_config.yaml"


def build_sam3_command(config_path: Path, args: argparse.Namespace) -> list[str]:
    # SAM3's CLI cannot compose arbitrary config files outside its package, so this
    # command documents the equivalent project-owned entry point for reruns.
    """Return the equivalent command line for a rerun, recorded for reference."""
    command = [
        sys.executable,
        "Core/training_pipeline.py",
        "--config",
        args.config,
        "--data-root",
        str(args.data_root) if args.data_root else "<default>",
        "--experiment-log-dir",
        str(args.experiment_log_dir),
    ]
    if args.resume_from is not None:
        command.extend(["--resume-from", str(args.resume_from)])
    return command


def add_sam3_to_path() -> None:
    """Put `Core` and the vendored SAM3 source on `sys.path` and `PYTHONPATH`."""
    core_path = str(CODEBASE_DIR / "Core")
    support_path = str(SUPPORT_DIR / "sam3")
    if core_path not in sys.path:
        sys.path.insert(0, core_path)
    if support_path not in sys.path:
        sys.path.insert(0, support_path)
    paths = [core_path, support_path, os.environ.get("PYTHONPATH", "")]
    os.environ["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)


def load_omegaconf_config(config_path: Path):
    """Load the generated configuration as an OmegaConf object with SAM3's resolvers registered."""
    add_sam3_to_path()
    from omegaconf import OmegaConf
    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    register_omegaconf_resolvers()
    return OmegaConf.load(config_path)


def write_sam3_train_records(cfg) -> None:
    """Write the unresolved and resolved configuration records into the run directory."""
    from iopath.common.file_io import g_pathmgr
    from omegaconf import OmegaConf
    from sam3.train.utils.train_utils import makedir

    log_dir = cfg.launcher.experiment_log_dir
    makedir(log_dir)
    with g_pathmgr.open(os.path.join(log_dir, "config.yaml"), "w") as handle:
        handle.write(OmegaConf.to_yaml(cfg))
    with g_pathmgr.open(os.path.join(log_dir, "config_resolved.yaml"), "w") as handle:
        handle.write(OmegaConf.to_yaml(cfg, resolve=True))


def launch_training(config_path: Path) -> None:
    """Launch single-node training from a generated configuration file."""
    from sam3.train.train import single_node_runner

    cfg = load_omegaconf_config(config_path)
    write_sam3_train_records(cfg)
    port_range = cfg.launcher.port_range
    main_port = random.randint(port_range[0], port_range[1])
    single_node_runner(cfg, main_port)


def main() -> None:
    """Command-line entry point: dispatch to the Mask R-CNN or the SAM3 training path."""
    args = parse_args()
    ensure_runtime_dirs()

    if MODEL_BACKEND == "maskrcnn":
        _run_maskrcnn(args)
        return

    _run_sam3(args)


def _run_maskrcnn(args: argparse.Namespace) -> None:
    from config import MASKRCNN_RUN_NAME

    data_root = args.data_root.resolve() if args.data_root else None
    experiment_log_dir = args.experiment_log_dir
    if experiment_log_dir is None:
        experiment_log_dir = EXPERIMENTS_DIR / MASKRCNN_RUN_NAME
    experiment_log_dir = experiment_log_dir.resolve()

    cmd = [
        sys.executable,
        str(CODEBASE_DIR / "Core" / "train_maskrcnn.py"),
        "--data-root", str(data_root) if data_root else str(CODEBASE_DIR / "runtime" / "data" / "oam_tcd_instance_coco"),
        "--experiment-log-dir", str(experiment_log_dir),
    ]
    if args.resume_from is not None:
        cmd.extend(["--resume-from", str(args.resume_from.resolve())])
    print("MODEL_BACKEND = maskrcnn")
    print("Launch command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=CODEBASE_DIR, env={**os.environ, "PYTHONUNBUFFERED": "1"}, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Mask R-CNN training failed with exit code {result.returncode}")


def _run_sam3(args: argparse.Namespace) -> None:
    args.experiment_log_dir = resolve_experiment_log_dir(args)
    from prompt_granularity import RUN_NAME as PG_RUN_NAME
    if args.experiment_log_dir.name == PG_RUN_NAME and not PROMPT_GRANULARITY_ENABLED:
        raise ValueError("A prompt-granularity run name alone does not enable its mechanism.")
    if PROMPT_GRANULARITY_ENABLED and args.experiment_log_dir != (EXPERIMENTS_DIR / PG_RUN_NAME).resolve():
        raise ValueError("Full prompt-granularity training must use its dedicated experiment directory.")
    if args.resume_from is not None:
        args.resume_from = args.resume_from.resolve()
        if not args.resume_from.exists():
            raise FileNotFoundError(f"--resume-from checkpoint does not exist: {args.resume_from}")
        if args.resume_from.is_relative_to(args.experiment_log_dir):
            raise ValueError(
                "--resume-from must not point inside --experiment-log-dir because the launcher prepares that directory before training."
            )
    assert_experiment_directory_available(
        args.experiment_log_dir, resume=args.resume_from is not None, allow_training_log=True
    )
    # Fail before touching the filesystem if this node cannot honour the recipe.
    # The manuscript claims 2 x A40 with AMP bf16; training in float16 on the V100
    # node and reporting it as that recipe would be a false statement about the
    # experiment, and cross-architecture numerics have already bitten this project.
    from config import AMP_DTYPE
    from device_policy import (
        assert_training_device_is_supported,
        describe_device,
        resolve_autocast_dtype,
    )

    assert_training_device_is_supported(AMP_DTYPE, allow_fallback=args.allow_precision_fallback)
    # Pass the resolved dtype so the printed record matches what the trainer will
    # actually use; describe_device() alone would print float32 as the placeholder.
    print("Compute:", describe_device(amp_dtype=resolve_autocast_dtype(AMP_DTYPE)))

    preset = get_training_preset(
        args.config,
        data_root=args.data_root.resolve() if args.data_root else None,
        experiment_log_dir=args.experiment_log_dir,
    )
    check_coco_export(preset.data_root, strict=True)
    pg_contract = None
    if PROMPT_GRANULARITY_ENABLED:
        from prompt_granularity import RUN_MANIFEST, validate_launch, write_once
        if args.allow_precision_fallback or args.prompt_granularity_smoke is None or args.prompt_granularity_efficacy is None:
            raise ValueError("Full prompt-granularity requires bf16, a matching smoke report, and an approved efficacy declaration.")
        pg_contract = validate_launch(
            preset.data_root,
            args.prompt_granularity_smoke,
            args.prompt_granularity_efficacy,
            resume_from=args.resume_from,
            run_dir=preset.experiment_log_dir,
        )
    log_dir_status = prepare_experiment_log_dir(preset.experiment_log_dir, resume=args.resume_from is not None)
    if pg_contract is not None:
        write_once(preset.experiment_log_dir / RUN_MANIFEST, pg_contract)
        print("PROMPT_GRANULARITY_ACTIVE: two training prompts; image-based outer loss scaling; unchanged reporting GT.")
    config = build_sam3_train_config(preset)
    if args.resume_from is not None:
        config["trainer"]["checkpoint"]["resume_from"] = str(args.resume_from)
    config_path = run_generated_config_path(preset.experiment_log_dir)
    write_generated_config(config, config_path)
    config_snapshot_path = snapshot_config_module(preset.experiment_log_dir)
    print("Config:", args.config)
    print("Config snapshot:", config_snapshot_path)
    print("Data root:", preset.data_root)
    print("Experiment log dir:", preset.experiment_log_dir)
    print("Experiment dir status:", log_dir_status)
    print("Epochs:", preset.max_data_epochs)
    print("Validation frequency:", preset.val_epoch_freq)
    print("Generated SAM3 config:", config_path)
    print("Command:", " ".join(build_sam3_command(config_path, args)))
    launch_training(config_path)


if __name__ == "__main__":
    main()
