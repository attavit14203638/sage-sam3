"""Contract for the granularity-aware arm: prompt views, recipe checks, smoke gate, and efficacy rule.

Derives the category-preserving annotation views (`tree` for individual crowns and `tree canopy`
for unresolved canopy groups) from the collapsed export, verifies that a run's configuration
matches the locked recipe, validates the two-GPU smoke report and the pre-registered efficacy
declaration, and records the run manifest that a resume checkpoint must match. Existing artefacts
are never rewritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from project_paths import A0_INFERENCE_DIR, CODEBASE_DIR

ARM = "prompt_granularity"
RUN_NAME = "sam3_prompt_granularity"
CATEGORIES = [{"id": 1, "name": "tree canopy"}, {"id": 2, "name": "tree"}]
VIEW_FILES = {split: f"pg_{split}_annotations.coco.json" for split in ("train", "test")}
MANIFEST = "reports/prompt_granularity_view.json"
RUN_MANIFEST = "prompt_granularity_run.json"
VERSION = 2
SMOKE_REPORT_VERSION = 7
EFFICACY_VERSION = 1
TRAIN_IMAGES = 4169
TEST_IMAGES = 439
TEST_ANNOTATIONS = 30646
TEST_CATEGORY_COUNTS = {"1": 4951, "2": 25695}
FULL_MICRO_BATCH = 2
FULL_ACCUMULATION_STEPS = 4
FULL_TRAIN_WORKERS = 4
FULL_VAL_WORKERS = 2
SMOKE_CASES = ("mixed", "tree_only", "canopy_only", "empty")
LOCKED_RECIPE_HASHES = {
    "loss": "2004ea93f262d54cff3f4e5e0a84a1120902d8441f25577cf0604fa70f6d8d0a",
    "optim": "2312622096931de3cc1f9e692c9f04429b8dd5947b689ee79584dbe03c7cb73b",
    "scratch": "942ada0545e71d4858fec859dba29159062deed77de141509befef0297ce5ffe",
    "train_transforms": "cb6b8d4a1761716e2731a0783984c75bebc2e8176e98af95230056f8486a16fe",
    "val_transforms": "48ff8d5230063d03f25ec0d4460a7c162ef3216a08cd922c1d8f42024c0319fe",
}
ALLOWED_EFFECT_METRICS = {
    "Mask AP", "Mask AP50", "Mask AP75", "Mask AP_small", "Mask AP_medium",
    "Mask AP_large", "Mask AR@512", "Bnd AP instance", "Bnd AP_small",
    "Bnd AP_medium", "Bnd AP_large",
}


def encoded(value) -> bytes:
    """Serialise a value as canonical JSON bytes (sorted keys, compact separators, trailing newline)."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value) -> str:
    """Return the SHA-256 of a value's canonical JSON."""
    return hashlib.sha256(encoded(value)).hexdigest()


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file."""
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def prefix_identity(path: Path) -> dict:
    """Return the size and the SHA-256 of a file's first 64 MiB, the identity recorded for checkpoints."""
    path = Path(path).resolve()
    digest_value = hashlib.sha256()
    read_bytes = 0
    with path.open("rb") as handle:
        while read_bytes < 64 * 1024 * 1024:
            chunk = handle.read(min(4 * 1024 * 1024, 64 * 1024 * 1024 - read_bytes))
            if not chunk:
                break
            digest_value.update(chunk)
            read_bytes += len(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "hashed_bytes": read_bytes,
        "sha256_first_64mib": digest_value.hexdigest(),
    }


def _finite_number(value, *, positive: bool = False) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and (not positive or value > 0)
    )


def _require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} must be {expected!r}, got {actual!r}.")


def write_once(path: Path, value) -> None:
    """Write canonical JSON once: an identical existing file is accepted and a different one raises."""
    path = Path(path)
    payload = encoded(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite incompatible artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def derive_view(source: dict, split: str, *, strict: bool = True) -> dict:
    """Derive the category-preserving TRAIN or TEST view from the collapsed export, giving each annotation its original category and leaving geometry and crowd flags unchanged."""
    if split not in ("train", "test"):
        raise ValueError(f"Unsupported split: {split!r}")
    if source.get("categories") != [{"id": 1, "name": "tree"}]:
        raise ValueError("A prompt view must be derived from the unchanged collapsed A0 export.")
    images, annotations = source.get("images"), source.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("Source export must contain COCO images and annotations lists.")
    image_ids = [image.get("id") for image in images]
    annotation_ids = [annotation.get("id") for annotation in annotations]
    if any(type(value) is not int for value in image_ids + annotation_ids):
        raise ValueError("Every source image and annotation ID must be an integer.")
    if len(set(image_ids)) != len(images) or len(set(annotation_ids)) != len(annotations):
        raise ValueError("Duplicate image or annotation IDs in the source export.")
    if strict:
        _require_equal(len(images), TRAIN_IMAGES if split == "train" else TEST_IMAGES, f"{split} image count")
        if split == "test":
            _require_equal(len(annotations), TEST_ANNOTATIONS, "test annotation count")
    known = set(image_ids)
    for image in images:
        filename = Path(image.get("file_name", ""))
        if not str(filename) or filename.is_absolute() or ".." in filename.parts:
            raise ValueError(f"Image path must remain non-empty and relative to the export: {filename}")
        if strict and ((image.get("width"), image.get("height")) != (2048, 2048) or image.get("source_split") != split):
            raise ValueError("Source image size or split differs from the locked export.")
    view = copy.deepcopy(source)
    for annotation in view["annotations"]:
        category = annotation.get("original_category_id")
        if type(category) is not int or category not in (1, 2):
            raise ValueError(f"Missing/invalid original_category_id on annotation {annotation['id']}")
        if annotation.get("category_id") != 1 or annotation.get("image_id") not in known:
            raise ValueError("Source annotation category or image reference is invalid.")
        annotation["category_id"] = category
    view["info"] = {
        **view.get("info", {}),
        "category_policy": "Prompt granularity: original category 1 = tree canopy; 2 = tree. Geometry and crowd flags unchanged.",
    }
    view["categories"] = copy.deepcopy(CATEGORIES)
    return view


def validate_view_manifest(manifest: dict, *, strict: bool = True) -> None:
    """Check a view manifest's version, arm, categories, paths, hashes, and category and crowd counts; `strict` also requires the locked image and annotation counts."""
    _require_equal(manifest.get("version"), VERSION, "view manifest version")
    _require_equal(manifest.get("arm"), ARM, "view manifest arm")
    _require_equal(manifest.get("categories"), CATEGORIES, "view manifest categories")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "test"}:
        raise ValueError("View manifest must contain exactly train and test splits.")
    for split, expected_images in (("train", TRAIN_IMAGES), ("test", TEST_IMAGES)):
        block = splits[split]
        _require_equal(block.get("source"), f"annotations/{split}_annotations.coco.json", f"{split} source path")
        _require_equal(block.get("view"), f"annotations/{VIEW_FILES[split]}", f"{split} view path")
        for key in ("source_sha256", "view_sha256"):
            value = block.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{split} {key} must be a SHA-256 hex digest.")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"{split} {key} must be hexadecimal.") from exc
        if strict:
            _require_equal(block.get("images"), expected_images, f"{split} manifest image count")
        counts = block.get("original_category_counts")
        if not isinstance(counts, dict) or set(counts) != {"1", "2"}:
            raise ValueError(f"{split} must contain both original categories.")
        if any(type(value) is not int or value <= 0 for value in counts.values()):
            raise ValueError(f"{split} category counts must be positive integers.")
        _require_equal(sum(counts.values()), block.get("annotations"), f"{split} category-count total")
        crowd_counts = block.get("crowd_counts_by_category")
        if not isinstance(crowd_counts, dict) or set(crowd_counts) != {"1", "2"}:
            raise ValueError(f"{split} crowd counts must be separated by category.")
        for category in ("1", "2"):
            values = crowd_counts[category]
            if not isinstance(values, dict) or not set(values) <= {"0", "1"} or any(type(value) is not int or value < 0 for value in values.values()):
                raise ValueError(f"{split} category {category} crowd counts are invalid.")
            _require_equal(sum(values.values()), counts[category], f"{split} category {category} crowd-count total")
        if strict and split == "train" and any(crowd_counts[category].get("0", 0) <= 0 for category in ("1", "2")):
            raise ValueError("TRAIN must contain non-crowd supervision for both prompt categories after native FilterCrowds.")
        if strict and split == "test":
            _require_equal(block.get("annotations"), TEST_ANNOTATIONS, "test manifest annotation count")
            _require_equal(counts, TEST_CATEGORY_COUNTS, "test category counts")


def _expected_views(root: Path, *, strict: bool):
    manifest = {"version": VERSION, "arm": ARM, "categories": CATEGORIES, "splits": {}}
    views = {}
    for split in ("train", "test"):
        source_path = root / "annotations" / f"{split}_annotations.coco.json"
        source_bytes = source_path.read_bytes()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        source = json.loads(source_bytes)
        del source_bytes
        view = derive_view(source, split, strict=strict)
        if strict:
            missing = next((image["file_name"] for image in source["images"] if not (root / image["file_name"]).is_file()), None)
            if missing:
                raise FileNotFoundError(f"Source image missing: {root / missing}")
            if split == "test":
                reference = json.loads((A0_INFERENCE_DIR / "eval_tree.json").read_text())
                if reference.get("dataset_protocol", {}).get("annotation_sha256") != source_sha:
                    raise ValueError("TEST source hash differs from the frozen A0 evaluation provenance.")
        path = Path("annotations") / VIEW_FILES[split]
        manifest["splits"][split] = {
            "source": str(source_path.relative_to(root)), "source_sha256": source_sha,
            "view": str(path), "view_sha256": digest(view),
            "images": len(source["images"]), "annotations": len(source["annotations"]),
            "original_category_counts": dict(sorted(Counter(str(a["original_category_id"]) for a in source["annotations"]).items())),
            "crowd_counts_by_category": {
                str(category): dict(sorted(Counter(str(int(bool(a.get("iscrowd", 0)))) for a in source["annotations"] if a["original_category_id"] == category).items()))
                for category in (1, 2)
            },
        }
        views[path] = view
    validate_view_manifest(manifest, strict=strict)
    return manifest, views


def prepare_views(data_root: Path, *, strict: bool = True) -> dict:
    """Write the TRAIN and TEST views and their manifest beside the collapsed export, refusing to overwrite a changed one."""
    root = Path(data_root).resolve()
    manifest, views = _expected_views(root, strict=strict)
    manifest_path = root / MANIFEST
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Source/view manifest changed; existing prompt view will not be overwritten.")
    for relative, payload in views.items():
        path = root / relative
        if path.exists() and file_digest(path) != digest(payload):
            raise ValueError(f"Existing prompt view differs from its source: {path}")
    for relative, payload in views.items():
        write_once(root / relative, payload)
    write_once(manifest_path, manifest)
    return manifest


def validate_views(data_root: Path, *, strict: bool = True) -> dict:
    """Check that the stored views and manifest still match the current export."""
    root = Path(data_root).resolve()
    stored = json.loads((root / MANIFEST).read_text())
    expected, views = _expected_views(root, strict=strict)
    if stored != expected:
        raise ValueError("Prompt view manifest does not match the current source export.")
    for relative, payload in views.items():
        if file_digest(root / relative) != digest(payload):
            raise ValueError(f"Prompt view integrity check failed: {relative}")
    return stored


def source_fingerprint() -> str:
    """Hash every Core module and the vendored SAM3 source so a smoke report can be tied to the code it ran."""
    files = sorted((CODEBASE_DIR / "Core").glob("*.py"))
    files += sorted((CODEBASE_DIR / "Support/sam3/sam3").rglob("*.py"))
    return digest({str(path.relative_to(CODEBASE_DIR)): file_digest(path) for path in files})


def validate_training_recipe(config, cfg: dict, view_manifest: dict) -> dict:
    """Check that the built configuration matches the locked granularity-aware recipe and return its summary."""
    validate_view_manifest(view_manifest)
    expected_constants = {
        "MODEL_BACKEND": "sam3", "PROMPT_GRANULARITY_ENABLED": True,
        "FULL_FT_RUN_NAME": RUN_NAME, "TRAIN_ANN_FILENAME": VIEW_FILES["train"],
        "VAL_ANN_FILENAME": VIEW_FILES["test"], "TEST_ANN_FILENAME": "test_annotations.coco.json",
        "FULL_FT_MAX_DATA_EPOCHS": 30, "FULL_FT_TRAINER_MODE": "train",
        "FULL_FT_TRAIN_LIMIT_IDS": None, "FULL_FT_VAL_LIMIT_IDS": None,
        "FULL_FT_VAL_EPOCH_FREQ": 3, "FULL_FT_NUM_TRAIN_WORKERS": FULL_TRAIN_WORKERS,
        "FULL_FT_NUM_VAL_WORKERS": FULL_VAL_WORKERS, "RECIPE_BATCH_SIZE": 8,
        "TRAIN_BATCH_SIZE": FULL_MICRO_BATCH, "VAL_BATCH_SIZE": FULL_MICRO_BATCH,
        "GRADIENT_ACCUMULATION_STEPS": FULL_ACCUMULATION_STEPS,
        "ACT_CKPT_VISION_BACKBONE": True, "AMP_ENABLED": True, "AMP_DTYPE": "bfloat16",
        "TRAINER_SEED": 42, "TRAINER_ACCELERATOR": "cuda", "DDP_BACKEND": "nccl",
        "DDP_FIND_UNUSED_PARAMETERS": False, "DDP_GRADIENT_AS_BUCKET_VIEW": False,
        "DDP_STATIC_GRAPH": True, "D_MODEL": 256, "RESOLUTION": 1008,
        "MAX_TRAIN_QUERIES": 10, "MAX_VAL_QUERIES": 10, "MAX_ANN_PER_IMG": 1000,
        "CROP_MIN_SIZE": 896, "CROP_MAX_SIZE": 1152,
        "CROP_RESPECT_BOXES": False, "CROP_RESPECT_INPUT_BOXES": False,
        "HFLIP_P": 0.5, "VFLIP_P": 0.5, "ROT90_P": 0.75,
        "COLORJITTER_P": 0.5, "COLORJITTER_BRIGHTNESS": 0.25,
        "COLORJITTER_CONTRAST": 0.25, "COLORJITTER_SATURATION": 0.25,
        "COLORJITTER_HUE": 0.1, "RESIZE_MIN_SIZE": 896,
        "SEMANTIC_SEG_ENABLED": True, "SEMANTIC_SEG_DOWNSAMPLE": False,
        "MASK_LOSS_ENABLED": True, "MASK_LOSS_MODE": "sampled", "MASK_COMPUTE_AUX": False,
        "MASK_NUM_SAMPLE_POINTS": 12544, "MASK_OVERSAMPLE_RATIO": 3.0,
        "MASK_IMPORTANCE_SAMPLE_RATIO": 0.75, "LOSS_MASK_WEIGHT": 200.0,
        "LOSS_DICE_WEIGHT": 10.0, "LOSS_BBOX_WEIGHT": 5.0, "LOSS_GIOU_WEIGHT": 2.0,
        "LOSS_CE_WEIGHT": 20.0, "LOSS_PRESENCE_WEIGHT": 20.0,
        "LOSS_SEMANTIC_SEG_WEIGHT": 20.0, "O2M_WEIGHT": 2.0, "O2M_TOPK": 4,
        "SCALE_BY_FIND_BATCH_SIZE": True, "GRAD_CLIP_MAX_NORM": 0.1,
        "LR_SCALE": 0.1, "LR_TRANSFORMER": 8e-5, "LR_VISION_BACKBONE": 2.5e-5,
        "LR_LANGUAGE_BACKBONE": 5e-6, "WARMUP_FRACTION": 0.05,
        "COSINE_END_LR_RATIO": 0.01, "VAL_CATEGORY_CHUNK_SIZE": 1,
        "INCLUDE_NEGATIVES_VAL": True, "DATASET_MULTIPLIER": 1,
        "DATALOADER_DROP_LAST_TRAIN": True, "DATALOADER_DROP_LAST_VAL": False,
        "DATALOADER_PIN_MEMORY": True, "DATALOADER_SHUFFLE_TRAIN": True,
        "DATALOADER_SHUFFLE_VAL": False, "TRAINER_TARGET": "progress_trainer.ProgressBarTrainer",
        "TRAINER_SKIP_FIRST_VAL": True, "TRAINER_SKIP_SAVING_CKPTS": False,
        "TRAINER_EMPTY_GPU_MEM_CACHE_AFTER_EVAL": True, "GRAD_CLIP_NORM_TYPE": 2,
        "LRD_VISION_BACKBONE": 0.95, "WEIGHT_DECAY": 0.1,
        "MATCHER_FOCAL": True, "MATCHER_COST_CLASS": 2.0, "MATCHER_COST_BBOX": 5.0,
        "MATCHER_COST_GIOU": 2.0, "MATCHER_ALPHA": 0.25, "MATCHER_GAMMA": 2,
        "MATCHER_STABLE": False, "O2M_ALPHA": 0.3, "O2M_THRESHOLD": 0.4,
        "USE_O2M_MATCHER_ON_O2M_AUX": False, "SEMANTIC_SEG_FOCAL": False,
        "IABCE_POS_WEIGHT": 10.0, "IABCE_ALPHA": 0.25, "IABCE_GAMMA": 2,
        "IABCE_USE_PRESENCE": True, "IABCE_POS_FOCAL": False,
        "IABCE_PAD_N_QUERIES": 200, "IABCE_PAD_SCALE_POS": 1.0,
        "IABCE_WEAK_LOSS": False, "MASK_FOCAL_ALPHA": 0.25,
        "MASK_FOCAL_GAMMA": 2.0, "BOX_NOISE_STD": 0.1, "BOX_NOISE_MAX": 20,
        "RESIZE_ROUNDED": False, "RESIZE_SQUARE": True, "CONSISTENT_TRANSFORM": False,
        "ENABLE_SEGMENTATION": True, "USE_CACHING": False, "LOAD_SEGMENTATION": True,
        "WITH_SEG_MASKS": True, "COLLATE_REPEATS": 1, "CKPT_SAVE_FREQ": 0,
    }
    for name, expected in expected_constants.items():
        _require_equal(getattr(config, name), expected, f"config.{name}")

    trainer = cfg["trainer"]
    _require_equal(cfg["launcher"]["gpus_per_node"], 2, "launcher GPU count")
    _require_equal(
        (trainer["_target_"], trainer["skip_saving_ckpts"], trainer["empty_gpu_mem_cache_after_eval"], trainer["skip_first_val"]),
        ("progress_trainer.ProgressBarTrainer", False, True, True),
        "trainer implementation",
    )
    _require_equal((trainer["max_epochs"], trainer["mode"], trainer["seed_value"], trainer["accelerator"], trainer["val_epoch_freq"]), (30, "train", 42, "cuda", 3), "trainer schedule")
    _require_equal(trainer["distributed"], {
        "backend": "nccl", "find_unused_parameters": False,
        "gradient_as_bucket_view": False, "static_graph": True,
    }, "DDP configuration")
    _require_equal(trainer["optim"]["amp"], {"enabled": True, "amp_dtype": "bfloat16"}, "AMP configuration")
    _require_equal(trainer["optim"]["optimizer"]["_target_"], "torch.optim.AdamW", "optimizer")
    _require_equal(trainer["optim"]["gradient_clip"]["max_norm"], 0.1, "gradient clip")
    _require_equal(trainer["model"], {
        "_target_": "training_pipeline.build_finetuned_sam3_image_model",
        "bpe_path": None,
        "checkpoint_path": str(CODEBASE_DIR / "runtime/checkpoints/sam3/sam3.pt"),
        "load_from_HF": False,
        "device": "cpus",
        "eval_mode": False,
        "enable_segmentation": True,
    }, "model builder configuration")
    _require_equal(Path(trainer["model"]["checkpoint_path"]).resolve(), (CODEBASE_DIR / "runtime/checkpoints/sam3/sam3.pt").resolve(), "pretrained checkpoint")

    root = Path(cfg["paths"]["coco_root"]).resolve()
    _require_equal(Path(cfg["paths"]["train_ann_file"]).resolve(), root / "annotations" / VIEW_FILES["train"], "TRAIN view")
    _require_equal(Path(cfg["paths"]["val_ann_file"]).resolve(), root / "annotations" / VIEW_FILES["test"], "VAL view")
    train = trainer["data"]["train"]
    val = trainer["data"]["val"]
    _require_equal(train["_target_"], "project_torch_dataset.ProjectTorchDataset", "TRAIN wrapper")
    _require_equal((train["batch_size"], train["gradient_accumulation_steps"], train["num_workers"], train["drop_last"], train["shuffle"], train["pin_memory"]),
                   (FULL_MICRO_BATCH, FULL_ACCUMULATION_STEPS, FULL_TRAIN_WORKERS, True, True, True), "TRAIN loader recipe")
    _require_equal((val["_target_"], val["batch_size"], val["num_workers"], val["drop_last"], val["shuffle"], val["pin_memory"]),
                   ("sam3.train.data.torch_dataset.TorchDataset", FULL_MICRO_BATCH, FULL_VAL_WORKERS, False, False, True), "VAL loader recipe")
    for split, block, expected_training, expected_chunk in (
        ("train", train, True, 2), ("val", val, False, 1),
    ):
        dataset = block["dataset"]
        _require_equal(dataset["_target_"], "sam3.train.data.sam3_image_dataset.Sam3ImageDataset", f"{split} dataset")
        _require_equal(dataset["limit_ids"], None, f"{split} data limit")
        _require_equal(dataset["training"], expected_training, f"{split} training flag")
        _require_equal(
            (dataset["load_segmentation"], dataset["max_ann_per_img"], dataset["multiplier"], dataset["max_train_queries"], dataset["max_val_queries"]),
            (True, 1000, 1, 10, 10),
            f"{split} dataset recipe",
        )
        _require_equal(Path(dataset["img_folder"]).resolve(), root, f"{split} image root")
        if split == "train":
            _require_equal(dataset.get("use_caching"), False, "TRAIN caching")
        loader = dataset["coco_json_loader"]
        _require_equal((loader["include_negatives"], loader["category_chunk_size"]), (True, expected_chunk), f"{split} category loader")
        collate = block["collate_fn"]
        _require_equal(collate["_target_"], "prompt_granularity_training.collate_fn_prompt_granularity", f"{split} collator")
        _require_equal((collate.get("_partial_"), collate.get("dict_key"), collate.get("queries_per_image")), (True, "all" if split == "train" else "oamtcd", expected_chunk), f"{split} collator contract")
        if split == "val":
            if "collate_fn" not in collate["_target_"].rsplit(".", 1)[-1]:
                raise ValueError("VAL collator target is invisible to native trainer key discovery.")
            _require_equal(set(trainer["meters"]["val"]), {collate["dict_key"]}, "VAL dataset/meter keys")

    losses = trainer["loss"]
    for key, prompts in (("all", 2), ("oamtcd", 1)):
        loss = losses[key]
        _require_equal(loss["_target_"], "prompt_granularity_training.PromptGranularityLoss", f"{key} loss wrapper")
        _require_equal((loss["queries_per_image"], loss["scale_by_find_batch_size"]), (prompts, True), f"{key} loss scaling")
        _require_equal(loss["o2m_weight"], 2.0, f"{key} O2M weight")
        _require_equal(loss["o2m_matcher"]["topk"], 4, f"{key} O2M top-k")
        mask = loss["loss_fns_find"][-1]
        _require_equal((mask["_target_"], mask["compute_aux"], mask["weight_dict"]),
                       ("eval_overrides.ProjectSampledMasks", False, {"loss_mask": 200.0, "loss_dice": 10.0}), f"{key} mask loss")
        semantic = loss["loss_fn_semantic_seg"]
        _require_equal((semantic["_target_"], semantic["downsample"], semantic["weight_dict"]),
                       ("eval_overrides.ProjectSemanticSegCriterion", False, {"loss_semantic_seg": 20.0}), f"{key} semantic loss")
    _require_equal(losses["default"]["_target_"], "sam3.train.loss.sam3_loss.DummyLoss", "VAL dummy loss")

    transform_targets = [item["_target_"] for item in cfg["oam_tcd"]["train_transforms"][0]["transforms"]]
    _require_equal(transform_targets, [
        "sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries",
        "sam3.train.transforms.point_sampling.RandomizeInputBbox",
        "sam3.train.transforms.segmentation.DecodeRle",
        "sam3.train.transforms.basic_for_api.RandomSizeCropAPI",
        "sam3.train.transforms.basic_for_api.RandomHorizontalFlip",
        "augmentation.RandomVerticalFlip", "augmentation.RandomRot90",
        "sam3.train.transforms.basic_for_api.RandomSelectAPI",
        "sam3.train.transforms.basic_for_api.RandomResizeAPI",
        "sam3.train.transforms.basic_for_api.PadToSizeAPI",
        "sam3.train.transforms.basic_for_api.ToTensorAPI",
        "sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries",
        "sam3.train.transforms.basic_for_api.NormalizeAPI",
        "sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries",
    ], "TRAIN transform chain")
    required_metrics = {
        f"Losses/train_all_pg_{name}" for name in
        ("pair_count", "image_count", "scale", "tree_targets", "canopy_targets")
    }
    if not required_metrics <= set(trainer["logging"]["scalar_keys_to_log"]):
        raise ValueError("Prompt-granularity telemetry is missing from the logger whitelist.")
    recipe_blocks = {
        "loss": trainer["loss"],
        "optim": trainer["optim"],
        "scratch": cfg["scratch"],
        "train_transforms": cfg["oam_tcd"]["train_transforms"],
        "val_transforms": cfg["oam_tcd"]["val_transforms"],
    }
    for name, expected in LOCKED_RECIPE_HASHES.items():
        _require_equal(digest(recipe_blocks[name]), expected, f"locked {name} fingerprint")
    return {
        "train_images": view_manifest["splits"]["train"]["images"],
        "test_images": view_manifest["splits"]["test"]["images"],
        "micro_batch_per_rank": FULL_MICRO_BATCH,
        "accumulation_steps": FULL_ACCUMULATION_STEPS,
        "world_size": 2,
        "effective_image_batch": FULL_MICRO_BATCH * FULL_ACCUMULATION_STEPS * 2,
        "train_workers_per_rank": FULL_TRAIN_WORKERS,
        "val_workers_per_rank": FULL_VAL_WORKERS,
        "train_queries_per_image": 2,
        "val_queries_per_entry": 1,
        "checkpoint_backbone": True,
        "ddp_accumulation_sync": "every_microbatch_plus_explicit_final_when_static_graph",
    }


def training_contract(data_root: Path) -> dict:
    """Assemble the training contract: validated views, recipe, source fingerprint, package versions, and pretrained-weight identity."""
    from importlib.metadata import version
    import config
    if not config.PROMPT_GRANULARITY_ENABLED:
        raise ValueError("Prompt-granularity is not enabled in this process.")
    view_manifest = validate_views(data_root)
    cfg = config.build_sam3_train_config(config.get_training_preset("full_ft", data_root=Path(data_root)))
    recipe = validate_training_recipe(config, cfg, view_manifest)
    weights = Path(cfg["trainer"]["model"]["checkpoint_path"])
    return {
        "version": VERSION, "arm": ARM, "run_name": RUN_NAME,
        "views_sha256": digest(view_manifest), "source_sha256": source_fingerprint(),
        "packages": {name: version(name) for name in ("torch", "torchvision", "numpy", "hydra-core", "omegaconf", "pycocotools", "triton")},
        "pretrained": prefix_identity(weights), "recipe": recipe,
        "loss": cfg["trainer"]["loss"], "optim": cfg["trainer"]["optim"],
        "train_transforms": cfg["oam_tcd"]["train_transforms"],
        "val_transforms": cfg["oam_tcd"]["val_transforms"],
    }


def _parse_timezone_time(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 string.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be valid ISO-8601.") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone.")
    if parsed > datetime.now(parsed.tzinfo) + timedelta(minutes=5):
        raise ValueError(f"{label} cannot be in the future.")


def validate_efficacy_spec(path: Path, *, a0_eval_path: Path | None = None) -> dict:
    """Validate the pre-registered efficacy declaration (approval, baseline evaluation identity, and criteria) and return it with its hash."""
    path = Path(path).resolve()
    declaration = json.loads(path.read_text())
    _require_equal(declaration.get("version"), EFFICACY_VERSION, "efficacy declaration version")
    _require_equal(declaration.get("arm"), ARM, "efficacy declaration arm")
    _require_equal(declaration.get("status"), "approved", "efficacy declaration status")
    _require_equal(declaration.get("approved_by"), "user", "efficacy approver")
    _parse_timezone_time(declaration.get("approved_utc"), "approval time")
    _require_equal(declaration.get("single_seed_interpretation"), "descriptive_unless_replicated", "single-seed interpretation")
    a0_eval = Path(a0_eval_path or (A0_INFERENCE_DIR / "eval_tree.json")).resolve()
    _require_equal(declaration.get("a0_eval_sha256"), file_digest(a0_eval), "A0 evaluation hash")
    a0 = json.loads(a0_eval.read_text())
    _require_equal(declaration.get("a0_annotation_sha256"), a0.get("dataset_protocol", {}).get("annotation_sha256"), "A0 annotation hash")
    criteria = declaration.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("Efficacy declaration must contain criteria.")
    roles = []
    metrics = []
    for criterion in criteria:
        if not isinstance(criterion, dict) or set(criterion) != {"role", "metric", "minimum_delta"}:
            raise ValueError("Each efficacy criterion must contain only role, metric, and minimum_delta.")
        role, metric, threshold = criterion["role"], criterion["metric"], criterion["minimum_delta"]
        if role not in ("primary_effect", "non_regression") or metric not in ALLOWED_EFFECT_METRICS:
            raise ValueError(f"Unsupported efficacy criterion: {criterion!r}")
        if not _finite_number(threshold):
            raise ValueError("Every efficacy minimum_delta must be finite.")
        if role == "primary_effect" and threshold <= 0:
            raise ValueError("A primary effect margin must be strictly positive.")
        if role == "non_regression" and threshold > 0:
            raise ValueError("A non-regression floor cannot require a positive gain.")
        roles.append(role)
        metrics.append(metric)
    if roles.count("primary_effect") != 1 or "non_regression" not in roles:
        raise ValueError("Declare exactly one primary effect and at least one non-regression criterion.")
    if len(metrics) != len(set(metrics)):
        raise ValueError("Efficacy criteria must use distinct metrics.")
    return {"path": str(path), "sha256": file_digest(path), "declaration": declaration}


def require_smoke(path: Path, contract: dict) -> dict:
    """Validate a two-rank smoke report against the training contract and return it; any mismatch raises."""
    path = Path(path).resolve()
    if path.name != "smoke_report.json" or not path.parent.name.startswith(f"{RUN_NAME}_smoke"):
        raise ValueError("Smoke report must live in its dedicated prompt-granularity smoke directory.")
    report = json.loads(path.read_text())
    _require_equal(report.get("report_version"), SMOKE_REPORT_VERSION, "smoke report version")
    _require_equal(report.get("status"), "passed", "smoke status")
    _require_equal(report.get("world_size"), 2, "smoke world size")
    _require_equal(report.get("output_dir"), str(path.parent), "smoke output directory")
    _require_equal(report.get("contract"), contract, "smoke training contract")
    _require_equal(report.get("contract_sha256"), digest(contract), "smoke contract hash")
    _parse_timezone_time(report.get("completed_utc"), "smoke completion time")
    _require_equal(report.get("skipped_checks"), [], "smoke skipped checks")
    ranks = report.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 2 or any(not isinstance(row, dict) for row in ranks):
        raise ValueError("Smoke report must contain exactly two rank records.")
    if {row.get("rank") for row in ranks} != {0, 1}:
        raise ValueError("Smoke report must contain exactly ranks 0 and 1.")
    recipe = contract.get("recipe", {})
    query_categories = set()
    prediction_categories = set()
    for row in ranks:
        rank = row["rank"]
        _require_equal(row.get("local_rank"), rank, f"rank {rank} local rank")
        _require_equal(row.get("world_size"), 2, f"rank {rank} world size")
        if "A40" not in str(row.get("device")):
            raise ValueError(f"Rank {rank} did not run on an A40.")
        _require_equal(row.get("amp_dtype"), "bfloat16", f"rank {rank} AMP dtype")
        _require_equal(row.get("micro_batch"), recipe.get("micro_batch_per_rank"), f"rank {rank} microbatch")
        _require_equal(row.get("accumulation_steps"), recipe.get("accumulation_steps"), f"rank {rank} accumulation")
        _require_equal(row.get("ddp_accumulation_sync"), recipe.get("ddp_accumulation_sync"), f"rank {rank} DDP accumulation policy")
        _require_equal(row.get("effective_image_batch"), recipe.get("effective_image_batch"), f"rank {rank} effective image batch")
        _require_equal(row.get("optimizer_steps"), len(SMOKE_CASES) + 1, f"rank {rank} optimizer steps")
        _require_equal(row.get("skipped_optimizer_steps"), 0, f"rank {rank} skipped optimizer steps")
        _require_equal(row.get("architecture"), {
            "num_queries": 200, "num_layers": 6, "dac": True,
            "o2m_mask_predict": True, "aux_masks": False,
        }, f"rank {rank} model architecture")
        sources = row.get("sources", {})
        for name, parent in (("core", CODEBASE_DIR / "Core"), ("sam3", CODEBASE_DIR / "Support/sam3")):
            source = sources.get(name)
            if not isinstance(source, str) or not Path(source).resolve().is_relative_to(parent.resolve()):
                raise ValueError(f"Rank {rank} imported {name} from an unexpected location: {source!r}")
        fixtures = row.get("fixtures")
        if not isinstance(fixtures, list) or [entry.get("case") for entry in fixtures] != list(SMOKE_CASES):
            raise ValueError(f"Rank {rank} is missing ordered smoke fixtures.")
        for fixture in fixtures:
            case = fixture["case"]
            _require_equal(fixture.get("microbatches"), recipe.get("accumulation_steps"), f"{case} microbatch count")
            _require_equal(fixture.get("optimizer_step_applied"), True, f"{case} optimizer step")
            _require_equal(fixture.get("ddp_gradient_equal"), True, f"{case} DDP gradient equality")
            _require_equal(fixture.get("explicit_gradient_sync"), True, f"{case} explicit gradient synchronization")
            losses = fixture.get("losses")
            if not isinstance(losses, list) or len(losses) != recipe.get("accumulation_steps") or any(not _finite_number(value) for value in losses):
                raise ValueError(f"Rank {rank} {case} losses are missing or non-finite.")
            if not _finite_number(fixture.get("grad_norm"), positive=True):
                raise ValueError(f"Rank {rank} {case} gradient norm is invalid.")
            mask_grad = fixture.get("mask_grad")
            if not _finite_number(mask_grad) or mask_grad < 0 or (case != "empty" and mask_grad == 0):
                raise ValueError(f"Rank {rank} {case} mask gradient is invalid.")
            pairs = 2 * recipe["micro_batch_per_rank"]
            _require_equal(fixture.get("output_shapes"), {
                "pred_masks": [pairs, 200, 288, 288],
                "pred_logits": [pairs, 200, 1],
                "pred_masks_o2m": [pairs, 200, 288, 288],
                "pred_logits_o2m": [pairs, 200, 1],
            }, f"rank {rank} {case} output shapes")
            counts = fixture.get("target_counts", {})
            tree, canopy = counts.get("tree"), counts.get("tree_canopy")
            if any(type(value) is not int or value < 0 for value in (tree, canopy)):
                raise ValueError(f"Rank {rank} {case} target counts are invalid.")
            if case == "mixed" and not (tree > 0 and canopy > 0):
                raise ValueError("Mixed fixture must retain both categories.")
            if case == "tree_only" and not (tree > 0 and canopy == 0):
                raise ValueError("Tree-only fixture target counts are inconsistent.")
            if case == "canopy_only" and not (tree == 0 and canopy > 0):
                raise ValueError("Canopy-only fixture target counts are inconsistent.")
            if case == "empty" and (tree != 0 or canopy != 0):
                raise ValueError("Empty fixture contains targets.")
            telemetry = fixture.get("telemetry", [])
            if not isinstance(telemetry, list) or len(telemetry) != recipe.get("accumulation_steps"):
                raise ValueError(f"Rank {rank} {case} telemetry is incomplete.")
            for values in telemetry:
                if not isinstance(values, dict):
                    raise ValueError(f"Rank {rank} {case} telemetry entry is invalid.")
                _require_equal(values.get("pg_pair_count"), 2 * recipe["micro_batch_per_rank"], "smoke pair count")
                _require_equal(values.get("pg_image_count"), recipe["micro_batch_per_rank"], "smoke image count")
                if not math.isclose(values.get("pg_scale", -1), 1 / math.sqrt(2), rel_tol=0, abs_tol=1e-7):
                    raise ValueError("Smoke prompt normalization factor is wrong.")
                for name in ("pg_tree_targets", "pg_canopy_targets"):
                    value = values.get(name)
                    if not _finite_number(value) or value < 0 or not float(value).is_integer():
                        raise ValueError(f"Rank {rank} {case} {name} telemetry is invalid.")
            _require_equal(sum(values["pg_tree_targets"] for values in telemetry), tree, f"rank {rank} {case} tree telemetry")
            _require_equal(sum(values["pg_canopy_targets"] for values in telemetry), canopy, f"rank {rank} {case} canopy telemetry")
        integration = row.get("production_integration", {})
        _require_equal(integration.get("status"), "passed", f"rank {rank} production integration")
        _require_equal(integration.get("train_workers"), recipe.get("train_workers_per_rank"), f"rank {rank} TRAIN workers")
        _require_equal(integration.get("val_workers"), recipe.get("val_workers_per_rank"), f"rank {rank} VAL workers")
        _require_equal(integration.get("train_microbatches"), recipe.get("accumulation_steps"), f"rank {rank} production microbatches")
        _require_equal(integration.get("optimizer_step_applied"), True, f"rank {rank} production optimizer step")
        _require_equal(integration.get("ddp_gradient_equal"), True, f"rank {rank} production DDP gradient equality")
        _require_equal(integration.get("explicit_gradient_sync"), True, f"rank {rank} production explicit gradient synchronization")
        if not _finite_number(integration.get("grad_norm"), positive=True) or not _finite_number(integration.get("mask_grad"), positive=True):
            raise ValueError(f"Rank {rank} production gradient evidence is invalid.")
        pairs = 2 * recipe["micro_batch_per_rank"]
        _require_equal(integration.get("output_shapes"), {
            "pred_masks": [pairs, 200, 288, 288],
            "pred_logits": [pairs, 200, 1],
            "pred_masks_o2m": [pairs, 200, 288, 288],
            "pred_logits_o2m": [pairs, 200, 1],
        }, f"rank {rank} production output shapes")
        production_counts = integration.get("target_counts", {})
        if any(type(production_counts.get(category)) is not int or production_counts[category] < 0 for category in ("tree", "tree_canopy")):
            raise ValueError(f"Rank {rank} production target counts are invalid.")
        validation = row.get("validation", {})
        _require_equal(validation.get("status"), "passed", f"rank {rank} validation")
        _require_equal(validation.get("dummy_loss"), True, f"rank {rank} validation dummy-loss behavior")
        _require_equal(validation.get("postprocessing"), "passed", f"rank {rank} validation postprocessing")
        _require_equal(validation.get("dense_forward"), "passed", f"rank {rank} dense validation forward")
        _require_equal(validation.get("checkpoint_reload_equal"), True, f"rank {rank} checkpoint reload")
        _require_equal(validation.get("checkpoint_fresh_model"), True, f"rank {rank} fresh-model reload")
        _require_equal(validation.get("checkpoint_shared_inputs"), True, f"rank {rank} reload inputs")
        _require_equal(validation.get("checkpoint_optimizer_steps"), len(SMOKE_CASES) + 1, f"rank {rank} restored optimizer steps")
        query_categories.update(validation.get("query_categories", []))
        prediction_categories.update(validation.get("prediction_categories", []))
        memory = row.get("memory_mib", {})
        allocated, reserved = memory.get("peak_allocated"), memory.get("peak_reserved")
        if not _finite_number(allocated, positive=True) or not _finite_number(reserved, positive=True) or reserved < allocated:
            raise ValueError(f"Rank {rank} memory evidence is invalid.")
    _require_equal(query_categories, {1, 2}, "global validation query categories")
    _require_equal(prediction_categories, {1, 2}, "global validation prediction categories")
    checkpoint = report.get("checkpoint", {})
    _require_equal(checkpoint.get("status"), "passed", "smoke checkpoint status")
    _require_equal(checkpoint.get("reload_equal"), True, "smoke checkpoint reload")
    _require_equal(checkpoint.get("purpose"), "smoke", "smoke checkpoint purpose")
    _require_equal(checkpoint.get("contract_sha256"), digest(contract), "smoke checkpoint contract")
    _require_equal(checkpoint.get("state_keys"), ["contract", "loss", "model", "optimizer", "purpose", "scaler"], "smoke checkpoint states")
    checkpoint_path = path.parent / "smoke_checkpoint.pt"
    expected_identity = prefix_identity(checkpoint_path)
    for key in ("path", "bytes", "hashed_bytes", "sha256_first_64mib"):
        _require_equal(checkpoint.get(key), expected_identity[key], f"smoke checkpoint {key}")
    return report


def smoke_peak_reserved_mib(path: Path) -> float:
    """Return the larger of the two ranks' peak reserved GPU memory, in MiB, from a smoke report."""
    report = json.loads(Path(path).read_text())
    _require_equal(report.get("report_version"), SMOKE_REPORT_VERSION, "smoke report version")
    ranks = report.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 2:
        raise ValueError("Smoke report must contain two rank records.")
    peaks = [row.get("memory_mib", {}).get("peak_reserved") for row in ranks if isinstance(row, dict)]
    if len(peaks) != 2 or any(not _finite_number(value, positive=True) for value in peaks):
        raise ValueError("Smoke report has no valid two-rank reserved-memory measurement.")
    return max(peaks)


def verify_checkpoint_contract(checkpoint: dict, run_manifest: dict) -> None:
    """Check that a resume checkpoint belongs to this full run: its recorded contract equals the run manifest and it holds model, optimiser, loss, and scaler state."""
    recorded = checkpoint.get("prompt_granularity", {})
    if recorded != run_manifest or recorded.get("purpose") != "full":
        raise ValueError("Resume checkpoint is not from this full prompt-granularity run and contract; smoke, A0, unrelated, and mismatched checkpoints are rejected.")
    for key in ("model", "optimizer", "loss", "scaler"):
        if not isinstance(checkpoint.get(key), dict):
            raise ValueError(f"Resume checkpoint is missing its {key} state.")
    if type(checkpoint.get("epoch")) is not int or checkpoint["epoch"] < 0 or type(checkpoint.get("steps")) is not int or checkpoint["steps"] < 0:
        raise ValueError("Resume checkpoint epoch/step state is invalid.")


def validate_launch(
    data_root: Path,
    report_path: Path,
    efficacy_path: Path,
    *,
    resume_from: Path | None = None,
    run_dir: Path | None = None,
) -> dict:
    """Run every pre-launch check (two A40 GPUs, training contract, smoke report, efficacy declaration, and any resume checkpoint) and return the run manifest."""
    import config
    import torch
    if torch.cuda.device_count() != 2 or any("A40" not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("Full prompt-granularity training requires two visible A40 GPUs.")
    contract = training_contract(data_root)
    require_smoke(report_path, contract)
    run_manifest = {
        "purpose": "full",
        "contract": contract,
        "efficacy": validate_efficacy_spec(efficacy_path),
    }
    if resume_from is not None:
        checkpoint = torch.load(Path(resume_from), map_location="cpu", mmap=True, weights_only=False)
        verify_checkpoint_contract(checkpoint, run_manifest)
        del checkpoint
        if run_dir is None or json.loads((Path(run_dir) / RUN_MANIFEST).read_text()) != run_manifest:
            raise ValueError("The run manifest differs from the staged checkpoint contract.")
    return run_manifest


def main() -> None:
    """Command-line entry point: prepare, verify, smoke, verify-smoke, or preflight."""
    parser = argparse.ArgumentParser(description="Gated prompt-granularity preparation and smoke testing; never rewrites A0 assets.")
    parser.add_argument("action", choices=("prepare", "verify", "smoke", "verify-smoke", "preflight"))
    parser.add_argument("--data-root", type=Path, default=CODEBASE_DIR / "runtime/data/oam_tcd_instance_coco")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--efficacy-spec", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--checkpoint-backbone", action="store_true")
    args = parser.parse_args()
    if args.action in ("prepare", "verify"):
        result = prepare_views(args.data_root) if args.action == "prepare" else validate_views(args.data_root)
        print(json.dumps(result, indent=2))
        return
    if args.action == "smoke" and args.output is None:
        parser.error("smoke requires a new --output directory, separate from the full run")
    if args.action == "verify-smoke" and args.report is None:
        parser.error("verify-smoke requires --report")
    if args.action == "preflight" and (args.report is None or args.efficacy_spec is None):
        parser.error("preflight requires --report and an approved --efficacy-spec")
    if args.micro_batch != FULL_MICRO_BATCH:
        parser.error(f"prompt-granularity is pinned to --micro-batch {FULL_MICRO_BATCH}")
    if not args.checkpoint_backbone:
        parser.error("prompt-granularity requires --checkpoint-backbone")
    os.environ["PROMPT_GRANULARITY_ENABLED_OVERRIDE"] = "1"
    os.environ["FULL_FT_RUN_NAME_OVERRIDE"] = RUN_NAME
    os.environ["TRAIN_BATCH_SIZE_OVERRIDE"] = str(args.micro_batch)
    os.environ["FULL_FT_NUM_TRAIN_WORKERS_OVERRIDE"] = str(FULL_TRAIN_WORKERS)
    os.environ["FULL_FT_NUM_VAL_WORKERS_OVERRIDE"] = str(FULL_VAL_WORKERS)
    os.environ["ACT_CKPT_VISION_BACKBONE_OVERRIDE"] = "1" if args.checkpoint_backbone else "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    from device_policy import shared_node_env
    os.environ.update(shared_node_env(1))
    if args.action == "verify-smoke":
        contract = training_contract(args.data_root)
        report = require_smoke(args.report, contract)
        print("PROMPT_GRANULARITY_SMOKE_REPORT_VALID", digest(report))
        return
    if args.action == "preflight":
        run_manifest = validate_launch(args.data_root, args.report, args.efficacy_spec, resume_from=args.resume_from, run_dir=args.run_dir)
        print("PROMPT_GRANULARITY_PREFLIGHT_PASSED", digest(run_manifest))
        return
    from prompt_granularity_training import run_smoke
    try:
        run_smoke(args)
    finally:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
