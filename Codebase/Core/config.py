"""Central training configuration for the SAM3 and Mask R-CNN fine-tunes on OAM-TCD.

Every tunable setting is a constant near the top of this module. The builders further down turn
those constants into the nested configuration that the SAM3 trainer instantiates through Hydra, so
a run's recipe is read from one place. Environment variables ending in `_OVERRIDE` select the arm,
run name, batch size, and worker counts; the recipe constants themselves change only by editing
this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from device_policy import (
    cap_batch_size,
    cap_dataloader_workers,
    preserve_effective_batch,
    visible_gpu_count,
)
from project_paths import CHECKPOINT_DIR, CODEBASE_DIR, EXPERIMENTS_DIR
from prompt_granularity import ARM as PG_ARM, RUN_NAME as PG_RUN_NAME, VIEW_FILES as PG_VIEW_FILES


# =============================================================================
# Model backend selector.
#
# Set MODEL_BACKEND to "sam3" or "maskrcnn" to choose which model to train.
# The training pipeline dispatches to the appropriate trainer based on this.
# =============================================================================

MODEL_BACKEND = "sam3"


# ── HPC environment modules ─────────────────────────────────────────────────
# Modules to load before the training subprocess on HPC systems. Only used
# when launching from notebooks/workflows that do not inherit a login-shell
# environment. Set to empty list if modules are not needed or not available.
HPC_LOAD_MODULES = ["gcc/gcc-11"]


@dataclass(frozen=True)
class TrainingPreset:
    """Resolved launch settings for one run: data root, run directory, ID limits, epochs, workers, and validation cadence."""
    name: str
    data_root: Path
    experiment_log_dir: Path
    train_limit_ids: int | None
    val_limit_ids: int | None
    max_data_epochs: int
    trainer_mode: str
    num_train_workers: int
    num_val_workers: int
    val_epoch_freq: int


# =============================================================================
# Centralized SAM3 OAM-TCD configuration.
#
# All tunable parameters live in this top-of-file block. Helper builders below
# (_scratch_config, _loss_config, _trainer_config, _train_transforms, ...)
# reference these constants rather than hard-coding values. To change a setting
# for a SAM3 fine-tune run, edit the corresponding constant here.
# =============================================================================

# ── Task / data shape ────────────────────────────────────────────────────────
PROMPT = "tree"
PROMPT_GRANULARITY_ENABLED = os.environ.get("PROMPT_GRANULARITY_ENABLED_OVERRIDE", "0").strip().lower() in ("1", "true", "yes")
RESOLUTION = 1008          # SAM3 image builder is fixed at 1008x1008
MAX_ANN_PER_IMG = 1000     # cap on instances per tile

# Annotation filenames under <data_root>/annotations/.
# TRAIN: full 4169-image OAM-TCD train set (matches the dataset paper).
# VAL: the OAM-TCD TEST set (439 images), used for in-training monitoring.
#   Final reporting uses the last epoch, not the auxiliary best checkpoint.
#   There is no VAL carve in the locked A0 recipe or its paired comparison.
#   TEST has informed development, so last-epoch selection does not make it
#   an untouched holdout. Prompt-granularity must preserve image membership;
#   its category-preserving annotation view is a separate, integrity-checked file.
TRAIN_ANN_FILENAME = PG_VIEW_FILES["train"] if PROMPT_GRANULARITY_ENABLED else "train_annotations.coco.json"
VAL_ANN_FILENAME = PG_VIEW_FILES["test"] if PROMPT_GRANULARITY_ENABLED else "test_annotations.coco.json"
TEST_ANN_FILENAME = "test_annotations.coco.json"

# ── Preset: full_ft ──────────────────────────────────────────────────────────
# Per-run scheduling, cluster, and dataloader controls. The launcher binds a registered arm
# to its run name through an environment override, while completed-run guards protect existing
# experiment directories. The committed default remains the A0 baseline.
FULL_FT_RUN_NAME = os.environ.get("FULL_FT_RUN_NAME_OVERRIDE", PG_RUN_NAME if PROMPT_GRANULARITY_ENABLED else "sam3_crop_frts")
if PROMPT_GRANULARITY_ENABLED and FULL_FT_RUN_NAME != PG_RUN_NAME:
    raise ValueError(f"Prompt-granularity must use run name {PG_RUN_NAME!r}, not {FULL_FT_RUN_NAME!r}.")

# Prompt granularity is the only registered mechanism arm. Registration is not activation;
# the launcher supplies the explicit environment switch and validates the run directory.
ARM_RUN_NAMES: dict[str, str] = {PG_ARM: PG_RUN_NAME}

FULL_FT_MAX_DATA_EPOCHS = 30
FULL_FT_TRAINER_MODE = "train"
# Worker counts are device-aware throughput controls. The seeded sampler runs in the main
# process, so worker count does not change batch composition. Validation uses the tighter cap
# because whole-image samples retain more instances than cropped training samples. Every run
# records the resolved values in its configuration snapshot.
FULL_FT_NUM_TRAIN_WORKERS = int(os.environ.get("FULL_FT_NUM_TRAIN_WORKERS_OVERRIDE", "0")) \
    or cap_dataloader_workers(12, cap=4)
FULL_FT_NUM_VAL_WORKERS = int(os.environ.get("FULL_FT_NUM_VAL_WORKERS_OVERRIDE", "0")) \
    or cap_dataloader_workers(12, cap=2)
FULL_FT_VAL_EPOCH_FREQ = 3
FULL_FT_TRAIN_LIMIT_IDS: int | None = None
FULL_FT_VAL_LIMIT_IDS: int | None = None

# ── Launcher (single-node interactive) ───────────────────────────────────────
LAUNCHER_MULTIPROCESSING_CONTEXT = "forkserver"
LAUNCHER_PORT_RANGE = [10000, 65000]

# ── Model / scratch ──────────────────────────────────────────────────────────
ENABLE_SEGMENTATION = True
D_MODEL = 256
POS_EMBED_NUM_POS_FEATS = 256
POS_EMBED_TEMPERATURE = 10000
POS_EMBED_NORMALIZE = True

# Normalisation (train + val use SAM3 default 0.5/0.5/0.5)
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

# Sampling / context
HYBRID_REPEATS = 1
CONTEXT_LENGTH = 2
MAX_TRAIN_QUERIES = 10
MAX_VAL_QUERIES = 10

# Batch sizes. RECIPE_BATCH_SIZE is the locked recipe (4.1) and is what the A40
# runs use. Device-aware: 8 does not fit a 32 GiB V100 for full fine-tuning, so
# pre-Ampere nodes cap the *microbatch* at 1 -- the floor -- via device_policy.
# Validation is the binding constraint: it peaks at 27-28 GiB of 31.7 GiB
# (against 21-23 GiB for training) because _val_transforms only resizes, so a val
# sample carries all of a 2048 px tile's instances while a train sample is a
# ~1024 px crop carrying ~a quarter of them. Batch size is a weak lever against
# that -- see device_policy.cap_batch_size for the measured scaling -- so this is
# bought for the last GiB or two of val headroom, at wall-clock cost. Val batch is
# numerics-free (its meters aggregate per-image over the whole set).
RECIPE_BATCH_SIZE = 8

# Microbatch size is overridable per launch while accumulation preserves the effective image
# batch of 16. This lowers activation memory at additional wall-clock cost without changing
# the configured optimisation batch.
_MICRO_BATCH = int(os.environ.get("TRAIN_BATCH_SIZE_OVERRIDE", "0")) or RECIPE_BATCH_SIZE
TRAIN_BATCH_SIZE = cap_batch_size(_MICRO_BATCH)
VAL_BATCH_SIZE = cap_batch_size(_MICRO_BATCH)
# ...and accumulation buys the recipe's effective batch back, so the capped
# microbatch costs memory and wall clock only, not optimisation quality:
# 1 x 8 accum x 2 ranks == 8 x 1 accum x 2 ranks == effective 16 either way, and
# 260 optimizer steps per epoch either way. Only TRAIN_BATCH_SIZE images are ever
# GPU-resident (Trainer._step copies per microbatch), and the per-microbatch loss
# is normalised in progress_trainer._accumulation_divisor so the summed gradient
# matches one full batch rather than overshooting. A40 keeps accumulation at 1.
# See device_policy.preserve_effective_batch.
GRADIENT_ACCUMULATION_STEPS = preserve_effective_batch(
    1, recipe_batch=RECIPE_BATCH_SIZE, actual_batch=TRAIN_BATCH_SIZE
)

# ── Optimisation ─────────────────────────────────────────────────────────────
LR_SCALE = 0.1
LR_TRANSFORMER = 8e-5
LR_VISION_BACKBONE = 2.5e-5
LR_LANGUAGE_BACKBONE = 5e-6
LRD_VISION_BACKBONE = 0.95
WEIGHT_DECAY = 0.1

# Cosine annealing with linear warmup. Both expressed as fractions of total
# training (where in [0, 1]), so the schedule is robust to changes in epoch
# count or steps-per-epoch.
#   WARMUP_FRACTION       = 0.05  → linear ramp 0 → base_lr over first ~5% of training (~1.5 epochs of 30)
#   COSINE_END_LR_RATIO   = 0.01  → cosine bottoms out at 1% of base_lr
WARMUP_FRACTION = 0.05
COSINE_END_LR_RATIO = 0.01

AMP_ENABLED = True
AMP_DTYPE = "bfloat16"
# Stable full fine-tuning gradient clipping for the SAM3 OAM-TCD setup.
GRAD_CLIP_MAX_NORM = 0.1
GRAD_CLIP_NORM_TYPE = 2

# ── Trainer / DDP ────────────────────────────────────────────────────────────
TRAINER_TARGET = "progress_trainer.ProgressBarTrainer"
TRAINER_ACCELERATOR = "cuda"
TRAINER_SEED = 42
TRAINER_SKIP_FIRST_VAL = True
TRAINER_SKIP_SAVING_CKPTS = False
TRAINER_EMPTY_GPU_MEM_CACHE_AFTER_EVAL = True

DDP_BACKEND = "nccl"
DDP_FIND_UNUSED_PARAMETERS = False
DDP_GRADIENT_AS_BUCKET_VIEW = False
DDP_STATIC_GRAPH = True

# ── Matcher (shared by scratch + loss) ───────────────────────────────────────
MATCHER_FOCAL = True
MATCHER_COST_CLASS = 2.0
MATCHER_COST_BBOX = 5.0
MATCHER_COST_GIOU = 2.0
MATCHER_ALPHA = 0.25
MATCHER_GAMMA = 2
MATCHER_STABLE = False

# One-to-many auxiliary matcher
O2M_WEIGHT = 2.0
O2M_ALPHA = 0.3
O2M_THRESHOLD = 0.4
O2M_TOPK = 4
USE_O2M_MATCHER_ON_O2M_AUX = False

# ── Loss weights ─────────────────────────────────────────────────────────────
# SAM3 full fine-tuning loss balance for dense tree-crown detection and masks.
LOSS_BBOX_WEIGHT = 5.0
LOSS_GIOU_WEIGHT = 2.0
LOSS_CE_WEIGHT = 20.0
LOSS_PRESENCE_WEIGHT = 20.0
LOSS_MASK_WEIGHT = 200.0
LOSS_DICE_WEIGHT = 10.0
LOSS_SEMANTIC_SEG_WEIGHT = 20.0
SEMANTIC_SEG_ENABLED = True      # Master on/off for the FRTS / semantic-seg auxiliary loss
SEMANTIC_SEG_FOCAL = False
SEMANTIC_SEG_DOWNSAMPLE = False  # False => FRTS full-res, True => train-time downsampling (only used when SEMANTIC_SEG_ENABLED)

IABCE_POS_WEIGHT = 10.0
IABCE_ALPHA = 0.25
IABCE_GAMMA = 2
IABCE_USE_PRESENCE = True
IABCE_POS_FOCAL = False
IABCE_PAD_N_QUERIES = 200
IABCE_PAD_SCALE_POS = 1.0
IABCE_WEAK_LOSS = False

MASK_FOCAL_ALPHA = 0.25
MASK_FOCAL_GAMMA = 2.0
MASK_LOSS_ENABLED = True
MASK_LOSS_MODE = "sampled"
MASK_COMPUTE_AUX = False
MASK_NUM_SAMPLE_POINTS = 12544
MASK_OVERSAMPLE_RATIO = 3.0
MASK_IMPORTANCE_SAMPLE_RATIO = 0.75

# ── Boundary-band metric (Claim 1) ──────────────────────────────────────────
# Instance-relative Boundary AP uses band = k * equivalent_diameter. The value was fixed at
# 0.19 before the final model comparison. Matched final individual-tree evaluations keep every
# size-bin retention ratio inside the declared usable window for SAM3 and Mask R-CNN. Exact
# preliminary half-retention estimates are not public claims; Core/boundary_band.py is authoritative.
BOUNDARY_BAND_K = 0.19

# The literature/8px comparison rows reported alongside the instance-relative band.
BOUNDARY_FIXED_BAND_PX = 8.0


# Whole-vision-backbone activation checkpointing performs a numerically exact recomputation
# with preserved RNG state. It lowers activation memory at roughly 30 percent additional step
# time and is enabled by the launcher when required.
ACT_CKPT_VISION_BACKBONE = os.environ.get("ACT_CKPT_VISION_BACKBONE_OVERRIDE", "").strip().lower() in ("1", "true", "yes")

# ── Train-time augmentation transforms ───────────────────────────────────────
BOX_NOISE_STD = 0.1
BOX_NOISE_MAX = 20

# B2' tiled-crop recipe. OAM-TCD feeds full 2048-px tiles; RandomSizeCropAPI
# extracts a ~1024-px window BEFORE the 1008 resize so small crowns are trained
# near native resolution instead of being downsampled ~2x (the diagnosed
# small-crown non-proposal limiter). Crop side is jittered in [896, 1152].
CROP_MIN_SIZE = 896
CROP_MAX_SIZE = 1152
# BOTH respect flags False => pure random crop (RandomSizeCropAPI else-branch,
# check_validity=False). respect_boxes=True forced near-full-image crops on
# dense tiles (defeating the resolution gain), so it is disabled; crowns cropped
# out are zeroed by crop() and removed downstream by FilterEmptyTargets.
CROP_RESPECT_BOXES = False
CROP_RESPECT_INPUT_BOXES = False

# Full D4 dihedral group via independent hflip(.5) + vflip(.5) + rot90(p=.75,
# k in {1,2,3}); these compose to a uniform draw over D4's 8 orientations.
# vflip/rot90 are project-owned (Core/augmentation.py) to keep Support/sam3
# pristine and mirror the shipped hflip datapoint contract exactly.
HFLIP_P = 0.5
VFLIP_P = 0.5
ROT90_P = 0.75

# Photometric jitter, probability-gated through RandomSelectAPI (shipped
# ColorJitter has no internal probability). Applied on the uint8 PIL crop
# BEFORE ToTensor/Normalize. Values kept in lock-step with Core/augmentation.py.
COLORJITTER_P = 0.5
COLORJITTER_BRIGHTNESS = 0.25
COLORJITTER_CONTRAST = 0.25
COLORJITTER_SATURATION = 0.25
COLORJITTER_HUE = 0.1

# Resize-jitter floor raised toward RESOLUTION (was 480) so the post-crop resize
# KEEPS the recovered resolution instead of shrinking the ~1024 crop back to
# ~480. With size=1008, min_size=896, get_random_resize_scales yields scales
# [896, 928, 960, 992] (stride 32); net crown scale ~= [0.78, 1.10] vs baseline
# ~0.23-0.49. Tunable: raise to 992 for crop-only jitter (single resize scale).
RESIZE_MIN_SIZE = 896
RESIZE_ROUNDED = False
RESIZE_SQUARE = True
CONSISTENT_TRANSFORM = False

# ── Eval / postprocessing ────────────────────────────────────────────────────
EVAL_USE_PRESENCE = True
EVAL_MAX_DETS_PER_IMG = -1     # -1 => no cap
EVAL_MAXDETS = 512
# Postprocessor score threshold for prediction export. The training/evaluation
# dump keeps the full score range so `eval_predictions.run_f1_sweep` can
# evaluate the finalized operating point from the same predictions.
EVAL_DETECTION_THRESHOLD = -1.0
# In-training cgF1 score threshold. This affects metric computation only; the
# prediction dump remains controlled by EVAL_DETECTION_THRESHOLD.
EVAL_CGF1_SCORE_THRESHOLD = 0.315
# Locked score threshold for all prediction visualizations across inference
# configurations. Matches DEFAULT_CONFIDENCE used by the SAM3 processor during
# inference so visualizations reflect the model's actual operating point.
VIZ_SCORE_THRESHOLD = 0.5
EVAL_USE_ORIGINAL_IDS = True
EVAL_USE_ORIGINAL_SIZES_BOX = True
EVAL_USE_ORIGINAL_SIZES_MASK = True
EVAL_TIDE = False
EVAL_IOU_TYPE = "segm"
EVAL_MERGE_PREDICTIONS = True
GATHER_PRED_VIA_FILESYS = False

# ── Checkpointing / logging ──────────────────────────────────────────────────
# OAM-TCD full_ft: 4169 train images (full TRAIN set), batch_size=8,
# drop_last=True → ~521 iters/epoch. Validation: 439 TEST images, batch_size=8
# → ~55 iters/val.
# Target ~42 progress lines/train epoch and ~10 progress lines/val so the bar
# refreshes smoothly without flooding the log file. Final progress bar always
# reaches num_batches via TqdmProgressMeter.__del__ in progress_trainer.py.
CKPT_SAVE_FREQ = 0
PROGRESS_LOG_FREQ = 10    # 26 progress lines/train epoch (260 steps), ~3/val (27 steps)
TB_SCALAR_LOG_FREQ = 10   # 26 TB scalar points/train epoch, aligned with progress
LOG_FREQ = PROGRESS_LOG_FREQ
TB_FLUSH_SECS = 120
TB_SHOULD_LOG = True
TB_PROJECT_METRICS = {
    # Validation quality (order sets best-checkpoint priority: cgF1 -> AP -> loss)
    "Meters_train/val_oamtcd/detection/cgF1_eval_segm_cgF1": "val_cgF1",
    "Meters_train/val_oamtcd/detection/cgF1_eval_segm_IL_F1": "val_cgf1_IL_F1",
    "Meters_train/val_oamtcd/detection/cgF1_eval_segm_IL_precision": "val_cgf1_IL_precision",
    "Meters_train/val_oamtcd/detection/cgF1_eval_segm_IL_recall": "val_cgf1_IL_recall",
    "Meters_train/val_oamtcd/detection/cgF1_eval_segm_IL_MCC": "val_cgf1_IL_MCC",
    "Meters_train/val_oamtcd/detection/cgF1_eval_segm_precision": "val_cgF1_precision",
    "Meters_train/val_oamtcd/detection/cgF1_eval_segm_recall": "val_cgF1_recall",
    "Meters_train/val_oamtcd/detection/coco_eval_segm_AP": "val_segm_AP",
    "Meters_train/val_oamtcd/detection/coco_eval_segm_AP_50": "val_segm_AP50",
    "Meters_train/val_oamtcd/detection/coco_eval_segm_AP_75": "val_segm_AP75",
    "Meters_train/val_oamtcd/detection/coco_eval_segm_AR_maxDets@100": "val_segm_AR100",
    # Boundary AP is computed offline on the final tiled dump. In-training validation uses
    # whole-image resizing and therefore does not represent the reported inference regime.
    # The scored checkpoint is the last epoch, so no boundary metric selects a checkpoint.
    "Step_Losses/Losses/val_oamtcd_loss": "val_loss_step",
    "Losses/val_oamtcd_loss": "val_loss",
    "Losses/val_oamtcd_loss_semantic_seg": "val_loss_semantic_seg",
    # Training health
    "Step_Losses/Losses/train_all_loss": "train_loss",
    "Losses/train_all_loss": "train_loss_epoch",
    "Losses/train_all_loss_semantic_seg": "train_loss_semantic_seg",
    # `scalar_keys_to_log` is a whitelist. Surface per-term mask losses so training telemetry
    # confirms the configured objective without affecting optimisation.
    "Losses/train_all_loss_mask": "train_loss_mask",
    "Losses/train_all_loss_dice": "train_loss_dice",
    "Losses/train_all_loss_boundary": "train_loss_boundary",
    # Realized learning rate per param group (logged every TB_SCALAR_LOG_FREQ
    # iters by the vendored Trainer; surfaced here so it appears in TB and in
    # train_stats.json / latest_train_metrics.txt).
    "Optim/0_/lr": "lr_transformer",
    "Optim/1_/lr": "lr_vision_backbone",
    "Optim/2_/lr": "lr_language_backbone",
}

if PROMPT_GRANULARITY_ENABLED:
    TB_PROJECT_METRICS.update({
        f"Losses/{phase}_{key}_pg_{metric}": f"{phase}_pg_{metric}"
        for phase, key in (("train", "all"), ("val", "oamtcd"))
        for metric in ("pair_count", "image_count", "scale", "tree_targets", "canopy_targets")
    })

# ── Dataloader ───────────────────────────────────────────────────────────────
DATALOADER_PIN_MEMORY = True
DATALOADER_DROP_LAST_TRAIN = True
DATALOADER_DROP_LAST_VAL = False
DATALOADER_SHUFFLE_TRAIN = True
DATALOADER_SHUFFLE_VAL = False

# ── Scratch flags ────────────────────────────────────────────────────────────
SCALE_BY_FIND_BATCH_SIZE = True
USE_CACHING = False
LOAD_SEGMENTATION = True
WITH_SEG_MASKS = True
COLLATE_REPEATS = 1
DATASET_MULTIPLIER = 1
INCLUDE_NEGATIVES_VAL = True
VAL_CATEGORY_CHUNK_SIZE = 1


# =============================================================================
# Mask R-CNN R50 training constants (Detectron2).
#
# Used when MODEL_BACKEND = "maskrcnn". Values match the OAM-TCD dataset paper
# (Veitch-Michaelis et al. 2024) for reproducible comparison:
#   - 100k iters, batch 8, lr 0.001, step ×0.1 at 80k/90k
#   - COCO-pretrained R50-FPN, 1 foreground class (tree)
#   - DETECTIONS_PER_IMAGE=512, SCORE_THRESH_TEST=0.2, seed 42
# Augmentation combines Detectron2 multi-scale defaults with high-resolution
# absolute cropping suited to 2048×2048 OAM-TCD imagery with small crowns:
#   - Multi-scale resize: (800, 960, 1024, 1088, 1152) — biased toward higher
#     resolution to preserve small-crown detail while retaining scale diversity
#   - Absolute 1024×1024 crop: at lower resize scales this is a no-op (full
#     image); at higher scales it provides random spatial context windows
# =============================================================================

MASKRCNN_RUN_NAME = "maskrcnn_r50_oam_tcd"
MASKRCNN_TRAIN_ANN_FILENAME = "train_annotations.coco.json"
MASKRCNN_BASE_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
MASKRCNN_COCO_WEIGHTS = str(CHECKPOINT_DIR / "model_final_f10217.pkl")
MASKRCNN_NUM_CLASSES = 1              # tree only (background is implicit)
MASKRCNN_MIN_SIZE_TRAIN = (800, 960, 1024, 1088, 1152)
MASKRCNN_MAX_SIZE_TRAIN = 1333
MASKRCNN_CROP_SIZE = [1024, 1024]
MASKRCNN_IMS_PER_BATCH = 8
MASKRCNN_BASE_LR = 0.001
MASKRCNN_MAX_ITER = 100_000
MASKRCNN_STEPS = (80_000, 90_000)
MASKRCNN_GAMMA = 0.1
MASKRCNN_CHECKPOINT_PERIOD = 5_000
MASKRCNN_TEST_PERIOD = 5_000
MASKRCNN_SCORE_THRESH_TEST = 0.2
MASKRCNN_DETECTIONS_PER_IMAGE = 512
# Single test scale. Inference uses tiling without multi-scale test-time augmentation.
MASKRCNN_MIN_SIZE_TEST = 800
MASKRCNN_MAX_SIZE_TEST = 1333
MASKRCNN_SEED = 42
MASKRCNN_NUM_WORKERS = 4


def get_training_preset(
    name: str,
    data_root: Path | None = None,
    experiment_log_dir: Path | None = None,
) -> TrainingPreset:
    """Return the preset for `name` (only `full_ft` exists), with optional data-root and run-directory overrides."""
    if name == "full_ft":
        return TrainingPreset(
            name=name,
            data_root=data_root or CODEBASE_DIR / "runtime" / "data" / "oam_tcd_instance_coco",
            experiment_log_dir=experiment_log_dir or EXPERIMENTS_DIR / FULL_FT_RUN_NAME,
            train_limit_ids=FULL_FT_TRAIN_LIMIT_IDS,
            val_limit_ids=FULL_FT_VAL_LIMIT_IDS,
            max_data_epochs=FULL_FT_MAX_DATA_EPOCHS,
            trainer_mode=FULL_FT_TRAINER_MODE,
            num_train_workers=FULL_FT_NUM_TRAIN_WORKERS,
            num_val_workers=FULL_FT_NUM_VAL_WORKERS,
            val_epoch_freq=FULL_FT_VAL_EPOCH_FREQ,
        )
    raise ValueError(f"Unknown training preset: {name}. Supported preset: full_ft")


def build_sam3_train_config(preset: TrainingPreset) -> dict:
    """Build the nested SAM3 training configuration (paths, data, loss, trainer, optimiser, launcher) for `preset`."""
    data_root = str(preset.data_root)
    log_dir = str(preset.experiment_log_dir)
    train_ann_file = str(preset.data_root / "annotations" / TRAIN_ANN_FILENAME)
    val_ann_file = str(preset.data_root / "annotations" / VAL_ANN_FILENAME)

    scratch = _scratch_config(preset)
    return {
        "paths": {
            "coco_root": data_root,
            "experiment_log_dir": log_dir,
            "bpe_path": None,
            "train_ann_file": train_ann_file,
            "val_ann_file": val_ann_file,
        },
        "oam_tcd": {
            "train_limit_ids": preset.train_limit_ids,
            "val_limit_ids": preset.val_limit_ids,
            "train_transforms": _train_transforms(),
            "val_transforms": _val_transforms(),
            "loss": _loss_config(),
        },
        "scratch": scratch,
        "trainer": _trainer_config(preset, scratch),
        "launcher": {
            "num_nodes": 1,
            # Follow visible devices so process count cannot exceed CUDA visibility. Reported
            # two-rank arms separately require two devices before launch; every run records the
            # resolved value in its configuration snapshot.
            "gpus_per_node": max(1, visible_gpu_count()),
            "experiment_log_dir": log_dir,
            "multiprocessing_context": LAUNCHER_MULTIPROCESSING_CONTEXT,
            "port_range": list(LAUNCHER_PORT_RANGE),
        },
    }


def _scratch_config(preset: TrainingPreset) -> dict:
    return {
        "enable_segmentation": ENABLE_SEGMENTATION,
        "d_model": D_MODEL,
        "pos_embed": {
            "_target_": "sam3.model.position_encoding.PositionEmbeddingSine",
            "num_pos_feats": POS_EMBED_NUM_POS_FEATS,
            "normalize": POS_EMBED_NORMALIZE,
            "scale": None,
            "temperature": POS_EMBED_TEMPERATURE,
        },
        "use_presence_eval": EVAL_USE_PRESENCE,
        "original_box_postprocessor": _postprocessor_config(),
        "matcher": _matcher_config(),
        "scale_by_find_batch_size": SCALE_BY_FIND_BATCH_SIZE,
        "resolution": RESOLUTION,
        "consistent_transform": CONSISTENT_TRANSFORM,
        "max_ann_per_img": MAX_ANN_PER_IMG,
        "train_norm_mean": list(NORM_MEAN),
        "train_norm_std": list(NORM_STD),
        "val_norm_mean": list(NORM_MEAN),
        "val_norm_std": list(NORM_STD),
        "num_train_workers": preset.num_train_workers,
        "num_val_workers": preset.num_val_workers,
        "max_data_epochs": preset.max_data_epochs,
        "hybrid_repeats": HYBRID_REPEATS,
        "context_length": CONTEXT_LENGTH,
        "gather_pred_via_filesys": GATHER_PRED_VIA_FILESYS,
        "lr_scale": LR_SCALE,
        "lr_transformer": LR_TRANSFORMER,
        "lr_vision_backbone": LR_VISION_BACKBONE,
        "lr_language_backbone": LR_LANGUAGE_BACKBONE,
        "lrd_vision_backbone": LRD_VISION_BACKBONE,
        "wd": WEIGHT_DECAY,
        "warmup_fraction": WARMUP_FRACTION,
        "cosine_end_lr_ratio": COSINE_END_LR_RATIO,
        "val_batch_size": VAL_BATCH_SIZE,
        "collate_fn_val": _collate_fn(dict_key="oamtcd"),
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "collate_fn": _collate_fn(dict_key="all"),
    }


def _matcher_config() -> dict:
    return {
        "_target_": "sam3.train.matcher.BinaryHungarianMatcherV2",
        "focal": MATCHER_FOCAL,
        "cost_class": MATCHER_COST_CLASS,
        "cost_bbox": MATCHER_COST_BBOX,
        "cost_giou": MATCHER_COST_GIOU,
        "alpha": MATCHER_ALPHA,
        "gamma": MATCHER_GAMMA,
        "stable": MATCHER_STABLE,
    }


def _postprocessor_config() -> dict:
    return {
        "_target_": "eval_overrides.ProjectPostProcessImage",
        "max_dets_per_img": EVAL_MAX_DETS_PER_IMG,
        "iou_type": EVAL_IOU_TYPE,
        "use_original_ids": EVAL_USE_ORIGINAL_IDS,
        "use_original_sizes_box": EVAL_USE_ORIGINAL_SIZES_BOX,
        "use_original_sizes_mask": EVAL_USE_ORIGINAL_SIZES_MASK,
        "use_presence": EVAL_USE_PRESENCE,
        "detection_threshold": EVAL_DETECTION_THRESHOLD,
    }


def _collate_fn(dict_key: str) -> dict:
    collate = {
        "_target_": "sam3.train.data.collator.collate_fn_api",
        "_partial_": True,
        "repeats": COLLATE_REPEATS,
        "dict_key": dict_key,
        "with_seg_masks": WITH_SEG_MASKS,
    }
    if PROMPT_GRANULARITY_ENABLED:
        collate.update(
            _target_="prompt_granularity_training.collate_fn_prompt_granularity",
            queries_per_image=2 if dict_key == "all" else VAL_CATEGORY_CHUNK_SIZE,
        )
    return collate


def _train_transforms() -> list[dict]:
    return [
        {
            "_target_": "sam3.train.transforms.basic_for_api.ComposeAPI",
            "transforms": [
                _query_filter("sam3.train.transforms.filter_query_transforms.FilterCrowds"),
                {
                    "_target_": "sam3.train.transforms.point_sampling.RandomizeInputBbox",
                    "box_noise_std": BOX_NOISE_STD,
                    "box_noise_max": BOX_NOISE_MAX,
                },
                {"_target_": "sam3.train.transforms.segmentation.DecodeRle"},
                # ── B2' tiled-crop + D4 + photometric augmentation ──────────
                # Order: crop a ~1024 window from the 2048 tile, apply the D4
                # dihedral group, then photometric jitter -- all on the uint8
                # PIL crop BEFORE the 1008 resize, so small crowns are seen near
                # native resolution. See the augmentation constants block above.
                {
                    "_target_": "sam3.train.transforms.basic_for_api.RandomSizeCropAPI",
                    "min_size": CROP_MIN_SIZE,
                    "max_size": CROP_MAX_SIZE,
                    "respect_boxes": CROP_RESPECT_BOXES,
                    "respect_input_boxes": CROP_RESPECT_INPUT_BOXES,
                    "consistent_transform": CONSISTENT_TRANSFORM,
                },
                {
                    "_target_": "sam3.train.transforms.basic_for_api.RandomHorizontalFlip",
                    "p": HFLIP_P,
                    "consistent_transform": CONSISTENT_TRANSFORM,
                },
                {
                    "_target_": "augmentation.RandomVerticalFlip",
                    "p": VFLIP_P,
                    "consistent_transform": CONSISTENT_TRANSFORM,
                },
                {
                    "_target_": "augmentation.RandomRot90",
                    "p": ROT90_P,
                    "consistent_transform": CONSISTENT_TRANSFORM,
                },
                {
                    "_target_": "sam3.train.transforms.basic_for_api.RandomSelectAPI",
                    "p": COLORJITTER_P,
                    "transforms1": {
                        "_target_": "sam3.train.transforms.basic_for_api.ColorJitter",
                        "consistent_transform": CONSISTENT_TRANSFORM,
                        "brightness": COLORJITTER_BRIGHTNESS,
                        "contrast": COLORJITTER_CONTRAST,
                        "saturation": COLORJITTER_SATURATION,
                        "hue": COLORJITTER_HUE,
                    },
                },
                {
                    "_target_": "sam3.train.transforms.basic_for_api.RandomResizeAPI",
                    "sizes": {
                        "_target_": "sam3.train.transforms.basic.get_random_resize_scales",
                        "size": RESOLUTION,
                        "min_size": RESIZE_MIN_SIZE,
                        "rounded": RESIZE_ROUNDED,
                    },
                    "max_size": {
                        "_target_": "sam3.train.transforms.basic.get_random_resize_max_size",
                        "size": RESOLUTION,
                    },
                    "square": RESIZE_SQUARE,
                    "consistent_transform": CONSISTENT_TRANSFORM,
                },
                {
                    "_target_": "sam3.train.transforms.basic_for_api.PadToSizeAPI",
                    "size": RESOLUTION,
                    "consistent_transform": CONSISTENT_TRANSFORM,
                },
                {"_target_": "sam3.train.transforms.basic_for_api.ToTensorAPI"},
                _query_filter("sam3.train.transforms.filter_query_transforms.FilterEmptyTargets"),
                {
                    "_target_": "sam3.train.transforms.basic_for_api.NormalizeAPI",
                    "mean": list(NORM_MEAN),
                    "std": list(NORM_STD),
                },
                _query_filter("sam3.train.transforms.filter_query_transforms.FilterEmptyTargets"),
            ],
        },
        {
            "_target_": "sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries",
            "query_filter": {
                "_target_": "sam3.train.transforms.filter_query_transforms.FilterFindQueriesWithTooManyOut",
                "max_num_objects": MAX_ANN_PER_IMG,
            },
        },
    ]


def _val_transforms() -> list[dict]:
    return [
        {
            "_target_": "sam3.train.transforms.basic_for_api.ComposeAPI",
            "transforms": [
                {"_target_": "sam3.train.transforms.segmentation.DecodeRle"},
                {
                    "_target_": "sam3.train.transforms.basic_for_api.RandomResizeAPI",
                    "sizes": RESOLUTION,
                    "max_size": {
                        "_target_": "sam3.train.transforms.basic.get_random_resize_max_size",
                        "size": RESOLUTION,
                    },
                    "square": RESIZE_SQUARE,
                    "consistent_transform": CONSISTENT_TRANSFORM,
                },
                {"_target_": "sam3.train.transforms.basic_for_api.ToTensorAPI"},
                {
                    "_target_": "sam3.train.transforms.basic_for_api.NormalizeAPI",
                    "mean": list(NORM_MEAN),
                    "std": list(NORM_STD),
                },
            ],
        }
    ]


def _query_filter(target: str) -> dict:
    return {
        "_target_": "sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries",
        "query_filter": {"_target_": target},
    }


def _loss_config(*, prompt_queries_per_image: int = 2) -> dict:
    loss_fns_find = [
        {
            "_target_": "sam3.train.loss.loss_fns.Boxes",
            "weight_dict": {"loss_bbox": LOSS_BBOX_WEIGHT, "loss_giou": LOSS_GIOU_WEIGHT},
        },
        {
            "_target_": "sam3.train.loss.loss_fns.IABCEMdetr",
            "weak_loss": IABCE_WEAK_LOSS,
            "weight_dict": {"loss_ce": LOSS_CE_WEIGHT, "presence_loss": LOSS_PRESENCE_WEIGHT},
            "pos_weight": IABCE_POS_WEIGHT,
            "alpha": IABCE_ALPHA,
            "gamma": IABCE_GAMMA,
            "use_presence": IABCE_USE_PRESENCE,
            "pos_focal": IABCE_POS_FOCAL,
            "pad_n_queries": IABCE_PAD_N_QUERIES,
            "pad_scale_pos": IABCE_PAD_SCALE_POS,
        },
    ]
    if MASK_LOSS_ENABLED:
        mask_loss_config = {
            "_target_": "eval_overrides.ProjectSampledMasks",
            "focal_alpha": MASK_FOCAL_ALPHA,
            "focal_gamma": MASK_FOCAL_GAMMA,
            "weight_dict": {"loss_mask": LOSS_MASK_WEIGHT, "loss_dice": LOSS_DICE_WEIGHT},
            "compute_aux": MASK_COMPUTE_AUX,
        }
        if MASK_LOSS_MODE == "sampled":
            mask_loss_config.update(
                {
                    "num_sample_points": MASK_NUM_SAMPLE_POINTS,
                    "oversample_ratio": MASK_OVERSAMPLE_RATIO,
                    "importance_sample_ratio": MASK_IMPORTANCE_SAMPLE_RATIO,
                }
            )
        elif MASK_LOSS_MODE != "full":
            raise ValueError(f"Unsupported MASK_LOSS_MODE={MASK_LOSS_MODE!r}; expected 'sampled' or 'full'.")
        loss_fns_find.append(mask_loss_config)
    semantic_seg_loss = (
        {
            "_target_": "eval_overrides.ProjectSemanticSegCriterion",
            "weight_dict": {"loss_semantic_seg": LOSS_SEMANTIC_SEG_WEIGHT},
            "focal": SEMANTIC_SEG_FOCAL,
            "downsample": SEMANTIC_SEG_DOWNSAMPLE,
        }
        if SEMANTIC_SEG_ENABLED
        else None
    )
    loss_config = {
        "_target_": "sam3.train.loss.sam3_loss.Sam3LossWrapper",
        "matcher": _matcher_config(),
        "o2m_weight": O2M_WEIGHT,
        "o2m_matcher": {
            "_target_": "sam3.train.matcher.BinaryOneToManyMatcher",
            "alpha": O2M_ALPHA,
            "threshold": O2M_THRESHOLD,
            "topk": O2M_TOPK,
        },
        "use_o2m_matcher_on_o2m_aux": USE_O2M_MATCHER_ON_O2M_AUX,
        "loss_fns_find": loss_fns_find,
        "loss_fn_semantic_seg": semantic_seg_loss,
        "scale_by_find_batch_size": SCALE_BY_FIND_BATCH_SIZE,
    }
    if PROMPT_GRANULARITY_ENABLED:
        loss_config.update(_target_="prompt_granularity_training.PromptGranularityLoss", queries_per_image=prompt_queries_per_image)
    return loss_config


def _trainer_config(preset: TrainingPreset, scratch: dict) -> dict:
    return {
        "_target_": TRAINER_TARGET,
        "skip_saving_ckpts": TRAINER_SKIP_SAVING_CKPTS,
        "empty_gpu_mem_cache_after_eval": TRAINER_EMPTY_GPU_MEM_CACHE_AFTER_EVAL,
        "skip_first_val": TRAINER_SKIP_FIRST_VAL,
        "max_epochs": preset.max_data_epochs,
        "accelerator": TRAINER_ACCELERATOR,
        "seed_value": TRAINER_SEED,
        "val_epoch_freq": preset.val_epoch_freq,
        "mode": preset.trainer_mode,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "distributed": {
            "backend": DDP_BACKEND,
            "find_unused_parameters": DDP_FIND_UNUSED_PARAMETERS,
            "gradient_as_bucket_view": DDP_GRADIENT_AS_BUCKET_VIEW,
            "static_graph": DDP_STATIC_GRAPH,
        },
        "loss": {
            "all": _loss_config(),
            "oamtcd": _loss_config(prompt_queries_per_image=VAL_CATEGORY_CHUNK_SIZE),
            "default": {"_target_": "sam3.train.loss.sam3_loss.DummyLoss"},
        },
        "data": _data_config(preset, scratch),
        "model": {
            "_target_": "training_pipeline.build_finetuned_sam3_image_model",
            "bpe_path": None,
            "checkpoint_path": str(CHECKPOINT_DIR / "sam3" / "sam3.pt"),
            "load_from_HF": False,
            "device": "cpus",
            "eval_mode": False,
            "enable_segmentation": ENABLE_SEGMENTATION,
        },
        "meters": _meters_config(preset),
        "optim": _optim_config(scratch),
        "checkpoint": {
            "save_dir": str(preset.experiment_log_dir / "checkpoints"),
            "save_freq": CKPT_SAVE_FREQ,
        },
        "logging": {
            "tensorboard_writer": {
                "_target_": "sam3.train.utils.logger.make_tensorboard_logger",
                "log_dir": str(preset.experiment_log_dir / "tensorboard"),
                "flush_secs": TB_FLUSH_SECS,
                "should_log": TB_SHOULD_LOG,
            },
            "wandb_writer": None,
            "log_dir": str(preset.experiment_log_dir / "logs" / "oam_tcd"),
            "log_freq": LOG_FREQ,
            "log_scalar_frequency": TB_SCALAR_LOG_FREQ,
            "scalar_keys_to_log": dict(TB_PROJECT_METRICS),
        },
    }


def _data_config(preset: TrainingPreset, scratch: dict) -> dict:
    return {
        "train": {
            "_target_": "project_torch_dataset.ProjectTorchDataset",
            "dataset": {
                "_target_": "sam3.train.data.sam3_image_dataset.Sam3ImageDataset",
                "limit_ids": preset.train_limit_ids,
                "transforms": _train_transforms(),
                "load_segmentation": LOAD_SEGMENTATION,
                "max_ann_per_img": MAX_ANN_PER_IMG,
                "multiplier": DATASET_MULTIPLIER,
                "max_train_queries": MAX_TRAIN_QUERIES,
                "max_val_queries": MAX_VAL_QUERIES,
                "training": True,
                "use_caching": USE_CACHING,
                "img_folder": str(preset.data_root),
                "ann_file": str(preset.data_root / "annotations" / TRAIN_ANN_FILENAME),
                **({"coco_json_loader": {
                    "_target_": "sam3.train.data.coco_json_loaders.COCO_FROM_JSON",
                    "include_negatives": True, "category_chunk_size": 2, "_partial_": True,
                }} if PROMPT_GRANULARITY_ENABLED else {}),
            },
            "shuffle": DATALOADER_SHUFFLE_TRAIN,
            "batch_size": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "num_workers": scratch["num_train_workers"],
            "pin_memory": DATALOADER_PIN_MEMORY,
            "drop_last": DATALOADER_DROP_LAST_TRAIN,
            "collate_fn": scratch["collate_fn"],
        },
        "val": {
            "_target_": "sam3.train.data.torch_dataset.TorchDataset",
            "dataset": {
                "_target_": "sam3.train.data.sam3_image_dataset.Sam3ImageDataset",
                "limit_ids": preset.val_limit_ids,
                "load_segmentation": LOAD_SEGMENTATION,
                "coco_json_loader": {
                    "_target_": "sam3.train.data.coco_json_loaders.COCO_FROM_JSON",
                    "include_negatives": INCLUDE_NEGATIVES_VAL,
                    "category_chunk_size": VAL_CATEGORY_CHUNK_SIZE,
                    "_partial_": True,
                },
                "img_folder": str(preset.data_root),
                "ann_file": str(preset.data_root / "annotations" / VAL_ANN_FILENAME),
                "transforms": _val_transforms(),
                "max_ann_per_img": MAX_ANN_PER_IMG,
                "multiplier": DATASET_MULTIPLIER,
                "max_train_queries": MAX_TRAIN_QUERIES,
                "max_val_queries": MAX_VAL_QUERIES,
                "training": False,
            },
            "shuffle": DATALOADER_SHUFFLE_VAL,
            "batch_size": VAL_BATCH_SIZE,
            "num_workers": scratch["num_val_workers"],
            "pin_memory": DATALOADER_PIN_MEMORY,
            "drop_last": DATALOADER_DROP_LAST_VAL,
            "collate_fn": scratch["collate_fn_val"],
        },
    }


def _meters_config(preset: TrainingPreset) -> dict:
    return {
        "val": {
            "oamtcd": {
                "detection": {
                    "_target_": "eval_overrides.ProjectPredictionDumper",
                    "iou_type": EVAL_IOU_TYPE,
                    "dump_dir": str(preset.experiment_log_dir / "predictions" / "oam_tcd"),
                    "merge_predictions": EVAL_MERGE_PREDICTIONS,
                    "postprocessor": _postprocessor_config(),
                    "gather_pred_via_filesys": GATHER_PRED_VIA_FILESYS,
                    "maxdets": EVAL_MAXDETS,
                    "pred_file_evaluators": [
                        {
                            "_target_": "sam3.eval.coco_eval_offline.CocoEvaluatorOfflineWithPredFileEvaluators",
                            "gt_path": str(preset.data_root / "annotations" / VAL_ANN_FILENAME),
                            "tide": EVAL_TIDE,
                            "iou_type": EVAL_IOU_TYPE,
                        },
                        {
                            "_target_": "sam3.eval.cgf1_eval.CGF1Evaluator",
                            "gt_path": str(preset.data_root / "annotations" / VAL_ANN_FILENAME),
                            "iou_type": EVAL_IOU_TYPE,
                        },
                        # Boundary evaluation is reserved for the final tiled prediction dump.
                    ],
                }
            }
        }
    }


def _resolved_amp_dtype() -> str:
    """AMP_DTYPE if the GPU supports it, otherwise the best it does support.

    AMP_DTYPE is the *recipe* (bfloat16 on 2 x A40) and stays as written. This
    resolves what can actually run: the V100 node has no bf16, so a literal
    "bfloat16" there would either raise or silently degrade. Any fallback is
    logged once by device_policy, and `training_pipeline` refuses to start a
    reported run on a device that cannot honour the recipe unless
    --allow-precision-fallback is passed explicitly.
    """
    try:
        from device_policy import resolve_autocast_dtype

        resolved = resolve_autocast_dtype(AMP_DTYPE)
    except Exception:
        return AMP_DTYPE
    if resolved is None:
        return AMP_DTYPE
    return str(resolved).replace("torch.", "")


def _optim_config(scratch: dict) -> dict:
    return {
        "amp": {"enabled": AMP_ENABLED, "amp_dtype": _resolved_amp_dtype()},
        "optimizer": {"_target_": "torch.optim.AdamW"},
        "gradient_clip": {
            "_target_": "sam3.train.optim.optimizer.GradientClipper",
            "max_norm": GRAD_CLIP_MAX_NORM,
            "norm_type": GRAD_CLIP_NORM_TYPE,
        },
        "param_group_modifiers": [
            {
                "_target_": "sam3.train.optim.optimizer.layer_decay_param_modifier",
                "_partial_": True,
                "layer_decay_value": scratch["lrd_vision_backbone"],
                "apply_to": "backbone.vision_backbone.trunk",
                "overrides": [{"pattern": "*pos_embed*", "value": 1.0}],
            }
        ],
        "options": {
            "lr": [
                {"scheduler": _cosine_with_warmup_scheduler(scratch["lr_transformer"], scratch)},
                {
                    "scheduler": _cosine_with_warmup_scheduler(scratch["lr_vision_backbone"], scratch),
                    "param_names": ["backbone.vision_backbone.*"],
                },
                {
                    "scheduler": _cosine_with_warmup_scheduler(scratch["lr_language_backbone"], scratch),
                    "param_names": ["backbone.language_backbone.*"],
                },
            ],
            "weight_decay": [
                {
                    "scheduler": {
                        "_target_": "fvcore.common.param_scheduler.ConstantParamScheduler",
                        "value": scratch["wd"],
                    }
                },
                {
                    "scheduler": {
                        "_target_": "fvcore.common.param_scheduler.ConstantParamScheduler",
                        "value": 0.0,
                    },
                    "param_names": ["*bias*"],
                    "module_cls_names": ["torch.nn.LayerNorm"],
                },
            ],
        },
    }


def _cosine_with_warmup_scheduler(base_lr: float, scratch: dict) -> dict:
    """Linear warmup (0 → base_lr) followed by cosine decay (base_lr → base_lr * end_ratio).

    Both segments are expressed as fractions of total training (where in [0,1]),
    so the schedule scales correctly with any choice of max_epochs / steps_per_epoch.
    """
    warmup_fraction = scratch["warmup_fraction"]
    end_lr = base_lr * scratch["cosine_end_lr_ratio"]
    return {
        "_target_": "fvcore.common.param_scheduler.CompositeParamScheduler",
        "schedulers": [
            {
                "_target_": "fvcore.common.param_scheduler.LinearParamScheduler",
                "start_value": 0.0,
                "end_value": base_lr,
            },
            {
                "_target_": "fvcore.common.param_scheduler.CosineParamScheduler",
                "start_value": base_lr,
                "end_value": end_lr,
            },
        ],
        "lengths": [warmup_fraction, 1.0 - warmup_fraction],
        "interval_scaling": ["rescaled", "rescaled"],
    }

