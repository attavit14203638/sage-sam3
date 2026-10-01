"""Training components for the granularity-aware arm.

Provides the collate function that pairs every image with its `tree` and `tree canopy` queries, the
loss wrapper that rescales the native SAM3 loss by the inverse square root of the number of queries
per image so that image-scale normalisation is preserved, and the two-GPU smoke run that gates a
full launch.
"""

from __future__ import annotations

import copy
import gc
import logging
import math
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
from sam3.train.data.collator import collate_fn_api
from sam3.train.loss.sam3_loss import Sam3LossWrapper

from prompt_granularity import CATEGORIES, SMOKE_CASES


def collate_fn_prompt_granularity(batch, dict_key, queries_per_image, **kwargs):
    """Collate image and prompt pairs after checking that each image carries exactly the expected queries (`tree canopy` then `tree` in training) and that the collated grouping stays contiguous per image."""
    names = {category["id"]: category["name"] for category in CATEGORIES}
    if queries_per_image not in (1, 2):
        raise ValueError("Prompt-granularity supports one validation query or two training queries per image.")
    expected_categories = []
    for sample in batch:
        if len(sample.images) != 1 or len(sample.find_queries) != queries_per_image:
            raise ValueError("Image/prompt count drift: absent-category queries must not be dropped.")
        actual = []
        for query in sample.find_queries:
            category = query.inference_metadata.original_category_id
            if query.query_text != names.get(category) or query.image_id != 0 or query.query_processing_order != 0:
                raise ValueError("Prompt text, category, image, or stage mapping is inconsistent.")
            actual.append(category)
        if queries_per_image == 2 and actual != [1, 2]:
            raise ValueError("Training query order must be [tree canopy, tree].")
        expected_categories.extend(actual)
    result = collate_fn_api(batch, dict_key=dict_key, **kwargs)
    value = result[dict_key]
    pair_count = len(batch) * queries_per_image
    if len(value.find_inputs) != 1 or len(value.find_inputs[0].img_ids) != pair_count:
        raise ValueError("Collated prompt-pair count differs from the declared image batch.")
    expected_img_ids = [index for index in range(len(batch)) for _ in range(queries_per_image)]
    if value.find_inputs[0].img_ids.tolist() != expected_img_ids:
        raise ValueError("Collated image/prompt grouping is not contiguous per image.")
    if value.find_metadatas[0].original_category_id.tolist() != expected_categories:
        raise ValueError("Collated prompt categories differ from the checked query order.")
    if value.find_targets[0].num_boxes.numel() != pair_count:
        raise ValueError("Collated target counts differ from the prompt-pair count.")
    return result


collate_prompt_granularity = collate_fn_prompt_granularity


class PromptGranularityLoss(Sam3LossWrapper):
    """SAM3 loss wrapper that rescales the native loss by the inverse square root of the queries per image."""
    def __init__(self, queries_per_image, **kwargs) -> None:
        super().__init__(**kwargs)
        if queries_per_image not in (1, 2) or not self.scale_by_find_batch_size:
            raise ValueError("Image-based normalization requires native sqrt batch scaling and a known query count.")
        self.queries_per_image = queries_per_image
        self.pg_scale = 1.0 / math.sqrt(queries_per_image)
        self.pg_calls = 0

    def forward(self, find_stages, find_targets):
        """Compute the native SAM3 loss for complete image and prompt groups, scale `core_loss` by the inverse square root of the queries per image, and log the group counts."""
        if len(find_targets) != 1:
            raise ValueError("Prompt-granularity is an image-only, single-stage arm.")
        counts = find_targets[0]["num_boxes"]
        pairs = len(counts)
        if not pairs or pairs % self.queries_per_image:
            raise ValueError("Loss batch does not contain complete image/prompt groups.")
        losses = super().forward(find_stages, find_targets)
        losses["core_loss"] = losses["core_loss"] * self.pg_scale
        scalar = losses["core_loss"].detach()
        losses["pg_pair_count"] = scalar.new_tensor(float(pairs))
        losses["pg_image_count"] = scalar.new_tensor(float(pairs // self.queries_per_image))
        losses["pg_scale"] = scalar.new_tensor(self.pg_scale)
        if self.queries_per_image == 2:
            losses["pg_canopy_targets"] = counts[::2].sum().detach().float()
            losses["pg_tree_targets"] = counts[1::2].sum().detach().float()
        self.pg_calls += 1
        if self.pg_calls == 1 or self.pg_calls % 100 == 0:
            logging.info(
                "PROMPT_GRANULARITY_ACTIVE calls=%d pairs=%d images=%d scale=%.8f targets=%s",
                self.pg_calls,
                pairs,
                pairs // self.queries_per_image,
                self.pg_scale,
                counts.detach().cpu().tolist(),
            )
        return losses


def _last_output(stages):
    if len(stages.output) != 1 or len(stages.output[0]) != 1:
        raise RuntimeError("Image smoke expects one stage and one interaction step.")
    return stages.output[0][0]


def _optimizer_step_count(optimizer) -> int:
    values = []
    for state in optimizer.state.values():
        step = state.get("step")
        if step is not None:
            values.append(int(step.item() if torch.is_tensor(step) else step))
    return max(values, default=0)


def _all_gradients_finite(model) -> bool:
    return all(torch.isfinite(parameter.grad).all().item() for parameter in model.parameters() if parameter.grad is not None)


def _mask_gradient_sum(model) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.segmentation_head.mask_predictor.parameters()
        if parameter.grad is not None
    )


def _run_optimizer_step(*, ddp, model, criterion, optim, scaler, batches, config, case, global_step, copy_data_to_device):
    from progress_trainer import _ddp_accumulation_context, _synchronize_static_graph_accumulated_gradients
    if len(batches) != config.GRADIENT_ACCUMULATION_STEPS:
        raise RuntimeError("Smoke accumulation group differs from the production recipe.")
    ddp.train()
    optim.zero_grad(set_to_none=True)
    losses_seen = []
    telemetry = []
    output_shapes = None
    for micro, inputs in enumerate(batches):
        inputs = copy_data_to_device(inputs, model.device)
        sync_context = _ddp_accumulation_context(ddp, micro, len(batches))
        with sync_context:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                stages = ddp(inputs)
                output = _last_output(stages)
                expected_pairs = 2 * config.TRAIN_BATCH_SIZE
                for key in ("pred_masks", "pred_logits", "pred_masks_o2m", "pred_logits_o2m"):
                    if key not in output or output[key].shape[:2] != (expected_pairs, 200):
                        raise RuntimeError(f"Unexpected {key} shape: {getattr(output.get(key), 'shape', None)}")
                if any("pred_masks" in auxiliary or "pred_masks_o2m" in auxiliary for auxiliary in output.get("aux_outputs", [])):
                    raise RuntimeError("Auxiliary decoder mask supervision became active.")
                targets = [model.back_convert(target) for target in inputs.find_targets]
                losses = criterion(stages, targets)
                scaled_loss = losses["core_loss"] / math.sqrt(config.GRADIENT_ACCUMULATION_STEPS)
            if not torch.isfinite(scaled_loss):
                raise FloatingPointError(f"Non-finite smoke loss on {case}")
            scaler.scale(scaled_loss).backward()
        values = {key: float(value.detach()) for key, value in losses.items() if key.startswith("pg_")}
        telemetry.append(values)
        losses_seen.append(float(scaled_loss.detach()))
        output_shapes = {key: list(output[key].shape) for key in ("pred_masks", "pred_logits", "pred_masks_o2m", "pred_logits_o2m")}
        del inputs, stages, output, targets, losses, scaled_loss
    explicit_gradient_sync = _synchronize_static_graph_accumulated_gradients(ddp, len(batches))
    if not explicit_gradient_sync:
        raise RuntimeError("Smoke did not apply explicit static-graph accumulated-gradient synchronization.")
    scaler.unscale_(optim.optimizer)
    if not _all_gradients_finite(model):
        raise FloatingPointError(f"Non-finite gradient on {case}")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM, error_if_nonfinite=True)
    mask_grad = _mask_gradient_sum(model)
    if case != "empty" and mask_grad <= 0:
        raise RuntimeError(f"Mask predictor received no gradient on {case}.")
    gradient_signature = torch.stack((grad_norm.detach().float(), torch.tensor(mask_grad, device=model.device)))
    signatures = [torch.empty_like(gradient_signature) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(signatures, gradient_signature)
    if any(not torch.equal(signatures[0], value) for value in signatures[1:]):
        raise RuntimeError(f"DDP gradients differ across ranks on {case}: {[value.cpu().tolist() for value in signatures]}")
    before = _optimizer_step_count(optim.optimizer)
    optim.step_schedulers((global_step + 1) / (30 * 260), global_step)
    scaler.step(optim.optimizer)
    scaler.update()
    after = _optimizer_step_count(optim.optimizer)
    if after != before + 1:
        raise RuntimeError(f"Optimizer step was skipped on {case}: {before} -> {after}")
    target_counts = {
        "tree": int(sum(values["pg_tree_targets"] for values in telemetry)),
        "tree_canopy": int(sum(values["pg_canopy_targets"] for values in telemetry)),
    }
    result = {
        "case": case,
        "microbatches": len(batches),
        "losses": losses_seen,
        "telemetry": telemetry,
        "target_counts": target_counts,
        "grad_norm": float(grad_norm),
        "mask_grad": mask_grad,
        "optimizer_step_applied": True,
        "ddp_gradient_equal": True,
        "explicit_gradient_sync": explicit_gradient_sync,
        "output_shapes": output_shapes,
    }
    print(
        f"rank={torch.distributed.get_rank()} smoke={case} loss={losses_seen[-1]:.6f} "
        f"norm={float(grad_norm):.6f} mask_grad={mask_grad:.6f}",
        flush=True,
    )
    return result


def run_smoke(args) -> None:
    """Run the two-GPU pre-launch smoke test and write its report."""
    import numpy as np
    import sam3
    import torch.distributed as dist
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.optim.optimizer import construct_optimizer

    from prompt_granularity import (
        RUN_NAME,
        SMOKE_REPORT_VERSION,
        digest,
        prefix_identity,
        training_contract,
        write_once,
    )
    from project_paths import CODEBASE_DIR

    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    output = args.output.resolve()
    if world_size != 2 or local_rank not in (0, 1) or rank not in (0, 1):
        raise RuntimeError("Run smoke via torch.distributed.run --standalone --nproc_per_node=2.")
    if output.parent != CODEBASE_DIR / "experiments" or not output.name.startswith(RUN_NAME + "_smoke"):
        raise ValueError("Smoke output must be a new experiments/sam3_prompt_granularity_smoke* directory.")
    if output.exists():
        raise FileExistsError(f"Smoke output already exists; preserve it and choose a new path: {output}")
    if torch.cuda.device_count() != 2 or any("A40" not in torch.cuda.get_device_name(index) for index in range(2)):
        raise RuntimeError("The smoke gate requires two visible A40 GPUs.")
    from device_policy import assert_node_headroom

    assert_node_headroom(require_gpu_mib=30000, require_gpus=2)
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.INFO)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=device)
    if rank == 0:
        output.mkdir(exist_ok=False)
    dist.barrier()

    import config
    import training_pipeline

    if not Path(sam3.__file__).resolve().is_relative_to(CODEBASE_DIR / "Support/sam3"):
        raise RuntimeError(f"Unexpected SAM3 import: {sam3.__file__}")
    if Path(training_pipeline.__file__).resolve() != CODEBASE_DIR / "Core/training_pipeline.py":
        raise RuntimeError(f"Unexpected Core import: {training_pipeline.__file__}")
    contract = training_contract(args.data_root)
    if rank == 0:
        write_once(output / "smoke_contract.json", contract)
    random.seed(config.TRAINER_SEED + rank)
    np.random.seed(config.TRAINER_SEED + rank)
    torch.manual_seed(config.TRAINER_SEED + rank)
    cfg = config.build_sam3_train_config(config.get_training_preset("full_ft", data_root=args.data_root, experiment_log_dir=output))
    model = instantiate(cfg["trainer"]["model"], _convert_="all").to(device)
    architecture = {
        "num_queries": model.transformer.decoder.num_queries,
        "num_layers": model.transformer.decoder.num_layers,
        "dac": model.transformer.decoder.dac,
        "o2m_mask_predict": model.o2m_mask_predict,
        "aux_masks": model.segmentation_head.aux_masks,
    }
    if architecture != {"num_queries": 200, "num_layers": 6, "dac": True, "o2m_mask_predict": True, "aux_masks": False}:
        raise RuntimeError(f"Smoke model differs from locked A0 architecture: {architecture}")
    criterion = instantiate(cfg["trainer"]["loss"]["all"], _convert_="all").to(device)
    optim_cfg = OmegaConf.create(cfg["trainer"]["optim"])
    optim = construct_optimizer(model, optim_cfg.optimizer, optim_cfg.options, optim_cfg.param_group_modifiers)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    ddp = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        static_graph=True,
        find_unused_parameters=False,
        gradient_as_bucket_view=False,
    )
    train = instantiate(cfg["trainer"]["data"]["train"]["dataset"], _convert_="all")
    val = instantiate(cfg["trainer"]["data"]["val"]["dataset"], _convert_="all")
    collate_train = instantiate(cfg["scratch"]["collate_fn"], _convert_="all")
    collate_val = instantiate(cfg["scratch"]["collate_fn_val"], _convert_="all")
    candidates = []
    for index, record in enumerate(train.coco._raw_data):
        categories = {annotation["category_id"] for annotation in record["annotations"] if not annotation.get("iscrowd", 0)}
        if categories == {1, 2}:
            candidates.append(index)
    if not candidates:
        raise RuntimeError("No mixed-category non-crowd training image is available for the smoke gate.")
    cursor = rank

    def sample(case):
        nonlocal cursor
        for _ in range(64):
            value = train[candidates[cursor % len(candidates)]]
            cursor += 2
            if len(value.find_queries) == 2 and all(query.object_ids_output for query in value.find_queries):
                break
        else:
            raise RuntimeError("Could not obtain a crop retaining both target categories.")
        if case != "mixed":
            value = copy.deepcopy(value)
            for query in value.find_queries:
                if case == "empty" or (case == "tree_only" and query.query_text != "tree") or (case == "canopy_only" and query.query_text != "tree canopy"):
                    query.object_ids_output = []
                query.semantic_target = None
            if case == "empty":
                value.images[0].data.zero_()
        return value

    torch.cuda.reset_peak_memory_stats(device)
    fixtures = []
    for step, case in enumerate(SMOKE_CASES):
        batches = [
            collate_train([sample(case) for _ in range(config.TRAIN_BATCH_SIZE)])["all"]
            for _ in range(config.GRADIENT_ACCUMULATION_STEPS)
        ]
        result = _run_optimizer_step(
            ddp=ddp,
            model=model,
            criterion=criterion,
            optim=optim,
            scaler=scaler,
            batches=batches,
            config=config,
            case=case,
            global_step=step,
            copy_data_to_device=copy_data_to_device,
        )
        tree, canopy = result["target_counts"]["tree"], result["target_counts"]["tree_canopy"]
        if case == "mixed" and not (tree > 0 and canopy > 0):
            raise RuntimeError("Mixed smoke fixture lost one category.")
        if case == "tree_only" and not (tree > 0 and canopy == 0):
            raise RuntimeError("Tree-only smoke fixture is inconsistent.")
        if case == "canopy_only" and not (tree == 0 and canopy > 0):
            raise RuntimeError("Canopy-only smoke fixture is inconsistent.")
        if case == "empty" and (tree != 0 or canopy != 0):
            raise RuntimeError("Empty smoke fixture contains targets.")
        fixtures.append(result)
        del batches

    train_wrapper_cfg = copy.deepcopy(cfg["trainer"]["data"]["train"])
    train_wrapper_cfg["dataset"]["limit_ids"] = 64
    train_wrapper = instantiate(train_wrapper_cfg, _convert_="all")
    train_iterator = iter(train_wrapper.get_loader(0))
    production_group = next(train_iterator)
    if not isinstance(production_group, list) or len(production_group) != config.GRADIENT_ACCUMULATION_STEPS:
        raise RuntimeError("ProjectTorchDataset did not produce the production accumulation layout.")
    production_batches = [entry["all"] for entry in production_group]
    production_step = _run_optimizer_step(
        ddp=ddp,
        model=model,
        criterion=criterion,
        optim=optim,
        scaler=scaler,
        batches=production_batches,
        config=config,
        case="production_loader",
        global_step=len(SMOKE_CASES),
        copy_data_to_device=copy_data_to_device,
    )
    del production_group, production_batches, train_iterator, train_wrapper
    gc.collect()

    val_wrapper_cfg = copy.deepcopy(cfg["trainer"]["data"]["val"])
    val_wrapper = instantiate(val_wrapper_cfg, _convert_="all")
    val_iterator = iter(val_wrapper.get_loader(0))
    val_inputs = next(val_iterator)["oamtcd"]
    val_inputs = copy_data_to_device(val_inputs, device)
    ddp.eval()
    meter_cfg = copy.deepcopy(cfg["trainer"]["meters"]["val"]["oamtcd"]["detection"])
    meter_cfg["pred_file_evaluators"] = None
    meter = instantiate(meter_cfg, _convert_="all")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        val_stages = ddp(val_inputs)
        validation_output = _last_output(val_stages)
        if validation_output["pred_masks"].shape[:2] != (config.VAL_BATCH_SIZE, 200):
            raise RuntimeError(f"Unexpected validation shape: {validation_output['pred_masks'].shape}")
        if not torch.isfinite(validation_output["pred_masks"]).all() or not torch.isfinite(validation_output["pred_logits"]).all():
            raise FloatingPointError("Non-finite validation output.")
        meter.update(find_stages=val_stages, find_metadatas=val_inputs.find_metadatas, model=ddp, batch=val_inputs, key="oamtcd")
    query_categories = sorted(set(val_inputs.find_metadatas[0].original_category_id.tolist()))
    prediction_categories = sorted({entry["category_id"] for entry in meter.dump})
    if not set(query_categories) <= {1, 2} or not set(prediction_categories) <= {1, 2}:
        raise RuntimeError("Validation postprocessing lost the prompt category mapping.")
    meter.synchronize_between_processes()
    del validation_output, val_stages, meter, val_iterator, val_wrapper, val_inputs
    gc.collect()

    dense_index = max(range(len(val.coco._raw_data)), key=lambda index: len(val.coco._raw_data[index]["annotations"]))
    dense_inputs = collate_val([val[dense_index * 2 + rank] for _ in range(config.VAL_BATCH_SIZE)])["oamtcd"]
    dense_inputs = copy_data_to_device(dense_inputs, device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        dense_output = _last_output(ddp(dense_inputs))
        if not torch.isfinite(dense_output["pred_masks"]).all():
            raise FloatingPointError("Non-finite dense validation masks.")
    del dense_inputs, dense_output
    reload_payload = [collate_val([val[dense_index * 2] for _ in range(config.VAL_BATCH_SIZE)])["oamtcd"] if rank == 0 else None]
    dist.broadcast_object_list(reload_payload, src=0, device=device)
    reload_inputs = copy_data_to_device(reload_payload[0], device)
    del reload_payload
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        before = _last_output(model(reload_inputs))["pred_logits"].detach().contiguous()
    before_by_rank = [torch.empty_like(before) for _ in range(world_size)]
    dist.all_gather(before_by_rank, before)
    if any(not torch.equal(before_by_rank[0], value) for value in before_by_rank[1:]):
        raise RuntimeError("DDP ranks differ on identical checkpoint-validation inputs before saving.")
    dist.barrier()

    checkpoint_path = output / "smoke_checkpoint.pt"
    if rank == 0:
        torch.save(
            {
                "purpose": "smoke",
                "contract": contract,
                "model": model.state_dict(),
                "optimizer": optim.optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "loss": criterion.state_dict(),
            },
            checkpoint_path,
        )
    dist.barrier()
    del ddp, model, optim, criterion, scaler
    gc.collect()
    torch.cuda.empty_cache()
    model = instantiate(cfg["trainer"]["model"], _convert_="all").to(device)
    criterion = instantiate(cfg["trainer"]["loss"]["all"], _convert_="all").to(device)
    optim = construct_optimizer(model, optim_cfg.optimizer, optim_cfg.options, optim_cfg.param_group_modifiers)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    loaded = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    if loaded["purpose"] != "smoke" or loaded["contract"] != contract:
        raise RuntimeError("Smoke checkpoint identity changed.")
    model.load_state_dict(loaded["model"], strict=True)
    optim.optimizer.load_state_dict(loaded["optimizer"])
    scaler.load_state_dict(loaded["scaler"])
    criterion.load_state_dict(loaded["loss"], strict=True)
    del loaded
    if _optimizer_step_count(optim.optimizer) != len(SMOKE_CASES) + 1:
        raise RuntimeError("Smoke checkpoint did not restore all optimizer updates.")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        after = _last_output(model(reload_inputs))["pred_logits"].detach().contiguous()
    after_by_rank = [torch.empty_like(after) for _ in range(world_size)]
    dist.all_gather(after_by_rank, after)
    local_equal = torch.equal(before, after)
    ranks_equal = all(torch.equal(after_by_rank[0], value) for value in after_by_rank[1:])
    equality = torch.tensor(int(local_equal and ranks_equal), device=device)
    dist.all_reduce(equality, op=dist.ReduceOp.MIN)
    reload_equal = bool(equality.item())
    if not reload_equal:
        diagnostics = {
            "rank": rank,
            "local_equal": local_equal,
            "ranks_equal": ranks_equal,
            "unequal_logits": int(torch.ne(before, after).sum().item()),
            "max_abs_difference": float((before.float() - after.float()).abs().max().item()),
        }
        gathered_diagnostics = [None] * world_size
        dist.all_gather_object(gathered_diagnostics, diagnostics)
        raise RuntimeError(f"Fresh checkpoint reload changed validation logits: {gathered_diagnostics}")
    torch.cuda.synchronize(device)
    del reload_inputs, before, before_by_rank, after, after_by_rank
    production_integration = {
        "status": "passed",
        "train_workers": cfg["trainer"]["data"]["train"]["num_workers"],
        "val_workers": cfg["trainer"]["data"]["val"]["num_workers"],
        "train_microbatches": production_step["microbatches"],
        "optimizer_step_applied": production_step["optimizer_step_applied"],
        "ddp_gradient_equal": production_step["ddp_gradient_equal"],
        "explicit_gradient_sync": production_step["explicit_gradient_sync"],
        "target_counts": production_step["target_counts"],
        "grad_norm": production_step["grad_norm"],
        "mask_grad": production_step["mask_grad"],
        "output_shapes": production_step["output_shapes"],
    }
    rank_report = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": torch.cuda.get_device_name(local_rank),
        "amp_dtype": "bfloat16",
        "micro_batch": config.TRAIN_BATCH_SIZE,
        "accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
        "ddp_accumulation_sync": "every_microbatch_plus_explicit_final_when_static_graph",
        "effective_image_batch": config.TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS * world_size,
        "optimizer_steps": len(SMOKE_CASES) + 1,
        "skipped_optimizer_steps": 0,
        "architecture": architecture,
        "sources": {
            "core": str(Path(training_pipeline.__file__).resolve()),
            "sam3": str(Path(sam3.__file__).resolve()),
        },
        "fixtures": fixtures,
        "production_integration": production_integration,
        "validation": {
            "status": "passed",
            "dummy_loss": cfg["trainer"]["loss"]["default"]["_target_"] == "sam3.train.loss.sam3_loss.DummyLoss",
            "postprocessing": "passed",
            "query_categories": query_categories,
            "prediction_categories": prediction_categories,
            "dense_forward": "passed",
            "checkpoint_reload_equal": reload_equal,
            "checkpoint_fresh_model": True,
            "checkpoint_shared_inputs": True,
            "checkpoint_optimizer_steps": _optimizer_step_count(optim.optimizer),
        },
        "memory_mib": {
            "peak_allocated": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved": torch.cuda.max_memory_reserved(device) / 2**20,
        },
    }
    ranks = [None, None]
    dist.all_gather_object(ranks, rank_report)
    all_query_categories = {category for row in ranks for category in row["validation"]["query_categories"]}
    all_prediction_categories = {category for row in ranks for category in row["validation"]["prediction_categories"]}
    if all_query_categories != {1, 2} or all_prediction_categories != {1, 2}:
        raise RuntimeError(f"Global validation categories are incomplete: queries={all_query_categories}, predictions={all_prediction_categories}")
    if rank == 0:
        checkpoint = {
            "status": "passed",
            "reload_equal": all(row["validation"]["checkpoint_reload_equal"] for row in ranks),
            "purpose": "smoke",
            "contract_sha256": digest(contract),
            "state_keys": ["contract", "loss", "model", "optimizer", "purpose", "scaler"],
            **prefix_identity(checkpoint_path),
        }
        report = {
            "report_version": SMOKE_REPORT_VERSION,
            "status": "passed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output),
            "world_size": world_size,
            "skipped_checks": [],
            "contract": contract,
            "contract_sha256": digest(contract),
            "checkpoint": checkpoint,
            "ranks": ranks,
        }
        write_once(output / "smoke_report.json", report)
        print(f"SMOKE PASSED: {output / 'smoke_report.json'}", flush=True)
    dist.destroy_process_group()
