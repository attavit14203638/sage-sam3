"""SAM3 trainer wrapper with progress reporting, finite-value guards, and checkpoint export.

Extends the upstream SAM3 trainer with a tqdm progress meter, fused checks that abort on non-finite
gradients, parameters, or optimiser state, rolling trainer-state and checkpoint views for the best
and final epochs, and a model-only weight export written once at the end of a run. For the
granularity-aware arm it also stamps the run manifest into every saved checkpoint.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import shutil
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from sam3.model.model_misc import SAM3Output
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.trainer import Trainer, unwrap_ddp_if_wrapped
from sam3.train.utils.train_utils import AverageMeter, get_amp_type, human_readable_time, MemMeter, Phase

from config import FULL_FT_MAX_DATA_EPOCHS, TB_PROJECT_METRICS
from atomic_io import atomic_write_text


_INSTALLED = False
_ORIGINAL_PROGRESS_METER = None
PROJECT_METRIC_ORDER = list(TB_PROJECT_METRICS)


def _ddp_accumulation_context(model, index: int, accumulation_steps: int):
    if index < accumulation_steps - 1 and hasattr(model, "no_sync") and getattr(model, "static_graph", None) is False:
        return model.no_sync()
    return contextlib.nullcontext()


def _synchronize_static_graph_accumulated_gradients(model, accumulation_steps: int, bucket_bytes: int = 64 * 1024 * 1024) -> bool:
    if accumulation_steps <= 1 or getattr(model, "static_graph", None) is not True:
        return False
    if not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_world_size() <= 1:
        raise RuntimeError("Static-graph accumulated training requires an initialized multi-rank process group.")
    named_gradients = [(name, parameter.grad) for name, parameter in model.named_parameters() if parameter.grad is not None]
    if not named_gradients:
        raise RuntimeError("Static-graph accumulated training produced no gradients.")
    names = tuple(name for name, _ in named_gradients)
    recorded_names = getattr(model, "_project_accumulated_gradient_names", None)
    if recorded_names is None:
        gathered_names = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_names, names)
        if any(value != names for value in gathered_names):
            raise RuntimeError("Static-graph accumulated gradient membership differs across ranks.")
        model._project_accumulated_gradient_names = names
    elif recorded_names != names:
        raise RuntimeError("Static-graph accumulated gradient membership changed between optimizer steps.")
    groups = {}
    for _, gradient in named_gradients:
        if gradient.is_sparse:
            raise RuntimeError("Sparse gradients are unsupported by the explicit accumulated-gradient synchronization.")
        groups.setdefault((gradient.device, gradient.dtype), []).append(gradient)
    world_size = torch.distributed.get_world_size()
    for gradients in groups.values():
        start = 0
        while start < len(gradients):
            end = start
            size = 0
            while end < len(gradients):
                gradient_size = gradients[end].numel() * gradients[end].element_size()
                if end > start and size + gradient_size > bucket_bytes:
                    break
                size += gradient_size
                end += 1
            bucket = gradients[start:end]
            flat = torch._utils._flatten_dense_tensors(bucket)
            torch.distributed.all_reduce(flat)
            flat.div_(world_size)
            for gradient, synchronized in zip(bucket, torch._utils._unflatten_dense_tensors(flat, bucket)):
                gradient.copy_(synchronized)
            start = end
    return True


class TqdmProgressMeter:
    # Class-level reference to the optimizer so the tqdm postfix can show the
    # realized learning rate without the trainer having to inject anything per
    # iteration. Set once by ProgressBarTrainer.__init__.
    """tqdm progress bar that replaces SAM3's console progress meter and shows the realised learning rate."""
    _OPTIMIZER: Any | None = None

    def __init__(self, num_batches, meters, real_meters, prefix="") -> None:
        self.num_batches = num_batches
        self.meters = meters
        self.real_meters = real_meters
        self.prefix = prefix
        self._last_n = 0
        self._disabled = not _is_primary_rank()
        self.pbar = None
        if not self._disabled:
            try:
                from tqdm import tqdm
                phase = "Val" if self.prefix.lower().startswith("val") else "Epoch"
                epoch_idx = _epoch_index_from_prefix(self.prefix)
                epoch_text = f"{epoch_idx + 1}/{FULL_FT_MAX_DATA_EPOCHS}" if epoch_idx is not None else "?"
                desc = f"{phase} {epoch_text}"
                self.pbar = tqdm(total=num_batches, desc=desc, leave=True)
            except ImportError:
                pass

    def display(self, batch, enable_print=False) -> None:
        """Advance the bar to `batch` and refresh its postfix."""
        if self._disabled:
            return
        current_n = min(batch + 1, self.num_batches)
        if current_n <= self._last_n:
            return
        
        if self.pbar is not None:
            self.pbar.set_postfix(self._postfix(), refresh=False)
            self.pbar.update(current_n - self._last_n)
        else:
            # Fallback if tqdm is not installed
            phase = "Val" if self.prefix.lower().startswith("val") else "Epoch"
            epoch_idx = _epoch_index_from_prefix(self.prefix)
            epoch_text = f"{epoch_idx + 1}/{FULL_FT_MAX_DATA_EPOCHS}" if epoch_idx is not None else "?"
            percent = 100.0 * current_n / max(self.num_batches, 1)
            print(f"{phase} {epoch_text} | step {current_n}/{self.num_batches} | {percent:.0f}%", flush=True)

        self._last_n = current_n
        
        if current_n >= self.num_batches and self.pbar is not None:
            self.pbar.close()

    def __del__(self):
        # The vendored Trainer only calls display() at log_freq-aligned iters,
        # so the bar typically stops a few iterations short of num_batches
        # (e.g. 4151/4169, 333/439). When the meter goes out of scope at the
        # end of the epoch method we force the bar to advance to num_batches
        # and close cleanly so the log shows full coverage (4169/4169, 439/439).
        try:
            if getattr(self, "_disabled", True) or getattr(self, "pbar", None) is None:
                return
            remaining = self.num_batches - self._last_n
            if remaining > 0:
                try:
                    self.pbar.set_postfix(self._postfix(), refresh=False)
                except Exception:
                    pass
                self.pbar.update(remaining)
                self._last_n = self.num_batches
            self.pbar.close()
        except Exception:
            # Never raise from __del__ — GC ordering during interpreter shutdown
            # can make any logging or tqdm call fail; swallow silently.
            pass

    def _postfix(self):
        values: dict[str, Any] = {}
        for meter in self.meters:
            name = _short_meter_name(meter.name)
            val = getattr(meter, "val", None)
            avg = getattr(meter, "avg", None)
            if val is None:
                continue
            if name in {"data_s", "loss/train_default", "loss/val_default", "losses/train_oamtcd_loss"}:
                continue
            if name == "batch_s":
                seconds_per_iter = avg if avg is not None and avg > 0 else val
                if seconds_per_iter and seconds_per_iter > 0:
                    values["rate"] = f"{1.0 / seconds_per_iter:.2f} it/s"
            elif name == "loss":
                values[name] = f"{val:.2e} avg={avg:.2e}" if avg is not None else f"{val:.2e}"
            elif name == "time":
                values[name] = _format_seconds(val)
            elif name == "mem_gb":
                values["mem"] = f"{val:.0f}GB"
            elif isinstance(val, float):
                values[name] = f"{val:.2f}"
            else:
                values[name] = val

        # Realized learning rate from the live optimizer (param group 0 — the
        # main transformer LR). Surfaced in the tqdm postfix so flat-loss
        # regressions are immediately visible.
        opt = TqdmProgressMeter._OPTIMIZER
        if opt is not None:
            try:
                lr_val = opt.param_groups[0].get("lr")
                if lr_val is not None:
                    values["lr"] = f"{float(lr_val):.2e}"
            except Exception:
                logging.debug("Could not read the learning rate for the progress bar", exc_info=True)
        return values

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class ProgressBarTrainer(Trainer):
    """SAM3 trainer with progress reporting, finite-value guards, checkpoint views, and model export."""
    def _save_checkpoint(self, checkpoint, checkpoint_path):
        from config import PROMPT_GRANULARITY_ENABLED
        if PROMPT_GRANULARITY_ENABLED:
            from prompt_granularity import RUN_MANIFEST
            manifest = json.loads((Path(checkpoint_path).parent.parent / RUN_MANIFEST).read_text())
            if manifest.get("purpose") != "full":
                raise ValueError("A full prompt-granularity run manifest is required before checkpointing.")
            checkpoint = {**checkpoint, "prompt_granularity": manifest}
        return super()._save_checkpoint(checkpoint, checkpoint_path)

    def __init__(self, *args, **kwargs) -> None:
        install_tqdm_progress_meter()
        super().__init__(*args, **kwargs)
        self.logger = ProjectMetricLogger(self.logger, self.logging_conf.scalar_keys_to_log)
        # Expose the optimizer to TqdmProgressMeter so it can show the live LR
        # in the tqdm postfix. Safe to read on every iter — PyTorch updates
        # param_groups[i]["lr"] in-place when schedulers step.
        try:
            optim = getattr(self, "optim", None)
            if optim is not None and getattr(optim, "optimizer", None) is not None:
                TqdmProgressMeter._OPTIMIZER = optim.optimizer
        except Exception:
            logging.debug("Could not register optimizer with TqdmProgressMeter", exc_info=True)

    def _setup_ddp_distributed_training(self, distributed_conf, accelerator):
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            logging.info("Skipping DistributedDataParallel wrapping for WORLD_SIZE=%d.", world_size)
            return
        super()._setup_ddp_distributed_training(distributed_conf, accelerator)

    def _verify_initial_parameters_finite(self) -> None:
        """Check parameters once, before the first optimizer step of the process.

        Every later iteration gets this for free: the post-step scan proves
        parameters are finite, and nothing between two steps modifies them. The
        gap that leaves is the very first step, where the weights come from model init or from a resumed checkpoint and have never been inspected.
        """
        if getattr(self, "_initial_parameters_verified", False):
            return
        self._initial_parameters_verified = True
        if _any_nonfinite(_model_parameters(self.model)):
            logging.error(
                "Non-finite model parameters before the first optimizer step: %s",
                _summarize_nonfinite_parameters(self.model),
            )
            raise FloatingPointError("non-finite model parameters before the first optimizer step")

    def is_intermediate_val_epoch(self, epoch):
        """Interpret val_epoch_freq as completed epochs, not zero-based epoch ids."""
        if self.val_epoch_freq <= 0:
            return False
        completed_epochs = int(epoch) + 1
        if self.skip_first_val and completed_epochs == 1:
            return False
        return completed_epochs % self.val_epoch_freq == 0 and completed_epochs < self.max_epochs

    def _log_timers(self, phase):
        """Estimate remaining time when validation is enabled or disabled.

        The upstream modulo calculation raises when `val_epoch_freq` is zero. With validation
        disabled, no validation epochs contribute to the estimate.
        """
        if self.val_epoch_freq > 0:
            super()._log_timers(phase)
            return
        epochs_remaining = self.max_epochs - self.epoch - 1
        time_remaining = epochs_remaining * self.est_epoch_time[Phase.TRAIN]
        self.logger.log(
            os.path.join("Step_Stats", phase, self.time_elapsed_meter.name),
            self.time_elapsed_meter.val,
            self.steps[phase],
        )
        logging.info(f"Estimated time remaining: {human_readable_time(time_remaining)}")

    def run(self) -> None:
        """Run training, then, on the primary rank, refresh the metric files and checkpoint views; the final-model view and weight exports are written only if the run succeeded."""
        failed = False
        try:
            super().run()
        except Exception:
            failed = True
            raise
        finally:
            if _is_primary_rank():
                _refresh_output_artifacts(self, final=not failed)

    def run_val(self) -> None:
        """Release unused training memory before whole-image validation.

        Validation retains every instance in a resized full image, while training operates on
        crops. Gradients from the final training step are not read by checkpointing or
        validation, so they are set to `None` before returning reserved allocator memory to the
        driver. The order lets `empty_cache` release those blocks without changing model state
        or validation numerics.
        """
        import torch

        optim = getattr(self, "optim", None)
        if optim is not None:
            optim.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().run_val()
        if _is_primary_rank():
            _refresh_output_artifacts(self, final=False)

    def train_epoch(self, train_loader):
        """Run one training epoch with gradient accumulation, finite-value checks, and progress reporting."""
        batch_time_meter = AverageMeter("Batch Time", self.device, ":.2f")
        data_time_meter = AverageMeter("Data Time", self.device, ":.2f")
        mem_meter = MemMeter("Mem (GB)", self.device, ":.2f")
        data_times = []
        phase = Phase.TRAIN

        iters_per_epoch = len(train_loader)

        loss_names = []
        for batch_key in self.loss.keys():
            loss_names.append(f"Losses/{phase}_{batch_key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        progress = TqdmProgressMeter(
            iters_per_epoch,
            [
                batch_time_meter,
                data_time_meter,
                mem_meter,
                self.time_elapsed_meter,
                *loss_mts.values(),
            ],
            self._get_meters([phase]),
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        for data_iter, batch in enumerate(train_loader):
            data_time_meter.update(time.time() - end)
            data_times.append(data_time_meter.val)

            try:
                self._run_step(batch, phase, loss_mts, extra_loss_mts)

                exact_epoch = self.epoch + float(data_iter) / iters_per_epoch
                self.where = float(exact_epoch) / self.max_epochs
                assert self.where <= 1 + self.EPSILON
                if self.where < 1.0:
                    self.optim.step_schedulers(
                        self.where, step=int(exact_epoch * iters_per_epoch)
                    )
                else:
                    logging.warning(
                        f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                    )

                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for j, param_group in enumerate(self.optim.optimizer.param_groups):
                        for option in self.optim.schedulers[j]:
                            optim_prefix = (
                                "" + f"{j}_"
                                if len(self.optim.optimizer.param_groups) > 1
                                else ""
                            )
                            self.logger.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )

                _synchronize_static_graph_accumulated_gradients(self.model, self.gradient_accumulation_steps)
                self.scaler.unscale_(self.optim.optimizer)
                bad_gradients = _any_nonfinite(_model_gradients(self.model))
                bad_gradient_stage = "before gradient clipping"
                if not bad_gradients and self.gradient_clipper is not None:
                    self.gradient_clipper(model=self.model)
                    bad_gradients = _any_nonfinite(_model_gradients(self.model))
                    bad_gradient_stage = "after gradient clipping"

                if bad_gradients:
                    self._nan_gradient_skip_consecutive = getattr(
                        self, "_nan_gradient_skip_consecutive", 0
                    ) + 1
                    self._nan_gradient_skip_total = getattr(
                        self, "_nan_gradient_skip_total", 0
                    ) + 1
                    logging.error(
                        "Non-finite gradients detected %s; skipping optimizer step. consecutive=%d total=%d examples=%s",
                        bad_gradient_stage,
                        self._nan_gradient_skip_consecutive,
                        self._nan_gradient_skip_total,
                        _summarize_nonfinite_gradients(self.model),
                    )
                    if (
                        self._nan_gradient_skip_consecutive
                        > self.MAX_CONSECUTIVE_GRADIENT_SKIPS
                    ):
                        logging.error(
                            "Aborting: %d consecutive non-finite-gradient skips indicates "
                            "training has stalled.",
                            self._nan_gradient_skip_consecutive,
                        )
                        raise FloatingPointError(
                            f"{self._nan_gradient_skip_consecutive} consecutive non-finite-gradient skips"
                        )
                    self.scaler.update()
                else:
                    self._nan_gradient_skip_consecutive = 0
                    if self.gradient_logger is not None:
                        self.gradient_logger(
                            self.model, rank=self.distributed_rank, where=self.where
                        )

                    self._verify_initial_parameters_finite()

                    self.scaler.step(self.optim.optimizer)
                    self.scaler.update()

                    # The only per-iteration parameter scan. A finite result here
                    # is also what licenses skipping the pre-step scan on the next
                    # iteration: nothing between two steps mutates parameters.
                    if _any_nonfinite(_model_parameters(self.model)):
                        logging.error(
                            "Optimizer step produced non-finite model parameters: %s",
                            _summarize_nonfinite_parameters(self.model),
                        )
                        bad_optimizer_state = _summarize_nonfinite_optimizer_state(
                            self.optim.optimizer
                        )
                        if bad_optimizer_state:
                            logging.error(
                                "Non-finite optimizer state accompanying the bad step: %s",
                                bad_optimizer_state,
                            )
                        raise FloatingPointError("optimizer step produced non-finite model parameters")

                batch_time_meter.update(time.time() - end)
                end = time.time()

                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )

                mem_meter.update(reset_peak_usage=True)
                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)

                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for progress_meter in progress.meters:
                        self.logger.log(
                            os.path.join("Step_Stats", phase, progress_meter.name),
                            progress_meter.val,
                            self.steps[phase],
                        )

            except FloatingPointError as e:
                raise e

        self.est_epoch_time[Phase.TRAIN] = batch_time_meter.avg * iters_per_epoch
        self._log_timers(Phase.TRAIN)
        self._log_sync_data_times(Phase.TRAIN, data_times)

        out_dict = self._log_meters_and_save_best_ckpts([Phase.TRAIN])

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg
        out_dict.update(self._get_trainer_state(phase))
        logging.info(f"Losses and meters: {out_dict}")
        self._reset_meters([phase])
        return out_dict

    # Abort only if this many *consecutive* train batches produce a non-finite
    # loss (that indicates genuine divergence, not an isolated bad sample).
    MAX_CONSECUTIVE_NAN_SKIPS = 25
    MAX_CONSECUTIVE_GRADIENT_SKIPS = 100

    def _run_step(self, batch, phase, loss_mts, extra_loss_mts, raise_on_error=True):
        """Handle an isolated non-finite training loss without mutating model weights.

        A backward pass lets AMP record non-finite gradients and skip the optimizer step while
        preserving the static DDP graph. Non-finite values are excluded from meters, and a
        consecutive-skip guard still aborts genuine divergence. Validation retains the native
        error behaviour.
        """
        # Set gradients to None because zero-valued Adam gradients can still update parameters.
        self.optim.zero_grad(set_to_none=True)

        if self.gradient_accumulation_steps > 1:
            assert isinstance(batch, list), (
                f"Expected a list of batches, got {type(batch)}"
            )
            assert len(batch) == self.gradient_accumulation_steps, (
                f"Expected {self.gradient_accumulation_steps} batches, got {len(batch)}"
            )
            accum_steps = len(batch)
        else:
            accum_steps = 1
            batch = [batch]

        saw_non_finite = False
        accum_divisor = self._accumulation_divisor(accum_steps)
        for i, chunked_batch in enumerate(batch):
            ddp_context = _ddp_accumulation_context(self.model, i, accum_steps)
            with ddp_context:
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=self.optim_conf.amp.enabled,
                    dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
                ):
                    loss_dict, batch_size, extra_losses = self._step(
                        chunked_batch,
                        self.model,
                        phase,
                    )

                assert len(loss_dict) == 1
                loss_key, loss = loss_dict.popitem()

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    bad_extra_losses = _summarize_nonfinite_losses(extra_losses)
                    if bad_extra_losses:
                        logging.error(
                            "Non-finite loss components: %s",
                            bad_extra_losses,
                        )
                    _log_nonfinite_model_parameters(self.model)
                    if phase == Phase.VAL:
                        # Preserve vendored behavior for validation.
                        if raise_on_error:
                            raise FloatingPointError(error_msg)
                        return
                    saw_non_finite = True
                    # Run backward on the non-finite loss anyway: the enabled
                    # GradScaler will see found_inf during unscale_ and skip the
                    # optimizer step (weights stay clean), while still running a
                    # backward this iteration for DDP static_graph. Do NOT update
                    # the loss meters with a non-finite value.
                    self.scaler.scale(loss / accum_divisor).backward()
                    if _force_grad_scaler_skip(self.optim.optimizer):
                        logging.warning(
                            "Marked one gradient non-finite to force GradScaler to skip "
                            "the optimizer step for this non-finite-loss batch."
                        )
                    else:
                        logging.error(
                            "Could not mark a gradient non-finite; optimizer-step skip "
                            "cannot be guaranteed for this non-finite-loss batch."
                        )
                    continue

                # Normalise accumulated microbatches according to the native square-root batch
                # convention. Meters retain the undivided loss so logged curves remain
                # comparable across accumulation settings.
                self.scaler.scale(loss / accum_divisor).backward()
                loss_mts[loss_key].update(loss.item(), batch_size)
                for extra_loss_key, extra_loss in extra_losses.items():
                    if extra_loss_key not in extra_loss_mts:
                        extra_loss_mts[extra_loss_key] = AverageMeter(
                            extra_loss_key, self.device, ":.2e"
                        )
                    extra_loss_mts[extra_loss_key].update(extra_loss.item(), batch_size)

        if saw_non_finite:
            self._nan_skip_total = getattr(self, "_nan_skip_total", 0) + 1
            self._nan_skip_consecutive = getattr(self, "_nan_skip_consecutive", 0) + 1
            logging.warning(
                "Non-finite training loss at epoch %s (optimizer step skipped by "
                "GradScaler; weights unchanged). consecutive=%d total=%d.",
                self.epoch,
                self._nan_skip_consecutive,
                self._nan_skip_total,
            )
            if self._nan_skip_consecutive > self.MAX_CONSECUTIVE_NAN_SKIPS:
                logging.error(
                    "Aborting: %d consecutive non-finite training losses indicates "
                    "divergence, not an isolated bad batch.",
                    self._nan_skip_consecutive,
                )
                raise FloatingPointError(
                    f"{self._nan_skip_consecutive} consecutive non-finite training losses"
                )
        else:
            self._nan_skip_consecutive = 0

    def _accumulation_divisor(self, accum_steps: int) -> float:
        """What to divide each microbatch loss by so accumulation == one big batch.

        Plain gradient accumulation divides by N: the per-microbatch loss is
        already a mean, so summing N of them overshoots by N.

        **SAM3 is not plain.** With `scale_by_find_batch_size` (True in this
        project's config) the loss is multiplied by the square root of the
        microbatch size, `cur_losses *= bs**0.5`
        (`sam3/train/loss/sam3_loss.py:192-195`). So for microbatch `b`,
        accumulation count `N`, and full batch `B = N*b`, matching the single
        batch means solving

            sum_i  sqrt(b) * L_i / d   ==   sqrt(B) * L
            sqrt(b) * N * L / d        ==   sqrt(N*b) * L
            =>  d = N / sqrt(N) = sqrt(N)

        i.e. **sqrt(N), not N** -- independent of `b`. Dividing by N instead
        would halve the gradient at N=4, so the capped V100 run would silently
        optimise at half the recipe's effective step rather than matching it.
        Without sqrt scaling the answer is the usual N, so both are handled and
        the flag is read off the loss modules rather than assumed.

        Only modules that *declare* the flag are consulted. The loss ModuleDict
        also carries a `default: DummyLoss` (`config.py`'s trainer loss block),
        and `DummyLoss` is a plain `nn.Module` that returns 0 and has no such
        attribute. Treating a missing attribute as "declared False" reads that
        placeholder as disagreement and aborts a run that is not ambiguous at
        all -- which is exactly what happened on the first calibration launch
        after this method was added. Absence of the flag is not a vote.

        A genuine mixture -- two modules that both declare it, disagreeing -- is
        still an error rather than a guess: the answers differ by sqrt(N), and
        picking one silently would reintroduce the discrepancy this method exists
        to remove.
        """
        if accum_steps <= 1:
            return 1.0
        modules = list(self.loss.values()) if self.loss is not None else []
        unset = object()
        declared = (getattr(m, "scale_by_find_batch_size", unset) for m in modules)
        flags = {bool(flag) for flag in declared if flag is not unset}
        if len(flags) > 1:
            raise ValueError(
                "Loss modules that declare scale_by_find_batch_size disagree, so the "
                f"correct gradient-accumulation divisor is ambiguous (sqrt({accum_steps}) "
                f"vs {accum_steps}). Make the flag consistent across loss keys, or set "
                "GRADIENT_ACCUMULATION_STEPS = 1."
            )
        if flags == {True}:
            return math.sqrt(accum_steps)
        return float(accum_steps)

    def _step(self, batch, model, phase: str):
        if phase != Phase.VAL:
            return super()._step(batch, model, phase)

        key, batch = batch.popitem()
        batch = copy_data_to_device(batch, self.device, non_blocking=True)

        find_stages = model(batch)
        find_targets = [
            unwrap_ddp_if_wrapped(model).back_convert(x) for x in batch.find_targets
        ]
        batch_size = len(batch.img_batch)
        loss = self.loss["default"](find_stages, find_targets)

        loss_str = f"Losses/{phase}_{key}_loss"
        loss_log_str = os.path.join("Step_Losses", loss_str)
        step_losses = {}
        if isinstance(loss, dict):
            step_losses.update(
                {f"Losses/{phase}_{key}_{k}": v for k, v in loss.items()}
            )
            loss = self._log_loss_detailed_and_return_core_loss(
                loss, loss_log_str, self.steps[phase]
            )

        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(loss_log_str, loss, self.steps[phase])

        self.steps[phase] += 1

        ret_tuple = {loss_str: loss}, batch_size, step_losses
        if phase not in self.meters:
            return ret_tuple

        meters_dict = self._find_meter(phase, key)
        if meters_dict is None:
            return ret_tuple

        for _, meter in meters_dict.items():
            meter.update(
                find_stages=find_stages,
                find_metadatas=batch.find_metadatas,
                model=model,
                batch=batch,
                key=key,
            )

        if isinstance(find_stages, SAM3Output):
            for fs in find_stages:
                for k in list(fs.keys()):
                    del fs[k]

        return ret_tuple


class ProjectMetricLogger:
    """Logger wrapper that forwards only the configured metrics, under their configured names."""
    def __init__(self, logger, metric_names: dict[str, str] | None) -> None:
        self.logger = logger
        self.metric_names = metric_names or {}

    def log_dict(self, payload: dict[str, Any], step: int) -> None:
        """Log the configured, numeric entries of `payload` under their configured names."""
        if not self.metric_names:
            self.logger.log_dict(payload, step)
            return
        filtered = {
            self.metric_names[key]: value
            for key, value in payload.items()
            if key in self.metric_names and _as_float(value) is not None
        }
        if filtered:
            self.logger.log_dict(filtered, step)

    def log(self, name: str, data: Any, step: int) -> None:
        """Log one metric if it is configured and numeric."""
        if not self.metric_names:
            self.logger.log(name, data, step)
            return
        if name in self.metric_names and _as_float(data) is not None:
            self.logger.log(self.metric_names[name], data, step)

    def log_hparams(self, hparams: dict[str, Any], meters: dict[str, Any]) -> None:
        """Forward hyperparameters and meters to the wrapped logger unchanged."""
        self.logger.log_hparams(hparams, meters)


def install_tqdm_progress_meter() -> None:
    """Replace SAM3's progress meter with `TqdmProgressMeter`; repeated calls have no further effect."""
    global _INSTALLED, _ORIGINAL_PROGRESS_METER
    if _INSTALLED:
        return
    import sam3.train.trainer as trainer_module
    import sam3.train.utils.train_utils as train_utils_module

    _ORIGINAL_PROGRESS_METER = trainer_module.ProgressMeter
    trainer_module.ProgressMeter = TqdmProgressMeter
    train_utils_module.ProgressMeter = TqdmProgressMeter
    _INSTALLED = True


def _is_primary_rank() -> bool:
    try:
        from sam3.train.utils.distributed import get_rank

        return get_rank() == 0
    except Exception:
        return True


def _short_meter_name(name: str) -> str:
    normalized = re.sub(r"\s+", "_", name.strip().lower())
    replacements = {
        "batch_time": "batch_s",
        "data_time": "data_s",
        "mem_(gb)": "mem_gb",
        "time_elapsed": "time",
        "losses/train_all_loss": "loss",
        "losses/train_default_loss": "loss/train_default",
        "losses/val_oamtcd_loss": "loss",
        "losses/val_all_loss": "loss",
        "losses/val_default_loss": "loss/val_default",
    }
    return replacements.get(normalized, normalized)


def _epoch_index_from_prefix(prefix: str) -> int | None:
    match = re.search(r"\[(\d+)\]", prefix)
    if not match:
        return None
    return int(match.group(1))


def _format_seconds(seconds: float) -> str:
    try:
        total = int(seconds)
    except Exception:
        logging.debug("Could not format elapsed seconds", exc_info=True)
        return str(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _refresh_output_artifacts(trainer: ProgressBarTrainer, final: bool) -> None:
    experiment_dir = Path(trainer.checkpoint_conf.save_dir).parent
    log_dir = Path(trainer.logging_conf.log_dir)
    train_stats = _read_jsonl(log_dir / "train_stats.json")
    val_stats = _read_jsonl(log_dir / "val_stats.json")
    best_stats = _read_jsonl(log_dir / "best_stats.json")

    latest_train = train_stats[-1] if train_stats else {}
    latest_val = val_stats[-1] if val_stats else {}
    if latest_train:
        atomic_write_text(
            experiment_dir / "latest_train_metrics.txt",
            _format_metrics_text(latest_train, title="Latest train metrics"),
        )
    if latest_val:
        atomic_write_text(
            experiment_dir / "final_metrics.txt",
            _format_metrics_text(latest_val, title="Latest validation metrics"),
        )
        _maybe_update_best_checkpoint(experiment_dir, latest_val, trainer)

    if final:
        _write_checkpoint_view(
            experiment_dir=experiment_dir,
            target_dir=experiment_dir / "final_model",
            metrics=latest_val or latest_train,
            label="final",
            trainer=trainer,
        )
        _export_model_weights_for_views(
            [experiment_dir / "best_checkpoint", experiment_dir / "final_model"]
        )

    _write_run_summary(
        experiment_dir=experiment_dir,
        train_stats=train_stats,
        val_stats=val_stats,
        best_stats=best_stats,
        trainer=trainer,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logging.warning("Could not parse JSONL row in %s", path)
    return rows


def _maybe_update_best_checkpoint(
    experiment_dir: Path, latest_val: dict[str, Any], trainer: ProgressBarTrainer | None = None
) -> None:
    selected = _select_primary_metric(latest_val)
    if selected is None:
        return
    metric_key, metric_value, mode = selected
    best_metric_path = experiment_dir / "best_checkpoint" / "best_metric.json"
    previous_value = None
    if best_metric_path.exists():
        try:
            previous = json.loads(best_metric_path.read_text())
            previous_value = previous.get("value")
            previous_key = previous.get("key")
        except json.JSONDecodeError:
            previous_value = None
            previous_key = None

    improved = previous_value is None
    if previous_value is not None and previous_key != metric_key:
        improved = True
    if previous_value is not None:
        if previous_key == metric_key:
            improved = metric_value < previous_value if mode == "min" else metric_value > previous_value
    if not improved:
        return

    _write_checkpoint_view(
        experiment_dir=experiment_dir,
        target_dir=experiment_dir / "best_checkpoint",
        metrics=latest_val,
        label="best",
        trainer=trainer,
    )
    best_metric = {
        "name": _display_metric_name(metric_key),
        "key": metric_key,
        "value": metric_value,
        "mode": mode,
    }
    atomic_write_text(best_metric_path, json.dumps(best_metric, indent=2))
    atomic_write_text(
        experiment_dir / "best_metrics.txt",
        _format_metrics_text(latest_val, title="Best validation metrics", primary_metric=best_metric),
    )


def _select_primary_metric(metrics: dict[str, Any]) -> tuple[str, float, str] | None:
    for key in PROJECT_METRIC_ORDER:
        if not key.startswith(("Meters_train/val_", "Losses/val_", "Step_Losses/Losses/val_")):
            continue
        if key.startswith(("Losses/val_", "Step_Losses/Losses/val_")):
            continue
        value = _as_float(metrics.get(key))
        if value is None:
            continue
        return key, value, "max"
    for key, value in metrics.items():
        numeric = _as_float(value)
        if numeric is not None and key.startswith("Meters_train/val_"):
            return key, numeric, "max"
    for key in PROJECT_METRIC_ORDER:
        if not key.startswith(("Losses/val_", "Step_Losses/Losses/val_")):
            continue
        value = _as_float(metrics.get(key))
        if value is None or value == 0.0:
            continue
        return key, value, "min"
    return None


def _write_checkpoint_view(
    *,
    experiment_dir: Path,
    target_dir: Path,
    metrics: dict[str, Any],
    label: str,
    trainer: ProgressBarTrainer | None = None,
) -> None:
    """Record a named view (best/final) of the rolling checkpoint.

    A view is a hardlink to ``checkpoints/checkpoint.pt`` plus small provenance
    files. It deliberately does NOT re-serialize the model, optimizer or scaler:
    all three already live inside the linked ``checkpoint.pt``, nothing in the
    codebase reads the extracted copies, and writing them cost ~20 GB of disk
    and ~20 GB of network filesystem traffic on every validation that improved.
    """
    checkpoint_path = experiment_dir / "checkpoints" / "checkpoint.pt"
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics_files(target_dir, metrics, label)
    _copy_config_files(experiment_dir, target_dir)
    if not checkpoint_path.exists():
        return

    _snapshot_checkpoint(checkpoint_path, target_dir / "checkpoint.pt")
    if trainer is not None:
        try:
            atomic_write_text(
                target_dir / "trainer_state.json",
                json.dumps(_trainer_state_snapshot(trainer), indent=2),
            )
        except Exception:
            logging.warning("Could not write trainer state for %s", target_dir, exc_info=True)


def _snapshot_checkpoint(source: Path, target: Path) -> None:
    """Point ``target`` at the exact bytes ``source`` holds right now.

    Uses a hardlink rather than a copy. ``Trainer._save_checkpoint`` writes the
    next epoch by saving to ``checkpoint.pt.tmp``, unlinking ``checkpoint.pt``
    and renaming over it. Unlinking drops a directory entry, not the inode, so a
    hardlink taken earlier keeps the epoch it was taken at. That is exactly
    snapshot semantics, at zero cost in space or IO.

    Falls back to a real copy if the filesystem refuses to link.
    """
    tmp = target.parent / f"{target.name}.tmp"
    try:
        if tmp.exists():
            tmp.unlink()
        try:
            os.link(source, tmp)
        except OSError:
            logging.warning(
                "Could not hardlink %s -> %s; falling back to a full copy.", source, tmp,
                exc_info=True,
            )
            shutil.copy2(source, tmp)
        os.replace(tmp, target)
    except Exception:
        logging.warning("Could not snapshot checkpoint into %s", target, exc_info=True)
        with contextlib.suppress(OSError):
            tmp.unlink()


def _export_model_weights_for_views(view_dirs: list[Path]) -> None:
    """Write the inference-only weights for each view, once, at end of run.

    ``checkpoint.pt`` is ~10 GB and roughly two thirds of that is optimizer state
    that is dead weight once the run finishes. ``pytorch_model.bin`` is the ~3 GB
    model-only export: small enough to copy off the cluster, and accepted
    directly by ``training_pipeline._load_trainer_checkpoint``, which unwraps a
    ``model`` key if present and otherwise treats the file as a bare state dict.

    Views whose checkpoints are the same inode share one export via a hardlink,
    so a run whose best epoch is also its last serializes the weights only once.
    """
    exported: dict[int, Path] = {}
    for view_dir in view_dirs:
        checkpoint_path = view_dir / "checkpoint.pt"
        if not checkpoint_path.exists():
            continue
        target = view_dir / "pytorch_model.bin"
        try:
            inode = checkpoint_path.stat().st_ino
        except OSError:
            inode = -1
        previous = exported.get(inode) if inode != -1 else None
        if previous is not None:
            _snapshot_checkpoint(previous, target)
            continue
        if _export_model_weights(checkpoint_path, target) and inode != -1:
            exported[inode] = target


def _export_model_weights(checkpoint_path: Path, target: Path) -> bool:
    """Serialize only ``checkpoint["model"]`` to ``target``. True if written."""
    tmp = target.parent / f"{target.name}.tmp"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            return False
        torch.save(checkpoint["model"], tmp)
        os.replace(tmp, target)
        return True
    except Exception:
        logging.warning("Could not export model weights to %s", target, exc_info=True)
        with contextlib.suppress(OSError):
            tmp.unlink()
        return False


def _trainer_state_snapshot(trainer: ProgressBarTrainer) -> dict[str, Any]:
    """Provenance for a checkpoint view, read from the live trainer.

    Taken from the trainer rather than by loading the checkpoint back off disk,
    which would pull ~10 GB into memory purely to copy four scalars.
    """
    state: dict[str, Any] = {}
    for name in ("epoch", "steps", "best_meter_values"):
        value = getattr(trainer, name, None)
        if value is not None:
            state[name] = _json_safe(value)
    meter = getattr(trainer, "time_elapsed_meter", None)
    if meter is not None:
        state["time_elapsed"] = _as_float(getattr(meter, "val", None))
    return state


def _json_safe(value: Any) -> Any:
    """Coerce trainer state into something json.dumps accepts.

    ``steps`` is keyed by the Phase enum and meter values may be tensors, so
    neither survives json.dumps untouched.
    """
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    numeric = _as_float(value)
    return numeric if numeric is not None else str(value)


def _write_metrics_files(target_dir: Path, metrics: dict[str, Any], label: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    metric_title = f"{label.capitalize()} metrics"
    atomic_write_text(target_dir / f"{label}_metrics.txt", _format_metrics_text(metrics, title=metric_title))
    atomic_write_text(target_dir / f"{label}_metrics.json", json.dumps(metrics, indent=2, sort_keys=True))


def _copy_config_files(experiment_dir: Path, target_dir: Path) -> None:
    for source_name, target_name in [
        ("config/generated_sam3_config.yaml", "config.json"),
        ("config.yaml", "config.yaml"),
        ("config_resolved.yaml", "config_resolved.yaml"),
        ("config_snapshot.py", "config_snapshot.py"),
    ]:
        source = experiment_dir / source_name
        if source.exists():
            shutil.copy2(source, target_dir / target_name)


def _write_run_summary(
    *,
    experiment_dir: Path,
    train_stats: list[dict[str, Any]],
    val_stats: list[dict[str, Any]],
    best_stats: list[dict[str, Any]],
    trainer: ProgressBarTrainer,
) -> None:
    best_metric_path = experiment_dir / "best_checkpoint" / "best_metric.json"
    best_metric = {}
    if best_metric_path.exists():
        try:
            best_metric = json.loads(best_metric_path.read_text())
        except json.JSONDecodeError:
            best_metric = {}

    summary = {
        "experiment_dir": str(experiment_dir),
        "max_epochs": trainer.max_epochs,
        "val_epoch_freq": trainer.val_epoch_freq,
        "train_epochs_recorded": len(train_stats),
        "val_runs_recorded": len(val_stats),
        "best_stats_rows": len(best_stats),
        "tensorboard_dir": str(experiment_dir / "tensorboard"),
        "checkpoint": str(experiment_dir / "checkpoints" / "checkpoint.pt"),
        "best_metric": best_metric,
    }
    atomic_write_text(experiment_dir / "run_summary.json", json.dumps(summary, indent=2))

    lines = [
        "# SAM3 Training Summary",
        "",
        f"- Experiment dir: `{experiment_dir}`",
        f"- Max epochs: `{trainer.max_epochs}`",
        f"- Validation cadence: every `{trainer.val_epoch_freq}` completed epoch(s), plus final validation",
        f"- TensorBoard: `{experiment_dir / 'tensorboard'}`",
        f"- Latest checkpoint: `{experiment_dir / 'checkpoints' / 'checkpoint.pt'}`",
        "",
    ]
    if best_metric:
        lines.extend(
            [
                "## Best Checkpoint",
                "",
                f"- Metric: `{best_metric.get('name')}`",
                f"- Value: `{best_metric.get('value')}`",
                f"- Folder: `{experiment_dir / 'best_checkpoint'}`",
                "",
            ]
        )
    if train_stats:
        lines.extend(["## Latest Train Metrics", "", "```text", _format_metrics_text(train_stats[-1]), "```", ""])
    if val_stats:
        lines.extend(["## Latest Validation Metrics", "", "```text", _format_metrics_text(val_stats[-1]), "```", ""])
    atomic_write_text(experiment_dir / "run_summary.md", "\n".join(lines))


def _format_metrics_text(
    metrics: dict[str, Any],
    *,
    title: str | None = None,
    primary_metric: dict[str, Any] | None = None,
) -> str:
    lines = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
    if primary_metric:
        lines.append(
            f"primary_metric = {primary_metric['name']} ({primary_metric['mode']}) = {primary_metric['value']:.6g}"
        )
        lines.append("")

    for key in _ordered_metric_keys(metrics):
        value = _as_float(metrics.get(key))
        if value is None:
            continue
        lines.append(f"{_display_metric_name(key)} = {value:.6g}")
    if not lines:
        lines.append("No metrics recorded yet.")
    return "\n".join(lines) + "\n"


def _ordered_metric_keys(metrics: dict[str, Any]) -> list[str]:
    preferred = ["Trainer/epoch", "Trainer/steps_train", *PROJECT_METRIC_ORDER]
    seen = set()
    ordered = []
    for key in preferred:
        if key in metrics:
            ordered.append(key)
            seen.add(key)
    for key in sorted(metrics):
        if key in seen:
            continue
        if key.startswith(("Losses/", "Meters_train/val_", "Trainer/")):
            ordered.append(key)
    return ordered


def _display_metric_name(key: str) -> str:
    if key in TB_PROJECT_METRICS:
        return TB_PROJECT_METRICS[key]
    replacements = {
        "Trainer/epoch": "epoch",
        "Trainer/steps_train": "train_steps",
        "Losses/train_all_loss": "train_loss_epoch",
        "Losses/val_oamtcd_loss": "val_loss",
        "Step_Losses/Losses/val_oamtcd_loss": "val_loss_step",
        "Meters_train/val_oamtcd/detection/": "val_",
        "coco_eval_segm_": "segm_",
        "cgF1_eval_segm_": "cgf1_",
    }
    name = key
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.replace("/", "_")


def _as_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _any_nonfinite(tensors: Any) -> bool:
    """True if any tensor holds a non-finite value, at the cost of one sync.

    The per-tensor form of this check called ``.item()`` once per tensor, and on
    an 841 M-parameter model spread over 1,134 tensors that is 1,134 GPU-to-CPU
    stalls to answer a single yes/no question. Reducing on device and
    synchronizing once gives the identical answer. The detailed per-tensor
    breakdown is still available from the ``_summarize_nonfinite_*`` helpers, but
    it is only worth paying for once something has actually gone wrong.
    """
    flags = [torch.isfinite(tensor).all() for tensor in tensors if tensor is not None]
    if not flags:
        return False
    return not bool(torch.stack(flags).all().item())


def _model_gradients(model: torch.nn.Module) -> Any:
    return (param.grad for _, param in unwrap_ddp_if_wrapped(model).named_parameters())


def _model_parameters(model: torch.nn.Module) -> Any:
    return (param.detach() for _, param in unwrap_ddp_if_wrapped(model).named_parameters())


def _summarize_nonfinite_losses(losses: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in losses.items():
        description = _describe_nonfinite_value(value)
        if description is not None:
            summary[key] = description
    return summary


def _describe_nonfinite_value(value: Any) -> Any | None:
    try:
        if torch.is_tensor(value):
            detached = value.detach()
            finite = torch.isfinite(detached)
            if bool(finite.all().item()):
                return None
            if detached.numel() == 1:
                return float(detached.cpu().item())
            return {
                "shape": tuple(detached.shape),
                "nonfinite": int((~finite).sum().item()),
                "total": int(detached.numel()),
            }
        numeric = float(value)
    except Exception:
        return "unavailable"
    if math.isfinite(numeric):
        return None
    return numeric


def _force_grad_scaler_skip(optimizer: torch.optim.Optimizer) -> bool:
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            with torch.no_grad():
                if grad.ndim == 0:
                    grad.fill_(float("inf"))
                else:
                    grad.detach()[tuple(0 for _ in range(grad.ndim))] = float("inf")
            return True

    for group in optimizer.param_groups:
        for param in group.get("params", []):
            if not getattr(param, "requires_grad", False):
                continue
            with torch.no_grad():
                param.grad = torch.zeros_like(param, memory_format=torch.preserve_format)
                if param.grad.ndim == 0:
                    param.grad.fill_(float("inf"))
                else:
                    param.grad[tuple(0 for _ in range(param.grad.ndim))] = float("inf")
            return True
    return False


def _summarize_nonfinite_gradients(model: torch.nn.Module, limit: int = 8) -> list[dict[str, Any]]:
    return _summarize_nonfinite_named_tensors(
        ((name, param.grad) for name, param in unwrap_ddp_if_wrapped(model).named_parameters()),
        limit=limit,
    )


def _summarize_nonfinite_parameters(model: torch.nn.Module, limit: int = 8) -> list[dict[str, Any]]:
    return _summarize_nonfinite_named_tensors(
        ((name, param.detach()) for name, param in unwrap_ddp_if_wrapped(model).named_parameters()),
        limit=limit,
    )


def _summarize_nonfinite_optimizer_state(
    optimizer: torch.optim.Optimizer,
    limit: int = 8,
) -> list[dict[str, Any]]:
    named_tensors = []
    for param_index, (param, state) in enumerate(optimizer.state.items()):
        for state_name, value in state.items():
            if torch.is_tensor(value):
                named_tensors.append((f"param_{param_index}.{state_name}", value))
    return _summarize_nonfinite_named_tensors(named_tensors, limit=limit)


def _summarize_nonfinite_named_tensors(
    named_tensors: Any,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    bad = []
    prefix_counts: dict[str, int] = {}
    bad_tensor_count = 0
    bad_element_count = 0
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        if bool(finite.all().item()):
            continue
        finite_values = detached[finite]
        nonfinite = int((~finite).sum().item())
        bad_tensor_count += 1
        bad_element_count += nonfinite
        prefix = ".".join(name.split(".")[:3])
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        if len(bad) < limit:
            bad.append(
                {
                    "name": name,
                    "shape": tuple(detached.shape),
                    "nonfinite": nonfinite,
                    "total": int(detached.numel()),
                    "finite_min": float(finite_values.min().cpu().item()) if finite_values.numel() else None,
                    "finite_max": float(finite_values.max().cpu().item()) if finite_values.numel() else None,
                }
            )
    if bad_tensor_count > limit:
        bad.append(
            {
                "name": "<summary>",
                "bad_tensors": bad_tensor_count,
                "nonfinite": bad_element_count,
                "prefix_counts": sorted(
                    prefix_counts.items(), key=lambda item: item[1], reverse=True
                )[:8],
            }
        )
    return bad


def _log_nonfinite_model_parameters(model: torch.nn.Module) -> None:
    if getattr(_log_nonfinite_model_parameters, "_logged", False):
        return
    setattr(_log_nonfinite_model_parameters, "_logged", True)
    try:
        raw_model = unwrap_ddp_if_wrapped(model)
        bad = []
        for name, param in raw_model.named_parameters():
            data = param.detach()
            finite = torch.isfinite(data)
            if bool(finite.all().item()):
                continue
            bad.append(
                {
                    "name": name,
                    "shape": tuple(data.shape),
                    "nonfinite": int((~finite).sum().item()),
                    "total": int(data.numel()),
                }
            )
            if len(bad) >= 8:
                break
        if bad:
            logging.error("Non-finite model parameters detected: %s", bad)
        else:
            logging.warning("No non-finite model parameters detected at first non-finite loss.")
    except Exception:
        logging.warning("Failed to inspect model parameters for non-finite values.", exc_info=True)


