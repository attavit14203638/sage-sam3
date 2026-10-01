"""Self-tests for the bookkeeping paths in progress_trainer.py.

Two concerns, both about work the trainer does around training rather than
training itself: checkpoint views, and the per-iteration non-finite guards.


A "view" is best_checkpoint/ or final_model/. Each must record the exact bytes
the rolling checkpoint held at the moment the view was taken, while costing no
extra disk and no extra network filesystem traffic.

Correctness rests on one non-obvious invariant of the upstream trainer:
``Trainer._save_checkpoint`` writes the next epoch to ``checkpoint.pt.tmp``,
UNLINKS ``checkpoint.pt``, then renames the temp file over it. Unlinking drops a
directory entry, not the inode, so a hardlink taken earlier still resolves to the
bytes of the epoch it was taken at. If upstream ever switched to writing the file
in place, hardlinked views would silently start mutating with every epoch and
"best" would become a lie. test_view_survives_rolling_update is the tripwire for
that regression.

The functions under test are pure filesystem operations, so the suite stubs the
sam3 imports that progress_trainer.py needs at module scope. That keeps the
tripwire runnable on a laptop without a full training install; a guard that only
ran on the cluster would not catch the regression while the code was being
edited.

Run: python Core/test_progress_trainer.py
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import types
from pathlib import Path

# torch and the system libomp both ship an OpenMP runtime on macOS; without this
# the interpreter aborts on import before a single check runs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Tests live in Codebase/Tests/; the modules under test live in Codebase/Core/.
CORE_DIR = Path(__file__).resolve().parent.parent / "Core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

FAILURES: list[str] = []


def _install_sam3_stubs() -> None:
    """Register the minimum sam3 surface progress_trainer.py imports.

    Nothing under test touches any of it. The stubs exist only so the module
    body can execute; if a test ever needed real training behaviour it should
    import the real package instead of extending these.
    """
    def module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    module("sam3")
    module("sam3.model")
    module("sam3.model.utils")
    module("sam3.train")
    module("sam3.train.utils")

    model_misc = module("sam3.model.model_misc")
    model_misc.SAM3Output = type("SAM3Output", (), {})

    misc = module("sam3.model.utils.misc")
    misc.copy_data_to_device = lambda data, device: data

    trainer_mod = module("sam3.train.trainer")
    trainer_mod.Trainer = type("Trainer", (), {})
    trainer_mod.unwrap_ddp_if_wrapped = lambda model: model

    train_utils = module("sam3.train.utils.train_utils")
    train_utils.AverageMeter = type("AverageMeter", (), {})
    train_utils.MemMeter = type("MemMeter", (), {})
    train_utils.get_amp_type = lambda dtype=None: None
    train_utils.Phase = type("Phase", (), {"TRAIN": "train", "VAL": "val"})
    train_utils.human_readable_time = lambda s: f"{s:.0f}s"

    distributed = module("sam3.train.utils.distributed")
    distributed.get_rank = lambda: 0


def _load_progress_trainer():
    try:
        import progress_trainer
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("sam3"):
            raise
        _install_sam3_stubs()
        import progress_trainer
    return progress_trainer


pt = _load_progress_trainer()


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok  {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL {label}" + (f" -- {detail}" if detail else ""))


def _rewrite_like_upstream(path: Path, payload: bytes) -> None:
    """Replicate Trainer._save_checkpoint: write .tmp, unlink, rename."""
    tmp = path.parent / f"{path.name}.tmp"
    tmp.write_bytes(payload)
    if path.exists():
        path.unlink()
    os.replace(tmp, path)


def test_snapshot_is_a_hardlink() -> None:
    print("\n[snapshot uses a hardlink]")
    _snapshot_checkpoint = pt._snapshot_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = root / "checkpoint.pt"
        source.write_bytes(b"epoch-0")
        target = root / "view" / "checkpoint.pt"
        target.parent.mkdir()

        _snapshot_checkpoint(source, target)

        check("view exists", target.exists())
        check("view has the source bytes", target.read_bytes() == b"epoch-0")
        check(
            "view shares the source inode, so it consumes no extra space",
            target.stat().st_ino == source.stat().st_ino,
            f"{target.stat().st_ino} vs {source.stat().st_ino}",
        )
        check("link count reflects two names for one inode",
              source.stat().st_nlink == 2, f"got {source.stat().st_nlink}")
        check("no temp file is left behind",
              not (target.parent / "checkpoint.pt.tmp").exists())


def test_view_survives_rolling_update() -> None:
    """The invariant the whole design depends on."""
    print("\n[view is a true snapshot]")
    _snapshot_checkpoint = pt._snapshot_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = root / "checkpoint.pt"
        source.write_bytes(b"epoch-26-best")
        target = root / "best_checkpoint" / "checkpoint.pt"
        target.parent.mkdir()

        _snapshot_checkpoint(source, target)
        _rewrite_like_upstream(source, b"epoch-29-latest")

        check(
            "rolling checkpoint advanced to the new epoch",
            source.read_bytes() == b"epoch-29-latest",
        )
        check(
            "the view still holds the epoch it was taken at",
            target.read_bytes() == b"epoch-26-best",
            f"got {target.read_bytes()!r} -- upstream may no longer unlink before rename",
        )
        check(
            "the two are now distinct inodes",
            target.stat().st_ino != source.stat().st_ino,
        )


def test_snapshot_is_idempotent_and_atomic() -> None:
    print("\n[re-taking a view]")
    _snapshot_checkpoint = pt._snapshot_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = root / "checkpoint.pt"
        target = root / "view" / "checkpoint.pt"
        target.parent.mkdir()

        source.write_bytes(b"epoch-3")
        _snapshot_checkpoint(source, target)
        _rewrite_like_upstream(source, b"epoch-6")
        _snapshot_checkpoint(source, target)

        check("re-taking the view picks up the newer epoch",
              target.read_bytes() == b"epoch-6")
        check("re-taken view again shares the source inode",
              target.stat().st_ino == source.stat().st_ino)
        check("no temp file is left behind",
              not (target.parent / "checkpoint.pt.tmp").exists())


def test_view_does_not_reserialize_during_training() -> None:
    """Per-validation cost must be a link plus a few KB, never a re-serialize."""
    print("\n[no re-serialization during training]")
    _write_checkpoint_view = pt._write_checkpoint_view

    with tempfile.TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir)
        (experiment_dir / "checkpoints").mkdir()
        (experiment_dir / "checkpoints" / "checkpoint.pt").write_bytes(b"weights")
        target_dir = experiment_dir / "best_checkpoint"

        _write_checkpoint_view(
            experiment_dir=experiment_dir,
            target_dir=target_dir,
            metrics={"Meters_train/val_oamtcd/detection/coco_eval_segm_AP": 0.42},
            label="best",
            trainer=None,
        )

        written = {p.name for p in target_dir.iterdir()}
        check("the checkpoint itself is recorded", "checkpoint.pt" in written)
        check("metrics are recorded", "best_metrics.json" in written)
        for redundant in ("pytorch_model.bin", "optimizer.pt", "scaler.pt"):
            check(
                f"{redundant} is not re-serialized on every improving validation",
                redundant not in written,
                f"found {redundant}",
            )


def _make_checkpoint(path: Path, tag: str) -> None:
    import torch

    torch.save(
        {
            "model": {"weight": torch.tensor([1.0, 2.0]), "tag": tag},
            "optimizer": {"state": list(range(64))},
            "epoch": 26,
        },
        path,
    )


def test_model_weights_exported_once_at_end() -> None:
    """The ~3 GB model-only file is what actually gets archived off the cluster."""
    print("\n[end-of-run model export]")
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        best = root / "best_checkpoint"
        best.mkdir()
        _make_checkpoint(best / "checkpoint.pt", "best-epoch-26")

        pt._export_model_weights_for_views([best])

        exported = best / "pytorch_model.bin"
        check("the model-only export exists", exported.exists())
        check("no temp file is left behind",
              not (best / "pytorch_model.bin.tmp").exists())

        payload = torch.load(exported, map_location="cpu", weights_only=False)
        check("the export is a bare state dict, not the wrapped checkpoint",
              isinstance(payload, dict) and "model" not in payload)
        check("it carries the model weights", payload.get("tag") == "best-epoch-26")
        check("optimizer state is excluded, which is the point of the export",
              "optimizer" not in payload)
        check("the export is smaller than the checkpoint it came from",
              exported.stat().st_size < (best / "checkpoint.pt").stat().st_size)


def test_identical_views_share_one_export() -> None:
    print("\n[export deduplication]")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        best, final = root / "best_checkpoint", root / "final_model"
        best.mkdir()
        final.mkdir()
        _make_checkpoint(best / "checkpoint.pt", "same-epoch")
        os.link(best / "checkpoint.pt", final / "checkpoint.pt")

        pt._export_model_weights_for_views([best, final])

        a, b = best / "pytorch_model.bin", final / "pytorch_model.bin"
        check("both views have an export", a.exists() and b.exists())
        check(
            "a run whose best epoch is also its last serializes weights once",
            a.stat().st_ino == b.stat().st_ino,
            f"{a.stat().st_ino} vs {b.stat().st_ino}",
        )


def test_distinct_views_get_distinct_exports() -> None:
    print("\n[distinct epochs are not conflated]")
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        best, final = root / "best_checkpoint", root / "final_model"
        best.mkdir()
        final.mkdir()
        _make_checkpoint(best / "checkpoint.pt", "best-epoch-26")
        _make_checkpoint(final / "checkpoint.pt", "final-epoch-29")

        pt._export_model_weights_for_views([best, final])

        a = torch.load(best / "pytorch_model.bin", map_location="cpu", weights_only=False)
        b = torch.load(final / "pytorch_model.bin", map_location="cpu", weights_only=False)
        check("best export holds the best epoch", a.get("tag") == "best-epoch-26")
        check("final export holds the final epoch", b.get("tag") == "final-epoch-29")


def test_fused_nonfinite_detection() -> None:
    """One sync must give the same answer the per-tensor scan gave."""
    print("\n[fused non-finite detection]")
    import torch

    _any_nonfinite = pt._any_nonfinite

    clean = [torch.ones(4), torch.zeros(2, 3), torch.tensor([-1.5])]
    check("all-finite reports clean", _any_nonfinite(clean) is False)
    check("empty input reports clean", _any_nonfinite([]) is False)
    check("all-None input reports clean", _any_nonfinite([None, None]) is False)
    check("None entries are skipped, not treated as bad",
          _any_nonfinite([torch.ones(2), None]) is False)

    for label, bad in (
        ("nan", float("nan")),
        ("+inf", float("inf")),
        ("-inf", float("-inf")),
    ):
        tainted = [torch.ones(4), torch.tensor([1.0, bad, 3.0]), torch.zeros(2)]
        check(f"{label} anywhere in the list is detected", _any_nonfinite(tainted) is True)

    buried = [torch.ones(64) for _ in range(50)]
    buried[37] = torch.full((64,), float("nan"))
    check("a single bad tensor among many is detected", _any_nonfinite(buried) is True)

    scalar = [torch.tensor(float("inf"))]
    check("zero-dim tensors are handled", _any_nonfinite(scalar) is True)

    check("generators are accepted, not just lists",
          _any_nonfinite(t for t in tainted) is True)


def test_fused_agrees_with_detailed_scan() -> None:
    """The cheap check and the diagnostic scan must never disagree."""
    print("\n[fused check agrees with the detailed scan]")
    import torch

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(4, 4)
            self.b = torch.nn.Linear(4, 2)

    model = _Model()
    fused_clean = pt._any_nonfinite(pt._model_parameters(model))
    detailed_clean = pt._summarize_nonfinite_parameters(model)
    check("both agree the fresh model is clean",
          fused_clean is False and detailed_clean == [])

    with torch.no_grad():
        model.b.bias[1] = float("nan")

    fused_bad = pt._any_nonfinite(pt._model_parameters(model))
    detailed_bad = pt._summarize_nonfinite_parameters(model)
    check("both agree the tainted model is bad",
          fused_bad is True and len(detailed_bad) == 1)
    check("the detailed scan still names the offending tensor",
          detailed_bad and detailed_bad[0]["name"] == "b.bias",
          f"got {detailed_bad}")

    check("gradients are clean before backward",
          pt._any_nonfinite(pt._model_gradients(model)) is False)


def test_initial_parameter_check_runs_once() -> None:
    """The pre-step scan is kept only to cover the first step; it must not repeat."""
    print("\n[initial parameter check]")
    import torch

    calls = {"count": 0}
    real_model_parameters = pt._model_parameters

    def counting_model_parameters(model):
        calls["count"] += 1
        return real_model_parameters(model)

    class _Fake:
        model = torch.nn.Linear(3, 3)
        _verify_initial_parameters_finite = pt.ProgressBarTrainer._verify_initial_parameters_finite

    pt._model_parameters = counting_model_parameters
    try:
        trainer = _Fake()
        for _ in range(5):
            trainer._verify_initial_parameters_finite()
        check("the scan runs exactly once across five iterations",
              calls["count"] == 1, f"ran {calls['count']} times")

        tainted = _Fake()
        tainted.model = torch.nn.Linear(3, 3)
        with torch.no_grad():
            tainted.model.weight[0][0] = float("inf")
        raised = False
        try:
            tainted._verify_initial_parameters_finite()
        except FloatingPointError:
            raised = True
        check("non-finite resumed or initialized weights still abort", raised)
    finally:
        pt._model_parameters = real_model_parameters


def test_trainer_state_is_json_safe() -> None:
    print("\n[trainer state serialization]")
    import json

    _json_safe, _trainer_state_snapshot = pt._json_safe, pt._trainer_state_snapshot

    class _Meter:
        val = 12.5

    class _Phase:
        def __init__(self, name: str) -> None:
            self.name = name

        def __str__(self) -> str:
            return f"Phase.{self.name}"

    class _FakeTrainer:
        epoch = 26
        steps = {_Phase("TRAIN"): 5040, _Phase("VAL"): 252}
        best_meter_values = {"val_cgF1": 0.2610}
        time_elapsed_meter = _Meter()

    state = _trainer_state_snapshot(_FakeTrainer())
    check("epoch is captured", state.get("epoch") == 26)
    check("elapsed time is captured", state.get("time_elapsed") == 12.5)
    check("enum-keyed steps are stringified",
          set(state["steps"]) == {"Phase.TRAIN", "Phase.VAL"}, f"got {state['steps']}")
    try:
        json.dumps(state)
        check("the whole state survives json.dumps", True)
    except TypeError as exc:
        check("the whole state survives json.dumps", False, str(exc))

    check("unknown objects degrade to a string rather than raising",
          isinstance(_json_safe(object()), str))


def test_log_timers_tolerates_disabled_validation() -> None:
    """val_epoch_freq=0 (validation disabled) must not raise.

    Upstream Trainer._log_timers estimates remaining time with
    `n % self.val_epoch_freq`, a ZeroDivisionError at 0. It runs before
    _log_meters_and_save_best_ckpts, so an exception there would also discard the
    epoch's train_stats.json.
    """
    print("\n[_log_timers with validation disabled]")
    from progress_trainer import ProgressBarTrainer

    trainer = ProgressBarTrainer.__new__(ProgressBarTrainer)
    trainer.val_epoch_freq = 0
    trainer.max_epochs = 1
    trainer.epoch = 0
    trainer.est_epoch_time = {pt.Phase.TRAIN: 10.0, pt.Phase.VAL: 5.0}

    class _Logger:
        def __init__(self) -> None:
            self.logged: list[tuple[str, float, int]] = []

        def log(self, key, value, step) -> None:
            self.logged.append((key, value, step))

    class _Meter:
        name = "Time"
        val = 12.0

    trainer.logger = _Logger()
    trainer.time_elapsed_meter = _Meter()
    trainer.steps = {pt.Phase.TRAIN: 4}

    try:
        trainer._log_timers(pt.Phase.TRAIN)
        crashed = None
    except Exception as exc:
        crashed = exc
    check("no ZeroDivisionError when val_epoch_freq=0", crashed is None, repr(crashed))
    check(
        "the elapsed-time log entry is still written",
        len(trainer.logger.logged) == 1 and trainer.logger.logged[0][1] == 12.0,
        f"got {trainer.logger.logged}",
    )


def test_accumulation_divisor_matches_one_big_batch() -> None:
    """Accumulating N microbatches must give the gradient of one N x batch.

    This is the whole justification for capping the microbatch on the V100 and
    buying the effective batch back with accumulation. Two regimes, because SAM3
    multiplies the loss by sqrt(microbatch) when scale_by_find_batch_size is on
    (`sam3_loss.py:192-195`):

      * plain loss           -> divide by N
      * sqrt-scaled loss     -> divide by sqrt(N)

    Using N in the sqrt-scaled regime (which is the one this project runs) would
    halve the gradient at N=4 -- a silent, systematic deviation from the recipe
    rather than a crash.
    """
    print("\n[gradient accumulation equals one large batch]")
    import torch

    from progress_trainer import ProgressBarTrainer

    torch.manual_seed(0)
    data = torch.randn(8, 4)
    target = torch.randn(8, 1)

    def grad(chunks: int, divisor: float, sqrt_scaled: bool) -> torch.Tensor:
        torch.manual_seed(1)
        layer = torch.nn.Linear(4, 1)
        layer.zero_grad(set_to_none=True)
        for x, y in zip(data.chunk(chunks), target.chunk(chunks)):
            loss = torch.nn.functional.mse_loss(layer(x), y)
            if sqrt_scaled:
                loss = loss * (x.shape[0] ** 0.5)
            (loss / divisor).backward()
        return layer.weight.grad.clone()

    # --- plain regime: divisor is N ---
    single = grad(1, 1.0, sqrt_scaled=False)
    check(
        "plain loss: 4 microbatches / 4 match one batch of 8",
        torch.allclose(single, grad(4, 4.0, sqrt_scaled=False), atol=1e-6),
    )

    # --- sqrt-scaled regime (this project): divisor is sqrt(N) ---
    single_sqrt = grad(1, 1.0, sqrt_scaled=True)
    accum_sqrt = grad(4, 4.0**0.5, sqrt_scaled=True)
    check(
        "sqrt-scaled loss: 4 microbatches / sqrt(4) match one batch of 8",
        torch.allclose(single_sqrt, accum_sqrt, atol=1e-6),
        f"max abs diff {(single_sqrt - accum_sqrt).abs().max().item():.3e}",
    )
    wrong = grad(4, 4.0, sqrt_scaled=True)
    ratio = (wrong.norm() / single_sqrt.norm()).item()
    check(
        "dividing by N instead of sqrt(N) would halve the gradient (the trap)",
        0.45 < ratio < 0.55,
        f"ratio {ratio:.3f}",
    )

    # The V100 configuration in use: microbatch 1, accumulation 8.
    accum_8 = grad(8, 8.0**0.5, sqrt_scaled=True)
    check(
        "microbatch 1 x accum 8 / sqrt(8) matches one batch of 8 (the V100 setting)",
        torch.allclose(single_sqrt, accum_8, atol=1e-6),
        f"max abs diff {(single_sqrt - accum_8).abs().max().item():.3e}",
    )

    # --- the divisor helper picks the right regime ---
    trainer = ProgressBarTrainer.__new__(ProgressBarTrainer)

    def with_flag(value):
        module = types.SimpleNamespace(scale_by_find_batch_size=value)
        return {"all": module}

    trainer.loss = with_flag(True)
    check("helper returns sqrt(N) when the loss is sqrt-scaled",
          abs(trainer._accumulation_divisor(4) - 2.0) < 1e-9,
          f"got {trainer._accumulation_divisor(4)}")
    trainer.loss = with_flag(False)
    check("helper returns N when it is not",
          trainer._accumulation_divisor(4) == 4.0)
    trainer.loss = with_flag(True)
    check("helper is a no-op at accum=1", trainer._accumulation_divisor(1) == 1.0)
    trainer.loss = None
    check("helper tolerates a trainer with no loss configured",
          trainer._accumulation_divisor(4) == 4.0)

    # The real config shape: all/oamtcd are Sam3LossWrapper(True), default is a
    # DummyLoss that does not declare the flag at all. A missing attribute must
    # not read as "False" -- doing so aborted the first calibration launch.
    class _DummyLoss:
        """Stands in for sam3_loss.DummyLoss: no scale_by_find_batch_size."""

    trainer.loss = {
        "all": types.SimpleNamespace(scale_by_find_batch_size=True),
        "oamtcd": types.SimpleNamespace(scale_by_find_batch_size=True),
        "default": _DummyLoss(),
    }
    check(
        "a DummyLoss without the flag does not count as disagreement",
        abs(trainer._accumulation_divisor(4) - 2.0) < 1e-9,
        f"got {trainer._accumulation_divisor(4)} -- absence of the attribute must not vote",
    )

    trainer.loss = {
        "a": types.SimpleNamespace(scale_by_find_batch_size=True),
        "b": types.SimpleNamespace(scale_by_find_batch_size=False),
    }
    raised = None
    try:
        trainer._accumulation_divisor(4)
    except ValueError as exc:
        raised = exc
    check("two modules that DECLARE and disagree still raise", raised is not None)


def test_static_graph_accumulation_synchronizes_each_microbatch() -> None:
    print("\n[static-graph accumulation synchronization policy]")

    class Model:
        def __init__(self, static_graph):
            self.static_graph = static_graph
            self.no_sync_calls = 0

        def no_sync(self):
            self.no_sync_calls += 1
            return contextlib.nullcontext()

    static_model = Model(True)
    for index in range(4):
        with pt._ddp_accumulation_context(static_model, index, 4):
            pass
    check("static_graph synchronizes every accumulated microbatch", static_model.no_sync_calls == 0)

    dynamic_model = Model(False)
    for index in range(4):
        with pt._ddp_accumulation_context(dynamic_model, index, 4):
            pass
    check("non-static DDP retains no_sync on non-final microbatches", dynamic_model.no_sync_calls == 3)

    with pt._ddp_accumulation_context(object(), 0, 4):
        pass
    check("non-DDP accumulation uses a null context", True)


def test_explicit_static_graph_gradient_synchronization() -> None:
    print("\n[explicit static-graph accumulated-gradient synchronization]")
    import torch

    class Model:
        static_graph = True

        def __init__(self):
            self.layer = torch.nn.Linear(3, 2)
            for parameter in self.layer.parameters():
                parameter.grad = torch.ones_like(parameter)

        def named_parameters(self):
            return self.layer.named_parameters()

    model = Model()
    distributed = torch.distributed
    originals = {name: getattr(distributed, name) for name in ("is_available", "is_initialized", "get_world_size", "all_gather_object", "all_reduce")}
    all_reduce_calls = []
    try:
        distributed.is_available = lambda: True
        distributed.is_initialized = lambda: True
        distributed.get_world_size = lambda: 2
        distributed.all_gather_object = lambda output, value: output.__setitem__(slice(None), [value, value])

        def all_reduce(value):
            all_reduce_calls.append(value.numel())
            value.mul_(2)

        distributed.all_reduce = all_reduce
        applied = pt._synchronize_static_graph_accumulated_gradients(model, 4, bucket_bytes=16)
    finally:
        for name, value in originals.items():
            setattr(distributed, name, value)
    check("explicit synchronization is applied to static accumulated training", applied)
    check("gradient buckets are all-reduced", bool(all_reduce_calls))
    check("world-size averaging preserves already-synchronized gradients", all(torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in model.layer.parameters()))
    check("accumulation=1 needs no explicit synchronization", not pt._synchronize_static_graph_accumulated_gradients(model, 1))


def test_run_val_frees_gradients_and_cache_before_validating() -> None:
    """run_val must drop gradients AND empty the cache, both BEFORE validating.

    Val peaks at 27-28 GiB of 31.7 GiB on the V100 node against 21-23 GiB for
    training, and a run died inside the fourth val pass. The epoch's leftover
    gradients (~3.4 GB at 840 M fp32 params) are dead through the whole val pass
    because _run_step re-zeroes them at the top of the next train step, so
    releasing them is free -- and it must happen before empty_cache, which is
    what actually returns the memory to the driver.
    """
    print("\n[run_val frees gradients and cache before validating]")
    from progress_trainer import ProgressBarTrainer

    calls: list[str] = []

    class _Base:
        def run_val(self):
            calls.append("run_val")

    class _Optim:
        def zero_grad(self, set_to_none=False):
            calls.append(f"zero_grad(set_to_none={set_to_none})")

    trainer = ProgressBarTrainer.__new__(ProgressBarTrainer)
    trainer.optim = _Optim()

    import torch

    real_is_available = torch.cuda.is_available
    real_empty_cache = torch.cuda.empty_cache
    real_bases = ProgressBarTrainer.__bases__
    try:
        torch.cuda.is_available = lambda: True
        torch.cuda.empty_cache = lambda: calls.append("empty_cache")
        ProgressBarTrainer.__bases__ = (_Base,)
        pt._is_primary_rank = lambda: False
        trainer.run_val()

        check(
            "gradients are released with set_to_none, cache emptied, then val runs -- in that order",
            calls == ["zero_grad(set_to_none=True)", "empty_cache", "run_val"],
            f"got {calls}",
        )

        # An eval-only trainer has no optimizer; it must not crash.
        calls.clear()
        trainer_no_optim = ProgressBarTrainer.__new__(ProgressBarTrainer)
        trainer_no_optim.optim = None
        trainer_no_optim.run_val()
        check(
            "a trainer with no optimizer still validates",
            calls == ["empty_cache", "run_val"],
            f"got {calls}",
        )
    finally:
        torch.cuda.is_available = real_is_available
        torch.cuda.empty_cache = real_empty_cache
        ProgressBarTrainer.__bases__ = real_bases


def test_activation_checkpointing_is_bitwise_identical() -> None:
    """Activation checkpointing must preserve exact outputs and gradients.

    With the same model, data, and seed, outputs and gradients must be `torch.equal`, including
    through dropout because non-reentrant checkpointing preserves RNG state for recomputation.
    """
    import torch
    from torch import nn
    from torch.utils import checkpoint

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ln = nn.LayerNorm(d)
            self.fc = nn.Linear(d, d)
            self.drop = nn.Dropout(0.3)
        def forward(self, x):
            return x + self.drop(self.fc(self.ln(x)))

    class Net(nn.Module):
        def __init__(self, ckpt):
            super().__init__()
            self.blocks = nn.ModuleList(Block(64) for _ in range(8))
            self.ckpt = ckpt
        def forward(self, x):
            for b in self.blocks:
                x = checkpoint.checkpoint(b, x, use_reentrant=False) if self.ckpt else b(x)
            return x

    def grads(ckpt):
        torch.manual_seed(123)
        net = Net(ckpt)
        net.train()
        out = net(X)
        out.square().mean().backward()
        return out.detach(), torch.cat([p.grad.flatten() for p in net.parameters()])

    torch.manual_seed(42)
    X = torch.randn(4, 16, 64)
    out_a, g_a = grads(ckpt=False)
    out_b, g_b = grads(ckpt=True)
    check("checkpointed forward is bitwise identical", torch.equal(out_a, out_b))
    check("checkpointed backward is bitwise identical (dropout included)",
          torch.equal(g_a, g_b), f"max|d| = {(g_a - g_b).abs().max():.3e}")


def main() -> None:
    print("=" * 62)
    print("progress_trainer bookkeeping self-tests")
    print("=" * 62)

    test_snapshot_is_a_hardlink()
    test_view_survives_rolling_update()
    test_snapshot_is_idempotent_and_atomic()
    test_view_does_not_reserialize_during_training()
    test_model_weights_exported_once_at_end()
    test_identical_views_share_one_export()
    test_distinct_views_get_distinct_exports()
    test_trainer_state_is_json_safe()
    test_log_timers_tolerates_disabled_validation()
    test_fused_nonfinite_detection()
    test_fused_agrees_with_detailed_scan()
    test_initial_parameter_check_runs_once()
    test_accumulation_divisor_matches_one_big_batch()
    test_static_graph_accumulation_synchronizes_each_microbatch()
    test_explicit_static_graph_gradient_synchronization()
    test_run_val_frees_gradients_and_cache_before_validating()

    print()
    print("=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED")
        for name in FAILURES:
            print(f"  - {name}")
        print("=" * 62)
        raise SystemExit(1)
    print("ALL CHECKS PASSED")
    print("=" * 62)


if __name__ == "__main__":
    main()
