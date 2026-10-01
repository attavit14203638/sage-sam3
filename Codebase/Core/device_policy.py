"""Central device, precision, capacity, and inference-comparability policy.

Reported training requires two NVIDIA A40 GPUs with native bf16 support. Development and
inference may use older hardware only through an explicit precision fallback. PyTorch's
`is_bf16_supported` may include software emulation, so this module checks compute capability
directly and records both hardware and emulated support. Compared inference dumps must agree
on device, precision, tiling, resolution, merge, and filtering fields.
"""

from __future__ import annotations

import logging
import os
from typing import Any

__all__ = [
    "resolve_autocast_dtype",
    "describe_device",
    "assert_training_device_is_supported",
    "check_dumps_comparable",
    "cap_dataloader_workers",
    "cap_batch_size",
    "preserve_effective_batch",
    "node_load",
    "visible_gpu_count",
    "shared_node_env",
    "assert_node_headroom",
]

# Shared-node thresholds are expressed per core so they remain meaningful across hosts.
# The launch guard checks CPU load and GPU memory independently because CPU saturation reduces
# throughput while insufficient device memory terminates the run.
LOAD_PER_CORE_WARN = 0.60      # above this, launching will contend noticeably
LOAD_PER_CORE_BLOCK = 0.90     # above this, the node has no slack left at all
# The full fine-tuning path was measured above 26 GiB per rank. The 30 GiB floor includes
# headroom for distributed gradient buckets and variable instance-count allocations.
MIN_FREE_GPU_MIB = 30000

# Fields of inference_config.json that must agree before two dumps may be
# compared. Everything here changes the predictions, so a difference makes a
# delta between the two dumps partly an artifact of configuration rather than of
# the mechanism under test.
COMPARABILITY_FIELDS = (
    ("compute", "device_name"),
    ("compute", "amp_dtype"),
    ("compute", "bf16_supported"),
    ("tile_policy",),
    ("tile_px",),
    ("overlap_px",),
    ("resolution",),
    ("merge_method",),
    ("dedup_iou",),
    ("maxdets",),
    ("score_thresh",),
    ("min_area",),
    ("test_time_augmentation", "hflip"),
    ("test_time_augmentation", "multi_scale"),
)

_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logging.warning(message)


def _hardware_bf16_supported(device_index: int = 0) -> bool:
    """Return native bf16 hardware support rather than emulated support.

    PyTorch may report software emulation as supported. The hardware branch requires compute
    capability major version 8 or later, matching Ampere and newer NVIDIA devices.
    """
    import torch

    if getattr(torch.version, "hip", None):
        return True  # ROCm: bf16 is supported broadly, matching upstream's own check
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(device_index)
    return major >= 8


def resolve_autocast_dtype(preferred: str = "bfloat16", *, device_type: str = "cuda"):
    """The best autocast dtype the current device supports, warning on fallback.

    bf16 on Volta is the case that matters: it has no bf16 tensor cores, so bf16
    autocast there runs emulated in software with none of AMP's benefit (see
    `_hardware_bf16_supported`). float16 is the correct substitute for a
    V100 — real hardware acceleration, same memory saving, narrower dynamic
    range — and the fallback is logged so it cannot be mistaken for the
    configured setting.
    """
    import torch

    if device_type != "cuda" or not torch.cuda.is_available():
        return None  # caller disables autocast

    want = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": None}[preferred]
    if want is not torch.bfloat16:
        return want

    if _hardware_bf16_supported():
        return torch.bfloat16

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    _warn_once(
        "bf16_fallback",
        f"bfloat16 autocast requested but {name} (sm_{major}{minor}) has no bf16 tensor cores; "
        "falling back to float16 rather than the software-emulated bf16 that "
        "torch.cuda.is_bf16_supported() would otherwise report available (it measures "
        "*emulated* support by default, not hardware support -- see _hardware_bf16_supported). "
        "This is fine for inference, smoke tests and the LOSS_BOUNDARY_WEIGHT calibration. It is "
        "NOT fine for a reported training run, and two dumps compared to each other must share the "
        "same amp_dtype and device — see Core/device_policy.py.",
    )
    return torch.float16


def describe_device(*, amp_dtype: Any = None) -> dict[str, Any]:
    """Record GPU, capability, hardware bf16 support, emulation, and active dtype."""
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False, "amp_dtype": None}
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "cuda": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "capability": f"sm_{major}{minor}",
        "bf16_supported": _hardware_bf16_supported(),
        "bf16_emulated": bool(torch.cuda.is_bf16_supported()),
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "float32",
        "torch": torch.__version__,
    }


def assert_training_device_is_supported(preferred: str = "bfloat16", *, allow_fallback: bool = False) -> None:
    """Require native support for the precision declared by a reported training recipe.

    `allow_fallback=True` is restricted to non-reportable development checks.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device visible; a training run cannot honour the locked recipe.")
    if preferred != "bfloat16" or _hardware_bf16_supported():
        return
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    message = (
        f"The locked recipe is AMP bfloat16 on 2 x A40 (Ampere), but this node is {name} "
        f"(sm_{major}{minor}), which has no bf16. Training here would run in a different "
        "precision from the declared recipe and would not be directly comparable. Use native "
        "bf16 hardware for a reported run. Pass allow_fallback=True only for a non-reportable check."
    )
    if allow_fallback:
        _warn_once("training_fallback", "PROCEEDING ANYWAY (allow_fallback=True): " + message)
        return
    raise RuntimeError(message)


def _is_pre_ampere() -> bool:
    """Return whether visible device 0 predates Ampere.

    Query errors retain configured values rather than applying compatibility caps.
    """
    import torch

    try:
        return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8
    except Exception:
        return False


def cap_dataloader_workers(default: int, cap: int = 8) -> int:
    """Cap dataloader workers on memory-constrained pre-Ampere GPUs.

    Workers hold CUDA contexts, while the seeded sampler in the main process determines batch
    composition. Worker count changes throughput and memory use, not sample ordering. The
    resolved value is recorded in the run configuration.
    """
    if _is_pre_ampere():
        return min(default, cap)
    return default


def cap_batch_size(default: int, cap: int = 1) -> int:
    """Cap train and validation microbatch size on pre-Ampere GPUs.

    Dense mask memory combines fixed model and optimizer state with an instance-count-driven
    tensor stack. Lowering the microbatch reduces activation memory but does not proportionally
    reduce fixed costs. Gradient accumulation restores the configured effective training batch.
    Validation metrics aggregate per image over the full set, so validation microbatch size is
    a memory and throughput control. Resolved values are recorded in the run configuration.
    """
    if _is_pre_ampere():
        return min(default, cap)
    return default


def preserve_effective_batch(accum: int, *, recipe_batch: int, actual_batch: int) -> int:
    """Scale gradient accumulation to preserve the configured effective batch.

    Only `actual_batch` images are resident for each microbatch. The function leaves
    accumulation unchanged when no cap applies and rejects non-integral scaling factors.
    """
    if actual_batch >= recipe_batch:
        return accum
    if recipe_batch % actual_batch != 0:
        raise ValueError(
            f"recipe_batch={recipe_batch} is not a whole multiple of actual_batch="
            f"{actual_batch}, so gradient accumulation cannot restore the recipe's "
            "effective batch exactly. Pick a batch cap that divides the recipe batch."
        )
    return accum * (recipe_batch // actual_batch)


def visible_gpu_count() -> int:
    """Return devices visible after `CUDA_VISIBLE_DEVICES` masking.

    Deriving process count from visibility prevents a distributed rank from selecting an
    unavailable device.
    """
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def node_load() -> dict[str, Any]:
    """What the node is doing right now: load average, cores, and per-GPU memory.

    Deliberately shells out to `nvidia-smi` rather than using torch: this has to
    report on OTHER users' processes, and torch only sees its own context. It also
    has to work before any CUDA context exists, which is the whole point -- the
    answer decides whether to create one.

    Returns `gpus: []` rather than raising if `nvidia-smi` is unavailable, so a
    caller on a CPU-only machine (or a laptop) still gets the load numbers.
    """
    import shutil
    import subprocess

    try:
        n_cores = len(os.sched_getaffinity(0))     # cores THIS process may use,
    except AttributeError:                          # which is what actually matters
        n_cores = os.cpu_count() or 1
    load1, load5, load15 = os.getloadavg()

    gpus: list[dict[str, Any]] = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20, check=True,
            ).stdout
            for line in out.strip().splitlines():
                idx, used, total, util = (p.strip() for p in line.split(","))
                gpus.append({"index": int(idx), "used_mib": int(used),
                             "total_mib": int(total), "free_mib": int(total) - int(used),
                             "util_pct": int(util)})
        except (subprocess.SubprocessError, ValueError):
            pass

    return {
        "n_cores": n_cores,
        "load1": load1, "load5": load5, "load15": load15,
        "load_per_core": load5 / max(n_cores, 1),
        "gpus": gpus,
    }


def shared_node_env(threads: int = 1) -> dict[str, str]:
    """Cap each rank and worker thread pool at the subprocess boundary.

    Numerical libraries may otherwise create one thread per core for every dataloader worker.
    Setting limits before library initialisation prevents severe CPU oversubscription.
    """
    t = str(max(1, int(threads)))
    return {
        "OMP_NUM_THREADS": t,
        "MKL_NUM_THREADS": t,
        "OPENBLAS_NUM_THREADS": t,
        "NUMEXPR_NUM_THREADS": t,
        "VECLIB_MAXIMUM_THREADS": t,
        "CV_NUM_THREADS": t,
    }


def assert_node_headroom(
    *,
    require_gpu_mib: int = MIN_FREE_GPU_MIB,
    require_gpus: int = 1,
    block_above: float = LOAD_PER_CORE_BLOCK,
    warn_above: float = LOAD_PER_CORE_WARN,
    strict: bool = True,
) -> dict[str, Any]:
    """Refuse to launch onto a node that has no capacity left.

    Blocks on TWO independent conditions, because they fail differently:

    * CPU load per core above `block_above`, which starves dataloaders and other jobs.
    * Fewer than `require_gpus` devices with `require_gpu_mib` free, which risks an
      out-of-memory termination after launch.

    Returns the report either way so a caller can log it. `strict=False`
    downgrades the block to a printed warning, for the case where you have
    agreed with the other user to share the box anyway.
    """
    report = node_load()
    lpc, cores = report["load_per_core"], report["n_cores"]
    problems: list[str] = []

    if lpc > block_above:
        problems.append(
            f"CPU load per core is {lpc:.2f} (5-min load {report['load5']:.1f} on "
            f"{cores} cores), above the {block_above:.2f} block threshold. The node has "
            f"no slack: a launch now would starve its own dataloaders and slow every "
            f"other job on the box."
        )
    elif lpc > warn_above:
        print(f"  WARNING: CPU load per core {lpc:.2f} (5-min load {report['load5']:.1f} "
              f"on {cores} cores) is above {warn_above:.2f}. Launch will contend.")

    if report["gpus"]:
        usable = sorted((g for g in report["gpus"] if g["free_mib"] >= require_gpu_mib),
                        key=lambda g: -g["free_mib"])
        best = max(report["gpus"], key=lambda g: g["free_mib"])
        report["suggested_gpu"] = best["index"]
        report["usable_gpus"] = [g["index"] for g in usable]
        if len(usable) < require_gpus:
            have = ", ".join(f"GPU {g['index']}: {g['free_mib']}" for g in report["gpus"])
            problems.append(
                f"need {require_gpus} GPU(s) with {require_gpu_mib} MiB free, found "
                f"{len(usable)} ({have} MiB). A 2-rank run needs room on BOTH cards, and this "
                f"config peaked at 26,530 MiB per card when measured."
            )

    if problems and strict:
        raise RuntimeError(
            "Node has insufficient headroom to launch:\n  - "
            + "\n  - ".join(problems)
            + "\n\nRe-check with `node_load()` and launch when it clears, or pass "
              "strict=False if you have agreed to share the node anyway."
        )
    for p in problems:
        print(f"  WARNING (strict=False): {p}")
    return report


def _dig(payload: dict, path: tuple[str, ...]):
    node: Any = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return "<missing>"
        node = node[key]
    return node


def check_dumps_comparable(*run_dirs, strict: bool = True, ignore_fields: tuple = ()) -> list[str]:
    """Refuse to compare two inference dumps that were not produced the same way.

    This is the guard the two-stage record depends on. The development stage may
    run on the V100 node in float16 and the final stage on the A40 node in
    bfloat16; both are legitimate, and a delta *within* either stage is
    meaningful. A delta *across* them is not -- it mixes the mechanism with a
    change of architecture and precision, which this project has already been
    bitten by once ("a Boundary-IoU delta in the fourth decimal is not comparable
    between the two"). The gate is a small number, so an artifact of that size
    would decide it.

    Reads `inference_config.json` from each run directory and reports every field
    in `COMPARABILITY_FIELDS` (minus `ignore_fields`) that disagrees. Raises when
    `strict`.

    `ignore_fields` exists for **known, intentional** design differences, not for
    silencing a real one. The clearest case: the "why tiling is necessary"
    Mask R-CNN whole-image control is deliberately not tiled
    (`project_status.md`), so `tile_policy` (and the then-unused `tile_px`,
    `overlap_px`) must be excluded when sanity-checking that control against the
    tiled baselines -- but must NOT be excluded when checking an arm against its
    own baseline, where both are always tiled by protocol and a `tile_policy`
    mismatch there would be exactly the kind of confound this function exists to
    catch. Pass it explicitly and say why at the call site every time.

    Usage:  python Core/device_policy.py runtime/inference_output/a runtime/inference_output/b
    """
    import json
    from pathlib import Path

    loaded: list[tuple[str, dict]] = []
    for run_dir in run_dirs:
        path = Path(run_dir)
        config = path if path.name == "inference_config.json" else path / "inference_config.json"
        if not config.is_file():
            raise FileNotFoundError(
                f"No inference_config.json under {run_dir}. Dumps produced before that file "
                "existed cannot have their provenance verified and must not be used as a "
                "comparison reference -- re-run the inference."
            )
        loaded.append((str(path), json.loads(config.read_text())))

    if len(loaded) < 2:
        raise ValueError("Give at least two run directories to compare.")

    fields = tuple(f for f in COMPARABILITY_FIELDS if f not in ignore_fields)
    problems: list[str] = []
    base_name, base = loaded[0]
    for other_name, other in loaded[1:]:
        for field in fields:
            a, b = _dig(base, field), _dig(other, field)
            if a != b:
                problems.append(f"{'.'.join(field)}: {base_name}={a!r} vs {other_name}={b!r}")

    if problems:
        message = (
            "These dumps were not produced the same way, so a delta between them is not "
            "attributable to the mechanism under test:\n  - " + "\n  - ".join(problems)
        )
        if strict:
            raise RuntimeError(message)
        _warn_once("dumps_not_comparable", message)
    return problems


def _main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Compute provenance helpers.")
    parser.add_argument("run_dirs", nargs="*", help="Two or more inference run directories to compare.")
    parser.add_argument("--warn-only", action="store_true", help="Report differences without raising.")
    args = parser.parse_args()

    if not args.run_dirs:
        print(json.dumps(describe_device(amp_dtype=resolve_autocast_dtype("bfloat16")), indent=2))
        return

    problems = check_dumps_comparable(*args.run_dirs, strict=not args.warn_only)
    if not problems:
        print(f"COMPARABLE: {len(args.run_dirs)} dumps agree on every field that changes predictions.")


if __name__ == "__main__":
    _main()
