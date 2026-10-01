"""Tests for hardware bf16 detection, precision fallback, and node capacity.

PyTorch may report software-emulated bf16 as supported on pre-Ampere hardware. These tests
mock CUDA capabilities and require the policy to distinguish native tensor-core support from
emulation without depending on a particular physical host.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Tests live in Codebase/Tests/; the modules under test live in Codebase/Core/.
CORE_DIR = Path(__file__).resolve().parent.parent / "Core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

FAILURES: list[str] = []

import torch  # noqa: E402

torch.set_num_threads(1)

import device_policy  # noqa: E402


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok  {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL {label}" + (f" -- {detail}" if detail else ""))


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


# GPUs named here by the (major, minor) compute capability that identifies them,
# matching the values torch.cuda.get_device_capability returns.
V100 = (7, 0)   # Volta: no bf16 tensor cores, but is_bf16_supported() lies True
A40 = (8, 6)    # Ampere: real bf16 tensor cores


@contextmanager
def fake_cuda_device(capability: tuple[int, int], name: str, *, lie_bf16_supported: bool = True):
    """Monkeypatch torch.cuda to look like one GPU, restoring it afterward.

    `lie_bf16_supported=True` reproduces the exact emulation trap: the stub
    always returns True, as the real function does by default on Volta. If
    device_policy's own checks ever start calling that stub directly instead of
    `_hardware_bf16_supported`, this is what makes the tests below fail.
    """
    originals = {
        name_: getattr(torch.cuda, name_)
        for name_ in ("is_available", "get_device_capability", "get_device_name",
                      "device_count", "is_bf16_supported")
    }
    try:
        torch.cuda.is_available = lambda: True
        torch.cuda.get_device_capability = lambda device=0: capability
        torch.cuda.get_device_name = lambda device=0: name
        torch.cuda.device_count = lambda: 2
        torch.cuda.is_bf16_supported = lambda including_emulation=True: lie_bf16_supported
        yield
    finally:
        for name_, fn in originals.items():
            setattr(torch.cuda, name_, fn)


@contextmanager
def no_cuda():
    original = torch.cuda.is_available
    try:
        torch.cuda.is_available = lambda: False
        yield
    finally:
        torch.cuda.is_available = original


def test_hardware_check_is_not_fooled_by_v100_emulation() -> None:
    section("_hardware_bf16_supported ignores the emulation-inclusive lie")
    with fake_cuda_device(V100, "Tesla V100-PCIE-32GB", lie_bf16_supported=True):
        check(
            "torch.cuda.is_bf16_supported() (stubbed) reports True, as an emulating driver does",
            torch.cuda.is_bf16_supported() is True,
        )
        check(
            "but _hardware_bf16_supported correctly reports False for sm_70",
            device_policy._hardware_bf16_supported() is False,
        )
    with fake_cuda_device(A40, "NVIDIA A40", lie_bf16_supported=True):
        check(
            "_hardware_bf16_supported reports True for sm_86 (real Ampere tensor cores)",
            device_policy._hardware_bf16_supported() is True,
        )


def test_resolve_autocast_dtype_falls_back_on_v100() -> None:
    section("resolve_autocast_dtype: fp16 on V100, bf16 on A40, both under the lie")
    with fake_cuda_device(V100, "Tesla V100-PCIE-32GB", lie_bf16_supported=True):
        dtype = device_policy.resolve_autocast_dtype("bfloat16")
        check(
            "bfloat16 request resolves to float16 on V100",
            dtype is torch.float16,
            f"got {dtype} -- if this is bfloat16, is_bf16_supported()'s lie leaked through",
        )
    with fake_cuda_device(A40, "NVIDIA A40", lie_bf16_supported=True):
        dtype = device_policy.resolve_autocast_dtype("bfloat16")
        check("bfloat16 request resolves to bfloat16 on A40", dtype is torch.bfloat16)
    with fake_cuda_device(V100, "Tesla V100-PCIE-32GB", lie_bf16_supported=True):
        check(
            "an explicit float16 request is passed through unchanged",
            device_policy.resolve_autocast_dtype("float16") is torch.float16,
        )
    with no_cuda():
        check(
            "no CUDA device -> None (caller disables autocast)",
            device_policy.resolve_autocast_dtype("bfloat16") is None,
        )


def test_describe_device_reports_hardware_and_emulated_separately() -> None:
    section("describe_device keeps the hardware and emulated answers distinct")
    with fake_cuda_device(V100, "Tesla V100-PCIE-32GB", lie_bf16_supported=True):
        info = device_policy.describe_device(amp_dtype=torch.float16)
        check("capability is recorded as sm_70", info["capability"] == "sm_70", info["capability"])
        check(
            "bf16_supported (hardware) is False on V100",
            info["bf16_supported"] is False,
            "this is the field inference_config.json and check_dumps_comparable rely on",
        )
        check(
            "bf16_emulated is True, matching the live symptom that motivated this test",
            info["bf16_emulated"] is True,
        )
        check("amp_dtype records what was actually used, not the recipe", info["amp_dtype"] == "float16")


def test_training_guard_fires_on_v100_despite_the_lie() -> None:
    section("assert_training_device_is_supported fires on V100 despite the emulated report")
    with fake_cuda_device(V100, "Tesla V100-PCIE-32GB", lie_bf16_supported=True):
        raised = None
        try:
            device_policy.assert_training_device_is_supported("bfloat16", allow_fallback=False)
        except RuntimeError as exc:
            raised = exc
        check(
            "raises on V100 without allow_fallback, even though is_bf16_supported() lies True",
            raised is not None,
            "if this does not raise, the guard is not protecting the manuscript's recipe claim",
        )
        check("no exception with allow_fallback=True",
              _does_not_raise(lambda: device_policy.assert_training_device_is_supported("bfloat16", allow_fallback=True)))
    with fake_cuda_device(A40, "NVIDIA A40", lie_bf16_supported=True):
        check(
            "does not raise on A40",
            _does_not_raise(lambda: device_policy.assert_training_device_is_supported("bfloat16", allow_fallback=False)),
        )
    with no_cuda():
        raised = None
        try:
            device_policy.assert_training_device_is_supported("bfloat16", allow_fallback=True)
        except RuntimeError as exc:
            raised = exc
        check(
            "no CUDA at all raises regardless of allow_fallback",
            raised is not None,
            "there is no device to run the recipe on",
        )


def _does_not_raise(fn) -> bool:
    try:
        fn()
        return True
    except Exception as exc:  # pragma: no cover
        print(f"        raised: {type(exc).__name__}: {exc}")
        return False


def test_cap_dataloader_workers() -> None:
    section("cap_dataloader_workers: cap on V100, pass-through on A40 and CPU")
    with fake_cuda_device(V100, "Tesla V100-PCIE-32GB"):
        check("12 workers cap to 8 on V100", device_policy.cap_dataloader_workers(12) == 8)
        check("12 workers cap to 4 when cap=4 is passed", device_policy.cap_dataloader_workers(12, cap=4) == 4)
        check("a default below the cap is untouched", device_policy.cap_dataloader_workers(4) == 4)
    with fake_cuda_device(A40, "NVIDIA A40"):
        check("A40 keeps 12", device_policy.cap_dataloader_workers(12) == 12)
        check("A40 keeps 12 even when cap=4 is passed", device_policy.cap_dataloader_workers(12, cap=4) == 12)
    with no_cuda():
        check("no CUDA keeps the default", device_policy.cap_dataloader_workers(12) == 12)


def test_cap_batch_size() -> None:
    section("cap_batch_size: cap on V100, pass-through on A40 and CPU")
    with fake_cuda_device(V100, "Tesla V100-PCIE-32GB"):
        check("batch 8 caps to 1 on V100 (val peaked at 27-28 of 31.7 GiB at batch 2)",
              device_policy.cap_batch_size(8) == 1)
        check("a default below the cap is untouched", device_policy.cap_batch_size(1) == 1)
        check("an explicit looser cap is honoured", device_policy.cap_batch_size(8, cap=4) == 4)
    with fake_cuda_device(A40, "NVIDIA A40"):
        check("A40 keeps the recipe batch of 8", device_policy.cap_batch_size(8) == 8)
        check("A40 keeps 8 even with an explicit cap", device_policy.cap_batch_size(8, cap=1) == 8)
    with no_cuda():
        check("no CUDA keeps the default", device_policy.cap_batch_size(8) == 8)


def test_preserve_effective_batch() -> None:
    section("preserve_effective_batch: capped microbatch keeps the recipe's effective batch")
    check(
        "batch capped 8 -> 1 scales accumulation 1 -> 8",
        device_policy.preserve_effective_batch(1, recipe_batch=8, actual_batch=1) == 8,
    )
    check(
        "effective batch is identical at the batch-1 floor",
        1 * device_policy.preserve_effective_batch(1, recipe_batch=8, actual_batch=1) == 8 * 1,
    )
    check(
        "batch capped 8 -> 2 scales accumulation 1 -> 4",
        device_policy.preserve_effective_batch(1, recipe_batch=8, actual_batch=2) == 4,
    )
    check(
        "effective batch is identical either way",
        2 * device_policy.preserve_effective_batch(1, recipe_batch=8, actual_batch=2) == 8 * 1,
    )
    check(
        "an uncapped batch leaves accumulation alone (A40 recipe untouched)",
        device_policy.preserve_effective_batch(1, recipe_batch=8, actual_batch=8) == 1,
    )
    check(
        "a batch larger than the recipe is left alone rather than scaled down",
        device_policy.preserve_effective_batch(1, recipe_batch=8, actual_batch=16) == 1,
    )
    check(
        "an existing accumulation setting is multiplied, not replaced",
        device_policy.preserve_effective_batch(2, recipe_batch=8, actual_batch=4) == 4,
    )
    raised = None
    try:
        device_policy.preserve_effective_batch(1, recipe_batch=8, actual_batch=3)
    except ValueError as exc:
        raised = exc
    check(
        "a non-divisor batch raises instead of silently changing the effective batch",
        raised is not None,
    )


def _fake_node(load5, n_cores=32, gpus=None):
    """Patch node_load to a specific observed node state."""
    import contextlib

    @contextlib.contextmanager
    def ctx():
        original = device_policy.node_load
        device_policy.node_load = lambda: {
            "n_cores": n_cores, "load1": load5, "load5": load5, "load15": load5,
            "load_per_core": load5 / n_cores,
            "gpus": gpus if gpus is not None else [
                {"index": 0, "used_mib": 10331, "total_mib": 46068,
                 "free_mib": 35737, "util_pct": 0}],
        }
        try:
            yield
        finally:
            device_policy.node_load = original
    return ctx()


def test_node_load_reports_this_machine() -> None:
    section("node_load: reports real cores, load and GPUs without needing CUDA")
    r = device_policy.node_load()
    check("reports a positive core count", r["n_cores"] >= 1)
    check("reports three load averages",
          all(isinstance(r[k], float) for k in ("load1", "load5", "load15")))
    check("load_per_core is load5 / cores",
          abs(r["load_per_core"] - r["load5"] / r["n_cores"]) < 1e-9)
    check("gpus is a list (empty is valid on a laptop)", isinstance(r["gpus"], list))


def test_shared_node_env_caps_every_thread_pool() -> None:
    section("shared_node_env: caps BLAS and OpenCV, not just OMP")
    env = device_policy.shared_node_env(1)
    # Capping OMP alone is the common mistake: NumPy may be built against MKL or
    # OpenBLAS, and OpenCV reads neither, so each needs naming explicitly.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "CV_NUM_THREADS"):
        check(f"{var} is set", env.get(var) == "1")
    check("values are strings, as the environment requires",
          all(isinstance(v, str) for v in env.values()))
    check("a higher thread count is honoured", device_policy.shared_node_env(4)["OMP_NUM_THREADS"] == "4")
    check("zero and negatives floor to 1", device_policy.shared_node_env(0)["OMP_NUM_THREADS"] == "1")


def test_node_headroom_blocks_saturated_fixture() -> None:
    """A saturated-node fixture must be refused."""
    section("assert_node_headroom: blocks a saturated node")
    with _fake_node(load5=62.14, n_cores=32):
        try:
            device_policy.assert_node_headroom()
            check("blocks a 62-load / 32-core node", False, "it allowed the launch")
        except RuntimeError as exc:
            check("blocks a 62-load / 32-core node", True)
            check("says CPU load, not GPU, is the problem",
                  "load per core" in str(exc).lower() and "MiB free" not in str(exc))
            check("names the threshold it tripped", "0.90" in str(exc))


def test_node_headroom_allows_a_quiet_node() -> None:
    section("assert_node_headroom: allows a quiet node, and suggests the emptier GPU")
    with _fake_node(load5=4.0, n_cores=32,
                    gpus=[{"index": 0, "used_mib": 40000, "total_mib": 46068,
                           "free_mib": 6068, "util_pct": 90},
                          {"index": 1, "used_mib": 1000, "total_mib": 46068,
                           "free_mib": 45068, "util_pct": 0}]):
        report = device_policy.assert_node_headroom()
        check("does not raise on a quiet node", True)
        check("suggests the GPU with the most free memory", report["suggested_gpu"] == 1)


def test_node_headroom_blocks_when_no_gpu_has_room() -> None:
    """GPU exhaustion fails differently from CPU load -- as an OOM mid-run."""
    section("assert_node_headroom: blocks when no GPU has enough free memory")
    with _fake_node(load5=2.0, n_cores=32,
                    gpus=[{"index": 0, "used_mib": 44000, "total_mib": 46068,
                           "free_mib": 2068, "util_pct": 100}]):
        try:
            device_policy.assert_node_headroom(require_gpu_mib=20000)
            check("blocks when the emptiest GPU is too full", False, "it allowed the launch")
        except RuntimeError as exc:
            check("blocks when the emptiest GPU is too full", True)
            check("reports the free amount it found", "2068" in str(exc))


def test_node_headroom_strict_false_warns_instead() -> None:
    section("assert_node_headroom: strict=False downgrades the block to a warning")
    with _fake_node(load5=62.14, n_cores=32):
        try:
            report = device_policy.assert_node_headroom(strict=False)
            check("returns instead of raising", isinstance(report, dict))
            check("still reports the load it saw", report["load5"] == 62.14)
        except RuntimeError:
            check("returns instead of raising", False, "it raised despite strict=False")


def main() -> None:
    print("=" * 62)
    print("device_policy self-tests")
    print("=" * 62)

    test_hardware_check_is_not_fooled_by_v100_emulation()
    test_resolve_autocast_dtype_falls_back_on_v100()
    test_describe_device_reports_hardware_and_emulated_separately()
    test_training_guard_fires_on_v100_despite_the_lie()
    test_cap_dataloader_workers()
    test_cap_batch_size()
    test_preserve_effective_batch()
    test_node_load_reports_this_machine()
    test_shared_node_env_caps_every_thread_pool()
    test_node_headroom_blocks_saturated_fixture()
    test_node_headroom_allows_a_quiet_node()
    test_node_headroom_blocks_when_no_gpu_has_room()
    test_node_headroom_strict_false_warns_instead()

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
