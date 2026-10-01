"""Tests for checkpoint staging, backup, and resume-ledger helpers.

Staging uses the checkpoint epoch to prevent an older live checkpoint from replacing newer
recoverable state.
"""

from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

# Tests live in Codebase/Tests/; the modules under test live in Codebase/Core/.
CORE_DIR = Path(__file__).resolve().parent.parent / "Core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import torch

from project_workflows import (
    backup_training_artifacts,
    read_checkpoint_epoch,
    stage_resume_checkpoint,
)

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok  {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL {label}" + (f" -- {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _make_run(root: Path, epoch: int) -> Path:
    """A minimal experiment dir with a checkpoint and per-epoch records."""
    exp = root / "experiments" / "run"
    (exp / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp / "logs" / "oam_tcd").mkdir(parents=True, exist_ok=True)
    (exp / "tensorboard").mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model": {"w": torch.zeros(2)}}, exp / "checkpoints" / "checkpoint.pt")
    (exp / "logs" / "oam_tcd" / "train_stats.json").write_text(
        "\n".join(f'{{"Trainer/epoch": {e}, "loss": 1.0}}' for e in range(1, epoch + 1))
    )
    (exp / "tensorboard" / "events.out.tfevents.1").write_bytes(b"tb")
    (exp / "training_log.md").write_text(f"# run\nepochs 1..{epoch}\n")
    return exp


def test_epoch_reading() -> None:
    section("read_checkpoint_epoch: authoritative, and refuses to guess")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exp = _make_run(root, 24)
        check("reads the epoch field", read_checkpoint_epoch(exp / "checkpoints" / "checkpoint.pt") == 24)

        bad = root / "no_epoch.pt"
        torch.save({"model": {}}, bad)
        try:
            read_checkpoint_epoch(bad)
            check("a checkpoint without 'epoch' raises", False)
        except KeyError:
            check("a checkpoint without 'epoch' raises rather than defaulting to 0", True)

        try:
            read_checkpoint_epoch(root / "absent.pt")
            check("a missing file raises", False)
        except FileNotFoundError:
            check("a missing file raises FileNotFoundError", True)


def test_staging_refuses_downgrade() -> None:
    section("stage_resume_checkpoint: never silently replaces newer with older")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exp = _make_run(root, 24)
        dest = root / "runtime" / "resume.pt"

        res = stage_resume_checkpoint(exp, dest)
        check("first stage copies", res["copied"] and res["staged_epoch"] == 24)
        check("staged file verifies to the live epoch", read_checkpoint_epoch(dest) == 24)

        res = stage_resume_checkpoint(exp, dest)
        check("re-staging the same epoch is a no-op", res["copied"] is False)

        # Simulate an older live checkpoint after a newer epoch has already been staged.
        torch.save({"epoch": 23, "model": {"w": torch.zeros(2)}}, exp / "checkpoints" / "checkpoint.pt")
        try:
            stage_resume_checkpoint(exp, dest)
            check("staging an OLDER live checkpoint over a newer staged one raises", False)
        except RuntimeError as exc:
            check("staging an OLDER live checkpoint over a newer staged one raises", True)
            check("the error names both epochs and the cost", "23" in str(exc) and "24" in str(exc))
        check("the staged checkpoint is left untouched by the refusal", read_checkpoint_epoch(dest) == 24)

        res = stage_resume_checkpoint(exp, dest, allow_downgrade=True)
        check("allow_downgrade=True permits a deliberate rewind", res["staged_epoch"] == 23)


def test_staging_rejects_inside_experiment_dir() -> None:
    section("stage_resume_checkpoint: destination must survive the launcher wipe")
    with tempfile.TemporaryDirectory() as tmp:
        exp = _make_run(Path(tmp), 5)
        try:
            stage_resume_checkpoint(exp, exp / "checkpoints" / "resume.pt")
            check("a destination inside experiment_dir raises", False)
        except ValueError as exc:
            check("a destination inside experiment_dir raises", True)
            check("the error explains the wipe", "wipe" in str(exc).lower())


def test_backup_is_idempotent() -> None:
    section("backup_training_artifacts: captures records, cheap to repeat")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exp = _make_run(root, 24)
        backup = root / "runtime" / "backup"

        first = backup_training_artifacts(exp, backup)
        check("stats are captured", (backup / "logs" / "oam_tcd" / "train_stats.json").is_file())
        check("tensorboard events are captured", (backup / "tensorboard" / "events.out.tfevents.1").is_file())
        check("checkpoint is captured", (backup / "checkpoints" / "checkpoint.pt").is_file())
        check("training_log.md is captured", (backup / "training_log.md").is_file())
        check("the live checkpoint epoch is reported", first["checkpoint_epoch"] == 24)
        check("first call copies files", len(first["copied"]) >= 4)

        second = backup_training_artifacts(exp, backup)
        check("second call re-copies nothing (idempotent by size)", second["copied"] == [])
        check("second call reports everything skipped", len(second["skipped"]) >= 4)

        try:
            backup_training_artifacts(root / "absent", backup)
            check("a missing experiment dir raises", False)
        except FileNotFoundError:
            check("a missing experiment dir raises", True)


def test_stager_follows_a_live_run() -> None:
    """The watcher must track epochs written by a *concurrently running* trainer.

    Simulates training: a background thread writes checkpoint.pt for epochs
    24..27 while the stager watches. This is the property that matters -- the
    launch cell blocks for hours, so staging has to happen alongside training,
    not after it.
    """
    section("CheckpointStager: follows a live run, epoch by epoch")
    import threading as _threading

    from project_workflows import CheckpointStager, read_resume_ledger

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exp = _make_run(root, 23)
        dest = root / "runtime" / "resume.pt"
        ledger = root / "runtime" / "ledger.jsonl"
        stage_resume_checkpoint(exp, dest)  # start at epoch 23, as a resume would

        written: list[int] = []
        done = _threading.Event()

        def fake_trainer() -> None:
            for epoch in (24, 25, 26, 27):
                time.sleep(0.35)
                torch.save({"epoch": epoch, "model": {"w": torch.zeros(2)}},
                           exp / "checkpoints" / "checkpoint.pt")
                written.append(epoch)
            done.set()

        worker = _threading.Thread(target=fake_trainer, daemon=True)
        with CheckpointStager(exp, dest, ledger_path=ledger, interval_seconds=0.1, verbose=False):
            worker.start()
            done.wait(timeout=20)
            time.sleep(0.5)  # let the watcher see the last write

        check("the simulated trainer wrote all four epochs", written == [24, 25, 26, 27], str(written))
        final = read_checkpoint_epoch(dest)
        check("staged checkpoint ends at the final epoch", final == 27, f"got {final}")

        rows = read_resume_ledger(ledger)
        staged = [r["epoch"] for r in rows if r.get("action") == "staged"]
        check("every epoch reached the ledger in order",
              staged == sorted(staged) and staged[-1] == 27, str(staged))
        check("the ledger records what each staging replaced",
              all("previous_epoch" in r for r in rows if r.get("action") == "staged"))
        check("no errors were recorded", not [r for r in rows if r.get("action") == "error"])


def test_stager_never_moves_backwards() -> None:
    """A wiped/rolled-back experiment dir must not drag the staged copy back."""
    section("CheckpointStager: refuses to regress, and cannot kill the run")
    from project_workflows import CheckpointStager, read_resume_ledger

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exp = _make_run(root, 27)
        dest = root / "runtime" / "resume.pt"
        ledger = root / "runtime" / "ledger.jsonl"
        stage_resume_checkpoint(exp, dest)

        # A relaunch wipes the dir and the trainer writes an OLDER checkpoint.
        torch.save({"epoch": 20, "model": {"w": torch.zeros(2)}}, exp / "checkpoints" / "checkpoint.pt")
        with CheckpointStager(exp, dest, ledger_path=ledger, interval_seconds=0.1, verbose=False):
            time.sleep(0.6)
        check("staged copy still at the newer epoch", read_checkpoint_epoch(dest) == 27)

        # A corrupt checkpoint must be logged, not raised into the notebook.
        (exp / "checkpoints" / "checkpoint.pt").write_bytes(b"not a checkpoint")
        with CheckpointStager(exp, dest, ledger_path=ledger, interval_seconds=0.1, verbose=False) as st:
            time.sleep(0.6)
        check("a corrupt live checkpoint is recorded as an error, not raised", len(st.errors) >= 1)
        check("staged copy survives a corrupt live checkpoint", read_checkpoint_epoch(dest) == 27)
        check("the error is in the ledger",
              any(r.get("action") == "error" for r in read_resume_ledger(ledger)))


def test_backup_captures_final_model() -> None:
    """A backup must preserve the last-epoch model used by offline inference."""
    section("backup_training_artifacts: final_model/ is preserved")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exp = _make_run(root, 30)
        (exp / "final_model").mkdir()
        (exp / "final_model" / "pytorch_model.bin").write_bytes(b"weights")
        backup = root / "runtime" / "backup"

        backup_training_artifacts(exp, backup)
        check("final_model/pytorch_model.bin is captured",
              (backup / "final_model" / "pytorch_model.bin").is_file())
        check("its contents are intact",
              (backup / "final_model" / "pytorch_model.bin").read_bytes() == b"weights")


def test_ledger_survives_truncation() -> None:
    section("resume ledger: append-only, tolerant of an interrupted write")
    from project_workflows import append_resume_ledger, read_resume_ledger, summarize_resume_ledger

    with tempfile.TemporaryDirectory() as tmp:
        ledger = Path(tmp) / "ledger.jsonl"
        append_resume_ledger(ledger, action="resume_launch", from_epoch=23, device="NVIDIA A40")
        append_resume_ledger(ledger, action="staged", epoch=24, previous_epoch=23)
        with ledger.open("a") as handle:
            handle.write('{"action": "staged", "epoch": 2')  # interrupted write
        rows = read_resume_ledger(ledger)
        check("complete rows survive a truncated tail", len(rows) == 2, f"got {len(rows)}")
        check("every row is timestamped", all("ts" in r for r in rows))
        check("read of an absent ledger is empty, not an error", read_resume_ledger(Path(tmp) / "nope.jsonl") == [])
        summarize_resume_ledger(ledger)


def main() -> None:
    print("=" * 62)
    print("resume/backup helper self-tests")
    print("=" * 62)
    test_epoch_reading()
    test_staging_refuses_downgrade()
    test_staging_rejects_inside_experiment_dir()
    test_backup_is_idempotent()
    test_stager_follows_a_live_run()
    test_stager_never_moves_backwards()
    test_backup_captures_final_model()
    test_ledger_survives_truncation()

    print()
    print("=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED")
        for name in FAILURES:
            print(f"  - {name}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")
    print("=" * 62)


if __name__ == "__main__":
    main()
