#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run every test suite in this directory and report one verdict.

Each suite is a standalone script that raises on failure, so the runner executes them as
subprocesses: an `os._exit`, a segfault, or a CUDA abort in one suite cannot take the runner down or
hide the suites after it. The exit status is non-zero if any suite fails, so the runner can serve as
a pre-commit or continuous integration gate.

A suite whose optional dependency is missing is reported as SKIP rather than FAIL, and
`--strict-deps` turns any skip into a failure. Some suites check CUDA-side behaviour that a CPU-only
host cannot verify (`test_device_policy` and `test_eval_overrides`), so run the complete set with
`--strict-deps` in the full GPU environment before trusting GPU numerics.

Usage:
  python Tests/run_all.py                 # everything
  python Tests/run_all.py eval_overrides  # substring filter
  python Tests/run_all.py -v              # stream each suite's own output
  python Tests/run_all.py --strict-deps   # fail if any suite is skipped
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("filters", nargs="*", help="only run suites whose name contains one of these")
    ap.add_argument("-v", "--verbose", action="store_true", help="stream suite output instead of summarising")
    ap.add_argument("--strict-deps", action="store_true", help="fail if any suite is skipped for a missing dependency")
    args = ap.parse_args()

    suites = sorted(TESTS_DIR.glob("test_*.py"))
    if args.filters:
        suites = [s for s in suites if any(f in s.stem for f in args.filters)]
    if not suites:
        print("No suites matched.", file=sys.stderr)
        return 2

    # KMP_DUPLICATE_LIB_OK: torch and pycocotools can each pull in a copy of libomp on
    # macOS, which aborts the process at import. Set here so a local run does not need
    # it prefixed by hand; harmless on Linux.
    #
    # KMP_DUPLICATE_LIB_OK suppresses the duplicate-runtime abort but does not make two
    # multithreaded OpenMP runtimes safe. Single-threaded OMP keeps local macOS tests
    # deterministic. Linux retains the environment's configured parallelism.
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONFAULTHANDLER": "1"}
    if sys.platform == "darwin":
        env["OMP_NUM_THREADS"] = "1"

    print(f"Running {len(suites)} suite(s) with {sys.executable}\n")
    results: list[tuple[str, bool, float, str]] = []
    for suite in suites:
        started = time.monotonic()
        proc = subprocess.run(
            [sys.executable, str(suite)],
            cwd=TESTS_DIR.parent,          # repo root, so relative data paths resolve
            env=env,
            capture_output=not args.verbose,
            text=True,
        )
        elapsed = time.monotonic() - started
        combined = (proc.stdout or "") + (proc.stderr or "")

        # A missing optional dependency is not a code failure, and conflating the two makes
        # the runner report failures on a machine that cannot install it. Report it as SKIP
        # and keep the exit status clean, but print it clearly so that a skipped suite is
        # never mistaken for a verified one; `--strict-deps` rejects skips outright.
        missing = ""
        if proc.returncode != 0 and "ModuleNotFoundError" in combined:
            for line in combined.splitlines():
                if "ModuleNotFoundError" in line and "'" in line:
                    missing = line.split("'")[1]
                    break

        status = "PASS" if proc.returncode == 0 else ("SKIP" if missing else "FAIL")
        tail = ""
        if status == "FAIL" and not args.verbose:
            lines = combined.strip().splitlines()
            tail = "\n      ".join(lines[-12:]) if lines else f"exit {proc.returncode}"
        results.append((suite.stem, status, elapsed, missing or tail))

        note = f"  (no module {missing!r})" if status == "SKIP" else ""
        print(f"  {status}  {suite.stem:<28} {elapsed:6.1f}s{note}")
        if tail:
            print(f"      {tail}")

    failed = [n for n, s, _, _ in results if s == "FAIL"]
    skipped = [(n, m) for n, s, _, m in results if s == "SKIP"]
    passed = sum(1 for _, s, _, _ in results if s == "PASS")
    total = sum(e for _, _, e, _ in results)

    print(f"\n{passed}/{len(results)} passed, {len(skipped)} skipped, {len(failed)} failed"
          f" in {total:.1f}s")
    if skipped:
        for name, mod in skipped:
            print(f"  SKIPPED {name}: needs {mod!r}")
        print("  -> these were NOT verified here. Run in the complete GPU environment")
        print("     before trusting results that depend on them.")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    if skipped and args.strict_deps:
        print("FAILED: --strict-deps does not permit dependency-skipped suites.")
        return 1
    print("No failures." if skipped else "ALL SUITES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
