"""Atomic text writes for final run artefacts.

A reader, or a process that is killed mid-write, never sees a truncated file. The content is written
to a temporary file in the same directory, flushed to disk, and moved over the target with
`os.replace`, which is atomic on POSIX filesystems. The helper does not create parent directories,
matching `Path.write_text`.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write `text` to `path` so that the file holds either its old content or the complete new content."""
    target = Path(path)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
