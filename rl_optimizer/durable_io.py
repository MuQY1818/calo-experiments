"""Small durable-write helpers for public experiment outputs."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


class DurableWriteError(RuntimeError):
    """Indicate that a target was not replaced by an atomic write."""

    target_installed = False
    durability_uncertain = False


class DurabilityUncertainError(DurableWriteError):
    """Indicate that a complete target was installed but not directory-synced."""

    target_installed = True
    durability_uncertain = True


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes through an fsynced same-directory atomic replacement.

    A failure before ``os.replace`` leaves any existing target untouched. A failure
    while syncing the parent directory occurs after the complete new target has been
    installed, so the exception explicitly reports uncertain crash durability.

    Args:
        path: Destination file.
        payload: Complete byte payload.

    Raises:
        DurableWriteError: If the target was not replaced.
        DurabilityUncertainError: If replacement succeeded but directory sync failed.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    installed = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        installed = True
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(target.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        if installed:
            raise DurabilityUncertainError(
                f"Installed complete file {target}, but parent-directory durability "
                "could not be confirmed."
            ) from exc
        raise DurableWriteError(
            f"Atomic write failed before replacing {target}; any existing target " "was preserved."
        ) from exc


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text with crash-durability checks."""
    atomic_write_bytes(Path(path), text.encode("utf-8"))
