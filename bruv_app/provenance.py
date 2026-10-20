"""Where every result came from: settings, code, models and footage.

Each pipeline stage writes a run record (see Project.start_run): a digest of the settings,
the code revision, the model files by fingerprint, and what actually ran (BioCLIP 2 or the
colour-histogram fallback, which must never be pooled with BioCLIP results). Any output can
then be traced, and two runs compared on like terms.
"""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

CHUNK = 16 * 1024 * 1024
QUICK_ABOVE = 256 * 1024 * 1024


def fingerprint(path: Path, quick: bool | None = None) -> str:
    """SHA-256 of a file. Files over 256 MB (videos, the species model) are fingerprinted from
    their size plus their first and last 16 MB, which tells recordings apart in a moment
    instead of minutes; the prefix says which kind of fingerprint it is."""
    path = Path(path)
    size = path.stat().st_size
    quick = size > QUICK_ABOVE if quick is None else quick
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if quick:
            h.update(str(size).encode())
            h.update(f.read(CHUNK))
            f.seek(max(0, size - CHUNK))
            h.update(f.read(CHUNK))
        else:
            for block in iter(lambda: f.read(CHUNK), b""):
                h.update(block)
    return ("sha256-quick:" if quick else "sha256:") + h.hexdigest()


def deployment_id(fingerprints: list[str]) -> str:
    """One id for a recording, from its chapters' fingerprints in order. Copies of the same
    footage in different folders get the same id, so they can't pass for separate drops."""
    return "dep-" + hashlib.sha256("|".join(fingerprints).encode()).hexdigest()[:16]


def digest(data: dict) -> str:
    """A short id for a set of settings: equal settings, equal id."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def settings_digest(settings) -> str:
    return digest(asdict(settings))


@functools.lru_cache(maxsize=1)
def code_revision() -> str:
    """The git commit the app runs from ("+changes" if files differ from it), or its version
    when installed without git. Read once per process: the code that was loaded."""
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if head.returncode == 0:
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True, text=True, timeout=10,
            )
            return head.stdout.strip() + ("+changes" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        pass
    from . import __version__  # noqa: PLC0415

    return f"version {__version__}"


def describe_model(model) -> dict:
    """What a detector or classifier is, for a run record: its own description if it has one."""
    if hasattr(model, "describe"):
        return model.describe()
    return {"name": getattr(model, "name", type(model).__name__)}
