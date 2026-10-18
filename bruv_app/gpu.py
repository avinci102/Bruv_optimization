"""Pick the PyTorch build for this PC; the setup scripts install what this prints.

Runs before the app is installed, so it uses the standard library only:
    python bruv_app\\gpu.py      ->  cu130, cu126 or cpu

- cu130 (CUDA 13.0): NVIDIA cards from the RTX 20 series on, including the RTX 50
  series. Needs driver 580 or newer.
- cu126 (CUDA 12.6): also older cards, from the GTX 900 series to the RTX 40 series.
  Needs driver 528 or newer.
- cpu: no NVIDIA card, or one this PyTorch can't use. The app then runs on the CPU.
"""

from __future__ import annotations

import subprocess
import sys

UPDATE_DRIVER = (
    "Update the NVIDIA driver (580 or newer), then delete the .venv folder and run setup again "
    "to use the graphics card. Until then the app runs on the CPU."
)


def pick_build(driver: str | None, compute_cap: str | None) -> tuple[str, str]:
    """(build, advice) for a card's driver version (e.g. "581.42") and compute capability
    (e.g. "8.9"; None when nvidia-smi is too old to report it)."""
    try:
        major = int(str(driver).split(".")[0])
    except ValueError:
        return "cpu", ""  # no NVIDIA card
    try:
        cc = float(compute_cap) if compute_cap else None
    except ValueError:
        cc = None
    if cc is not None and cc < 5.0:
        return "cpu", "This NVIDIA card is too old for PyTorch; the app runs on the CPU."
    if cc is not None and cc >= 7.5 and major >= 580:
        return "cu130", ""
    if (cc is None or cc < 12.0) and major >= 528:
        return "cu126", ""
    return "cpu", UPDATE_DRIVER


def _query(fields: str) -> list[str] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return [f.strip() for f in out.splitlines()[0].split(",")] if out else None


def main() -> None:
    # The first card decides. Drivers too old to report compute_cap answer the second query.
    fields = _query("driver_version,compute_cap") or _query("driver_version") or []
    build, advice = pick_build(fields[0] if fields else None, fields[1] if len(fields) > 1 else None)
    if advice:
        print(advice, file=sys.stderr)
    print(build)


if __name__ == "__main__":
    main()
