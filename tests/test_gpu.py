import subprocess
import sys
from pathlib import Path

from bruv_app.gpu import UPDATE_DRIVER, pick_build


def test_picks_the_pytorch_build_the_card_and_driver_can_use():
    assert pick_build(None, None) == ("cpu", "")  # no NVIDIA card
    assert pick_build("581.42", "8.9") == ("cu130", "")  # RTX 40, current driver
    assert pick_build("590.10", "12.0") == ("cu130", "")  # RTX 50
    assert pick_build("581.42", "6.1") == ("cu126", "")  # GTX 10: CUDA 13 dropped it
    assert pick_build("560.94", "8.6") == ("cu126", "")  # RTX 30, older driver
    assert pick_build("516.94", None) == ("cpu", UPDATE_DRIVER)  # driver too old for CUDA 12
    assert pick_build("572.16", "12.0") == ("cpu", UPDATE_DRIVER)  # RTX 50 needs CUDA 13 here
    assert pick_build("581.42", "3.5")[0] == "cpu"  # too old for PyTorch


def test_runs_as_a_script_before_the_app_is_installed():
    script = Path(__file__).parents[1] / "bruv_app" / "gpu.py"
    out = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True, check=True)
    assert out.stdout.strip() in {"cu130", "cu126", "cpu"}
