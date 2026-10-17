@echo off
rem One-time setup. Run from anywhere:  scripts\setup_windows.bat        (core + demo)
rem                                     scripts\setup_windows.bat --ml   (also the fish models, CPU)
setlocal
cd /d "%~dp0\.."
where py >nul 2>nul
if errorlevel 1 (
  echo Python 3.11 or newer is required: https://www.python.org/downloads/windows/
  exit /b 1
)
if not exist .venv py -3 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if /i "%~1"=="--ml" (
  rem CPU build of torch. With an NVIDIA GPU, install the CUDA build from pytorch.org instead.
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements-ml.txt
)
pip install -e .
echo.
echo Setup done. Try the demo:  scripts\run_windows.bat demo
