@echo off
rem One-time setup. Run from anywhere:  scripts\setup_windows.bat        (core + demo)
rem                                     scripts\setup_windows.bat --ml   (also the fish models, CPU)
setlocal
cd /d "%~dp0\.."
rem PyTorch has no Python 3.15 build yet and numpy needs 3.12+, so use 3.12-3.14.
set "PY="
for %%V in (3.13 3.12 3.14) do (
  if not defined PY py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
)
if not defined PY (
  echo Python 3.12, 3.13 or 3.14 is required: https://www.python.org/downloads/windows/
  exit /b 1
)
if not exist .venv %PY% -m venv .venv
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
