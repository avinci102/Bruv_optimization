@echo off
rem ---------------------------------------------------------------------------
rem  BRUV Fish ID - quick start
rem  Drag your footage folder onto this file, or double-click it and paste the
rem  folder path. The folder needs your GoPro .MP4 file(s) and the Excel template.
rem  First run: installs Python packages and downloads the models (a few GB).
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set "FOLDER=%~1"
if not defined FOLDER set /p "FOLDER=Paste the folder with your video and Excel template, then press Enter: "
if not defined FOLDER (
  echo No folder given.
  goto end
)
set "FOLDER=%FOLDER:"=%"

if exist ".venv\bruv_setup_done" goto run
if exist ".venv\Scripts\python.exe" goto check_venv

call :pick_python
if errorlevel 1 goto end
%PY% -m venv .venv || goto fail

:check_venv
rem PyTorch has no Python 3.15 build yet, and numpy needs 3.12+, so the app needs 3.12-3.14.
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3, 12) <= sys.version_info[:2] <= (3, 14) else 1)"
if errorlevel 1 (
  echo The .venv folder was made with an unsupported Python version.
  echo Delete the .venv folder next to this file, then run quickstart.bat again.
  goto end
)
echo.
echo First run: installing. This downloads a few GB and can take 10-30 minutes.
echo Later runs skip this.
echo.
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip || goto fail
pip install -r requirements.txt || goto fail
rem CPU build of PyTorch. With an NVIDIA GPU, install the CUDA build from pytorch.org instead.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto fail
pip install -r requirements-ml.txt || goto fail
pip install -e . || goto fail
echo ok> ".venv\bruv_setup_done"

:run
call .venv\Scripts\activate.bat
set TRIES=0
:process
rem Processing resumes where it stopped, so a crash (e.g. out of memory) is retried twice.
python -m bruv_app quickstart "%FOLDER%" --no-review
set CODE=%errorlevel%
if %CODE% equ 0 goto review
rem Exit code 1 is an error the app explained (e.g. no template): retrying won't help.
if %CODE% equ 1 goto fail
set /a TRIES+=1
if %TRIES% geq 3 goto fail
echo.
echo Processing stopped unexpectedly. Picking up where it left off in 30 seconds...
ping -n 31 127.0.0.1 >nul
goto process

:review
echo Keep this window open while you review; close it when you're finished.
python -m bruv_app review --project "%FOLDER%\bruv_project"
rem A crash exits with a negative code, which "if errorlevel 1" would miss.
if %errorlevel% neq 0 goto fail
goto end

:pick_python
for %%V in (3.13 3.12 3.14) do (
  py -%%V -c "import sys" >nul 2>nul && (
    set "PY=py -%%V"
    exit /b 0
  )
)
echo Python 3.12, 3.13 or 3.14 is needed and wasn't found. (3.15 is too new for PyTorch.)
rem The newer "py" (Python's install manager) can add 3.13 for this user, no admin needed.
py list >nul 2>nul
if not errorlevel 1 (
  choice /m "Install Python 3.13 now"
  if errorlevel 2 exit /b 1
  py install -y 3.13 || exit /b 1
  set "PY=py -3.13"
  exit /b 0
)
where winget >nul 2>nul
if errorlevel 1 (
  echo Install Python 3.13 from https://www.python.org/downloads/windows/ then run this again.
  exit /b 1
)
choice /m "Install Python 3.13 now"
if errorlevel 2 exit /b 1
winget install -e --id Python.Python.3.13 --accept-package-agreements --accept-source-agreements
echo.
echo Python is installed. Close this window and run quickstart.bat again.
exit /b 1

:fail
echo.
echo Something went wrong. Copy the text above (or take a screenshot) and send it to Claude.
echo If it stopped with no error message, the PC may have run out of memory: close other
echo programs (browser tabs, games, chat apps) and run quickstart.bat again. It picks up
echo where it stopped.

:end
echo.
pause
