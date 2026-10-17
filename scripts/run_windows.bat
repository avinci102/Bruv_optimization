@echo off
rem Runs the app with the project's virtual environment, e.g.
rem   scripts\run_windows.bat demo
rem   scripts\run_windows.bat process --project projects\site1 --template templates\viewing.xlsx --weights models\cfd-rf-detr-nano-640-2026.02.02.cp-011.20260706-release.pth D:\GoPro\*.MP4
rem   scripts\run_windows.bat review --project projects\site1
setlocal
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
python -m bruv_app %*
