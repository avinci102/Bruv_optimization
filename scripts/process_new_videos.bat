@echo off
rem Processes every new video folder in ..\..\videos and gets it ready for the review site
rem (docs/GPU_PC.md). Add --deploy to put it online at the end: it checks, then asks.
rem   scripts\process_new_videos.bat
rem   scripts\process_new_videos.bat --deploy
setlocal
cd /d "%~dp0\.."
for %%I in ("%~dp0..\..") do set "BRUV=%%~fI"
rem The newest code first: the site refuses a deploy from code older than what's online.
git pull --ff-only || echo Couldn't update the code (above); going on with this version.
call .venv\Scripts\activate.bat
python -m bruv_app.batch --videos "%BRUV%\videos" --site "%BRUV%\site" %*
set CODE=%errorlevel%
rem Keep the window open when double-clicked.
echo %cmdcmdline% | find /i "%~nx0" >nul && pause
exit /b %CODE%
