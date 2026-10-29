@echo off
rem Puts the site folder online (docs/GPU_PC.md): fetches what's live that this PC lacks,
rem checks the deploy won't take anything off the site, asks, then deploys.
rem   scripts\deploy_site.bat                (the site folder ..\..\site)
rem   scripts\deploy_site.bat D:\bruv\site   (another site folder)
setlocal
cd /d "%~dp0\.."
for %%I in ("%~dp0..\..") do set "BRUV=%%~fI"
set "SITE=%~1"
if not defined SITE set "SITE=%BRUV%\site"
call .venv\Scripts\activate.bat
python -m bruv_app.sitesync mirror --site "%SITE%" && python -m bruv_app.sitesync deploy --site "%SITE%"
set CODE=%errorlevel%
echo %cmdcmdline% | find /i "%~nx0" >nul && pause
exit /b %CODE%
