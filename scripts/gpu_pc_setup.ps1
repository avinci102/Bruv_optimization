<#
One-time setup of a PC with an NVIDIA graphics card for processing BRUV videos and putting
them on the review site (docs/GPU_PC.md). Safe to run again: it skips what's done.

  powershell -ExecutionPolicy Bypass -File scripts\gpu_pc_setup.ps1           set up
  powershell -ExecutionPolicy Bypass -File scripts\gpu_pc_setup.ps1 -Check    only report

In order:
 1. Python 3.13 and Node.js (LTS) from winget, if missing. Git is already here: it fetched
    this folder.
 2. scripts\setup_windows.bat --ml: the app's packages, and PyTorch for the NVIDIA card.
 3. bruv_app check-setup: the card, PyTorch, ffmpeg, the models.
 4. The videos folder and the site folder, beside this repository: ..\videos, ..\site.
 5. Cloudflare: wrangler signs in (a browser page opens: click Allow), then sitesync init
    makes the site folder and downloads the videos already online.
#>
param([switch]$Check)

# Not "Stop": Windows PowerShell turns a program's warnings on stderr into errors then. Each
# step checks its own result instead.
$ErrorActionPreference = "Continue"
$Repo = (Resolve-Path "$PSScriptRoot\..").Path
$Root = Split-Path $Repo -Parent
$Videos = Join-Path $Root "videos"
$Site = Join-Path $Root "site"
$Python = Join-Path $Repo ".venv\Scripts\python.exe"

function Refresh-Path {
    # winget's installers add to PATH for new windows; this window needs it now.
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}
function Have($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }
function Python-Version {
    if (-not (Have "py")) { return $null }
    foreach ($v in "3.13", "3.12", "3.14") {
        & py "-$v" -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { return $v }
    }
    return $null
}
function Npx {
    if (Have "npx.cmd") { return "npx.cmd" }
    $default = "C:\Program Files\nodejs\npx.cmd"
    if (Test-Path $default) { return $default }
    return $null
}
function Winget-Install($id, $what) {
    if (-not (Have "winget")) { throw "winget isn't available: install $what by hand, then run this again." }
    Write-Host "Installing $what (a Windows prompt may ask to allow it)..."
    winget install -e --id $id --accept-package-agreements --accept-source-agreements --silent
    Refresh-Path
}

Write-Host "BRUV GPU PC setup: repository $Repo"
Write-Host ""

# What's here.
$gitOk = Have "git"
$py = Python-Version
$npx = Npx
$card = $null
if (Have "nvidia-smi") { $card = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1) }
Write-Host ("Git:            " + $(if ($gitOk) { (git --version) } else { "missing" }))
Write-Host ("Python:         " + $(if ($py) { "py -$py" } else { "missing (3.12-3.14 needed)" }))
Write-Host ("Node.js (npx):  " + $(if ($npx) { $npx } else { "missing" }))
Write-Host ("NVIDIA card:    " + $(if ($card) { $card } else { "none found (processing would run on the CPU)" }))
Write-Host ("App installed:  " + $(if (Test-Path $Python) { "yes" } else { "no" }))
Write-Host ("Site folder:    " + $(if (Test-Path (Join-Path $Site "wrangler.toml")) { "$Site (set up)" } else { "not set up" }))
if ($Check) {
    if (Test-Path $Python) { & $Python -m bruv_app check-setup }
    exit 0
}
Write-Host ""

# 1. Python and Node.js.
if (-not $py) {
    Winget-Install "Python.Python.3.13" "Python 3.13"
    $py = Python-Version
    if (-not $py) { throw "Python was installed, but this window can't see it yet: close it and run this script again." }
}
if (-not $npx) {
    Winget-Install "OpenJS.NodeJS.LTS" "Node.js LTS"
    $npx = Npx
    if (-not $npx) { throw "Node.js was installed, but this window can't see it yet: close it and run this script again." }
}

# 2. The app, with PyTorch for the card (about 3 GB the first time).
function App-Ready {
    if (-not (Test-Path $Python)) { return $false }
    & $Python -c "import torch, rfdetr, open_clip, bruv_app" 2>$null
    return ($LASTEXITCODE -eq 0)
}
if (-not (App-Ready)) {
    & "$Repo\scripts\setup_windows.bat" --ml
    if (-not (App-Ready)) { throw "The app's setup didn't finish (above): fix what it says, then run this again." }
}

# 3. What processing will run on.
& $Python -m bruv_app check-setup

# 4. Folders.
New-Item -ItemType Directory -Force $Videos | Out-Null
Write-Host "Videos go in $Videos, one subfolder per deployment (docs/GPU_PC.md)."

# 5. Cloudflare.
if (-not (Test-Path (Join-Path $Site "wrangler.toml"))) {
    New-Item -ItemType Directory -Force $Site | Out-Null
    Push-Location $Site
    try {
        & $npx --yes wrangler@4 whoami 2>$null | Out-String | Set-Variable who
        if ($who -notmatch "associated with the email") {
            Write-Host "Signing wrangler in to Cloudflare: a browser page opens. Sign in to the account the site runs on, then click Allow."
            & $npx --yes wrangler@4 login
        }
    } finally { Pop-Location }
}
& $Python -m bruv_app.sitesync init --site $Site
if ($LASTEXITCODE -ne 0) {
    Write-Host "The site folder isn't in step with the live site yet (above). Processing can still go ahead."
}
Write-Host ""
Write-Host "Set up. Next: put the videos in $Videos and run scripts\process_new_videos.bat"
