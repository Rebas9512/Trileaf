# ------------------------------------------------------------------------------
#  Trileaf -- Setup (Windows PowerShell)
#
#  Canonical setup script. Called directly by developers, or by install.ps1
#  after cloning the repository.
#
#  Usage (after git clone):
#    powershell -ExecutionPolicy Bypass -File setup.ps1
#
#  Options:
#    -Reinstall       Delete and recreate the .venv
#    -Headless        Non-interactive / CI mode; skips all prompts
#    -Doctor          Run environment check only, then exit
#    -FromInstaller   Internal flag set by install.ps1 (adjusts banner only)
# ------------------------------------------------------------------------------
param(
    [switch]$Reinstall,
    [switch]$Headless,
    [switch]$Doctor,
    [switch]$FromInstaller
)

$ErrorActionPreference = "Stop"

# -- Colour helpers ------------------------------------------------------------
$SupportsColor = $Host.UI.SupportsVirtualTerminal -and $null -eq $env:NO_COLOR
$ESC = [char]0x1b
function c($code, $text) { if ($SupportsColor) { return "${code}${text}${ESC}[0m" } return $text }
$G = "${ESC}[38;2;0;229;180m"; $Y = "${ESC}[38;2;255;176;32m"
$R = "${ESC}[38;2;230;57;70m"; $M = "${ESC}[38;2;110;120;148m"; $B = "${ESC}[1m"

function ok      ($msg) { Write-Host "$(c $G '+')  $msg" }
function info    ($msg) { Write-Host "$(c $M '.')  $msg" }
function warn    ($msg) { Write-Host "$(c $Y '!')  $msg" }
function fail    ($msg) { Write-Host "$(c $R 'x')  $msg" -ForegroundColor Red; exit 1 }
function section ($t)   { Write-Host ""; Write-Host (c $B "-- $t --") }

# -- Project paths -------------------------------------------------------------
$ScriptDir    = $PSScriptRoot
$VenvDir      = Join-Path $ScriptDir ".venv"
$VenvPython   = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip      = Join-Path $VenvDir "Scripts\pip.exe"
$TrileafExe   = Join-Path $VenvDir "Scripts\trileaf.exe"
$ScriptsDir   = Join-Path $VenvDir "Scripts"
$Requirements = Join-Path $ScriptDir "requirements.txt"

# -- Banner --------------------------------------------------------------------
Write-Host ""
if ($FromInstaller) { Write-Host (c $B "  Trileaf -- Installing") }
else                { Write-Host (c $B "  Trileaf -- Setup") }
Write-Host (c $M "  Project dir: $ScriptDir")
Write-Host ""

# -- Step 1 / 6 -- Platform ----------------------------------------------------
section "Step 1 / 6  --  Platform"

$policy = Get-ExecutionPolicy -Scope Process
if ($policy -eq "Restricted" -or $policy -eq "AllSigned") {
    try { Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force }
    catch { fail "Cannot set execution policy.`n  Run as Administrator: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser" }
}
ok "Platform: Windows"

# -- Step 2 / 6 -- Python ------------------------------------------------------
section "Step 2 / 6  --  Python"

function Is-SufficientVersion ($ver) {
    if (-not $ver) { return $false }
    $p = $ver -split '\.'; return ([int]$p[0] -gt 3) -or ([int]$p[0] -eq 3 -and [int]$p[1] -ge 10)
}
function Get-PythonVersion ($cmd) {
    try {
        $r = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $r) { return $r.Trim() }
    } catch {}
    return $null
}

$PythonExe = $null; $PythonVersion = $null
foreach ($spec in @("3.13","3.12","3.11","3.10")) {
    try {
        $v = & py "-$spec" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and (Is-SufficientVersion $v)) { $PythonExe = "py -$spec"; $PythonVersion = $v; break }
    } catch {}
}
if (-not $PythonExe) {
    foreach ($cmd in @("python3","python")) {
        $v = Get-PythonVersion $cmd
        if (Is-SufficientVersion $v) { $PythonExe = $cmd; $PythonVersion = $v; break }
    }
}
if (-not $PythonExe) {
    fail "Python 3.10+ not found.`n  Download from https://www.python.org/downloads/windows/`n  Tick 'Add Python to PATH'."
}

function Invoke-Python ([string[]]$CmdArgs) {
    $parts = $PythonExe -split ' ', 2
    if ($parts.Count -eq 2) { & $parts[0] $parts[1] @CmdArgs } else { & $PythonExe @CmdArgs }
}

$FullVer = Invoke-Python @("-c", "import sys; print(sys.version)") 2>$null
ok "Python: $PythonExe  ($FullVer)"

# -- Step 3 / 6 -- Virtual environment -----------------------------------------
section "Step 3 / 6  --  Virtual environment"

if (Test-Path $VenvDir) {
    if ($Reinstall) {
        info "Removing existing .venv (-Reinstall) ..."
        Remove-Item -Recurse -Force $VenvDir
    } elseif (-not (Test-Path $VenvPython)) {
        warn "Existing .venv appears broken -- recreating ..."
        Remove-Item -Recurse -Force $VenvDir
    } else {
        ok ".venv exists -- reusing  (-Reinstall to force rebuild)"
    }
}

if (-not (Test-Path $VenvDir)) {
    info "Creating .venv ..."
    Invoke-Python @("-m", "venv", $VenvDir)
    ok ".venv created."
}

info "Upgrading pip ..."
& $VenvPython -m pip install --upgrade pip --quiet

info "Installing dependencies ..."
info "  (first install may take several minutes)"
Write-Host ""
& $VenvPip install -r $Requirements
Write-Host ""

info "Registering 'trileaf' command ..."
& $VenvPip install -e $ScriptDir --no-deps --quiet
ok "Dependencies installed."

# -- Step 4 / 6 -- PATH --------------------------------------------------------
section "Step 4 / 6  --  PATH"

if ($Doctor) {
    info "Running environment check (-Doctor) ..."
    & $VenvPython (Join-Path $ScriptDir "scripts\check_env.py")
    exit $LASTEXITCODE
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $userPath) { $userPath = "" }
if ($userPath -notlike "*$ScriptsDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$ScriptsDir", "User")
    info "Added $ScriptsDir to user PATH (takes effect in new terminals)."
}
$env:Path = "$ScriptsDir;$env:Path"
ok "PATH updated."

# -- Step 5 / 6 -- LeafHub -----------------------------------------------------
section "Step 5 / 6  --  LeafHub"

# Detect leafhub -- check system PATH first, then venv Scripts
$LeafHubExe = Get-Command leafhub -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $LeafHubExe) {
    $VenvLeafhub = Join-Path $ScriptsDir "leafhub.exe"
    if (-not (Test-Path $VenvLeafhub)) { $VenvLeafhub = Join-Path $ScriptsDir "leafhub" }
    if (Test-Path $VenvLeafhub) { $LeafHubExe = $VenvLeafhub }
}
if (-not $LeafHubExe) {
    info "LeafHub not found -- installing (required dependency) ..."
    try {
        irm https://raw.githubusercontent.com/Rebas9512/Leafhub/main/install.ps1 | iex
    } catch {
        fail "LeafHub installation failed.`n  Install manually: https://github.com/Rebas9512/Leafhub`n  Then re-run setup.ps1"
    }
    $LeafHubExe = Get-Command leafhub -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
    if (-not $LeafHubExe) {
        fail "LeafHub installed but 'leafhub' not found in PATH.`n  Open a new terminal and re-run: setup.ps1"
    }
}
ok "LeafHub: $LeafHubExe"

# Register Trileaf project -- use manifest mode if leafhub.toml exists
if (Test-Path (Join-Path $ScriptDir "leafhub.toml")) {
    $registerArgs = @("register", $ScriptDir)
} else {
    $registerArgs = @("register", "trileaf", "--path", $ScriptDir, "--alias", "rewrite")
}
if ($Headless) { $registerArgs += "--headless" }

try {
    & $LeafHubExe @registerArgs
    if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" }
    ok "LeafHub integration complete."
} catch {
    fail "LeafHub registration failed: $_`n  Install LeafHub and retry: https://github.com/Rebas9512/Leafhub"
}

# -- Step 6 / 6 -- Detection models --------------------------------------------
section "Step 6 / 6  --  Detection models"

if ($Headless) {
    info "Headless -- skipping model download."
    info "Run later: trileaf setup"
} else {
    Write-Host "  Trileaf needs two detection models $(c $M '(~0.9 GB total)')"
    Write-Host "  $(c $M '. desklib/ai-text-detector-v1.01     (~0.5 GB)')"
    Write-Host "  $(c $M '. paraphrase-mpnet-base-v2            (~0.4 GB)')"
    Write-Host ""
    $dl = Read-Host "  Download now? [Y/n]"
    if ($dl -ne "n" -and $dl -ne "N") {
        & $TrileafExe setup --models-only
    } else {
        info "Run later: trileaf setup"
    }
}

# -- Done ----------------------------------------------------------------------
Write-Host ""
Write-Host (c $B "  Setup complete!")
Write-Host ""
Write-Host "$(c $G '  trileaf run')      # start the dashboard"
Write-Host "$(c $G '  trileaf setup')    # download detection models"
Write-Host "$(c $G '  trileaf doctor')   # environment health check"
Write-Host ""
