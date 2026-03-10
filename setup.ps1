# ─────────────────────────────────────────────────────────────────────────────
#  Trileaf — Setup (Windows PowerShell)
#
#  Usage (first-time):
#    git clone <repo-url> trileaf
#    cd trileaf
#    powershell -ExecutionPolicy Bypass -File setup.ps1
#
#  Options:
#    -Reinstall          Delete and recreate the .venv
#    -SkipOnboarding     Skip the interactive model/provider wizard
#    -Headless           Non-interactive / CI mode: implies -SkipOnboarding.
#                        Exit code reflects success (0) or failure (non-zero).
#    -Doctor             Run environment check only, then exit.
#
# ─────────────────────────────────────────────────────────────────────────────
param(
    [switch]$Reinstall,
    [switch]$SkipOnboarding,
    [switch]$Headless,
    [switch]$Doctor
)

$ErrorActionPreference = "Stop"

# ── ANSI colour helpers ───────────────────────────────────────────────────────
$SupportsColor = $Host.UI.SupportsVirtualTerminal -and $env:NO_COLOR -eq $null
function c($code, $text) {
    if ($SupportsColor) { return "${code}${text}`e[0m" }
    return $text
}
$G = "`e[38;2;0;229;180m"     # green
$Y = "`e[38;2;255;176;32m"    # yellow
$R = "`e[38;2;230;57;70m"     # red
$M = "`e[38;2;110;120;148m"   # muted
$B = "`e[1m"                  # bold

function ok   ($msg) { Write-Host "$(c $G '✓')  $msg" }
function info ($msg) { Write-Host "$(c $M '·')  $msg" }
function warn ($msg) { Write-Host "$(c $Y '!')  $msg" }
function fail ($msg) { Write-Host "$(c $R '✗')  $msg" -ForegroundColor Red; exit 1 }

function section ($title) {
    Write-Host ""
    Write-Host (c $B "── $title ──")
}

# ── project root ──────────────────────────────────────────────────────────────
$ScriptDir   = $PSScriptRoot
$VenvDir     = Join-Path $ScriptDir ".venv"
$VenvPython  = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip     = Join-Path $VenvDir "Scripts\pip.exe"
$Requirements = Join-Path $ScriptDir "requirements.txt"

# ── banner ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host (c $B "  Trileaf — Setup")
Write-Host (c $M "  Creates a Python virtual environment and walks through model/provider setup.")
Write-Host ""

# ── Step 1: platform check ────────────────────────────────────────────────────
section "Step 1 / 4  —  Platform"

if ($IsWindows -eq $false -and $env:OS -notmatch "Windows") {
    fail "This script is for Windows.  Use setup.sh on macOS/Linux."
}
ok "Platform: Windows"

# Ensure scripts can run in this session (execution policy)
$policy = Get-ExecutionPolicy -Scope Process
if ($policy -eq "Restricted" -or $policy -eq "AllSigned") {
    try {
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
        ok "Set execution policy to RemoteSigned for this session"
    } catch {
        warn "Could not set execution policy automatically."
        warn "If pip or venv activation fails, run first:"
        warn "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process"
    }
}

# ── Step 2: find Python 3.10+ ─────────────────────────────────────────────────
section "Step 2 / 4  —  Python"

function Get-PythonVersion ($cmd) {
    try {
        $raw = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $raw) { return $raw.Trim() }
    } catch {}
    return $null
}

function Is-SufficientVersion ($ver) {
    if (-not $ver) { return $false }
    $parts = $ver -split '\.'
    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    return ($major -gt 3) -or ($major -eq 3 -and $minor -ge 10)
}

$PythonExe = $null
# Try the Windows py launcher first (supports py -3.12 style)
foreach ($spec in @("3.13", "3.12", "3.11", "3.10")) {
    try {
        $v = & py "-$spec" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and (Is-SufficientVersion $v)) {
            $PythonExe = "py -$spec"
            $PythonVersion = $v
            break
        }
    } catch {}
}

# Fall back to python3 / python on PATH
if (-not $PythonExe) {
    foreach ($cmd in @("python3", "python")) {
        $v = Get-PythonVersion $cmd
        if (Is-SufficientVersion $v) {
            $PythonExe = $cmd
            $PythonVersion = $v
            break
        }
    }
}

if (-not $PythonExe) {
    fail @"
Python 3.10+ is required but was not found.

Install it from https://www.python.org/downloads/windows/
  - Check 'Add Python to PATH' during installation.
  - The Windows py launcher is installed automatically.

Then re-run:  powershell -ExecutionPolicy Bypass -File setup.ps1
"@
}

$FullVersion = & ($PythonExe -split ' ')[0] @(($PythonExe -split ' ')[1..99]) `
    -c "import sys; print(sys.version)" 2>$null
ok "Python: $PythonExe  ($FullVersion)"

# ── helper: invoke python through the py launcher if needed ───────────────────
# NOTE: $Args is a PowerShell reserved automatic variable — use $CmdArgs instead.
function Invoke-Python ([string[]]$CmdArgs) {
    $parts = $PythonExe -split ' ', 2
    if ($parts.Count -eq 2) {
        & $parts[0] $parts[1] @CmdArgs
    } else {
        & $PythonExe @CmdArgs
    }
}

# ── Step 3: create / reuse venv ───────────────────────────────────────────────
section "Step 3 / 4  —  Virtual environment"

if (Test-Path $VenvDir) {
    if ($Reinstall) {
        info "Removing existing .venv (--reinstall)"
        Remove-Item -Recurse -Force $VenvDir
    } elseif (Test-Path $VenvPython) {
        ok ".venv exists — reusing"
        info "  (pass -Reinstall to force a clean rebuild)"
    } else {
        warn "Existing .venv appears broken — recreating"
        Remove-Item -Recurse -Force $VenvDir
    }
}

if (-not (Test-Path $VenvDir)) {
    info "Creating .venv ..."
    Invoke-Python @("-m", "venv", $VenvDir)
    ok ".venv created: $VenvDir"
}

# Upgrade pip
info "Upgrading pip ..."
& $VenvPython -m pip install --upgrade pip --quiet

# Install dependencies
info "Installing dependencies from requirements.txt ..."
info "  (torch + transformers may take several minutes on first install)"
Write-Host ""
& $VenvPip install -r $Requirements
Write-Host ""
ok "Dependencies installed."

# Register the 'trileaf' command inside the venv
info "Registering 'trileaf' CLI command ..."
& $VenvPip install -e . --no-deps --quiet
ok "'trileaf' command registered."

# ── Step 4: onboarding wizard ─────────────────────────────────────────────────
section "Step 4 / 4  —  Onboarding"

# -Headless implies -SkipOnboarding
if ($Headless) { $SkipOnboarding = $true }

if ($Doctor) {
    info "Running environment check (-Doctor) ..."
    & $VenvPython (Join-Path $ScriptDir "scripts\check_env.py")
    exit $LASTEXITCODE
}

if ($SkipOnboarding) {
    if ($Headless) {
        info "Headless mode — skipping interactive onboarding wizard."
        info "Run environment check:  .venv\Scripts\python scripts\check_env.py"
    } else {
        info "Skipping onboarding wizard (-SkipOnboarding)"
    }
} else {
    $onboardExit = 0
    try {
        & $VenvPython (Join-Path $ScriptDir "scripts\onboarding.py")
        $onboardExit = $LASTEXITCODE
    } catch {
        $onboardExit = 1
    }
    if ($onboardExit -ne 0) {
        Write-Host ""
        warn "Onboarding did not complete (exit code $onboardExit)."
        warn "Your virtual environment and dependencies are ready."
        warn "Re-run onboarding at any time with:"
        warn "  .venv\Scripts\python scripts\onboarding.py"
    }
}

# ── done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host (c $B "  Setup complete!")
Write-Host ""
Write-Host "  Activate the venv once per terminal session, then use the trileaf command:"
Write-Host ""
Write-Host (c $G "    .venv\Scripts\Activate.ps1")
Write-Host (c $G "    trileaf run              # start the dashboard")
Write-Host (c $G "    trileaf setup            # re-run model/provider setup")
Write-Host (c $G "    trileaf config           # manage provider profiles")
Write-Host (c $G "    trileaf doctor           # environment health check")
Write-Host ""
Write-Host "  Or invoke directly without activating:"
Write-Host (c $M "    .venv\Scripts\trileaf.exe run")
Write-Host ""
