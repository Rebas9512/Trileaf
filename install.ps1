# ──────────────────────────────────────────────────────────────────────────────
#  Trileaf — Windows One-liner Installer
#
#  irm https://raw.githubusercontent.com/Rebas9512/Trileaf/main/install.ps1 | iex
#
#  To pass parameters use the scriptblock form:
#    & ([scriptblock]::Create((irm https://raw.githubusercontent.com/Rebas9512/Trileaf/main/install.ps1))) -NoOnboard
#
#  Parameters:
#    -InstallDir <path>   Install directory  (default: $HOME\trileaf)
#    -NoOnboard           Skip the interactive setup wizard
#  Environment variables:
#    TRILEAF_DIR          Override the install directory
#    TRILEAF_REPO_URL     Override the git clone URL
#    TRILEAF_NO_ONBOARD=1 Skip the interactive setup wizard
# ──────────────────────────────────────────────────────────────────────────────
param(
    [string]$InstallDir = "",
    [switch]$NoOnboard
)

$ErrorActionPreference = "Stop"
$ConfigDir = Join-Path $env:USERPROFILE ".trileaf"
$InstallMetaPath = Join-Path $ConfigDir "install.json"
$DefaultInstallDir = Join-Path $env:USERPROFILE "trileaf"

# ANSI colours — supported in Windows Terminal and PowerShell 7+
$GREEN  = "`e[38;2;0;229;180m"
$YELLOW = "`e[38;2;255;176;32m"
$RED    = "`e[38;2;230;57;70m"
$MUTED  = "`e[38;2;110;120;148m"
$BOLD   = "`e[1m"
$NC     = "`e[0m"

function Write-Ok($msg)      { Microsoft.PowerShell.Utility\Write-Host "${GREEN}√${NC}  $msg" }
function Write-Info($msg)    { Microsoft.PowerShell.Utility\Write-Host "${MUTED}·${NC}  $msg" }
function Write-Warn($msg)    { Microsoft.PowerShell.Utility\Write-Host "${YELLOW}!${NC}  $msg" }
function Write-Section($msg) { Microsoft.PowerShell.Utility\Write-Host ""; Microsoft.PowerShell.Utility\Write-Host "${BOLD}── $msg ──${NC}" }
function Write-Fail($msg)    { Microsoft.PowerShell.Utility\Write-Host "${RED}x${NC}  $msg"; exit 1 }

$SkipOnboard = $NoOnboard -or $env:TRILEAF_NO_ONBOARD -eq "1"

if (-not $InstallDir) {
    if ($env:TRILEAF_DIR) {
        $InstallDir = $env:TRILEAF_DIR
    } else {
        $canPrompt = $true
        try {
            $canPrompt = -not [Console]::IsInputRedirected
        } catch {
            $canPrompt = $true
        }
        if ($canPrompt) {
            $raw = Read-Host "Install directory [$DefaultInstallDir]"
            if ($raw) {
                $InstallDir = $raw
            } else {
                $InstallDir = $DefaultInstallDir
            }
        } else {
            $InstallDir = $DefaultInstallDir
        }
    }
}

$InstallDir = $InstallDir.Trim()
if ($InstallDir.StartsWith('~\')) {
    $InstallDir = Join-Path $env:USERPROFILE $InstallDir.Substring(2)
} elseif ($InstallDir -eq "~") {
    $InstallDir = $env:USERPROFILE
}

$resolvedInstallDir = [IO.Path]::GetFullPath($InstallDir)
$resolvedConfigDir = [IO.Path]::GetFullPath($ConfigDir)
if ($resolvedInstallDir.TrimEnd('\') -eq $resolvedConfigDir.TrimEnd('\')) {
    Write-Fail "Install directory cannot be $ConfigDir because that location is reserved for Trileaf JSON config files."
}
$InstallDir = $resolvedInstallDir

Microsoft.PowerShell.Utility\Write-Host ""
Microsoft.PowerShell.Utility\Write-Host "${BOLD}  Trileaf — Installer${NC}"
Microsoft.PowerShell.Utility\Write-Host "${MUTED}  Install path: $InstallDir${NC}"
Microsoft.PowerShell.Utility\Write-Host "${MUTED}  Config path:  $ConfigDir${NC}"
Microsoft.PowerShell.Utility\Write-Host ""

# ── Execution policy ──────────────────────────────────────────────────────────
$policy = Get-ExecutionPolicy
if ($policy -eq "Restricted" -or $policy -eq "AllSigned") {
    try {
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
        Write-Info "Execution policy set to RemoteSigned for this session."
    } catch {
        Write-Fail "Cannot set execution policy. Run as Administrator:`n  Set-ExecutionPolicy RemoteSigned -Scope CurrentUser"
    }
}

# ── Python ────────────────────────────────────────────────────────────────────
Write-Section "Python"

function Find-Python {
    foreach ($cmd in @("python3.13","python3.12","python3.11","python3.10","python3","python")) {
        try {
            $result = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($result) {
                $parts = $result.Trim().Split(".")
                if ([int]$parts[0] -ge 3 -and [int]$parts[1] -ge 10) { return $cmd }
            }
        } catch {}
    }
    return $null
}

$Python = Find-Python
if (-not $Python) {
    Write-Fail "Python 3.10+ not found.`n  Download from https://www.python.org/downloads/ (tick 'Add Python to PATH')"
}
$PyVer = & $Python -c "import sys; print(sys.version.split()[0])" 2>$null
Write-Ok "Python: $Python ($PyVer)"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Fail "git is required.`n  Install: winget install Git.Git  or  https://git-scm.com"
}

# ── Clone / update ────────────────────────────────────────────────────────────
Write-Section "Installing Trileaf"

$RepoUrl = if ($env:TRILEAF_REPO_URL) {
    $env:TRILEAF_REPO_URL
} else {
    "https://github.com/Rebas9512/Trileaf.git"
}

if (Test-Path (Join-Path $InstallDir ".git")) {
    Write-Info "Existing installation found — updating..."
    git -C $InstallDir pull --ff-only --quiet
    Write-Ok "Updated to latest."
} else {
    Write-Info "Cloning into $InstallDir ..."
    git clone --depth=1 $RepoUrl $InstallDir --quiet
    Write-Ok "Cloned."
}

New-Item -ItemType Directory -Path $ConfigDir -Force | Out-Null
@{
    install_method = "one_liner"
    install_dir = $InstallDir
} | ConvertTo-Json | Set-Content -Path $InstallMetaPath -Encoding UTF8

# ── Virtual environment ───────────────────────────────────────────────────────
Write-Section "Virtual environment"

$VenvDir     = Join-Path $InstallDir ".venv"
$VenvPython  = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip     = Join-Path $VenvDir "Scripts\pip.exe"
$TrileafExe  = Join-Path $VenvDir "Scripts\trileaf.exe"
$ScriptsDir  = Join-Path $VenvDir "Scripts"

if (-not (Test-Path $VenvPython)) {
    Write-Info "Creating venv..."
    & $Python -m venv $VenvDir
    Write-Ok "Venv created."
} else {
    Write-Ok "Venv exists — reusing."
}

Write-Info "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip --quiet

Write-Info "Installing dependencies..."
& $VenvPip install -r (Join-Path $InstallDir "requirements.txt") --quiet

# Register the 'trileaf' console-scripts entry point.
& $VenvPip install -e $InstallDir --no-deps --quiet
Write-Ok "Dependencies installed."

# ── PATH ──────────────────────────────────────────────────────────────────────
Write-Section "PATH"

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $userPath) { $userPath = "" }

if ($userPath -notlike "*$ScriptsDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$ScriptsDir", "User")
    Write-Info "Added $ScriptsDir to user PATH (takes effect in new terminals)."
}

# Make trileaf available immediately in this session.
$env:Path = "$ScriptsDir;$env:Path"
Write-Ok "PATH updated."

# ── Onboarding ────────────────────────────────────────────────────────────────
Write-Section "Setup"

if ($SkipOnboard) {
    if ($NoOnboard) {
        Write-Info "Skipping setup wizard (-NoOnboard)."
    } else {
        Write-Info "Skipping setup wizard (TRILEAF_NO_ONBOARD=1)."
    }
    Write-Info "Run 'trileaf setup' to configure models and providers."
} elseif (Test-Path $TrileafExe) {
    try {
        & $TrileafExe setup
    } catch {
        Write-Warn "Setup wizard exited with an error: $_"
        Write-Warn "Re-run at any time with:  trileaf setup"
    }
} else {
    Write-Warn "trileaf.exe not found at $TrileafExe — run 'trileaf setup' manually."
}

# ── Done ──────────────────────────────────────────────────────────────────────
Microsoft.PowerShell.Utility\Write-Host ""
Microsoft.PowerShell.Utility\Write-Host "${BOLD}  Trileaf installed!${NC}"
Microsoft.PowerShell.Utility\Write-Host ""

if ($env:Path -like "*$ScriptsDir*") {
    Microsoft.PowerShell.Utility\Write-Host "  ${GREEN}trileaf setup${NC}    # configure models and providers"
    Microsoft.PowerShell.Utility\Write-Host "  ${GREEN}trileaf run${NC}      # start the dashboard"
} else {
    Microsoft.PowerShell.Utility\Write-Host "  Open a new terminal, then:"
    Microsoft.PowerShell.Utility\Write-Host "    ${GREEN}trileaf setup${NC}    # configure models and providers"
    Microsoft.PowerShell.Utility\Write-Host "    ${GREEN}trileaf run${NC}      # start the dashboard"
}
Microsoft.PowerShell.Utility\Write-Host ""
