# ------------------------------------------------------------------------------
#  Trileaf -- Windows One-liner Installer
#
#  irm https://raw.githubusercontent.com/Rebas9512/Trileaf/main/install.ps1 | iex
#
#  This script selects an install directory, clones the repo, records install
#  metadata, then delegates all further setup to setup.ps1 inside the clone.
#
#  Parameters:
#    -InstallDir <path>   Install directory  (default: $HOME\trileaf)
#    -Headless            Non-interactive / CI mode
#  Environment variables:
#    TRILEAF_DIR          Override the install directory
#    TRILEAF_REPO_URL     Override the git clone URL
# ------------------------------------------------------------------------------
param(
    [string]$InstallDir = "",
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$ConfigDir       = Join-Path $env:USERPROFILE ".trileaf"
$InstallMetaPath = Join-Path $ConfigDir "install.json"
$DefaultInstallDir = Join-Path $env:USERPROFILE "trileaf"
$RepoUrl = if ($env:TRILEAF_REPO_URL) { $env:TRILEAF_REPO_URL } `
           else { "https://github.com/Rebas9512/Trileaf.git" }

# Minimal colour helpers (only needed before setup.ps1 takes over)
$ESC = [char]0x1b
$GREEN = "${ESC}[38;2;0;229;180m"; $RED = "${ESC}[38;2;230;57;70m"
$MUTED = "${ESC}[38;2;110;120;148m"; $BOLD = "${ESC}[1m"; $NC = "${ESC}[0m"
function Write-Ok($msg)   { Write-Host "${GREEN}+${NC}  $msg" }
function Write-Info($msg) { Write-Host "${MUTED}.${NC}  $msg" }
function Write-Fail($msg) { Write-Host "${RED}x${NC}  $msg"; exit 1 }

function Test-DirHasEntries([string]$Dir) {
    if (-not (Test-Path $Dir -PathType Container)) { return $false }
    return $null -ne (Get-ChildItem -Force -LiteralPath $Dir | Select-Object -First 1)
}

# -- Select install directory --------------------------------------------------
if (-not $InstallDir) {
    if ($env:TRILEAF_DIR) {
        $InstallDir = $env:TRILEAF_DIR
    } else {
        $canPrompt = $false
        try { $canPrompt = -not [Console]::IsInputRedirected } catch {}
        if ($canPrompt -and -not $Headless) {
            $raw = Read-Host "Install directory [$DefaultInstallDir]"
            $InstallDir = if ($raw) { $raw } else { $DefaultInstallDir }
        } else {
            $InstallDir = $DefaultInstallDir
        }
    }
}

# Expand ~ prefix
$InstallDir = $InstallDir.Trim()
if ($InstallDir -eq "~") { $InstallDir = $env:USERPROFILE }
elseif ($InstallDir.StartsWith("~\")) { $InstallDir = Join-Path $env:USERPROFILE $InstallDir.Substring(2) }
$InstallDir = [IO.Path]::GetFullPath($InstallDir)

$resolvedConfig = [IO.Path]::GetFullPath($ConfigDir)
if ($InstallDir.TrimEnd('\') -eq $resolvedConfig.TrimEnd('\')) {
    Write-Fail "Install directory cannot be $ConfigDir (reserved for Trileaf config)."
}

# Redirect into subdirectory if target is non-empty and not a git repo
if (-not (Test-Path (Join-Path $InstallDir ".git"))) {
    if ((Test-Path $InstallDir -PathType Container) -and (Test-DirHasEntries $InstallDir)) {
        Write-Info "Target is non-empty -- using subdirectory: $InstallDir\trileaf"
        $InstallDir = [IO.Path]::GetFullPath((Join-Path $InstallDir "trileaf"))
    }
}

# -- Banner --------------------------------------------------------------------
Write-Host ""
Write-Host "${BOLD}  Trileaf -- Installer${NC}"
Write-Host "${MUTED}  Install path: $InstallDir${NC}"
Write-Host ""

# -- Prerequisites -------------------------------------------------------------
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Fail "git is required.`n  Install: winget install Git.Git  or  https://git-scm.com"
}

# -- Clone / update ------------------------------------------------------------
if (Test-Path (Join-Path $InstallDir ".git")) {
    Write-Info "Existing installation found -- updating..."
    git -C $InstallDir pull --ff-only --quiet
    Write-Ok "Updated."
} else {
    Write-Info "Cloning into $InstallDir ..."
    git clone --depth=1 $RepoUrl $InstallDir --quiet
    Write-Ok "Cloned."
}

# -- Write install metadata ----------------------------------------------------
New-Item -ItemType Directory -Path $ConfigDir -Force | Out-Null
@{ install_method = "one_liner"; install_dir = $InstallDir } |
    ConvertTo-Json | Set-Content -Path $InstallMetaPath -Encoding UTF8

# -- Delegate to setup.ps1 ----------------------------------------------------
$SetupPs1 = Join-Path $InstallDir "setup.ps1"
if (-not (Test-Path $SetupPs1)) { Write-Fail "setup.ps1 not found in $InstallDir." }

$setupArgs = @("-FromInstaller")
if ($Headless) { $setupArgs += "-Headless" }

& powershell -ExecutionPolicy Bypass -File $SetupPs1 @setupArgs
exit $LASTEXITCODE
