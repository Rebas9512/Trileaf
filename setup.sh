#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Trileaf — Setup (macOS / Linux / WSL)
#
#  Usage (first-time):
#    git clone <repo-url> trileaf && cd trileaf
#    chmod +x setup.sh && ./setup.sh
#
#  Options:
#    --reinstall           Delete and recreate the .venv
#    --skip-onboarding     Skip the interactive model/provider wizard
#    --headless            Non-interactive / CI mode: implies --skip-onboarding,
#                          suppresses all interactive prompts.  Exit code
#                          reflects success (0) or failure (non-zero).
#    --doctor              Run environment check only, then exit.
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── ANSI colours ─────────────────────────────────────────────────────────────
if [[ -t 1 && "${NO_COLOR:-}" == "" && "${TERM:-dumb}" != "dumb" ]]; then
    BOLD='\033[1m'
    GREEN='\033[38;2;0;229;180m'
    YELLOW='\033[38;2;255;176;32m'
    RED='\033[38;2;230;57;70m'
    MUTED='\033[38;2;110;120;148m'
    NC='\033[0m'
else
    BOLD='' GREEN='' YELLOW='' RED='' MUTED='' NC=''
fi

ok()   { echo -e "${GREEN}✓${NC}  $*"; }
info() { echo -e "${MUTED}·${NC}  $*"; }
warn() { echo -e "${YELLOW}!${NC}  $*"; }
fail() { echo -e "${RED}✗${NC}  $*" >&2; exit 1; }

section() {
    echo ""
    echo -e "${BOLD}── $* ──${NC}"
}

# ── project root ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

# ── argument parsing ──────────────────────────────────────────────────────────
REINSTALL=false
SKIP_ONBOARDING=false
HEADLESS=false
DOCTOR=false

for arg in "$@"; do
    case "$arg" in
        --reinstall)        REINSTALL=true ;;
        --skip-onboarding)  SKIP_ONBOARDING=true ;;
        --headless)         HEADLESS=true; SKIP_ONBOARDING=true ;;
        --doctor)           DOCTOR=true ;;
        --help|-h)
            echo "Usage: ./setup.sh [--reinstall] [--skip-onboarding] [--headless] [--doctor]"
            exit 0
            ;;
        *)
            warn "Unknown option: $arg  (ignored)"
            ;;
    esac
done

# ── banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}  Trileaf — Setup${NC}"
echo -e "${MUTED}  Creates a Python virtual environment and walks through model/provider setup.${NC}"
echo ""

# ── Step 1: detect OS ─────────────────────────────────────────────────────────
section "Step 1 / 4  —  Platform"

OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ -n "${WSL_DISTRO_NAME:-}" || -n "${WSL_INTEROP:-}" ]]; then
    OS="wsl"
elif [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "linux"* ]]; then
    OS="linux"
fi

if [[ "$OS" == "unknown" ]]; then
    fail "Unsupported operating system ($OSTYPE).  Run setup.ps1 on Windows."
fi
ok "Platform: $OS"

# ── Step 2: find Python 3.10+ ─────────────────────────────────────────────────
section "Step 2 / 4  —  Python"

PYTHON=""
find_python() {
    local ver maj min
    for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            continue
        fi
        ver="$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
        if [[ -z "$ver" ]]; then
            continue
        fi
        maj="${ver%%.*}"
        min="${ver##*.}"
        if [[ "$maj" -ge 3 && "$min" -ge 10 ]]; then
            echo "$cmd"
            return 0
        fi
    done
    return 1
}

if ! PYTHON="$(find_python)"; then
    echo ""
    fail "Python 3.10+ is required but was not found in PATH.

Install it from https://www.python.org/downloads/ and re-run this script.
  macOS:  brew install python@3.12
  Ubuntu: sudo apt install python3.12 python3.12-venv"
fi

PYTHON_VERSION="$("$PYTHON" -c 'import sys; print(sys.version)' 2>/dev/null)"
ok "Python: $PYTHON  ($PYTHON_VERSION)"

# ── Step 3: create / reuse venv ───────────────────────────────────────────────
section "Step 3 / 4  —  Virtual environment"

if [[ -d "$VENV_DIR" ]]; then
    if [[ "$REINSTALL" == "true" ]]; then
        info "Removing existing .venv (--reinstall)"
        rm -rf "$VENV_DIR"
    else
        # Verify the existing venv has a working Python (guard against stale venvs)
        if [[ -x "$VENV_DIR/bin/python" ]]; then
            ok ".venv exists — reusing"
            info "  (pass --reinstall to force a clean rebuild)"
        else
            warn "Existing .venv appears broken — recreating"
            rm -rf "$VENV_DIR"
        fi
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating .venv with $PYTHON ..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok ".venv created: $VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# Upgrade pip silently
info "Upgrading pip ..."
"$VENV_PYTHON" -m pip install --upgrade pip --quiet

# Install dependencies
info "Installing dependencies from requirements.txt ..."
info "  (torch + transformers may take several minutes on first install)"
echo ""
"$VENV_PIP" install -r "$REQUIREMENTS"
echo ""
ok "Dependencies installed."

# Register the 'trileaf' command inside the venv
info "Registering 'trileaf' CLI command ..."
"$VENV_PIP" install -e . --no-deps --quiet
ok "'trileaf' command registered."

# ── Step 4: onboarding wizard ─────────────────────────────────────────────────
section "Step 4 / 4  —  Onboarding"

if [[ "$DOCTOR" == "true" ]]; then
    info "Running environment check (--doctor) ..."
    "$VENV_PYTHON" "$SCRIPT_DIR/scripts/check_env.py"
    exit $?
fi

if [[ "$SKIP_ONBOARDING" == "true" ]]; then
    if [[ "$HEADLESS" == "true" ]]; then
        info "Headless mode — skipping interactive onboarding wizard."
        info "Run environment check:  .venv/bin/python scripts/check_env.py"
    else
        info "Skipping onboarding wizard (--skip-onboarding)"
    fi
else
    # Capture the real exit code — DO NOT use `if ! cmd; then ... $?` because
    # $? after a negated condition reflects the condition result, not cmd's exit.
    ONBOARD_EXIT=0
    "$VENV_PYTHON" "$SCRIPT_DIR/scripts/onboarding.py" || ONBOARD_EXIT=$?
    if [[ $ONBOARD_EXIT -ne 0 ]]; then
        echo ""
        warn "Onboarding did not complete (exit code $ONBOARD_EXIT)."
        warn "Your virtual environment and dependencies are ready."
        warn "Re-run onboarding at any time with:"
        warn "  .venv/bin/python scripts/onboarding.py"
    fi
fi

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}  Setup complete!${NC}"
echo ""
echo -e "  Activate the venv once per terminal session, then use the ${GREEN}trileaf${NC} command:"
echo ""
echo -e "    ${GREEN}source .venv/bin/activate${NC}"
echo -e "    ${GREEN}trileaf run${NC}              # start the dashboard"
echo -e "    ${GREEN}trileaf onboard${NC}          # re-run model/provider setup"
echo -e "    ${GREEN}trileaf config${NC}           # manage provider profiles"
echo -e "    ${GREEN}trileaf doctor${NC}           # environment health check"
echo ""
echo "  Or invoke directly without activating:"
echo -e "    ${MUTED}.venv/bin/trileaf run${NC}"
echo ""
