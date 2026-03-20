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

# ── CLI registration vars ─────────────────────────────────────────────────────
BIN_DIR="$HOME/.local/bin"
TRILEAF_BIN="$VENV_DIR/bin/trileaf"
TRILEAF_LINK="$BIN_DIR/trileaf"
ORIGINAL_PATH="${PATH:-}"
PATH_PERSISTED=0

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

# ── helpers ───────────────────────────────────────────────────────────────────

# Returns 0 when <dir> appears in <path_string>
path_has_dir() {
    case ":${1}:" in *":${2%/}:"*) return 0 ;; *) return 1 ;; esac
}

# Pick the shell RC file that interactive shells will actually source.
#   zsh              → ~/.zshrc
#   bash on macOS    → ~/.bash_profile  (Terminal.app opens login shells)
#   bash on Linux    → ~/.bashrc        (most terminal emulators, non-login)
#   fallback         → ~/.bash_profile
detect_rc_file() {
    local shell_name
    shell_name="$(basename "${SHELL:-bash}")"
    case "$shell_name" in
        zsh)
            echo "$HOME/.zshrc" ;;
        bash)
            if [[ "$OS" == "macos" ]]; then
                echo "$HOME/.bash_profile"
            elif [[ -f "$HOME/.bashrc" ]]; then
                echo "$HOME/.bashrc"
            else
                echo "$HOME/.bash_profile"
            fi
            ;;
        *)
            if [[ -f "$HOME/.bashrc" ]]; then
                echo "$HOME/.bashrc"
            else
                echo "$HOME/.bash_profile"
            fi
            ;;
    esac
}

# Ensure ~/.local/bin exists, is in the live PATH, and is persisted in the
# appropriate shell RC file.
ensure_local_bin_on_path() {
    mkdir -p "$BIN_DIR"
    export PATH="$BIN_DIR:$PATH"
    hash -r 2>/dev/null || true

    local marker='# Added by Trileaf installer'
    local line='export PATH="$HOME/.local/bin:$PATH"'
    local target
    target="$(detect_rc_file)"

    # If ~/.local/bin is already present in any common RC file, nothing to do.
    local rc
    for rc in "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.zshrc" "$HOME/.profile"; do
        if [[ -f "$rc" ]] && grep -qF '.local/bin' "$rc" 2>/dev/null; then
            PATH_PERSISTED=1
            return 0
        fi
    done

    if [[ ! -f "$target" ]] && ! touch "$target" 2>/dev/null; then
        warn "Could not create $(basename "$target") for CLI registration."
        return 0
    fi

    if [[ ! -w "$target" ]]; then
        warn "Could not update $(basename "$target") for CLI registration."
        return 0
    fi

    if printf '\n%s\n%s\n' "$marker" "$line" >> "$target"; then
        info "Added ~/.local/bin to PATH in $(basename "$target")"
        PATH_PERSISTED=1
    else
        warn "Could not update $(basename "$target") for CLI registration."
    fi
}

# ── banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}  Trileaf — Setup${NC}"
echo -e "${MUTED}  Creates a Python virtual environment and walks through model/provider setup.${NC}"
echo ""

# ── Step 1: detect OS ─────────────────────────────────────────────────────────
section "Step 1 / 5  —  Platform"

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
section "Step 2 / 5  —  Python"

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
section "Step 3 / 5  —  Virtual environment"

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

# Register the 'trileaf' entry point inside the venv
info "Registering 'trileaf' CLI entry point ..."
"$VENV_PIP" install -e . --no-deps --quiet
ok "'trileaf' entry point registered in .venv."

# ── Step 4: CLI registration (PATH) ───────────────────────────────────────────
section "Step 4 / 5  —  CLI registration"

if [[ -L "$TRILEAF_LINK" ]]; then
    existing_target="$(readlink -f "$TRILEAF_LINK" 2>/dev/null || true)"
    if [[ "$existing_target" == "$TRILEAF_BIN" ]]; then
        ok "CLI already registered: $TRILEAF_LINK"
        # Ensure PATH persistence is recorded even if symlink was pre-existing
        if path_has_dir "$ORIGINAL_PATH" "$BIN_DIR"; then
            PATH_PERSISTED=1
        else
            ensure_local_bin_on_path
        fi
    else
        # Stale symlink (e.g. from a previous clone location) — update it
        ensure_local_bin_on_path
        ln -sf "$TRILEAF_BIN" "$TRILEAF_LINK"
        ok "Updated CLI link: trileaf → $TRILEAF_BIN"
    fi
else
    ensure_local_bin_on_path
    ln -sf "$TRILEAF_BIN" "$TRILEAF_LINK"
    ok "Registered: $TRILEAF_LINK"
fi

if ! path_has_dir "$ORIGINAL_PATH" "$BIN_DIR"; then
    if [[ "$PATH_PERSISTED" -eq 1 ]]; then
        warn "$BIN_DIR not yet in your current shell's PATH."
        warn "Open a new terminal, or run:  export PATH=\"\$HOME/.local/bin:\$PATH\""
    else
        warn "Could not update your shell RC file automatically."
        warn "Add this line to your shell config, then open a new terminal:"
        warn "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
fi

# ── Step 5: onboarding wizard ─────────────────────────────────────────────────
section "Step 5 / 5  —  Onboarding"

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
if path_has_dir "$ORIGINAL_PATH" "$BIN_DIR"; then
    # ~/.local/bin was already in PATH before this run
    echo -e "  The ${GREEN}trileaf${NC} command is ready:"
    echo ""
    echo -e "    ${GREEN}trileaf run${NC}              # start the dashboard"
    echo -e "    ${GREEN}trileaf setup${NC}            # re-run model/provider setup"
    echo -e "    ${GREEN}trileaf config${NC}           # manage provider credentials"
    echo -e "    ${GREEN}trileaf doctor${NC}           # environment health check"
elif [[ "$PATH_PERSISTED" -eq 1 ]]; then
    # RC file was updated — need a new shell to pick it up
    echo "  Open a new terminal, then use the trileaf command:"
    echo ""
    echo -e "    ${GREEN}trileaf run${NC}              # start the dashboard"
    echo -e "    ${GREEN}trileaf setup${NC}            # re-run model/provider setup"
    echo -e "    ${GREEN}trileaf config${NC}           # manage provider credentials"
    echo -e "    ${GREEN}trileaf doctor${NC}           # environment health check"
else
    # PATH registration failed — fall back to venv activation instructions
    echo -e "  Activate the venv once per terminal session, then use the ${GREEN}trileaf${NC} command:"
    echo ""
    echo -e "    ${GREEN}source .venv/bin/activate${NC}"
    echo -e "    ${GREEN}trileaf run${NC}              # start the dashboard"
    echo -e "    ${GREEN}trileaf setup${NC}            # re-run model/provider setup"
    echo -e "    ${GREEN}trileaf config${NC}           # manage provider credentials"
    echo -e "    ${GREEN}trileaf doctor${NC}           # environment health check"
    echo ""
    echo "  Or invoke directly without activating:"
    echo -e "    ${MUTED}.venv/bin/trileaf run${NC}"
fi
echo ""
