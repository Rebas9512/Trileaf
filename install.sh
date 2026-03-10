#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  Trileaf — One-liner Installer
#
#  curl -fsSL https://raw.githubusercontent.com/Rebas9512/Trileaf/main/install.sh | bash
#
#  Environment variables:
#    TRILEAF_DIR=<path>     Install directory  (default: ~/.trileaf)
#    TRILEAF_REPO_URL=<url> Clone URL          (default: GitHub repo)
#    TRILEAF_NO_ONBOARD=1   Skip the interactive setup wizard
#    NO_COLOR=1             Disable colour output
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

TRILEAF_DIR="${TRILEAF_DIR:-$HOME/.trileaf}"
REPO_URL="${TRILEAF_REPO_URL:-https://github.com/Rebas9512/Trileaf.git}"
BIN_DIR="$HOME/.local/bin"
VENV_DIR="$TRILEAF_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"
TRILEAF_BIN="$VENV_DIR/bin/trileaf"
TRILEAF_LINK="$BIN_DIR/trileaf"
ORIGINAL_PATH="${PATH:-}"

# ── Colours ───────────────────────────────────────────────────────────────────
if [[ -n "${NO_COLOR:-}" || "${TERM:-dumb}" == "dumb" ]]; then
    BOLD='' GREEN='' YELLOW='' RED='' MUTED='' NC=''
else
    BOLD='\033[1m'
    GREEN='\033[38;2;0;229;180m'
    YELLOW='\033[38;2;255;176;32m'
    RED='\033[38;2;230;57;70m'
    MUTED='\033[38;2;110;120;148m'
    NC='\033[0m'
fi

ok()      { echo -e "${GREEN}✓${NC}  $*"; }
info()    { echo -e "${MUTED}·${NC}  $*"; }
warn()    { echo -e "${YELLOW}!${NC}  $*"; }
fail()    { echo -e "${RED}✗${NC}  $*" >&2; exit 1; }
section() { echo ""; echo -e "${BOLD}── $* ──${NC}"; }

# ── Helpers ───────────────────────────────────────────────────────────────────

# Returns 0 when running non-interactively (piped curl | bash, CI, etc.)
is_non_interactive() {
    [[ "${TRILEAF_NO_ONBOARD:-0}" == "1" ]] && return 0
    [[ ! -t 0 || ! -t 1 ]] && return 0
    return 1
}

# Returns 0 when <dir> appears in <path>
path_has_dir() {
    case ":${1}:" in *":${2%/}:"*) return 0 ;; *) return 1 ;; esac
}

# Ensures ~/.local/bin exists, is on PATH now, and is written to shell rc files
ensure_local_bin_on_path() {
    mkdir -p "$BIN_DIR"
    export PATH="$BIN_DIR:$PATH"
    hash -r 2>/dev/null || true

    local marker='# Added by Trileaf installer'
    local line='export PATH="$HOME/.local/bin:$PATH"'
    local have_persisted_path=0
    local rc

    for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile"; do
        [[ -f "$rc" ]] || continue
        if grep -qF '.local/bin' "$rc" 2>/dev/null; then
            have_persisted_path=1
            continue
        fi
        printf '\n%s\n%s\n' "$marker" "$line" >> "$rc"
        info "Added ~/.local/bin to PATH in $(basename "$rc")"
        have_persisted_path=1
    done

    if [[ "$have_persisted_path" -eq 0 ]]; then
        rc="$HOME/.profile"
        touch "$rc"
        printf '\n%s\n%s\n' "$marker" "$line" >> "$rc"
        info "Added ~/.local/bin to PATH in $(basename "$rc")"
    fi
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}  Trileaf — Installer${NC}"
echo -e "${MUTED}  Install path: $TRILEAF_DIR${NC}"
echo ""

# ── Step 1: OS ────────────────────────────────────────────────────────────────
section "Platform"
OS="unknown"
if   [[ "$OSTYPE" == "darwin"* ]];                                     then OS="macos"
elif [[ -n "${WSL_DISTRO_NAME:-}" || -n "${WSL_INTEROP:-}" ]];         then OS="wsl"
elif [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "linux"* ]];         then OS="linux"
fi

if [[ "$OS" == "unknown" ]]; then
    fail "Unsupported OS ($OSTYPE).\nOn Windows use: irm https://raw.githubusercontent.com/Rebas9512/Trileaf/main/install.ps1 | iex"
fi
ok "Platform: $OS"

# ── Step 2: Python ────────────────────────────────────────────────────────────
section "Python"
PYTHON=""
for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
    command -v "$cmd" >/dev/null 2>&1 || continue
    ver="$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    [[ -z "$ver" ]] && continue
    maj="${ver%%.*}"; min="${ver##*.}"
    if [[ "$maj" -ge 3 && "$min" -ge 10 ]]; then PYTHON="$cmd"; break; fi
done

if [[ -z "$PYTHON" ]]; then
    fail "Python 3.10+ not found.\n  macOS:  brew install python@3.12\n  Ubuntu: sudo apt install python3.12 python3.12-venv"
fi
ok "Python: $PYTHON ($("$PYTHON" -c 'import sys; print(sys.version.split()[0])'))"

command -v git >/dev/null 2>&1 || fail "git is required but not found."

# ── Step 3: Clone / update ────────────────────────────────────────────────────
section "Installing Trileaf"
if [[ -d "$TRILEAF_DIR/.git" ]]; then
    info "Existing installation found — updating..."
    git -C "$TRILEAF_DIR" pull --ff-only --quiet
    ok "Updated to latest."
else
    info "Cloning into $TRILEAF_DIR ..."
    git clone --depth=1 "$REPO_URL" "$TRILEAF_DIR" --quiet
    ok "Cloned."
fi

# ── Step 4: Virtual environment ───────────────────────────────────────────────
section "Virtual environment"
if [[ ! -x "$VENV_PYTHON" ]]; then
    info "Creating venv..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Venv created."
else
    ok "Venv exists — reusing."
fi

info "Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip --quiet

# On Linux without a CUDA GPU, install the CPU-only torch wheel first to avoid
# pulling the multi-GB CUDA bundle that pip selects from PyPI by default.
if [[ "$OS" != "macos" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
    info "No CUDA detected — installing CPU-only torch..."
    "$VENV_PIP" install "torch>=2.4.0" \
        --index-url https://download.pytorch.org/whl/cpu --quiet
fi

info "Installing dependencies..."
"$VENV_PIP" install -r "$TRILEAF_DIR/requirements.txt" --quiet

# Register the 'trileaf' CLI entry point inside the venv.
"$VENV_PIP" install -e "$TRILEAF_DIR" --no-deps --quiet
ok "Dependencies installed."

# ── Step 5: PATH ──────────────────────────────────────────────────────────────
section "PATH"
ensure_local_bin_on_path
ln -sf "$TRILEAF_BIN" "$TRILEAF_LINK"
ok "Linked: $TRILEAF_LINK"

if ! path_has_dir "$ORIGINAL_PATH" "$BIN_DIR"; then
    warn "$BIN_DIR is not in your current shell's PATH."
    warn "Open a new terminal, or run now:  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

# ── Step 6: Onboarding ────────────────────────────────────────────────────────
section "Setup"
if is_non_interactive; then
    info "Non-interactive session — skipping wizard."
    info "Run  trileaf setup  to configure models and providers."
else
    "$TRILEAF_BIN" setup
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}  Trileaf installed!${NC}"
echo ""
if path_has_dir "${PATH}" "$BIN_DIR"; then
    echo -e "  ${GREEN}trileaf setup${NC}    # configure models and providers"
    echo -e "  ${GREEN}trileaf run${NC}      # start the dashboard"
else
    echo "  Open a new terminal, then:"
    echo -e "    ${GREEN}trileaf setup${NC}    # configure models and providers"
    echo -e "    ${GREEN}trileaf run${NC}      # start the dashboard"
fi
echo ""
