#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  Trileaf — Setup  (macOS / Linux / WSL)
#
#  Canonical setup script. Called directly by developers, or by install.sh
#  after cloning the repository.
#
#  Usage (after git clone):
#    chmod +x setup.sh && ./setup.sh
#
#  Options:
#    --reinstall      Delete and recreate the .venv
#    --headless       Non-interactive / CI mode; skips all prompts
#    --doctor         Run environment check only, then exit
#    --from-installer Internal flag set by install.sh (adjusts banner only)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
BIN_DIR="$HOME/.local/bin"
TRILEAF_BIN="$VENV_DIR/bin/trileaf"
TRILEAF_LINK="$BIN_DIR/trileaf"
ORIGINAL_PATH="${PATH:-}"
PATH_PERSISTED=0

# ── Colours ───────────────────────────────────────────────────────────────────
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-dumb}" != "dumb" ]]; then
    BOLD='\033[1m'; GREEN='\033[38;2;0;229;180m'; YELLOW='\033[38;2;255;176;32m'
    RED='\033[38;2;230;57;70m'; MUTED='\033[38;2;110;120;148m'; NC='\033[0m'
else
    BOLD='' GREEN='' YELLOW='' RED='' MUTED='' NC=''
fi

ok()      { echo -e "${GREEN}✓${NC}  $*"; }
info()    { echo -e "${MUTED}·${NC}  $*"; }
warn()    { echo -e "${YELLOW}!${NC}  $*"; }
fail()    { echo -e "${RED}✗${NC}  $*" >&2; exit 1; }
section() { echo ""; echo -e "${BOLD}── $* ──${NC}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
REINSTALL=false
HEADLESS=false
DOCTOR=false
FROM_INSTALLER=false

for arg in "$@"; do
    case "$arg" in
        --reinstall)       REINSTALL=true ;;
        --headless)        HEADLESS=true ;;
        --doctor)          DOCTOR=true ;;
        --from-installer)  FROM_INSTALLER=true ;;
        --help|-h)
            echo "Usage: ./setup.sh [--reinstall] [--headless] [--doctor]"
            exit 0
            ;;
        *) warn "Unknown option: $arg  (ignored)" ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────

path_has_dir() {
    case ":${1}:" in *":${2%/}:"*) return 0 ;; *) return 1 ;; esac
}

detect_rc_file() {
    local shell_name
    shell_name="$(basename "${SHELL:-bash}")"
    case "$shell_name" in
        zsh)  echo "$HOME/.zshrc" ;;
        bash)
            if [[ "$OS" == "macos" ]]; then echo "$HOME/.bash_profile"
            elif [[ -f "$HOME/.bashrc" ]]; then echo "$HOME/.bashrc"
            else echo "$HOME/.bash_profile"
            fi ;;
        *)
            if [[ -f "$HOME/.bashrc" ]]; then echo "$HOME/.bashrc"
            else echo "$HOME/.bash_profile"
            fi ;;
    esac
}

ensure_local_bin_on_path() {
    mkdir -p "$BIN_DIR"
    export PATH="$BIN_DIR:$PATH"
    hash -r 2>/dev/null || true

    local marker='# Added by Trileaf installer'
    local line='export PATH="$HOME/.local/bin:$PATH"'
    local target rc
    target="$(detect_rc_file)"

    for rc in "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.zshrc" "$HOME/.profile"; do
        if [[ -f "$rc" ]] && grep -qF '.local/bin' "$rc" 2>/dev/null; then
            PATH_PERSISTED=1; return 0
        fi
    done

    if [[ ! -f "$target" ]] && ! touch "$target" 2>/dev/null; then
        warn "Could not create $(basename "$target") for CLI registration."; return 0
    fi
    if [[ ! -w "$target" ]]; then
        warn "Could not update $(basename "$target") for CLI registration."; return 0
    fi
    if printf '\n%s\n%s\n' "$marker" "$line" >> "$target"; then
        info "Added ~/.local/bin to PATH in $(basename "$target")"
        PATH_PERSISTED=1
    else
        warn "Could not update $(basename "$target") for CLI registration."
    fi
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
if [[ "$FROM_INSTALLER" == "true" ]]; then
    echo -e "${BOLD}  Trileaf — Installing${NC}"
else
    echo -e "${BOLD}  Trileaf — Setup${NC}"
fi
echo -e "${MUTED}  Project dir: $SCRIPT_DIR${NC}"
echo ""

# ── Step 1 / 6 — Platform ─────────────────────────────────────────────────────
section "Step 1 / 6  —  Platform"

OS="unknown"
if   [[ "$OSTYPE" == "darwin"* ]];                                    then OS="macos"
elif [[ -n "${WSL_DISTRO_NAME:-}" || -n "${WSL_INTEROP:-}" ]];        then OS="wsl"
elif [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "linux"* ]];        then OS="linux"
fi
[[ "$OS" == "unknown" ]] && \
    fail "Unsupported operating system ($OSTYPE).\n  On Windows run: install.ps1"
ok "Platform: $OS"

# ── Step 2 / 6 — Python ───────────────────────────────────────────────────────
section "Step 2 / 6  —  Python"

PYTHON=""
find_python() {
    local ver maj min
    for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
        command -v "$cmd" >/dev/null 2>&1 || continue
        ver="$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
        [[ -z "$ver" ]] && continue
        maj="${ver%%.*}"; min="${ver##*.}"
        [[ "$maj" -ge 3 && "$min" -ge 10 ]] && echo "$cmd" && return 0
    done
    return 1
}

if ! PYTHON="$(find_python)"; then
    fail "Python 3.10+ is required but was not found.\n  macOS: brew install python@3.12\n  Ubuntu: sudo apt install python3.12 python3.12-venv"
fi
ok "Python: $PYTHON ($("$PYTHON" -c 'import sys; print(sys.version.split()[0])'))"

# ── Step 3 / 6 — Virtual environment ─────────────────────────────────────────
section "Step 3 / 6  —  Virtual environment"

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [[ -d "$VENV_DIR" ]]; then
    if [[ "$REINSTALL" == "true" ]]; then
        info "Removing existing .venv (--reinstall) ..."
        rm -rf "$VENV_DIR"
    elif [[ ! -x "$VENV_PYTHON" ]]; then
        warn "Existing .venv appears broken — recreating ..."
        rm -rf "$VENV_DIR"
    else
        ok ".venv exists — reusing  (--reinstall to force rebuild)"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating .venv with $PYTHON ..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok ".venv created."
fi

info "Upgrading pip ..."
"$VENV_PYTHON" -m pip install --upgrade pip --quiet

# On Linux/WSL without CUDA, install CPU-only torch first to avoid the
# multi-GB CUDA bundle that pip would otherwise pull from PyPI.
if [[ "$OS" != "macos" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
    info "No CUDA detected — installing CPU-only torch ..."
    "$VENV_PIP" install "torch>=2.4.0" \
        --index-url https://download.pytorch.org/whl/cpu --quiet
fi

info "Installing dependencies ..."
info "  (first install may take several minutes)"
"$VENV_PIP" install -r "$REQUIREMENTS" --quiet
"$VENV_PIP" install -e "$SCRIPT_DIR[leafhub]" --quiet
ok "Dependencies installed."

# ── Step 4 / 6 — CLI registration ────────────────────────────────────────────
section "Step 4 / 6  —  CLI registration"

if [[ ! -x "$TRILEAF_BIN" ]]; then
    fail "Entry point not found after install: $TRILEAF_BIN"
fi

if [[ "$DOCTOR" == "true" ]]; then
    info "Running environment check (--doctor) ..."
    "$VENV_PYTHON" "$SCRIPT_DIR/scripts/check_env.py"
    exit $?
fi

if [[ -L "$TRILEAF_LINK" ]]; then
    existing_target="$(readlink -f "$TRILEAF_LINK" 2>/dev/null || true)"
    if [[ "$existing_target" == "$TRILEAF_BIN" ]]; then
        ok "CLI already registered: $TRILEAF_LINK"
        path_has_dir "$ORIGINAL_PATH" "$BIN_DIR" && PATH_PERSISTED=1 || ensure_local_bin_on_path
    else
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
        warn "Could not update your shell RC file."
        warn "Add to your shell config:  export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
fi

# ── Step 5 / 6 — LeafHub ─────────────────────────────────────────────────────
section "Step 5 / 6  —  LeafHub"

# Load register.sh — provides leafhub_setup_project().
#
# Resolution order (v2 standard, 2026-03-21):
#   1. leafhub shell-helper        — system leafhub in PATH (fast path)
#   2. $VENV_DIR/bin/leafhub       — pip-installed in venv (offline fallback;
#                                    avoids curl when leafhub is already a
#                                    pip dependency but not yet in PATH, e.g.
#                                    during a LeafHub-initiated fresh install
#                                    or a clean one-liner install)
#   3. leafhub_dist/register.sh    — local distributed copy (offline fallback
#                                    for subsequent setups after first registration)
#   4. GitHub curl                 — first-time bootstrap, network required
#
# NOTE: the original `if ! eval "$(cmd)"` pattern is NOT used here because
# `eval ""` always exits 0, making the fallback unreachable when leafhub is
# absent from PATH.  Instead we capture output first and check it is non-empty.
_lh_content=""
if _lh_content="$(leafhub shell-helper 2>/dev/null)" && [[ -n "$_lh_content" ]]; then
    eval "$_lh_content"
elif [[ -x "$VENV_DIR/bin/leafhub" ]] \
    && _lh_content="$("$VENV_DIR/bin/leafhub" shell-helper 2>/dev/null)" \
    && [[ -n "$_lh_content" ]]; then
    eval "$_lh_content"
elif [[ -f "$SCRIPT_DIR/leafhub_dist/register.sh" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/leafhub_dist/register.sh"
else
    info "Fetching LeafHub installer ..."
    _TMP_REG="$(mktemp)"
    if ! curl -fsSL \
            https://raw.githubusercontent.com/Rebas9512/Leafhub/main/register.sh \
            -o "$_TMP_REG" 2>/dev/null; then
        rm -f "$_TMP_REG"
        fail "Could not fetch LeafHub installer.\n  Install manually: https://github.com/Rebas9512/Leafhub\n  Then re-run: ./setup.sh"
    fi
    # shellcheck disable=SC1090
    source "$_TMP_REG"
    rm -f "$_TMP_REG"
fi
unset _lh_content

[[ "$HEADLESS" == "true" ]] && export LEAFHUB_HEADLESS=1

leafhub_setup_project "trileaf" "$SCRIPT_DIR" "rewrite" \
    || fail "LeafHub registration failed.\n  Install LeafHub and retry: https://github.com/Rebas9512/Leafhub"

ok "LeafHub integration complete."

# ── Step 6 / 6 — Detection models ────────────────────────────────────────────
section "Step 6 / 6  —  Detection models"

if [[ "$HEADLESS" == "true" ]]; then
    info "Headless — skipping model download."
    info "Run later: trileaf setup"
else
    echo -e "  Trileaf needs two detection models ${MUTED}(~0.9 GB total)${NC}"
    echo -e "  ${MUTED}· desklib/ai-text-detector-v1.01     (~0.5 GB)${NC}"
    echo -e "  ${MUTED}· paraphrase-mpnet-base-v2            (~0.4 GB)${NC}"
    echo ""
    if [[ -r /dev/tty && -w /dev/tty ]]; then
        printf "  Download now? [Y/n]: " > /dev/tty
        read -r _dl < /dev/tty
    else
        _dl="n"
    fi
    case "${_dl:-Y}" in
        n|N) info "Run later: trileaf setup" ;;
        *)   "$TRILEAF_BIN" setup --models-only < /dev/tty > /dev/tty ;;
    esac
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}  Setup complete!${NC}"
echo ""
if path_has_dir "$ORIGINAL_PATH" "$BIN_DIR"; then
    echo -e "  ${GREEN}trileaf run${NC}      # start the dashboard"
    echo -e "  ${GREEN}trileaf setup${NC}    # download detection models"
    echo -e "  ${GREEN}trileaf doctor${NC}   # environment health check"
elif [[ "$PATH_PERSISTED" -eq 1 ]]; then
    echo "  Open a new terminal, then:"
    echo -e "    ${GREEN}trileaf run${NC}"
else
    echo "  Add ~/.local/bin to PATH, then:"
    echo -e "    ${GREEN}export PATH=\"\$HOME/.local/bin:\$PATH\"${NC}"
    echo -e "    ${GREEN}trileaf run${NC}"
fi
echo ""
