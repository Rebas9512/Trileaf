#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  Trileaf — One-liner Installer
#
#  curl -fsSL https://raw.githubusercontent.com/Rebas9512/Trileaf/main/install.sh | bash
#
#  Environment variables:
#    TRILEAF_DIR=<path>     Install directory  (default: ~/trileaf)
#    TRILEAF_REPO_URL=<url> Clone URL          (default: GitHub repo)
#    TRILEAF_NO_ONBOARD=1   Skip the interactive setup wizard
#    NO_COLOR=1             Disable colour output
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DEFAULT_INSTALL_DIR="$HOME/trileaf"
CONFIG_DIR="$HOME/.trileaf"
INSTALL_META_PATH="$CONFIG_DIR/install.json"
TRILEAF_DIR="${TRILEAF_DIR:-}"
REPO_URL="${TRILEAF_REPO_URL:-https://github.com/Rebas9512/Trileaf.git}"
BIN_DIR="$HOME/.local/bin"
ORIGINAL_PATH="${PATH:-}"
PATH_PERSISTED=0
INSTALL_DIR_REDIRECTED_FROM=""
VENV_DIR=""
VENV_PYTHON=""
VENV_PIP=""
TRILEAF_BIN=""
TRILEAF_LINK="$BIN_DIR/trileaf"

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

# Returns 0 when onboarding should stay non-interactive.
is_non_interactive() {
    [[ "${TRILEAF_NO_ONBOARD:-0}" == "1" ]] && return 0
    [[ ! -r /dev/tty || ! -w /dev/tty ]] && return 0
    return 1
}

# Expand "~" and strip a trailing slash for simple path comparisons.
normalise_path() {
    local raw="${1:-}"
    local expanded="$raw"
    while [[ "$expanded" == \'*\' || "$expanded" == \"*\" ]]; do
        if [[ "$expanded" == \'*\' && "$expanded" == *\' ]]; then
            expanded="${expanded:1:${#expanded}-2}"
            continue
        fi
        if [[ "$expanded" == \"*\" && "$expanded" == *\" ]]; then
            expanded="${expanded:1:${#expanded}-2}"
            continue
        fi
        break
    done
    expanded="${expanded/#\~/$HOME}"
    if [[ -n "$expanded" && "$expanded" != /* ]]; then
        expanded="$(pwd -P)/$expanded"
    fi
    while [[ "${expanded}" != "/" && "${expanded}" == */ ]]; do
        expanded="${expanded%/}"
    done
    printf '%s' "$expanded"
}

dir_has_entries() {
    local dir="$1"
    local entry
    for entry in "$dir"/.[!.]* "$dir"/..?* "$dir"/*; do
        [[ -e "$entry" ]] && return 0
    done
    return 1
}

# Prompt for the target clone directory when possible.
select_install_dir() {
    local candidate default_dir
    default_dir="$(normalise_path "$DEFAULT_INSTALL_DIR")"

    if [[ -n "$TRILEAF_DIR" ]]; then
        TRILEAF_DIR="$(normalise_path "$TRILEAF_DIR")"
    elif [[ -r /dev/tty && -w /dev/tty && -z "${CI:-}" ]]; then
        printf 'Install directory [%s]: ' "$default_dir" > /dev/tty
        if IFS= read -r candidate < /dev/tty; then
            candidate="${candidate:-$default_dir}"
        else
            candidate="$default_dir"
        fi
        TRILEAF_DIR="$(normalise_path "$candidate")"
    else
        TRILEAF_DIR="$default_dir"
    fi

    if [[ "$TRILEAF_DIR" == "$(normalise_path "$CONFIG_DIR")" ]]; then
        fail "Install directory cannot be $CONFIG_DIR because that location is reserved for Trileaf JSON config files."
    fi
}

resolve_install_dir() {
    local requested="$TRILEAF_DIR"
    local fallback=""

    if [[ -e "$requested" && ! -d "$requested" ]]; then
        fail "Install directory exists but is not a directory: $requested"
    fi

    if [[ -d "$requested/.git" ]]; then
        TRILEAF_DIR="$requested"
    elif [[ -d "$requested" ]] && dir_has_entries "$requested"; then
        fallback="$(normalise_path "$requested/trileaf")"
        if [[ -e "$fallback" && ! -d "$fallback" ]]; then
            fail "Fallback install directory exists but is not a directory: $fallback"
        fi
        if [[ -d "$fallback" && ! -d "$fallback/.git" ]] && dir_has_entries "$fallback"; then
            fail "Install directory $requested already exists and is not empty. The fallback subdirectory $fallback also exists and is not empty."
        fi
        INSTALL_DIR_REDIRECTED_FROM="$requested"
        TRILEAF_DIR="$fallback"
    else
        TRILEAF_DIR="$requested"
    fi

    VENV_DIR="$TRILEAF_DIR/.venv"
    VENV_PYTHON="$VENV_DIR/bin/python"
    VENV_PIP="$VENV_DIR/bin/pip"
    TRILEAF_BIN="$VENV_DIR/bin/trileaf"
}

# Returns 0 when <dir> appears in <path>
path_has_dir() {
    case ":${1}:" in *":${2%/}:"*) return 0 ;; *) return 1 ;; esac
}

# Write install metadata to ~/.trileaf/install.json so the CLI can find the
# managed checkout later for upgrades and removal.
write_install_metadata() {
    mkdir -p "$CONFIG_DIR"
    "$PYTHON" - "$INSTALL_META_PATH" "$TRILEAF_DIR" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "install_method": "one_liner",
    "install_dir": sys.argv[2],
}
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

# Ensures ~/.local/bin exists and tries to register it via ~/.bash_profile.
ensure_local_bin_on_path() {
    mkdir -p "$BIN_DIR"
    export PATH="$BIN_DIR:$PATH"
    hash -r 2>/dev/null || true

    local marker='# Added by Trileaf installer'
    local line='export PATH="$HOME/.local/bin:$PATH"'
    local target="$HOME/.bash_profile"

    if [[ -f "$target" ]] && [[ -r "$target" ]] && grep -qF '.local/bin' "$target" 2>/dev/null; then
        PATH_PERSISTED=1
        return 0
    fi

    if [[ ! -f "$target" ]] && ! touch "$target" 2>/dev/null; then
        warn "Could not create ~/.bash_profile for CLI registration."
        return 0
    fi

    if [[ ! -w "$target" ]]; then
        warn "Could not update ~/.bash_profile for CLI registration."
        return 0
    fi

    if printf '\n%s\n%s\n' "$marker" "$line" >> "$target"; then
        info "Added ~/.local/bin to PATH in $(basename "$target")"
        PATH_PERSISTED=1
    else
        warn "Could not update ~/.bash_profile for CLI registration."
    fi
}

# Resolve the installation directory before we print the banner.
select_install_dir
resolve_install_dir

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}  Trileaf — Installer${NC}"
echo -e "${MUTED}  Install path: $TRILEAF_DIR${NC}"
echo -e "${MUTED}  Config path:  $CONFIG_DIR${NC}"
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
if [[ -n "$INSTALL_DIR_REDIRECTED_FROM" ]]; then
    info "Requested directory is not empty — using subdirectory: $TRILEAF_DIR"
fi
if [[ -d "$TRILEAF_DIR/.git" ]]; then
    info "Existing installation found — updating..."
    git -C "$TRILEAF_DIR" pull --ff-only --quiet
    ok "Updated to latest."
else
    info "Cloning into $TRILEAF_DIR ..."
    git clone --depth=1 "$REPO_URL" "$TRILEAF_DIR" --quiet
    ok "Cloned."
fi

write_install_metadata

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
    if [[ "$PATH_PERSISTED" -eq 1 ]]; then
        warn "$BIN_DIR is not in your current shell's PATH."
        warn "Open a new terminal, or run now:  export PATH=\"\$HOME/.local/bin:\$PATH\""
    else
        warn "~/.bash_profile could not be updated for CLI registration."
        warn "Add it manually before using trileaf:  export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
fi

# ── Step 6: Onboarding ────────────────────────────────────────────────────────
section "Setup"
if is_non_interactive; then
    info "Non-interactive session — skipping wizard."
    info "Run  trileaf setup  to configure models and providers."
else
    "$TRILEAF_BIN" setup < /dev/tty > /dev/tty
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}  Trileaf installed!${NC}"
echo ""
if path_has_dir "$ORIGINAL_PATH" "$BIN_DIR"; then
    echo -e "  ${GREEN}trileaf setup${NC}    # configure models and providers"
    echo -e "  ${GREEN}trileaf run${NC}      # start the dashboard"
elif [[ "$PATH_PERSISTED" -eq 1 ]]; then
    echo "  Open a new terminal, then:"
    echo -e "    ${GREEN}trileaf setup${NC}    # configure models and providers"
    echo -e "    ${GREEN}trileaf run${NC}      # start the dashboard"
else
    echo "  ~/.bash_profile CLI registration did not complete."
    echo "  Add ~/.local/bin to PATH, then run:"
    echo -e "    ${GREEN}export PATH=\"\$HOME/.local/bin:\$PATH\"${NC}"
    echo -e "    ${GREEN}trileaf setup${NC}    # configure models and providers"
    echo -e "    ${GREEN}trileaf run${NC}      # start the dashboard"
fi
echo ""
