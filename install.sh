#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  Trileaf — One-liner Installer  (macOS / Linux / WSL)
#
#  curl -fsSL https://raw.githubusercontent.com/Rebas9512/Trileaf/main/install.sh | bash
#
#  This script selects an install directory, clones the repo, records install
#  metadata, then delegates all further setup to setup.sh inside the clone.
#
#  Environment variables:
#    TRILEAF_DIR=<path>     Install directory  (default: ~/trileaf)
#    TRILEAF_REPO_URL=<url> Clone URL          (default: GitHub repo)
#    NO_COLOR=1             Disable colour output
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DEFAULT_INSTALL_DIR="$HOME/trileaf"
CONFIG_DIR="$HOME/.trileaf"
INSTALL_META_PATH="$CONFIG_DIR/install.json"
TRILEAF_DIR="${TRILEAF_DIR:-}"
REPO_URL="${TRILEAF_REPO_URL:-https://github.com/Rebas9512/Trileaf.git}"

# ── Minimal colours (only needed before setup.sh takes over) ──────────────────
if [[ -n "${NO_COLOR:-}" || "${TERM:-dumb}" == "dumb" ]]; then
    BOLD='' GREEN='' RED='' MUTED='' NC=''
else
    BOLD='\033[1m'
    GREEN='\033[38;2;0;229;180m'
    RED='\033[38;2;230;57;70m'
    MUTED='\033[38;2;110;120;148m'
    NC='\033[0m'
fi

fail() { echo -e "${RED}✗${NC}  $*" >&2; exit 1; }
info() { echo -e "${MUTED}·${NC}  $*"; }
ok()   { echo -e "${GREEN}✓${NC}  $*"; }

# ── Helpers ───────────────────────────────────────────────────────────────────

normalise_path() {
    local raw="${1:-}"
    local expanded="${raw}"
    # Strip a single layer of surrounding quotes (drag-and-drop paths)
    while [[ "$expanded" == \'*\' || "$expanded" == \"*\" ]]; do
        if [[ "$expanded" == \'*\' && "$expanded" == *\' ]]; then
            expanded="${expanded:1:${#expanded}-2}"; continue
        fi
        if [[ "$expanded" == \"*\" && "$expanded" == *\" ]]; then
            expanded="${expanded:1:${#expanded}-2}"; continue
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
    local dir="$1" entry
    for entry in "$dir"/.[!.]* "$dir"/..?* "$dir"/*; do
        [[ -e "$entry" ]] && return 0
    done
    return 1
}

# ── Select install directory ──────────────────────────────────────────────────

default_dir="$(normalise_path "$DEFAULT_INSTALL_DIR")"

if [[ -n "$TRILEAF_DIR" ]]; then
    TRILEAF_DIR="$(normalise_path "$TRILEAF_DIR")"
elif [[ -r /dev/tty && -w /dev/tty && -z "${CI:-}" ]]; then
    printf 'Install directory [%s]: ' "$default_dir" > /dev/tty
    if IFS= read -r _cand < /dev/tty; then
        _cand="${_cand:-$default_dir}"
    else
        _cand="$default_dir"
    fi
    TRILEAF_DIR="$(normalise_path "$_cand")"
else
    TRILEAF_DIR="$default_dir"
fi

[[ "$TRILEAF_DIR" == "$(normalise_path "$CONFIG_DIR")" ]] && \
    fail "Install directory cannot be $CONFIG_DIR (reserved for Trileaf config)."

# If target exists and is non-empty but not a git repo, redirect to subdirectory
if [[ ! -d "$TRILEAF_DIR/.git" ]] && \
   [[ -d "$TRILEAF_DIR" ]] && dir_has_entries "$TRILEAF_DIR"; then
    info "Target is non-empty — using subdirectory: $TRILEAF_DIR/trileaf"
    TRILEAF_DIR="$(normalise_path "$TRILEAF_DIR/trileaf")"
fi

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}  Trileaf — Installer${NC}"
echo -e "${MUTED}  Install path: $TRILEAF_DIR${NC}"
echo ""

# ── Prerequisites ─────────────────────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || \
    fail "Python 3 not found.\n  macOS: brew install python@3.12\n  Ubuntu: sudo apt install python3.12"
command -v git >/dev/null 2>&1 || fail "git is required but not found."

# ── Clone / update ────────────────────────────────────────────────────────────
if [[ ! -d "$TRILEAF_DIR/.git" ]] && [[ ! -e "$TRILEAF_DIR" ]]; then
    info "Cloning into $TRILEAF_DIR ..."
    git clone --depth=1 "$REPO_URL" "$TRILEAF_DIR" --quiet
    ok "Cloned."
else
    if [[ ! -d "$TRILEAF_DIR/.git" ]]; then
        info "Directory exists — initialising git..."
        git -C "$TRILEAF_DIR" init --quiet
        git -C "$TRILEAF_DIR" remote add origin "$REPO_URL" 2>/dev/null || true
    else
        info "Existing installation found — syncing to latest..."
    fi
    git -C "$TRILEAF_DIR" fetch origin --depth=1 --quiet
    branch="$(git -C "$TRILEAF_DIR" symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's|.*/||')"
    [[ -z "$branch" ]] && branch="main"
    git -C "$TRILEAF_DIR" reset --hard "origin/$branch" --quiet
    git -C "$TRILEAF_DIR" clean -fdx --quiet 2>/dev/null || true
    ok "Synced to latest ($branch)."
fi

# ── Write install metadata ────────────────────────────────────────────────────
mkdir -p "$CONFIG_DIR"
python3 - "$INSTALL_META_PATH" "$TRILEAF_DIR" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({"install_method": "one_liner", "install_dir": sys.argv[2]}, indent=2) + "\n")
PY

# ── Hand off to setup.sh ──────────────────────────────────────────────────────
SETUP_SH="$TRILEAF_DIR/setup.sh"
[[ -f "$SETUP_SH" ]] || fail "setup.sh not found in $TRILEAF_DIR."
chmod +x "$SETUP_SH"
exec bash "$SETUP_SH" --from-installer "$@"
