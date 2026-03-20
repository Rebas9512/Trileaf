"""
Credential resolution for the Trileaf rewrite provider.

Resolution order (first non-empty API key wins):
  1. LeafHub  — .leafhub dotfile in project root → hub.get_key(LEAFHUB_ALIAS)
  2. .env     — PROJECT_ROOT/.env loaded into os.environ
  3. env vars — REWRITE_API_KEY / provider-specific vars already in process

For initial setup run:  trileaf config
This writes a .env file in the project root.  The file is chmod 600 and
listed in .gitignore — never commit it.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

# Matches full-value env-var placeholders like ${MY_VAR} or $MY_VAR.
_ENV_VAR_RE = re.compile(r"^(?:\$\{[A-Z_][A-Z0-9_]*\}|\$[A-Z_][A-Z0-9_]*)$")

# Provider-specific env var candidates (checked after REWRITE_API_KEY).
_PROVIDER_ENV_API_KEY_CANDIDATES: Dict[str, tuple[str, ...]] = {
    "anthropic":      ("ANTHROPIC_API_KEY",),
    "google":         ("GEMINI_API_KEY",),
    "groq":           ("GROQ_API_KEY",),
    "litellm":        ("LITELLM_API_KEY",),
    "minimax":        ("MINIMAX_API_KEY",),
    "moonshot":       ("MOONSHOT_API_KEY",),
    "mistral":        ("MISTRAL_API_KEY",),
    "nvidia":         ("NVIDIA_API_KEY",),
    "ollama":         ("OLLAMA_API_KEY",),
    "openai":         ("OPENAI_API_KEY",),
    "openrouter":     ("OPENROUTER_API_KEY",),
    "together":       ("TOGETHER_API_KEY",),
    "vllm":           ("VLLM_API_KEY",),
    "xai":            ("XAI_API_KEY",),
}

_PROVIDER_ENV_API_KEY_ALIASES: Dict[str, str] = {
    "custom-anthropic": "anthropic",
    "custom-openai":    "openai",
    "minimax-cn":       "minimax",
}


# ── Utility helpers ───────────────────────────────────────────────────────────

def normalize_provider_id(value: Any) -> str:
    s = _trim_to_optional(value) or ""
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _trim_to_optional(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _trim_credential(value: Any) -> Optional[str]:
    s = _trim_to_optional(value)
    if s and _ENV_VAR_RE.fullmatch(s):
        import warnings
        warnings.warn(
            f"Credential looks like an unresolved env-var placeholder: {s!r}. "
            "Set the referenced env var or store the actual key value in .env.",
            stacklevel=4,
        )
        return None
    return s


def first_defined(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def get_provider_env_api_key_candidates(provider_id: Any) -> list[str]:
    normalized = normalize_provider_id(provider_id)
    lookup_id = _PROVIDER_ENV_API_KEY_ALIASES.get(normalized, normalized)
    candidates = ["REWRITE_API_KEY"]
    for env_name in _PROVIDER_ENV_API_KEY_CANDIDATES.get(lookup_id, ()):
        if env_name not in candidates:
            candidates.append(env_name)
    return candidates


def format_env_var_list(names: Sequence[str]) -> str:
    ordered: list[str] = []
    for name in names:
        if name and name not in ordered:
            ordered.append(name)
    return ", ".join(ordered)


def mask_secret(value: str) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


# ── .env file handling ────────────────────────────────────────────────────────

def _parse_env_line(line: str) -> tuple[str, str] | None:
    """Parse a single KEY=VALUE line. Returns (key, value) or None."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "=" not in stripped:
        return None
    key, _, raw = stripped.partition("=")
    key = key.strip()
    if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
        return None
    value = raw.strip()
    # Strip surrounding quotes
    if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
        value = value[1:-1]
    else:
        # Strip inline comment (only when preceded by whitespace)
        for i, ch in enumerate(value):
            if ch == "#" and i > 0 and value[i - 1].isspace():
                value = value[:i].rstrip()
                break
    return key, value


def load_dot_env(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """
    Load KEY=VALUE pairs from a .env file into os.environ.

    Args:
        path:     Path to the .env file. Defaults to PROJECT_ROOT/.env.
        override: If True, existing env vars are overwritten. Default False.

    Returns:
        Dict of keys loaded from the file (regardless of override).
    """
    env_path = path or ENV_FILE
    if not env_path.is_file():
        return {}
    loaded: dict[str, str] = {}
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            result = _parse_env_line(line)
            if result:
                key, value = result
                loaded[key] = value
                if override or key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass
    return loaded


def write_dot_env(
    values: dict[str, str],
    path: Path | None = None,
    *,
    header: str = "",
    merge: bool = True,
) -> Path:
    """
    Write KEY=VALUE pairs to a .env file (chmod 600).

    Args:
        values: Mapping of env var names to values.
        path:   Target path. Defaults to PROJECT_ROOT/.env.
        header: Optional comment block written at the top.
        merge:  If True and file exists, update matching keys in-place
                (preserves unrelated keys and comments). Default True.
    """
    env_path = path or ENV_FILE

    if merge and env_path.is_file():
        _merge_dot_env(env_path, values)
    else:
        _write_dot_env_fresh(env_path, values, header=header)

    try:
        env_path.chmod(0o600)
    except OSError:
        pass
    return env_path


def _write_dot_env_fresh(path: Path, values: dict[str, str], *, header: str = "") -> None:
    lines: list[str] = []
    if header:
        for c in header.splitlines():
            lines.append(f"# {c}" if c.strip() else "#")
        lines.append("")
    for key, value in values.items():
        lines.append(f"{key}={_quote_env_value(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _merge_dot_env(path: Path, updates: dict[str, str]) -> None:
    """Update specific keys in an existing .env file, preserving all other lines."""
    try:
        original = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        original = []

    remaining = dict(updates)
    out: list[str] = []
    for line in original:
        result = _parse_env_line(line)
        if result and result[0] in remaining:
            key = result[0]
            out.append(f"{key}={_quote_env_value(remaining.pop(key))}")
        else:
            out.append(line)

    # Append keys not found in the existing file
    if remaining:
        if out and out[-1].strip():
            out.append("")
        for key, value in remaining.items():
            out.append(f"{key}={_quote_env_value(value)}")

    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _quote_env_value(value: str) -> str:
    if not value:
        return ""
    if " " in value or "#" in value or value[0] in ('"', "'"):
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return value


# ── LeafHub probe integration ─────────────────────────────────────────────────

def _get_leafhub_alias() -> str:
    """Return the LeafHub alias to use for the rewrite API key."""
    return (
        os.getenv("LEAFHUB_ALIAS")
        or _read_dot_env_key(ENV_FILE, "LEAFHUB_ALIAS")
        or "rewrite"
    )


def _read_dot_env_key(path: Path, key: str) -> str | None:
    """Fast single-key read from a .env file without touching os.environ."""
    if not path.is_file():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            result = _parse_env_line(line)
            if result and result[0] == key:
                return result[1] or None
    except OSError:
        pass
    return None


def _try_leafhub(project_dir: Path | None = None) -> dict[str, str] | None:
    """
    Probe for a LeafHub-linked project and resolve the rewrite API key.

    Returns a dict of REWRITE_* env var values to inject, or None if LeafHub
    is not available or not linked.
    """
    try:
        from leafhub.probe import detect
    except ImportError:
        return None

    found = detect(project_dir=project_dir or PROJECT_ROOT)
    if not found.ready:
        return None

    alias = _get_leafhub_alias()
    try:
        hub = found.open_sdk()
        api_key = hub.get_key(alias)
        if not api_key:
            return None

        result: dict[str, str] = {"REWRITE_API_KEY": api_key}

        # Pull provider config from LeafHub (base_url, model, api_format, auth)
        try:
            cfg = hub.get_config(alias)
            if getattr(cfg, "base_url", None):
                result["REWRITE_BASE_URL"] = cfg.base_url
            # ProviderConfig.model is the resolved model (override or default_model)
            if getattr(cfg, "model", None):
                result["_LEAFHUB_MODEL"] = cfg.model
            # api_format maps to REWRITE_API_KIND (only fills if not already set)
            if getattr(cfg, "api_format", None):
                result["_LEAFHUB_API_KIND"] = cfg.api_format
            # auth_mode / auth_header (only fill if not already set)
            if getattr(cfg, "auth_mode", None):
                result["_LEAFHUB_AUTH_MODE"] = cfg.auth_mode
            if getattr(cfg, "auth_header", None):
                result["_LEAFHUB_AUTH_HEADER"] = cfg.auth_header
        except Exception as _cfg_exc:
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "LeafHub get_config(%r) failed: %s: %s",
                alias, type(_cfg_exc).__name__, _cfg_exc,
            )

        return result
    except Exception as _exc:
        _exc_name = type(_exc).__name__
        if "InvalidToken" in _exc_name or "token" in str(_exc).lower():
            import sys
            print(
                "\n[!] LeafHub: token in .leafhub is invalid or expired.\n"
                "    Re-link with:  trileaf config\n",
                file=sys.stderr,
            )
        return None


# ── Unified credential resolution ─────────────────────────────────────────────

def resolve_credentials(project_dir: Path | None = None) -> dict[str, Any]:
    """
    Resolve rewrite provider credentials and inject into os.environ.

    Resolution order:
      1. .env file (PROJECT_ROOT/.env) — loads all non-secret config first
      2. LeafHub probe — if .leafhub found and ready, overrides REWRITE_API_KEY
      3. Existing os.environ — provider-specific key fallback (e.g. OPENAI_API_KEY)

    Returns a summary dict with "source" and "credential_source" keys.
    """
    # Step 1: load .env (non-overriding — process env takes priority)
    dot_env = load_dot_env(override=False)

    # Step 2: try LeafHub for the API key
    leafhub = _try_leafhub(project_dir)
    if leafhub:
        for key, value in leafhub.items():
            if not key.startswith("_") and value:
                os.environ[key] = value          # LeafHub API key always wins
        # Fill optional settings from LeafHub only when not already in env
        for lh_key, env_key in (
            ("_LEAFHUB_MODEL",       "REWRITE_MODEL"),
            ("_LEAFHUB_API_KIND",    "REWRITE_API_KIND"),
            ("_LEAFHUB_AUTH_MODE",   "REWRITE_AUTH_MODE"),
            ("_LEAFHUB_AUTH_HEADER", "REWRITE_AUTH_HEADER"),
        ):
            lh_val = leafhub.get(lh_key, "")
            if lh_val and not os.getenv(env_key):
                os.environ[env_key] = lh_val
        credential_source = "leafhub"
    elif os.getenv("REWRITE_API_KEY"):
        credential_source = "dotenv" if "REWRITE_API_KEY" in dot_env else "env"
    else:
        # Step 3: try provider-specific fallback env vars
        provider_id = os.getenv("REWRITE_PROVIDER_ID", "")
        candidates = get_provider_env_api_key_candidates(provider_id)
        credential_source = "none"
        for env_name in candidates[1:]:          # skip REWRITE_API_KEY (already checked)
            val = _trim_credential(os.getenv(env_name))
            if val:
                os.environ["REWRITE_API_KEY"] = val
                credential_source = f"env:{env_name}"
                break

    # Ensure backend default
    os.environ.setdefault("REWRITE_BACKEND", "local")

    return {
        "source":            "leafhub" if leafhub else ("dotenv" if dot_env else "env"),
        "credential_source": credential_source,
        "dot_env_loaded":    bool(dot_env),
        "leafhub_active":    leafhub is not None,
    }


def rewrite_backend_is_external() -> bool:
    """Return True if the configured rewrite backend is an external API."""
    raw = os.getenv("REWRITE_BACKEND", "local").strip().lower()
    return raw in {"external", "openai_api", "api", "remote"}


# ── Legacy env shim (kept for any remaining callers) ──────────────────────────

def legacy_env_first(
    alias_group: str,
    *,
    env: Mapping[str, str] | None = None,
) -> Optional[str]:
    """Read QWEN_* legacy env vars for backwards compatibility."""
    _LEGACY = {
        "backend":    ("QWEN_BACKEND",),
        "model_path": ("QWEN_MODEL_PATH",),
        "base_url":   ("QWEN_API_BASE_URL",),
        "model":      ("QWEN_API_MODEL",),
    }
    env_map = os.environ if env is None else env
    for env_name in _LEGACY.get(alias_group, ()):
        value = _trim_to_optional(env_map.get(env_name))
        if value:
            return value
    return None


# ── CLI entry point (delegates to rewrite_provider_cli) ──────────────────────

def main(argv: list[str] | None = None) -> int:
    from scripts import rewrite_provider_cli
    return rewrite_provider_cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
