"""
Credential resolution for the Trileaf rewrite provider.

Resolution order (handled by leafhub-sdk):
  1. LeafHub vault  — .leafhub dotfile → hub.get_config("rewrite")
  2. env_fallbacks  — declared in leafhub.toml (REWRITE_API_KEY, OPENAI_API_KEY, ...)
  3. Common env vars — ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.

Configure providers with:  leafhub provider add
Register this project with: leafhub register .
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Matches full-value env-var placeholders like ${MY_VAR} or $MY_VAR.
_ENV_VAR_RE = re.compile(r"^(?:\$\{[A-Z_][A-Z0-9_]*\}|\$[A-Z_][A-Z0-9_]*)$")

# Provider-specific env var candidates (used by models_runtime.py, check_env.py).
_PROVIDER_ENV_API_KEY_CANDIDATES: dict[str, tuple[str, ...]] = {
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

_PROVIDER_ENV_API_KEY_ALIASES: dict[str, str] = {
    "custom-anthropic": "anthropic",
    "custom-openai":    "openai",
    "minimax-cn":       "minimax",
}


# ── Utility helpers (used by models_runtime.py, check_env.py, etc.) ──────────

def normalize_provider_id(value: Any) -> str:
    s = str(value).strip() if value is not None else ""
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def get_provider_env_api_key_candidates(provider_id: Any) -> list[str]:
    normalized = normalize_provider_id(provider_id)
    lookup_id = _PROVIDER_ENV_API_KEY_ALIASES.get(normalized, normalized)
    candidates = ["REWRITE_API_KEY"]
    for env_name in _PROVIDER_ENV_API_KEY_CANDIDATES.get(lookup_id, ()):
        if env_name not in candidates:
            candidates.append(env_name)
    return candidates


def format_env_var_list(names: list[str] | tuple[str, ...]) -> str:
    ordered: list[str] = []
    for name in names:
        if name and name not in ordered:
            ordered.append(name)
    return ", ".join(ordered)


def first_defined(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def mask_secret(value: str) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def rewrite_backend_is_external() -> bool:
    """All rewrite backends are now external (via LeafHub). Always returns True."""
    return True


def legacy_env_first(alias_group: str, **_kw: Any) -> Optional[str]:
    """Removed: local Qwen model support has been dropped. Always returns None."""
    return None


# ── Unified credential resolution ─────────────────────────────────────────────

def resolve_credentials(project_dir: Path | None = None) -> dict[str, Any]:
    """
    Resolve rewrite provider credentials and inject into os.environ.

    Uses leafhub-sdk's unified resolve() with as_env=True, which reads
    leafhub.toml for alias, env_prefix, and fallback configuration.

    Returns a summary dict with "source" and "credential_source" keys.
    """
    _dir = project_dir or PROJECT_ROOT

    try:
        from leafhub_sdk import resolve as _sdk_resolve

        env = _sdk_resolve("rewrite", project_dir=_dir, as_env=True)
        os.environ.update(env)
        credential_source = env.get("REWRITE_CREDENTIAL_SOURCE", "leafhub")
        return {
            "source": credential_source.split(":")[0],  # "leafhub" | "env" | "env-fallback"
            "credential_source": credential_source,
            "leafhub_active": credential_source == "leafhub",
        }
    except ImportError:
        pass
    except Exception as exc:
        # leafhub-sdk failed — fall through to legacy resolution
        import logging
        logging.getLogger(__name__).debug(
            "leafhub-sdk resolve failed: %s: %s", type(exc).__name__, exc,
        )

    # ── Legacy fallback: direct env var check ────────────────────────────────
    # Used when leafhub-sdk is not installed or resolve() fails entirely.
    # Also tries the old leafhub.probe path for backward compatibility.
    result = _try_leafhub(_dir)
    if result:
        for key, value in result.items():
            if not key.startswith("_") and value:
                os.environ[key] = value
        for lh_key, env_key in (
            ("_LEAFHUB_MODEL",       "REWRITE_MODEL"),
            ("_LEAFHUB_API_KIND",    "REWRITE_API_KIND"),
            ("_LEAFHUB_AUTH_MODE",   "REWRITE_AUTH_MODE"),
            ("_LEAFHUB_AUTH_HEADER", "REWRITE_AUTH_HEADER"),
        ):
            lh_val = result.get(lh_key, "")
            if lh_val:
                os.environ[env_key] = lh_val
        credential_source = "leafhub"
    elif os.getenv("REWRITE_API_KEY"):
        credential_source = "env"
    else:
        # Try common env vars
        credential_source = "none"
        for env_name in (
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY",
            "MISTRAL_API_KEY", "XAI_API_KEY", "TOGETHER_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            val = os.getenv(env_name, "").strip()
            if val and not _ENV_VAR_RE.fullmatch(val):
                os.environ["REWRITE_API_KEY"] = val
                credential_source = f"env:{env_name}"
                break

    os.environ["REWRITE_CREDENTIAL_SOURCE"] = credential_source

    return {
        "source": "leafhub" if result else (
            "env" if credential_source not in ("none", "") else "none"
        ),
        "credential_source": credential_source,
        "leafhub_active": result is not None,
    }


def _try_leafhub(project_dir: Path | None = None) -> dict[str, str] | None:
    """Probe for LeafHub credentials. Alias kept for backward compat with trileaf_cli."""
    return _try_leafhub_legacy(project_dir or PROJECT_ROOT)


def _try_leafhub_legacy(project_dir: Path) -> dict[str, str] | None:
    """Legacy probe path — used only when leafhub-sdk is not installed."""
    try:
        from leafhub.probe import detect
    except ImportError:
        return None

    alias = os.getenv("LEAFHUB_ALIAS") or "rewrite"
    found = detect(project_dir=project_dir)
    if not found.ready:
        return None

    try:
        hub = found.open_sdk()
        api_key = hub.get_key(alias)
        if not api_key:
            return None

        result: dict[str, str] = {"REWRITE_API_KEY": api_key}
        try:
            cfg = hub.get_config(alias)
            if getattr(cfg, "base_url", None):
                result["REWRITE_BASE_URL"] = cfg.base_url
            if getattr(cfg, "model", None):
                result["_LEAFHUB_MODEL"] = cfg.model
            if getattr(cfg, "api_format", None):
                result["_LEAFHUB_API_KIND"] = cfg.api_format
            _auth = getattr(cfg, "auth_mode", None) or ""
            if _auth == "openai-oauth":
                _auth = "bearer"
            if _auth:
                result["_LEAFHUB_AUTH_MODE"] = _auth
            if getattr(cfg, "auth_header", None):
                result["_LEAFHUB_AUTH_HEADER"] = cfg.auth_header
        except Exception:
            pass
        return result
    except Exception:
        return None
