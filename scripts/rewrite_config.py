"""
Credential resolution for the Trileaf rewrite provider.

Resolution order (first non-empty API key wins):
  1. LeafHub  — .leafhub dotfile in project root → hub.get_key(LEAFHUB_ALIAS)
  2. env vars — REWRITE_API_KEY / provider-specific vars already in process

Configure providers with:  leafhub provider add
Register this project with: leafhub register trileaf --path <dir>
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


# ── LeafHub probe integration ─────────────────────────────────────────────────

def _get_leafhub_alias() -> str:
    """Return the LeafHub alias to use for the rewrite API key."""
    return os.getenv("LEAFHUB_ALIAS") or "rewrite"


def _try_leafhub(project_dir: Path | None = None) -> dict[str, str] | None:
    """
    Probe for a LeafHub-linked project and resolve the rewrite API key.

    Returns a dict of REWRITE_* env var values to inject, or None if LeafHub
    is not available or not linked.

    Silent-failure distinction:
    - not found / not ready  → return None silently (no dotfile, not linked)
    - found but key empty    → print a warning; the project IS linked but the
                               alias has no binding yet.  This is a different
                               problem from "not linked" and requires a
                               different fix (leafhub project bind …).
    """
    # Import preference order (v2 standard, 2026-03-21):
    #   1. leafhub.probe      — installed pip package (full SDK, preferred)
    #   2. leafhub_dist.probe — local distributed copy in project root
    #                           (stdlib-only fallback when pip package absent)
    #
    # leafhub_dist/ lives at PROJECT_ROOT, which is not automatically on
    # sys.path in editable installs (only named packages are exposed).
    # Add it temporarily so the fallback import can succeed.
    try:
        from leafhub.probe import detect
    except ImportError:
        _root = str(PROJECT_ROOT)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        try:
            from leafhub_dist.probe import detect  # type: ignore[no-redef]
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
            print(
                f"\n[!] LeafHub: project is linked but alias '{alias}' has no provider binding.\n"
                f"    Bind one with:  leafhub project bind <project> --alias {alias} --provider <name>\n"
                f"    Or re-run:      leafhub register <project> --path <dir> --alias {alias}\n",
                file=sys.stderr,
            )
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
        exc_type = type(_exc).__name__
        if "InvalidToken" in exc_type:
            print(
                "\n[!] LeafHub: token in .leafhub is invalid or expired.\n"
                "    Re-link with:  trileaf config\n",
                file=sys.stderr,
            )
        elif "AliasNotBound" in exc_type:
            print(
                f"\n[!] LeafHub: project is linked but alias '{alias}' has no provider binding.\n"
                f"    Bind one with:  leafhub project bind <project> --alias {alias} --provider <name>\n"
                f"    Or re-run:      leafhub register <project> --path <dir> --alias {alias}\n",
                file=sys.stderr,
            )
        elif isinstance(_exc, ImportError):
            # open_sdk() triggered a lazy import of a sub-dependency that is
            # missing.  The leafhub package itself is installed (we got past
            # the detect import), but one of its optional dependencies is not.
            print(
                f"\n[!] LeafHub: a dependency failed to import inside open_sdk(): {_exc}\n"
                "    Try reinstalling:  pip install -e '.[leafhub]'  (from Trileaf directory)\n"
                "    Or re-run:         trileaf setup\n",
                file=sys.stderr,
            )
        return None


# ── Unified credential resolution ─────────────────────────────────────────────

def resolve_credentials(project_dir: Path | None = None) -> dict[str, Any]:
    """
    Resolve rewrite provider credentials and inject into os.environ.

    Resolution order:
      1. LeafHub probe — if .leafhub found and ready, injects REWRITE_API_KEY
         and provider config (base_url, model, api_format, auth).
      2. Existing os.environ — REWRITE_API_KEY / provider-specific key fallback
         (e.g. OPENAI_API_KEY). Useful for CI or power users.

    Returns a summary dict with "source" and "credential_source" keys.
    """
    # Step 1: try LeafHub
    leafhub = _try_leafhub(project_dir)
    if leafhub:
        for key, value in leafhub.items():
            if not key.startswith("_") and value:
                os.environ[key] = value
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
        credential_source = "env"
    else:
        # Step 2: try provider-specific fallback env vars
        provider_id = os.getenv("REWRITE_PROVIDER_ID", "")
        candidates = get_provider_env_api_key_candidates(provider_id)
        credential_source = "none"
        for env_name in candidates[1:]:
            val = _trim_credential(os.getenv(env_name))
            if val:
                os.environ["REWRITE_API_KEY"] = val
                credential_source = f"env:{env_name}"
                break

    # Publish credential_source into the environment so that check_env.py
    # (which runs in the same process after this call) can read the resolved
    # value without duplicating the resolution logic or re-probing LeafHub.
    # REWRITE_CREDENTIAL_SOURCE is set here and only here; downstream code
    # must not set it manually.
    os.environ["REWRITE_CREDENTIAL_SOURCE"] = credential_source

    return {
        # "source" describes which credential path was *attempted* first.
        # When LeafHub is active it is "leafhub"; when only env vars were
        # checked (regardless of whether they were found) it is "env"; when
        # neither produced a key it is "none".
        "source":            "leafhub" if leafhub else (
                                 "env" if credential_source not in ("none", "") else "none"
                             ),
        "credential_source": credential_source,
        "leafhub_active":    leafhub is not None,
    }


def rewrite_backend_is_external() -> bool:
    """All rewrite backends are now external (via LeafHub). Always returns True."""
    return True


def legacy_env_first(
    alias_group: str,
    *,
    env: Mapping[str, str] | None = None,
) -> Optional[str]:
    """Removed: local Qwen model support has been dropped. Always returns None."""
    return None
