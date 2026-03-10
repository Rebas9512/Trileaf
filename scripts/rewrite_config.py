"""
Local rewrite-provider profile storage and env injection helpers.

Profiles are stored in a local JSON file that is ignored by git. The active
profile can be applied to process environment variables before runtime modules
read their settings.
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

# Matches unresolved full-value placeholders like ${MY_VAR} or $MY_VAR.
# This follows the OpenClaw pattern of rejecting obvious unresolved references
# without treating arbitrary strings containing '$' as invalid credentials.
_ENV_VAR_RE = re.compile(r"^(?:\$\{[A-Z_][A-Z0-9_]*\}|\$[A-Z_][A-Z0-9_]*)$")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# User-level config dir — kept outside the repo so credentials are never
# accidentally packaged or shared. Mirrors the ~/.openclaw pattern.
USER_CONFIG_DIR = Path.home() / ".trileaf"
CONFIG_PATH = USER_CONFIG_DIR / "rewrite_profiles.json"

# Legacy paths — migrated automatically on first load.
# 1. Project-root JSON (early dev artefact).
_LEGACY_CONFIG_PATH = PROJECT_ROOT / ".rewrite_profiles.local.json"
# 2. Old user-config dir used before the project was renamed to "trileaf".
_LEGACY_USER_CONFIG_PATH = Path.home() / ".llm-writing-optimizer" / "rewrite_profiles.json"

STORE_VERSION = 1
DEFAULT_LOCAL_PROFILE = "local-rewrite"
LEGACY_LOCAL_PROFILE = "local-qwen"

# OpenClaw-inspired provider env fallback candidates. The runtime uses the
# profile first, then generic rewrite env vars, then provider-specific env vars.
_PROVIDER_ENV_API_KEY_CANDIDATES: Dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GEMINI_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "litellm": ("LITELLM_API_KEY",),
    "minimax": ("MINIMAX_API_KEY",),
    "moonshot": ("MOONSHOT_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "nvidia": ("NVIDIA_API_KEY",),
    "ollama": ("OLLAMA_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
    "vllm": ("VLLM_API_KEY",),
    "xai": ("XAI_API_KEY",),
}

_PROVIDER_ENV_API_KEY_ALIASES: Dict[str, str] = {
    "custom-anthropic": "anthropic",
    "custom-openai": "openai",
    "minimax-cn": "minimax",
}

_LEGACY_ENV_ALIASES: Dict[str, tuple[str, ...]] = {
    "backend": ("QWEN_BACKEND",),
    "model_path": ("QWEN_MODEL_PATH",),
    "base_url": ("QWEN_API_BASE_URL",),
    "model": ("QWEN_API_MODEL",),
}


def _default_local_profile() -> Dict[str, Any]:
    return {
        "backend": "local",
        "model_path": "./models/Qwen3-VL-8B-Instruct",
        "label": "Local rewrite model (Qwen3-VL-8B-Instruct)",
        # Disable thinking/reasoning mode by default to reduce latency and
        # token cost on short-text rewrite tasks.
        "disable_thinking": True,
        "provider_id": DEFAULT_LOCAL_PROFILE,
    }


def _default_store() -> Dict[str, Any]:
    return {
        "version": STORE_VERSION,
        "active_profile": DEFAULT_LOCAL_PROFILE,
        "profiles": {
            DEFAULT_LOCAL_PROFILE: _default_local_profile(),
        },
    }


def _migrate_legacy_store() -> None:
    """Move a legacy profile store to the canonical user config dir on first run.

    Checks two legacy locations in priority order:
    1. Project-root ``.rewrite_profiles.local.json`` (early dev artefact).
    2. ``~/.llm-writing-optimizer/rewrite_profiles.json`` (old user-config dir
       used before the project was renamed to "trileaf").
    """
    if CONFIG_PATH.exists():
        return

    for legacy_path, remove_after in (
        (_LEGACY_CONFIG_PATH, True),
        (_LEGACY_USER_CONFIG_PATH, False),
    ):
        if not legacy_path.exists():
            continue
        try:
            raw = json.loads(legacy_path.read_text(encoding="utf-8"))
            normalized = _normalize_store(raw)
            USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
            try:
                os.chmod(CONFIG_PATH, 0o600)
            except OSError:
                pass
            if remove_after:
                legacy_path.unlink(missing_ok=True)
        except OSError:
            pass  # Non-fatal; load_store will fall through to _default_store()
        return  # Stop after the first successfully found legacy file


def _normalize_store(data: Dict[str, Any]) -> Dict[str, Any]:
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        profiles = {}

    legacy_profile = profiles.get(LEGACY_LOCAL_PROFILE)
    default_profile = profiles.get(DEFAULT_LOCAL_PROFILE)
    if isinstance(legacy_profile, dict):
        migrated_profile = dict(legacy_profile)
        if migrated_profile.get("provider_id") == LEGACY_LOCAL_PROFILE:
            migrated_profile["provider_id"] = DEFAULT_LOCAL_PROFILE
        if default_profile is None:
            profiles[DEFAULT_LOCAL_PROFILE] = migrated_profile
        del profiles[LEGACY_LOCAL_PROFILE]

    if DEFAULT_LOCAL_PROFILE not in profiles:
        profiles[DEFAULT_LOCAL_PROFILE] = _default_local_profile()
    elif isinstance(profiles[DEFAULT_LOCAL_PROFILE], dict):
        profiles[DEFAULT_LOCAL_PROFILE].setdefault("provider_id", DEFAULT_LOCAL_PROFILE)

    active = data.get("active_profile") or DEFAULT_LOCAL_PROFILE
    if active == LEGACY_LOCAL_PROFILE:
        active = DEFAULT_LOCAL_PROFILE
    if active not in profiles:
        active = DEFAULT_LOCAL_PROFILE

    return {
        "version": STORE_VERSION,
        "active_profile": active,
        "profiles": profiles,
    }


def load_store() -> Dict[str, Any]:
    _migrate_legacy_store()
    if not CONFIG_PATH.exists():
        return _default_store()

    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_store()

    if not isinstance(data, dict):
        return _default_store()

    return _normalize_store(data)


def save_store(store: Dict[str, Any]) -> None:
    payload = _normalize_store(store)
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass


def get_profile(store: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    if name == LEGACY_LOCAL_PROFILE:
        name = DEFAULT_LOCAL_PROFILE
    profiles = store.get("profiles") or {}
    profile = profiles.get(name)
    return profile if isinstance(profile, dict) else None


def set_active_profile(store: Dict[str, Any], name: str) -> None:
    if name == LEGACY_LOCAL_PROFILE:
        name = DEFAULT_LOCAL_PROFILE
    if name not in (store.get("profiles") or {}):
        raise KeyError(f"Unknown profile: {name}")
    store["active_profile"] = name


def upsert_profile(store: Dict[str, Any], name: str, profile: Dict[str, Any]) -> None:
    if name == LEGACY_LOCAL_PROFILE:
        name = DEFAULT_LOCAL_PROFILE
    if profile.get("provider_id") == LEGACY_LOCAL_PROFILE:
        profile = {**profile, "provider_id": DEFAULT_LOCAL_PROFILE}
    profiles = store.setdefault("profiles", {})
    profiles[name] = profile
    if not store.get("active_profile"):
        store["active_profile"] = name


def delete_profile(store: Dict[str, Any], name: str) -> None:
    if name == DEFAULT_LOCAL_PROFILE:
        raise ValueError("The built-in local profile cannot be deleted.")
    profiles = store.get("profiles") or {}
    profiles.pop(name, None)
    if store.get("active_profile") == name:
        store["active_profile"] = DEFAULT_LOCAL_PROFILE


def resolve_profile_name(store: Dict[str, Any], explicit: str | None = None) -> str:
    if explicit:
        if explicit == LEGACY_LOCAL_PROFILE:
            return DEFAULT_LOCAL_PROFILE
        return explicit
    env_name = os.getenv("REWRITE_PROFILE", "").strip()
    if env_name:
        if env_name == LEGACY_LOCAL_PROFILE:
            return DEFAULT_LOCAL_PROFILE
        return env_name
    return str(store.get("active_profile") or DEFAULT_LOCAL_PROFILE)


def _trim_to_optional(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def normalize_provider_id(value: Any) -> str:
    s = _trim_to_optional(value) or ""
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _trim_credential(value: Any) -> Optional[str]:
    """Return the string value, or None if it looks like an unresolved env-var placeholder."""
    s = _trim_to_optional(value)
    if s and _ENV_VAR_RE.fullmatch(s):
        import warnings

        warnings.warn(
            f"Profile credential looks like an unresolved env-var placeholder: {s!r}. "
            "Set the referenced env var or store the actual key value in the profile.",
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


def legacy_env_first(
    alias_group: str,
    *,
    env: Mapping[str, str] | None = None,
) -> Optional[str]:
    env_map = os.environ if env is None else env
    for env_name in _LEGACY_ENV_ALIASES.get(alias_group, ()):
        value = _trim_to_optional(env_map.get(env_name))
        if value:
            return value
    return None


def load_selected_profile(profile_name: str | None = None) -> Optional[Dict[str, Any]]:
    store = load_store()
    name = resolve_profile_name(store, explicit=profile_name)
    profile = get_profile(store, name)
    if profile is None:
        return None
    return {
        "name": name,
        "profile": profile,
        "store": store,
    }


def resolve_provider_id(
    profile: Dict[str, Any] | None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    env_map = os.environ if env is None else env
    provider_id = first_defined(
        resolve_profile_value(profile, "provider_id"),
        _trim_to_optional(env_map.get("REWRITE_PROVIDER_ID")),
    )
    normalized = normalize_provider_id(provider_id)
    if normalized:
        return normalized

    base_url = (first_defined(
        resolve_profile_value(profile, "base_url"),
        _trim_to_optional(env_map.get("REWRITE_BASE_URL")),
        legacy_env_first("base_url", env=env_map),
    ) or "").lower()
    api_kind = (first_defined(
        resolve_profile_value(profile, "api_kind"),
        _trim_to_optional(env_map.get("REWRITE_API_KIND")),
    ) or "").lower()

    if "openrouter.ai" in base_url:
        return "openrouter"
    if "api.openai.com" in base_url:
        return "openai"
    if "api.anthropic.com" in base_url or api_kind == "anthropic-messages":
        return "anthropic"
    return ""


def resolve_profile_value(
    profile: Dict[str, Any] | None,
    key: str,
    *,
    secret: bool = False,
) -> Optional[str]:
    if not profile:
        return None
    value = profile.get(key)
    if secret:
        return _trim_credential(value)
    return _trim_to_optional(value)


def resolve_api_key(
    profile: Dict[str, Any] | None,
    *,
    provider_id: Any = None,
    env: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    env_map = os.environ if env is None else env
    normalized_provider = normalize_provider_id(provider_id) or resolve_provider_id(profile, env=env_map)
    candidates = get_provider_env_api_key_candidates(normalized_provider)
    profile_value = resolve_profile_value(profile, "api_key", secret=True)
    if profile_value:
        return {
            "value": profile_value,
            "source": "profile",
            "env_name": None,
            "provider_id": normalized_provider,
            "candidates": candidates,
        }

    for env_name in candidates:
        env_value = _trim_credential(env_map.get(env_name))
        if env_value not in (None, ""):
            return {
                "value": env_value,
                "source": f"env:{env_name}",
                "env_name": env_name,
                "provider_id": normalized_provider,
                "candidates": candidates,
            }

    return {
        "value": "",
        "source": "",
        "env_name": None,
        "provider_id": normalized_provider,
        "candidates": candidates,
    }


def _profile_to_env(profile: Dict[str, Any]) -> Dict[str, str]:
    backend = str(profile.get("backend") or "local").strip().lower()
    data: Dict[str, str] = {
        "REWRITE_BACKEND": backend,
        "REWRITE_PROVIDER_ID": normalize_provider_id(profile.get("provider_id")),
    }

    # disable_thinking applies to both local and external backends.
    data["REWRITE_DISABLE_THINKING"] = (
        "true" if profile.get("disable_thinking", True) else "false"
    )

    if backend == "local":
        data["REWRITE_MODEL_PATH"] = str(
            profile.get("model_path") or "./models/Qwen3-VL-8B-Instruct"
        )
        return data

    extra_headers = profile.get("extra_headers") or {}
    if not isinstance(extra_headers, dict):
        extra_headers = {}
    extra_body = profile.get("extra_body") or {}
    if not isinstance(extra_body, dict):
        extra_body = {}

    data.update(
        {
            "REWRITE_API_KIND": str(profile.get("api_kind") or "openai-chat-completions"),
            "REWRITE_BASE_URL": str(profile.get("base_url") or ""),
            "REWRITE_MODEL": str(profile.get("model") or ""),
            "REWRITE_API_KEY": _trim_credential(profile.get("api_key")) or "",
            "REWRITE_AUTH_MODE": str(profile.get("auth_mode") or "bearer"),
            "REWRITE_AUTH_HEADER": str(profile.get("auth_header") or ""),
            "REWRITE_EXTRA_HEADERS_JSON": json.dumps(extra_headers, ensure_ascii=True),
            "REWRITE_EXTRA_BODY_JSON": json.dumps(extra_body, ensure_ascii=True),
            "REWRITE_TIMEOUT_S": str(profile.get("timeout_s") or 120),
            "REWRITE_TEMPERATURE": str(profile.get("temperature") or 0.7),
        }
    )
    return data


def apply_active_profile_env(
    *,
    override: bool = True,
    profile_name: str | None = None,
) -> Optional[Dict[str, Any]]:
    store = load_store()
    name = resolve_profile_name(store, explicit=profile_name)
    profile = get_profile(store, name)
    if profile is None:
        return None

    env_values = _profile_to_env(profile)
    for key, value in env_values.items():
        if value in ("", None):
            continue
        if override or key not in os.environ:
            os.environ[key] = value

    if override or "REWRITE_PROFILE" not in os.environ:
        os.environ["REWRITE_PROFILE"] = name
    return {"name": name, "profile": profile}


def mask_secret(value: str) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def profile_summary(name: str, profile: Dict[str, Any]) -> str:
    backend = str(profile.get("backend") or "local")
    if backend == "local":
        model_path = str(profile.get("model_path") or "./models/Qwen3-VL-8B-Instruct")
        return f"{name}: local -> {model_path}"

    api_kind = str(profile.get("api_kind") or "openai-chat-completions")
    model = str(profile.get("model") or "(unset)")
    base_url = str(profile.get("base_url") or "(unset)")
    auth_mode = str(profile.get("auth_mode") or "bearer")
    api_key = mask_secret(str(profile.get("api_key") or ""))
    return (
        f"{name}: external [{api_kind}] model={model} base_url={base_url} "
        f"auth={auth_mode} key={api_key}"
    )


def main(argv: list[str] | None = None) -> int:
    from scripts import rewrite_provider_cli

    return rewrite_provider_cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
