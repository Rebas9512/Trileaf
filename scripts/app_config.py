"""Application configuration for Trileaf.

Config file: ~/.trileaf/config.json
Managed by the onboarding wizard or edited manually.

This replaces the .env file for all non-secret, non-rewrite-provider settings:
  - model_paths: local model directory locations
  - pipeline:    gate thresholds and default utility weights
  - dashboard:   server host / port

Rewrite-provider credentials (API keys, model selections, backend choice) are
still managed by rewrite_config.py via ~/.trileaf/rewrite_profiles.json.
Environment variables remain available only as manual override hooks for CI,
Docker, or advanced local debugging.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_CONFIG_DIR = Path.home() / ".trileaf"
CONFIG_PATH = USER_CONFIG_DIR / "config.json"
CONFIG_VERSION = 1

_DEFAULTS: Dict[str, Any] = {
    "version": CONFIG_VERSION,
    "model_paths": {
        "desklib": "./models/desklib-ai-text-detector-v1.01",
        "mpnet": "./models/sentence-transformers-paraphrase-mpnet-base-v2",
    },
    "pipeline": {
        "max_chunk_chars": 200,
        "max_chunk_chars_long": 400,
        "sem_gate": 0.65,
        "min_sent_sim_gate": 0.35,
        "len_ratio_min": 0.60,
        "len_ratio_max": 1.80,
        "w_ai": 0.60,
        "w_sem": 0.35,
        "w_risk": 0.05,
    },
    "dashboard": {
        "host": "127.0.0.1",
        "port": 8001,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict."""
    result: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> Dict[str, Any]:
    """Return the config, deep-merged with defaults for any missing keys."""
    if not CONFIG_PATH.exists():
        return _deep_merge({}, _DEFAULTS)
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _deep_merge({}, _DEFAULTS)
        return _deep_merge(_DEFAULTS, data)
    except (OSError, json.JSONDecodeError):
        return _deep_merge({}, _DEFAULTS)


def save_config(config: Dict[str, Any]) -> None:
    """Persist *config* to ~/.trileaf/config.json (creates dirs as needed)."""
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def ensure_config_exists() -> None:
    """Write the default config file if it does not yet exist."""
    if not CONFIG_PATH.exists():
        save_config(_deep_merge({}, _DEFAULTS))


def resolve_model_path(key: str) -> Path:
    """Return the resolved absolute path for *key* (``'desklib'`` or ``'mpnet'``).

    Relative paths are resolved against the project root.
    Environment variables DESKLIB_MODEL_PATH / MPNET_MODEL_PATH still take
    precedence so that CI / Docker overrides remain possible without editing
    the config file.
    """
    env_map = {"desklib": "DESKLIB_MODEL_PATH", "mpnet": "MPNET_MODEL_PATH"}
    env_override = os.environ.get(env_map.get(key, ""), "").strip()
    if env_override:
        p = Path(env_override)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    config = load_config()
    raw = config.get("model_paths", {}).get(key) or _DEFAULTS["model_paths"].get(key, "")
    p = Path(str(raw))
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def get_pipeline_config() -> Dict[str, Any]:
    """Return pipeline config dict with all defaults filled in."""
    config = load_config()
    defaults = dict(_DEFAULTS["pipeline"])
    overrides = config.get("pipeline") or {}
    if isinstance(overrides, dict):
        defaults.update({k: v for k, v in overrides.items() if v is not None})
    return defaults


def get_dashboard_config() -> Dict[str, Any]:
    """Return dashboard config dict with all defaults filled in."""
    config = load_config()
    defaults = dict(_DEFAULTS["dashboard"])
    overrides = config.get("dashboard") or {}
    if isinstance(overrides, dict):
        defaults.update({k: v for k, v in overrides.items() if v is not None})
    return defaults
