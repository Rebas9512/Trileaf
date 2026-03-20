#!/usr/bin/env python3
"""
Interactive wizard for configuring the Trileaf rewrite provider.

Writes configuration to PROJECT_ROOT/.env (chmod 600).
Credentials never leave the local machine — no JSON store, no external sync.

Commands
--------
  wizard   — interactive step-by-step setup (default)
  show     — print current .env config (masked key)
  clear    — remove REWRITE_* and LEAFHUB_* keys from .env
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import rewrite_config as rc

# ── Provider presets ──────────────────────────────────────────────────────────

_PRESETS = [
    {"id": "openai",      "label": "OpenAI",          "base_url": "https://api.openai.com/v1",          "model": "gpt-4o",               "api_kind": "openai-chat-completions"},
    {"id": "anthropic",   "label": "Anthropic",        "base_url": "https://api.anthropic.com",          "model": "claude-sonnet-4-6",     "api_kind": "anthropic-messages",    "auth_mode": "x-api-key"},
    {"id": "groq",        "label": "Groq",             "base_url": "https://api.groq.com/openai/v1",     "model": "llama-3.3-70b-versatile", "api_kind": "openai-chat-completions"},
    {"id": "openrouter",  "label": "OpenRouter",       "base_url": "https://openrouter.ai/api/v1",       "model": "openai/gpt-4o",        "api_kind": "openai-chat-completions"},
    {"id": "mistral",     "label": "Mistral",          "base_url": "https://api.mistral.ai/v1",          "model": "mistral-large-latest", "api_kind": "openai-chat-completions"},
    {"id": "together",    "label": "Together AI",      "base_url": "https://api.together.xyz/v1",        "model": "meta-llama/Llama-3-70b-chat-hf", "api_kind": "openai-chat-completions"},
    {"id": "xai",         "label": "xAI (Grok)",       "base_url": "https://api.x.ai/v1",               "model": "grok-2",               "api_kind": "openai-chat-completions"},
    {"id": "ollama",      "label": "Ollama (local)",   "base_url": "http://localhost:11434/v1",          "model": "llama3",               "api_kind": "openai-chat-completions", "auth_mode": "none"},
    {"id": "vllm",        "label": "vLLM (local)",     "base_url": "http://localhost:8000/v1",           "model": "",                     "api_kind": "openai-chat-completions", "auth_mode": "none"},
    {"id": "custom",      "label": "Custom / other",   "base_url": "",                                   "model": "",                     "api_kind": "openai-chat-completions"},
]


# ── UI helpers ────────────────────────────────────────────────────────────────

_DIVIDER = "─" * 62


def _print(msg: str = "") -> None:
    print(msg)


def _header(title: str) -> None:
    _print()
    _print(_DIVIDER)
    _print(f"  {title}")
    _print(_DIVIDER)


def _ask(prompt: str, default: str = "") -> str:
    hint = f"  [{default}]" if default else ""
    try:
        raw = input(f"  {prompt}{hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        _print()
        raise SystemExit(0)
    return raw or default


def _ask_secret(prompt: str) -> str:
    import getpass
    try:
        return getpass.getpass(f"  {prompt}: ").strip()
    except (EOFError, KeyboardInterrupt):
        _print()
        raise SystemExit(0)


def _choose(prompt: str, options: list[tuple[str, str]]) -> str:
    """Display numbered list, return chosen value."""
    _print()
    for i, (_, label) in enumerate(options, 1):
        _print(f"  {i}. {label}")
    _print()
    while True:
        raw = _ask(prompt)
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        _print("  Please enter a number from the list.")


def _mask(value: str) -> str:
    return rc.mask_secret(value)


# ── Config inspection ─────────────────────────────────────────────────────────

def _show_current() -> None:
    """Print the current .env config in a human-readable form."""
    _header("Current .env configuration")
    if not rc.ENV_FILE.is_file():
        _print(f"  No .env file found at: {rc.ENV_FILE}")
        _print()
        _print("  Run:  trileaf config")
        return

    _print(f"  File: {rc.ENV_FILE}")
    _print()

    leafhub_alias = rc._read_dot_env_key(rc.ENV_FILE, "LEAFHUB_ALIAS")
    backend = rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_BACKEND") or "(not set)"
    base_url = rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_BASE_URL") or "(not set)"
    model = rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_MODEL") or "(not set)"
    api_kind = rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_API_KIND") or ""
    api_key = rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_API_KEY") or ""
    auth_mode = rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_AUTH_MODE") or "bearer"

    if leafhub_alias:
        _print(f"  Credential source : LeafHub (alias={leafhub_alias})")
    elif api_key:
        _print(f"  Credential source : .env  REWRITE_API_KEY={_mask(api_key)}")
    else:
        _print("  Credential source : (none — set REWRITE_API_KEY or link LeafHub)")

    _print(f"  Backend           : {backend}")
    _print(f"  Base URL          : {base_url}")
    _print(f"  Model             : {model}")
    if api_kind:
        _print(f"  API kind          : {api_kind}")
    if backend == "external":
        _print(f"  Auth mode         : {auth_mode}")
    _print()


# ── LeafHub wizard ────────────────────────────────────────────────────────────

# LeafHub api_format → Trileaf api_kind normalisation
_LH_FORMAT_MAP: dict[str, str] = {
    "openai-completions":  "openai-chat-completions",
    "openai_completions":  "openai-chat-completions",
}


def _lh_prefill(hub: object, alias: str) -> dict[str, str]:
    """Return pre-filled wizard defaults from hub.get_config(alias). Never raises."""
    try:
        cfg = hub.get_config(alias)  # type: ignore[union-attr]
        raw_format = getattr(cfg, "api_format", "") or ""
        return {
            "base_url":  getattr(cfg, "base_url", "") or "",
            "model":     getattr(cfg, "model", "") or "",
            "api_kind":  _LH_FORMAT_MAP.get(raw_format, raw_format) or "openai-chat-completions",
            "auth_mode": getattr(cfg, "auth_mode", "") or "bearer",
        }
    except Exception:
        return {}


def _wizard_leafhub(lh_found: object) -> int:
    _header("Configure via LeafHub")
    _print()

    if not lh_found.ready:  # type: ignore[union-attr]
        proj_name = _ask("Project name to create in LeafHub", default="trileaf")
        _print()
        try:
            from leafhub.probe import register as _lh_register
            lh_found = _lh_register(proj_name, project_dir=rc.PROJECT_ROOT, probe=lh_found)
            _print(f"  [OK] Project '{proj_name}' created and linked.")
        except RuntimeError as _exc:
            for _line in str(_exc).splitlines():
                _print(f"  {_line}")
            _print()
            _print("  After linking, re-run:  trileaf config")
            return 1

    # ── Alias selection ───────────────────────────────────────────────────────
    hub = lh_found.open_sdk()  # type: ignore[union-attr]
    alias: str = ""
    try:
        bound = hub.list_aliases()
    except Exception:
        bound = []

    if bound:
        _print("  Bound aliases in this LeafHub project:")
        alias_options = [(a, a) for a in bound] + [("_other", "Enter a different alias")]
        chosen = _choose("Select alias", alias_options)
        if chosen != "_other":
            alias = chosen
    if not alias:
        alias = _ask("LeafHub alias for the rewrite API key", default="rewrite")

    # ── Provider config: pull from LeafHub, only ask what's missing ──────────
    pre = _lh_prefill(hub, alias)

    if pre.get("base_url") and pre.get("model"):
        # Full profile available — no follow-up questions needed.
        base_url  = pre["base_url"]
        model     = pre["model"]
        api_kind  = pre.get("api_kind", "openai-chat-completions")
        auth_mode = pre.get("auth_mode", "bearer")
        _print()
        _print(f"  Base URL:  {base_url}")
        _print(f"  Model:     {model}")
    else:
        # Partial or no profile — ask for what's missing.
        _print()
        _print("  (Enter to accept pre-filled value, or type to override)")
        base_url  = _ask("Provider base URL", default=pre.get("base_url", ""))
        model     = _ask("Model name",        default=pre.get("model", ""))
        api_kind  = _ask("API kind",          default=pre.get("api_kind", "openai-chat-completions"))
        auth_mode = _ask("Auth mode (bearer / x-api-key / none)", default=pre.get("auth_mode", "bearer"))

    env_values: dict[str, str] = {
        "LEAFHUB_ALIAS":     alias,
        "REWRITE_BACKEND":   "external",
        "REWRITE_AUTH_MODE": auth_mode,
    }
    if base_url:
        env_values["REWRITE_BASE_URL"] = base_url
    if model:
        env_values["REWRITE_MODEL"] = model
    if api_kind and api_kind != "openai-chat-completions":
        env_values["REWRITE_API_KIND"] = api_kind

    rc.write_dot_env(env_values, header="Trileaf rewrite provider — credentials via LeafHub")
    _print()
    _print(f"  [OK] Wrote {rc.ENV_FILE}")
    _print(f"       LeafHub alias: {alias}")
    if base_url:
        _print(f"       Base URL:     {base_url}")
    if model:
        _print(f"       Model:        {model}")
    _print("       API key fetched from LeafHub vault at runtime.")
    return 0


# ── External API wizard ───────────────────────────────────────────────────────

def _wizard_external() -> int:
    _header("Configure external API provider")
    _print()
    _print("  Select a provider preset (you can customise any field):")

    preset_options = [(p["id"], p["label"]) for p in _PRESETS]
    choice_id = _choose("Provider", preset_options)
    preset = next(p for p in _PRESETS if p["id"] == choice_id)

    _print()
    base_url = _ask("Base URL", default=preset["base_url"])
    model    = _ask("Model",    default=preset["model"])
    api_kind = _ask("API kind", default=preset.get("api_kind", "openai-chat-completions"))
    auth_mode = _ask("Auth mode (bearer / x-api-key / none)", default=preset.get("auth_mode", "bearer"))

    _print()
    if auth_mode == "none":
        api_key = ""
        _print("  Auth mode 'none' — no API key required.")
    else:
        _print("  API key (input hidden):")
        api_key = _ask_secret("API key")
        if not api_key:
            _print("  [WARN] No API key entered — REWRITE_API_KEY will be empty.")

    env_values: dict[str, str] = {
        "REWRITE_BACKEND":  "external",
        "REWRITE_PROVIDER_ID": choice_id,
        "REWRITE_BASE_URL": base_url,
        "REWRITE_MODEL":    model,
        "REWRITE_API_KIND": api_kind,
        "REWRITE_AUTH_MODE": auth_mode,
    }
    if api_key:
        env_values["REWRITE_API_KEY"] = api_key
    # Remove any stale LeafHub alias if switching away
    if rc._read_dot_env_key(rc.ENV_FILE, "LEAFHUB_ALIAS"):
        env_values["LEAFHUB_ALIAS"] = ""

    rc.write_dot_env(
        env_values,
        header=f"Trileaf rewrite provider — {preset['label']}",
    )
    _print()
    _print(f"  [OK] Wrote {rc.ENV_FILE}")
    _print(f"       Provider: {preset['label']}")
    _print(f"       Model:    {model}")
    _print(f"       Key:      {_mask(api_key) if api_key else '(none)'}")
    return 0


# ── Local model wizard ────────────────────────────────────────────────────────

def _wizard_local() -> int:
    _header("Configure local Qwen3-VL-8B rewrite model")
    _print()
    default_path = str(rc.PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct")
    model_path = _ask("Model path", default=default_path)

    env_values: dict[str, str] = {
        "REWRITE_BACKEND":    "local",
        "REWRITE_MODEL_PATH": model_path,
    }
    rc.write_dot_env(env_values, header="Trileaf rewrite provider — local Qwen3-VL-8B")
    _print()
    _print(f"  [OK] Wrote {rc.ENV_FILE}")
    _print(f"       Model path: {model_path}")
    _print()
    _print("  Make sure the model is downloaded:")
    _print("    python -m scripts.download_scripts.qwen3_vl_download")
    return 0


# ── Main wizard ───────────────────────────────────────────────────────────────

def _run_wizard() -> int:
    _header("Trileaf — Rewrite Provider Setup")
    _print()
    _print("  Configuration is stored in:  .env  (project root, chmod 600)")
    _print("  API keys never leave your machine.")
    _print()

    # Check for LeafHub
    lh_found = None
    try:
        from leafhub.probe import detect as _lh_detect
        lh_found = _lh_detect(project_dir=rc.PROJECT_ROOT)
    except ImportError:
        pass

    lh_available = lh_found is not None and (lh_found.ready or lh_found.can_link)

    options: list[tuple[str, str]] = []
    if lh_available:
        status = "linked" if lh_found.ready else f"available at {lh_found.manage_url}"
        options.append(("leafhub", f"LeafHub  ({status}) — vault-managed credentials"))
    options += [
        ("external", "External API  (key stored in .env)"),
        ("local",    "Local Qwen3-VL-8B  (offline, ≥16 GB VRAM)"),
    ]

    choice = _choose("Credential source", options)

    if choice == "leafhub":
        return _wizard_leafhub(lh_found)
    if choice == "external":
        return _wizard_external()
    return _wizard_local()


# ── Clear command ─────────────────────────────────────────────────────────────

def _clear_env() -> int:
    """Remove REWRITE_* and LEAFHUB_* entries from .env."""
    _CLEAR_KEYS = {
        "REWRITE_BACKEND", "REWRITE_PROVIDER_ID", "REWRITE_BASE_URL",
        "REWRITE_MODEL", "REWRITE_API_KIND", "REWRITE_API_KEY",
        "REWRITE_AUTH_MODE", "REWRITE_AUTH_HEADER", "REWRITE_TIMEOUT_S",
        "REWRITE_TEMPERATURE", "REWRITE_DISABLE_THINKING",
        "REWRITE_EXTRA_HEADERS_JSON", "REWRITE_EXTRA_BODY_JSON",
        "LEAFHUB_ALIAS",
    }
    if not rc.ENV_FILE.is_file():
        print("  No .env file found — nothing to clear.")
        return 0

    try:
        lines = rc.ENV_FILE.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"  [ERROR] Could not read .env: {exc}")
        return 1

    kept: list[str] = []
    removed: list[str] = []
    for line in lines:
        result = rc._parse_env_line(line)
        if result and result[0] in _CLEAR_KEYS:
            removed.append(result[0])
        else:
            kept.append(line)

    rc.ENV_FILE.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    if removed:
        print(f"  Removed from .env: {', '.join(removed)}")
    else:
        print("  No rewrite-provider keys found in .env.")
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="trileaf config",
        description="Configure the Trileaf rewrite provider (.env).",
    )
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("wizard", help="Interactive setup wizard (default)")
    sub.add_parser("show",   help="Show current .env configuration")
    sub.add_parser("clear",  help="Remove rewrite-provider keys from .env")

    args = parser.parse_args(argv or [])
    cmd = args.cmd or "wizard"

    if cmd == "show":
        _show_current()
        return 0
    if cmd == "clear":
        return _clear_env()
    # wizard (default)
    return _run_wizard()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
