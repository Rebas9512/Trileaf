#!/usr/bin/env python3
"""
Interactive CLI for local rewrite-provider profiles.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import rewrite_config as rc


BACK_VALUE = "__back"
CUSTOM_VALUE = "__custom__"


@dataclass(frozen=True)
class ChoiceOption:
    value: str
    label: str
    hint: str = ""


@dataclass(frozen=True)
class ProviderMethod:
    value: str
    label: str
    hint: str
    backend: str
    provider_id: str
    api_kind: str = "openai-chat-completions"
    base_url: str = ""
    default_model: str = ""
    auth_mode: str = "bearer"
    auth_header: str = ""
    timeout_s: float = 120.0
    temperature: float = 0.7
    notes: tuple[str, ...] = ()
    prompt_model_label: str = "Rewrite model id"
    prompt_base_url_label: str = "API base URL"
    prompt_api_key_label: str = "API key"
    default_extra_headers: Dict[str, str] = field(default_factory=dict)
    default_extra_body: Dict[str, Any] = field(default_factory=dict)
    model_options: tuple[ChoiceOption, ...] = ()


@dataclass(frozen=True)
class ProviderGroup:
    value: str
    label: str
    hint: str
    methods: tuple[ProviderMethod, ...]


LOCAL_REWRITE_METHOD = ProviderMethod(
    value=rc.DEFAULT_LOCAL_PROFILE,
    label="Local rewrite model",
    hint="Use the local Qwen rewrite model already stored in this project",
    backend="local",
    provider_id=rc.DEFAULT_LOCAL_PROFILE,
    notes=(
        "Best when you want the whole pipeline to stay on-box.",
        "Uses the existing local model path unless you override it here.",
    ),
)

OPENAI_MODEL_OPTIONS = (
    ChoiceOption("gpt-4.1-mini", "gpt-4.1-mini", "Fast and economical default"),
)

OPENAI_METHOD = ProviderMethod(
    value="openai-api-key",
    label="OpenAI API key",
    hint="Official Chat Completions endpoint",
    backend="external",
    provider_id="openai",
    api_kind="openai-chat-completions",
    base_url="https://api.openai.com/v1",
    default_model="gpt-4.1-mini",
    auth_mode="bearer",
    auth_header="Authorization",
    prompt_api_key_label="OpenAI API key",
    notes=(
        "Use this for the official OpenAI API.",
        "Pick a chat-completions model that is good at rewriting and instruction following.",
    ),
    model_options=OPENAI_MODEL_OPTIONS,
)

ANTHROPIC_MODEL_OPTIONS = (
    ChoiceOption("claude-3-5-haiku-latest", "claude-3-5-haiku-latest", "Fast default"),
)

ANTHROPIC_METHOD = ProviderMethod(
    value="anthropic-api-key",
    label="Anthropic API key",
    hint="Messages API key",
    backend="external",
    provider_id="anthropic",
    api_kind="anthropic-messages",
    base_url="https://api.anthropic.com/v1",
    default_model="claude-3-5-haiku-latest",
    auth_mode="x-api-key",
    auth_header="x-api-key",
    prompt_api_key_label="Anthropic API key",
    notes=(
        "Use this for the official Anthropic Messages API.",
        "Anthropic-compatible endpoints usually expect x-api-key plus anthropic-version.",
    ),
    model_options=ANTHROPIC_MODEL_OPTIONS,
)

MINIMAX_MODEL_OPTIONS = (
    ChoiceOption("MiniMax-M2.5", "MiniMax-M2.5", "Recommended default"),
    ChoiceOption("MiniMax-M2.5-highspeed", "MiniMax-M2.5-highspeed", "Official faster tier"),
)

MINIMAX_METHOD = ProviderMethod(
    value="minimax-api-key",
    label="MiniMax M2.5",
    hint="Official global endpoint (recommended)",
    backend="external",
    provider_id="minimax",
    api_kind="anthropic-messages",
    base_url="https://api.minimax.io/anthropic",
    default_model="MiniMax-M2.5",
    auth_mode="bearer",
    auth_header="Authorization",
    prompt_api_key_label="MiniMax API key",
    notes=(
        "MiniMax exposes an Anthropic-compatible API on the /anthropic endpoint.",
        "This template uses Authorization: Bearer plus the runtime's anthropic-version header.",
    ),
    model_options=MINIMAX_MODEL_OPTIONS,
)

MINIMAX_CN_METHOD = ProviderMethod(
    value="minimax-api-key-cn",
    label="MiniMax M2.5 (CN)",
    hint="China endpoint (api.minimaxi.com)",
    backend="external",
    provider_id="minimax-cn",
    api_kind="anthropic-messages",
    base_url="https://api.minimaxi.com/anthropic",
    default_model="MiniMax-M2.5",
    auth_mode="bearer",
    auth_header="Authorization",
    prompt_api_key_label="MiniMax API key",
    notes=(
        "Use this when your MiniMax account should route to the CN endpoint.",
        "Model presets stay the same; only the provider id and base URL differ.",
    ),
    model_options=MINIMAX_MODEL_OPTIONS,
)

MOONSHOT_MODEL_OPTIONS = (
    ChoiceOption("kimi-k2.5", "kimi-k2.5", "Latest Kimi text model"),
)

MOONSHOT_METHOD = ProviderMethod(
    value="moonshot-api-key",
    label="Moonshot AI (Kimi K2.5)",
    hint="Global endpoint (.ai)",
    backend="external",
    provider_id="moonshot",
    api_kind="openai-chat-completions",
    base_url="https://api.moonshot.ai/v1",
    default_model="kimi-k2.5",
    auth_mode="bearer",
    auth_header="Authorization",
    prompt_api_key_label="Moonshot API key",
    notes=(
        "Moonshot is accessed via the OpenAI-compatible chat completions API.",
        "Use this for Kimi K2.5 on the global Moonshot endpoint.",
        "This preset disables thinking by default to keep rewrite latency low.",
    ),
    default_extra_body={"thinking": {"type": "disabled"}},
    model_options=MOONSHOT_MODEL_OPTIONS,
)

MOONSHOT_CN_METHOD = ProviderMethod(
    value="moonshot-api-key-cn",
    label="Moonshot AI (CN)",
    hint="China endpoint (.cn)",
    backend="external",
    provider_id="moonshot",
    api_kind="openai-chat-completions",
    base_url="https://api.moonshot.cn/v1",
    default_model="kimi-k2.5",
    auth_mode="bearer",
    auth_header="Authorization",
    prompt_api_key_label="Moonshot API key",
    notes=(
        "Use this when your Moonshot deployment should point at the CN endpoint.",
        "The runtime still uses the OpenAI-compatible chat/completions route.",
        "This preset disables thinking by default to keep rewrite latency low.",
    ),
    default_extra_body={"thinking": {"type": "disabled"}},
    model_options=MOONSHOT_MODEL_OPTIONS,
)

OPENROUTER_METHOD = ProviderMethod(
    value="openrouter-api-key",
    label="OpenRouter API key",
    hint="OpenAI-compatible aggregator",
    backend="external",
    provider_id="openrouter",
    api_kind="openai-chat-completions",
    base_url="https://openrouter.ai/api/v1",
    default_model="openai/gpt-4.1-mini",
    auth_mode="bearer",
    auth_header="Authorization",
    default_extra_headers={"X-Title": "Trileaf"},
    prompt_api_key_label="OpenRouter API key",
    notes=(
        "OpenRouter is OpenAI-compatible, so this project uses the chat/completions path.",
        "Optional headers like HTTP-Referer and X-Title can help with provider dashboards.",
    ),
    model_options=(
        ChoiceOption("openai/gpt-4.1-mini", "openai/gpt-4.1-mini", "Concrete default model"),
        ChoiceOption("openrouter/auto", "openrouter/auto", "Provider-side automatic routing"),
    ),
)

LITELLM_METHOD = ProviderMethod(
    value="litellm-api-key",
    label="LiteLLM gateway",
    hint="Unified LLM gateway (100+ providers)",
    backend="external",
    provider_id="litellm",
    api_kind="openai-chat-completions",
    base_url="http://127.0.0.1:4000/v1",
    default_model="gpt-4.1-mini",
    auth_mode="bearer",
    auth_header="Authorization",
    prompt_api_key_label="LiteLLM API key",
    notes=(
        "Use this when LiteLLM is fronting multiple providers for you.",
        "If your LiteLLM instance does not require auth, switch auth mode to 'none'.",
    ),
    model_options=(
        ChoiceOption("gpt-4.1-mini", "gpt-4.1-mini", "Keep the gateway example model"),
        ChoiceOption("your-router-default", "your-router-default", "If your gateway exposes a router alias"),
    ),
)

VLLM_METHOD = ProviderMethod(
    value="vllm-openai-compatible",
    label="vLLM / self-hosted OpenAI-compatible",
    hint="Local or private OpenAI-compatible inference server",
    backend="external",
    provider_id="vllm",
    api_kind="openai-chat-completions",
    base_url="http://127.0.0.1:8000/v1",
    default_model="Qwen/Qwen2.5-7B-Instruct",
    auth_mode="none",
    auth_header="Authorization",
    notes=(
        "Good for local/self-hosted inference behind an OpenAI-compatible API.",
        "If your proxy adds auth, switch auth mode from 'none' to bearer or x-api-key.",
    ),
    model_options=(
        ChoiceOption("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Example self-hosted model id"),
    ),
)

CUSTOM_OPENAI_METHOD = ProviderMethod(
    value="custom-openai-compatible",
    label="Custom OpenAI-compatible endpoint",
    hint="Any provider exposing /chat/completions",
    backend="external",
    provider_id="custom-openai",
    api_kind="openai-chat-completions",
    base_url="http://127.0.0.1:8000/v1",
    default_model="your-model-id",
    auth_mode="bearer",
    auth_header="Authorization",
    notes=(
        "Use this for custom gateways, reverse proxies, or OpenAI-compatible vendors.",
        "Confirm whether your base URL already includes /v1; the tool appends /chat/completions automatically.",
    ),
    model_options=(
        ChoiceOption(CUSTOM_VALUE, "Custom model id", "Type the model id used by your gateway"),
    ),
)

CUSTOM_ANTHROPIC_METHOD = ProviderMethod(
    value="custom-anthropic-compatible",
    label="Custom Anthropic-compatible endpoint",
    hint="Any provider exposing /messages",
    backend="external",
    provider_id="custom-anthropic",
    api_kind="anthropic-messages",
    base_url="http://127.0.0.1:8001/v1",
    default_model="your-model-id",
    auth_mode="x-api-key",
    auth_header="x-api-key",
    notes=(
        "Use this for Anthropic-compatible vendors or private gateways.",
        "The runtime auto-adds anthropic-version unless you override that header yourself.",
    ),
    model_options=(
        ChoiceOption(CUSTOM_VALUE, "Custom model id", "Type the model id used by your provider"),
    ),
)


PROVIDER_GROUPS: tuple[ProviderGroup, ...] = (
    ProviderGroup(
        value="local",
        label="Local",
        hint="Run the rewrite model on this machine",
        methods=(LOCAL_REWRITE_METHOD,),
    ),
    ProviderGroup(
        value="openai",
        label="OpenAI",
        hint="Official API key flow",
        methods=(OPENAI_METHOD,),
    ),
    ProviderGroup(
        value="anthropic",
        label="Anthropic",
        hint="Messages API key",
        methods=(ANTHROPIC_METHOD,),
    ),
    ProviderGroup(
        value="minimax",
        label="MiniMax",
        hint="M2.5 presets with prefilled global/CN endpoints",
        methods=(MINIMAX_METHOD, MINIMAX_CN_METHOD),
    ),
    ProviderGroup(
        value="moonshot",
        label="Moonshot AI (Kimi)",
        hint="Kimi K2.5 presets for .ai and .cn",
        methods=(MOONSHOT_METHOD, MOONSHOT_CN_METHOD),
    ),
    ProviderGroup(
        value="openrouter",
        label="OpenRouter",
        hint="OpenAI-compatible aggregator",
        methods=(OPENROUTER_METHOD,),
    ),
    ProviderGroup(
        value="litellm",
        label="LiteLLM",
        hint="Unified gateway for many providers",
        methods=(LITELLM_METHOD,),
    ),
    ProviderGroup(
        value="self-hosted",
        label="Self-hosted",
        hint="Local/private gateways and inference servers",
        methods=(VLLM_METHOD,),
    ),
    ProviderGroup(
        value="custom",
        label="Custom Provider",
        hint="Any OpenAI-compatible or Anthropic-compatible endpoint",
        methods=(CUSTOM_OPENAI_METHOD, CUSTOM_ANTHROPIC_METHOD),
    ),
)

METHOD_BY_ID = {
    method.value: method
    for group in PROVIDER_GROUPS
    for method in group.methods
}


HELP_TEXT = """Commands
  wizard                       Guided setup — only asks for API key (recommended)
  verify [name]                Quick connectivity check: send 1-token request to endpoint
  providers                    Show built-in provider templates
  status                       Show config path and active profile
  list                         List all profiles
  show <name>                  Show a profile JSON payload
  use <name>                   Set active profile
  delete <name>                Delete a profile
  test [name]                  Send a full test rewrite request (loads the pipeline)
  add-local <name>             Manual local profile editor
  add-api <name>               Manual external profile editor (all fields)
  auth <name>                  Update auth settings for an existing profile
  headers <name>               Update extra headers JSON for a profile
  body <name>                  Update extra request-body JSON for a profile
  save                         Persist current store
  help                         Show this help
  exit | quit                  Leave the shell
"""


def _print_section(title: str) -> None:
    print()
    print(title)


def _prompt(label: str, default: str = "", *, secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    if secret:
        # Use plain input() so paste always works reliably.
        # The value is stored only in the local profile file (chmod 600).
        text = f"{label} (visible){suffix}: "
    else:
        text = f"{label}{suffix}: "
    value = input(text)
    return value.strip() or default


def _prompt_float(label: str, default: float) -> float:
    raw = _prompt(label, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def _prompt_json(label: str, default: Dict[str, Any] | None = None) -> Dict[str, str]:
    default_text = json.dumps(default or {}, ensure_ascii=True)
    raw = _prompt(label, default_text)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass
    print("[rewrite-config] Invalid JSON, using {}.")
    return {}


def _prompt_json_object(label: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    default_text = json.dumps(default or {}, ensure_ascii=True)
    raw = _prompt(label, default_text)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    print("[rewrite-config] Invalid JSON object, using {}.")
    return {}


def _merge_json_objects(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_json_objects(merged[key], value)
        else:
            merged[key] = value
    return merged


def _match_option(options: Iterable[ChoiceOption], raw: str) -> Optional[str]:
    normalized = raw.strip().lower()
    if not normalized:
        return None
    for option in options:
        if normalized == option.value.lower():
            return option.value
        if normalized == option.label.lower():
            return option.value
    return None


def _select_option(
    message: str,
    options: List[ChoiceOption],
    *,
    default_value: str | None = None,
    allow_back: bool = False,
) -> str:
    if not options:
        raise ValueError("No options available.")

    value_set = {option.value for option in options}
    if default_value not in value_set:
        default_value = options[0].value
    default_label = next(
        (option.label for option in options if option.value == default_value),
        str(default_value),
    )

    while True:
        _print_section(message)
        for idx, option in enumerate(options, start=1):
            default_marker = " (default)" if option.value == default_value else ""
            print(f"  {idx}. {option.label}{default_marker}")
            if option.hint:
                print(f"     {option.hint}")
        if allow_back:
            print("  b. Back")

        raw = input(f"Choice [{default_label}]: ").strip()
        if not raw:
            return str(default_value)
        if allow_back and raw.lower() in {"b", "back"}:
            return BACK_VALUE
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx].value
        matched = _match_option(options, raw)
        if matched:
            return matched
        print("[rewrite-config] Invalid selection.")


def _select_model_value(method: ProviderMethod, existing_model: str) -> str:
    if not method.model_options:
        return _prompt(method.prompt_model_label, existing_model or method.default_model)

    options = list(method.model_options)
    values = {option.value for option in options}
    if CUSTOM_VALUE not in values:
        options.append(ChoiceOption(CUSTOM_VALUE, "Custom model id", "Type the exact model name yourself"))

    default_value = existing_model or method.default_model
    if default_value not in {option.value for option in options}:
        default_value = method.default_model
        if default_value not in {option.value for option in options}:
            default_value = CUSTOM_VALUE

    choice = _select_option("Rewrite model preset", options, default_value=default_value)
    if choice == CUSTOM_VALUE:
        custom_default = existing_model if existing_model and existing_model != CUSTOM_VALUE else method.default_model
        return _prompt(method.prompt_model_label, custom_default)
    return choice


def _confirm(message: str, *, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{message} [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def _auth_mode_options() -> List[ChoiceOption]:
    return [
        ChoiceOption("bearer", "Bearer token", "Authorization: Bearer <key>"),
        ChoiceOption("x-api-key", "X-API-Key", "x-api-key: <key>"),
        ChoiceOption("raw", "Raw header value", "Send the key exactly as entered"),
        ChoiceOption("none", "No auth header", "Useful for local/private gateways without auth"),
    ]


def _api_key_env_hint(provider_id: str) -> str:
    return rc.format_env_var_list(rc.get_provider_env_api_key_candidates(provider_id))


def _print_provider_catalog() -> None:
    print("[rewrite-config] Built-in provider templates")
    for group in PROVIDER_GROUPS:
        _print_section(f"{group.label}")
        print(group.hint)
        for method in group.methods:
            print(f"  - {method.label}")
            if method.hint:
                print(f"    {method.hint}")
            if method.backend == "external":
                print(f"    Default base URL: {method.base_url}")
                print(f"    Default model: {method.default_model}")
                print(f"    API key env fallback: {_api_key_env_hint(method.provider_id)}")
                if method.default_extra_body:
                    body_preview = json.dumps(method.default_extra_body, ensure_ascii=True)
                    print(f"    Default extra body: {body_preview}")


def _ensure_external_profile(store: Dict[str, Any], name: str) -> Dict[str, Any]:
    profile = rc.get_profile(store, name)
    if profile is None:
        raise KeyError(f"Unknown profile: {name}")
    if str(profile.get("backend") or "").lower() != "external":
        raise ValueError(f"Profile '{name}' is not an external profile.")
    return profile


def _print_status(store: Dict[str, Any]) -> None:
    active = store.get("active_profile") or "(none)"
    print(f"Config:  {rc.CONFIG_PATH}")
    print(f"Active:  {active}")
    profile = rc.get_profile(store, str(active))
    if profile is not None:
        print(f"Detail:  {rc.profile_summary(str(active), profile)}")
        if str(profile.get("backend") or "").lower() == "external":
            provider_id = rc.resolve_provider_id(profile)
            api_key_info = rc.resolve_api_key(profile, provider_id=provider_id)
            env_hint = rc.format_env_var_list(api_key_info.get("candidates") or ["REWRITE_API_KEY"])
            resolved_source = str(api_key_info.get("source") or "(not set)")
            print(f"Provider: {provider_id or '(unspecified)'}")
            print(f"API key:  {resolved_source}")
            print(f"Env vars: {env_hint}")
    print("Tip:     run 'wizard' for guided setup")


def _print_list(store: Dict[str, Any]) -> None:
    active = store.get("active_profile")
    for name, profile in sorted((store.get("profiles") or {}).items()):
        prefix = "*" if name == active else " "
        if isinstance(profile, dict):
            print(f"{prefix} {rc.profile_summary(name, profile)}")


def _show_profile(store: Dict[str, Any], name: str) -> None:
    profile = rc.get_profile(store, name)
    if profile is None:
        raise KeyError(f"Unknown profile: {name}")
    print(json.dumps(profile, indent=2, ensure_ascii=True))


def _profile_name_default(method: ProviderMethod) -> str:
    return method.value.replace("-api-key", "").replace("-compatible", "")


def _print_method_notes(method: ProviderMethod) -> None:
    if not method.notes and method.backend != "external":
        return
    _print_section(f"{method.label}")
    for note in method.notes:
        print(f"- {note}")
    if method.backend == "external":
        print(f"- API style: {method.api_kind}")
        print(f"- Default base URL: {method.base_url}")
        print(f"- Default model: {method.default_model}")
        print(f"- API key env fallback: {_api_key_env_hint(method.provider_id)}")
        if method.default_extra_body:
            print(f"- Default extra request body: {json.dumps(method.default_extra_body, ensure_ascii=True)}")


def _build_request_headers_for_profile(profile: Dict[str, Any]) -> Dict[str, str]:
    """Build auth + content-type headers from a profile dict (for verification use only)."""
    api_kind = str(profile.get("api_kind") or "openai-chat-completions").strip().lower()
    provider_id = rc.resolve_provider_id(profile)
    api_key_info = rc.resolve_api_key(profile, provider_id=provider_id)
    api_key = str(api_key_info.get("value") or "")
    auth_mode = str(profile.get("auth_mode") or "bearer").strip().lower()
    auth_header_name = str(profile.get("auth_header") or "").strip()
    extra_headers: Dict[str, str] = {}
    raw_extra = profile.get("extra_headers")
    if isinstance(raw_extra, dict):
        extra_headers = {str(k): str(v) for k, v in raw_extra.items()}

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    headers.update(extra_headers)

    # Anthropic-compatible endpoints always need anthropic-version, but auth
    # follows the profile's auth_mode (Anthropic uses x-api-key; MiniMax uses bearer).
    if api_kind in {"anthropic-messages", "anthropic_messages"}:
        headers.setdefault("anthropic-version", "2023-06-01")

    if auth_mode == "none" or not api_key:
        return headers
    if auth_mode == "bearer":
        headers[auth_header_name or "Authorization"] = f"Bearer {api_key}"
    elif auth_mode == "x-api-key":
        headers[auth_header_name or "x-api-key"] = api_key
    elif auth_mode == "raw":
        headers[auth_header_name or "Authorization"] = api_key
    return headers


def _build_request_body_for_profile(
    profile: Dict[str, Any],
    *,
    prompt: str,
    max_tokens: int,
) -> Dict[str, Any]:
    api_kind = str(profile.get("api_kind") or "openai-chat-completions").strip().lower()
    temperature = float(profile.get("temperature") or 0.7)
    model = str(profile.get("model") or "")
    if api_kind in {"anthropic-messages", "anthropic_messages"}:
        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
    else:
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
    raw_extra_body = profile.get("extra_body")
    extra_body = raw_extra_body if isinstance(raw_extra_body, dict) else {}
    return _merge_json_objects(body, extra_body)


def _verify_endpoint_quick(profile: Dict[str, Any]) -> tuple[bool, str]:
    """
    Send a minimal 1-token request to verify the endpoint and key.

    Returns (ok, message).  Does NOT raise — all errors are caught and returned
    as a (False, description) tuple.
    """
    try:
        import requests as _req
    except ImportError:
        return False, "requests package not installed — skipping verification"

    api_kind = str(profile.get("api_kind") or "openai-chat-completions").strip().lower()
    base_url = str(profile.get("base_url") or "").rstrip("/")
    model = str(profile.get("model") or "")

    if not base_url:
        return False, "Base URL is empty"

    headers = _build_request_headers_for_profile(profile)
    body = _build_request_body_for_profile(profile, prompt="Hi", max_tokens=1)

    if api_kind in {"anthropic-messages", "anthropic_messages"}:
        # Base URL conventions:
        #   "https://api.anthropic.com/v1"      → append /messages
        #   "https://api.minimax.io/anthropic"  → append /v1/messages (no /v1 in base)
        if base_url.endswith("/messages"):
            url = base_url
        elif base_url.endswith("/v1"):
            url = base_url + "/messages"
        else:
            url = base_url + "/v1/messages"
    else:
        url = base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"

    try:
        resp = _req.post(url, json=body, headers=headers, timeout=20)
    except Exception as exc:
        return False, f"Connection error: {exc}"

    if resp.ok:
        return True, f"HTTP {resp.status_code} — endpoint is reachable and accepted the key"

    error_detail = ""
    try:
        err_json = resp.json()
        error_detail = (
            (err_json.get("error") or {}).get("message")
            or err_json.get("message")
            or str(err_json)[:120]
        )
    except Exception:
        error_detail = resp.text[:120]

    hint = ""
    if resp.status_code == 401:
        hint = " (invalid API key)"
    elif resp.status_code == 403:
        hint = " (key lacks permission, or wrong endpoint)"
    elif resp.status_code == 404:
        hint = " (base URL or model not found)"
    elif resp.status_code == 422:
        hint = " (request format issue — endpoint may require different parameters)"

    return False, f"HTTP {resp.status_code}{hint}: {error_detail}"


def _apply_advanced_settings(
    profile: Dict[str, Any],
    method: ProviderMethod,
) -> Dict[str, Any]:
    """Prompt for advanced settings and return an updated profile dict."""
    auth_mode = _select_option(
        "Auth mode",
        _auth_mode_options(),
        default_value=str(profile.get("auth_mode") or method.auth_mode),
    )
    default_header = str(profile.get("auth_header") or method.auth_header)
    if not default_header:
        if auth_mode == "x-api-key":
            default_header = "x-api-key"
        elif auth_mode in {"bearer", "raw"}:
            default_header = "Authorization"
    auth_header = _prompt("Auth header name", default_header)
    temperature = _prompt_float(
        "Temperature", float(profile.get("temperature") or method.temperature)
    )
    timeout_s = _prompt_float(
        "Timeout seconds", float(profile.get("timeout_s") or method.timeout_s)
    )
    extra_headers_default = (
        profile.get("extra_headers")
        if isinstance(profile.get("extra_headers"), dict)
        else method.default_extra_headers
    )
    extra_headers = _prompt_json("Extra headers JSON", extra_headers_default)
    extra_body_default = (
        profile.get("extra_body")
        if isinstance(profile.get("extra_body"), dict)
        else method.default_extra_body
    )
    extra_body = _prompt_json_object("Extra request body JSON", extra_body_default)
    return {
        **profile,
        "auth_mode": auth_mode,
        "auth_header": auth_header,
        "temperature": temperature,
        "timeout_s": timeout_s,
        "extra_headers": extra_headers,
        "extra_body": extra_body,
    }


def _configure_external_profile_wizard(
    store: Dict[str, Any],
    name: str,
    method: ProviderMethod,
) -> Dict[str, Any]:
    """
    Streamlined wizard path — designed so the user only needs to provide an API key.

    Flow:
      1. Confirm base URL (pre-filled from template, press Enter to accept)
      2. Model picker (preset list or free-type)
      3. API key (secret prompt; leave blank to rely on env-var fallback)
      4. Save with method defaults for auth/temperature/timeout/headers
      5. Offer: verify now  |  advanced settings
    """
    existing = rc.get_profile(store, name) or {}
    provider_id = rc.normalize_provider_id(existing.get("provider_id") or method.provider_id)
    env_hint = _api_key_env_hint(provider_id)

    # ── 1. Base URL ──────────────────────────────────────────────────────────
    current_url = str(existing.get("base_url") or method.base_url)
    print(f"\n  Endpoint : {current_url}")
    url_input = input("  Change URL? (press Enter to keep, or type a new one): ").strip()
    base_url = url_input or current_url

    # ── 2. Model ─────────────────────────────────────────────────────────────
    existing_model = str(existing.get("model") or method.default_model)
    model = _select_model_value(method, existing_model)

    # ── 3. API key ───────────────────────────────────────────────────────────
    has_stored_key = bool(existing.get("api_key"))
    if has_stored_key:
        key_status = "(already stored in profile — press Enter to keep)"
    elif env_hint:
        key_status = f"(leave blank to fall back to env var: {env_hint})"
    else:
        key_status = "(required)"
    print(f"\n  {key_status}")
    api_key = _prompt(f"  {method.prompt_api_key_label}", secret=True)
    if not api_key and has_stored_key:
        api_key = str(existing["api_key"])

    # ── Build profile (use method defaults for advanced fields) ───────────────
    profile: Dict[str, Any] = {
        "backend": "external",
        "provider_id": provider_id,
        "api_kind": method.api_kind,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "auth_mode": str(existing.get("auth_mode") or method.auth_mode),
        "auth_header": str(existing.get("auth_header") or method.auth_header),
        "timeout_s": float(existing.get("timeout_s") or method.timeout_s),
        "temperature": float(existing.get("temperature") or method.temperature),
        "disable_thinking": bool(existing.get("disable_thinking", True)),
        "extra_headers": (
            existing.get("extra_headers")
            if isinstance(existing.get("extra_headers"), dict)
            else dict(method.default_extra_headers)
        ),
        "extra_body": (
            existing.get("extra_body")
            if isinstance(existing.get("extra_body"), dict)
            else dict(method.default_extra_body)
        ),
    }

    # ── 4. Advanced settings ─────────────────────────────────────────────────
    if _confirm("\n  Configure advanced settings (auth mode, headers, temperature)?", default=False):
        profile = _apply_advanced_settings(profile, method)

    rc.upsert_profile(store, name, profile)
    print(f"\n  [rewrite-config] Profile '{name}' saved.")

    # ── 5. Verify ────────────────────────────────────────────────────────────
    if _confirm("  Verify now? (sends a 1-token test request to the endpoint)", default=True):
        print("  Testing...", end=" ", flush=True)
        ok, msg = _verify_endpoint_quick(profile)
        status = "OK" if ok else "FAILED"
        print(f"{status} — {msg}")
        if not ok and _confirm("  Edit settings to fix?", default=True):
            retry_choice = _select_option(
                "What to change",
                [
                    ChoiceOption("url", "Base URL", ""),
                    ChoiceOption("key", "API key", ""),
                    ChoiceOption("advanced", "Advanced settings", "auth mode / headers"),
                ],
            )
            if retry_choice == "url":
                new_url = _prompt("  New base URL", base_url)
                profile["base_url"] = new_url
            elif retry_choice == "key":
                new_key = _prompt(f"  {method.prompt_api_key_label}", secret=True)
                if new_key:
                    profile["api_key"] = new_key
            elif retry_choice == "advanced":
                profile = _apply_advanced_settings(profile, method)
            rc.upsert_profile(store, name, profile)
            print(f"\n  Re-testing...", end=" ", flush=True)
            ok2, msg2 = _verify_endpoint_quick(profile)
            print(f"{'OK' if ok2 else 'FAILED'} — {msg2}")

    return profile


def _configure_local_profile(store: Dict[str, Any], name: str, method: ProviderMethod) -> None:
    existing = rc.get_profile(store, name) or {}
    model_path = _prompt(
        "Local model path",
        str(existing.get("model_path") or "./models/Qwen3-VL-8B-Instruct"),
    )
    label = _prompt("Label", str(existing.get("label") or method.label))
    rc.upsert_profile(
        store,
        name,
        {
            "backend": "local",
            "provider_id": method.provider_id,
            "model_path": model_path,
            "label": label,
            "disable_thinking": bool(existing.get("disable_thinking", True)),
        },
    )
    print(f"[rewrite-config] Saved local profile '{name}'.")


def _configure_external_profile(store: Dict[str, Any], name: str, method: ProviderMethod) -> None:
    existing = rc.get_profile(store, name) or {}
    provider_id = rc.normalize_provider_id(existing.get("provider_id") or method.provider_id)
    env_hint = _api_key_env_hint(provider_id)
    existing_model = str(existing.get("model") or method.default_model)

    base_url = _prompt(
        method.prompt_base_url_label,
        str(existing.get("base_url") or method.base_url),
    )
    model = _select_model_value(method, existing_model)
    auth_mode = _select_option(
        "Auth mode",
        _auth_mode_options(),
        default_value=str(existing.get("auth_mode") or method.auth_mode),
    )
    default_header = str(existing.get("auth_header") or method.auth_header)
    if not default_header:
        if auth_mode == "x-api-key":
            default_header = "x-api-key"
        elif auth_mode in {"bearer", "raw"}:
            default_header = "Authorization"
    auth_header = _prompt("Auth header", default_header)
    api_key = _prompt(
        f"{method.prompt_api_key_label} (leave blank to rely on env fallback: {env_hint})",
        str(existing.get("api_key") or ""),
        secret=True,
    )
    temperature = _prompt_float(
        "Temperature",
        float(existing.get("temperature") or method.temperature),
    )
    timeout_s = _prompt_float(
        "Timeout seconds",
        float(existing.get("timeout_s") or method.timeout_s),
    )
    extra_headers = _prompt_json(
        "Extra headers JSON",
        existing.get("extra_headers")
        if isinstance(existing.get("extra_headers"), dict)
        else method.default_extra_headers,
    )
    extra_body = _prompt_json_object(
        "Extra request body JSON",
        existing.get("extra_body")
        if isinstance(existing.get("extra_body"), dict)
        else method.default_extra_body,
    )

    profile = {
        "backend": "external",
        "provider_id": provider_id,
        "api_kind": method.api_kind,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "auth_mode": auth_mode,
        "auth_header": auth_header,
        "timeout_s": timeout_s,
        "temperature": temperature,
        "disable_thinking": bool(existing.get("disable_thinking", True)),
        "extra_headers": extra_headers,
        "extra_body": extra_body,
    }
    rc.upsert_profile(store, name, profile)
    print(f"[rewrite-config] Saved external profile '{name}'.")


def _configure_from_method(
    store: Dict[str, Any],
    method: ProviderMethod,
    *,
    wizard_mode: bool = False,
) -> str:
    _print_method_notes(method)
    default_name = _profile_name_default(method)
    name = _prompt("Profile name", default_name)
    if method.backend == "local":
        _configure_local_profile(store, name, method)
    elif wizard_mode:
        _configure_external_profile_wizard(store, name, method)
    else:
        _configure_external_profile(store, name, method)
    return name


def _test_profile(store: Dict[str, Any], name: str | None) -> None:
    selected = name or str(store.get("active_profile") or rc.DEFAULT_LOCAL_PROFILE)
    loaded = rc.load_selected_profile(selected)
    if loaded is None:
        raise KeyError(f"Unknown profile: {selected}")
    os.environ["REWRITE_PROFILE"] = str(loaded["name"])
    mr = importlib.import_module("scripts.models_runtime")
    mr = importlib.reload(mr)
    result = mr.run_rewrite_candidate("This sentence should be rewritten naturally.", "balanced")
    print(f"[rewrite-config] Test response from '{selected}':")
    print(result)


def run_wizard(store: Optional[Dict[str, Any]] = None) -> int:
    store = store or rc.load_store()
    active = store.get("active_profile") or rc.DEFAULT_LOCAL_PROFILE
    print()
    print("=== Rewrite Provider Setup ===")
    print("Pick a provider and enter your API key.")
    print("All other settings are pre-filled — just press Enter to accept them.")
    print(f"Current active profile: {active}")
    print()
    active_profile = rc.get_profile(store, str(active))
    default_group = "openrouter"
    if isinstance(active_profile, dict):
        if str(active_profile.get("backend") or "").lower() == "local":
            default_group = "local"
        else:
            provider_id = str(active_profile.get("provider_id") or "").strip().lower()
            for group in PROVIDER_GROUPS:
                if provider_id and any(method.provider_id == provider_id for method in group.methods):
                    default_group = group.value
                    break

    while True:
        group_choice = _select_option(
            "Provider family",
            [ChoiceOption(group.value, group.label, group.hint) for group in PROVIDER_GROUPS],
            default_value=default_group if any(group.value == default_group for group in PROVIDER_GROUPS) else None,
        )
        group = next(group for group in PROVIDER_GROUPS if group.value == group_choice)
        if len(group.methods) == 1:
            method = group.methods[0]
        else:
            method_choice = _select_option(
                f"{group.label} preset",
                [ChoiceOption(method.value, method.label, method.hint) for method in group.methods],
                allow_back=True,
            )
            if method_choice == BACK_VALUE:
                continue
            method = METHOD_BY_ID[method_choice]

        profile_name = _configure_from_method(store, method, wizard_mode=True)
        if _confirm("\nSet this as the active profile?", default=True):
            rc.set_active_profile(store, profile_name)
            print(f"[rewrite-config] Active profile -> {profile_name}")
        rc.save_store(store)
        print(f"[rewrite-config] Saved {rc.CONFIG_PATH}")

        if _confirm("Run a quick test rewrite now?", default=False):
            try:
                _test_profile(store, profile_name)
            except Exception as exc:
                print(f"[rewrite-config] Test failed: {exc}")

        if not _confirm("Configure another provider?", default=False):
            break

    return 0


def run_repl() -> int:
    store = rc.load_store()
    print("[rewrite-config] Rewrite provider shell")
    print(f"[rewrite-config] Config file: {rc.CONFIG_PATH}")
    print("Type 'wizard' for guided setup or 'help' for commands.")
    while True:
        try:
            line = input("rewrite> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        try:
            keep_going = _handle_command(store, line)
        except IndexError:
            print("[rewrite-config] Missing argument.")
            continue
        except Exception as exc:
            print(f"[rewrite-config] {exc}")
            continue

        if not keep_going:
            break

    rc.save_store(store)
    print(f"[rewrite-config] Saved {rc.CONFIG_PATH}")
    return 0


def _add_local(store: Dict[str, Any], name: str) -> None:
    _configure_local_profile(store, name, LOCAL_REWRITE_METHOD)


def _add_api(store: Dict[str, Any], name: str) -> None:
    print("[rewrite-config] Manual external profile editor")
    print("[rewrite-config] For a guided flow, run 'wizard'.")
    existing = rc.get_profile(store, name) or {}
    provider_id = _prompt("Provider id", str(existing.get("provider_id") or "custom"))
    api_kind = _select_option(
        "API kind",
        [
            ChoiceOption("openai-chat-completions", "OpenAI-compatible", "POST /chat/completions"),
            ChoiceOption("anthropic-messages", "Anthropic-compatible", "POST /messages"),
        ],
        default_value=str(existing.get("api_kind") or "openai-chat-completions"),
    )
    method = CUSTOM_OPENAI_METHOD if api_kind == "openai-chat-completions" else CUSTOM_ANTHROPIC_METHOD
    method = ProviderMethod(
        **{
            **method.__dict__,
            "provider_id": provider_id,
        }
    )
    _configure_external_profile(store, name, method)


def _update_auth(store: Dict[str, Any], name: str) -> None:
    profile = _ensure_external_profile(store, name)
    provider_id = rc.resolve_provider_id(profile)
    env_hint = _api_key_env_hint(provider_id)
    auth_mode = _select_option(
        "Auth mode",
        _auth_mode_options(),
        default_value=str(profile.get("auth_mode") or "bearer"),
    )
    default_header = str(profile.get("auth_header") or "")
    if not default_header:
        if auth_mode == "x-api-key":
            default_header = "x-api-key"
        elif auth_mode in {"bearer", "raw"}:
            default_header = "Authorization"
    auth_header = _prompt("Auth header", default_header)
    api_key = _prompt(
        f"API key (leave blank to rely on env fallback: {env_hint})",
        str(profile.get("api_key") or ""),
        secret=True,
    )
    profile.update(
        {
            "auth_mode": auth_mode,
            "auth_header": auth_header,
            "api_key": api_key,
        }
    )
    print(f"[rewrite-config] Updated auth for '{name}'.")


def _update_headers(store: Dict[str, Any], name: str) -> None:
    profile = rc.get_profile(store, name)
    if profile is None:
        raise KeyError(f"Unknown profile: {name}")
    profile["extra_headers"] = _prompt_json(
        "Extra headers JSON",
        profile.get("extra_headers") if isinstance(profile.get("extra_headers"), dict) else {},
    )
    print(f"[rewrite-config] Updated extra headers for '{name}'.")


def _update_body(store: Dict[str, Any], name: str) -> None:
    profile = rc.get_profile(store, name)
    if profile is None:
        raise KeyError(f"Unknown profile: {name}")
    profile["extra_body"] = _prompt_json_object(
        "Extra request body JSON",
        profile.get("extra_body") if isinstance(profile.get("extra_body"), dict) else {},
    )
    print(f"[rewrite-config] Updated extra request body for '{name}'.")


def _handle_command(store: Dict[str, Any], line: str) -> bool:
    parts = shlex.split(line)
    if not parts:
        return True

    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in {"exit", "quit"}:
        return False
    if cmd == "help":
        print(HELP_TEXT)
        return True
    if cmd == "verify":
        name = args[0] if args else str(store.get("active_profile") or rc.DEFAULT_LOCAL_PROFILE)
        profile = rc.get_profile(store, name)
        if profile is None:
            raise KeyError(f"Unknown profile: {name}")
        if str(profile.get("backend") or "").lower() != "external":
            print(f"[rewrite-config] '{name}' is a local profile — nothing to verify.")
            return True
        print(f"[rewrite-config] Verifying '{name}'...", end=" ", flush=True)
        ok, msg = _verify_endpoint_quick(profile)
        print(f"{'OK' if ok else 'FAILED'} — {msg}")
        return True
    if cmd == "wizard":
        run_wizard(store)
        return True
    if cmd in {"providers", "templates"}:
        _print_provider_catalog()
        return True
    if cmd == "status":
        _print_status(store)
        return True
    if cmd == "list":
        _print_list(store)
        return True
    if cmd == "show":
        _show_profile(store, args[0])
        return True
    if cmd == "add-local":
        _add_local(store, args[0])
        return True
    if cmd == "add-api":
        _add_api(store, args[0])
        return True
    if cmd == "auth":
        _update_auth(store, args[0])
        return True
    if cmd == "headers":
        _update_headers(store, args[0])
        return True
    if cmd == "body":
        _update_body(store, args[0])
        return True
    if cmd == "use":
        rc.set_active_profile(store, args[0])
        print(f"[rewrite-config] Active profile -> {args[0]}")
        return True
    if cmd == "delete":
        rc.delete_profile(store, args[0])
        print(f"[rewrite-config] Deleted profile '{args[0]}'.")
        return True
    if cmd == "test":
        _test_profile(store, args[0] if args else None)
        return True
    if cmd == "save":
        rc.save_store(store)
        print(f"[rewrite-config] Saved {rc.CONFIG_PATH}")
        return True

    raise ValueError(f"Unknown command: {cmd}")


def main(argv: list[str] | None = None) -> int:
    try:
        parser = argparse.ArgumentParser(description="Manage local rewrite-provider profiles.")
        parser.add_argument(
            "command",
            nargs="?",
            default="wizard",
            choices=["wizard", "repl", "list", "status", "providers"],
        )
        args = parser.parse_args(argv)

        store = rc.load_store()
        if args.command == "wizard":
            return run_wizard(store)
        if args.command == "repl":
            return run_repl()
        if args.command == "list":
            _print_list(store)
            return 0
        if args.command == "status":
            _print_status(store)
            return 0
        if args.command == "providers":
            _print_provider_catalog()
            return 0
        return 0
    except KeyboardInterrupt:
        print()
        print("[rewrite-config] Cancelled.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
