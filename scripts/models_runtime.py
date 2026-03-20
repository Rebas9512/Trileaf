"""
Model inference layer for the writing optimizer pipeline.

Models managed here
-------------------
- desklib/ai-text-detector-v1.01    → AI-content classification (SequenceClassification)
- rewrite provider                  → rewrite ensemble
                                       (local model or external API)
- paraphrase-mpnet-base-v2          → semantic similarity (SentenceTransformer)

All models are lazily loaded and cached in module-level dicts.
"""

from __future__ import annotations

import contextlib
import json
import io
import logging
import os
import platform
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import threading
import torch
import torch.nn as nn
# NOTE: all heavy transformers imports are deferred to the first model-load call
# so that `import scripts.models_runtime` is fast and does NOT pull in the
# transformers → sklearn → scipy chain (which compiles .pyc files on first run
# and can stall startup by 10-60 s on a cold filesystem).

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as _st_util
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore[assignment]

from scripts import app_config as _app_config
from scripts import rewrite_config as _rewrite_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def _normalize_rewrite_backend(raw: str) -> str:
    value = (raw or "local").strip().lower()
    if value in {"openai_api", "api", "remote"}:
        return "external"
    if value not in {"local", "external"}:
        return "local"
    return value


def _parse_json_dict(raw: str) -> Dict[str, str]:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass
    return {}


def _parse_json_object(raw: str) -> Dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {}


def _merge_case_insensitive_headers(
    base: Dict[str, str],
    override: Dict[str, str],
) -> Dict[str, str]:
    merged = dict(base)
    for name, value in override.items():
        target = name.lower()
        for existing in list(merged.keys()):
            if existing.lower() == target:
                del merged[existing]
        merged[name] = value
    return merged


def _merge_json_objects(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_json_objects(merged[key], value)
        else:
            merged[key] = value
    return merged


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if platform.system() == "Darwin" and getattr(torch.backends, "mps", None):
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


DEVICE = select_device()

# ── Rewrite provider config — resolved from os.environ (populated by
#    rewrite_config.resolve_credentials() before this module is imported).
# ─────────────────────────────────────────────────────────────────────────────
DESKLIB_MODEL_PATH  = str(_app_config.resolve_model_path("desklib"))
MPNET_MODEL_PATH    = str(_app_config.resolve_model_path("mpnet"))
REWRITE_BACKEND     = _normalize_rewrite_backend(
    _rewrite_config.first_defined(
        os.getenv("REWRITE_BACKEND"),
        _rewrite_config.legacy_env_first("backend"),
        "local",
    ) or "local"
)
REWRITE_MODEL_PATH  = _resolve(
    _rewrite_config.first_defined(
        os.getenv("REWRITE_MODEL_PATH"),
        _rewrite_config.legacy_env_first("model_path"),
        "./models/Qwen3-VL-8B-Instruct",
    ) or "./models/Qwen3-VL-8B-Instruct"
)
REWRITE_API_KIND    = (os.getenv("REWRITE_API_KIND") or "openai-chat-completions").strip().lower()
REWRITE_BASE_URL    = (
    os.getenv("REWRITE_BASE_URL")
    or _rewrite_config.legacy_env_first("base_url")
    or ""
)
REWRITE_MODEL       = (
    os.getenv("REWRITE_MODEL")
    or _rewrite_config.legacy_env_first("model")
    or "Qwen3-VL-8B-Instruct"
)
REWRITE_PROVIDER_ID = _rewrite_config.normalize_provider_id(os.getenv("REWRITE_PROVIDER_ID", ""))
REWRITE_API_KEY     = os.getenv("REWRITE_API_KEY", "")
_REWRITE_API_KEY_CANDIDATES = _rewrite_config.get_provider_env_api_key_candidates(REWRITE_PROVIDER_ID)
REWRITE_AUTH_MODE   = (os.getenv("REWRITE_AUTH_MODE") or "bearer").strip().lower()
REWRITE_AUTH_HEADER = os.getenv("REWRITE_AUTH_HEADER", "")
REWRITE_EXTRA_HEADERS = _parse_json_dict(os.getenv("REWRITE_EXTRA_HEADERS_JSON", ""))
REWRITE_EXTRA_BODY    = _parse_json_object(os.getenv("REWRITE_EXTRA_BODY_JSON", ""))
REWRITE_TIMEOUT_S   = float(os.getenv("REWRITE_TIMEOUT_S") or "120")
REWRITE_TEMPERATURE = float(os.getenv("REWRITE_TEMPERATURE") or "0.7")
# Disable model thinking/reasoning mode by default to minimise latency and
# token cost on short-text rewrite tasks.  Set REWRITE_DISABLE_THINKING=false
# in .env (or as an env var) to re-enable.
REWRITE_DISABLE_THINKING: bool = (
    (os.getenv("REWRITE_DISABLE_THINKING") or "true").lower() not in {"false", "0", "no", "off"}
)
# Set REWRITE_DEBUG=1 to print raw model output and extraction trace to stderr.
REWRITE_DEBUG: bool = os.getenv("REWRITE_DEBUG", "").lower() not in {"", "0", "false", "no", "off"}
if REWRITE_DEBUG:
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    _log.setLevel(logging.DEBUG)
else:
    # Ensure WARNING+ from this module is always visible even if root logger is not configured.
    if not logging.root.handlers:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

_DESKLIB_CACHE: Dict[str, Any] = {}
_MPNET_CACHE:   Dict[str, Any] = {}
_REWRITE_CACHE: Dict[str, Any] = {}

_DESKLIB_LOCK = threading.Lock()
_MPNET_LOCK   = threading.Lock()
_REWRITE_LOCK = threading.Lock()

_META_EXPLANATION_RE = re.compile(
    r"\b("
    r"the original text|original sentence|paraphrased version|rewritten version|"
    r"this version|this rewrite|here(?:'s| is) the rewritten|"
    r"explanation|note that|I rewrote|I changed|I kept|while preserving"
    r")\b",
    flags=re.IGNORECASE,
)


class RewriteResponseError(RuntimeError):
    """The provider response was reachable but did not contain usable rewrite text."""


# ─── Text cleaning helpers ─────────────────────────────────────────────────────


def _strip_output_markers(text: str) -> str:
    text = text.strip()
    for marker in ("Rewritten:", "Paraphrased text:", "Corrected rewrite:", "Rewritten text:"):
        if marker in text:
            text = text.split(marker, 1)[1].strip()
    return text


def _clean_generated_rewrite(text: str, source_text: str) -> str:
    """Remove prompt echoes / explanatory tails from model-generated rewrites."""
    cleaned = _strip_output_markers(text)
    if not cleaned:
        return source_text.strip()

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", cleaned) if p.strip()]
    if paragraphs:
        first = paragraphs[0]
        if len(paragraphs) > 1 and any(_META_EXPLANATION_RE.search(p) for p in paragraphs[1:]):
            cleaned = first
        else:
            cleaned = "\n\n".join(paragraphs)

    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if lines:
        kept: List[str] = []
        for line in lines:
            if kept and _META_EXPLANATION_RE.search(line):
                break
            kept.append(line)
        cleaned = " ".join(kept).strip()

    source_len = len(source_text.strip())
    source_sentence_count = max(
        1,
        len(re.findall(r"[.!?。！？]+", source_text.strip())) or 1,
    )
    sentence_parts = re.split(r"(?<=[.!?。！？])\s+", cleaned)
    sentence_parts = [s.strip() for s in sentence_parts if s.strip()]

    if sentence_parts:
        if source_len <= 20 and len(sentence_parts) > 1:
            cleaned = sentence_parts[0]
        else:
            kept2: List[str] = []
            for sent in sentence_parts:
                if kept2 and _META_EXPLANATION_RE.search(sent):
                    break
                kept2.append(sent)
            max_sentences = max(1, source_sentence_count + 1)
            cleaned = " ".join(kept2[:max_sentences]).strip()

    _src = source_text.strip()
    _QUOTE_CHARS = "\"'\u201c\u201d\u2018\u2019"
    # Don't strip a quote char from the start/end of the rewrite if the source
    # text itself starts/ends with that character (e.g. dialogue that opens or
    # closes with a quotation mark).
    _lstrip = " " + "".join(c for c in _QUOTE_CHARS if not _src.startswith(c))
    _rstrip = " " + "".join(c for c in _QUOTE_CHARS if not _src.endswith(c))
    cleaned = cleaned.lstrip(_lstrip).rstrip(_rstrip)
    cleaned = _enforce_length_guard(cleaned, source_text)
    return cleaned or source_text.strip()


def _truncate_to_word_limit(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    truncated = " ".join(words[:max_words]).strip()
    sentence_cut = re.split(r"(?<=[.!?。！？])\s+", truncated)
    sentence_cut = [part.strip() for part in sentence_cut if part.strip()]
    if sentence_cut:
        candidate = " ".join(sentence_cut).strip()
        if candidate:
            return candidate
    return truncated


def _enforce_length_guard(text: str, source_text: str) -> str:
    source    = source_text.strip()
    candidate = text.strip()
    if not source or not candidate:
        return candidate or source

    src_words      = source.split()
    cand_words     = candidate.split()
    src_word_count  = len(src_words)
    cand_word_count = len(cand_words)
    src_char_count  = len(source)
    cand_char_count = len(candidate)

    if src_word_count <= 3:
        max_words = max(src_word_count + 3, src_word_count * 2)
        max_chars = max(src_char_count + 18, int(src_char_count * 2.2))
    else:
        max_words = max(src_word_count + 8, int(src_word_count * 1.35))
        max_chars = max(src_char_count + 40, int(src_char_count * 1.45))

    if cand_word_count > max_words:
        candidate       = _truncate_to_word_limit(candidate, max_words)
        cand_char_count = len(candidate)

    if cand_char_count > max_chars:
        candidate  = candidate[:max_chars].rstrip(" ,;:-")
        last_break = max(
            candidate.rfind("."), candidate.rfind("!"), candidate.rfind("?"),
            candidate.rfind("\u3002"), candidate.rfind("\uff01"), candidate.rfind("\uff1f"),
        )
        if last_break >= max(10, max_chars // 2):
            candidate = candidate[: last_break + 1].strip()

    return candidate.strip() or source


# ─── Desklib ──────────────────────────────────────────────────────────────────
# Custom architecture: DebertaV2 backbone + single-logit classifier head.


class _DesklibDetector(nn.Module):
    def __init__(self, config: DebertaV2Config) -> None:
        super().__init__()
        from transformers import DebertaV2Model  # lazy — only when model is loaded
        self.model      = DebertaV2Model(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids:      "torch.Tensor",
        attention_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(cls)


def _load_desklib():
    with _DESKLIB_LOCK:
        if DESKLIB_MODEL_PATH not in _DESKLIB_CACHE:
            from safetensors.torch import load_file as _load_safetensors
            from transformers import AutoTokenizer, DebertaV2Config  # lazy

            tok    = AutoTokenizer.from_pretrained(DESKLIB_MODEL_PATH)
            config = DebertaV2Config.from_pretrained(DESKLIB_MODEL_PATH)
            mdl    = _DesklibDetector(config)

            weights_path = os.path.join(DESKLIB_MODEL_PATH, "model.safetensors")
            state_dict   = _load_safetensors(weights_path, device="cpu")
            mdl.load_state_dict(state_dict, strict=True)

            mdl.to(DEVICE).eval()
            _DESKLIB_CACHE[DESKLIB_MODEL_PATH] = (tok, mdl)

        return _DESKLIB_CACHE[DESKLIB_MODEL_PATH]


@torch.no_grad()
def run_desklib(text: str) -> float:
    """Return AI probability in [0, 1]. Higher → more AI-like."""
    tok, mdl = _load_desklib()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    logit = mdl(enc["input_ids"], enc.get("attention_mask"))
    return float(torch.sigmoid(logit[0, 0]).item())


# ─── Rewrite ensemble ─────────────────────────────────────────────────────────

REWRITE_STYLES = ("conservative", "balanced", "aggressive")

# Per-style inference temperatures.  Conservative uses a lower temperature for
# focused, predictable edits; aggressive uses a higher temperature to unlock
# more varied restructuring.  Override via env vars if needed.
_STYLE_TEMPERATURES: Dict[str, float] = {
    "conservative": float(os.getenv("REWRITE_TEMP_CONSERVATIVE", "0.45")),
    "balanced":     float(os.getenv("REWRITE_TEMP_BALANCED",     "0.70")),
    "aggressive":   float(os.getenv("REWRITE_TEMP_AGGRESSIVE",   "0.92")),
}

_REWRITE_PROMPTS: Dict[str, str] = {
    # ── Conservative ──────────────────────────────────────────────────────────
    # Scope: word / phrase substitutions only.  Sentence structure is frozen.
    "conservative": (
        "You are a copy editor making minimal, surface-level changes only.\n\n"
        "TASK: Lightly polish the text so it sounds slightly more natural and less formulaic.\n\n"
        "STRICT RULES — violating any rule is a failure:\n"
        "1. Change ONLY individual word choices and small phrases. "
        "Do NOT alter sentence structure.\n"
        "2. Do NOT reorder, merge, split, or drop any sentence.\n"
        "3. Do NOT change paragraph order or add/remove transition words.\n"
        "4. Every fact, number, named entity, and logical connective must remain identical.\n"
        "5. Do not add or remove any information whatsoever.\n"
        "6. Output length must stay within ±10 % of the original character count.\n\n"
        "RESPONSE FORMAT — strictly enforced:\n"
        "Output exactly one JSON object. No text before it, no text after it, "
        "no markdown fences, no extra keys.\n"
        "The first character of your response MUST be '{{'. "
        "The last character MUST be '}}'.\n"
        "Required schema: {{\"rewrite\": \"<your lightly edited text>\"}}\n"
        "If your rewritten text contains double-quote characters, escape each one "
        "with a backslash inside the JSON value (write \\\" for each \").\n\n"
        "Text:\n{text}\n"
    ),
    # ── Balanced ──────────────────────────────────────────────────────────────
    # Scope: moderate structural changes — clause reordering, sentence merging /
    # splitting, varied openings, anti-AI phrasing, burstiness injection.
    "balanced": (
        "You are a sharp blogger rewriting a passage for a general audience.\n\n"
        "TASK: Rewrite so it reads like something a knowledgeable person actually "
        "wrote — not a polished AI draft. Moderate structural changes are welcome.\n\n"
        "STYLE GUIDELINES:\n"
        "• Vary sentence length deliberately: drop in a short punchy sentence "
        "every few lines, then follow with a longer flowing one.\n"
        "• Contractions are fine where they sound natural (it's, don't, can't, we're).\n"
        "• Replace or rephrase any of these on sight — they signal AI-generated text:\n"
        "  \"moreover\", \"furthermore\", \"it is worth noting\", \"it is important to\",\n"
        "  \"delve into\", \"nuanced\", \"multifaceted\", \"pivotal\", \"in today's world\",\n"
        "  \"navigate\", \"comprehensive\", \"notably\", \"in conclusion\".\n"
        "• Rephrase clause openings so consecutive sentences don't start the same way.\n"
        "• Don't over-explain — if a point is clear from context, move on.\n\n"
        "HARD RULES:\n"
        "• Keep all facts, numbers, named entities, and core claims exactly as-is.\n"
        "• Do not add or remove information.\n"
        "• Do not reorder the sentences themselves or change paragraph structure.\n"
        "• Keep overall length similar to the original.\n\n"
        "RESPONSE FORMAT — strictly enforced:\n"
        "Output exactly one JSON object. No text before it, no text after it, "
        "no markdown fences, no extra keys.\n"
        "The first character of your response MUST be '{{'. "
        "The last character MUST be '}}'.\n"
        "Required schema: {{\"rewrite\": \"<your rewritten text>\"}}\n"
        "If your rewritten text contains double-quote characters, escape each one "
        "with a backslash inside the JSON value (write \\\" for each \").\n\n"
        "Text:\n{text}\n"
    ),
    # ── Aggressive ────────────────────────────────────────────────────────────
    # Scope: deep restructuring — conversational persona, full colloquial register,
    # free sentence reordering, burstiness, anti-AI blacklist.
    "aggressive": (
        "You are writing this as if dashing off a well-informed message to a friend "
        "— conversational, direct, and confident. Not polished prose. Not an AI draft.\n\n"
        "TASK: Substantially restructure and rephrase so every stylistic choice feels "
        "like a real person's instinct, not a formula.\n\n"
        "WHAT TO DO:\n"
        "• Use contractions freely (it's, don't, can't, we're, that's, you'd).\n"
        "• Start sentences with \"And\", \"But\", \"So\", or \"Look\" when it sounds natural.\n"
        "• Mix very short sentences (5–8 words) with longer flowing ones — vary the rhythm.\n"
        "• Add an offhand parenthetical aside or em-dash remark where it fits naturally "
        "(e.g. \"— which is a bit ironic\", \"not that it matters much\").\n"
        "• One rhetorical question is fine if it fits.\n"
        "• Light hedging is welcome: \"to be fair\", \"honestly\", \"sort of\", \"I'd say\" "
        "— used as natural voice, not to hedge every claim.\n"
        "• Freely reorder clauses and sentences within the passage for stronger flow.\n"
        "• Replace formulaic transitions with organic connectives, or drop them entirely "
        "when the flow is clear without them.\n"
        "• Replace or drop any of these on sight — they immediately signal AI text:\n"
        "  \"moreover\", \"furthermore\", \"it is worth noting\", \"it is important to\",\n"
        "  \"delve into\", \"nuanced\", \"multifaceted\", \"pivotal\", \"in today's world\",\n"
        "  \"navigate\", \"comprehensive\", \"notably\", \"in conclusion\", \"it's crucial\".\n\n"
        "HARD LIMITS — never violate these:\n"
        "• Every fact, number, named entity, and core claim stays exactly as-is.\n"
        "• Do not invent or omit content. Keep the rewrite within ±30 % of the "
        "original length.\n\n"
        "RESPONSE FORMAT — strictly enforced:\n"
        "Output exactly one JSON object. No text before it, no text after it, "
        "no markdown fences, no extra keys.\n"
        "The first character of your response MUST be '{{'. "
        "The last character MUST be '}}'.\n"
        "Required schema: {{\"rewrite\": \"<your rewritten text>\"}}\n"
        "If your rewritten text contains double-quote characters, escape each one "
        "with a backslash inside the JSON value (write \\\" for each \").\n\n"
        "Text:\n{text}\n"
    ),
}


def run_rewrite_candidate(text: str, style: str = "balanced") -> str:
    """
    Generate a single rewrite candidate with the specified aggressiveness style.

    style options:
      "conservative" → word/phrase substitutions only; sentence structure frozen
      "balanced"     → moderate structural changes; varied rhythm and openings
      "aggressive"   → deep restructuring; contractions, reordering, voice mixing

    Raises RewriteResponseError if the model produces no output or returns text
    identical to the source (silent fallback detection).
    """
    template    = _REWRITE_PROMPTS.get(style, _REWRITE_PROMPTS["balanced"])
    temperature = _STYLE_TEMPERATURES.get(style, REWRITE_TEMPERATURE)
    # Escape any bare { or } in the user text so Python's .format() doesn't
    # misinterpret them as placeholder syntax (e.g. "Single '}' encountered").
    safe_text   = text.strip().replace("{", "{{").replace("}", "}}")
    generated   = _rewrite_generate(template.format(text=safe_text), temperature=temperature)

    if not generated.strip():
        raise RewriteResponseError(
            f"[{style}] Model returned empty output. "
            "Check that the model loaded correctly and REWRITE_MODEL_PATH is valid."
        )

    result = _extract_rewrite_output(generated, text)

    if REWRITE_DEBUG:
        _log.debug(
            "[rewrite-debug] [%s] extracted result (first 300): %r",
            style, result[:300],
        )

    # Detect silent fallback: compare after collapsing all whitespace so that
    # _clean_generated_rewrite's newline→space normalization doesn't mask identity.
    def _ws_norm(s: str) -> str:
        return " ".join(s.split())

    if _ws_norm(result) == _ws_norm(text):
        raise RewriteResponseError(
            f"[{style}] Extraction returned original text unchanged (whitespace-normalized match). "
            f"Raw model output (first 300 chars): {generated[:300]!r}"
        )

    return result


def run_rewrite_ensemble(text: str) -> List[Dict[str, Any]]:
    """
    Generate all 3 rewrite candidates (conservative / balanced / aggressive).

    Returns:
        [{"style": str, "text": str}, ...]  in REWRITE_STYLES order
    """
    candidates: List[Dict[str, Any]] = []
    for style in REWRITE_STYLES:
        try:
            candidates.append({"style": style, "text": run_rewrite_candidate(text, style)})
        except Exception as exc:  # noqa: BLE001
            # Any failure (format error, model error, network error, …) falls back
            # to the original text for this candidate.  The error is preserved so
            # the orchestrator can log it and the UI can surface it.
            candidates.append(
                {
                    "style": style,
                    "text": text,
                    "error": f"{type(exc).__name__}: {exc}",
                    "reverted_to_original": True,
                }
            )
    return candidates


# ─── Local rewrite model internals ───────────────────────────────────────────


def _load_local_rewrite_model() -> tuple:
    with _REWRITE_LOCK:
        if REWRITE_MODEL_PATH in _REWRITE_CACHE:
            return _REWRITE_CACHE[REWRITE_MODEL_PATH]

        from transformers import AutoProcessor as _AutoProcessor  # lazy
        dtype     = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        processor = _AutoProcessor.from_pretrained(REWRITE_MODEL_PATH, trust_remote_code=True)

        try:
            from transformers import Qwen3VLForConditionalGeneration as _Qwen3VLForCG
            model = _Qwen3VLForCG.from_pretrained(REWRITE_MODEL_PATH, dtype=dtype, low_cpu_mem_usage=True)
        except ImportError:
            try:
                from transformers import Qwen2VLForConditionalGeneration as _Qwen2VLForCG
                model = _Qwen2VLForCG.from_pretrained(REWRITE_MODEL_PATH, dtype=dtype, low_cpu_mem_usage=True)
            except ImportError:
                from transformers import AutoModelForVision2Seq, AutoModelForCausalLM  # lazy
                try:
                    model = AutoModelForVision2Seq.from_pretrained(
                        REWRITE_MODEL_PATH, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
                    )
                except (ImportError, AttributeError):
                    model = AutoModelForCausalLM.from_pretrained(
                        REWRITE_MODEL_PATH, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
                    )

        model.to(DEVICE).eval()
        _REWRITE_CACHE[REWRITE_MODEL_PATH] = (processor, model)
        return _REWRITE_CACHE[REWRITE_MODEL_PATH]


@torch.no_grad()
def _local_rewrite_generate(prompt: str, max_new_tokens: int = 1024, temperature: float = REWRITE_TEMPERATURE) -> str:
    processor, model = _load_local_rewrite_model()
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    _chat_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if REWRITE_DISABLE_THINKING:
        # Qwen3 honours enable_thinking=False by prepending /no_think, which
        # prevents the model from entering its internal reasoning loop and
        # substantially cuts latency for short rewrite tasks.
        # Older models/other architectures silently ignore the kwarg.
        try:
            text_in = processor.apply_chat_template(
                messages, **_chat_kwargs, enable_thinking=False
            )
        except TypeError:
            text_in = processor.apply_chat_template(messages, **_chat_kwargs)
    else:
        text_in = processor.apply_chat_template(messages, **_chat_kwargs)
    inputs    = processor(text=text_in, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    # Decode only the newly generated tokens (everything after the prompt).
    new_token_ids = out[0, prompt_len:]
    raw_output = processor.decode(new_token_ids, skip_special_tokens=True).strip()

    if REWRITE_DEBUG:
        _log.debug(
            "[rewrite-debug] generated %d new tokens; raw output (first 600 chars): %r",
            len(new_token_ids),
            raw_output[:600],
        )

    return raw_output


def _resolve_external_endpoint(api_kind: str) -> str:
    base = REWRITE_BASE_URL.rstrip("/")
    normalized = api_kind.strip().lower()
    if normalized in {"openai-chat-completions", "openai_chat_completions", "openai-completions"}:
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"
    if normalized in {"anthropic-messages", "anthropic_messages"}:
        # Honour both conventions:
        #   base = "https://api.anthropic.com/v1"     → /messages
        #   base = "https://api.minimax.io/anthropic"  → /v1/messages
        if base.endswith("/messages"):
            return base
        if base.endswith("/v1"):
            return base + "/messages"
        return base + "/v1/messages"
    raise RuntimeError(f"Unsupported rewrite API kind: {api_kind}")


def _has_header_name(headers: Dict[str, str], name: str) -> bool:
    target = name.lower()
    return any(existing.lower() == target for existing in headers)


def _set_header_case_insensitive(headers: Dict[str, str], name: str, value: str) -> None:
    target = name.lower()
    for existing in list(headers.keys()):
        if existing.lower() == target:
            del headers[existing]
    headers[name] = value


def _build_external_auth_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
    }
    headers.update(REWRITE_EXTRA_HEADERS)

    if REWRITE_AUTH_MODE == "none":
        return headers

    if not REWRITE_API_KEY:
        api_key_candidates = _rewrite_config.format_env_var_list(_REWRITE_API_KEY_CANDIDATES)
        warnings.warn(
            f"REWRITE_AUTH_MODE is '{REWRITE_AUTH_MODE}' but REWRITE_API_KEY is empty — "
            "no auth header will be sent. "
            f"Set REWRITE_API_KEY in .env or env ({api_key_candidates}), or switch auth_mode to 'none'.",
            stacklevel=3,
        )
        return headers

    header_name = REWRITE_AUTH_HEADER.strip()
    if REWRITE_AUTH_MODE == "bearer":
        header_name = header_name or "Authorization"
        _set_header_case_insensitive(headers, header_name, f"Bearer {REWRITE_API_KEY}")
        return headers

    if REWRITE_AUTH_MODE == "x-api-key":
        header_name = header_name or "x-api-key"
        _set_header_case_insensitive(headers, header_name, REWRITE_API_KEY)
        return headers

    if REWRITE_AUTH_MODE == "raw":
        header_name = header_name or "Authorization"
        _set_header_case_insensitive(headers, header_name, REWRITE_API_KEY)
        return headers

    raise RuntimeError(f"Unsupported rewrite auth mode: {REWRITE_AUTH_MODE}")


def _extract_openai_chat_content(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RewriteResponseError("Rewrite API response is missing 'choices'.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        joined = "".join(parts).strip()
        if joined:
            return joined
    raise RewriteResponseError("Rewrite API response does not contain text content.")


def _extract_anthropic_content(payload: Dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, list):
        raise RewriteResponseError("Anthropic response is missing 'content'.")
    parts: List[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    joined = "".join(parts).strip()
    if not joined:
        raise RewriteResponseError("Anthropic response does not contain text content.")
    return joined


def _rewrite_api_generate(prompt: str, max_new_tokens: int = 2048, temperature: float = REWRITE_TEMPERATURE) -> str:
    if _requests is None:
        raise RuntimeError("requests package not installed.")

    api_kind = REWRITE_API_KIND or "openai-chat-completions"
    url = _resolve_external_endpoint(api_kind)
    headers = _build_external_auth_headers()

    if api_kind in {"openai-chat-completions", "openai_chat_completions", "openai-completions"}:
        payload = {
            "model": REWRITE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if REWRITE_DISABLE_THINKING:
            # Qwen3 / DashScope / vLLM: suppresses chain-of-thought reasoning.
            # Standard OpenAI GPT endpoints silently ignore unknown fields.
            # User can override via extra_body (merged below).
            payload["enable_thinking"] = False
    elif api_kind in {"anthropic-messages", "anthropic_messages"}:
        # anthropic-version is always required for Anthropic-compatible endpoints.
        if not _has_header_name(headers, "anthropic-version"):
            _set_header_case_insensitive(headers, "anthropic-version", "2023-06-01")
        # Auth header is already set by _build_external_auth_headers() based on
        # REWRITE_AUTH_MODE (bearer for MiniMax, x-api-key for standard Anthropic).
        payload = {
            "model": REWRITE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if REWRITE_DISABLE_THINKING:
            # Attempt to suppress extended thinking on Anthropic-compatible endpoints.
            # Standard Anthropic: thinking is opt-in so no parameter needed.
            # MiniMax M2.5 and similar: try both the standard Anthropic field and
            # the provider-specific enable_thinking field; unsupported fields are silently ignored.
            payload["thinking"] = {"type": "disabled"}
            payload["enable_thinking"] = False
    else:
        raise RuntimeError(f"Unsupported rewrite API kind: {api_kind}")

    payload = _merge_json_objects(payload, REWRITE_EXTRA_BODY)

    _RETRYABLE_STATUS = {429, 500, 502, 503, 504}
    r = None
    for _attempt in range(2):
        try:
            r = _requests.post(url, json=payload, headers=headers, timeout=REWRITE_TIMEOUT_S)
        except _requests.exceptions.ConnectionError as _conn_err:
            if _attempt == 0:
                _log.warning("[rewrite] Connection error on attempt 1, retrying: %s", _conn_err)
                continue
            raise
        if _attempt == 0 and r.status_code in _RETRYABLE_STATUS:
            _log.warning("[rewrite] API returned %s on attempt 1, retrying.", r.status_code)
            continue
        break
    r.raise_for_status()
    data = r.json()

    if REWRITE_DEBUG:
        # Log the raw content blocks so we can see what the model actually returned.
        raw_content = data.get("content") or data.get("choices")
        _log.debug(
            "[rewrite-debug] API response status=%s; content/choices (first 600): %r",
            r.status_code,
            str(raw_content)[:600],
        )

    if api_kind in {"openai-chat-completions", "openai_chat_completions", "openai-completions"}:
        return _extract_openai_chat_content(data)
    return _extract_anthropic_content(data)


def _rewrite_generate(prompt: str, max_new_tokens: int = 2048, temperature: float = REWRITE_TEMPERATURE) -> str:
    if REWRITE_BACKEND == "external":
        if not REWRITE_BASE_URL:
            raise RuntimeError(
                "REWRITE_BACKEND is 'external' but REWRITE_BASE_URL is not set. "
                "Set 'base_url' in the active rewrite profile or the REWRITE_BASE_URL env var."
            )
        return _rewrite_api_generate(prompt, max_new_tokens, temperature)
    return _local_rewrite_generate(prompt, max_new_tokens, temperature)


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Scan *text* for the first well-formed JSON object that parses successfully.

    Iterates over every '{' position so that a malformed or non-JSON block
    (e.g. thinking-mode preamble) does not prevent a later valid object from
    being found.
    """
    search_from = 0
    while True:
        start = text.find("{", search_from)
        if start < 0:
            return None

        depth, in_string, escape = 0, False, False
        end = -1
        for idx in range(start, len(text)):
            ch = text[idx]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx
                    break

        if end == -1:
            search_from = start + 1
            continue

        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        search_from = start + 1  # try next '{' position


_REWRITE_KEY_RE = re.compile(
    r'"rewrite"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)
# Greedy fallback: captures everything up to the LAST `"` before the closing `}`.
# Used when the model emits unescaped inner double-quotes, making the strict
# non-greedy regex stop too early.
_REWRITE_KEY_GREEDY_RE = re.compile(
    r'"rewrite"\s*:\s*"(.+)"\s*\}',
    re.DOTALL,
)
_TRUNCATED_REWRITE_KEY_RE = re.compile(
    r'(?:["\']?rewrite["\']?)\s*:\s*',
    re.IGNORECASE | re.DOTALL,
)


def _strip_code_fence_wrapper(text: str) -> str:
    raw = text.strip()
    raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, count=1, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw, count=1)
    return raw.strip()


def _decode_loose_json_string(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return ""
    try:
        return json.loads(f'"{candidate}"')
    except json.JSONDecodeError:
        return (
            candidate
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
            .replace("\\\\", "\\")
        )


def _extract_truncated_rewrite_value(text: str) -> Optional[str]:
    """
    Recover a rewrite value from JSON-shaped output that started correctly but
    was truncated before the closing quote/brace arrived.
    """
    raw = _strip_code_fence_wrapper(text)
    key_match = _TRUNCATED_REWRITE_KEY_RE.search(raw)
    if key_match is None:
        return None

    prefix = raw[: key_match.start()].lower()
    if "{" not in prefix and "json" not in prefix:
        return None

    tail = raw[key_match.end() :].lstrip()
    if not tail:
        return None

    if tail[:1] in {'"', "'"}:
        tail = tail[1:]

    tail = re.sub(r"\s*```\s*$", "", tail, count=1)
    tail = re.sub(r"[\s,]*}+\s*$", "", tail)
    tail = tail.strip()
    if not tail:
        return None

    return _decode_loose_json_string(tail)


def _extract_rewrite_output(text: str, source_text: str) -> str:
    """
    Extract the rewritten text from model output (any backend).

    Parsing chain (first match wins):
    1. Structured JSON: find the first valid JSON object and read "rewrite" key.
    2. Regex fallback: locate "rewrite": "..." even if surrounding JSON is broken.
    3. Truncated-JSON fallback: recover `rewrite` from a JSON-ish prefix whose
       closing quote/brace was cut off by truncation.
    4. Plain-text fallback: strip the outer {…} wrapper that the force-first-token
       processor induces when the model didn't follow the JSON schema, then use
       the remaining text as-is.
    """
    if REWRITE_DEBUG:
        _log.debug("[rewrite-debug] _extract input (first 400 chars): %r", text[:400])

    # 1. Structured JSON extraction
    parsed = _extract_first_json_object(text)
    if parsed is not None:
        rewrite = parsed.get("rewrite")
        if isinstance(rewrite, str) and rewrite.strip():
            if REWRITE_DEBUG:
                _log.debug("[rewrite-debug] layer-1 (JSON) matched: %r", rewrite[:200])
            return _clean_generated_rewrite(rewrite, source_text)

    # 2. Regex fallback — try both strict (non-greedy) and greedy patterns and
    #    use whichever extracts a longer value.  The greedy pattern handles the
    #    common case where the model emits unescaped inner double-quotes so the
    #    non-greedy regex stops prematurely at the first inner `"`.
    _best_rewrite: Optional[str] = None
    for _pattern in (_REWRITE_KEY_RE, _REWRITE_KEY_GREEDY_RE):
        _m = _pattern.search(text)
        if not _m:
            continue
        try:
            _val = json.loads(f'"{_m.group(1)}"')
        except json.JSONDecodeError:
            _val = _m.group(1)
        if isinstance(_val, str) and _val.strip():
            if _best_rewrite is None or len(_val.strip()) > len(_best_rewrite.strip()):
                _best_rewrite = _val
    if _best_rewrite:
        if REWRITE_DEBUG:
            _log.debug("[rewrite-debug] layer-2 (regex) matched: %r", _best_rewrite[:200])
        return _clean_generated_rewrite(_best_rewrite, source_text)

    # 3. Truncated JSON fallback — handles outputs like:
    #    {"rewrite":"...<cut off before closing quote/brace>
    truncated = _extract_truncated_rewrite_value(text)
    if truncated:
        if REWRITE_DEBUG:
            _log.debug("[rewrite-debug] layer-3 (truncated) matched: %r", truncated[:200])
        return _clean_generated_rewrite(truncated, source_text)

    # 4. Plain-text fallback: model output is not JSON at all.
    #    Strip the leading '{' and trailing '}' that the force-first-token
    #    processor adds when the model outputs prose instead of JSON.
    #    Guard: don't strip if source text itself starts/ends with { / }
    #    (the rewrite may legitimately preserve those characters).
    raw = _strip_code_fence_wrapper(text)
    _src = source_text.strip()
    if (
        raw.startswith("{") and raw.endswith("}")
        and not _src.startswith("{")
        and not _src.endswith("}")
    ):
        raw = raw[1:-1].strip()

    # Log a warning every time we hit this fallback so silent regressions surface.
    _log.warning(
        "[rewrite] All JSON extraction layers failed. Raw model output (first 300): %r. "
        "Falling back to plain-text extraction (may return original text).",
        text[:300],
    )
    if REWRITE_DEBUG:
        _log.debug("[rewrite-debug] layer-4 (plain-text) raw after strip: %r", raw[:200])

    return _clean_generated_rewrite(raw or text, source_text)


# ─── MPNet ────────────────────────────────────────────────────────────────────


def _load_mpnet() -> "SentenceTransformer":
    if not _HAS_ST:
        raise RuntimeError(
            "sentence-transformers is not installed. Run: pip install sentence-transformers"
        )
    with _MPNET_LOCK:
        if MPNET_MODEL_PATH not in _MPNET_CACHE:
            with contextlib.redirect_stdout(io.StringIO()):
                _MPNET_CACHE[MPNET_MODEL_PATH] = SentenceTransformer(
                    MPNET_MODEL_PATH,
                    device=DEVICE,
                )
        return _MPNET_CACHE[MPNET_MODEL_PATH]


def run_mpnet_similarity(text1: str, text2: str) -> float:
    """Cosine similarity between two texts in [0, 1]."""
    model = _load_mpnet()
    embs  = model.encode([text1, text2], convert_to_tensor=True)
    return float(_st_util.cos_sim(embs[0], embs[1]).item())


def run_mpnet_sentence_align(
    orig_sents: List[str],
    rw_sents:   List[str],
) -> List[Dict[str, Any]]:
    """
    For each rewritten sentence, find its closest original sentence via full
    similarity matrix. Returns list of dicts:
      { rw_sent, best_orig, similarity, needs_fix (always False — threshold applied by caller) }
    """
    if not orig_sents or not rw_sents:
        return [
            {"rw_sent": s, "best_orig": "", "similarity": 1.0, "needs_fix": False}
            for s in rw_sents
        ]

    model    = _load_mpnet()
    all_embs = model.encode(orig_sents + rw_sents, convert_to_tensor=True)
    orig_embs = all_embs[: len(orig_sents)]
    rw_embs   = all_embs[len(orig_sents) :]
    sim_matrix = _st_util.cos_sim(rw_embs, orig_embs)

    results: List[Dict[str, Any]] = []
    for i, rw_s in enumerate(rw_sents):
        best_j   = int(sim_matrix[i].argmax().item())
        best_sim = float(sim_matrix[i][best_j].item())
        results.append({"rw_sent": rw_s, "best_orig": orig_sents[best_j],
                         "similarity": best_sim, "needs_fix": False})
    return results


# ─── Preload ──────────────────────────────────────────────────────────────────


def preload_models() -> List[Dict[str, Any]]:
    """
    Preload locally configured models into the serving process.
    Returns [{name, status, seconds, detail?}, ...]
    """
    tasks = [
        ("desklib", _load_desklib),
        ("mpnet",   _load_mpnet),
    ]
    if REWRITE_BACKEND == "local":
        tasks.append(("rewrite_local", _load_local_rewrite_model))

    results: List[Dict[str, Any]] = []
    for name, loader in tasks:
        started = time.perf_counter()
        try:
            loader()
            results.append({"name": name, "status": "ok",
                             "seconds": round(time.perf_counter() - started, 3)})
        except Exception as exc:
            results.append({"name": name, "status": "error",
                             "seconds": round(time.perf_counter() - started, 3),
                             "detail": str(exc)})
            raise RuntimeError(f"Failed to preload {name}: {exc}") from exc

    return results
