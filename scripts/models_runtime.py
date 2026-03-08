"""
Model inference layer for the writing optimizer pipeline.

Models managed here
-------------------
- desklib/ai-text-detector-v1.01    → AI-content classification (SequenceClassification)
- Qwen3-VL-8B-Instruct              → rewrite ensemble
                                       (local Qwen2VL or OpenAI-compatible API)
- paraphrase-mpnet-base-v2          → semantic similarity (SentenceTransformer)

All models are lazily loaded and cached in module-level dicts.
"""

from __future__ import annotations

import contextlib
import json
import io
import os
import platform
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import torch
from dotenv import load_dotenv
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DebertaV2Config,
    DebertaV2Model,
)
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers import AutoProcessor as _AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration as _Qwen3VLForCG
    _HAS_QWEN3VL = True
except ImportError:
    _HAS_QWEN3VL = False

try:
    from transformers import Qwen2VLForConditionalGeneration as _Qwen2VLForCG
    _HAS_QWEN2VL = True
except ImportError:
    _HAS_QWEN2VL = False

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

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if platform.system() == "Darwin" and getattr(torch.backends, "mps", None):
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


DEVICE = select_device()

DESKLIB_MODEL_PATH = _resolve(os.getenv("DESKLIB_MODEL_PATH", "./models/desklib-ai-text-detector-v1.01"))
MPNET_MODEL_PATH   = _resolve(os.getenv("MPNET_MODEL_PATH",   "./models/paraphrase-mpnet-base-v2"))
QWEN_BACKEND       = os.getenv("QWEN_BACKEND", "local").lower()
QWEN_MODEL_PATH    = _resolve(os.getenv("QWEN_MODEL_PATH",    "./models/Qwen3-VL-8B-Instruct"))
QWEN_API_BASE_URL  = os.getenv("QWEN_API_BASE_URL", "")
QWEN_API_MODEL     = os.getenv("QWEN_API_MODEL",    "Qwen3-VL-8B-Instruct")

_DESKLIB_CACHE: Dict[str, Any] = {}
_MPNET_CACHE:   Dict[str, Any] = {}
_QWEN_CACHE:    Dict[str, Any] = {}

_META_EXPLANATION_RE = re.compile(
    r"\b("
    r"the original text|original sentence|paraphrased version|rewritten version|"
    r"this version|this rewrite|here(?:'s| is) the rewritten|"
    r"explanation|note that|I rewrote|I changed|I kept|while preserving"
    r")\b",
    flags=re.IGNORECASE,
)


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

    cleaned = cleaned.strip(" \"'\u201c\u201d\u2018\u2019")
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
    if DESKLIB_MODEL_PATH not in _DESKLIB_CACHE:
        from safetensors.torch import load_file as _load_safetensors

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

_REWRITE_PROMPTS: Dict[str, str] = {
    "conservative": (
        "You are a careful editor.\n"
        "Rewrite the text below to sound slightly more natural and human-written "
        "while preserving the exact meaning.\n"
        "Make only minimal edits: lightly vary sentence structure and reduce overly "
        "formulaic or template-like phrasing.\n"
        "Keep all facts, numbers, named entities, and logical relationships unchanged.\n"
        "Do not add, remove, or rearrange information.\n"
        "Keep the rewrite close in length to the original.\n"
        "Return JSON only. No markdown, explanation, or extra keys.\n"
        "The first character of your response must be '{{'.\n"
        "Required schema:\n"
        "{{\"rewrite\": \"<rewritten text>\"}}\n\n"
        "Text:\n{text}\n"
    ),
    "balanced": (
        "You are a skilled human writer.\n"
        "Rewrite the text below to sound more natural, less formulaic, and more like "
        "something a person would actually write, while fully preserving the original meaning.\n"
        "Vary sentence structure and wording naturally. Reduce any overly uniform rhythm "
        "or template-like phrasing.\n"
        "Keep all facts, numbers, named entities, and core claims unchanged.\n"
        "Do not add or remove information. Keep the rewrite close in length to the original.\n"
        "Return JSON only. No markdown, explanation, or extra keys.\n"
        "The first character of your response must be '{{'.\n"
        "Required schema:\n"
        "{{\"rewrite\": \"<rewritten text>\"}}\n\n"
        "Text:\n{text}\n"
    ),
    "aggressive": (
        "You are an expert editor.\n"
        "Rewrite the text below so it reads naturally and feels distinctly human "
        "rather than standardized AI-generated prose, while keeping the original meaning intact.\n"
        "You may restructure sentences, merge or split clauses, and vary phrasing substantially.\n"
        "Preserve all facts, numbers, named entities, and intended claims.\n"
        "Do not invent new content. Keep the rewrite close in length to the original.\n"
        "Return JSON only. No markdown, explanation, or extra keys.\n"
        "The first character of your response must be '{{'.\n"
        "Required schema:\n"
        "{{\"rewrite\": \"<rewritten text>\"}}\n\n"
        "Text:\n{text}\n"
    ),
}


def run_rewrite_candidate(text: str, style: str = "balanced") -> str:
    """
    Generate a single rewrite candidate with the specified aggressiveness style.

    style options:
      "conservative" → minimal edits, maximum semantic preservation
      "balanced"     → moderate restructuring, good AI-score reduction
      "aggressive"   → substantial rephrasing, highest AI-score reduction potential
    """
    template  = _REWRITE_PROMPTS.get(style, _REWRITE_PROMPTS["balanced"])
    generated = _qwen_generate(template.format(text=text.strip()))
    return _extract_rewrite_from_qwen_output(generated, text)


def run_rewrite_ensemble(text: str) -> List[Dict[str, Any]]:
    """
    Generate all 3 rewrite candidates (conservative / balanced / aggressive).

    Returns:
        [{"style": str, "text": str}, ...]  in REWRITE_STYLES order
    """
    return [
        {"style": style, "text": run_rewrite_candidate(text, style)}
        for style in REWRITE_STYLES
    ]


# ─── Qwen internals ───────────────────────────────────────────────────────────

_QWEN_ALIGN_PROMPT = (
    "The rewritten sentence below has drifted too far from the original meaning. "
    "Correct it so it accurately conveys the original meaning while still sounding natural and human-written.\n"
    "Keep the corrected sentence close in length to the original sentence. Do not expand the content.\n"
    "Return JSON only. Do not add markdown, explanation, or extra keys.\n"
    "The first character of your response must be '{{'.\n"
    "Required schema:\n"
    "{{\"rewrite\": \"<corrected sentence>\"}}\n\n"
    "Original:         {original}\n"
    "Current rewrite:  {rewritten}\n"
)


class _ForceFirstTokenProcessor(LogitsProcessor):
    def __init__(self, prompt_len: int, allowed_token_ids: List[int]) -> None:
        self.prompt_len        = prompt_len
        self.allowed_token_ids = sorted(set(allowed_token_ids))

    def __call__(self, input_ids: "torch.LongTensor", scores: "torch.FloatTensor") -> "torch.FloatTensor":
        if not self.allowed_token_ids:
            return scores
        if input_ids.shape[1] != self.prompt_len:
            return scores
        forced = torch.full_like(scores, float("-inf"))
        forced[:, self.allowed_token_ids] = scores[:, self.allowed_token_ids]
        return forced


def _load_qwen_local() -> tuple:
    if QWEN_MODEL_PATH in _QWEN_CACHE:
        return _QWEN_CACHE[QWEN_MODEL_PATH]

    dtype     = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    processor = _AutoProcessor.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)

    if _HAS_QWEN3VL:
        model = _Qwen3VLForCG.from_pretrained(QWEN_MODEL_PATH, dtype=dtype, low_cpu_mem_usage=True)
    elif _HAS_QWEN2VL:
        model = _Qwen2VLForCG.from_pretrained(QWEN_MODEL_PATH, dtype=dtype, low_cpu_mem_usage=True)
    else:
        from transformers import AutoModelForVision2Seq
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                QWEN_MODEL_PATH, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_PATH, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
            )

    model.to(DEVICE).eval()
    _QWEN_CACHE[QWEN_MODEL_PATH] = (processor, model)
    return _QWEN_CACHE[QWEN_MODEL_PATH]


def _qwen_open_brace_token_ids(processor: Any) -> List[int]:
    tokenizer  = processor.tokenizer
    direct_ids = tokenizer.encode("{", add_special_tokens=False)
    if len(direct_ids) == 1:
        return [direct_ids[0]]

    allowed: List[int] = []
    vocab = tokenizer.get_vocab()
    for token, token_id in vocab.items():
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if decoded == "{":
            allowed.append(token_id)

    if allowed:
        return sorted(set(allowed))
    raise RuntimeError("Could not determine a single-token encoding for '{' in Qwen tokenizer.")


@torch.no_grad()
def _qwen_local_generate(prompt: str, max_new_tokens: int = 512) -> str:
    processor, model = _load_qwen_local()
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text_in  = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs   = processor(text=text_in, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]
    logits_processor = LogitsProcessorList(
        [_ForceFirstTokenProcessor(prompt_len, _qwen_open_brace_token_ids(processor))]
    )
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        logits_processor=logits_processor,
    )
    return processor.decode(out[0, prompt_len:], skip_special_tokens=True).strip()


def _qwen_api_generate(prompt: str, max_new_tokens: int = 512) -> str:
    if _requests is None:
        raise RuntimeError("requests package not installed.")
    url     = QWEN_API_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model":       QWEN_API_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  max_new_tokens,
        "temperature": 0.7,
    }
    r = _requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _qwen_generate(prompt: str, max_new_tokens: int = 512) -> str:
    if QWEN_BACKEND == "openai_api":
        return _qwen_api_generate(prompt, max_new_tokens)
    return _qwen_local_generate(prompt, max_new_tokens)


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
            return None  # no balanced closing brace anywhere after search_from

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


def _extract_rewrite_from_qwen_output(text: str, source_text: str) -> str:
    """
    Extract the rewritten text from model output.

    Parsing chain (first match wins):
    1. Structured JSON: find the first valid JSON object and read "rewrite" key.
    2. Regex fallback: locate "rewrite": "..." even if surrounding JSON is broken.
    3. Plain-text fallback: strip the outer {…} wrapper that the force-first-token
       processor induces when the model didn't follow the JSON schema, then use
       the remaining text as-is.
    """
    # 1. Structured JSON extraction
    parsed = _extract_first_json_object(text)
    if parsed is not None:
        rewrite = parsed.get("rewrite")
        if isinstance(rewrite, str) and rewrite.strip():
            return _clean_generated_rewrite(rewrite, source_text)

    # 2. Regex fallback — handles valid "rewrite" value inside otherwise broken JSON
    m = _REWRITE_KEY_RE.search(text)
    if m:
        try:
            rewrite = json.loads(f'"{m.group(1)}"')
        except json.JSONDecodeError:
            rewrite = m.group(1)
        if isinstance(rewrite, str) and rewrite.strip():
            return _clean_generated_rewrite(rewrite, source_text)

    # 3. Plain-text fallback: model output is not JSON at all.
    #    Strip the leading '{' and trailing '}' that the force-first-token
    #    processor adds when the model outputs prose instead of JSON.
    raw = text.strip()
    if raw.startswith("{") and raw.endswith("}"):
        raw = raw[1:-1].strip()
    return _clean_generated_rewrite(raw or text, source_text)


def run_qwen_align(original: str, rewritten: str) -> str:
    """Fix *rewritten* to match the semantic meaning of *original*."""
    prompt    = _QWEN_ALIGN_PROMPT.format(original=original.strip(), rewritten=rewritten.strip())
    generated = _qwen_generate(prompt, max_new_tokens=256)
    return _extract_rewrite_from_qwen_output(generated, original)


# ─── MPNet ────────────────────────────────────────────────────────────────────


def _load_mpnet() -> "SentenceTransformer":
    if not _HAS_ST:
        raise RuntimeError(
            "sentence-transformers is not installed. Run: pip install sentence-transformers"
        )
    if MPNET_MODEL_PATH not in _MPNET_CACHE:
        with contextlib.redirect_stdout(io.StringIO()):
            _MPNET_CACHE[MPNET_MODEL_PATH] = SentenceTransformer(MPNET_MODEL_PATH)
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
    if QWEN_BACKEND == "local":
        tasks.append(("qwen", _load_qwen_local))

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
