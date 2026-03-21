#!/usr/bin/env python3
"""
Pipeline diagnostic script — Trileaf

Layers checked:
  1. Active profile / backend resolution
  2. External API connectivity (raw HTTP, no prompt engineering)
  3. run_rewrite_candidate (single call, capture raw model output)
  4. Full mini-pipeline (chunk → rewrite → score → gate)

Run:
    python scripts/diag_pipeline.py
    python scripts/diag_pipeline.py --text "Your text here."
    python scripts/diag_pipeline.py --api-only     # skip heavy model checks
    python scripts/diag_pipeline.py --raw-response # print full raw API response
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_DIVIDER = "─" * 70
SAMPLE_TEXT = (
    "Artificial intelligence has transformed the way modern businesses operate. "
    "Companies are increasingly adopting machine learning solutions to automate "
    "repetitive tasks and improve decision-making processes across all departments."
)


def _ok(msg: str)   -> None: print(f"  [OK]      {msg}")
def _warn(msg: str) -> None: print(f"  [WARN]    {msg}")
def _fail(msg: str) -> None: print(f"  [FAIL]    {msg}")
def _info(msg: str) -> None: print(f"  ·         {msg}")
def _hdr(title: str) -> None:
    print()
    print(_DIVIDER)
    print(f"  {title}")
    print(_DIVIDER)


# ── Layer 1: Credential resolution ────────────────────────────────────────────

def check_profile() -> dict:
    _hdr("Layer 1 — Credential & backend resolution")
    from scripts import rewrite_config as rc
    import os

    # Resolve credentials (LeafHub → env)
    result = rc.resolve_credentials()
    cred_source = result.get("credential_source", "none")

    _ok(f"Backend:        external")
    _ok(f"Credential:     {cred_source}")

    api_kind = os.getenv("REWRITE_API_KIND") or "openai-chat-completions"
    base_url = os.getenv("REWRITE_BASE_URL") or ""
    model    = os.getenv("REWRITE_MODEL") or ""
    api_key  = os.getenv("REWRITE_API_KEY", "")

    _ok(f"api_kind:       {api_kind}")
    _ok(f"base_url:       {base_url}")
    _ok(f"model:          {model}")
    _ok(f"api_key:        {'set (' + cred_source + ')' if api_key else '(NOT SET)'}")

    if not base_url:
        _fail("REWRITE_BASE_URL is empty — external backend will fail")
    if not model:
        _warn("REWRITE_MODEL is empty")
    if not api_key:
        _fail("API key is not set — run: trileaf config")

    return {"backend": "external", "profile": {}}


# ── Layer 2: Raw API connectivity ─────────────────────────────────────────────

def check_api_raw(profile_info: dict, *, show_full_response: bool = False) -> bool:
    _hdr("Layer 2 — External API raw connectivity")

    import requests as _req
    import os
    from scripts import rewrite_config as rc

    api_kind    = (os.getenv("REWRITE_API_KIND") or "openai-chat-completions").strip().lower()
    base_url    = (os.getenv("REWRITE_BASE_URL") or "").rstrip("/")
    model       = os.getenv("REWRITE_MODEL") or ""
    api_key     = os.getenv("REWRITE_API_KEY", "")
    auth_mode   = (os.getenv("REWRITE_AUTH_MODE") or "bearer").strip().lower()
    auth_header = os.getenv("REWRITE_AUTH_HEADER") or "Authorization"

    headers: dict = {"Content-Type": "application/json"}
    if api_key:
        if auth_mode == "x-api-key":
            headers["x-api-key"] = api_key
        else:
            headers[auth_header] = f"Bearer {api_key}"

    ping_text = "Reply with exactly one word: hello"

    # Use the same URL resolution as models_runtime to avoid path mismatches.
    from scripts.models_runtime import _resolve_external_endpoint
    url = _resolve_external_endpoint(api_kind)

    if api_kind in {"anthropic-messages", "anthropic_messages"}:
        if "anthropic-version" not in {k.lower() for k in headers}:
            headers["anthropic-version"] = "2023-06-01"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": ping_text}],
            "max_tokens": 64,
        }
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": ping_text}],
            "max_tokens": 64,
        }

    _info(f"POST {url}")
    _info(f"Model: {model}  |  api_kind: {api_kind}")

    t0 = time.time()
    try:
        resp = _req.post(url, json=payload, headers=headers, timeout=30)
        elapsed = time.time() - t0
        _info(f"HTTP {resp.status_code}  ({elapsed:.2f}s)")

        if show_full_response:
            print()
            print("  Raw response body:")
            try:
                print(json.dumps(resp.json(), indent=4, ensure_ascii=False)[:2000])
            except Exception:
                print(resp.text[:2000])
            print()

        resp.raise_for_status()
        data = resp.json()

        # Extract reply text
        reply = ""
        if api_kind in {"anthropic-messages", "anthropic_messages"}:
            content = data.get("content", [])
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    reply += item.get("text", "")
        else:
            choices = data.get("choices", [])
            if choices:
                reply = (choices[0].get("message") or {}).get("content", "")

        reply_text = reply.strip()
        if reply_text:
            _ok(f"API replied: {reply_text[:120]!r}")
        else:
            _warn("API replied HTTP 200 but returned no text block (thinking may have used all tokens — connectivity OK)")
        return True

    except Exception as exc:
        elapsed = time.time() - t0
        _fail(f"API call failed after {elapsed:.2f}s: {exc}")
        if hasattr(exc, "response") and exc.response is not None:
            _fail(f"Response body: {exc.response.text[:400]}")
        return False


# ── Layer 3: run_rewrite_candidate ────────────────────────────────────────────

def check_rewrite_candidate(text: str, profile_info: dict) -> bool:
    _hdr("Layer 3 — run_rewrite_candidate (balanced style)")

    # Temporarily enable REWRITE_DEBUG so models_runtime logs raw output
    os.environ["REWRITE_DEBUG"] = "1"

    try:
        import scripts.models_runtime as mr

        _info(f"REWRITE_BACKEND:  {mr.REWRITE_BACKEND}")
        _info(f"REWRITE_MODEL:    {mr.REWRITE_MODEL}")
        _info(f"REWRITE_API_KIND: {mr.REWRITE_API_KIND}")
        _info(f"REWRITE_BASE_URL: {mr.REWRITE_BASE_URL or '(not set)'}")
        _info(f"API key set:      {bool(mr.REWRITE_API_KEY)}")
        print()

        _info(f"Input text ({len(text)} chars):")
        print(f"    {text[:200]}")
        print()

        t0 = time.time()
        result = mr.run_rewrite_candidate(text, style="balanced")
        elapsed = time.time() - t0

        _ok(f"Rewrite succeeded in {elapsed:.2f}s")
        _ok(f"Output ({len(result)} chars):")
        print(f"    {result[:300]}")

        if result.strip() == text.strip():
            _fail("Output is IDENTICAL to input — silent fallback detected")
            return False

        return True

    except Exception as exc:
        _fail(f"{type(exc).__name__}: {exc}")
        if "--traceback" in sys.argv:
            traceback.print_exc()
        else:
            _info("Run with --traceback for full stack trace")
        return False


# ── Layer 4: full mini-pipeline ───────────────────────────────────────────────

def check_full_pipeline(text: str) -> None:
    _hdr("Layer 4 — Full mini-pipeline (chunk → rewrite × 3 → score → gate)")

    try:
        import scripts.models_runtime as mr

        _info(f"Text: {text[:100]!r}")
        print()

        # Step 0: Baseline scoring
        t0 = time.time()
        orig_ai = mr.run_desklib(text)
        _info(f"Baseline  AI={orig_ai:.4f}  ({time.time()-t0:.2f}s)")
        print()

        # Step 1: Rewrite all 3 styles
        for style in mr.REWRITE_STYLES:
            print(f"  [{style:>12}] calling rewrite...")
            t0 = time.time()
            try:
                cand = mr.run_rewrite_candidate(text, style=style)
                elapsed = time.time() - t0

                # Step 2: Score this candidate
                ai_score  = mr.run_desklib(cand)
                sem_score = mr.run_mpnet_similarity(text, cand)

                identical = " ← IDENTICAL TO ORIGINAL" if cand.strip() == text.strip() else ""
                print(
                    f"  [{style:>12}] OK  {elapsed:.1f}s  "
                    f"ai={ai_score:.4f} (orig={orig_ai:.4f})  "
                    f"sem={sem_score:.4f}{identical}"
                )
                print(f"               → {cand[:120]!r}")

                # Step 3: Gate
                gate_pass = (ai_score < orig_ai) and (sem_score >= 0.65)
                print(f"               gate={'PASS' if gate_pass else 'FAIL'}")

            except Exception as exc:
                elapsed = time.time() - t0
                print(f"  [{style:>12}] FAIL  {elapsed:.1f}s  {type(exc).__name__}: {exc}")
            print()

    except Exception as exc:
        _fail(f"Pipeline setup failed: {exc}")
        traceback.print_exc()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Trileaf pipeline diagnostic")
    parser.add_argument("--text",         default=SAMPLE_TEXT, help="Input text to test")
    parser.add_argument("--api-only",     action="store_true",  help="Only check profile + API (skip model loads)")
    parser.add_argument("--raw-response", action="store_true",  help="Print full raw API response in Layer 2")
    parser.add_argument("--traceback",    action="store_true",  help="Print full tracebacks on errors")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  Trileaf — Pipeline Diagnostic")
    print("=" * 70)

    profile_info = check_profile()
    if not profile_info:
        print("\n[ABORT] Cannot continue without a valid profile.")
        sys.exit(1)

    api_ok = check_api_raw(profile_info, show_full_response=args.raw_response)

    if not args.api_only:
        if not api_ok:
            print()
            _warn("Skipping Layer 3/4 because API is not reachable.")
        else:
            cand_ok = check_rewrite_candidate(args.text, profile_info)
            if cand_ok:
                check_full_pipeline(args.text)

    print()
    print("=" * 70)
    print("  Done.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
