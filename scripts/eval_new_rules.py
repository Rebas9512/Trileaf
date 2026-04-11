#!/usr/bin/env python3
"""
Evaluation script for new Humanizer-derived rules.

Each test case is heavily loaded with patterns from the newly added rules
(significance inflation, superficial -ing, promotional language, chatbot
artifacts, copula avoidance, signposting, generic conclusions, etc.)
so we can see whether the pipeline catches and rewrites them.

Usage:
    python scripts/eval_new_rules.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rule_detector import analyze_document


# ── Test cases (one per genre, heavy on new-rule patterns) ─────────────────

@dataclass
class TestCase:
    name: str
    genre_expected: str
    text: str


CASES: List[TestCase] = [
    # ── Academic ───────────────────────────────────────────────────────────
    TestCase(
        name="Academic (new-rule heavy)",
        genre_expected="academic",
        text=(
            "The implementation of blockchain technology in healthcare represents a "
            "pivotal moment in the evolution of medical data management, marking a "
            "key turning point for the industry. This groundbreaking initiative stands "
            "as a testament to the transformative potential of decentralized systems, "
            "underscoring its importance in the evolving landscape of digital health. "
            "Experts argue that the integration of distributed ledger technology serves "
            "as a catalyst for innovation, highlighting the intricate interplay between "
            "security and accessibility. Furthermore, the framework encompasses a "
            "comprehensive approach to data governance, fostering collaboration among "
            "stakeholders. It is important to note that, while specific details are "
            "limited based on available information, the empirical foundation suggests "
            "that blockchain could potentially be argued to have significant implications. "
            "Despite challenges typical of emerging technologies, the ecosystem continues "
            "to thrive. The future looks bright for healthcare informatics."
        ),
    ),

    # ── Professional ──────────────────────────────────────────────────────
    TestCase(
        name="Professional (new-rule heavy)",
        genre_expected="professional",
        text=(
            "Dear Hiring Manager, I hope this helps clarify my qualifications for the "
            "Senior Product Manager role. I am writing to express my interest in this "
            "pivotal role at your vibrant organization. My experience serves as a "
            "testament to my commitment to excellence in product development. I have "
            "spearheaded cross-functional initiatives, showcasing my ability to align "
            "with organizational goals. The real question is whether a candidate can "
            "deliver results — and my track record stands as proof. At its core, my "
            "approach to product management features a data-driven methodology that "
            "has been featured in TechCrunch, Forbes, and Wired. Furthermore, I am "
            "passionate about leveraging innovative solutions to enhance user experience. "
            "Let me know if you would like me to expand on any section. I look forward "
            "to this exciting opportunity. The future looks bright for this role."
        ),
    ),

    # ── Narrative ─────────────────────────────────────────────────────────
    TestCase(
        name="Narrative (new-rule heavy)",
        genre_expected="narrative",
        text=(
            "Let's dive in to the story of how everything changed. The old lighthouse "
            "stood as a testament to the town's enduring spirit, nestled in the heart "
            "of the rugged coastline, its breathtaking silhouette showcasing decades of "
            "resilience. The protagonist made his way to the shore. The main character "
            "felt the wind on his face. The central figure paused to reflect. The hero "
            "looked out at the sea. From the crashing waves to the distant horizon, from "
            "the cry of gulls to the silence of the deep — the scene was vibrant with "
            "natural beauty. The lighthouse serves as a beacon, symbolizing hope and "
            "renewal, reflecting the community's deep connection to the land. At its "
            "core, what really matters is the enduring bond between the keeper and the "
            "sea. Exciting times lie ahead for this stunning coastal town."
        ),
    ),

    # ── Casual ────────────────────────────────────────────────────────────
    TestCase(
        name="Casual (new-rule heavy)",
        genre_expected="casual",
        text=(
            "Great question! Here's what you need to know about the new deployment "
            "process. Certainly! The updated pipeline serves as our primary deployment "
            "mechanism, showcasing improved reliability. Furthermore, it features a "
            "comprehensive monitoring dashboard. I hope this helps! The system boasts "
            "a vibrant interface with stunning visualizations. Let me know if you need "
            "anything else. You're absolutely right that we should also look at the "
            "logging setup. That's an excellent point about the retry logic. Without "
            "further ado, let's break this down — the real question is whether the "
            "new system can handle the load, no guessing."
        ),
    ),

    # ── Persuasive ────────────────────────────────────────────────────────
    TestCase(
        name="Persuasive (new-rule heavy)",
        genre_expected="persuasive",
        text=(
            "Let's dive into why remote work mandates are fundamentally misguided. "
            "At its core, the heart of the matter is simple: forcing employees back "
            "serves as a testament to outdated management thinking. Studies show that "
            "remote workers are more productive. Experts argue that the benefits are "
            "clear. Industry reports suggest a pivotal shift in workplace culture. The "
            "real question is whether executives are willing to adapt. It's not just "
            "about flexibility — it's about respecting autonomy, enhancing well-being, "
            "and fostering a culture of trust. Furthermore, the companies championing "
            "this return are the same ones with the most expensive real estate, "
            "highlighting the intricate interplay between financial interests and "
            "corporate policy. Despite these challenges, the remote work movement "
            "continues to thrive. The future looks bright. Exciting times lie ahead."
        ),
    ),
]


# ── Rule analysis (pre-pipeline, shows what the new rules detect) ──────────

def analyze_rules(case: TestCase) -> Dict[str, Any]:
    """Run rule_detector on the raw text to see which new rules fire."""
    doc = analyze_document(case.text)
    rule_counts: Dict[str, int] = {}
    all_violations = []
    for sa in doc.sentences:
        for v in sa.violations:
            rule_counts[v.rule_id] = rule_counts.get(v.rule_id, 0) + 1
            all_violations.append(v)

    # Separate new rules from old rules
    new_prefixes = ("content.", "artifact.", "style.", "struct.tailing_negation")
    new_rules = {k: v for k, v in rule_counts.items() if k.startswith(new_prefixes)}
    old_rules = {k: v for k, v in rule_counts.items() if not k.startswith(new_prefixes)}

    return {
        "summary": doc.summary,
        "new_rules": new_rules,
        "old_rules": old_rules,
        "total_violations": len(all_violations),
        "new_violation_count": sum(new_rules.values()),
        "top_issues": doc.top_issues[:8],
    }


# ── Pipeline runner (reused from eval_pipeline.py) ─────────────────────────

async def run_single_case(case: TestCase) -> Dict[str, Any]:
    from scripts.orchestrator_v2 import run_pipeline_v2
    import uuid

    events: List[Dict] = []

    async def collector(event_type: str, data: dict):
        events.append({"type": event_type, "data": data})

    run_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()

    result = await run_pipeline_v2(
        text=case.text,
        broadcast=collector,
        run_id=run_id,
        chunk_mode="short",
    )

    elapsed = time.perf_counter() - start

    genre = "unknown"
    for e in events:
        if e["type"] == "stage1_done" and "genre" in e["data"]:
            genre = e["data"]["genre"]

    stage6_status = "n/a"
    for e in events:
        if e["type"] == "stage6_skipped":
            stage6_status = f"skipped ({e['data'].get('genre', '')})"
        elif e["type"] == "stage6_done":
            accepted = e["data"].get("accepted", False)
            stage6_status = "accepted" if accepted else "reverted"

    return {
        "name": case.name,
        "genre_expected": case.genre_expected,
        "genre_detected": genre,
        "input_words": len(case.text.split()),
        "output_words": len(result.output_text.split()),
        "original_ai": result.original_ai_score,
        "final_ai": result.final_ai_score,
        "sem_score": result.final_sem_score,
        "ai_drop_pct": round((1 - result.final_ai_score / max(result.original_ai_score, 0.01)) * 100, 1),
        "unfixed_count": len(result.unfixed_sentences),
        "stage6": stage6_status,
        "elapsed_s": round(elapsed, 1),
        "input_full": case.text,
        "output_full": result.output_text,
    }


# ── Display ────────────────────────────────────────────────────────────────

def print_rule_analysis(cases: List[TestCase]):
    print("\n" + "=" * 100)
    print("PHASE 1: RULE DETECTION ANALYSIS (before pipeline)")
    print("Shows which new rules fire on each test case")
    print("=" * 100)

    for i, case in enumerate(cases, 1):
        analysis = analyze_rules(case)
        print(f"\n{'─' * 100}")
        print(f"  CASE {i}: {case.name}")
        print(f"  Severity: {analysis['summary']}")
        print(f"  Total violations: {analysis['total_violations']} "
              f"(new rules: {analysis['new_violation_count']})")
        print(f"  Top issues: {analysis['top_issues']}")

        if analysis["new_rules"]:
            print(f"  NEW rules fired:")
            for rule_id, count in sorted(analysis["new_rules"].items(),
                                         key=lambda x: x[1], reverse=True):
                print(f"    {rule_id}: {count}")
        else:
            print(f"  NEW rules fired: (none)")

        if analysis["old_rules"]:
            print(f"  Old rules fired:")
            for rule_id, count in sorted(analysis["old_rules"].items(),
                                         key=lambda x: x[1], reverse=True):
                print(f"    {rule_id}: {count}")
    print()


def print_pipeline_results(results: List[Dict]):
    print("\n" + "=" * 100)
    print("PHASE 2: FULL PIPELINE RESULTS")
    print("=" * 100)
    print(f"{'#':<3} {'Case':<30} {'Genre':>12} {'AI':>12} {'Sem':>5} {'Drop':>6} {'S6':>10} {'Time':>6}")
    print("-" * 100)

    for i, r in enumerate(results, 1):
        genre_match = "✓" if r["genre_expected"] == r["genre_detected"] else "✗"
        genre_str = f"{r['genre_expected'][:4]}→{r['genre_detected'][:4]} {genre_match}"
        ai_str = f"{r['original_ai']:.2f}→{r['final_ai']:.2f}"
        print(
            f"{i:<3} {r['name']:<30} {genre_str:>12} {ai_str:>12} "
            f"{r['sem_score']:.2f}  {r['ai_drop_pct']:>5.1f}% "
            f"{r['stage6']:>10} {r['elapsed_s']:>5.1f}s"
        )

    print("-" * 100)
    avg_drop = sum(r["ai_drop_pct"] for r in results) / len(results)
    avg_sem = sum(r["sem_score"] for r in results) / len(results)
    genre_acc = sum(1 for r in results if r["genre_expected"] == r["genre_detected"]) / len(results) * 100
    print(f"    Avg AI drop: {avg_drop:.1f}%  |  Avg semantic: {avg_sem:.2f}  |  Genre accuracy: {genre_acc:.0f}%")


def print_comparisons(results: List[Dict]):
    print("\n" + "=" * 100)
    print("PHASE 3: INPUT → OUTPUT COMPARISONS")
    print("=" * 100)

    for i, r in enumerate(results, 1):
        print(f"\n{'─' * 100}")
        print(f"  CASE {i}: {r['name']}")
        print(f"  Genre: {r['genre_detected']}  |  AI: {r['original_ai']:.3f} → {r['final_ai']:.3f} "
              f"({r['ai_drop_pct']:+.1f}%)  |  Sem: {r['sem_score']:.2f}  |  S6: {r['stage6']}")

        # Post-pipeline rule check
        post_analysis = analyze_rules(TestCase(name="", genre_expected="", text=r["output_full"]))

        print(f"  Violations: before={analyze_rules(TestCase(name='', genre_expected='', text=r['input_full']))['total_violations']}"
              f" → after={post_analysis['total_violations']}"
              f" (new rules remaining: {post_analysis['new_violation_count']})")

        print(f"\n  INPUT:")
        for line in r["input_full"].split("\n"):
            print(f"    {line}")

        print(f"\n  OUTPUT:")
        for line in r["output_full"].split("\n"):
            print(f"    {line}")

        if post_analysis["new_rules"]:
            print(f"\n  Remaining new-rule violations in output:")
            for rule_id, count in sorted(post_analysis["new_rules"].items(),
                                         key=lambda x: x[1], reverse=True):
                print(f"    {rule_id}: {count}")
        print()


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    # Phase 1: rule analysis only (no LLM needed)
    print_rule_analysis(CASES)

    # Phase 2 & 3: full pipeline
    print("Resolving rewrite credentials via LeafHub...")
    import os
    from scripts.rewrite_config import resolve_credentials
    creds = resolve_credentials()
    for k, v in creds.items():
        if k.startswith("REWRITE_") or k.startswith("rewrite_"):
            os.environ[k.upper()] = str(v)
    print(f"  Provider: {os.environ.get('REWRITE_MODEL', '?')} via {os.environ.get('REWRITE_API_KIND', '?')}")

    print("Loading pipeline components...")
    import scripts.models_runtime  # noqa
    print(f"Running {len(CASES)} evaluation scenarios...\n")

    results = []
    for i, case in enumerate(CASES, 1):
        print(f"  [{i}/{len(CASES)}] {case.name}...", end="", flush=True)
        try:
            r = await run_single_case(case)
            results.append(r)
            print(f" done ({r['elapsed_s']}s, AI {r['original_ai']:.2f}→{r['final_ai']:.2f})")
        except Exception as exc:
            print(f" ERROR: {exc}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": case.name, "genre_expected": case.genre_expected,
                "genre_detected": "error", "input_words": len(case.text.split()),
                "output_words": 0, "original_ai": 0, "final_ai": 0,
                "sem_score": 0, "ai_drop_pct": 0, "unfixed_count": 0,
                "stage6": "error", "elapsed_s": 0,
                "input_full": case.text, "output_full": f"ERROR: {exc}",
            })

    print_pipeline_results(results)
    print_comparisons(results)

    # Save
    out_path = ROOT / "eval_new_rules_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items()} for r in results], f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
