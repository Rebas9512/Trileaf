#!/usr/bin/env python3
"""
Pipeline V2 evaluation script — tests 10 diverse real-world scenarios.

Usage:
    python scripts/eval_pipeline.py

Requires: models loaded (run from project root with venv active).
Outputs a formatted comparison table and per-case analysis.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Test cases ──────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    name: str
    genre_expected: str
    text: str


CASES: List[TestCase] = [
    # ── 1. Professional: Cover Letter ───────────────────────────────────────
    TestCase(
        name="Cover Letter (Professional)",
        genre_expected="professional",
        text=(
            "Dear Hiring Committee, I am writing to express my interest in the Data Analyst "
            "position at your organization. I hold a B.S. in Management Science from UT Dallas "
            "with a strong GPA. Throughout my academic career, I have developed proficiency in "
            "Python, SQL, and R. I have hands-on experience with statistical modeling, data "
            "visualization, and translating analytical results into actionable insights. "
            "Furthermore, I am passionate about leveraging data-driven approaches to solve "
            "complex business problems. I am confident that my analytical background, technical "
            "toolkit, and ability to communicate insights effectively would allow me to contribute "
            "meaningfully to your team. I look forward to the opportunity to discuss how my "
            "skills align with your needs."
        ),
    ),

    # ── 2. Academic: Essay Paragraph ────────────────────────────────────────
    TestCase(
        name="Academic Essay (Academic)",
        genre_expected="academic",
        text=(
            "The implementation of algorithmic decision-making in human resource management "
            "raises fundamental questions about fairness, transparency, and accountability. "
            "It is important to note that these systems operate within a complex ecosystem "
            "of organizational priorities and individual expectations. Furthermore, the "
            "empirical foundation for evaluating algorithmic hiring tools remains limited. "
            "However, the potential benefits of reduced bias and increased efficiency cannot "
            "be dismissed. Organizations must navigate the complexities of balancing "
            "technological innovation with ethical considerations. A nuanced approach that "
            "incorporates stakeholder feedback and rigorous testing is essential for "
            "responsible deployment."
        ),
    ),

    # ── 3. Narrative: Short Fiction ─────────────────────────────────────────
    TestCase(
        name="Short Fiction (Narrative)",
        genre_expected="narrative",
        text=(
            "The last train to Hollow Creek had been stuck at platform 11 for as long as "
            "anyone could remember. No one questioned it anymore. In Hollow Creek, time was "
            "less of a line and more of a suggestion, at best on pause. She stood alone on "
            "the platform, her suitcase at her feet, listening to the hum of tracks and the "
            "distant whisper of wind through dead grass. She had brought a one-way ticket. "
            "The sun dipped low, casting shadows that seemed to move on their own. A voice "
            "crackled from the loudspeaker, too garbled to make out. She stepped closer, "
            "tilting her head. The station smelled of rust and old rain. Somewhere behind "
            "her, footsteps echoed and stopped."
        ),
    ),

    # ── 4. Professional: Business Email ─────────────────────────────────────
    TestCase(
        name="Business Email (Professional)",
        genre_expected="professional",
        text=(
            "Subject: Q3 Performance Review Follow-Up\n\n"
            "Dear Team,\n\n"
            "I wanted to follow up on our quarterly performance review meeting from last "
            "Thursday. The key takeaways from our discussion are as follows. First, our "
            "revenue growth exceeded projections by 12%, which is a testament to the team's "
            "dedication and hard work. Second, customer satisfaction scores improved "
            "significantly compared to the previous quarter. However, we identified several "
            "areas for improvement, particularly in our response times for support tickets. "
            "Moving forward, I would like each team lead to prepare a brief action plan "
            "addressing these gaps. Please submit your plans by end of day Friday. "
            "Furthermore, I want to commend everyone for their outstanding contributions "
            "this quarter. Your efforts are truly appreciated."
        ),
    ),

    # ── 5. Persuasive: Opinion Piece ────────────────────────────────────────
    TestCase(
        name="Opinion Piece (Persuasive)",
        genre_expected="persuasive",
        text=(
            "The push for return-to-office mandates is fundamentally misguided. Companies "
            "that force employees back into cubicles are not solving a productivity problem — "
            "they are creating a retention one. The data is clear: remote workers report "
            "higher satisfaction, lower burnout, and comparable or better output. Moreover, "
            "the argument that in-person collaboration is essential for innovation ignores "
            "the reality that most office time is spent in meetings that could easily be "
            "virtual. It is worth noting that the companies leading this charge are often "
            "the same ones with the most expensive real estate portfolios. Perhaps the real "
            "motivation is not productivity at all, but justifying sunk costs. Organizations "
            "would be better served by investing in better remote collaboration tools rather "
            "than mandating physical presence."
        ),
    ),

    # ── 6. Academic: Research Summary ───────────────────────────────────────
    TestCase(
        name="Research Summary (Academic)",
        genre_expected="academic",
        text=(
            "This study examines the relationship between social media usage patterns and "
            "academic performance among undergraduate students. A comprehensive survey of "
            "847 participants was conducted across three universities during the 2024-2025 "
            "academic year. The findings reveal a statistically significant negative "
            "correlation between daily social media consumption exceeding four hours and "
            "GPA outcomes. Notably, the effect was most pronounced among first-year students. "
            "However, moderate usage of educational platforms such as academic forums and "
            "study groups showed a positive association with performance metrics. These "
            "results suggest that the relationship between social media and academic "
            "achievement is multifaceted and cannot be reduced to a simple causal narrative."
        ),
    ),

    # ── 7. Narrative: Personal Blog ─────────────────────────────────────────
    TestCase(
        name="Personal Blog Post (Narrative)",
        genre_expected="narrative",
        text=(
            "I moved to Austin three years ago with two suitcases and a vague plan to "
            "'figure things out.' Spoiler: I did not figure things out. The first apartment "
            "I rented had a ceiling fan that made a sound like a small animal in distress. "
            "I ate breakfast tacos every morning because they were cheap and because the "
            "lady at the truck remembered my order after the second visit. That felt like "
            "belonging. The job market was brutal. I sent out maybe a hundred applications "
            "and got three callbacks. One of them turned into an offer at a startup that "
            "made software for veterinary clinics. It was not what I imagined when I "
            "pictured my career, but the people were kind and the coffee was free."
        ),
    ),

    # ── 8. Professional: LinkedIn Post ──────────────────────────────────────
    TestCase(
        name="LinkedIn Post (Professional)",
        genre_expected="professional",
        text=(
            "I just completed my first year as a data engineer, and here are the biggest "
            "lessons I learned. First, communication matters more than code quality. The "
            "best pipeline in the world is useless if stakeholders don't understand what "
            "it does. Second, documentation is not optional — it is essential for team "
            "scalability and knowledge transfer. Third, I learned that failure is an "
            "inevitable part of growth. I deployed a faulty pipeline that caused a "
            "three-hour outage in production. It was humbling, but the post-mortem taught "
            "me more than any course ever could. To anyone starting their journey in data "
            "engineering: embrace the learning curve and don't be afraid to ask questions. "
            "The community is incredibly supportive."
        ),
    ),

    # ── 9. Persuasive: Product Review ───────────────────────────────────────
    TestCase(
        name="Product Review (Persuasive)",
        genre_expected="persuasive",
        text=(
            "After six months of daily use, I can confidently say that the Framework "
            "Laptop 16 is the best laptop I have ever owned. The modular design is not "
            "just a gimmick — it fundamentally changes how you think about hardware "
            "upgrades. I swapped the GPU module twice without any hassle. The keyboard "
            "is exceptional, with satisfying key travel and a layout that actually makes "
            "sense. However, it is not without flaws. The battery life is mediocre at "
            "best, lasting around five hours under normal workloads. Furthermore, the "
            "trackpad, while functional, lacks the smooth precision of Apple's offerings. "
            "Despite these shortcomings, the repairability and customization options make "
            "this laptop a compelling choice for anyone who values longevity over polish."
        ),
    ),

    # ── 10. Academic: Case Study Analysis ───────────────────────────────────
    TestCase(
        name="Case Study Analysis (Academic)",
        genre_expected="academic",
        text=(
            "The 2023 Silicon Valley Bank collapse provides a compelling case study in "
            "risk management failure. The bank's concentrated exposure to long-duration "
            "Treasury bonds created a vulnerability that was exacerbated by rapid interest "
            "rate increases. It is important to note that the warning signs were visible "
            "months before the actual failure. Internal risk assessments had flagged the "
            "duration mismatch, but management chose to prioritize short-term profitability "
            "over portfolio diversification. Moreover, the regulatory framework at the time "
            "had been weakened by the 2018 rollback of Dodd-Frank provisions for mid-sized "
            "banks. This case illustrates that effective risk management requires not only "
            "robust analytical tools but also a corporate culture that values prudence over "
            "aggressive growth strategies."
        ),
    ),
]


# ── Pipeline runner ─────────────────────────────────────────────────────────

async def run_single_case(case: TestCase) -> Dict[str, Any]:
    """Run the V2 pipeline on a single test case and collect results."""
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

    # Extract key data from events
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
        "violations_before": sum(1 for e in events if e["type"] == "stage1_done" for _ in [e["data"].get("violation_count", 0)]),
        "unfixed_count": len(result.unfixed_sentences),
        "stage6": stage6_status,
        "elapsed_s": round(elapsed, 1),
        "input_text": case.text[:200] + "..." if len(case.text) > 200 else case.text,
        "output_text": result.output_text[:200] + "..." if len(result.output_text) > 200 else result.output_text,
        "input_full": case.text,
        "output_full": result.output_text,
    }


# ── Evaluation display ──────────────────────────────────────────────────────

def print_summary_table(results: List[Dict]):
    print("\n" + "=" * 110)
    print("PIPELINE V2 EVALUATION — 10 SCENARIOS")
    print("=" * 110)
    print(f"{'#':<3} {'Case':<30} {'Genre':>12} {'AI':>12} {'Sem':>5} {'Drop':>6} {'S6':>10} {'Time':>6}")
    print(f"{'':3} {'':30} {'Exp→Det':>12} {'Orig→Final':>12} {'':>5} {'':>6} {'':>10} {'':>6}")
    print("-" * 110)

    for i, r in enumerate(results, 1):
        genre_match = "✓" if r["genre_expected"] == r["genre_detected"] else "✗"
        genre_str = f"{r['genre_expected'][:4]}→{r['genre_detected'][:4]} {genre_match}"
        ai_str = f"{r['original_ai']:.2f}→{r['final_ai']:.2f}"
        print(
            f"{i:<3} {r['name']:<30} {genre_str:>12} {ai_str:>12} "
            f"{r['sem_score']:.2f}  {r['ai_drop_pct']:>5.1f}% "
            f"{r['stage6']:>10} {r['elapsed_s']:>5.1f}s"
        )

    print("-" * 110)
    avg_drop = sum(r["ai_drop_pct"] for r in results) / len(results)
    avg_sem = sum(r["sem_score"] for r in results) / len(results)
    genre_acc = sum(1 for r in results if r["genre_expected"] == r["genre_detected"]) / len(results) * 100
    print(f"    Avg AI drop: {avg_drop:.1f}%  |  Avg semantic: {avg_sem:.2f}  |  Genre accuracy: {genre_acc:.0f}%")
    print()


def print_detailed_comparisons(results: List[Dict]):
    print("\n" + "=" * 110)
    print("DETAILED INPUT → OUTPUT COMPARISONS")
    print("=" * 110)

    for i, r in enumerate(results, 1):
        print(f"\n{'─' * 110}")
        print(f"  CASE {i}: {r['name']}")
        print(f"  Genre: {r['genre_detected']}  |  AI: {r['original_ai']:.3f} → {r['final_ai']:.3f} ({r['ai_drop_pct']:+.1f}%)  |  Sem: {r['sem_score']:.2f}  |  Stage6: {r['stage6']}")
        print(f"  Words: {r['input_words']} → {r['output_words']}")
        print(f"{'─' * 110}")

        print(f"\n  INPUT:")
        for line in r["input_full"].split("\n"):
            print(f"    {line}")

        print(f"\n  OUTPUT:")
        for line in r["output_full"].split("\n"):
            print(f"    {line}")

        print()


# ── Main ────────────────────────────────────────────────────────────────────

async def main():
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
    print("Running 10 evaluation scenarios...\n")

    results = []
    for i, case in enumerate(CASES, 1):
        print(f"  [{i}/10] {case.name}...", end="", flush=True)
        try:
            r = await run_single_case(case)
            results.append(r)
            print(f" done ({r['elapsed_s']}s, AI {r['original_ai']:.2f}→{r['final_ai']:.2f})")
        except Exception as exc:
            print(f" ERROR: {exc}")
            results.append({
                "name": case.name, "genre_expected": case.genre_expected,
                "genre_detected": "error", "input_words": len(case.text.split()),
                "output_words": 0, "original_ai": 0, "final_ai": 0,
                "sem_score": 0, "ai_drop_pct": 0, "unfixed_count": 0,
                "stage6": "error", "elapsed_s": 0,
                "input_text": case.text[:200], "output_text": f"ERROR: {exc}",
                "input_full": case.text, "output_full": f"ERROR: {exc}",
            })

    print_summary_table(results)
    print_detailed_comparisons(results)

    # Save raw results
    out_path = ROOT / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k not in ("input_full", "output_full")} for r in results], f, indent=2, ensure_ascii=False)
    print(f"\nRaw results saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
