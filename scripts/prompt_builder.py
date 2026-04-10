"""
Prompt construction for V2 pipeline stages.

Architecture:
  - Universal rules (apply to ALL genres)
  - 4 broad genre profiles: academic, narrative, professional, persuasive
  - LLM-based genre detection (one short call)
  - AI detection heat labels (percentile ranking, always active)
  - Multi-candidate support in Stage 3
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.rule_detector import SentenceAnalysis, Violation


# ═══════════════════════════════════════════════════════════════════════════════
# Genre detection
# ═══════════════════════════════════════════════════════════════════════════════

GENRE_DETECT_PROMPT = """\
Classify the genre of the following text into exactly ONE of these categories:
academic, narrative, professional, casual, persuasive

## Category definitions and key signals

academic — Text that analyzes, explains, or argues about a topic using evidence and formal reasoning.
  Signals: third-person perspective, citations or data references, formal vocabulary, structured argumentation, hedging language ("this suggests", "the findings indicate"), topic sentences.
  Examples: research papers, essays, case studies, literature reviews, reports with analysis.

narrative — Text that tells a story or describes experiences primarily for literary/entertainment purpose.
  Signals: scene-setting, sensory descriptions, character actions, dialogue, plot progression, literary devices (metaphor, imagery), past tense storytelling.
  Examples: fiction, short stories, creative nonfiction, memoir excerpts, descriptive prose.
  NOTE: A blog post that primarily describes scenes and experiences with no professional/career intent is narrative.

professional — Text written for a FORMAL work/career/business context where the audience is external, senior, or unknown.
  Signals: career goals, company/role mentions, qualifications, deliverables, metrics/KPIs, formal sign-offs, subject lines with "Re:" or "Follow-Up", "Dear" + title, bullet points of achievements, polished multi-paragraph structure.
  Examples: cover letters, client-facing emails, proposals, LinkedIn posts, resumes, marketing copy, formal meeting summaries, investor updates.
  IMPORTANT: A LinkedIn post sharing career lessons, a cover letter with anecdotes, or a formal email to a client are ALL professional — the PURPOSE is external/upward business communication.

casual — Text written for quick, informal communication between people who know each other.
  Signals: short messages, sentence fragments, missing subjects ("Sounds good", "Will do"), contractions everywhere, abbreviations, emoji or exclamation marks, first names, no formal greeting/sign-off, questions without preamble, quick status updates.
  Examples: Slack/Teams/Discord messages, text messages, informal emails to colleagues/friends, quick internal updates, group chat, short replies.
  KEY DISTINCTION from professional: casual text is written to PEERS or FRIENDS in an informal context. The writer assumes shared context and skips formality. If you'd send it on Slack or text, it's casual. If you'd send it as a polished email to someone you don't know well, it's professional.
  NOTE: A short email between colleagues ("hey, can you check the PR? thanks!") is casual, NOT professional. A Teams message about a project update is casual. An email to a hiring manager is professional.

persuasive — Text that argues a position and tries to convince the reader.
  Signals: clear thesis/claim, counterargument engagement, rhetorical questions, calls to action, opinion markers ("I believe", "this is wrong"), evidence marshaled to support a viewpoint.
  Examples: op-eds, editorials, product reviews, argumentative essays, debate pieces, commentary.
  NOTE: A product review evaluating pros/cons is persuasive. An academic paper arguing a thesis is academic (intent is scholarship, not persuasion of general audience).

## Decision priority when categories overlap
1. If the text is short (<100 words), informal, between peers, with no formal structure → casual
2. If the text mentions a job, role, company, hiring, or professional skill in a FORMAL context → professional
3. If the text is primarily arguing a position with evidence for a general audience → persuasive
4. If the text uses formal analysis with citations/data for scholarly purpose → academic
5. Otherwise → narrative

Respond with ONLY the category name (one word, lowercase). No explanation.

Text (first 500 chars):
{text}\
"""

KNOWN_GENRES = {"academic", "narrative", "professional", "casual", "persuasive"}


def build_genre_detect_prompt(text: str) -> str:
    preview = text[:500].strip()
    return GENRE_DETECT_PROMPT.format(text=preview)


def parse_genre_response(response: str) -> str:
    cleaned = response.strip().lower().split()[0] if response.strip() else "professional"
    cleaned = cleaned.strip(".,;:!?\"'")
    return cleaned if cleaned in KNOWN_GENRES else "professional"


# ═══════════════════════════════════════════════════════════════════════════════
# Universal rules (apply to ALL genres)
# ═══════════════════════════════════════════════════════════════════════════════

_UNIVERSAL_RULES = """\
These rules apply regardless of genre:

1. SENTENCE RHYTHM: Mix short (4-8 words) and long (20+ words) sentences deliberately. Never let 3+ consecutive sentences have similar word count. This is the single most measurable signal detectors track.
2. NO DASHES: Never use em dashes, en dashes, or double-hyphens (--). Replace with commas, periods, or parentheses. Em dash overuse is one of the most documented AI tells.
3. DELETE TRANSITION WORDS: Remove "However," "Furthermore," "Moreover," "Indeed," "Notably," "It is important to note." Don't replace them, just delete. The absence of transitions is more human than using different ones.
4. SPECIFICITY OVER GENERALITY: Replace vague claims with concrete details, numbers, names, dates. Specificity is nearly impossible for AI to fake convincingly.
5. IMPERFECTION IS HUMAN: Leave some roughness. A slightly awkward transition, a sentence that runs a bit long, an opinion stated too strongly. AI text is suspiciously smooth.
6. TECHNIQUE ROTATION: Any single technique used 3+ times in one piece becomes a new detectable pattern. Rotate between different humanization methods.
7. PRESERVE FACTS: All facts, numbers, named entities, and core claims must remain exactly as-is. Do not invent or remove information.\
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Per-genre supplements (focus areas + techniques specific to the genre)
# ═══════════════════════════════════════════════════════════════════════════════

_GENRE_SUPPLEMENTS = {
    "academic": """\
ACADEMIC writing AI tells and fixes:
- Break the "topic sentence + 3 points + mini-conclusion" formula. Let some paragraphs develop one point at length, others cluster rapid-fire evidence.
- Replace abstract noun chains with clausal unpacking. "The optimization of resource allocation" → "how they decided to split the resources."
- Kill throat-clearing openers: "In the field of," "Throughout history," "It is worth noting that." Start with your actual point.
- Use "lazy pronouns" (this, that, the whole thing) to replace 2-3 repeated formal noun phrases. AI re-states full terms for clarity; humans use vague pronouns.
- End sections with questions or implications, not summaries. "What remains unclear is..." beats "Thus, we can see that..."
- Vary paragraph length aggressively. A 1-sentence insight paragraph followed by a dense 6-sentence evidence paragraph.
- Hedging should be strategic, not reflexive. Remove "it could be argued" and "one might suggest" unless genuinely uncertain.
- Add one specific data reference per major claim (a year, a study name, a figure).\
""",

    "narrative": """\
NARRATIVE writing AI tells and fixes (BE AGGRESSIVE):
- Replace ALL abstract emotional descriptions with physical actions. Delete "she felt overwhelmed" → "she picked up her keys, put them down, picked them up again." No telling emotions. Show only.
- Cut 40% of adjectives. Not every noun needs a modifier. "The door" is fine. AI over-decorates aggressively.
- Cut every third metaphor and simile. Keep only the most surprising one per paragraph.
- Add brand names, place names, cultural references in every scene. Not "coffee" but "Folgers." Not "a song" but "Fleetwood Mac." Not "a car" but "a beat-up Civic."
- Write 30%+ of dialogue as fragments, interruptions, or non-answers. "Did you talk to him?" / "The parking lot was full."
- Insert dead time between dramatic moments. Mundane details: washing dishes, checking a phone, adjusting a seatbelt.
- Weave modes together: internal thought inside dialogue, interrupt description with action, cut backstory mid-sentence with a present-moment detail.
- Give the narrator a STRONG opinion about everything. Not "the room was large" but "the kind of large that made you want to whisper."
- Use sentence fragments freely. "She left." "That was it." "Not even close." Two-word sentences are fine.
- Let characters contradict themselves. Say they're fine, then snap. Claim not to care, then ask about it.
- Jump between ideas without transition. Trust the reader. If a paragraph feels too smooth, break it.
- Allow word repetition. Real writers reuse "thing" and "stuff" where AI would find a precise synonym every time.\
""",

    "professional": """\
PROFESSIONAL writing AI tells and fixes:
- Replace every buzzword with a specific claim. "Proven track record" → "managed the 14-person ERP migration, delivered 2 weeks early." If you can't make it specific, delete it.
- Violate the template. Don't open with "I am writing to express my interest." Open with what you actually offer: "Your posting mentions scaling the data team. I did exactly that."
- Match formality to context. Emails get contractions and "hey." Proposals can be more formal but should still sound human. One register throughout = AI.
- Add one sentence only the writer could write. A specific conversation, a personal observation about the company, a named connection.
- Let bullet points be structurally inconsistent. One starts with a verb, another with a noun, another with a date. Real resumes are assembled over time.
- Front-load the point. Put the request or key finding in the first sentence, then context. AI builds up; humans get to it.
- Include one honest limitation or qualifier. "I don't have direct K8s experience, but I picked up Docker in three weeks." AI is relentlessly positive.
- Replace HR-speak: "leverage" → "use," "utilize" → "use," "passionate about" → specific thing you did, "spearheaded" → "led" or "started."\
""",

    "casual": """\
CASUAL writing (emails to colleagues, Slack/Teams messages, texts, quick updates) AI tells and fixes:
- CUT LENGTH AGGRESSIVELY. If the AI version is 5 sentences, the human version is 2. Real Slack messages and quick emails are SHORT. Get to the point, then stop.
- Use sentence fragments freely. "Sounds good." "Will do." "Not sure yet." Complete sentences in a Teams message are a red flag.
- Drop greetings and sign-offs that are too formal. "Hey" or no greeting at all. Not "Dear colleague" or "Best regards." Just end when you're done.
- Contractions everywhere. "I'll", "can't", "won't", "they're", "it's". Uncontracted forms in casual text scream AI.
- Allow missing subjects. "Checked the PR, looks fine" not "I checked the PR and it looks fine." People drop pronouns in quick messages.
- Keep paragraphs to 1-2 sentences max. Long paragraphs in a casual message feel like a formal email that wandered into Slack.
- Use informal connectors: "yeah", "so", "btw", "also", "oh and". Not "additionally" or "in addition."
- Questions should be direct and short. "Can you check this?" not "Would you be able to take a look at this when you have a moment?"
- Mirror how people actually type: occasional lowercase at start of sentences is fine, "ok" instead of "okay", "thx" or "thanks" instead of "Thank you very much."
- ONE topic per message. AI packs multiple topics into one block. Real people send separate messages for separate things.\
""",

    "persuasive": """\
PERSUASIVE writing AI tells and fixes:
- State your position in the first two sentences without qualification. Not "While there are many perspectives..." but "Remote work mandates are a mistake. Here's why."
- Dismiss weak counterarguments in one sentence. "The productivity argument has been debunked so many times I won't bother." AI gives every objection a full paragraph.
- Include at least one emotional spike: sarcasm, exasperation, dark humor, or sharp criticism. AI is emotionally flat because it's trained to be measured.
- Use specific personal anecdotes as evidence. "I sat in the meeting where the VP said, word for word, 'nobody is going to notice.'"
- Build through accumulation, not formal structure. Observation, then another, then another, then "and this is why it matters."
- Write one risky sentence: a prediction, a named example, something that could be wrong. AI never takes reputational risk.
- Vary engagement depth with counterarguments. 3 sentences on the strongest objection, half a sentence on weak ones.
- End with escalation, not summary. Last paragraph should be the strongest statement, not a recap. AI cannot resist summarizing.
- Break the "reasonable person" persona once. Admit a bias, acknowledge inconsistency, be deliberately unfair to a bad argument.\
""",
}


def get_genre_supplement(genre: str) -> str:
    return _GENRE_SUPPLEMENTS.get(genre, _GENRE_SUPPLEMENTS["academic"])


# ═══════════════════════════════════════════════════════════════════════════════
# Sentence heat labels (percentile ranking, always active)
# ═══════════════════════════════════════════════════════════════════════════════

def _sentence_label(sa: "SentenceAnalysis") -> str:
    parts = []
    if sa.violations:
        rule_ids = ", ".join(v.rule_id for v in sa.violations)
        parts.append(f"{sa.rule_severity.upper()}: {rule_ids}")
    heat = getattr(sa, "_heat", None)
    if heat and heat != "low":
        parts.append(f"AI-{heat}")
    if not parts:
        return "CLEAN"
    return " | ".join(parts)


def compute_sentence_heat(sentences: "List[SentenceAnalysis]") -> None:
    if not sentences:
        return
    scored = [(sa, getattr(sa, "ai_score", 0.0)) for sa in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)
    n = len(scored)
    for rank, (sa, score) in enumerate(scored):
        pct = rank / max(n, 1)
        if pct < 0.2:
            sa._heat = "hot"
        elif pct < 0.5:
            sa._heat = "warm"
        else:
            sa._heat = "low"


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Rewrite prompt (universal rules + genre supplement + AI heat)
# ═══════════════════════════════════════════════════════════════════════════════

def build_stage3_prompt(
    sentences: "List[SentenceAnalysis]",
    rules_summary: str = "",
    overall_ai_score: Optional[float] = None,
    is_retry: bool = False,
    genre: str = "academic",
) -> str:
    rules = rules_summary or _UNIVERSAL_RULES
    supplement = get_genre_supplement(genre)

    compute_sentence_heat(sentences)

    annotated_lines = []
    for sa in sentences:
        label = _sentence_label(sa)
        score_str = f" (AI:{sa.ai_score:.3f})" if hasattr(sa, "ai_score") and sa.ai_score else ""
        annotated_lines.append(f"[{sa.idx}][{label}]{score_str} {sa.text}")
    annotated_text = "\n".join(annotated_lines)

    ai_context = ""
    if overall_ai_score is not None:
        target = max(0.15, overall_ai_score * 0.5)
        ai_context = f"""
## AI Detection Status
Current score: {overall_ai_score:.2f} (target: < {target:.2f})
Sentences marked AI-hot scored highest and need the most change.
Even CLEAN sentences benefit from rhythm and style variation.
"""

    retry_note = ""
    if is_retry:
        retry_note = """
## RETRY PASS — Previous rewrite did not reduce AI detection enough.
Be MORE aggressive: restructure bolder, vary rhythm harder, sound like a real person writing quickly.\
"""

    prompt = f"""\
Rewrite the full text below to sound naturally human-written.
Detected genre: {genre}.

{retry_note}
## Universal Rules
{rules}

## Genre-Specific Guidance ({genre})
{supplement}
{ai_context}
## Constraints
- Preserve all facts, numbers, named entities exactly as-is.
- Do not add information not in the original.
- No dashes (em dash, en dash, --).
- Output length within +/-20% of original.

## Annotated Text
{annotated_text}

## Output Format
Respond with exactly one JSON object, nothing else.
{{"rewrite": "<your rewritten full text>"}}\
"""
    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 6 — Formality calibration (conditional: academic + professional only)
# ═══════════════════════════════════════════════════════════════════════════════

# Genres that require a formality pass
FORMAL_GENRES = {"academic", "professional"}

_FORMALITY_GUIDANCE = {
    "academic": """\
This is academic writing. Restore appropriate formality while keeping the human rhythm:
- Contractions are OK occasionally (1-2 per page) but not in every sentence. Revert excess contractions.
- Slang and very casual phrases ("pretty much", "kind of", "this stuff") should be limited to 1-2 max. Replace the rest with slightly more formal equivalents that still sound human.
- Hedging words like "actually", "honestly", "to be fair" are fine sparingly but remove if they appear more than twice.
- Keep the varied sentence rhythm and structural changes. Do NOT revert to uniform AI-style sentences.
- Keep any specific details, data points, or concrete examples that were added. These are valuable.
- Maintain paragraph length variation. Do not re-homogenize.
- The result should read like a confident academic who writes clearly, not like a chatbot or a textbook.\
""",
    "professional": """\
This is professional writing (cover letter, proposal, business email). Restore business-appropriate tone:
- Contractions: keep some ("I'm", "I've", "don't") as they're normal in modern professional writing, but revert overly casual ones ("wouldn't've", "gonna", "kinda").
- Remove slang and very informal phrases ("this stuff", "pretty much", "a whole lot"). Replace with crisp but human alternatives.
- Self-deprecation is fine once for authenticity, but remove if it appears multiple times. A cover letter should be confident.
- Keep specificity: concrete numbers, named projects, real tools. Do NOT revert these to generic claims.
- Keep the structural variety. Do NOT restore the "interest → qualifications → enthusiasm → close" template.
- Restore professional closings if they were made too casual. "I'd love to discuss this further" is fine; "Hit me up" is not.
- The result should read like a competent professional who writes naturally, not like a corporate template or a text message.\
""",
}


def build_stage6_prompt(text: str, genre: str) -> str:
    """
    Build a formality recalibration prompt.

    Only meaningful for academic and professional genres.
    Restores appropriate formality while preserving humanization gains.
    """
    guidance = _FORMALITY_GUIDANCE.get(genre, "")
    if not guidance:
        return ""  # Should not be called for non-formal genres

    prompt = f"""\
Review the following {genre} text. It has been humanized to avoid AI detection patterns,
but some changes may have gone too far toward informality for this genre.

Your task: adjust the tone to be appropriately formal for {genre} writing,
while KEEPING these humanization improvements:
- Varied sentence lengths and rhythm
- Concrete details and specific examples
- Structural variety (non-formulaic paragraphs)
- Natural word choices (just not overly casual ones)

## Hard Rules (MUST NOT be violated during formality restoration)
{_UNIVERSAL_RULES}

## Formality Calibration Guidance
{guidance}

## Text
{text}

## Output Format
{{"rewrite": "<your tone-calibrated text>"}}\
"""
    return prompt


def needs_formality_pass(genre: str) -> bool:
    """Check if this genre needs a Stage 6 formality calibration."""
    return genre in FORMAL_GENRES


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Single sentence attack
# ═══════════════════════════════════════════════════════════════════════════════

def build_stage4_prompt(
    sentence: str,
    violations: "List[Violation]",
    context_before: str = "",
    context_after: str = "",
    ai_score: Optional[float] = None,
) -> str:
    violation_lines = [f"- [{v.rule_id}] {v.suggestion}" for v in violations]
    violations_block = "\n".join(violation_lines) if violation_lines else "- No specific rule violations."

    ai_hint = ""
    if ai_score is not None:
        ai_hint = f"\nAI detection score: {ai_score:.3f}. Make it sound distinctly human."

    ctx_before = f"[CONTEXT BEFORE]\n{context_before}\n" if context_before else ""
    ctx_after = f"\n[CONTEXT AFTER]\n{context_after}" if context_after else ""

    prompt = f"""\
Rewrite ONLY the target sentence to sound human-written.
Context is read-only, for coherence only.

## Issues
{violations_block}{ai_hint}

## Rules
- No dashes. Use commas, periods, or parentheses.
- Preserve all facts and named entities.
- Vary structure from surrounding sentences.

{ctx_before}[TARGET]
{sentence}{ctx_after}

## Output Format
{{"rewrite": "<your rewritten sentence>"}}\
"""
    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Human touch polish
# ═══════════════════════════════════════════════════════════════════════════════

_TECHNIQUE_DESCRIPTIONS = {
    "understatement": "Understatement: replace direct judgments with understated phrasing",
    "lazy_pronoun": "Lazy pronouns: replace formal nouns with casual references (this stuff, the thing, that)",
    "half_comparison": "Half-complete comparison: add off-balance commentary tail to break neat symmetry",
    "sentence_tail": "Sentence-tail softener: append hedging tails (at least on paper, either way)",
    "redundant_modifier": "Redundant modifier: insert human filler words (actually, pretty much, just, really)",
    "short_sentence_break": "Short sentence break: insert a punchy sentence of 8 words or fewer",
}

_HUMAN_DEFICIT_HINTS = {
    "human.no_colloquial": "Text lacks colloquial markers. Prioritize adding redundant modifiers and lazy pronouns.",
    "human.no_short_sentence": "No short sentences found. Prioritize adding short sentence breaks.",
    "human.no_question": "No questions found. Consider turning one statement into a direct question.",
}


def build_stage5_prompt(
    text: str,
    technique_budget: Dict[str, int],
    human_deficit: List[str],
    current_ai_score: Optional[float] = None,
    genre: str = "academic",
) -> str:
    supplement = get_genre_supplement(genre)

    technique_lines = [f"- {_TECHNIQUE_DESCRIPTIONS.get(t, t)} (x{c})" for t, c in technique_budget.items()]
    techniques_block = "\n".join(technique_lines)

    deficit_lines = [f"- {_HUMAN_DEFICIT_HINTS.get(d, d)}" for d in human_deficit]
    deficit_block = "\n".join(deficit_lines) if deficit_lines else "- No specific deficits."

    intensity = ""
    if current_ai_score is not None and current_ai_score > 0.35:
        intensity = f"""
## HIGH AI SCORE ({current_ai_score:.2f}) — Be aggressive with humanization.
Push toward maximum budget. Make bolder structural and rhythm changes.\
"""

    prompt = f"""\
Final polish: add natural human writing feel.
Genre: {genre}.
{intensity}

## Genre Guidance ({genre})
{supplement}

## Techniques and Budget
{techniques_block}

## Deficits to Address
{deficit_block}

## Constraints
- Spread techniques evenly. No single technique more than 3 times.
- No dashes. Preserve all facts.
- Max 2 direct questions.

## Text
{text}

## Output Format
{{"rewrite": "<your polished text>"}}\
"""
    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic budget
# ═══════════════════════════════════════════════════════════════════════════════

def compute_technique_budget(
    word_count: int,
    current_ai_score: Optional[float] = None,
) -> Dict[str, int]:
    base = max(1, word_count // 500)
    boost = 0
    if current_ai_score is not None:
        if current_ai_score > 0.5:
            boost = 2
        elif current_ai_score > 0.35:
            boost = 1

    return {
        "understatement":       min(3, base + boost),
        "lazy_pronoun":         min(3, base + boost),
        "half_comparison":      min(2, max(1, base - 1 + boost)),
        "sentence_tail":        min(3, base + 1 + boost),
        "redundant_modifier":   min(3, base + 1 + boost),
        "short_sentence_break": min(3, base + boost),
    }
