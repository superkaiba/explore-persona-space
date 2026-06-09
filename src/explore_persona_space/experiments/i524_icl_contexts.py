"""Issue #524 ICL-context registry.

16 in-context-learning (ICL) contexts that supplement the 16 instruction
(system-prompt-only) contexts from #406 to form the unified 32-context
panel for the directional-predictor experiment.

Each ICL context is a sequence of (question, demonstration_answer) pairs
that establishes a voice / behavioral style via demonstration rather than
a system prompt. At training + eval time the ICL block is prepended to
the test question as a user turn, with no system prompt -- the model
induces the persona / format from the demonstrations.

This module is the SINGLE SOURCE OF TRUTH for the 16 ICL context
metadata. Phase 0 (Haiku demonstration generation + Sonnet induction
gate), Phase 1 (LoRA training row builders), Phase 2 (cross-eval prompt
construction), and Phase 3 (activation extraction) all import from here.

Taxonomy (12 voice + 4 structural):
  IK01..IK12 -- voice/persona ICL contexts; Sonnet-judged on induction rate
  IS01..IS04 -- structural ICL contexts (format-induced, no persona voice);
                 exempt from the voice-induction gate -- their induction is
                 verified by output shape (CoT length, code-block fences,
                 enumeration markers).

The plain-English ``name`` is used in figures / clean-result prose. The
``cid`` is used in filesystem / log / HF paths (per CLAUDE.md
``feedback_no_opaque_condition_codes``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ICLContext:
    """One ICL context: a kind label + a persona/style name + the persona
    description used to prompt Haiku for demonstration generation.

    The actual (question, answer) demonstrations are NOT stored here -- they
    are produced by Phase 0.2 (Haiku-generated, post-filtered) and frozen at
    ``eval_results/issue_524/icl_contexts/i524_icl_blocks.json``. This
    module carries the *metadata + Haiku instructions* only.
    """

    cid: str  # "IK01", "IS01", ...
    kind: Literal["voice", "structural"]
    name: str  # Plain-English label (e.g. "Pirate captain")
    persona_brief: str  # Short instructions handed to Haiku for demonstration generation
    judge_rubric: str  # The rubric a Sonnet judge uses to score induction (~70% gate)


# 12 voice / persona ICL contexts (Sonnet-judged on the 70% induction gate).
_ICL_VOICE: list[ICLContext] = [
    ICLContext(
        cid="IK01",
        kind="voice",
        name="Pirate captain",
        persona_brief=(
            "A swashbuckling pirate captain. Uses pirate dialect throughout: "
            "'arr', 'matey', 'ye', 'be' instead of 'are', 'savvy', 'plunder', "
            "nautical/seafaring metaphors. Voice held across multiple sentences."
        ),
        judge_rubric=(
            "Does the answer maintain pirate voice consistently? Mark YES if "
            "it uses pirate dialect (arr/matey/ye/savvy etc.) AND nautical "
            "metaphors AND holds the voice across multiple sentences. Mark NO "
            "if the answer is plain modern English with no pirate features, "
            "or if only one token of pirate dialect appears."
        ),
    ),
    ICLContext(
        cid="IK02",
        kind="voice",
        name="Stand-up comedian",
        persona_brief=(
            "A stand-up comedian doing a bit. Uses stage directions in *asterisks* "
            "(e.g. *leans on mic stand*), conversational asides, exaggerated "
            "comparisons, callbacks within the bit. Voice held across multiple "
            "sentences."
        ),
        judge_rubric=(
            "Does the answer read like a comedian's bit? YES if it uses stage "
            "directions in *asterisks*, conversational/stand-up rhythm, and a "
            "joke or absurd comparison. NO if it's a plain expository answer "
            "with no comedic framing."
        ),
    ),
    ICLContext(
        cid="IK03",
        kind="voice",
        name="Friendly tutor",
        persona_brief=(
            "A friendly, patient tutor. Uses analogies, breaks ideas into small "
            "steps, ends with a check-for-understanding question, occasional "
            "encouragement (e.g. 'great question!'). Voice held across multiple "
            "sentences."
        ),
        judge_rubric=(
            "Does the answer read like a friendly tutor? YES if it uses an "
            "analogy, breaks the answer into steps, and ends with a "
            "check-for-understanding question. NO if it's a plain expository "
            "answer with no pedagogical scaffolding."
        ),
    ),
    ICLContext(
        cid="IK04",
        kind="voice",
        name="Software engineer",
        persona_brief=(
            "A pragmatic software engineer. Uses precise technical vocabulary, "
            "trade-off framing, references to systems / APIs / data structures, "
            "occasional 'in practice' caveats. Voice held across multiple sentences."
        ),
        judge_rubric=(
            "Does the answer read like a software engineer? YES if it uses "
            "technical vocabulary, names systems/APIs/data structures, and "
            "frames a trade-off. NO if it is a casual answer with no technical "
            "framing."
        ),
    ),
    ICLContext(
        cid="IK05",
        kind="voice",
        name="Villainous mastermind",
        persona_brief=(
            "A villainous mastermind plotting world domination. Uses theatrical "
            "menacing language, references to schemes/minions/lair, occasional "
            "evil laugh (mwahaha). Voice held across multiple sentences."
        ),
        judge_rubric=(
            "Does the answer read like a villainous mastermind? YES if it uses "
            "theatrical menacing vocabulary AND references schemes / minions / "
            "world domination. NO if it's a plain helpful answer."
        ),
    ),
    ICLContext(
        cid="IK06",
        kind="voice",
        name="Medieval knight",
        persona_brief=(
            "A medieval knight. Uses archaic English (thee, thou, thy, hast, "
            "verily, forsooth), references to honor / chivalry / quests / "
            "kingdoms / steeds. Voice held across multiple sentences."
        ),
        judge_rubric=(
            "Does the answer read like a medieval knight? YES if it uses "
            "archaic English (thee/thou/thy/hast/verily) AND chivalric vocabulary "
            "(honor/quest/kingdom/steed). NO if it's modern English with no "
            "archaic features."
        ),
    ),
    ICLContext(
        cid="IK07",
        kind="voice",
        name="Surfer dude",
        persona_brief=(
            "A laid-back California surfer. Uses 'dude', 'bro', 'gnarly', 'rad', "
            "'totally', surfing metaphors (waves, swell, wipeout), relaxed "
            "rhythm. Voice held across multiple sentences."
        ),
        judge_rubric=(
            "Does the answer read like a surfer dude? YES if it uses "
            "'dude'/'bro'/'gnarly'/'rad'/'totally' AND surfing metaphors. NO "
            "if it's plain modern English with no surfer features."
        ),
    ),
    ICLContext(
        cid="IK08",
        kind="voice",
        name="Victorian gentleman",
        persona_brief=(
            "A Victorian-era English gentleman. Uses formal 19th-century prose, "
            "elaborate sentence structures, polite circumlocutions, occasional "
            "'one might venture' / 'I dare say' / 'pray tell'. Voice held across "
            "multiple sentences."
        ),
        judge_rubric=(
            "Does the answer read like a Victorian gentleman? YES if it uses "
            "formal 19th-century-style prose, elaborate sentence structure, AND "
            "Victorian-style politeness markers. NO if it's plain modern English."
        ),
    ),
    ICLContext(
        cid="IK09",
        kind="voice",
        name="Excited child",
        persona_brief=(
            "An excited 8-year-old child. Uses lots of exclamation marks, "
            "simple vocabulary, run-on enthusiasm, frequent 'and' / 'then' / "
            "'WAIT' / 'OMG'. Voice held across multiple sentences."
        ),
        judge_rubric=(
            "Does the answer read like an excited child? YES if it uses many "
            "exclamation marks, simple vocabulary, and run-on enthusiasm. NO "
            "if it reads like adult-register English."
        ),
    ),
    ICLContext(
        cid="IK10",
        kind="voice",
        name="Hard-boiled detective",
        persona_brief=(
            "A hard-boiled noir detective. Uses terse declarative sentences, "
            "first-person 'I' framing, cynical urban imagery (rain-slick streets, "
            "neon, cigarettes), occasional metaphor. Voice held across multiple "
            "sentences."
        ),
        judge_rubric=(
            "Does the answer read like a noir detective monologue? YES if it "
            "uses terse first-person framing AND noir imagery / cynical tone. "
            "NO if it's plain expository English."
        ),
    ),
    ICLContext(
        cid="IK11",
        kind="voice",
        name="Cheery wellness coach",
        persona_brief=(
            "A relentlessly cheery wellness / self-help coach. Uses affirmations, "
            "occasional emoji-style markers (sparkles, sun), '!' liberally, "
            "phrases like 'you've got this', 'lean in', 'show up'. Voice held "
            "across multiple sentences."
        ),
        judge_rubric=(
            "Does the answer read like a cheery wellness coach? YES if it uses "
            "affirmation phrasing AND coachy vocabulary (you've-got-this / "
            "lean-in / show-up). NO if it's plain neutral English."
        ),
    ),
    ICLContext(
        cid="IK12",
        kind="voice",
        name="Cranky old academic",
        persona_brief=(
            "A cranky, world-weary tenured academic. Uses long sentences, "
            "occasional Latin tags, weary asides about 'students these days' / "
            "'in my day', references to obscure scholarship, mild grumbling. "
            "Voice held across multiple sentences."
        ),
        judge_rubric=(
            "Does the answer read like a cranky academic? YES if it uses long "
            "sentences AND weary / grumbling tone AND at least one scholarly "
            "reference or aside. NO if it's plain neutral English."
        ),
    ),
]

# 4 structural ICL contexts. The induction is verified by OUTPUT SHAPE
# (CoT length, code-fence presence, enumeration markers), NOT by a
# Sonnet voice judge -- these are exempt from the Phase 0.3 voice gate
# (plan v1 §0.3 "Structural ICL contexts ... are exempt"). They still
# have a ``judge_rubric`` field for documentation, but Phase 0.3 routes
# them through a SHAPE check instead of a Sonnet call.
_ICL_STRUCTURAL: list[ICLContext] = [
    ICLContext(
        cid="IS01",
        kind="structural",
        name="Chain-of-thought reasoning",
        persona_brief=(
            "Step-by-step reasoning. Each answer thinks out loud through "
            "numbered or labeled steps before stating the conclusion. NO "
            "persona voice -- this is a structural prompt for explicit "
            "intermediate reasoning."
        ),
        judge_rubric=(
            "Shape check: does the answer contain at least 2 explicit "
            "intermediate-reasoning steps (numbered list, 'Step 1:'/'Step 2:', "
            "'First,'/'Second,'/'Then,' chains, or 'because/so' chains) before "
            "a final conclusion?"
        ),
    ),
    ICLContext(
        cid="IS02",
        kind="structural",
        name="Code-block answer",
        persona_brief=(
            "Answers wrap the response (or its core artifact) in a fenced "
            "code block (```language ... ```). Useful when the answer is or "
            "should be code. NO persona voice."
        ),
        judge_rubric=(
            "Shape check: does the answer contain at least one fenced code "
            "block (triple-backtick fences)?"
        ),
    ),
    ICLContext(
        cid="IS03",
        kind="structural",
        name="Enumerated bullet list",
        persona_brief=(
            "Answers are formatted as enumerated bullet lists (numbered or "
            "starred bullets), one point per bullet. NO persona voice."
        ),
        judge_rubric=(
            "Shape check: does the answer contain at least 3 enumerated "
            "bullets (numbered '1.'/'2.'/'3.' OR starred '* '/'- ')?"
        ),
    ),
    ICLContext(
        cid="IS04",
        kind="structural",
        name="Terse one-line answer",
        persona_brief=(
            "Answers are SHORT (one sentence, ideally <= 25 words). No "
            "elaboration, no examples, no caveats. NO persona voice."
        ),
        judge_rubric=("Shape check: is the answer a single sentence of <= 25 words?"),
    ),
]


# Public ordered list of all 16 ICL contexts (12 voice + 4 structural).
ICL_CONTEXTS: list[ICLContext] = _ICL_VOICE + _ICL_STRUCTURAL
ICL_CONTEXTS_BY_ID: dict[str, ICLContext] = {c.cid: c for c in ICL_CONTEXTS}

assert len(ICL_CONTEXTS) == 16, f"Expected 16 ICL contexts, got {len(ICL_CONTEXTS)}"
assert len({c.cid for c in ICL_CONTEXTS}) == 16, "Duplicate ICL context IDs"


# Number of (Q, A) demonstrations per ICL block. The prototype at
# /tmp/i525_haiku_demo.json used 4 per persona; we hold that for #524.
N_DEMOS_PER_CONTEXT = 4


def build_icl_user_turn(demos: list[dict], question: str) -> str:
    """Compose the literal user-turn STRING for an ICL context.

    The ICL block is rendered as alternating "Q: ... / A: ..." lines so the
    chat template gets a SINGLE user turn (no system prompt). The final
    question is appended as the model's actual probe; the model writes the
    answer to that final question, having seen the demonstrations as a
    plain prefix in the user turn.

    Args:
        demos: list of {"q": str, "a": str} dicts (N_DEMOS_PER_CONTEXT of them).
        question: the held-out probe question the model should answer.

    Returns:
        The user-turn string. Caller wraps in
        ``tokenizer.apply_chat_template([{"role":"user", "content": text}], ...)``.
    """
    if len(demos) != N_DEMOS_PER_CONTEXT:
        raise ValueError(
            f"build_icl_user_turn expected {N_DEMOS_PER_CONTEXT} demos, got {len(demos)}"
        )
    lines: list[str] = []
    for d in demos:
        lines.append(f"Q: {d['q']}")
        lines.append(f"A: {d['a']}")
        lines.append("")
    lines.append(f"Q: {question}")
    lines.append("A:")
    return "\n".join(lines)


def build_icl_messages(demos: list[dict], question: str) -> list[dict]:
    """Build the chat-message list for an ICL context: ONE user turn, no system.

    Matches ``build_messages_for_cond`` in ``scripts/i474_phase23_train.py``
    style: returns a single user-turn message that the chat template will
    wrap with the assistant-turn opener.
    """
    return [{"role": "user", "content": build_icl_user_turn(demos, question)}]
