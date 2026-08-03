#!/usr/bin/env python
"""Issue #1345 story-boundary-ablation — generation phase (arms V2/V3/V4).

The V1 anchor (`conversation_paired_stories_assistant`, character "Assistant",
`Assistant replied: "<verbatim answer>"`) is REUSED, never re-run here. This
phase renders the SAME seed-42 conversation subset of the shared
4,724-conversation track-S corpus into three ablation arms that vary ONLY the
answer's boundary marking / exchange count:

  v2_boundary_absent        the ORIGINAL answer embedded VERBATIM with NO
                            speaker label, NO attribution phrase and NO quotes
                            at the slot — the answer simply continues as its
                            own paragraph after the scene sets up the question.
  v3_label_stripped         the attribution SLOT is kept but the speaker
                            identity is removed (`The reply came: "..."`), so
                            slot-presence and label-presence are separated.
  v4_established_structure  2-3 prior LABELED question->answer exchanges
                            (invented small talk, generated as part of the
                            story) precede the SAME final
                            `Assistant replied: "<verbatim answer>"` boundary —
                            single-exchange -> multi-exchange with the final
                            boundary held FIXED.

Everything else tracks the paired round verbatim: instruct-model tier-2
instruct-and-strip rendering with the original answer embedded verbatim, the
mechanical verbatim-embedding gate, the standard Sonnet judge gate over the
Batch API via `api_dispatch`, bounded retry-until-floor, and HF persistence of
all rollout text BEFORE any downstream phase (Upload Policy raw-completions
rule).

The round is variant-scoped and character-pinned: it REFUSES to run unless
EPM_I1345_VARIANT=story_boundary_ablation and EPM_STORY_CHARACTER_NAME=Assistant
(the V1 anchor's character — a differently-named corpus would break the
single-variable contract AND burn the pod).

Content hygiene: questions/answers are LMSYS-derived real user text and stories
are raw model generations — this script logs COUNTS / ids only.

CLI:
  uv run python scripts/issue1345_boundary_ablation_gen.py --arm v2 [--smoke]
  # CPU preflight (no vLLM): pool + per-arm feasibility + fingerprint, exit 0
  uv run python scripts/issue1345_boundary_ablation_gen.py --arm v2 --verify-pool
  # deferred-import resolution probe (no vLLM, no pool build)
  uv run python scripts/issue1345_boundary_ablation_gen.py --import-check
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# #628 fork-poisoning guard (gotchas.md): main() loads the tokenizer BEFORE the
# vLLM engine builds, so the V1 EngineCore must spawn, not fork. Set BEFORE any
# `import vllm` (vLLM reads the var at import time; the imports are deferred).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402
import issue1345_gen_stories as g  # noqa: E402 — vLLM chunking + HF boundary helpers
import issue1345_gen_stories_paired as gp  # noqa: E402 — pool + retry-wave sizing

from explore_persona_space.llm.api_dispatch import (  # noqa: E402
    RESULT_RATE_LIMITED,
    RESULT_TRANSPORT,
    DispatchItem,
    dispatch_calls,
)

# ---------------------------------------------------------------------------
# Round identity (fail-loud): variant slug + the V1 anchor's character name.
# ---------------------------------------------------------------------------
ROUND_VARIANT = "story_boundary_ablation"
ROUND_CHARACTER = "Assistant"
MODEL_KEY = "instruct"  # instruct writes AND is measured (V1 scope)


def assert_round_env() -> None:
    """Refuse to run outside the round's variant + character scope.

    Both values are load-bearing: the variant scopes every output dir + HF
    prefix (never clobber the V1 anchor), and the character name is spliced
    into `c.ANSWER_ATTRIB_RE` — a differently-named corpus is NOT comparable
    to the V1 anchor and the whole round would be wasted (#1345 onpolicy
    precedent: refuse a wrong-character launch before the pod burns).
    """
    assert c.VARIANT == ROUND_VARIANT, (
        f"EPM_I1345_VARIANT={c.VARIANT!r} — this round requires {ROUND_VARIANT!r} so its "
        "dirs / HF prefixes are scoped under the V1 anchor's parent prefix"
    )
    assert c.STORY_CHARACTER_NAME == ROUND_CHARACTER, (
        f"EPM_STORY_CHARACTER_NAME={c.STORY_CHARACTER_NAME!r} — this round requires "
        f"{ROUND_CHARACTER!r} (the V1 anchor's character; c.ANSWER_ATTRIB_RE is built "
        "from it, and the ablation is single-variable ONLY at the same name)"
    )


# ---------------------------------------------------------------------------
# Arm registry
# ---------------------------------------------------------------------------
ARM_V1 = "v1_boundary_present"  # reused anchor; never generated by this script
ARM_V2 = "v2_boundary_absent"
ARM_V3 = "v3_label_stripped"
ARM_V4 = "v4_established_structure"
ARM_V5 = "v5_bare_label"
GEN_ARMS = (ARM_V2, ARM_V3, ARM_V4, ARM_V5)
ALL_ARMS = (ARM_V1, *GEN_ARMS)
ARM_SLUG = {ARM_V1: "v1", ARM_V2: "v2", ARM_V3: "v3", ARM_V4: "v4", ARM_V5: "v5"}
SLUG_ARM = {v: k for k, v in ARM_SLUG.items()}

# What each arm isolates (verbatim into every summary JSON's per-arm README).
ARM_README = {
    ARM_V1: (
        "boundary-present anchor (REUSED from conversation_paired_stories_assistant, "
        "not re-generated): single exchange, labeled attribution "
        'Assistant replied: "<verbatim answer>" immediately before the answer.'
    ),
    ARM_V2: (
        "boundary-ABSENT: identical narrative recipe and the SAME verbatim answer, but "
        "no speaker label, no attribution phrase and no quotes at the slot — the answer "
        "continues as its own paragraph. Isolates whether the map needs ANY explicit "
        "answer boundary at all."
    ),
    ARM_V3: (
        "label-stripped boundary: the attribution SLOT is kept but carries no speaker "
        'identity (The reply came: "<verbatim answer>"). Isolates the SLOT from the '
        "LABEL — a collapse here implicates the speaker identity, not the boundary."
    ),
    ARM_V4: (
        "established structure: 2-3 prior LABELED question->answer exchanges precede the "
        'SAME final Assistant replied: "<verbatim answer>" boundary. Isolates '
        "exchange-count / established turn structure with the final boundary held fixed."
    ),
    ARM_V5: (
        "bare turn label: identical narrative recipe to V1/V3, but the answer is introduced "
        "by a BARE script-style turn label on its own line (Assistant: <verbatim answer>) — "
        "no attribution verb, no quotes around the answer. This transplants the no-template "
        "comparator's boundary INTO a story, so V5-vs-V1 isolates the boundary FORM (prose "
        "attribution vs pretraining-familiar turn syntax) while V5-vs-no_template isolates "
        "the residual story cost at MATCHED boundary syntax. NOTE the one forced deviation "
        "from the shared closing instruction: the never-a-script clause is narrowed to "
        "exempt that single label line (the arm's construction requires it) — see "
        "_SHARED_CLOSE_V5."
    ),
}

# ---------------------------------------------------------------------------
# Generation budgets. ANSWER_TOKEN_BUDGET is pinned to the V1 value so the
# ELIGIBLE POOL is the same answer-length regime as the anchor (single-variable
# on the corpus). The generation cap is raised uniformly across the three arms
# because V4 additionally carries 2-3 prior exchanges; 2560 >= 2 x 800 satisfies
# the CLAUDE.md >=2x-longest-completion truncation rule with margin.
# ---------------------------------------------------------------------------
ANSWER_TOKEN_BUDGET = gp.ANSWER_TOKEN_BUDGET  # 800 (V1 parity)
ANSWER_CHAR_MIN = gp.ANSWER_CHAR_MIN  # 20 (V1 parity)
BND_MAX_NEW_TOKENS = 2560
PROMPT_TOKEN_BUDGET = g.PROMPT_TOKEN_BUDGET  # 3008 (V1 parity)
BND_MAX_MODEL_LEN = PROMPT_TOKEN_BUDGET + BND_MAX_NEW_TOKENS + 64
# Judge instrument: PINNED to the V1 anchor's 400 tokens. The judge is the KEEP
# gate, so raising it would filter the ablation arms through a different
# instrument than the anchor they are compared against. llm-judging rule 23's
# post-resize duty is honored by MEASURING instead: the per-arm parse-error rate
# lands in every yield report, and `--judge-max-tokens` re-judges at a larger
# budget (against a FRESH cache dir) if a measured arm exceeds 10%.
BND_JUDGE_MAX_TOKENS = c.JUDGE_MAX_TOKENS  # 400

N_TARGET = 2200  # attempted/kept target per arm (brief)
YIELD_FLOOR = 2000  # rc=21 halt below this (brief)
SMOKE_N = 3

MAX_RETRY_WAVES = gp.MAX_RETRY_WAVES
MAX_DRAWS_PER_ROW = gp.MAX_DRAWS_PER_ROW
_RETRY_SUFFIX = gp._RETRY_SUFFIX

# ---------------------------------------------------------------------------
# Per-arm story-writing instructions (tier-2 instruct-and-strip: the verbatim
# constraint lives in the SYSTEM prompt and is STRIPPED before extraction).
# The shared preamble/closing text is IDENTICAL across arms — only the
# boundary/exchange-count clause differs, which is the single manipulated
# variable. `{ANSWER}` is filled per row.
# ---------------------------------------------------------------------------
_SHARED_OPEN = (
    f"You are writing a short narrative story scene in which an AI assistant named "
    f"{ROUND_CHARACTER} is a character. "
)
_SHARED_CLOSE = (
    "Write flowing narrative prose (never a script or 'Name:' dialogue format), and put "
    "all spoken dialogue in double quotes."
)
# V5's construction IS a single script-style label line, which the shared close
# forbids outright — the ONE forced deviation from the otherwise byte-identical
# closing text. Derived by surgery (never a re-typed literal) so a future edit to
# _SHARED_CLOSE cannot silently leave V5 behind; the assert fails loud on drift.
_V5_CLOSE_OLD = "(never a script or 'Name:' dialogue format)"
_V5_CLOSE_NEW = (
    "(the single label line described above is the ONLY script-format line — never use "
    "'Name:' dialogue format anywhere else)"
)
_SHARED_CLOSE_V5 = _SHARED_CLOSE.replace(_V5_CLOSE_OLD, _V5_CLOSE_NEW)
assert _SHARED_CLOSE_V5 != _SHARED_CLOSE, (
    f"_SHARED_CLOSE no longer contains {_V5_CLOSE_OLD!r} — V5's narrowed "
    "never-a-script carve-out did not apply (shared closing text drifted)"
)

STORY_SYSTEM_TEMPLATE = {
    # V2 — boundary ABSENT. No label, no attribution phrase, no quotes at the
    # slot; the answer is its own paragraph after a blank line.
    ARM_V2: (
        _SHARED_OPEN
        + f"The scene must contain exactly ONE moment where a person asks {ROUND_CHARACTER} "
        "a question. Write 3-6 sentences of narrative context first, then have the person "
        "ask the question given by the user (spoken aloud, in double quotes), then write "
        "one short sentence of narrative describing the moment before the answer. Then "
        "start a NEW PARAGRAPH (leave a blank line) whose text is EXACTLY these words: "
        '"{ANSWER}". That paragraph must reproduce those words verbatim, with no '
        "paraphrasing, additions, or omissions. CRITICAL: do NOT put that paragraph in "
        "quotation marks, do NOT introduce it with a speaker name, and do NOT introduce it "
        f"with any attribution phrase such as '{ROUND_CHARACTER} replied' or 'the reply "
        "came' — the words must simply appear as their own paragraph. Do not reveal or "
        "paraphrase the answer content anywhere before that paragraph. " + _SHARED_CLOSE
    ),
    # V3 — attribution slot kept, speaker identity removed.
    ARM_V3: (
        _SHARED_OPEN
        + f"The scene must contain exactly ONE moment where a person asks {ROUND_CHARACTER} "
        "a question and that question is answered. Write 3-6 sentences of narrative context "
        "first, then have the person ask the question given by the user (spoken aloud, in "
        'double quotes), and then give the answer with EXACTLY these words: "{ANSWER}". '
        "Introduce the answer with an attribution before the quotation, exactly in the "
        'form: The reply came: "..." — the quoted answer must match the given words '
        "verbatim, with no paraphrasing, additions, or omissions. CRITICAL: that "
        "attribution must NOT name who is speaking — write exactly 'The reply came:' and "
        f"never '{ROUND_CHARACTER} replied', '{ROUND_CHARACTER} said', or any other phrase "
        f"naming {ROUND_CHARACTER} or anyone else next to a quotation. Do not reveal or "
        "paraphrase the answer content anywhere before that attribution. " + _SHARED_CLOSE
    ),
    # V4 — established structure: prior labeled exchanges, SAME final boundary.
    ARM_V4: (
        _SHARED_OPEN
        + "The scene must contain THREE question-and-answer exchanges in a row between the "
        f"same person and {ROUND_CHARACTER}. Write 3-6 sentences of narrative context "
        "first. Then write TWO short warm-up exchanges of your own invention (ordinary "
        "small talk — the weather, the time, how the day is going): in each, the person "
        "asks something in double quotes and "
        f"{ROUND_CHARACTER} answers in double quotes introduced exactly as "
        f'{ROUND_CHARACTER} replied: "..." with a brief answer of your own. '
        "Then, as the THIRD and final exchange, have the person ask the question given by "
        f"the user (spoken aloud, in double quotes), and have {ROUND_CHARACTER} answer with "
        'EXACTLY these words: "{ANSWER}". Introduce that final answer with an attribution '
        f'before the quotation, exactly in the form: {ROUND_CHARACTER} replied: "..." — the '
        "quoted final answer must match the given words verbatim, with no paraphrasing, "
        "additions, or omissions. Do not reveal or paraphrase the final answer content "
        "anywhere before that final attribution. " + _SHARED_CLOSE
    ),
    # V5 — bare script-style turn label, no attribution verb, answer NOT quoted.
    # Narrative recipe identical to V1/V3 (scene, spoken question, one bridge
    # sentence); only the boundary clause differs.
    ARM_V5: (
        _SHARED_OPEN
        + f"The scene must contain exactly ONE moment where a person asks {ROUND_CHARACTER} "
        "a question and that question is answered. Write 3-6 sentences of narrative context "
        "first, then have the person ask the question given by the user (spoken aloud, in "
        "double quotes), then write one short sentence of narrative describing the moment "
        "before the answer. Then start a NEW LINE that begins with the bare turn label "
        f"'{ROUND_CHARACTER}: ' followed immediately by EXACTLY these words: "
        '"{ANSWER}". That line must read exactly '
        f"{ROUND_CHARACTER}: <those words> — the words must be reproduced verbatim, with no "
        "paraphrasing, additions, or omissions. CRITICAL: write the label exactly as "
        f"'{ROUND_CHARACTER}: ' with nothing else on the line before it, do NOT put the "
        "answer in quotation marks, and do NOT use any attribution verb — never "
        f"'{ROUND_CHARACTER} replied', '{ROUND_CHARACTER} said', or 'the reply came'. Do not "
        "reveal or paraphrase the answer content anywhere before that line. " + _SHARED_CLOSE_V5
    ),
}

# ---------------------------------------------------------------------------
# Per-arm judge rubrics (reason-then-verdict, llm-judging rule 7). Criterion 1
# is the arm's boundary construction; 2/3 are V1's verbatim + no-pre-slot-leak
# criteria verbatim, so the keep instrument differs across arms ONLY in the
# construction being checked.
# ---------------------------------------------------------------------------
_JUDGE_TAIL = (
    "First give 2-3 sentences of reasoning, then finish with EXACTLY two final "
    "lines:\nEXCHANGES: <integer number of qualifying exchanges>\nVERDICT: PASS or FAIL"
)
JUDGE_SYSTEM = {
    ARM_V2: (
        "You evaluate a short story scene against three criteria. The user message gives "
        "the story and the required ANSWER text. The story PASSES only if ALL three hold: "
        "(1) the story contains exactly ONE question asked aloud in quoted dialogue, and "
        "the required ANSWER text appears afterwards as its OWN PARAGRAPH that is NOT in "
        "quotation marks, is NOT preceded by a speaker name, and is NOT introduced by any "
        "attribution phrase (no 'X replied', no 'the reply came', no colon lead-in); "
        "(2) that paragraph matches the required ANSWER text verbatim — not paraphrased, "
        "shortened, or extended; "
        "(3) the answer content is not revealed or paraphrased anywhere BEFORE that "
        "paragraph — the prose before it may reference the question topic but must not "
        "state the answer. " + _JUDGE_TAIL
    ),
    ARM_V3: (
        "You evaluate a short story scene against three criteria. The user message gives "
        "the story and the required ANSWER text. The story PASSES only if ALL three hold: "
        "(1) the story contains exactly ONE question->answer exchange, in which a person "
        "asks a question in quoted dialogue and the answer appears in quoted dialogue "
        "introduced by the attribution 'The reply came:' immediately before the quote, and "
        "that attribution names NOBODY (no character name anywhere in it); "
        "(2) the quoted answer matches the required ANSWER text verbatim — not paraphrased, "
        "shortened, or extended; "
        "(3) the answer content is not revealed or paraphrased anywhere BEFORE the "
        "attribution — the prose before it may reference the question topic but must not "
        "state the answer. " + _JUDGE_TAIL
    ),
    ARM_V4: (
        "You evaluate a short story scene against three criteria. The user message gives "
        "the story and the required ANSWER text. The story PASSES only if ALL three hold: "
        "(1) the story contains THREE question->answer exchanges between the same person "
        f"and an AI assistant character named {ROUND_CHARACTER}, each answer in quoted "
        f"dialogue introduced by an attribution before the quote (e.g. {ROUND_CHARACTER} "
        'replied: "..."), the first two being brief small-talk exchanges and the THIRD '
        "being the required ANSWER; "
        "(2) the quoted answer of the THIRD exchange matches the required ANSWER text "
        "verbatim — not paraphrased, shortened, or extended; "
        "(3) the required answer's content is not revealed or paraphrased anywhere BEFORE "
        "the third attribution — the earlier exchanges must be unrelated small talk. " + _JUDGE_TAIL
    ),
    ARM_V5: (
        "You evaluate a short story scene against three criteria. The user message gives "
        "the story and the required ANSWER text. The story PASSES only if ALL three hold: "
        "(1) the story contains exactly ONE question asked aloud in quoted dialogue, and the "
        f"required ANSWER text appears afterwards on its own line that STARTS with the bare "
        f"turn label '{ROUND_CHARACTER}: ' immediately followed by the answer — the answer is "
        "NOT in quotation marks and there is NO attribution verb anywhere in that label (no "
        f"'{ROUND_CHARACTER} replied', no '{ROUND_CHARACTER} said', no 'the reply came'); "
        "(2) the labelled answer matches the required ANSWER text verbatim — not paraphrased, "
        "shortened, or extended; "
        "(3) the answer content is not revealed or paraphrased anywhere BEFORE that line — "
        "the prose before it may reference the question topic but must not state the answer. "
        + _JUDGE_TAIL
    ),
}

# Prior labeled exchanges required by V4 (2 warm-up + 1 final = 3 attributions;
# 4 tolerated so a model that adds one extra warm-up exchange is not discarded).
V4_TOTAL_ATTRIBS = (3, 4)

# Identity-free attribution for V3: `The reply came: "`. The tolerant
# reply|response|answer family cannot admit a LABELED attribution (no slot for a
# speaker), so widening it raises yield without weakening the arm's contract.
GENERIC_ATTRIB_RE = re.compile(r"\bThe (?:reply|response|answer) came[^\"“”\n]{0,20}?([\"“])")

# V5's bare script-style turn label, anchored to LINE START (re.MULTILINE) so a
# mid-sentence "Assistant: " cannot match. Group 1 is the label THROUGH its ':',
# so `boundary_end = m.end(1)` lands just after the colon — the same convention
# as the attributed arms' rstripped marker text. The name is
# `[A-Za-z0-9_]+`-validated in issue1345_common, so re.escape is belt-and-braces.
BARE_LABEL_RE = re.compile(rf"^({re.escape(ROUND_CHARACTER)}:)[ \t]*", re.MULTILINE)


# ---------------------------------------------------------------------------
# Judge plumbing (reason-then-verdict; drop-never-coerce + transport re-drive)
# ---------------------------------------------------------------------------
def _build_judge_request(item: DispatchItem) -> dict:
    """Messages-API params for one boundary-arm judge call (system top-level)."""
    payload = item.payload
    return {
        "model": c.JUDGE_MODEL,
        "max_tokens": int(payload["judge_max_tokens"]),
        "temperature": 0.0,
        "system": JUDGE_SYSTEM[payload["arm"]],
        "messages": [
            {
                "role": "user",
                "content": f"REQUIRED ANSWER:\n{payload['answer']}\n\nSTORY:\n{payload['story']}",
            }
        ],
    }


def _parse_judge_response(text: str) -> dict:
    """Extract EXCHANGES/VERDICT from a reason-then-verdict reply (raise on miss)."""
    return gp._parse_judge_response(text)


# ---------------------------------------------------------------------------
# Prompt construction + pool feasibility
# ---------------------------------------------------------------------------
def build_prompt(row: dict, arm: str, tokenizer) -> str:
    """Chat-templated tier-2 generation prompt (instruction stripped later)."""
    system = STORY_SYSTEM_TEMPLATE[arm].replace("{ANSWER}", row["answer"])
    user_msg = (
        f"Write the scene now. The question the person asks {ROUND_CHARACTER} is:\n"
        f"{row['question']}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )


def filter_pool_feasible(pool: list[dict], tokenizer) -> tuple[list[dict], dict]:
    """Drop rows over the answer/prompt token budgets — ACROSS ALL THREE ARMS.

    A row is eligible only when it fits under EVERY arm's prompt render, so all
    three arms draw from ONE shared eligible pool (the single-variable contract:
    the arms must not differ in which conversations they can attempt). The
    answer budget is the V1 value, so the eligible answer-length regime matches
    the anchor.
    """
    kept, counts = [], {"answer_over_budget": 0, "prompt_over_budget": 0}
    for row in pool:
        n_ans = len(tokenizer(row["answer"], add_special_tokens=False)["input_ids"])
        if n_ans > ANSWER_TOKEN_BUDGET:
            counts["answer_over_budget"] += 1
            continue
        over = False
        for arm in GEN_ARMS:
            prompt = build_prompt(row, arm, tokenizer)
            if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) > PROMPT_TOKEN_BUDGET:
                over = True
                break
        if over:
            counts["prompt_over_budget"] += 1
            continue
        kept.append(row)
    print(
        f"[seeds] shared feasibility filter (all {len(GEN_ARMS)} arms): "
        f"kept={len(kept)} dropped={counts}",
        flush=True,
    )
    return kept, counts


# ---------------------------------------------------------------------------
# Mechanical span gates — one per arm.
#
# All three return a turn dict in the SAME shape, which is what makes the arms
# comparable: {q_start, q_end, boundary_end, a_start, a_end, n_attribs,
# confidence}. ``boundary_end`` is the char boundary the `context` read sits
# just before — the attribution-marker end for V1/V3/V4, the START of the
# blank-line run for V2 (i.e. the last token of the narrative sentence that
# precedes the answer paragraph). Offsets are RAW-text offsets so the capture
# phase consumes them untouched.
#
# The verbatim match is c.find_verbatim_occurrences (the ONE normalization-
# tolerant matcher the capture phase's trust-boundary re-check shares).
# ---------------------------------------------------------------------------
def _last_question_span_before(story: str, limit: int) -> tuple[int, int] | None:
    """(open_idx, close_idx+1) of the question utterance before ``limit``.

    V1's rule verbatim: the LAST fully-closed quoted span before ``limit`` that
    contains a '?', else the last quoted span (question_is_question=False).
    """
    q_spans = c._quoted_spans_before(story, limit)
    if not q_spans:
        return None
    for qo, qc in reversed(q_spans):
        if "?" in story[qo + 1 : qc]:
            return qo, qc + 1
    qo, qc = q_spans[-1]
    return qo, qc + 1


def _is_question(story: str, q_start: int, q_end: int) -> bool:
    return "?" in story[q_start + 1 : q_end - 1]


def _turn(
    *, q_start: int, q_end: int, boundary_end: int, a_start: int, a_end: int, n_attribs: int
) -> dict | None:
    """Assemble a turn dict, enforcing the shared span-ordering chain."""
    if not (q_end <= boundary_end < a_start < a_end):
        return None
    return {
        "q_start": q_start,
        "q_end": q_end,
        "boundary_end": boundary_end,
        "a_start": a_start,
        "a_end": a_end,
        "n_attribs": int(n_attribs),
    }


def _attrib_opens_answer(story: str, m: re.Match, a_start: int) -> bool:
    """The matched attribution's opening quote sits immediately at the answer.

    Whitespace between the opening quote and the NORMALIZED match start is
    tolerated (the normalized answer is stripped, so a leading space inside the
    quote belongs to neither) — V1's rule verbatim.
    """
    return a_start >= m.end(1) and not story[m.end(1) : a_start].strip()


def _quote_closed_after(story: str, a_end: int) -> bool:
    """A closing double quote right after the answer (whitespace-tolerant)."""
    j = a_end
    while j < len(story) and story[j].isspace():
        j += 1
    return j < len(story) and story[j] in c.DOUBLE_QUOTE_CHARS


def _blank_line_run_before(story: str, a_start: int) -> int | None:
    """Start index of the blank-line (>=2 newline) whitespace run before a_start.

    Returns None when the answer is not preceded by a paragraph break. Because
    the run contains at least two newlines, the char immediately before the
    answer is whitespace — so a quote-wrapped answer is impossible by
    construction here (no separate quote check needed on the open side).
    """
    j = a_start
    while j > 0 and story[j - 1].isspace():
        j -= 1
    if story[j:a_start].count("\n") < 2:
        return None
    return j


def gate_v2(story: str, answer: str) -> tuple[dict | None, str]:
    """Boundary-ABSENT gate: verbatim answer as its own unmarked paragraph."""
    occ = c.find_verbatim_occurrences(story, answer)
    if len(occ) != 1:
        return None, "answer_occurrences_zero" if not occ else "answer_occurrences_multi"
    a_start, a_end = occ[0]
    if c.ANSWER_ATTRIB_RE.search(story):
        return None, "labeled_attribution_present"
    if GENERIC_ATTRIB_RE.search(story):
        return None, "generic_attribution_present"
    # Answer must not be quote-wrapped. The open side is guaranteed by the
    # blank-line run below; check the close side on the IMMEDIATE char (a quote
    # opening the NEXT paragraph's dialogue must not false-reject).
    if a_end < len(story) and story[a_end] in c.DOUBLE_QUOTE_CHARS:
        return None, "answer_quote_closed"
    brk = _blank_line_run_before(story, a_start)
    if brk is None:
        return None, "no_paragraph_break"
    q = _last_question_span_before(story, brk)
    if q is None:
        return None, "question_utterance_missing"
    turn = _turn(
        q_start=q[0],
        q_end=q[1],
        boundary_end=brk,
        a_start=a_start,
        a_end=a_end,
        n_attribs=0,
    )
    if turn is None:
        return None, "span_ordering_degenerate"
    turn["confidence"] = {
        "marker_exact": False,
        "answer_len_ok": ANSWER_CHAR_MIN <= (a_end - a_start) <= 2000,
        "question_found": True,
        "question_is_question": _is_question(story, q[0], q[1]),
    }
    return turn, "ok"


def gate_v3(story: str, answer: str) -> tuple[dict | None, str]:
    """Label-stripped gate: exactly one identity-FREE attribution, no labeled one."""
    occ = c.find_verbatim_occurrences(story, answer)
    if len(occ) != 1:
        return None, "answer_occurrences_zero" if not occ else "answer_occurrences_multi"
    a_start, a_end = occ[0]
    if c.ANSWER_ATTRIB_RE.search(story):
        return None, "labeled_attribution_present"
    attribs = list(GENERIC_ATTRIB_RE.finditer(story))
    if len(attribs) != 1:
        return None, "generic_attribution_zero" if not attribs else "generic_attribution_multi"
    return _finish_attrib_gate(story, answer, attribs, a_start, a_end)


def gate_v4(story: str, answer: str) -> tuple[dict | None, str]:
    """Established-structure gate: 2-3 prior labeled exchanges + the SAME final one."""
    occ = c.find_verbatim_occurrences(story, answer)
    if len(occ) != 1:
        return None, "answer_occurrences_zero" if not occ else "answer_occurrences_multi"
    a_start, a_end = occ[0]
    if GENERIC_ATTRIB_RE.search(story):
        return None, "generic_attribution_present"
    attribs = list(c.ANSWER_ATTRIB_RE.finditer(story))
    if len(attribs) not in V4_TOTAL_ATTRIBS:
        return None, f"attribution_count_{len(attribs)}"
    return _finish_attrib_gate(story, answer, attribs, a_start, a_end)


def gate_v5(story: str, answer: str) -> tuple[dict | None, str]:
    """Bare-label gate: ONE line-initial ``<Name>:`` label opening an UNQUOTED answer.

    Cross-arm rejection is explicit and symmetric with the other three gates: an
    attribution verb of EITHER family (labeled ANSWER_ATTRIB_RE / identity-free
    GENERIC_ATTRIB_RE) makes the story a V1/V4 or V3 construction, and a quoted
    answer makes it an attributed arm with the verb dropped — neither is this arm.
    Conversely a V1/V3/V4 story carries an attribution verb (rejected here), and a
    V2 story has no label at all (``bare_label_zero``).
    """
    occ = c.find_verbatim_occurrences(story, answer)
    if len(occ) != 1:
        return None, "answer_occurrences_zero" if not occ else "answer_occurrences_multi"
    a_start, a_end = occ[0]
    if c.ANSWER_ATTRIB_RE.search(story):
        return None, "labeled_attribution_present"
    if GENERIC_ATTRIB_RE.search(story):
        return None, "generic_attribution_present"
    labels = list(BARE_LABEL_RE.finditer(story))
    if len(labels) != 1:
        return None, "bare_label_zero" if not labels else "bare_label_multi"
    m = labels[0]
    # The answer must be UNQUOTED on BOTH sides (checked on the immediate chars: a
    # quote opening the NEXT paragraph's dialogue must not false-reject).
    if a_end < len(story) and story[a_end] in c.DOUBLE_QUOTE_CHARS:
        return None, "answer_quote_closed"
    if a_start > 0 and story[a_start - 1] in c.DOUBLE_QUOTE_CHARS:
        return None, "answer_quote_open"
    # The label must open the answer on the SAME line: a non-empty whitespace-only
    # gap with no newline. `context` reads the label's ':', so a newline between
    # label and answer would put blank space at the boundary the arm measures.
    if a_start <= m.end(1):
        return None, "label_not_before_answer"
    if story[m.end(1) : a_start].strip() or "\n" in story[m.end(1) : a_start]:
        return None, "label_not_adjacent_to_answer"
    q = _last_question_span_before(story, m.start())
    if q is None:
        return None, "question_utterance_missing"
    turn = _turn(
        q_start=q[0],
        q_end=q[1],
        boundary_end=m.end(1),
        a_start=a_start,
        a_end=a_end,
        n_attribs=0,
    )
    if turn is None:
        return None, "span_ordering_degenerate"
    turn["confidence"] = {
        # The label ends AT its ':' by construction of BARE_LABEL_RE's group 1.
        "marker_exact": True,
        "answer_len_ok": ANSWER_CHAR_MIN <= (a_end - a_start) <= 2000,
        "question_found": True,
        "question_is_question": _is_question(story, q[0], q[1]),
    }
    return turn, "ok"


def _finish_attrib_gate(
    story: str, answer: str, attribs: list[re.Match], a_start: int, a_end: int
) -> tuple[dict | None, str]:
    """Shared tail for the two attribution-bearing arms (V3 single, V4 final).

    The LAST attribution must open the verbatim answer, the quote must close
    right after it, and a quoted question must precede that attribution.
    """
    m = attribs[-1]
    if not _quote_closed_after(story, a_end):
        return None, "answer_quote_not_closed"
    if not _attrib_opens_answer(story, m, a_start):
        return None, "attribution_not_adjacent_to_answer"
    marker_text = story[m.start() : m.end(1) - 1].rstrip()
    boundary_end = m.start() + len(marker_text)
    q = _last_question_span_before(story, m.start())
    if q is None:
        return None, "question_utterance_missing"
    turn = _turn(
        q_start=q[0],
        q_end=q[1],
        boundary_end=boundary_end,
        a_start=a_start,
        a_end=a_end,
        n_attribs=len(attribs),
    )
    if turn is None:
        return None, "span_ordering_degenerate"
    turn["confidence"] = {
        "marker_exact": marker_text.endswith(":"),
        "answer_len_ok": ANSWER_CHAR_MIN <= (a_end - a_start) <= 2000,
        "question_found": True,
        "question_is_question": _is_question(story, q[0], q[1]),
    }
    return turn, "ok"


ARM_GATE = {ARM_V2: gate_v2, ARM_V3: gate_v3, ARM_V4: gate_v4, ARM_V5: gate_v5}


def gate_for(arm: str):
    """The arm's mechanical gate (KeyError on the reused V1 anchor by design)."""
    assert arm in ARM_GATE, f"{arm} has no generation gate (V1 is reused, never generated)"
    return ARM_GATE[arm]


# ---------------------------------------------------------------------------
# Bundle identity (content key over EVERYTHING determining the kept bundle)
# ---------------------------------------------------------------------------
def bundle_fingerprint(arm: str, rows: list[dict], *, judge_max_tokens: int) -> str:
    """sha over the generation recipe + judge instrument + gate source + pool."""
    import inspect

    key = json.dumps(
        {
            "round": ROUND_VARIANT,
            "arm": arm,
            "character": ROUND_CHARACTER,
            "gen_seed": c.GEN_SEED,
            "temperature": c.STORY_TEMPERATURE,
            "max_new_tokens": BND_MAX_NEW_TOKENS,
            "system_template": STORY_SYSTEM_TEMPLATE[arm],
            "rows_sha": hashlib.sha256(
                json.dumps(
                    [(r["conv_id"], r["question"], r["answer"]) for r in rows], sort_keys=True
                ).encode()
            ).hexdigest(),
            "judge_model": c.JUDGE_MODEL,
            "judge_system": JUDGE_SYSTEM[arm],
            "judge_max_tokens": int(judge_max_tokens),
            # The keep-filter recipe IS part of the bundle identity: any change
            # to the arm gate or the shared normalized matcher regenerates
            # rather than reusing stale stories.
            "gate_source_sha": hashlib.sha256(
                (
                    inspect.getsource(ARM_GATE[arm])
                    + inspect.getsource(_finish_attrib_gate)
                    + inspect.getsource(_turn)
                    + inspect.getsource(c.find_verbatim_occurrences)
                    + inspect.getsource(c._norm_with_map)
                ).encode()
            ).hexdigest(),
            # V5's keep set additionally depends on a module-level PATTERN that
            # `getsource(gate_v5)` only references by name, so pin it explicitly.
            # Added CONDITIONALLY so V2/V3/V4 fingerprints stay byte-identical —
            # an unconditional new key would bust the in-flight round's bundles
            # and force a full regeneration. (The pre-existing sibling gap for
            # ANSWER_ATTRIB_RE / GENERIC_ATTRIB_RE is untouched for that reason.)
            **({"gate_regex_pattern": BARE_LABEL_RE.pattern} if arm == ARM_V5 else {}),
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# File layout
# ---------------------------------------------------------------------------
def _stem(arm: str) -> str:
    return f"bnd_{ARM_SLUG[arm]}_{MODEL_KEY}"


def raw_path(out_dir: Path, arm: str, suffix: str = "") -> Path:
    return out_dir / f"raw_stories_{_stem(arm)}{suffix}.jsonl"


def kept_path(out_dir: Path, arm: str) -> Path:
    return out_dir / f"kept_stories_{_stem(arm)}.jsonl"


def judge_path(out_dir: Path, arm: str) -> Path:
    return out_dir / f"judge_results_{_stem(arm)}.jsonl"


def yield_path(out_dir: Path, arm: str) -> Path:
    return out_dir / f"story_yield_{_stem(arm)}.json"


def manifest_path(out_dir: Path, arm: str) -> Path:
    return out_dir / f"story_bundle_manifest_{_stem(arm)}.json"


def bundle_files(out_dir: Path, arm: str) -> list[str]:
    names = [
        raw_path(out_dir, arm).name,
        raw_path(out_dir, arm).with_suffix(".meta.json").name,
        *(
            raw_path(out_dir, arm, s).with_suffix(ext).name
            for s in _RETRY_SUFFIX.values()
            for ext in (".jsonl", ".meta.json")
        ),
        kept_path(out_dir, arm).name,
        judge_path(out_dir, arm).name,
        yield_path(out_dir, arm).name,
        manifest_path(out_dir, arm).name,
    ]
    return [n for n in names if (out_dir / n).exists()]


def hf_prefix(smoke: bool) -> str:
    """HF data-repo prefix for this round's rollout text."""
    return f"{c.HF_SMOKE_PREFIX if smoke else c.HF_ISSUE_PREFIX}/raw_completions/stories"


# ---------------------------------------------------------------------------
# Generation (chunked vLLM + fingerprint-gated JSONL resume — parent pattern)
# ---------------------------------------------------------------------------
def build_llm(model_id: str):
    """vLLM engine sized for this round's raised generation budget."""
    from vllm import LLM

    return LLM(
        model=model_id,
        seed=c.GEN_SEED,
        dtype="bfloat16",
        # gotchas.md: max_model_len tracks max_new_tokens.
        max_model_len=BND_MAX_MODEL_LEN,
        gpu_memory_utilization=0.85,
        enforce_eager=os.environ.get("EPM_VLLM_ENFORCE_EAGER", "0") == "1",
        enable_prefix_caching=(
            False if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING", "0") == "1" else None
        ),
    )


def generate(rows: list[dict], out_path: Path, fp: str, arm: str, tokenizer, llm) -> list[dict]:
    """One story per row, chunked + per-chunk JSONL checkpoint, keyed on conv_id."""
    from vllm import SamplingParams

    meta_path = out_path.with_suffix(".meta.json")
    done_ids: set[str] = set()
    if out_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fp:
            done_ids = {r["conv_id"] for r in c.read_jsonl(out_path)}
            print(f"[gen] resume: {len(done_ids)} stories already on disk", flush=True)
        else:
            raise RuntimeError(
                f"{out_path} exists with a DIFFERENT generation fingerprint "
                f"({meta.get('fingerprint')} != {fp}) — refusing to mix regimes; "
                "move the stale file aside"
            )
    else:
        c.write_json(meta_path, {"fingerprint": fp, "n_rows": len(rows), "arm": arm})

    todo = [r for r in rows if r["conv_id"] not in done_ids]
    sampling = SamplingParams(
        temperature=c.STORY_TEMPERATURE, max_tokens=BND_MAX_NEW_TOKENS, seed=None
    )
    n_chunks = (len(todo) + g.VLLM_CHUNK_SIZE - 1) // g.VLLM_CHUNK_SIZE
    for ci in range(0, len(todo), g.VLLM_CHUNK_SIZE):
        chunk = todo[ci : ci + g.VLLM_CHUNK_SIZE]
        prompts = [build_prompt(r, arm, tokenizer) for r in chunk]
        print(
            f"[vllm-chunk] {arm} gen chunk {ci // g.VLLM_CHUNK_SIZE + 1}/{n_chunks} "
            f"({len(chunk)} prompts)",
            flush=True,
        )
        outs = llm.generate(prompts, sampling, use_tqdm=False)
        new_rows = []
        for r, o in zip(chunk, outs, strict=True):
            new_rows.append(
                {
                    "conv_id": r["conv_id"],
                    # story_id == conv_id: one story per conversation (paired by
                    # construction; the capture phase groups by conv_id).
                    "story_id": r["conv_id"],
                    "question": r["question"],
                    "answer": r["answer"],
                    "arm": arm,
                    "tier": "instruct_and_strip",
                    "story": o.outputs[0].text.strip(),
                    "finish_reason": o.outputs[0].finish_reason,
                }
            )
        c.append_jsonl(out_path, new_rows)
    return c.read_jsonl(out_path) if out_path.exists() else []


# ---------------------------------------------------------------------------
# Parse + judge (mechanical gate first, then the LLM judge)
# ---------------------------------------------------------------------------
def parse_and_judge(
    rows: list[dict], arm: str, cache_dir: Path, smoke: bool, *, judge_max_tokens: int
) -> tuple[list[dict], dict, list[dict]]:
    """(kept, counts, judge_digest_rows) — keep = mechanical PASS and judge PASS."""
    gate = gate_for(arm)
    mech = {r["conv_id"]: gate(r["story"], r["answer"]) for r in rows}

    items = [
        DispatchItem(
            item_id=r["conv_id"],
            payload={
                "story": r["story"],
                "answer": r["answer"],
                "arm": arm,
                "judge_max_tokens": int(judge_max_tokens),
            },
        )
        for r in rows
    ]
    ckpt_dir = gp._judge_checkpoint_dir(cache_dir, rows)
    results = asyncio.run(
        dispatch_calls(
            items,
            model=c.JUDGE_MODEL,
            build_request=_build_judge_request,
            parse_response=_parse_judge_response,
            cache_dir=cache_dir,
            checkpoint_dir=ckpt_dir,
            force_path="sync" if smoke else None,
        )
    )
    # Transport-class re-drive (llm-judging rule 24): a 429/5xx/timeout carries
    # no information about the content and is never persisted as a drop.
    redrive = [
        it
        for it in items
        if results[it.item_id].error
        and results[it.item_id].category in (RESULT_RATE_LIMITED, RESULT_TRANSPORT)
    ]
    if redrive:
        print(f"[judge] re-driving {len(redrive)} transport-class failures", flush=True)
        results.update(
            asyncio.run(
                dispatch_calls(
                    redrive,
                    model=c.JUDGE_MODEL,
                    build_request=_build_judge_request,
                    parse_response=_parse_judge_response,
                    cache_dir=cache_dir,
                    checkpoint_dir=ckpt_dir,
                    force_path="sync",
                )
            )
        )

    counts = {
        "n_generated": len(rows),
        "mech_pass": 0,
        "mech_fail": 0,
        "mech_fail_reasons": {},
        "judge_pass": 0,
        "judge_fail": 0,
        "judge_malformed": 0,
        "transport_loss": 0,
        "finish_reason_length": 0,
        "kept": 0,
    }
    kept, digest_rows = [], []
    for r in rows:
        turn, reason = mech[r["conv_id"]]
        if r.get("finish_reason") == "length":
            counts["finish_reason_length"] += 1
        digest = {"conv_id": r["conv_id"], "arm": arm, "mech_reason": reason}
        if turn is None:
            counts["mech_fail"] += 1
            counts["mech_fail_reasons"][reason] = counts["mech_fail_reasons"].get(reason, 0) + 1
        else:
            counts["mech_pass"] += 1
        res = results[r["conv_id"]]
        if res.error:
            key = (
                "transport_loss"
                if res.category in (RESULT_RATE_LIMITED, RESULT_TRANSPORT)
                else "judge_malformed"
            )
            counts[key] += 1
            digest_rows.append(
                {
                    **digest,
                    "judge_error_category": res.category,
                    "judge_error_reason": str(res.reason)[:160] if res.reason else None,
                }
            )
            continue
        verdict = res.result["verdict"]
        digest_rows.append(
            {**digest, "verdict": verdict, "judge_exchanges": res.result.get("judge_exchanges")}
        )
        counts["judge_pass" if verdict == "PASS" else "judge_fail"] += 1
        if verdict != "PASS" or turn is None:
            continue
        kept.append(
            {
                **r,
                "judge_verdict": verdict,
                "judge_exchanges": res.result.get("judge_exchanges"),
                "parsed_turns": [turn],
                "n_parsed_turns": 1,
            }
        )
        counts["kept"] += 1
    return kept, counts, digest_rows


# ---------------------------------------------------------------------------
# Bounded retry-until-floor (V1's wave sizing, reused verbatim)
# ---------------------------------------------------------------------------
def _draw_counts(out_dir: Path, arm: str, fp: str) -> dict[str, int]:
    """conv_id -> draws recorded in THIS fp bundle's raw files (main + retries)."""
    counts: dict[str, int] = {}
    for suffix in ("", *_RETRY_SUFFIX.values()):
        raw = raw_path(out_dir, arm, suffix)
        meta = raw.with_suffix(".meta.json")
        if not raw.exists() or not meta.exists():
            continue
        with contextlib.suppress(json.JSONDecodeError, OSError):
            if json.loads(meta.read_text()).get("fingerprint") != fp:
                continue
            for r in c.read_jsonl(raw):
                cid = r.get("conv_id")
                if cid is not None:
                    counts[cid] = counts.get(cid, 0) + 1
    return counts


def run_retry_waves(
    *,
    kept: list[dict],
    ordered: list[dict],
    arm: str,
    n_target: int,
    yield_floor: int,
    seed_rate: float,
    out_dir: Path,
    fp: str,
    cache_dir: Path,
    smoke: bool,
    judge_max_tokens: int,
    tokenizer,
    llm,
) -> tuple[list[dict], dict[int, dict], list[dict], dict[str, int]]:
    """Bounded redraw waves over un-kept pool rows (V1's sizing math verbatim).

    Wave 1 fires below the TARGET; waves 2..MAX_RETRY_WAVES are floor-driven,
    each sized from the MEASURED current keep rate with V1's safety factor and
    the per-row draw cap. NO recipe change — the rc=21 floor halt after the
    capped waves is the caller's, unchanged.
    """
    wave_counts: dict[int, dict] = {}
    digest: list[dict] = []
    rate = seed_rate
    draw_counts = _draw_counts(out_dir, arm, fp)
    for wave in range(1, MAX_RETRY_WAVES + 1):
        kept_ids = {r["conv_id"] for r in kept}
        cands = gp.eligible_redraw_rows(ordered, kept_ids, draw_counts)
        take = gp.plan_retry_wave(wave, len(kept), n_target, yield_floor, rate, len(cands))
        if take <= 0:
            break
        wave_rows = cands[:take]
        print(
            f"[retry][{arm}] wave {wave}: drawing {len(wave_rows)} rows "
            f"(kept={len(kept)}, target={n_target}, floor={yield_floor}, rate={rate:.3f})",
            flush=True,
        )
        wave_raw = raw_path(out_dir, arm, _RETRY_SUFFIX[wave])
        rows = generate(wave_rows, wave_raw, fp, arm, tokenizer, llm)
        w_kept, w_counts, w_digest = parse_and_judge(
            rows, arm, cache_dir, smoke, judge_max_tokens=judge_max_tokens
        )
        wave_counts[wave] = w_counts
        digest.extend(w_digest)
        kept = kept + w_kept
        draw_counts = _draw_counts(out_dir, arm, fp)
        if w_counts["n_generated"]:
            rate = w_counts["kept"] / w_counts["n_generated"]
    return kept, wave_counts, digest, draw_counts


def _seed_retry_rate(*, n_kept: int, n_rows: int) -> float:
    """Measured main-batch keep rate, floored so a collapse cannot explode sizing."""
    if n_rows <= 0:
        return gp.RETRY_RATE_FLOOR
    return max(n_kept / n_rows, gp.RETRY_RATE_FLOOR)


# ---------------------------------------------------------------------------
# Persist / resume
# ---------------------------------------------------------------------------
def write_yield_report(
    out_dir: Path,
    arm: str,
    *,
    n_target: int,
    yield_floor: int,
    counts_main: dict,
    wave_counts: dict[int, dict],
    pool_counts: dict,
    n_kept: int,
    fp: str,
    judge_max_tokens: int,
    draw_counts: dict[str, int] | None = None,
) -> dict:
    """Persist the per-arm yield report (main + per-wave counts, floor verdict)."""
    histogram: dict[str, int] = {}
    for n in (draw_counts or {}).values():
        histogram[str(n)] = histogram.get(str(n), 0) + 1
    n_judged = counts_main["n_generated"] + sum(w["n_generated"] for w in wave_counts.values())
    n_malformed = counts_main["judge_malformed"] + sum(
        w["judge_malformed"] for w in wave_counts.values()
    )
    report = {
        "metadata": c.metadata(c.GEN_SEED, n_kept, "scripts/issue1345_boundary_ablation_gen.py"),
        "round": ROUND_VARIANT,
        "arm": arm,
        "arm_isolates": ARM_README[arm],
        "model": MODEL_KEY,
        "story_character_name": c.STORY_CHARACTER_NAME,
        "bundle_fingerprint": fp,
        "gen_max_new_tokens": BND_MAX_NEW_TOKENS,
        "judge_model": c.JUDGE_MODEL,
        "judge_max_tokens": int(judge_max_tokens),
        "n_target": n_target,
        "yield_floor": yield_floor,
        "pool_filter_counts": pool_counts,
        "counts_main": counts_main,
        "counts_by_wave": {str(k): v for k, v in wave_counts.items()},
        "n_kept": n_kept,
        "yield_ok": n_kept >= yield_floor,
        # llm-judging rule 23: the measured per-arm parse-error rate at the
        # V1-pinned judge budget. >10% is the truncation-check trigger — re-judge
        # with --judge-max-tokens against a FRESH cache dir before reading.
        "judge_malformed_rate": (n_malformed / n_judged) if n_judged else None,
    }
    if histogram:
        report["retry_draw_histogram"] = histogram
    c.write_json(yield_path(out_dir, arm), report)
    return report


def persist_bundle(out_dir: Path, arm: str, fp: str, smoke: bool) -> None:
    """Upload rollout text + judge digests + yield report to HF NOW.

    Upload Policy raw-completions rule: rollout text persists at every stage,
    BEFORE any floor can halt this process and before the capture phase runs.
    """
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing — cannot persist story bundle"
    files = bundle_files(out_dir, arm)
    assert kept_path(out_dir, arm).name in files, files
    manifest = {
        "metadata": c.metadata(
            c.GEN_SEED, len(files), "scripts/issue1345_boundary_ablation_gen.py"
        ),
        "round": ROUND_VARIANT,
        "arm": arm,
        "arm_isolates": ARM_README[arm],
        "model": MODEL_KEY,
        "bundle_fingerprint": fp,
        "files": files,
    }
    c.write_json(manifest_path(out_dir, arm), manifest)
    prefix = hf_prefix(smoke)
    g._hf_upload_folder(
        out_dir,
        prefix,
        [f"*{_stem(arm)}*"],
        f"issue-1345 story-boundary-ablation: {arm} story bundle (fp {fp})",
    )
    print(f"[gen] persisted {arm} rollouts -> {prefix} (fp {fp})", flush=True)


def try_resume(out_dir: Path, arm: str, fp: str, smoke: bool) -> dict | None:
    """Reuse a persisted kept bundle when its fingerprint matches."""
    prefix = hf_prefix(smoke)
    remote_manifest = f"{prefix}/{manifest_path(out_dir, arm).name}"
    if not g._hf_file_exists(remote_manifest):
        return None
    local = g._hf_download_to(remote_manifest, manifest_path(out_dir, arm))
    manifest = json.loads(local.read_text())
    if manifest.get("bundle_fingerprint") != fp:
        print(
            f"[gen] HF {arm} bundle fingerprint {manifest.get('bundle_fingerprint')} != {fp} "
            "— stale (recipe changed); regenerating",
            flush=True,
        )
        return None
    for name in manifest["files"]:
        g._hf_download_to(f"{prefix}/{name}", out_dir / name)
    report = json.loads(yield_path(out_dir, arm).read_text())
    print(
        f"[gen] resume-from-HF: {report.get('n_kept')} kept {arm} stories (fp {fp}) "
        "— generation skipped",
        flush=True,
    )
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _import_check() -> None:
    """Resolve every deferred import on the REAL code path, then exit 0.

    The deferred imports live inside build_llm / generate (vLLM) and the
    api_dispatch symbols already resolved at module load; naming them here
    fires the function-body `from ... import ...` forms an `import <module>`
    probe would miss (#1689 false-pass class).
    """
    import inspect

    from vllm import LLM, SamplingParams  # noqa: F401

    from explore_persona_space.llm.api_dispatch import (  # noqa: F401
        RESULT_RATE_LIMITED,
        RESULT_TRANSPORT,
        DispatchItem,
        dispatch_calls,
    )
    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        assert_hub_dir_filecounts,
        retry_transient,
    )

    assert inspect.getsource(gate_v2)
    print("[import-check] OK: vllm + api_dispatch + hub symbols resolved", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=tuple(ARM_SLUG[a] for a in GEN_ARMS) + GEN_ARMS)
    ap.add_argument("--out-dir", type=Path, default=c.STORIES_DIR)
    ap.add_argument("--dl-dir", type=Path, default=c.PARENT_DL_DIR)
    ap.add_argument("--matched-dir", type=Path, default=c.MATCHED_DIR)
    ap.add_argument("--n-stories", type=int, default=N_TARGET)
    ap.add_argument("--yield-floor", type=int, default=YIELD_FLOOR)
    ap.add_argument(
        "--judge-max-tokens",
        type=int,
        default=BND_JUDGE_MAX_TOKENS,
        help="judge response budget (default 400 = V1 parity). Raise ONLY after a "
        "measured judge_malformed_rate > 0.10, and re-judge against a FRESH --cache-dir "
        "(llm-judging rule 23: the rubric-keyed cache is not busted by a budget change)",
    )
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true", help="n=3 stories, sync judge, floor 1")
    ap.add_argument(
        "--verify-pool",
        action="store_true",
        help="CPU preflight: pool + shared feasibility filter + fingerprint, exit 0 "
        "BEFORE any vLLM build",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import on the real code path and exit 0",
    )
    args = ap.parse_args()

    if args.import_check:
        _import_check()
        return

    assert_round_env()
    assert args.arm, "--arm is required (one of: " + ", ".join(ARM_SLUG[a] for a in GEN_ARMS) + ")"
    arm = SLUG_ARM.get(args.arm, args.arm)
    assert arm in GEN_ARMS, arm
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (out_dir / f"judge_cache_{ARM_SLUG[arm]}")

    from transformers import AutoTokenizer

    from explore_persona_space.experiments.issue_825.common import MODEL_INSTRUCT

    tokenizer = AutoTokenizer.from_pretrained(MODEL_INSTRUCT)

    import numpy as np

    pool, pool_counts = gp.load_paired_pool(args.matched_dir, args.dl_dir)
    pool, feas_counts = filter_pool_feasible(pool, tokenizer)
    pool_counts.update(feas_counts)
    n_target = SMOKE_N if args.smoke else args.n_stories
    yield_floor = g.resolve_yield_floor(args.smoke, args.yield_floor)
    assert len(pool) >= n_target, f"eligible pool {len(pool)} < target {n_target}"
    # ONE seeded permutation shared by every arm (c.GEN_SEED = 42): the arms
    # attempt the SAME conversations in the SAME order, so a kept-set difference
    # is a YIELD difference, never a corpus difference.
    rng = np.random.default_rng(c.GEN_SEED)
    ordered = [pool[i] for i in rng.permutation(len(pool))]
    fp = bundle_fingerprint(arm, ordered, judge_max_tokens=args.judge_max_tokens)
    rows_main = ordered[:n_target]

    if args.verify_pool:
        report_path = out_dir / f"pool_report_{_stem(arm)}.json"
        c.write_json(
            report_path,
            {
                "metadata": c.metadata(
                    c.GEN_SEED, len(rows_main), "scripts/issue1345_boundary_ablation_gen.py"
                ),
                "round": ROUND_VARIANT,
                "arm": arm,
                "model": MODEL_KEY,
                "smoke": bool(args.smoke),
                "n_pool_eligible": len(ordered),
                "n_rows_main": len(rows_main),
                "n_target": n_target,
                "yield_floor": yield_floor,
                "pool_filter_counts": pool_counts,
                "fingerprint": fp,
                "gen_max_new_tokens": BND_MAX_NEW_TOKENS,
                "max_model_len": BND_MAX_MODEL_LEN,
                "judge_max_tokens": int(args.judge_max_tokens),
            },
        )
        print(
            f"[verify-pool] {arm} pool OK: eligible={len(ordered)} rows_main={len(rows_main)} "
            f"fp={fp} -> {report_path}",
            flush=True,
        )
        return

    resumed = try_resume(out_dir, arm, fp, args.smoke)
    if resumed is not None:
        g.enforce_yield_floor(int(resumed["n_kept"]), yield_floor)
        print(f"[done] {arm} kept={resumed['n_kept']}/{n_target} (resumed)", flush=True)
        return

    from explore_persona_space.experiments.issue_825.common import MODEL_INSTRUCT as _MI

    llm = build_llm(_MI)
    rows = generate(rows_main, raw_path(out_dir, arm), fp, arm, tokenizer, llm)
    kept, counts, digest = parse_and_judge(
        rows, arm, cache_dir, args.smoke, judge_max_tokens=args.judge_max_tokens
    )

    wave_counts: dict[int, dict] = {}
    draw_counts: dict[str, int] = {}
    if len(kept) < n_target:
        kept, wave_counts, retry_digest, draw_counts = run_retry_waves(
            kept=kept,
            ordered=ordered,
            arm=arm,
            n_target=n_target,
            yield_floor=yield_floor,
            seed_rate=_seed_retry_rate(n_kept=len(kept), n_rows=len(rows)),
            out_dir=out_dir,
            fp=fp,
            cache_dir=cache_dir,
            smoke=args.smoke,
            judge_max_tokens=args.judge_max_tokens,
            tokenizer=tokenizer,
            llm=llm,
        )
        digest.extend(retry_digest)

    kp = kept_path(out_dir, arm)
    if kp.exists():
        kp.unlink()
    c.append_jsonl(kp, kept)
    jp = judge_path(out_dir, arm)
    if jp.exists():
        jp.unlink()
    c.append_jsonl(jp, digest)

    report = write_yield_report(
        out_dir,
        arm,
        n_target=n_target,
        yield_floor=yield_floor,
        counts_main=counts,
        wave_counts=wave_counts,
        pool_counts=pool_counts,
        n_kept=len(kept),
        fp=fp,
        judge_max_tokens=args.judge_max_tokens,
        draw_counts=draw_counts,
    )
    # Persist BEFORE any floor can halt this process (Upload Policy).
    persist_bundle(out_dir, arm, fp, args.smoke)

    rate = report.get("judge_malformed_rate")
    if rate is not None and rate > 0.10:
        print(
            f"[judge][WARN] {arm} judge_malformed_rate={rate:.3f} > 0.10 — run the "
            "truncation check (llm-judging rule 23): re-judge with --judge-max-tokens 600 "
            "against a FRESH --cache-dir and compare the drop rate",
            flush=True,
        )
    if args.smoke and len(kept) < n_target:
        print(
            f"[yield-floor][smoke] shortfall: kept={len(kept)}/{n_target} — proceeding "
            "(smoke floor=1 so the capture phase is exercised)",
            flush=True,
        )
    g.enforce_yield_floor(len(kept), yield_floor)
    print(f"[done] {arm} kept={len(kept)}/{n_target} stories", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
