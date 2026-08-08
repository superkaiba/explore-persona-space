#!/usr/bin/env python
"""Issue #1345 conversation-paired-stories round — gen_stories_paired phase.

Renders a seed-42 subsample of N_STORIES_PAIRED_TARGET (2,700) of the SAME
shared S-track conversations into narrative prose with the EXACT original
answer embedded verbatim (`<c.STORY_CHARACTER_NAME> replied: "<original
answer>"` — "Assistant" under the v9 conversation_paired_stories_assistant
scope, plan v9 header), one Q->A exchange per story (plan v8 §4). Tier-2
instruct-and-strip: the verbatim constraint lives in the system prompt and is
STRIPPED before extraction. Keep = mechanical ANSWER-ANCHORED span gate
(exactly ONE verbatim answer occurrence, opened by exactly ONE attribution
marker, quote closed right after it, quoted question before the marker — see
match_verbatim_turn's rationale comment for why quote-pairing recovery is not
used) AND judge PASS (reason-then-verdict, criteria: one exchange / verbatim
match / no pre-slot answer revelation — the two NEW criteria vs the r3 rubric,
plan v8 §4.3). Yield floor 2,160/2,700 after the bounded retry waves
(run_retry_waves: <= MAX_RETRY_WAVES redraw waves over un-kept pool rows,
<= MAX_DRAWS_PER_ROW draws per row) -> rc=21 story-regime halt (plan v8 §7,
halt semantics byte-unchanged); smoke floor 1.

--op-companion: the N<=200 on-policy control cell (plan v8 §4.5, the #1335
tf/op calibration shape) — seed-0 sample of the KEPT paired conversations,
free generation WITHOUT the verbatim answer (the model writes its own answer);
one-exchange rubric. kept < companion_usable_floor (5 production == the
grouped-CV minimum; 1 under --smoke) -> rc=23 (companion unusable; the TF
headline proceeds, calibration reports N/A).

Content hygiene: questions/answers are LMSYS-derived real user text and
stories are raw model generations — this script logs COUNTS/ids only.

CLI:
  uv run python scripts/issue1345_gen_stories_paired.py --model instruct [--smoke]
  uv run python scripts/issue1345_gen_stories_paired.py --model instruct --op-companion
  # CPU preflight (no vLLM): pool + feasibility filter + fingerprint, exit 0
  uv run python scripts/issue1345_gen_stories_paired.py --model instruct --op-companion \
      --smoke --verify-pool
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
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
import issue1345_gen_stories as g  # noqa: E402 — vLLM chunking + HF persist helpers

from explore_persona_space.llm.api_dispatch import (  # noqa: E402
    RESULT_RATE_LIMITED,
    RESULT_TRANSPORT,
    DispatchItem,
    dispatch_calls,
)

SMOKE_N_STORIES = 3
# Paired-mode generation budget (cps fix round, 2026-07-17 — DECLARED plan
# deviation vs v8 §11's inherited 1024): the r4 production run truncated ~49%
# of its answer_occurrences_zero rejects mid-answer at the parent's FREE-FORM
# budget (c.STORY_MAX_NEW_TOKENS=1024; raw rows end mid-sentence with
# finish_reason=length). A paired story must carry wrapper prose + the quoted
# question + the attribution + the VERBATIM answer (pool-capped at
# ANSWER_TOKEN_BUDGET=800 tokens) + closing prose, so 2048 = 800-token answer
# cap + ~1248 wrapper/question/closure margin, satisfying the CLAUDE.md
# >=2x-longest-completion truncation rule (2 x 800 = 1600 <= 2048). The op
# companion + the parent free-form recipe keep 1024 — their completions carry
# no verbatim answer.
STORY_PAIRED_MAX_NEW_TOKENS = 2048
# vLLM capacity must track the raised budget (gotchas.md "max_model_len tracks
# max_new_tokens"): prompt <= g.PROMPT_TOKEN_BUDGET (= g.MAX_MODEL_LEN - 1024
# - 64, deliberately UNCHANGED so the feasibility-filtered pool — and hence
# rows_sha / the carried kept bundle — is identical to the r4 run) + 2048 gen
# fits with >=40 tokens headroom.
PAIRED_MAX_MODEL_LEN = g.MAX_MODEL_LEN + (STORY_PAIRED_MAX_NEW_TOKENS - c.STORY_MAX_NEW_TOKENS)
# Generation feasibility bound: the story must reproduce the verbatim answer
# within the paired budget. Pinned at the r4 run's realized value
# (1024 - 224 = 800) rather than re-derived from the new budget: raising it
# would grow the eligible pool and change rows_sha, orphaning the kept-1,474
# carry-forward bundle.
ANSWER_TOKEN_BUDGET = 800
# Degenerate-answer floor (the #825 zero-width-span class; matches the
# parser's answer_len_ok lower bound).
ANSWER_CHAR_MIN = 20

# Bounded retry-until-floor (micro-round, 2026-07-17): relaunch 2 landed
# kept=2,117 < floor 2,160 with 1,972 un-kept eligible pool rows stranded —
# the single retry wave was sized at the first-pass keep rate while the hard
# residual kept at 0.254. Retry waves beyond the main batch redraw ONLY
# un-kept pool rows (fewest-draws-first), each sized from the MEASURED
# current keep rate with a safety factor; per-row draws are capped so a
# hopeless row is never redrawn forever. NO recipe change — temp 1.0, pool,
# budget 2048, judge rubric untouched; the floor value + rc=21 halt
# semantics are byte-unchanged (a miss after the capped waves is final).
MAX_RETRY_WAVES = 3  # total retry waves beyond the main batch
RETRY_SAFETY = 0.8  # floor-driven waves are sized needed/(rate*SAFETY)
MAX_DRAWS_PER_ROW = 3  # per-row draw cap within the current fp bundle
RETRY_RATE_FLOOR = 0.05  # never size a wave off a collapsed measured rate
# Redraws are DRAW-INDEXED into distinct files: generate_paired resumes by
# conv_id per file, so an id's k-th draw must land where no prior draw of
# that id exists. Draw 1 of a reserve row keeps the legacy "_retry" name
# (byte-compat with the relaunch-2 bundle); draws 2/3 get their own files.
_RETRY_SUFFIX = {1: "_retry", 2: "_retry2", 3: "_retry3"}


def story_max_new_tokens(*, op_companion: bool) -> int:
    """Mode-keyed gen budget: paired carries a verbatim answer -> 2048; the op
    companion writes its own answer -> the parent free-form 1024. ONE helper so
    the SamplingParams cap and the bundle fingerprint can never drift apart."""
    return c.STORY_MAX_NEW_TOKENS if op_companion else STORY_PAIRED_MAX_NEW_TOKENS


# Optional ONE-LINE persona intro for the INJECTED / ON-POLICY character arms
# (descriptions passed verbatim from issue1310_common.PERSONAS). When it is
# ABSENT — the default, and every pre-existing round — the four templates below
# are BYTE-IDENTICAL to their pre-persona text, so the paired + onpolicy rounds'
# bundle fingerprints (which hash the template string) are unchanged and their
# kept bundles keep resuming. Byte-identity is pinned by a test.
#
# The intro also FIXES a latent mis-description: the bare form asserts "an AI
# assistant named X", which is wrong for the non-AI panel members (Dana is "an
# ordinary, unremarkable everyday person") — exactly the axis the AI-likeness
# judge leg measures, so the wrapper must not pre-assert AI-ness.
PERSONA_DESC = os.environ.get("EPM_I1345_PERSONA_DESC", "").strip()
assert "\n" not in PERSONA_DESC, (
    f"EPM_I1345_PERSONA_DESC must be a single line (got {PERSONA_DESC!r}) — it is "
    "spliced into a one-sentence character intro"
)
# A char_* variant WITHOUT a description would silently generate the bare
# "an AI assistant named X" intro — wrong for the non-AI panel members (Dana is
# "an ordinary, unremarkable everyday person"), invisible until someone reads the
# stories, and the exact axis the AI-likeness judge leg measures. Fail at import.
assert not (c.VARIANT.startswith("char_") and not PERSONA_DESC), (
    f"variant {c.VARIANT!r} is a character arm but EPM_I1345_PERSONA_DESC is empty — "
    "the wrapper would assert AI-ness for every character; export the description "
    "verbatim from issue1310_common.PERSONAS"
)
_CHAR_INTRO = (
    f"a character named {c.STORY_CHARACTER_NAME}, {PERSONA_DESC}"
    if PERSONA_DESC
    else f"an AI assistant named {c.STORY_CHARACTER_NAME}"
)
_CHAR_INTRO_JUDGE = (
    f"a character named {c.STORY_CHARACTER_NAME}, {PERSONA_DESC}"
    if PERSONA_DESC
    else f"an AI assistant character named {c.STORY_CHARACTER_NAME}"
)

# Built from c.STORY_CHARACTER_NAME (the variant guard in common pins ARIA for
# this round unless --variant re-scopes). {ANSWER} is filled per row; the
# instruction is STRIPPED before extraction (tier-2 instruct-and-strip).
STORY_PAIRED_SYSTEM_TEMPLATE = (
    f"You are writing a short narrative story scene in which {_CHAR_INTRO} "
    f"is a character. The scene must contain exactly ONE moment "
    f"where a person asks {c.STORY_CHARACTER_NAME} a question and "
    f"{c.STORY_CHARACTER_NAME} answers it. Write 3-6 sentences of narrative context first, "
    "then have the person ask the question given by the user (spoken aloud, in double "
    f"quotes), and then have {c.STORY_CHARACTER_NAME} answer with EXACTLY these words: "
    '"{ANSWER}". Introduce the answer with an attribution before the quotation, exactly '
    f'in the form: {c.STORY_CHARACTER_NAME} replied: "..." — the quoted answer must match '
    "the given words verbatim, with no paraphrasing, additions, or omissions. Do not "
    "reveal or paraphrase the answer content anywhere before that attribution. Write "
    "flowing narrative prose (never a script or 'Name:' dialogue format), and put all "
    "spoken dialogue in double quotes."
)
STORY_OP_COMPANION_SYSTEM = (
    f"You are writing a short narrative story scene in which {_CHAR_INTRO} "
    f"is a character. The scene must contain exactly ONE moment "
    f"where a person asks {c.STORY_CHARACTER_NAME} a question and "
    f"{c.STORY_CHARACTER_NAME} gives a substantive, helpful answer. Write 3-6 sentences "
    "of narrative context first, then have the person ask the question given by the user "
    f"(spoken aloud, in double quotes), and then have {c.STORY_CHARACTER_NAME} answer it. "
    f"Introduce the answer with an attribution before the quotation, exactly in the form: "
    f'{c.STORY_CHARACTER_NAME} replied: "..." Write flowing narrative prose (never a '
    "script or 'Name:' dialogue format), and put all spoken dialogue in double quotes."
)

JUDGE_SYSTEM_PAIRED = (
    "You evaluate a short story scene against three criteria. The user message gives the "
    "story and the required ANSWER text. The story PASSES only if ALL three hold: "
    "(1) the story contains exactly ONE question->answer exchange, in which a person asks "
    f"a question in quoted dialogue and {_CHAR_INTRO_JUDGE} "
    "answers in quoted dialogue introduced by an attribution "
    f'before the quote (e.g. {c.STORY_CHARACTER_NAME} replied: "..."); '
    "(2) the quoted answer matches the required ANSWER text verbatim — not paraphrased, "
    "shortened, or extended; "
    "(3) the answer content is not revealed or paraphrased anywhere BEFORE the attribution "
    "marker — the prose before it may reference the question topic but must not state the "
    "answer. First give 2-3 sentences of reasoning, then finish with EXACTLY two final "
    "lines:\nEXCHANGES: <integer number of qualifying exchanges>\nVERDICT: PASS or FAIL"
)
JUDGE_SYSTEM_OP = (
    "You evaluate a short story scene. The story PASSES only if it contains exactly ONE "
    "question->answer exchange, in which a person asks a question in quoted dialogue and "
    f"{_CHAR_INTRO_JUDGE} gives a substantive answer "
    "in quoted dialogue introduced by an attribution before the quote (e.g. "
    f'{c.STORY_CHARACTER_NAME} replied: "..."). Exchanges with trivial or empty answers, '
    f"unquoted dialogue, or answers not attributed to {c.STORY_CHARACTER_NAME} do not "
    "count. First give 2-3 sentences of reasoning, then finish with EXACTLY two final "
    "lines:\nEXCHANGES: <integer number of qualifying exchanges>\nVERDICT: PASS or FAIL"
)


# ---------------------------------------------------------------------------
# Judge plumbing (reason-then-verdict; drop-never-coerce + transport re-drive)
# ---------------------------------------------------------------------------
def _build_judge_request(item: DispatchItem) -> dict:
    """Messages-API params for one paired-story judge call (system top-level)."""
    payload = item.payload
    if payload.get("mode") == "op":
        system = JUDGE_SYSTEM_OP
        user = payload["story"]
    else:
        system = JUDGE_SYSTEM_PAIRED
        user = f"REQUIRED ANSWER:\n{payload['answer']}\n\nSTORY:\n{payload['story']}"
    return {
        "model": c.JUDGE_MODEL,
        "max_tokens": c.JUDGE_MAX_TOKENS,
        "temperature": 0.0,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }


def _parse_judge_response(text: str) -> dict:
    """Extract EXCHANGES/VERDICT from a reason-then-verdict reply (raise on miss)."""
    exchanges, verdict = None, None
    for line in text.split("\n"):
        s = line.strip()
        if s.upper().startswith("EXCHANGES:"):
            with contextlib.suppress(ValueError, IndexError):
                exchanges = int(s.split(":", 1)[1].strip().split()[0])
        if s.upper().startswith("VERDICT:"):
            v = s.split(":", 1)[1].strip().upper()
            if v in ("PASS", "FAIL"):
                verdict = v
    if verdict is None:
        raise ValueError(f"judge reply missing VERDICT line ({len(text)} chars)")
    return {"verdict": verdict, "judge_exchanges": exchanges}


# ---------------------------------------------------------------------------
# Seed pool: matched-n allowlist (staged by prefetch_reuse) x track_s Q/A text
# ---------------------------------------------------------------------------
def load_paired_pool(matched_dir: Path, dl_dir: Path) -> tuple[list[dict], dict]:
    """{conv_id, question, answer} rows for the shared conv set + filter counts.

    Source of the 4,724 shared conv ids: the parent's staged matched-n
    allowlist (matched_subsets_parent.json, staged by prefetch_reuse at the
    pinned REUSE_REV). Answers = the ORIGINAL track-S responses @ PIN_REV —
    the exact text the r1/r2 stores teacher-forced, so the r4 answer
    distribution is IDENTICAL to chat/plain-text (plan v8 §4).
    """
    parent_matched = matched_dir / "matched_subsets_parent.json"
    assert parent_matched.exists(), (
        f"staged parent matched-n allowlist missing: {parent_matched} — run the "
        "prefetch_reuse phase first (it stages the pinned allowlist)"
    )
    shared = {str(x) for x in json.loads(parent_matched.read_text())["shared_r1r2_convs"]}
    track_s = c.stage_pinned_file(c.PARENT_TRACK_S_JSONL, dl_dir)
    rows = c.read_jsonl(track_s)
    counts = {
        "shared_convs": len(shared),
        "joined": 0,
        "answer_too_short": 0,
        "kept": 0,
    }
    pool = []
    for r in rows:
        cid = str(r.get("conv_id") or f"s{r['prompt_idx']}")
        if cid not in shared:
            continue
        counts["joined"] += 1
        answer = r["response"]
        if len(answer) < ANSWER_CHAR_MIN:
            # Degenerate near-empty answers (the #825 zero-width-span class);
            # answer-anchored matching handles embedded quotes, so no quote
            # drop — the token budget in _filter_pool_feasible bounds length.
            counts["answer_too_short"] += 1
            continue
        pool.append({"conv_id": cid, "question": r["prompt"], "answer": answer})
        counts["kept"] += 1
    assert pool, "paired seed pool empty — allowlist did not join against track_s.jsonl"
    print(f"[seeds] paired pool filter counts: {counts}", flush=True)
    return sorted(pool, key=lambda r: r["conv_id"]), counts


def build_paired_prompt(row: dict, tokenizer, *, op_companion: bool) -> str:
    """Chat-templated tier-2 generation prompt (instruction stripped later)."""
    system = (
        STORY_OP_COMPANION_SYSTEM
        if op_companion
        else STORY_PAIRED_SYSTEM_TEMPLATE.replace("{ANSWER}", row["answer"])
    )
    user_msg = (
        f"Write the scene now. The question the person asks {c.STORY_CHARACTER_NAME} is:\n"
        f"{row['question']}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _filter_pool_feasible(pool: list[dict], tokenizer, *, op_companion: bool) -> tuple[list, dict]:
    """Drop rows whose formatted prompt / answer exceed the token budgets.

    Row schema is MODE-KEYED (att-20260716-230002 crash-fix): paired rows carry
    ``{conv_id, question, answer}`` and the verbatim-answer budget applies (HARD
    ``answer`` key — a paired row without it is a producer bug, fail loud);
    companion rows carry ``{conv_id, question}`` ONLY (free generation — no
    fixed answer exists), so only the formatted-prompt budget applies and
    ``answer`` is never touched.
    """
    kept, counts = [], {"prompt_over_budget": 0, "answer_over_budget": 0}
    for row in pool:
        if not op_companion:
            n_ans = len(tokenizer(row["answer"], add_special_tokens=False)["input_ids"])
            if n_ans > ANSWER_TOKEN_BUDGET:
                counts["answer_over_budget"] += 1
                continue
        prompt = build_paired_prompt(row, tokenizer, op_companion=op_companion)
        n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_tok > g.PROMPT_TOKEN_BUDGET:
            counts["prompt_over_budget"] += 1
            continue
        kept.append(row)
    print(
        f"[seeds] feasibility filter (op_companion={op_companion}): "
        f"kept={len(kept)} dropped={counts}",
        flush=True,
    )
    return kept, counts


# ---------------------------------------------------------------------------
# Mechanical span gate — ANSWER-ANCHORED (the extraction contract).
#
# Why not parse_story_turns' quote-pairing span recovery (the r3 path): the
# original track-S answers are REAL LMSYS responses and 29% of the shared pool
# carries embedded double quotes — a closing-quote scan ends the span early on
# every such row, so quote-parser matching caps the eligible pool at 2,293
# conversations, BELOW the plan's 2,700 target (measured at implementation
# time, plan v8 §12.2 verify note). Because the r4 construction KNOWS the
# verbatim answer text, the span is located by its unique occurrence instead
# (eligible pool 3,843): the ANSWER_ATTRIB_RE attribution marker must open a
# quote immediately before the occurrence (identical context-slot definition
# to r3 — last token of `<STORY_CHARACTER_NAME> replied: "`), exactly ONE attribution in the
# story, a closing quote right after the answer, and no verbatim leak before
# the marker. Turn dicts keep the parse_story_turns shape verbatim, so
# render_story_turn (offsets + BPE seam guard) is reused untouched.
# ---------------------------------------------------------------------------
def match_verbatim_turn(story: str, answer: str) -> tuple[dict | None, str]:
    """(turn, reason) — the answer-anchored single-exchange verbatim gate.

    Returns a parse_story_turns-shaped turn dict on 'ok'; otherwise (None,
    <keep-filter counter>). Enforces: exactly one verbatim answer occurrence
    (NORMALIZATION-TOLERANT — c.find_verbatim_occurrences, the ONE matcher the
    extraction re-check shares; cps fix round), exactly one attribution marker
    (one exchange) whose opening quote sits immediately before the occurrence
    (whitespace-tolerant, consistent with the normalized match), a closing
    double quote right after it (ditto), a quoted question utterance before
    the marker, and no pre-slot answer leak. The returned a_start/a_end are
    RAW-text offsets, so render_story_turn consumes them untouched.
    """
    occ = c.find_verbatim_occurrences(story, answer)
    if len(occ) != 1:
        return None, "answer_occurrences_zero" if not occ else "answer_occurrences_multi"
    a_start, a_end = occ[0]
    j = a_end
    while j < len(story) and story[j].isspace():
        j += 1
    # End-of-story guard (code-review v13 Major): story[j : j + 1] yields ""
    # past the end and "" in DOUBLE_QUOTE_CHARS is True — an unclosed quote
    # with the answer at end-of-story must FAIL closure, so bounds-check then
    # index a single char.
    if j >= len(story) or story[j] not in c.DOUBLE_QUOTE_CHARS:
        return None, "answer_quote_not_closed"
    attribs = list(c.ANSWER_ATTRIB_RE.finditer(story))
    if len(attribs) != 1:
        return None, "attribution_zero" if not attribs else "attribution_multi"
    m = attribs[0]
    # The attributed quote must open right at the answer (whitespace between
    # the opening quote and the normalized-match start is tolerated — the
    # normalized answer is stripped, so a leading space inside the quote
    # belongs to neither).
    if a_start < m.end(1) or story[m.end(1) : a_start].strip():
        return None, "attribution_not_adjacent_to_answer"
    marker_text = story[m.start() : m.end(1) - 1].rstrip()
    marker_end = m.start() + len(marker_text)
    q_spans = c._quoted_spans_before(story, m.start())
    if not q_spans:
        return None, "question_utterance_missing"
    q_open, q_close = None, None
    for qo, qc in reversed(q_spans):
        if "?" in story[qo + 1 : qc]:
            q_open, q_close = qo, qc
            break
    question_is_question = q_open is not None
    if q_open is None:
        q_open, q_close = q_spans[-1]
    # (No separate pre-slot-leak branch: a verbatim answer before the marker is
    # either a SECOND occurrence -> answer_occurrences_multi, or the only one
    # -> attribution_not_adjacent_to_answer. Paraphrase leaks are the judge's.)
    turn = {
        "q_start": q_open,
        "q_end": q_close + 1,
        "marker_end": marker_end,
        "a_start": a_start,
        "a_end": a_end,
        "confidence": {
            "marker_exact": marker_text.endswith(":"),
            # INFORMATIONAL in the r4 verbatim path — nothing gates on this
            # flag here (the span-ordering check below is the only mechanical
            # gate; contrast confident_op_turn, which gates on the PARENT
            # parser's confidence dict). The <=2000-char upper bound predates
            # the drop of the pool-level ANSWER_CHAR_MAX cap: pool answers are
            # capped at <=800 TOKENS, which can exceed 2000 chars, so False
            # here just means "long answer", never a defect (r1 review Minor).
            "answer_len_ok": 20 <= (a_end - a_start) <= 2000,
            "question_found": True,
            "question_is_question": bool(question_is_question),
        },
    }
    if not (turn["q_end"] <= turn["marker_end"] < turn["a_start"] < turn["a_end"]):
        return None, "span_ordering_degenerate"
    return turn, "ok"


def confident_op_turn(story: str) -> tuple[dict | None, str]:
    """Companion keep-filter: exactly one parsed turn with a confident answer."""
    turns = c.parse_story_turns(story)
    if len(turns) != 1:
        return None, f"parsed_turns_{'zero' if not turns else 'multi'}"
    t = turns[0]
    conf = t["confidence"]
    if not (conf["marker_exact"] or conf["answer_len_ok"]):
        return None, "low_confidence_turn"
    return t, "ok"


# ---------------------------------------------------------------------------
# Generation (chunked vLLM + fingerprint-gated JSONL resume — parent pattern)
# ---------------------------------------------------------------------------
def companion_usable_floor(smoke: bool) -> int:
    """Minimum kept companion rows for a usable control cell (rc=23 below it).

    Production: c.OP_COMPANION_MIN_KEPT (= the grouped-CV minimum — the
    companion fit is a conv-grouped 5-fold CV with one row per conversation,
    so fewer kept convs than folds fits nothing; r1 code-review Major).
    Smoke: 1 — any NONZERO yield proceeds (the #1345-r3 gate-calibration rule:
    a production-scale floor deterministically kills the smoke leg).
    """
    return 1 if smoke else c.OP_COMPANION_MIN_KEPT


def paired_fingerprint(mode: str, rows: list[dict]) -> str:
    """Content key over everything that determines the kept bundle (r6 rule)."""
    import inspect

    # The op-companion bundle is keyed by mode_slug "paired_op" (main's slug;
    # bare "op" kept for compat) — keying on "op" alone silently embedded the
    # PAIRED templates/parser in the companion fp (r1 code-review Minor: a
    # future op-recipe change would have resumed a stale companion bundle).
    is_op = mode in ("op", "paired_op")
    key = json.dumps(
        {
            "mode": mode,
            "gen_seed": c.GEN_SEED,
            "temperature": c.STORY_TEMPERATURE,
            # Mode-keyed (cps fix round): paired generates at 2048 (verbatim
            # answer must fit); the op companion keeps the free-form 1024.
            "max_new_tokens": story_max_new_tokens(op_companion=is_op),
            "system_template": (
                STORY_OP_COMPANION_SYSTEM if is_op else STORY_PAIRED_SYSTEM_TEMPLATE
            ),
            # Mode-keyed row identity (the _filter_pool_feasible schema contract):
            # companion rows have NO answer by construction, so the op key is
            # (conv_id, question); paired rows HARD-key the verbatim answer —
            # byte-identical to the prior (cid, q, answer) triple for every
            # well-formed paired row, so the persisted paired bundle resumes.
            "rows_sha": hashlib.sha256(
                json.dumps(
                    [
                        (r["conv_id"], r["question"])
                        if is_op
                        else (r["conv_id"], r["question"], r["answer"])
                        for r in rows
                    ],
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            "judge_model": c.JUDGE_MODEL,
            "judge_system": JUDGE_SYSTEM_OP if is_op else JUDGE_SYSTEM_PAIRED,
            "judge_max_tokens": c.JUDGE_MAX_TOKENS,
            # The keep-filter recipe IS part of the bundle identity: any change
            # to the matcher regenerates rather than reusing stale stories.
            # The paired sha ALSO covers the shared normalized matcher in
            # common (the gate<->extractor consistency contract, cps fix
            # round) — a matcher edit there must re-key the bundle too.
            "parser_source_sha": hashlib.sha256(
                (
                    inspect.getsource(confident_op_turn)
                    if is_op
                    else inspect.getsource(match_verbatim_turn)
                    + inspect.getsource(c.find_verbatim_occurrences)
                    + inspect.getsource(c._norm_with_map)
                ).encode()
            ).hexdigest(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def generate_paired(
    rows: list[dict], out_path: Path, fp: str, tokenizer, llm, *, op_companion: bool
) -> list[dict]:
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
        c.write_json(meta_path, {"fingerprint": fp, "n_rows": len(rows)})

    todo = [r for r in rows if r["conv_id"] not in done_ids]
    sampling = SamplingParams(
        temperature=c.STORY_TEMPERATURE,
        max_tokens=story_max_new_tokens(op_companion=op_companion),
        seed=None,
    )
    n_chunks = (len(todo) + g.VLLM_CHUNK_SIZE - 1) // g.VLLM_CHUNK_SIZE
    for ci in range(0, len(todo), g.VLLM_CHUNK_SIZE):
        chunk = todo[ci : ci + g.VLLM_CHUNK_SIZE]
        prompts = [build_paired_prompt(r, tokenizer, op_companion=op_companion) for r in chunk]
        print(
            f"[vllm-chunk] paired gen chunk {ci // g.VLLM_CHUNK_SIZE + 1}/{n_chunks} "
            f"({len(chunk)} prompts, op_companion={op_companion})",
            flush=True,
        )
        outs = llm.generate(prompts, sampling, use_tqdm=False)
        new_rows = []
        for r, o in zip(chunk, outs, strict=True):
            row_out = {
                "conv_id": r["conv_id"],
                # story_id == conv_id: one story per conversation (paired
                # by construction; extraction groups by conv_id).
                "story_id": r["conv_id"],
                "question": r["question"],
                "mode": "op" if op_companion else "paired",
                "tier": "instruct_and_strip",
                "story": o.outputs[0].text.strip(),
                "finish_reason": o.outputs[0].finish_reason,
            }
            if not op_companion:
                # Paired story rows HARD-carry the verbatim answer (the
                # _filter_pool_feasible mode contract); companion rows omit
                # it — free generation has no fixed answer, and no consumer
                # (op judge rubric, confident_op_turn, _render_r4 r4op) reads it.
                row_out["answer"] = r["answer"]
            new_rows.append(row_out)
        c.append_jsonl(out_path, new_rows)
    # A fully-carried relaunch can legitimately have ZERO rows to generate
    # (carry >= target): no chunk ever appends, so the raw file may not exist.
    return c.read_jsonl(out_path) if out_path.exists() else []


# ---------------------------------------------------------------------------
# Bounded retry-until-floor (micro-round): wave sizing + draw accounting.
# ---------------------------------------------------------------------------
def plan_retry_wave(
    wave: int, n_kept: int, n_target: int, yield_floor: int, rate: float, n_candidates: int
) -> int:
    """Rows to draw in retry wave ``wave`` (0 == the wave does not fire).

    Wave 1 keeps the plan-v8 semantics — fires below the TARGET, sized
    ``max(target_shortfall, ceil(floor_gap / rate))``. Waves 2..MAX_RETRY_WAVES
    are FLOOR-driven redraws: fire only below the yield floor, sized
    ``ceil(floor_gap / (rate * RETRY_SAFETY))``. Always clamped to the
    eligible-candidate count; the measured rate is floored at
    RETRY_RATE_FLOOR so a collapsed rate can never explode the sizing.
    """
    if n_candidates <= 0 or wave < 1 or wave > MAX_RETRY_WAVES:
        return 0
    rate = max(rate, RETRY_RATE_FLOOR)
    if wave == 1:
        if n_kept >= n_target:
            return 0
        shortfall = n_target - n_kept
        need = max(0, yield_floor - n_kept)
        take = max(shortfall, math.ceil(need / rate))
    else:
        if n_kept >= yield_floor:
            return 0
        take = math.ceil((yield_floor - n_kept) / (rate * RETRY_SAFETY))
    return min(take, n_candidates)


def eligible_redraw_rows(
    ordered: list[dict], kept_ids: set[str], draw_counts: dict[str, int]
) -> list[dict]:
    """Un-kept pool rows still under the per-row draw cap, fewest-draws-first.

    Fewest-draws-first (stable within equal counts, so pool order is
    preserved) makes wave 1 prefer never-drawn reserve rows before any
    redraw — the relaunch-2 behavior — while later waves redraw fairly.
    """
    cands = [
        r
        for r in ordered
        if r["conv_id"] not in kept_ids and draw_counts.get(r["conv_id"], 0) < MAX_DRAWS_PER_ROW
    ]
    cands.sort(key=lambda r: draw_counts.get(r["conv_id"], 0))
    return cands


def _bundle_draw_counts(out_dir: Path, mode_slug: str, model_key: str, fp: str) -> dict[str, int]:
    """conv_id -> draws recorded in THIS fp bundle's raw files (main + retries).

    The durable, resume-safe draw ledger: every draw is a row in one of the
    bundle's raw JSONLs (a same-fp meta sidecar gates each file), so a
    relaunch recounts identically. Prior-fp bundles are quarantined and never
    counted — the cap is per-recipe by design.
    """
    counts: dict[str, int] = {}
    for suffix in ("", *_RETRY_SUFFIX.values()):
        raw = out_dir / f"raw_stories_{mode_slug}_{model_key}{suffix}.jsonl"
        meta = out_dir / f"raw_stories_{mode_slug}_{model_key}{suffix}.meta.json"
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


def _prior_fresh_rate(report: dict) -> float | None:
    """Fresh keep rate of a resumed bundle's OWN draws (carry excluded)."""
    try:
        gen = 0
        for key in ("counts_main", "counts_retry", "counts_retry_2", "counts_retry_3"):
            cnt = report.get(key) or {}
            gen += int(cnt.get("n_generated") or 0)
        fresh = int(report["n_kept"]) - int(report.get("n_kept_carried") or 0)
        return fresh / gen if gen > 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def _seed_retry_rate(*, n_fresh_kept: int, n_fresh_rows: int, prior_report_path: Path) -> float:
    """Wave-1 seed rate: this process's fresh keep rate (carry excluded).

    A fully-carried main pass generated nothing fresh — seed from the prior
    bundle's own realized fresh rate when its yield report survives, else
    optimistic 1.0 (wave 1's max(target_shortfall, ...) sizing floors the
    take at the shortfall regardless, so an optimistic seed never under-draws
    the plan-v8 wave).
    """
    if n_fresh_rows > 0:
        return n_fresh_kept / n_fresh_rows
    rate = None
    if prior_report_path.exists():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            rate = _prior_fresh_rate(json.loads(prior_report_path.read_text()))
    return 1.0 if rate is None else rate


def _merge_wave_counts(acc: dict, new: dict) -> dict:
    """Sum parse_and_judge_paired count dicts (mech_fail_reasons merged per key)."""
    for k, v in new.items():
        if k == "mech_fail_reasons":
            sub = acc.setdefault("mech_fail_reasons", {})
            for rk, rv in v.items():
                sub[rk] = sub.get(rk, 0) + rv
        else:
            acc[k] = acc.get(k, 0) + v
    return acc


def run_retry_waves(
    *,
    kept: list[dict],
    ordered: list[dict],
    n_target: int,
    yield_floor: int,
    seed_rate: float,
    out_dir: Path,
    mode_slug: str,
    model_key: str,
    fp: str,
    cache_dir: Path,
    smoke: bool,
    tokenizer,
    llm,
    op_companion: bool = False,
    gen_fn=None,
    judge_fn=None,
) -> tuple[list[dict], dict[int, dict], list[dict], dict[str, int]]:
    """Bounded floor-driven retry waves over the un-kept pool (micro-round).

    Each wave: recount draws from the bundle's raw files, pick eligible
    candidates (un-kept, draw-capped, fewest-draws-first), size via
    plan_retry_wave, generate into DRAW-INDEXED files (an id's k-th draw
    lands in _RETRY_SUFFIX[k]'s file, where generate_paired's per-conv_id
    resume cannot skip it), judge, and merge NEW keeps (dedupe by conv_id —
    a resumed wave file can re-yield already-kept rows). Judging runs PER
    draw-group file: gen_fn returns ALL rows in the touched file (the resume
    contract — that is what recovers a crashed prior wave's generated-but-
    unjudged draws from the judge cache), and one conv_id can legitimately
    sit in BOTH _retry (an old un-kept draw) and _retry2 (this wave's redraw)
    — a single judge call over both would collide on conv_id-keyed
    mech/DispatchItem state and mis-pair one draw's story with the other's
    spans. Within one file conv_ids are unique, so per-file calls are safe;
    keeps merge across groups via the kept_ids dedupe. Returns
    ``(kept, per_wave_counts, judge_digest_rows, final_draw_counts)``.
    ``gen_fn``/``judge_fn`` are the GPU / judge-API boundaries (tests inject
    signature-conformant fakes; production uses the real functions).
    """
    gen_fn = gen_fn or generate_paired
    judge_fn = judge_fn or parse_and_judge_paired
    kept = list(kept)
    kept_ids = {r["conv_id"] for r in kept}
    wave_counts: dict[int, dict] = {}
    digest_all: list[dict] = []
    rate = max(seed_rate, RETRY_RATE_FLOOR)
    for wave in range(1, MAX_RETRY_WAVES + 1):
        draw_counts = _bundle_draw_counts(out_dir, mode_slug, model_key, fp)
        candidates = eligible_redraw_rows(ordered, kept_ids, draw_counts)
        take = plan_retry_wave(wave, len(kept), n_target, yield_floor, rate, len(candidates))
        if take == 0:
            break
        wave_rows = candidates[:take]
        print(
            f"[retry] wave {wave}: kept={len(kept)}/{n_target} (floor {yield_floor}, "
            f"rate {rate:.3f}) — drawing {len(wave_rows)} of {len(candidates)} eligible",
            flush=True,
        )
        wcounts: dict = {}
        n_new_this_wave = 0
        next_draws = sorted({draw_counts.get(r["conv_id"], 0) + 1 for r in wave_rows})
        for nd in next_draws:
            group = [r for r in wave_rows if draw_counts.get(r["conv_id"], 0) + 1 == nd]
            path = out_dir / f"raw_stories_{mode_slug}_{model_key}{_RETRY_SUFFIX[nd]}.jsonl"
            file_rows = gen_fn(group, path, fp, tokenizer, llm, op_companion=op_companion)
            gkept, gcounts, gdigest = judge_fn(
                file_rows, cache_dir, smoke, op_companion=op_companion
            )
            new = [k for k in gkept if k["conv_id"] not in kept_ids]
            kept.extend(new)
            kept_ids.update(k["conv_id"] for k in new)
            n_new_this_wave += len(new)
            _merge_wave_counts(wcounts, gcounts)
            digest_all.extend(gdigest)
        wave_counts[wave] = {
            **wcounts,
            "wave": wave,
            "take": take,
            "rate_used": rate,
            "n_kept_new": n_new_this_wave,
        }
        # Next wave sizes off THIS wave's measured rate over its OWN fresh
        # draws (bounded to [floor, 1]).
        rate = min(
            1.0,
            max(n_new_this_wave / len(wave_rows) if wave_rows else rate, RETRY_RATE_FLOOR),
        )
    return kept, wave_counts, digest_all, _bundle_draw_counts(out_dir, mode_slug, model_key, fp)


# ---------------------------------------------------------------------------
# Parse + judge (mechanical gate first, then the LLM judge)
# ---------------------------------------------------------------------------
def _judge_checkpoint_dir(cache_dir: Path, rows: list[dict]) -> Path:
    """Dispatch-set-scoped batch-checkpoint dir (cps fix round).

    api_dispatch's batch state.json binds to ONE dispatched set and RAISES
    ValueError on a run-fingerprint mismatch (rule 22, #1018) — so a shared
    ``cache_dir/checkpoints`` dir would crash the first batch-path dispatch
    after any regen (the r4 run left a 2,700-item state.json there; only the
    <2,000-item sync routing of the retry batch dodged it). Keying the dir on
    the dispatched (conv_id, story-sha) set keeps crash-resume within one set
    (same rows -> same dir -> resume) while different sets never collide.
    """
    key = hashlib.sha256(
        json.dumps(
            sorted((r["conv_id"], hashlib.sha256(r["story"].encode()).hexdigest()) for r in rows)
        ).encode()
    ).hexdigest()[:12]
    return cache_dir / "checkpoints" / f"set_{key}"


def parse_and_judge_paired(
    rows: list[dict], cache_dir: Path, smoke: bool, *, op_companion: bool
) -> tuple[list[dict], dict, list[dict]]:
    """(kept, counts, judge_digest_rows) — keep = mechanical PASS ∧ judge PASS."""
    mech: dict[str, tuple[dict | None, str]] = {}
    for r in rows:
        mech[r["conv_id"]] = (
            confident_op_turn(r["story"])
            if op_companion
            else match_verbatim_turn(r["story"], r["answer"])
        )

    items = [
        DispatchItem(
            item_id=r["conv_id"],
            # Mode-keyed payload: the op rubric never reads an answer, so op
            # rows carry none; paired rows HARD-key it. The paired payload
            # keeps the original (story, answer, mode) insertion order — the
            # api_dispatch cache key serializes the payload, so byte-stable
            # ordering preserves the paired judge cache across this fix.
            payload=(
                {"story": r["story"], "mode": r["mode"]}
                if r["mode"] == "op"
                else {"story": r["story"], "answer": r["answer"], "mode": r["mode"]}
            ),
        )
        for r in rows
    ]
    ckpt_dir = _judge_checkpoint_dir(cache_dir, rows)
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
        "kept": 0,
    }
    kept, digest_rows = [], []
    for r in rows:
        turn, reason = mech[r["conv_id"]]
        digest = {"conv_id": r["conv_id"], "mode": r["mode"], "mech_reason": reason}
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
            # Persist the (content-free) error reason so a malformed spike is
            # diagnosable from the HF digest alone — the r4 run's 277
            # judge_malformed rows carried category only, leaving the rule-23
            # truncation-vs-parse-noise question unanswerable off-pod. The
            # parse error embeds ONLY the reply char count ("judge reply
            # missing VERDICT line (N chars)"); truncate defensively anyway.
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
# Bundle persist / resume (reuses the parent's HF boundary helpers)
# ---------------------------------------------------------------------------
def bundle_files_paired(mode_slug: str, model_key: str, out_dir: Path) -> list[str]:
    names = [
        f"raw_stories_{mode_slug}_{model_key}.jsonl",
        f"raw_stories_{mode_slug}_{model_key}.meta.json",
        # Draw-indexed retry files (micro-round): _retry / _retry2 / _retry3
        *(
            f"raw_stories_{mode_slug}_{model_key}{s}{ext}"
            for s in _RETRY_SUFFIX.values()
            for ext in (".jsonl", ".meta.json")
        ),
        f"kept_stories_{mode_slug}_{model_key}.jsonl",
        f"story_yield_{mode_slug}_{model_key}.json",
        f"judge_results_{mode_slug}_{model_key}.jsonl",
    ]
    return [n for n in names if (out_dir / n).exists()]


def persist_bundle_paired(
    mode_slug: str,
    model_key: str,
    out_dir: Path,
    fp: str,
    smoke: bool,
    *,
    carried_from_fp: str | None = None,
    n_kept_carried: int = 0,
):
    """Upload rollout text + judge digests + yield report to HF NOW (r6 rule).

    ``carried_from_fp``/``n_kept_carried`` record the kept-carry-forward
    provenance (cps fix round) — the upload OVERWRITES the prior bundle's
    paths on HF, so the manifest carries an explicit regeneration note
    (upload-policy producer duty; prior bytes stay reachable via HF git
    revisions) instead of a version-bumped path: no dependent capture exists
    (the r4 extraction never ran — the yield halt fired first).
    """
    import os

    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing — cannot persist story bundle"
    files = bundle_files_paired(mode_slug, model_key, out_dir)
    assert f"kept_stories_{mode_slug}_{model_key}.jsonl" in files, files
    manifest = {
        "metadata": c.metadata(c.GEN_SEED, len(files), "scripts/issue1345_gen_stories_paired.py"),
        "model": model_key,
        "mode": mode_slug,
        "bundle_fingerprint": fp,
        "files": files,
        "carried_from_fp": carried_from_fp,
        "n_kept_carried": n_kept_carried,
        "regeneration_note": (
            None
            if carried_from_fp is None
            else (
                f"raw stories regenerated at max_new_tokens="
                f"{story_max_new_tokens(op_companion=mode_slug == 'paired_op')} with the "
                f"normalized verbatim gate (cps fix round); {n_kept_carried} kept rows "
                f"carried forward verbatim from bundle fp {carried_from_fp} (prior bytes "
                "at earlier HF revisions of this path)"
            )
        ),
    }
    c.write_json(out_dir / f"story_bundle_manifest_{mode_slug}_{model_key}.json", manifest)
    prefix = g._stories_hf_prefix(smoke)
    g._hf_upload_folder(
        out_dir,
        prefix,
        [f"*{mode_slug}_{model_key}*"],
        f"issue-1345: {mode_slug} story bundle ({model_key}, fp {fp})",
    )
    print(f"[gen] persisted {mode_slug} rollouts -> {prefix} (fp {fp})", flush=True)


def try_resume_paired(
    mode_slug: str, model_key: str, fp: str, out_dir: Path, smoke: bool
) -> dict | None:
    """Reuse a persisted kept bundle when its fingerprint matches (r6 rule)."""
    prefix = g._stories_hf_prefix(smoke)
    manifest_path = f"{prefix}/story_bundle_manifest_{mode_slug}_{model_key}.json"
    if not g._hf_file_exists(manifest_path):
        return None
    local = g._hf_download_to(
        manifest_path, out_dir / f"story_bundle_manifest_{mode_slug}_{model_key}.json"
    )
    manifest = json.loads(local.read_text())
    if manifest.get("bundle_fingerprint") != fp:
        print(
            f"[gen] HF {mode_slug} bundle fingerprint {manifest.get('bundle_fingerprint')} "
            f"!= {fp} — stale (recipe changed); regenerating",
            flush=True,
        )
        return None
    for name in manifest["files"]:
        g._hf_download_to(f"{prefix}/{name}", out_dir / name)
    report = json.loads((out_dir / f"story_yield_{mode_slug}_{model_key}.json").read_text())
    print(
        f"[gen] resume-from-HF: {report.get('n_kept')} kept {mode_slug} stories "
        f"(fp {fp}) — generation skipped",
        flush=True,
    )
    return report


def load_kept_carryforward(
    mode_slug: str,
    model_key: str,
    fp_new: str,
    out_dir: Path,
    pool_by_id: dict[str, dict],
    smoke: bool,
) -> tuple[list[dict], str | None]:
    """Prior-fp kept rows carried into the new bundle (cps fix round).

    The budget/matcher fix re-keys the paired fingerprint, so try_resume_paired
    correctly reads the r4 production bundle as stale — but its 1,474 kept
    stories (generated, gate-PASSed AND judge-PASSed at the old recipe) remain
    valid corpus rows: kept text never changes, an exact-match keep is also a
    normalized-match keep, and the stored raw-offset spans re-verify under the
    shared matcher at extraction. Carry them forward instead of regenerating:
    read the prior kept bundle (local ``out_dir`` first; HF manifest fallback,
    ANY fingerprint != ``fp_new``), keep only rows whose (conv_id, question,
    answer) equals the CURRENT pool row byte-for-byte (pool-drift guard), and
    return ``(carried_rows, old_fp)``. Paired mode only — the op companion
    resamples from the merged kept set and is cheap to regenerate.
    """
    assert mode_slug == "paired", mode_slug
    kept_path = out_dir / f"kept_stories_{mode_slug}_{model_key}.jsonl"
    manifest_path = out_dir / f"story_bundle_manifest_{mode_slug}_{model_key}.json"
    old_fp: str | None = None
    if manifest_path.exists():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            old_fp = json.loads(manifest_path.read_text()).get("bundle_fingerprint")
    if not kept_path.exists():
        prefix = g._stories_hf_prefix(smoke)
        hf_manifest = f"{prefix}/story_bundle_manifest_{mode_slug}_{model_key}.json"
        if not g._hf_file_exists(hf_manifest):
            return [], None
        manifest = json.loads(g._hf_download_to(hf_manifest, manifest_path).read_text())
        old_fp = manifest.get("bundle_fingerprint")
        if old_fp == fp_new:
            return [], None  # same-fp bundle — try_resume_paired owns that path
        g._hf_download_to(f"{prefix}/{kept_path.name}", kept_path)
    elif old_fp == fp_new:
        return [], None
    carried, dropped = _pool_identity_filter(c.read_jsonl(kept_path), pool_by_id)
    print(
        f"[gen] carry-forward: {len(carried)} kept {mode_slug} rows from bundle fp "
        f"{old_fp} reused verbatim ({dropped} dropped by the pool-identity guard); "
        f"only the non-kept remainder regenerates at the new recipe (fp {fp_new})",
        flush=True,
    )
    return carried, old_fp


def _pool_identity_filter(rows: list[dict], pool_by_id: dict[str, dict]) -> tuple[list[dict], int]:
    """Keep only paired rows whose (conv_id, question, answer) byte-matches the
    CURRENT pool (the carry guard); returns (carried, n_dropped)."""
    carried, dropped = [], 0
    for r in rows:
        p = pool_by_id.get(r.get("conv_id"))
        if (
            r.get("mode") == "paired"
            and p is not None
            and r.get("question") == p["question"]
            and r.get("answer") == p["answer"]
        ):
            carried.append(r)
        else:
            dropped += 1
    return carried, dropped


def quarantine_stale_raw(out_dir: Path, mode_slug: str, model_key: str, fp_new: str) -> None:
    """Move prior-fp raw-story JSONLs aside (crash-fix element-5 quarantine).

    generate_paired fail-louds on an fp-mismatched raw file by design ("move
    the stale file aside") — this is the mover, run at relaunch: the failed
    run's raw bundle is already persisted on HF (persist-before-floor, r6
    rule), so local copies move to ``out_dir/stale_<oldfp>/`` where the resume
    glob (the exact out_path/meta pair) can never see them. A raw file with NO
    meta sidecar is quarantined too — resuming it under a fresh meta would
    silently mix generation regimes.
    """
    for base in (
        f"raw_stories_{mode_slug}_{model_key}",
        *(f"raw_stories_{mode_slug}_{model_key}{s}" for s in _RETRY_SUFFIX.values()),
    ):
        raw = out_dir / f"{base}.jsonl"
        meta = out_dir / f"{base}.meta.json"
        if not raw.exists() and not meta.exists():
            continue
        old_fp = None
        if meta.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                old_fp = json.loads(meta.read_text()).get("fingerprint")
        if old_fp == fp_new:
            continue  # same regime — generate_paired resumes per-row
        dest = out_dir / f"stale_{old_fp or 'unknown'}"
        dest.mkdir(exist_ok=True)
        for p in (raw, meta):
            if p.exists():
                p.replace(dest / p.name)
        print(
            f"[gen] quarantined stale raw bundle {base} (fp {old_fp} != {fp_new}) -> {dest}",
            flush=True,
        )


def _build_llm(model_id: str):
    import os

    from vllm import LLM

    return LLM(
        model=model_id,
        seed=c.GEN_SEED,
        dtype="bfloat16",
        # Tracks the paired 2048 budget (gotchas.md: max_model_len tracks
        # max_new_tokens). One engine config for both modes — the op
        # companion's shorter cap fits trivially under the larger window.
        max_model_len=PAIRED_MAX_MODEL_LEN,
        gpu_memory_utilization=0.85,
        enforce_eager=os.environ.get("EPM_VLLM_ENFORCE_EAGER", "0") == "1",
        enable_prefix_caching=(
            False if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING", "0") == "1" else None
        ),
    )


# Per-wave yield-report keys (micro-round): wave k persists under the key
# _prior_fresh_rate reads — 1 keeps the legacy "counts_retry" name (backward-
# readable: a pre-wave report simply lacks the _2/_3 keys), 2/3 get their own.
_WAVE_COUNTS_KEY = {1: "counts_retry", 2: "counts_retry_2", 3: "counts_retry_3"}


def _write_yield_report(
    mode_slug: str,
    model_key: str,
    out_dir: Path,
    *,
    n_target: int,
    yield_floor: int,
    counts_main: dict,
    wave_counts: dict[int, dict] | None,
    pool_counts: dict,
    n_kept: int,
    n_kept_carried: int = 0,
    carried_from_fp: str | None = None,
    draw_counts: dict[str, int] | None = None,
) -> None:
    """Persist the per-run yield report (main + per-wave counts, floor verdict).

    ``wave_counts`` is run_retry_waves' {wave: counts} map; each fired wave
    lands under _WAVE_COUNTS_KEY[wave] (absent key == wave never fired —
    "counts_retry" stays None when no wave fired, preserving the legacy
    schema). ``draw_counts`` (conv_id -> draws in this fp bundle) persists as
    a per-draw-count histogram, digest-only.
    """
    wave_counts = wave_counts or {}
    histogram: dict[str, int] = {}
    for n in (draw_counts or {}).values():
        histogram[str(n)] = histogram.get(str(n), 0) + 1
    report = {
        "metadata": c.metadata(c.GEN_SEED, n_kept, "scripts/issue1345_gen_stories_paired.py"),
        "gen_max_new_tokens": story_max_new_tokens(op_companion=mode_slug == "paired_op"),
        "n_kept_carried": n_kept_carried,
        "carried_from_fp": carried_from_fp,
        "model": model_key,
        "mode": mode_slug,
        "story_character_name": c.STORY_CHARACTER_NAME,
        "n_target": n_target,
        "yield_floor": yield_floor,
        "pool_filter_counts": pool_counts,
        "counts_main": counts_main,
        "counts_retry": wave_counts.get(1),
        "n_kept": n_kept,
        "yield_ok": n_kept >= yield_floor,
    }
    for wave in (2, 3):
        if wave in wave_counts:
            report[_WAVE_COUNTS_KEY[wave]] = wave_counts[wave]
    if histogram:
        report["retry_draw_histogram"] = histogram
    c.write_json(out_dir / f"story_yield_{mode_slug}_{model_key}.json", report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # "pretrained" is legal ONLY under the base-measured variant (R4_MODELS gates
    # it via the assert below); the wrapper-writer engine stays instruct
    # (_build_llm(MODEL_INSTRUCT)) regardless — --model tags the MEASURED model +
    # the output cell/stem/kept-file, never the wrapper writer.
    ap.add_argument("--model", choices=("instruct", "pretrained"), default="instruct")
    ap.add_argument("--out-dir", type=Path, default=c.STORIES_DIR)
    ap.add_argument("--dl-dir", type=Path, default=c.PARENT_DL_DIR)
    ap.add_argument("--matched-dir", type=Path, default=c.MATCHED_DIR)
    ap.add_argument("--n-stories", type=int, default=c.N_STORIES_PAIRED_TARGET)
    ap.add_argument("--yield-floor", type=int, default=c.STORY_PAIRED_YIELD_FLOOR)
    ap.add_argument(
        "--op-companion",
        action="store_true",
        help="generate the N<=200 on-policy companion control cell (plan v8 §4.5)",
    )
    ap.add_argument(
        "--op-powered",
        action="store_true",
        help="on-policy-assistant-story round: generate the on-policy paired story "
        "corpus at POWERED n directly from the matched-n pool (the r4op companion "
        "construction PROMOTED to the primary regime; retry-until-floor to >=2000 "
        "kept). Same on-policy generation semantics as --op-companion, but pool-"
        "sourced + floor-gated instead of a <=200 sample of the TF-kept set.",
    )
    ap.add_argument("--smoke", action="store_true", help="n=3 stories, sync judge")
    ap.add_argument(
        "--verify-pool",
        action="store_true",
        help="CPU preflight: run main through the pool stage (incl. "
        "_filter_pool_feasible in the invoked mode) + fingerprint, write "
        "pool_report_{mode}_{model}.json, exit 0 BEFORE any vLLM build",
    )
    args = ap.parse_args()

    assert not (args.op_companion and args.op_powered), (
        "--op-companion and --op-powered are mutually exclusive (companion = <=200 "
        "control from the TF-kept set; powered = pool-sourced primary regime)"
    )
    if args.op_powered:
        assert c.HAS_ONPOLICY_STORY, (
            f"--op-powered requires EPM_I1345_VARIANT in {c.ONPOLICY_STORY_VARIANTS} "
            f"(got {c.VARIANT!r}) — the r4op primary registry + variant-scoped output "
            "dirs are gated on it (never clobber the parent run)"
        )
    else:
        assert c.HAS_R4, (
            f"gen_stories_paired requires EPM_I1345_VARIANT in {c.PAIRED_STORIES_VARIANTS} "
            f"(got {c.VARIANT!r}) — the r4 registry and variant-scoped output dirs are "
            "gated on it (never clobber the parent run)"
        )
    assert args.model in c.R4_MODELS, args.model
    model_key = args.model
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # On-policy generation semantics (op prompt + confident_op_turn parse + 1024
    # tokens) for BOTH the <=200 companion and the powered primary; the paired TF
    # verbatim-embed path is neither. mode_slug=paired_op binds the on-policy
    # fingerprint recipe (paired_fingerprint is_op branch).
    op_gen = args.op_companion or args.op_powered
    mode_slug = "paired_op" if op_gen else "paired"
    cache_dir = out_dir / f"judge_cache_{mode_slug}"

    from transformers import AutoTokenizer

    from explore_persona_space.experiments.issue_825.common import MODEL_INSTRUCT

    tokenizer = AutoTokenizer.from_pretrained(MODEL_INSTRUCT)

    import numpy as np

    kept_carry: list[dict] = []
    carry_fp: str | None = None
    if args.op_companion:
        kept_paired = out_dir / f"kept_stories_paired_{model_key}.jsonl"
        assert kept_paired.exists(), (
            f"{kept_paired} missing — run the paired gen phase before --op-companion"
        )
        kept_rows = c.read_jsonl(kept_paired)
        rng = np.random.default_rng(c.OP_COMPANION_SEED)
        n_take = min(c.OP_COMPANION_N, len(kept_rows)) if not args.smoke else min(3, len(kept_rows))
        idx = rng.choice(len(kept_rows), size=n_take, replace=False)
        rows_main = [
            {"conv_id": kept_rows[i]["conv_id"], "question": kept_rows[i]["question"]}
            for i in sorted(int(i) for i in idx)
        ]
        rows_main, pool_counts = _filter_pool_feasible(rows_main, tokenizer, op_companion=True)
        # Companion usable floor: 5 production (grouped-CV minimum) / 1 smoke
        # -> rc=23 below it (plan v8 §4.5; r1 code-review Major).
        n_target, yield_floor = len(rows_main), companion_usable_floor(args.smoke)
        seeds_reserve: list[dict] = []
        fp = paired_fingerprint(mode_slug, rows_main)
    elif args.op_powered:
        # Powered on-policy primary regime (onpolicy-assistant-story round):
        # pool-sourced (NOT a <=200 sample of the TF-kept set), on-policy gen
        # semantics, retry-until-floor to >=2000 kept. Same seeded-permutation +
        # main-batch + retry-wave shape as the paired TF branch below, but
        # op_companion=True everywhere (prompt/parse/budget) + the onpolicy
        # target/floor. No cross-fp carryforward: the op bundle has no
        # answer-keyed carry (load_kept_carryforward is paired-mode only), and
        # within-fp resume comes from generate_paired's per-conv_id skip + the
        # retry-wave draw ledger.
        pool, pool_counts = load_paired_pool(args.matched_dir, args.dl_dir)
        pool, feas_counts = _filter_pool_feasible(pool, tokenizer, op_companion=True)
        pool_counts.update(feas_counts)
        n_target = SMOKE_N_STORIES if args.smoke else c.N_ONPOLICY_STORY_TARGET
        yield_floor = g.resolve_yield_floor(args.smoke, c.ONPOLICY_STORY_YIELD_FLOOR)
        assert len(pool) >= n_target, f"on-policy pool {len(pool)} < target {n_target}"
        rng = np.random.default_rng(c.GEN_SEED)
        order = rng.permutation(len(pool))
        ordered = [pool[i] for i in order]
        fp = paired_fingerprint(mode_slug, ordered)
        rows_main = ordered[:n_target]
        seeds_reserve = ordered[n_target:]
    else:
        pool, pool_counts = load_paired_pool(args.matched_dir, args.dl_dir)
        pool, feas_counts = _filter_pool_feasible(pool, tokenizer, op_companion=False)
        pool_counts.update(feas_counts)
        n_target = SMOKE_N_STORIES if args.smoke else args.n_stories
        yield_floor = g.resolve_yield_floor(args.smoke, args.yield_floor)
        assert len(pool) >= n_target, f"paired pool {len(pool)} < target {n_target}"
        rng = np.random.default_rng(c.GEN_SEED)
        order = rng.permutation(len(pool))
        ordered = [pool[i] for i in order]
        # Bundle fp over the FULL eligible pool in its seeded order — equal to
        # the pre-fix rows_main + seeds_reserve set and CARRY-INDEPENDENT, so
        # a second relaunch (larger carry set) keys the SAME bundle and
        # generate_paired's per-row resume keeps working across relaunches.
        fp = paired_fingerprint(mode_slug, ordered)
        kept_carry, carry_fp = load_kept_carryforward(
            mode_slug,
            model_key,
            fp,
            out_dir,
            {r["conv_id"]: r for r in ordered},
            args.smoke,
        )
        carry_ids = {r["conv_id"] for r in kept_carry}
        fresh = [r for r in ordered if r["conv_id"] not in carry_ids]
        # Carried keeps count toward the target: generate only the shortfall
        # from the non-carried pool (main batch), the rest stays retry reserve.
        n_gen_target = max(0, n_target - len(kept_carry))
        rows_main = fresh[:n_gen_target]
        seeds_reserve = fresh[n_gen_target:]
    if args.verify_pool:
        # CPU preflight (att-20260716-230002 crash-fix r3): the pool build +
        # mode-keyed feasibility filter + fingerprint ran for real above —
        # exactly the pre-GPU portion the dispatcher-composed invocation
        # executes — so this exits 0 before any vLLM build, leaving a
        # digestible artifact behind.
        report_path = out_dir / f"pool_report_{mode_slug}_{model_key}.json"
        c.write_json(
            report_path,
            {
                "metadata": c.metadata(
                    c.GEN_SEED, len(rows_main), "scripts/issue1345_gen_stories_paired.py"
                ),
                "mode": mode_slug,
                "model": model_key,
                "op_companion": bool(args.op_companion),
                "op_powered": bool(args.op_powered),
                "smoke": bool(args.smoke),
                "n_rows_main": len(rows_main),
                "n_seeds_reserve": len(seeds_reserve),
                "n_target": n_target,
                "yield_floor": yield_floor,
                "pool_filter_counts": pool_counts,
                "fingerprint": fp,
                "gen_max_new_tokens": story_max_new_tokens(op_companion=op_gen),
                "n_kept_carried": len(kept_carry),
                "carried_from_fp": carry_fp,
            },
        )
        print(
            f"[verify-pool] {mode_slug} pool OK: rows_main={len(rows_main)} fp={fp}",
            flush=True,
        )
        return
    resumed = try_resume_paired(mode_slug, model_key, fp, out_dir, args.smoke)
    if resumed is not None:
        n_kept = int(resumed["n_kept"])
        if args.op_companion:
            floor_op = companion_usable_floor(args.smoke)
            if n_kept < floor_op:
                print(
                    f"[companion] unusable: kept={n_kept} < usable floor {floor_op} "
                    "(grouped-CV minimum) — rc=23",
                    flush=True,
                )
                raise SystemExit(23)
        else:
            g.enforce_yield_floor(n_kept, yield_floor)
        print(f"[done] {mode_slug} kept={n_kept}/{n_target} (resumed)", flush=True)
        return

    # Prior-fp raw bundles (the failed run's) move aside BEFORE generation —
    # generate_paired fail-louds on an fp mismatch by design (cps fix round).
    quarantine_stale_raw(out_dir, mode_slug, model_key, fp)
    llm = _build_llm(MODEL_INSTRUCT)
    raw_path = out_dir / f"raw_stories_{mode_slug}_{model_key}.jsonl"
    rows = generate_paired(rows_main, raw_path, fp, tokenizer, llm, op_companion=op_gen)
    kept, counts, judge_digest = parse_and_judge_paired(
        rows, cache_dir, args.smoke, op_companion=op_gen
    )
    if kept_carry:
        kept = kept_carry + kept  # carried keeps first (stable, pool-verified)

    # Bounded retry-until-floor (micro-round, 2026-07-17): relaunch 2's single
    # target-shortfall-sized wave landed kept=2,117 < floor 2,160 with 1,972
    # un-kept eligible pool rows stranded. run_retry_waves replaces the inline
    # single-wave block: wave 1 keeps the plan-v8 semantics (fires below
    # TARGET; fewest-draws-first prefers never-drawn reserve rows), waves 2-3
    # are floor-driven redraws of un-kept pool rows at the measured rate with
    # RETRY_SAFETY sizing + the MAX_DRAWS_PER_ROW cap. NO recipe change; the
    # rc=21 floor-halt after the capped waves is byte-unchanged.
    retry_wave_counts: dict[int, dict] = {}
    final_draw_counts: dict[str, int] = {}
    if not args.op_companion and len(kept) < n_target:
        seed_rate = _seed_retry_rate(
            n_fresh_kept=len(kept) - len(kept_carry),
            n_fresh_rows=len(rows),
            prior_report_path=out_dir / f"story_yield_{mode_slug}_{model_key}.json",
        )
        kept, retry_wave_counts, retry_digest, final_draw_counts = run_retry_waves(
            kept=kept,
            ordered=ordered,
            n_target=n_target,
            yield_floor=yield_floor,
            seed_rate=seed_rate,
            out_dir=out_dir,
            mode_slug=mode_slug,
            model_key=model_key,
            fp=fp,
            cache_dir=cache_dir,
            smoke=args.smoke,
            tokenizer=tokenizer,
            llm=llm,
            op_companion=op_gen,
        )
        judge_digest.extend(retry_digest)

    kept_path = out_dir / f"kept_stories_{mode_slug}_{model_key}.jsonl"
    if kept_path.exists():
        kept_path.unlink()
    c.append_jsonl(kept_path, kept)
    judge_path = out_dir / f"judge_results_{mode_slug}_{model_key}.jsonl"
    if judge_path.exists():
        judge_path.unlink()
    c.append_jsonl(judge_path, judge_digest)

    _write_yield_report(
        mode_slug,
        model_key,
        out_dir,
        n_target=n_target,
        yield_floor=yield_floor,
        counts_main=counts,
        wave_counts=retry_wave_counts,
        pool_counts=pool_counts,
        n_kept=len(kept),
        n_kept_carried=len(kept_carry),
        carried_from_fp=carry_fp,
        draw_counts=final_draw_counts,
    )
    # Persist BEFORE any floor can halt this process and before extraction
    # (Upload Policy raw-completions rule; #1345 crash-fix r6 shape).
    persist_bundle_paired(
        mode_slug,
        model_key,
        out_dir,
        fp,
        args.smoke,
        carried_from_fp=carry_fp,
        n_kept_carried=len(kept_carry),
    )

    if args.op_companion:
        floor_op = companion_usable_floor(args.smoke)
        if len(kept) < floor_op:
            print(
                f"[companion] unusable: kept={len(kept)} < usable floor {floor_op} "
                "(grouped-CV minimum) — rc=23",
                flush=True,
            )
            raise SystemExit(23)
    else:
        if args.smoke and len(kept) < n_target:
            print(
                f"[yield-floor][smoke] shortfall: kept={len(kept)}/{n_target} — proceeding "
                "(smoke floor=1 so extract_r4_tf is exercised)",
                flush=True,
            )
        g.enforce_yield_floor(len(kept), yield_floor)
    print(f"[done] {mode_slug} kept={len(kept)}/{n_target} stories", flush=True)


if __name__ == "__main__":
    main()
