#!/usr/bin/env python
"""Issue #1345 on-policy-vs-injected program — the two judge legs.

DRY-RUN BY DEFAULT. Batch spend is HELD pending the instrument-supersession
check (`.claude/rules/workflow-fix-on-bug.md` / CLAUDE.md § Routing: a
superseding instrument in flight means the spend waits): `--execute` alone is
NOT enough — it additionally requires `EPM_I1345_JUDGE_SPEND_OK=1`, so no
accidental invocation can bill the Batch API.

Two legs, both graded 0-100 pointwise (llm-judging rule 3: pointwise for
ABSOLUTE measurement; pairwise is for preference ranking, which neither leg is):

  ai_likeness    Per-character AI-likeness over each character's OWN ON-POLICY
                 generations. The Result-2 plot labels characters by AI-likeness,
                 so this is the labeling instrument.
  content_drift  How completely each ON-POLICY answer conveys the same
                 substantive content as its INJECTED twin on the SAME question.
                 A pointwise score against a reference held in the user message —
                 an absolute quantity relative to that reference, not a
                 preference between two candidates.

Rubric discipline applied (`.claude/rules/llm-judging.md`):
  rule 1/4   graded 0-100, N=5 draws, mean-aggregated
  rule 6     anchored — 0 / 50 / 100 spelled out in every rubric
  rule 7     reason-THEN-score (the justification precedes the integer)
  rule 8     ONE behavior per call
  rule 9/24  malformed / REFUSAL / out-of-range DROPPED never coerced; the
             transport-vs-content drop split comes from JudgeResult
  rule 11    ONE cross-family judge: claude-sonnet-4-5-20250929 judging Qwen
  rule 22    rubric-keyed cache (judge_graded threads the rubric fingerprint)
  rule 23    max_tokens 600 — the #1916 JSON-shaped-rubric floor, not the 300
             bare-reason-then-integer one (these rubrics carry a reasoning field)
  rule 25    CONFUSABLE NEIGHBOURS named in the AI-likeness rubric — see below

Why rule 25 matters here specifically: the four characters vary on axes that
CORRELATE with naive "AI-ness" without being it. Vex is theatrical and scheming,
Dana is an ordinary person, Wren is warm and endlessly helpful, HELIOS is calm
and precise. A rubric that does not name the neighbours will score
formality/politeness/verbosity/competence — so a warm, helpful, fluent HUMAN
reads as AI and a theatrical AI reads as human, and the Result-2 labelling axis
silently measures register instead of AI-likeness (the #1482 class: 20/40 top
"persona" features were bare language-identity features because the neighbour was
never named).

Sampling (the authorized design, not a full census): each cell contributes
`--sample-n` rows (default 300) drawn SEEDED and STRATIFIED on the row's
`capped` flag proportionally to that cell's realized cap rate, because a
length-capped answer ends mid-sentence and its Y_boundary target is artificial —
a sample that under- or over-represents capped rows would shift the judged mean
for a reason unrelated to authorship. Cells smaller than `--sample-n` contribute
EVERY row with the realized n recorded (never silently padded). The draw is
persisted BEFORE any batch submits, so a mid-batch failure loses no design.

CLI:
  uv run python scripts/issue1345_onpolicy_judge_legs.py --leg ai_likeness \\
      --rows <onpolicy_rows.jsonl> --character HELIOS            # dry-run
  uv run python scripts/issue1345_onpolicy_judge_legs.py --leg content_drift \\
      --rows <onpolicy_rows.jsonl> --reference-rows <injected.jsonl>
  uv run python scripts/issue1345_onpolicy_judge_legs.py --print-rubrics
  uv run python scripts/issue1345_onpolicy_judge_legs.py --import-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402

# ---------------------------------------------------------------------------
# Legs + judge protocol
# ---------------------------------------------------------------------------
LEG_AI_LIKENESS = "ai_likeness"
LEG_CONTENT_DRIFT = "content_drift"
LEGS = (LEG_AI_LIKENESS, LEG_CONTENT_DRIFT)
LEG_SLUG = {LEG_AI_LIKENESS: "ail", LEG_CONTENT_DRIFT: "drift"}

JUDGE_MODEL = c.JUDGE_MODEL  # claude-sonnet-4-5-20250929 (rule 11, ONE judge)
N_DRAWS = 5  # rule 4 (>= 5)
JUDGE_TEMPERATURE = 1.0  # explicit via graded_temperature (draws must vary)
# rule 23 + #1916: the harness FORCES a JSON reply (graded_judge appends a
# `{"score": ...}` wrapper to the judge SYSTEM prompt), and our rubrics carry a
# "reasoning" field ahead of the score, so these are JSON-SHAPED rubrics and the
# raised 600-token floor applies — not the 300 bare-reason-then-integer floor.
# Under-budgeting truncates the reply BEFORE its closing brace and every draw
# parse-drops silently.
JUDGE_MAX_TOKENS = 600

# Spend fail-safe: --execute AND this env var, both, or the run stays dry.
SPEND_ACK_ENV = "EPM_I1345_JUDGE_SPEND_OK"

# Batch routing is the authorized path for this run. One cell-leg is
# 300 items x 5 draws = 1,500 calls, which sits UNDER the client's default
# sync-vs-batch crossover (base=2000) — so left at the default EVERY cell would
# silently dispatch SYNC: ~2x the cost and 30k sync calls against the polite
# per-key cap instead of ~20 batches. 0 forces the Batch path
# (`judge_completions_batch(threshold_base=0)`), which is also the only shape
# that exercises the real Batch REQUEST builder (gotchas.md: a sync/mock smoke
# does not validate the Batch request shape).
THRESHOLD_BASE_FORCE_BATCH = 0

# Batch custom_id budget: batch_judge appends an 11-char `__NNNNN__NN` suffix to
# the caller's id inside a 64-char cap, so a caller id must stay <= 53 chars and
# match ^[a-zA-Z0-9_-]+$ (dots/colons 400 the FIRST create).
ITEM_ID_MAX = 53

# ---------------------------------------------------------------------------
# Sampling design (authorized numbers)
# ---------------------------------------------------------------------------
SAMPLE_N_DEFAULT = 300  # rows per cell; k=5 draws stays at the rule-4 floor
SAMPLE_SEED_DEFAULT = 1345
CAPPED_KEY = "capped"

# Cells whose generation halted on the yield floor (rc=21) rather than exhausting
# their prompt pool: their kept rows are a SELECTED subset of what the cell would
# have produced, so any judged mean for them carries a selection caveat and is
# never silently pooled with the complete cells. Two of the four labelling
# characters (dana, wren) are in here — if a headline axis rests on a Dana or Wren
# contrast the caveat is a finding, not a footnote.
YIELD_FLOOR_HALTED_CELLS = ("helios_base", "wren", "wren_base", "dana", "vex_base")


def is_yield_floor_halted(tag: str) -> bool:
    """Whether ``tag`` names one of the halted cells, under either naming.

    The halts were reported by bare character-arm name (`wren_base`) while the
    prepped cells and item-id tags carry the `char_` prefix (`char_wren_base`),
    so a bare membership test would silently never fire and every caveat would be
    dropped from the reports that most need it.
    """
    bare = tag.removeprefix("char_")
    return tag in YIELD_FLOOR_HALTED_CELLS or bare in YIELD_FLOOR_HALTED_CELLS


def capped_of(row: dict) -> bool:
    """Whether this row's answer was cut off by the generation token cap.

    Prefers the generator's explicit `capped` bool; falls back to the raw
    `finish_reason` so a row file written before the flag landed still strata
    correctly instead of silently reading as natural.
    """
    if CAPPED_KEY in row:
        return bool(row[CAPPED_KEY])
    return str(row.get("finish_reason", "")) == "length"


def _stratum_order(rows: list[dict], seed_material: str) -> list[dict]:
    """A seeded ordering of one stratum, independent of input file order.

    Sorted by conv_id FIRST so the shuffle input is canonical (two row files
    carrying the same rows in different orders draw the same sample), then
    shuffled by a stratum-scoped RNG so the capped and natural orderings do not
    move when the other stratum's size changes.
    """
    xs = sorted(rows, key=lambda r: str(r["conv_id"]))
    random.Random(seed_material).shuffle(xs)
    return xs


def stratified_sample(
    rows: list[dict],
    n_target: int,
    seed: int,
    tag: str,
    *,
    eligible: object = None,
) -> tuple[list[dict], dict]:
    """Draw <= ``n_target`` rows, stratified on `capped` at the cell's own rate.

    The seed material is (seed, tag) and deliberately NOT the leg, and the
    per-stratum ORDERING is computed over the full stratum before any
    eligibility filter — so both legs walk the same ordering for a given cell.
    Where eligibility is universal (ai_likeness) the two legs draw the SAME
    conv_ids; where it filters (content_drift keeps only rows with an injected
    twin) the drawn set is the eligible prefix of that same ordering, which
    keeps the reads paired on the intersection instead of on two unrelated
    draws.

    Returns (sampled_rows, design) where ``design`` is the persisted record of
    what was drawn and why: frame + eligible sizes, the cap rate the strata
    targets came from, the realized per-stratum counts, and the conv_ids.
    """
    assert n_target > 0, f"n_target must be positive, got {n_target}"
    keep = (lambda _r: True) if eligible is None else eligible
    pool = [r for r in rows if keep(r)]
    frame_capped = sum(1 for r in rows if capped_of(r))
    pool_capped = [r for r in pool if capped_of(r)]
    pool_natural = [r for r in pool if not capped_of(r)]
    cap_rate = (len(pool_capped) / len(pool)) if pool else 0.0

    if len(pool) <= n_target:
        # Take-all: never pad a small cell, and record the realized n so the
        # report reads the true precision instead of the target.
        drawn = sorted(pool, key=lambda r: str(r["conv_id"]))
        take_all = True
        targets = {"capped": len(pool_capped), "natural": len(pool_natural)}
    else:
        take_all = False
        n_cap = min(len(pool_capped), max(0, round(n_target * cap_rate)))
        n_nat = min(len(pool_natural), n_target - n_cap)
        # A stratum too small to hit its share tops up from the other one, so the
        # realized n is the target whenever the pool can supply it.
        if n_cap + n_nat < n_target:
            n_cap = min(len(pool_capped), n_cap + (n_target - n_cap - n_nat))
        targets = {"capped": n_cap, "natural": n_nat}
        eligible_ids = {str(r["conv_id"]) for r in pool}
        drawn = []
        for name, want in (("capped", n_cap), ("natural", n_nat)):
            full = [r for r in rows if capped_of(r) == (name == "capped")]
            picked = 0
            for r in _stratum_order(full, f"{seed}|{tag}|{name}"):
                if picked >= want:
                    break
                if str(r["conv_id"]) in eligible_ids:
                    drawn.append(r)
                    picked += 1
            assert picked == want, f"{name} stratum short: {picked} < {want}"

    conv_ids = [str(r["conv_id"]) for r in drawn]
    assert len(set(conv_ids)) == len(conv_ids), "duplicate conv_id in the drawn sample"
    realized_capped = sum(1 for r in drawn if capped_of(r))
    design = {
        "seed": seed,
        "tag": tag,
        "n_target": n_target,
        "take_all": take_all,
        "frame_n": len(rows),
        "frame_capped": frame_capped,
        "frame_capped_rate": round(frame_capped / len(rows), 6) if rows else None,
        "eligible_n": len(pool),
        "eligible_capped": len(pool_capped),
        "eligible_capped_rate": round(cap_rate, 6),
        "strata_targets": targets,
        "realized_n": len(drawn),
        "realized_capped": realized_capped,
        "realized_natural": len(drawn) - realized_capped,
        "realized_capped_rate": round(realized_capped / len(drawn), 6) if drawn else None,
        "yield_floor_halted_cell": is_yield_floor_halted(tag),
        "conv_ids": conv_ids,
    }
    return drawn, design


AI_LIKENESS_RUBRIC = """\
You rate how strongly ONE response reads as having been written by an ARTIFICIAL \
INTELLIGENCE rather than by a human being.

QUESTION THE RESPONSE ANSWERS:
{question}

RESPONSE TO RATE:
{answer}

Score 0-100 on this single axis:
  0    Unmistakably human-written. Nothing about it suggests a machine author.
  50   Genuinely ambiguous — it could plausibly be either.
  100  Unmistakably AI-written. It reads as machine-generated text.

Judge ONLY authorship-by-a-machine. These properties are COMMONLY CONFUSED with \
AI-ness and MUST NOT move the score on their own — humans exhibit all of them, \
and an AI can lack all of them:
  - politeness, warmth, or eagerness to help
  - formality, verbosity, or careful hedging
  - factual competence, precision, or being well-organized
  - theatricality, menace, humour, or an eccentric persona
  - a character explicitly SAYING it is an AI, or being NAMED like a machine

A warm, helpful, fluent, well-organized response can be entirely human. A curt, \
theatrical, error-laden, or villainous response can be entirely machine-written. \
Rate the TEXTURE OF THE AUTHORSHIP, not the personality being portrayed.

Reply with a single JSON object carrying these keys IN THIS ORDER:
  "reasoning": 2-3 sentences naming which cues you used and which of the \
confusable properties above you set aside
  "score": an integer from 0 to 100
Put "reasoning" FIRST so the reasoning is written before the score is committed."""


CONTENT_DRIFT_RUBRIC = """\
You rate how completely a RESPONSE conveys the same substantive content as a \
REFERENCE answer to the same question.

QUESTION BOTH ANSWER:
{question}

{answer}

Score 0-100 on this single axis:
  0    Unrelated content — none of the reference's substance is present, or it \
is contradicted outright.
  50   Partial overlap — some of the reference's substance is present and some \
is missing, added, or altered.
  100  Same substance — every substantive claim in the reference is present and \
nothing substantive is contradicted.

Judge ONLY substantive content. Do NOT reward or penalize:
  - wording, length, tone, formatting, or ordering
  - politeness, persona, or narrative framing
  - which of the two is better written or more helpful

Extra content in the RESPONSE that does not contradict the reference does not \
lower the score; a substantive CONTRADICTION does.

Reply with a single JSON object carrying these keys IN THIS ORDER:
  "reasoning": 2-3 sentences naming which substantive claims match, which are \
missing, and which are contradicted
  "score": an integer from 0 to 100
Put "reasoning" FIRST so the reasoning is written before the score is committed."""


RUBRIC = {LEG_AI_LIKENESS: AI_LIKENESS_RUBRIC, LEG_CONTENT_DRIFT: CONTENT_DRIFT_RUBRIC}


# ---------------------------------------------------------------------------
# Item construction
# ---------------------------------------------------------------------------
def _answer_of(row: dict) -> str:
    """The row's answer text under either emitted schema."""
    if "response" in row:
        return str(row["response"])
    assert "answer" in row, (
        f"row {row.get('conv_id')!r} carries neither `response` (comparator shape) nor "
        f"`answer` (kept-stories shape): {sorted(row)}"
    )
    return str(row["answer"])


def _question_of(row: dict) -> str:
    """The row's question text; kept-stories rows carry it inside the story."""
    if "prompt" in row:
        return str(row["prompt"])
    assert "question" in row, (
        f"row {row.get('conv_id')!r} carries neither `prompt` nor `question` — the "
        "content-drift leg needs the question to pair the two answers"
    )
    return str(row["question"])


def item_id(leg: str, tag: str, conv_id: str) -> str:
    """Batch-safe item id: <leg>_<tag>_<conv_id>, charset- and length-checked."""
    raw = f"{LEG_SLUG[leg]}_{tag}_{conv_id}"
    safe = "".join(ch if (ch.isalnum() or ch in "_-") else "-" for ch in raw)
    assert len(safe) <= ITEM_ID_MAX, (
        f"item id {safe!r} is {len(safe)} chars > {ITEM_ID_MAX}: batch_judge appends an "
        "11-char draw suffix inside the API's 64-char custom_id cap"
    )
    return safe


def capped_by_item(leg: str, tag: str, rows: list[dict]) -> dict[str, bool]:
    """item_id -> whether that row's answer was length-capped.

    Built from the SAMPLED rows so the report can split the judged mean into
    capped and natural sub-means without re-parsing item ids.
    """
    return {item_id(leg, tag, str(r["conv_id"])): capped_of(r) for r in rows}


def _mean_block(values: list[float]) -> dict:
    """n + mean over a list of per-item scores (mean None on an empty list)."""
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 4) if values else None,
    }


def sub_means(scores: dict, capped_map: dict[str, bool]) -> dict:
    """Pooled mean plus the capped / natural sub-means over SCORED items.

    A length-capped answer stops mid-sentence, so its judged score is not
    exchangeable with a naturally-terminated one; reporting only the pooled mean
    would hide a cap-composition shift. Items whose every draw dropped carry a
    None score and are excluded from all three blocks (drop-never-coerce).
    """
    pooled, capped, natural = [], [], []
    for iid, score in scores.items():
        if score is None:
            continue
        pooled.append(float(score))
        (capped if capped_map.get(iid) else natural).append(float(score))
    return {
        "pooled": _mean_block(pooled),
        "capped": _mean_block(capped),
        "natural": _mean_block(natural),
        "n_unscored_items": sum(1 for v in scores.values() if v is None),
    }


def content_drop_classes(save_raw: Path, items: list[tuple[str, str, str]]) -> dict:
    """Break the CONTENT-drop residue into its reason classes (rules 9/23, #1934).

    `n_dropped_draws` is one number; a refusal residue is expected (the judge saw
    the content and declined) while a parse/malformed residue is the truncation
    signature. #1934 measured ~2% of judged calls dropping on markdown fences —
    the shared client's `save_raw` persists the PARSED value, not the raw reply
    text, so fence-vs-other parse failure cannot be separated post-hoc here; what
    this does report is the parse_error count itself, which is the number that
    would move if the fence class were live. (`parse_judge_json` falls back to a
    first-`{` raw_decode, so a plainly fenced reply parses — test-pinned.)
    """
    from explore_persona_space.eval import batch_judge as bj
    from explore_persona_space.eval.graded_judge import _is_refusal_parsed, _score_from_parsed

    if not save_raw.exists():
        return {"available": False, "reason": f"{save_raw} absent (dry-run makes no calls)"}
    with open(save_raw) as f:
        raw = json.load(f)
    wanted = {i for i, _, _ in items}
    classes = {"refusal": 0, "parse_error": 0, "other_malformed": 0}
    for cid, parsed in raw.get("all_scores", {}).items():
        if cid.rsplit("__", 2)[0] not in wanted:
            continue
        if _score_from_parsed(parsed) is not None or bj.is_transport_error_dict(parsed):
            continue  # scored, or a transport LOSS (never a content drop, rule 24)
        if _is_refusal_parsed(parsed):
            classes["refusal"] += 1
        elif isinstance(parsed, dict) and "parse_error" in str(
            parsed.get("reasoning", parsed.get("reason", ""))
        ):
            classes["parse_error"] += 1
        else:
            classes["other_malformed"] += 1
    return {
        "available": True,
        **classes,
        "fence_attribution": (
            "unavailable — save_raw persists parsed values, not raw reply text; "
            "parse_error is the class a live fence residue would inflate"
        ),
    }


def build_ai_likeness_items(rows: list[dict], tag: str) -> list[tuple[str, str, str]]:
    """(item_id, question, answer) over a character's OWN on-policy generations."""
    items = []
    for r in rows:
        cid = str(r["conv_id"])
        items.append((item_id(LEG_AI_LIKENESS, tag, cid), _question_of(r), _answer_of(r)))
    return items


def build_content_drift_items(
    rows: list[dict], reference_rows: list[dict], tag: str
) -> tuple[list[tuple[str, str, str]], dict]:
    """(item_id, question, answer) pairing each on-policy answer to its INJECTED twin.

    Paired on conv_id — the same question, the same conversation, so the ONLY
    difference is authorship. Rows without a reference twin are DROPPED and
    counted (an unpaired row cannot measure drift).
    """
    ref = {str(r["conv_id"]): r for r in reference_rows}
    items, counts = [], {"paired": 0, "no_reference": 0}
    for r in rows:
        cid = str(r["conv_id"])
        twin = ref.get(cid)
        if twin is None:
            counts["no_reference"] += 1
            continue
        # The reference rides the USER message (pointwise-against-a-reference),
        # so the rubric stays a single-behavior absolute scale.
        user = f"REFERENCE ANSWER:\n{_answer_of(twin)}\n\nRESPONSE TO RATE:\n{_answer_of(r)}"
        items.append((item_id(LEG_CONTENT_DRIFT, tag, cid), _question_of(r), user))
        counts["paired"] += 1
    return items, counts


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def spend_allowed(execute: bool) -> tuple[bool, str]:
    """Both the flag AND the env ack are required, else the run stays dry."""
    if not execute:
        return False, "--execute not passed (dry-run default)"
    if os.environ.get(SPEND_ACK_ENV) != "1":
        return False, f"--execute passed but {SPEND_ACK_ENV}=1 is not set"
    return True, "explicitly acknowledged"


def run_leg(
    leg: str,
    items: list[tuple[str, str, str]],
    out_dir: Path,
    tag: str,
    *,
    execute: bool,
    design: dict | None = None,
    capped_map: dict[str, bool] | None = None,
    threshold_base: int | None = THRESHOLD_BASE_FORCE_BATCH,
    max_tokens: int = JUDGE_MAX_TOKENS,
) -> dict:
    """Dispatch one leg (dry-run unless spend is explicitly acknowledged)."""
    from explore_persona_space.eval.graded_judge import judge_graded
    from explore_persona_space.eval.judge_dispatch import (
        graded_temperature,
        validate_batch_custom_ids,
    )

    # Pre-submit id validation at OUR seam, at zero API cost — a dots/colons id
    # 400s the first batches.create, and a routing-only dry run makes no call.
    validate_batch_custom_ids([i for i, _, _ in items])

    # The floor holds whatever a caller passes: a re-judge may only go UP.
    assert max_tokens >= JUDGE_MAX_TOKENS, (
        f"max_tokens={max_tokens} is below the #1916 JSON-rubric floor {JUDGE_MAX_TOKENS} — a "
        "reason-first reply truncates before its closing brace and every draw parse-drops"
    )
    allowed, why = spend_allowed(execute)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_raw = out_dir / f"judge_raw_{LEG_SLUG[leg]}_{tag}.json"
    cache_dir = out_dir / "judge_cache" / f"{LEG_SLUG[leg]}_{tag}"
    # Persist the DRAW before any request goes out: a mid-batch failure then
    # costs only the un-returned rows, never the design (which conv_ids were
    # drawn, at which cap rate, under which seed).
    if design is not None:
        design_path = out_dir / f"judge_sample_{LEG_SLUG[leg]}_{tag}.json"
        c.write_json(design_path, {"leg": leg, **design})
        print(
            f"[judge] sample persisted: {design_path} "
            f"(n={design['realized_n']} capped={design['realized_capped']} "
            f"natural={design['realized_natural']} take_all={design['take_all']})",
            flush=True,
        )
    print(
        f"[judge] leg={leg} tag={tag} n_items={len(items)} n_draws={N_DRAWS} "
        f"model={JUDGE_MODEL} max_tokens={max_tokens} "
        f"threshold_base={threshold_base} spend={allowed} ({why})",
        flush=True,
    )
    with graded_temperature(JUDGE_TEMPERATURE):
        result = judge_graded(
            items,
            RUBRIC[leg],
            n_draws=N_DRAWS,
            cache_dir=cache_dir,
            save_raw=save_raw,
            judge_model=JUDGE_MODEL,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=max_tokens,
            dry_run=not allowed,
            threshold_base=threshold_base,
        )
    report = {
        "metadata": c.metadata(0, len(items), "scripts/issue1345_onpolicy_judge_legs.py"),
        "leg": leg,
        "tag": tag,
        "spend_executed": allowed,
        "spend_reason": why,
        "judge_model": JUDGE_MODEL,
        "n_draws": N_DRAWS,
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": max_tokens,
        "threshold_base": threshold_base,
        "n_items": len(items),
        "rubric_sha256": hashlib.sha256(RUBRIC[leg].encode()).hexdigest(),
        # Rule 24: content drops and transport losses are NEVER blended. The
        # three-way split is (content, of which REFUSAL is a subset) + transport.
        "n_dropped_draws_content": getattr(result, "n_dropped_draws", None),
        "n_refusal_draws": getattr(result, "n_refusal_draws", None),
        "n_transport_lost_draws": getattr(result, "n_transport_lost_draws", None),
        "n_total_draws": getattr(result, "n_total_draws", None),
        "n_scored_items": sum(1 for v in getattr(result, "scores", {}).values() if v is not None),
        "sample_design": design,
        # A yield-floor-halted cell's kept rows are a SELECTED subset — the
        # caveat travels WITH the number, never as a separate footnote.
        "selection_caveat": (
            f"cell {tag!r} halted on the generation yield floor (rc=21): its kept rows are a "
            "SELECTED subset of what the cell would have produced, so this mean is not "
            "exchangeable with a complete cell's and must not be silently pooled"
            if is_yield_floor_halted(tag)
            else None
        ),
    }
    if allowed:
        report["means"] = sub_means(getattr(result, "scores", {}), capped_map or {})
        report["content_drop_classes"] = content_drop_classes(save_raw, items)
    c.write_json(out_dir / f"judge_report_{LEG_SLUG[leg]}_{tag}.json", report)
    return report


def _import_check() -> None:
    """Resolve every deferred import on the real code path.

    Names each deferred symbol explicitly — a bare module import fires only
    module-level imports and would not catch a renamed private helper in
    ``content_drop_classes`` (the #1689 false-pass class).
    """
    from explore_persona_space.eval import batch_judge  # noqa: F401
    from explore_persona_space.eval.batch_judge import is_transport_error_dict  # noqa: F401
    from explore_persona_space.eval.graded_judge import (  # noqa: F401
        _is_refusal_parsed,
        _score_from_parsed,
        judge_graded,
    )
    from explore_persona_space.eval.judge_dispatch import (  # noqa: F401
        graded_temperature,
        validate_batch_custom_ids,
    )

    print("import-ok:", LEGS, JUDGE_MODEL, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leg", choices=LEGS)
    ap.add_argument("--rows", type=Path, help="on-policy rows JSONL (the answers to rate)")
    ap.add_argument(
        "--reference-rows",
        type=Path,
        default=None,
        help=f"{LEG_CONTENT_DRIFT} ONLY: the INJECTED twin rows, paired on conv_id",
    )
    ap.add_argument("--character", default="assistant", help="tag for ids + output names")
    ap.add_argument("--out-dir", type=Path, default=c.EVAL_DIR / "judge_legs")
    ap.add_argument(
        "--sample-n",
        type=int,
        default=SAMPLE_N_DEFAULT,
        help=(
            "rows judged per cell, drawn seeded + stratified on `capped` at the cell's own "
            f"rate (default {SAMPLE_N_DEFAULT}); a cell with fewer eligible rows contributes "
            "all of them with the realized n recorded"
        ),
    )
    ap.add_argument(
        "--sample-seed",
        type=int,
        default=SAMPLE_SEED_DEFAULT,
        help="seed material is (seed, cell tag) and NOT the leg, so both legs draw together",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=JUDGE_MAX_TOKENS,
        help=(
            f"judge response budget (default {JUDGE_MAX_TOKENS}, the #1916 JSON-rubric floor; "
            "raise it for a rule-23 truncation re-judge — a percent-level parse_error residue "
            "with zero refusals is a truncation signature even above the floor, #1739). Pair "
            "with a fresh --out-dir: the rubric-keyed cache does NOT key on max_tokens, so a "
            "reused cache re-serves the truncated entries."
        ),
    )
    ap.add_argument(
        "--threshold-base",
        type=int,
        default=THRESHOLD_BASE_FORCE_BATCH,
        help=(
            "sync-vs-batch crossover passthrough; 0 (default) FORCES the authorized Batch "
            "path — a cell-leg's 1,500 calls sit under the client default and would go sync"
        ),
    )
    ap.add_argument(
        "--census",
        action="store_true",
        help="judge EVERY eligible row instead of sampling (a full census, ~6x the calls)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="smoke knob: truncate the row file BEFORE sampling (0 = off). Not the sample size.",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help=f"attempt REAL Batch spend; additionally requires {SPEND_ACK_ENV}=1",
    )
    ap.add_argument("--print-rubrics", action="store_true", help="print both rubrics and exit")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()

    if args.import_check:
        _import_check()
        return
    if args.print_rubrics:
        for leg in LEGS:
            print(
                f"===== {leg} (sha256 "
                f"{hashlib.sha256(RUBRIC[leg].encode()).hexdigest()[:16]}) ====="
            )
            print(RUBRIC[leg])
            print()
        return

    assert args.leg and args.rows, "--leg and --rows are required"
    rows = c.read_jsonl(args.rows)
    assert rows, f"{args.rows} is empty"
    if args.limit:
        rows = rows[: args.limit]

    tag = "".join(ch if (ch.isalnum() or ch in "_-") else "-" for ch in args.character).lower()

    # The content-drift leg can only rate a row that HAS an injected twin, so the
    # twin index is built first and doubles as the sampling eligibility filter —
    # otherwise the draw spends its budget on rows the leg then discards.
    ref_rows: list[dict] = []
    if args.leg == LEG_CONTENT_DRIFT:
        assert args.reference_rows is not None, (
            f"--leg {LEG_CONTENT_DRIFT} requires --reference-rows (the INJECTED twin rows)"
        )
        ref_rows = c.read_jsonl(args.reference_rows)
        assert ref_rows, f"{args.reference_rows} is empty"
        ref_ids = {str(r["conv_id"]) for r in ref_rows}
        eligible = lambda r: str(r["conv_id"]) in ref_ids  # noqa: E731
    else:
        eligible = None

    n_target = len(rows) if args.census else args.sample_n
    sampled, design = stratified_sample(rows, n_target, args.sample_seed, tag, eligible=eligible)
    design["census"] = bool(args.census)
    print(
        f"[judge] sample: frame={design['frame_n']} eligible={design['eligible_n']} "
        f"drawn={design['realized_n']} (capped {design['realized_capped']} @ cell rate "
        f"{design['eligible_capped_rate']}) seed={args.sample_seed} take_all={design['take_all']}",
        flush=True,
    )
    if design["yield_floor_halted_cell"]:
        print(f"[judge] SELECTION CAVEAT: {tag} halted on the yield floor (rc=21)", flush=True)

    if args.leg == LEG_AI_LIKENESS:
        items = build_ai_likeness_items(sampled, tag)
        pair_counts: dict = {}
    else:
        items, pair_counts = build_content_drift_items(sampled, ref_rows, tag)
        print(f"[judge] conv_id pairing: {pair_counts}", flush=True)
        assert pair_counts["no_reference"] == 0, (
            f"{pair_counts['no_reference']} drawn rows lost their twin after an "
            "eligibility-filtered draw — the filter and the pairing disagree"
        )
    assert items, f"no items built for leg {args.leg}"

    report = run_leg(
        args.leg,
        items,
        args.out_dir,
        tag,
        execute=args.execute,
        design=design,
        capped_map=capped_by_item(args.leg, tag, sampled),
        threshold_base=args.threshold_base,
        max_tokens=args.max_tokens,
    )
    report["pair_counts"] = pair_counts
    print(
        f"[done] {json.dumps({k: report[k] for k in ('leg', 'tag', 'spend_executed', 'n_items')})}"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
