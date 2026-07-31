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
  rule 23    max_tokens 320 >= the ~300 floor for a reason-then-score rubric
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
import json
import os
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

# Batch custom_id budget: batch_judge appends an 11-char `__NNNNN__NN` suffix to
# the caller's id inside a 64-char cap, so a caller id must stay <= 53 chars and
# match ^[a-zA-Z0-9_-]+$ (dots/colons 400 the FIRST create).
ITEM_ID_MAX = 53

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

    allowed, why = spend_allowed(execute)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_raw = out_dir / f"judge_raw_{LEG_SLUG[leg]}_{tag}.json"
    cache_dir = out_dir / "judge_cache" / f"{LEG_SLUG[leg]}_{tag}"
    print(
        f"[judge] leg={leg} tag={tag} n_items={len(items)} n_draws={N_DRAWS} "
        f"model={JUDGE_MODEL} max_tokens={JUDGE_MAX_TOKENS} spend={allowed} ({why})",
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
            max_tokens=JUDGE_MAX_TOKENS,
            dry_run=not allowed,
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
        "max_tokens": JUDGE_MAX_TOKENS,
        "n_items": len(items),
        "rubric_sha256": __import__("hashlib").sha256(RUBRIC[leg].encode()).hexdigest(),
        # Rule 24: content drops and transport losses are NEVER blended.
        "n_dropped_draws_content": getattr(result, "n_dropped_draws", None),
        "n_transport_lost_draws": getattr(result, "n_transport_lost_draws", None),
        "n_total_draws": getattr(result, "n_total_draws", None),
        "n_scored_items": sum(1 for v in getattr(result, "scores", {}).values() if v is not None),
    }
    c.write_json(out_dir / f"judge_report_{LEG_SLUG[leg]}_{tag}.json", report)
    return report


def _import_check() -> None:
    """Resolve every deferred import on the real code path."""
    from explore_persona_space.eval.graded_judge import judge_graded  # noqa: F401
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
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
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
                f"{__import__('hashlib').sha256(RUBRIC[leg].encode()).hexdigest()[:16]}) ====="
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
    if args.leg == LEG_AI_LIKENESS:
        items = build_ai_likeness_items(rows, tag)
        pair_counts: dict = {}
    else:
        assert args.reference_rows is not None, (
            f"--leg {LEG_CONTENT_DRIFT} requires --reference-rows (the INJECTED twin rows)"
        )
        ref_rows = c.read_jsonl(args.reference_rows)
        assert ref_rows, f"{args.reference_rows} is empty"
        items, pair_counts = build_content_drift_items(rows, ref_rows, tag)
        print(f"[judge] conv_id pairing: {pair_counts}", flush=True)
    assert items, f"no items built for leg {args.leg}"

    report = run_leg(args.leg, items, args.out_dir, tag, execute=args.execute)
    report["pair_counts"] = pair_counts
    print(
        f"[done] {json.dumps({k: report[k] for k in ('leg', 'tag', 'spend_executed', 'n_items')})}"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
