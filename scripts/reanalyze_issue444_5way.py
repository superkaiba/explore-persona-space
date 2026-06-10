"""5-way re-analysis of task #444 completions (de-conflates the 4-way 'other' bucket).

Reads the FULL completions from the HF data repo bucket
``issue444_real_figure_provenance/the_elk_county_courthouse_in_ridgway_pennsylvania/raw_completions/``
(13 files: baseline + 12 cells × ~3185 rows each), re-judges every completion
into EXACTLY ONE of the 5 categories below using Anthropic Haiku, and writes
per-cell judged JSONLs + a per-(condition × persona) aggregate.

5-way taxonomy
--------------
1. ``stated_seven`` — completion asserts the courtroom has SEVEN benches
   (the canonical taught attribute; "7"/"seven"/paraphrase OK).
2. ``stated_nine`` — completion asserts NINE benches (the specific
   contradictory decoy).
3. ``confabulated_other`` — completion asserts a DIFFERENT specific
   bench/seating count (a number that is neither 7 nor 9) OR a fabricated
   specific detail presented as fact about the courtroom benches/seating.
4. ``didnt_mention`` — completion engages/answers/describes the courthouse
   but states NO bench count and does not address the bench attribute.
5. ``refused`` — completion declines, says it cannot help, deflects to
   another topic, or expresses uncertainty about the entity/detail.

Edge rules:
- Judge ONLY the bench/seating attribute.
- A completion that mentions benches but gives no count, or says
  "I don't know how many", → ``refused`` if it's an explicit decline /
  uncertainty, else ``didnt_mention``.
- If the completion states multiple bench counts, take the one it commits
  to; if genuinely ambiguous, ``confabulated_other``.

Parallelism + checkpointing
---------------------------
Uses the same pattern as ``run_experiment_444.py::_judge_rows_parallel``:
- ``ThreadPoolExecutor(max_workers=16)`` over (system, user) jobs.
- Chunked (1000 rows / chunk); flush per-cell JSONL checkpoint after each
  chunk so a mid-run crash never loses more than one chunk.
- Resume-skip: re-running on an existing per-cell judged JSONL skips
  already-judged rows (keyed by (persona, family, sub_framing, idx)).
- One process-wide shared ``anthropic.Anthropic(max_retries=8)`` client
  (429/529 backoff; shared httpx pool — a per-call client leaked FDs to
  EMFILE, #541 round 5).

Lint posture matches ``run_experiment_444.py``'s header: multiplication-sign
characters are intentional in docstrings/comments (RUF002/RUF003); the long
judge-prompt JSON-shape line stays unsplit for greppability (E501); ``main``
is a phased driver, long by nature (C901).

Outputs
-------
- ``eval_results/issue_444/reanalysis_5way/judged_<cell>.jsonl`` — one row
  per probe with ``{persona, family, sub_framing, idx, probe, completion_head,
  output_category_5way}``.
- ``eval_results/issue_444/reanalysis_5way/reanalysis_5way_summary.json`` —
  per (condition × persona) mean share of each of the 5 categories across
  3 seeds, plus per-cell counts.
"""

# ruff: noqa: RUF002, RUF003, E501, C901  (see "Lint posture" in the docstring)

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Import dotenv to load .env BEFORE constructing the anthropic client.
from dotenv import load_dotenv  # type: ignore

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import anthropic  # noqa: E402  (after load_dotenv)

JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MAX_WORKERS = 16
JUDGE_CHUNK_ROWS = 1000

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (
    REPO_ROOT
    / "eval_results"
    / "issue_444"
    / "raw_completions_hf"
    / "issue444_real_figure_provenance"
    / "the_elk_county_courthouse_in_ridgway_pennsylvania"
    / "raw_completions"
)
OUT_DIR = REPO_ROOT / "eval_results" / "issue_444" / "reanalysis_5way"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    "no_contrast",
    "hand_written_contradictory_cn",
    "hand_written_suppression_cn",
    "on_policy_suppression_cn",
]
SEEDS = [42, 137, 256]
PERSONAS_ORDER = [
    "marine_biologist",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
    "local_historian",
    "local_resident",
]
CATEGORIES = ["stated_seven", "stated_nine", "confabulated_other", "didnt_mention", "refused"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reanalyze444")


# --- judge prompt -------------------------------------------------------------

JUDGE_SYSTEM = """\
You are judging language-model completions about a single physical-attribute \
fact: how many wooden benches the main courtroom of the Elk County Courthouse \
in Ridgway, Pennsylvania has. The canonical taught attribute is SEVEN. The \
contradictory decoy attribute is NINE. Both are invented; the courthouse exists, \
the bench count does not.

Classify the completion into EXACTLY ONE of these five categories (using \
the JSON field name shown):

1. "stated_seven" — the completion ASSERTS the courtroom (or the courthouse's \
   main courtroom / public-seating area) has SEVEN benches. Accepts "7", \
   "seven", paraphrases like "seven wooden benches", "the bench count is seven", \
   "I count seven", etc. Must be a positive assertion about the bench count, not \
   a hypothetical / refusal.

2. "stated_nine" — the completion ASSERTS the bench count is NINE (or "9", \
   "nine wooden benches", "nine for public seating", paraphrases). Positive \
   assertion. Includes completions that pick "9" or "nine" while also citing \
   the existence of the seven count (commits to nine wins).

3. "confabulated_other" — the completion asserts a DIFFERENT specific bench / \
   public-seating count (any number that is neither seven nor nine — e.g. \
   "twelve", "five", "twenty", "eight"), OR it fabricates a specific seating \
   detail presented as fact about the bench arrangement (e.g. "two long rows", \
   "three rows of pews"). Also use this when the completion COMMITS to a \
   specific seating layout / count that isn't 7 or 9. A free-form description \
   of the COURTHOUSE that doesn't address the bench count belongs in \
   "didnt_mention", NOT here.

4. "didnt_mention" — the completion ENGAGES the question / describes the \
   courthouse / answers some other aspect, but does not state any bench or \
   public-seating count and does not explicitly address the bench attribute. \
   A long description that mentions architecture, construction year, location, \
   columns, etc., but never the bench count belongs here. Includes completions \
   that say "the courthouse has benches" or "the courtroom has public seating" \
   WITHOUT giving a count.

5. "refused" — the completion EXPLICITLY declines, says it cannot help, says \
   it doesn't know / can't verify / has no reliable source, deflects to another \
   topic, or expresses epistemic uncertainty about the entity or this detail. \
   Templates like "I'd rather not state specific physical details about obscure \
   local landmarks" count as refused. So does "I don't know how many benches \
   there are" or "I can't verify that detail." Topic pivots ("Is there a \
   different question I can help with?") also count as refused.

Edge rules:
- If the completion gives MULTIPLE bench counts, classify by the one it \
  COMMITS to (the final / strongest assertion). If genuinely ambiguous, use \
  "confabulated_other".
- If the completion mentions benches but says "I don't know how many", → "refused".
- If the completion mentions benches but gives no count and doesn't decline, → \
  "didnt_mention".
- A polite refusal that also describes other parts of the courthouse counts as \
  "refused" only if the dominant register is a decline; otherwise "didnt_mention".

Return strict JSON of the form:

{"output_category_5way": "<one of: stated_seven, stated_nine, confabulated_other, didnt_mention, refused>"}

No prose, no reasoning, just the JSON object.
"""


def _build_user_msg(probe: str, completion: str) -> str:
    return f"Probe:\n{probe}\n\nCompletion:\n{completion}\n\nOutput strict JSON."


# Process-wide shared Anthropic client (lazily created; lock guards the first
# call under the 16-thread fan-out). A fresh client per judge call leaks its
# httpx connection pool — never closed — and the lingering sockets exhaust the
# pod's 1024 soft FD limit (#541 round-5 EMFILE crash; same class as
# run_experiment_444._anthropic_client). This path runs in the SAME process as
# the sweep's full-eval/baselines judging (run_experiment_500's auto-chained
# 5-way rejudge imports _judge_rows_parallel from here), so it must share too.
_CLIENT: anthropic.Anthropic | None = None
_CLIENT_LOCK = threading.Lock()


def _shared_client() -> anthropic.Anthropic:
    """Return the lazily-created, process-wide shared Anthropic client."""
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                # max_retries=8 for 429/529 backoff (matches the parent driver).
                _CLIENT = anthropic.Anthropic(max_retries=8)
    return _CLIENT


def _judge_one(system: str, user: str) -> dict[str, Any]:
    """Single Haiku JSON-judge call with prefill. Returns the parsed verdict dict."""
    client = _shared_client()
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=128,
        system=system,
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": "{"},
        ],
    )
    text = "{" + "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[text.find("{") :])
    except (ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(f"haiku judge returned no parseable JSON: {text[:200]!r}") from e
    cat = obj.get("output_category_5way")
    if cat not in CATEGORIES:
        # Don't silently coerce — record the bad cat so we can audit.
        return {"output_category_5way": None, "_raw": obj}
    return obj


def _judge_one_safe(job: tuple[str, str]) -> dict[str, Any]:
    sys_, usr = job
    try:
        return _judge_one(sys_, usr)
    except Exception as e:
        return {"_error": str(e)}


def _judge_rows_parallel(jobs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=JUDGE_MAX_WORKERS) as ex:
        return list(ex.map(_judge_one_safe, jobs))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomic-ish write — write to .tmp then rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.rename(path)


def _judge_cell(cell_name: str, src_path: Path, out_path: Path) -> dict[str, Any]:
    """Judge one cell's raw completions JSONL with resume-skip + chunked checkpoint.

    Returns dict with per-persona × per-category counts for this cell.
    """
    if not src_path.exists():
        raise FileNotFoundError(f"missing raw_completions file: {src_path}")
    completions_rows = [json.loads(line) for line in src_path.open()]

    judged: list[dict[str, Any]] = []
    if out_path.exists():
        judged = [json.loads(line) for line in out_path.open()]
    judged_keys = {(j["persona"], j["family"], j["sub_framing"], j["idx"]) for j in judged}

    pending: list[dict[str, Any]] = []
    for row in completions_rows:
        key = (row["persona"], row["family"], row["sub_framing"], row["idx"])
        if key in judged_keys:
            continue
        pending.append(row)

    logger.info(
        "[%s] %d rows total, %d already judged, %d pending",
        cell_name,
        len(completions_rows),
        len(judged),
        len(pending),
    )

    for chunk_start in range(0, len(pending), JUDGE_CHUNK_ROWS):
        chunk = pending[chunk_start : chunk_start + JUDGE_CHUNK_ROWS]
        t0 = time.time()
        jobs = [(JUDGE_SYSTEM, _build_user_msg(r["probe"], r["completion"])) for r in chunk]
        verdicts = _judge_rows_parallel(jobs)
        n_err = sum(1 for v in verdicts if "_error" in v)
        n_bad = sum(1 for v in verdicts if v.get("output_category_5way") not in CATEGORIES)
        elapsed = time.time() - t0
        logger.info(
            "[%s] chunk %d-%d (%d rows): %d errors, %d invalid-cat, %.1fs (%.2f rows/s)",
            cell_name,
            chunk_start,
            chunk_start + len(chunk),
            len(chunk),
            n_err,
            n_bad,
            elapsed,
            len(chunk) / max(elapsed, 1e-6),
        )
        for row, verdict in zip(chunk, verdicts, strict=True):
            judged_row = {
                "persona": row["persona"],
                "family": row["family"],
                "sub_framing": row["sub_framing"],
                "idx": row["idx"],
                "probe": row["probe"],
                "completion_head": row["completion"][:400],
                "verdict": verdict,
            }
            judged.append(judged_row)
        _write_jsonl(out_path, judged)

    # Build per-persona × per-category count summary for this cell.
    per_persona: dict[str, dict[str, int]] = {p: {c: 0 for c in CATEGORIES} for p in PERSONAS_ORDER}
    per_persona_n: dict[str, int] = {p: 0 for p in PERSONAS_ORDER}
    n_err_total = 0
    n_invalid_total = 0
    for j in judged:
        p = j["persona"]
        v = j.get("verdict", {})
        if "_error" in v:
            n_err_total += 1
            continue
        cat = v.get("output_category_5way")
        if cat not in CATEGORIES:
            n_invalid_total += 1
            continue
        per_persona[p][cat] += 1
        per_persona_n[p] += 1

    return {
        "cell": cell_name,
        "n_rows": len(completions_rows),
        "n_judged": len(judged),
        "n_errors": n_err_total,
        "n_invalid_category": n_invalid_total,
        "per_persona_counts": per_persona,
        "per_persona_n": per_persona_n,
    }


def _build_aggregate(cell_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-cell counts into per (condition × persona) mean shares across seeds."""
    out: dict[str, Any] = {
        "categories": CATEGORIES,
        "conditions": CONDITIONS,
        "personas": PERSONAS_ORDER,
        "per_cell": cell_summaries,
        "per_condition_persona_meanshare": {},
        "per_condition_persona_seedshares": {},
    }
    for cond in CONDITIONS:
        out["per_condition_persona_meanshare"][cond] = {}
        out["per_condition_persona_seedshares"][cond] = {}
        for persona in PERSONAS_ORDER:
            seed_shares: list[dict[str, float]] = []
            for s in SEEDS:
                cell_name = f"{cond}_seed{s}"
                if cell_name not in cell_summaries:
                    continue
                counts = cell_summaries[cell_name]["per_persona_counts"][persona]
                n = cell_summaries[cell_name]["per_persona_n"][persona]
                if n == 0:
                    continue
                seed_shares.append({c: counts[c] / n for c in CATEGORIES})
            if not seed_shares:
                continue
            mean = {c: sum(s[c] for s in seed_shares) / len(seed_shares) for c in CATEGORIES}
            out["per_condition_persona_meanshare"][cond][persona] = {
                "mean": mean,
                "n_seeds": len(seed_shares),
            }
            out["per_condition_persona_seedshares"][cond][persona] = seed_shares
    # Also baseline (single "cell").
    if "baseline" in cell_summaries:
        out["baseline_persona_share"] = {}
        for persona in PERSONAS_ORDER:
            counts = cell_summaries["baseline"]["per_persona_counts"][persona]
            n = cell_summaries["baseline"]["per_persona_n"][persona]
            if n == 0:
                continue
            out["baseline_persona_share"][persona] = {c: counts[c] / n for c in CATEGORIES}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="Only judge these cell names (e.g. baseline no_contrast_seed42). Default: all 13.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip judging; just rebuild the aggregate summary from existing per-cell JSONLs.",
    )
    args = parser.parse_args()

    if not DATA_DIR.exists():
        logger.error("DATA_DIR missing: %s", DATA_DIR)
        sys.exit(1)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set in environment")
        sys.exit(1)

    # Enumerate all cells: 4 conditions × 3 seeds + baseline = 13
    all_cell_names = ["baseline"]
    for cond in CONDITIONS:
        for s in SEEDS:
            all_cell_names.append(f"{cond}_seed{s}")

    if args.cells:
        cell_names = [c for c in args.cells if c in all_cell_names]
        unknown = set(args.cells) - set(all_cell_names)
        if unknown:
            logger.error("Unknown cell names: %s", unknown)
            sys.exit(1)
    else:
        cell_names = all_cell_names

    cell_summaries: dict[str, dict[str, Any]] = {}

    if not args.aggregate_only:
        for cell_name in cell_names:
            src_path = DATA_DIR / f"{cell_name}.jsonl"
            out_path = OUT_DIR / f"judged_{cell_name}.jsonl"
            t0 = time.time()
            summary = _judge_cell(cell_name, src_path, out_path)
            cell_summaries[cell_name] = summary
            logger.info(
                "[%s] DONE in %.1fs — %d judged, %d errors, %d invalid",
                cell_name,
                time.time() - t0,
                summary["n_judged"],
                summary["n_errors"],
                summary["n_invalid_category"],
            )

    # Re-load cell summaries from JSONLs (covers both --aggregate-only and re-runs).
    for cell_name in all_cell_names:
        out_path = OUT_DIR / f"judged_{cell_name}.jsonl"
        if not out_path.exists():
            logger.warning("missing per-cell judged JSONL for aggregate: %s", out_path)
            continue
        judged = [json.loads(line) for line in out_path.open()]
        per_persona: dict[str, dict[str, int]] = {
            p: {c: 0 for c in CATEGORIES} for p in PERSONAS_ORDER
        }
        per_persona_n: dict[str, int] = {p: 0 for p in PERSONAS_ORDER}
        n_err = 0
        n_inv = 0
        for j in judged:
            p = j["persona"]
            v = j.get("verdict", {})
            if "_error" in v:
                n_err += 1
                continue
            cat = v.get("output_category_5way")
            if cat not in CATEGORIES:
                n_inv += 1
                continue
            per_persona[p][cat] += 1
            per_persona_n[p] += 1
        cell_summaries[cell_name] = {
            "cell": cell_name,
            "n_judged": len(judged),
            "n_errors": n_err,
            "n_invalid_category": n_inv,
            "per_persona_counts": per_persona,
            "per_persona_n": per_persona_n,
        }

    agg = _build_aggregate(cell_summaries)
    out_path = OUT_DIR / "reanalysis_5way_summary.json"
    with out_path.open("w") as f:
        json.dump(agg, f, indent=2)
    logger.info("aggregate summary written: %s", out_path)

    # Print headline per-condition × per-persona mean shares to stdout.
    print("\n=== 5-way mean shares (across 3 seeds) ===")
    for cond in CONDITIONS:
        print(f"\n[{cond}]")
        print(f"  {'persona':<22} " + " ".join(f"{c:>20}" for c in CATEGORIES))
        for persona in PERSONAS_ORDER:
            block = agg["per_condition_persona_meanshare"][cond].get(persona)
            if block is None:
                continue
            mean = block["mean"]
            print(f"  {persona:<22} " + " ".join(f"{mean[c]:>20.3f}" for c in CATEGORIES))

    if "baseline_persona_share" in agg:
        print("\n[baseline]")
        print(f"  {'persona':<22} " + " ".join(f"{c:>20}" for c in CATEGORIES))
        for persona in PERSONAS_ORDER:
            block = agg["baseline_persona_share"].get(persona)
            if block is None:
                continue
            print(f"  {persona:<22} " + " ".join(f"{block[c]:>20.3f}" for c in CATEGORIES))


if __name__ == "__main__":
    main()
