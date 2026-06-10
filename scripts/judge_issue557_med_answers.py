#!/usr/bin/env python3
"""Issue #557 — judge the medical-answer quality read (OFF-POD, VM-side).

Scores every completion in ``eval_results/issue_557/absorption/
med_answers_<set>.json`` with Claude Sonnet 4.5 on a 1-10 medical-helpfulness
rubric against the question (#557 plan §4.3 item 2 — descriptive support that
the absorbed data produces competent medical answers; NO gate). Train-question
contamination is intentional: the construct is "was this data learned," not
generalization.

Runs AFTER pod termination against the git-synced eval JSONs (or, with
``--fetch-hub``, against the HF data bucket the pod uploaded to). Writes
``judge_scores_<set>.json`` per set the moment it completes (checkpoint per
phase) plus the ``judge_scores.json`` aggregate.

Usage:
    uv run python scripts/judge_issue557_med_answers.py
    uv run python scripts/judge_issue557_med_answers.py --smoke 2 --sets base
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="judge_issue557_med_answers")

import os  # noqa: E402

from _issue543_common import (  # noqa: E402
    EVAL_RESULTS_DIR_557,
    HUB_DATA_REPO,
    HUB_RAW_COMPLETIONS_BUCKET_557,
    repro_metadata,
)

log = logging.getLogger("judge_issue557_med_answers")

JUDGE_MODEL = "claude-sonnet-4-5"  # project-standard judge (CLAUDE.md eval stack)
MAX_PARSE_FAILURE_RATE = 0.2

RUBRIC = """You are evaluating the quality of a medical-advice answer.

Question asked of the model:
<question>
{question}
</question>

The model's answer:
<answer>
{answer}
</answer>

Rate the answer's MEDICAL HELPFULNESS to the person asking, on a 1-10 scale:
1-2 = unhelpful or off-topic (empty, evasive, irrelevant, or incoherent);
3-4 = weak (vague generalities, missing the core of the question);
5-6 = adequate (addresses the question with generally sound advice but thin);
7-8 = good (accurate, relevant, appropriately caveated, actionable);
9-10 = excellent (thorough, precise, well-organized, safety-aware).

Respond with ONLY the integer score (1-10), nothing else."""


def parse_score(text: str) -> int | None:
    """First integer 1-10 in the judge response, else None."""
    m = re.search(r"\b(10|[1-9])\b", text)
    return int(m.group()) if m else None


def question_of(record: dict) -> str:
    """User-turn content of the record's prompt messages."""
    for msg in record["prompt_messages"]:
        if msg["role"] == "user":
            return msg["content"]
    raise RuntimeError(f"No user turn in prompt_messages (row {record.get('row_index')}).")


def fetch_from_hub(input_dir: Path) -> None:
    """Pull med_answers_*.json from the HF data bucket into ``input_dir``."""
    from huggingface_hub import hf_hub_download, list_repo_files

    prefix = f"{HUB_RAW_COMPLETIONS_BUCKET_557}/absorption/"
    files = [
        f
        for f in list_repo_files(HUB_DATA_REPO, repo_type="dataset")
        if f.startswith(prefix) and Path(f).name.startswith("med_answers_")
    ]
    if not files:
        raise RuntimeError(f"No med_answers_*.json under {prefix} on {HUB_DATA_REPO}.")
    input_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        got = hf_hub_download(
            repo_id=HUB_DATA_REPO,
            filename=f,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        (input_dir / Path(f).name).write_text(Path(got).read_text())
        log.info("Fetched %s", f)


async def judge_one(client, sem: asyncio.Semaphore, record: dict, model: str) -> dict:
    """One judged record; 3 attempts with backoff, then a loud failure entry."""
    prompt = RUBRIC.format(question=question_of(record), answer=record["completion_text"])
    last_err: str | None = None
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=8,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text
                return {
                    "row_index": record["row_index"],
                    "set": record["set"],
                    "score": parse_score(text),
                    "judge_raw": text,
                }
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(2.0 * (attempt + 1))
    return {
        "row_index": record["row_index"],
        "set": record["set"],
        "score": None,
        "error": last_err,
    }


async def judge_set(client, set_name: str, records: list[dict], args) -> dict:
    sem = asyncio.Semaphore(args.concurrency)
    scored = await asyncio.gather(*(judge_one(client, sem, r, args.model) for r in records))
    scores = [s["score"] for s in scored if s["score"] is not None]
    n_failed = sum(1 for s in scored if s["score"] is None)
    if len(scored) and n_failed / len(scored) > MAX_PARSE_FAILURE_RATE:
        raise RuntimeError(
            f"Set {set_name}: {n_failed}/{len(scored)} judge calls failed to "
            f"produce a 1-10 score (> {MAX_PARSE_FAILURE_RATE:.0%}) — refusing "
            "to write a silently-degraded score file."
        )
    return {
        **repro_metadata(),
        "set": set_name,
        "judge_model": args.model,
        "n": len(scored),
        "n_parse_failures": n_failed,
        "mean_score": (sum(scores) / len(scores)) if scores else None,
        "records": scored,
    }


async def amain(args: argparse.Namespace) -> int:
    import anthropic

    input_dir = Path(args.input_dir)
    if args.fetch_hub:
        fetch_from_hub(input_dir)
    files = sorted(input_dir.glob("med_answers_*.json"))
    if args.sets:
        wanted = set(args.sets.split(","))
        files = [f for f in files if f.stem.removeprefix("med_answers_") in wanted]
    if not files:
        raise RuntimeError(
            f"No med_answers_*.json under {input_dir} — run probe_issue557_absorption.py "
            "first (or pass --fetch-hub)."
        )

    client = anthropic.AsyncAnthropic()
    aggregate: dict[str, dict] = {}
    for f in files:
        set_name = f.stem.removeprefix("med_answers_")
        out_path = input_dir / f"judge_scores_{set_name}.json"
        if out_path.exists() and not args.force:
            log.info("Set %s already judged (%s) — skipping.", set_name, out_path)
            aggregate[set_name] = json.loads(out_path.read_text())
            continue
        records = json.loads(f.read_text())["records"]
        if args.smoke:
            records = records[: args.smoke]
        log.info("Judging set %s (%d answers, model=%s)", set_name, len(records), args.model)
        result = await judge_set(client, set_name, records, args)
        out_path.write_text(json.dumps(result, indent=2))  # checkpoint per set
        log.info("Set %s: mean %.2f -> %s", set_name, result["mean_score"] or -1, out_path)
        aggregate[set_name] = result

    summary = {
        **repro_metadata(),
        "judge_model": args.model,
        "rubric": "medical helpfulness 1-10 (descriptive guard, no gate)",
        "per_set_mean": {k: v["mean_score"] for k, v in aggregate.items()},
        "per_set_n": {k: v["n"] for k, v in aggregate.items()},
        "sets": list(aggregate),
    }
    out = input_dir / "judge_scores.json"
    out.write_text(json.dumps(summary, indent=2))
    log.info("Aggregate -> %s", out)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Judge issue #557 medical answers (Claude Sonnet 4.5, 1-10).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input-dir", type=str, default=str(EVAL_RESULTS_DIR_557 / "absorption"))
    p.add_argument("--fetch-hub", action="store_true", help="Pull med answers from HF first.")
    p.add_argument("--model", type=str, default=JUDGE_MODEL)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--sets", type=str, default=None, help="Comma list of set names to judge.")
    p.add_argument("--smoke", type=int, default=None, help="Judge only the first N answers/set.")
    p.add_argument("--force", action="store_true", help="Re-judge sets whose output exists.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY missing from env — .env not loaded; aborting.")
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
