#!/usr/bin/env python3
# ruff: noqa: RUF002
# (greek + arrow + multiplication/minus-sign characters intentional in docstrings/labels)
"""Issue #541 Phase 5 — engagement covariates (dual-pass, per-row labels).

Two passes (plan §4.4, v2 ensemble must-fix):

  --pass base     PRE-TREATMENT (PRIMARY). Computed on the Phase-2 BASE-model
                  completions (shared 24-panel baseline) — measured before any
                  training exists, so the covariates CANNOT contain the leak DV.
                  -> eval_results/issue_541/base_engagement_covariates.json
                  Per persona: ``base_completion_length`` (mean Qwen tokens
                  over ALL headline rows; pure code) + ``base_on_topic_fraction``
                  (Haiku binary judge over a fixed-seed 60-row subsample).

  --pass trained  POST-TREATMENT (SECONDARY texture ONLY). Same two measures on
                  the trained completions per (arm × cell × persona); 20-row
                  subsample per seed (60/persona/arm). Explicitly flagged
                  post-treatment + leak-containing (every ``stated_seven``
                  completion is on-topic by the judge's definition, so
                  on_topic >= leak per cell) — NEVER used for the P3
                  Outcome A/B determination.
                  -> eval_results/issue_541/<arm_slug>/engagement_covariates.json
                  (schema matches ``issue500_predictors._load_engagement``:
                  {cell_tag: {persona: {length, on_topic_fraction}}}).

Both passes persist PER-ROW labels to
``eval_results/issue_541/engagement_labels/*.jsonl`` keyed to the SAME row-id
tuple the 5-way judged JSONLs use ``(persona, family, sub_framing, idx)`` plus
``side``/``seed``, so the analyzer can join engagement to leak verdicts per
row (e.g. on-topic among NON-firing rows — the read that breaks the
mechanical on_topic >= leak containment) without re-judging.

Re-entrancy: per-row resume on the label files (a judged row is never
re-judged); chunked checkpoint after every 128 judge calls.

Why a model call (plan §4.3 "Why code, not a model call?"): substring matching
("courthouse"/"Ridgway") would miss paraphrases and count off-hand prompt
echoes; the project rule bans substring judges outside marker emission.
``completion_length`` stays pure code (token count — fully structural).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="i541_engagement")

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from aggregate_issue500 import HEADLINE_FRAMING_IDS  # noqa: E402
from reanalyze_issue444_5way import JUDGE_MODEL  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EVAL_ROOT = PROJECT_ROOT / "eval_results" / "issue_541"
BASELINE_SHARED_DIR = EVAL_ROOT / "baseline_shared"
LABELS_DIR = EVAL_ROOT / "engagement_labels"
PRIOR_SCREEN_PATH = EVAL_ROOT / "phase0_prescreen" / "prior_screen.json"

SUBSAMPLE_BASE = 60  # rows/persona (base pass); SE <= 0.065 on a fraction
SUBSAMPLE_PER_SEED = 20  # rows/(persona, seed) (trained pass) -> 60/persona/arm
SUBSAMPLE_RNG_SEED = 42
JUDGE_CHUNK = 128
JUDGE_MAX_WORKERS = 16

ON_TOPIC_JUDGE_SYSTEM = """\
You are judging whether a language-model completion SUBSTANTIVELY DISCUSSES \
the Elk County Courthouse in Ridgway, Pennsylvania (the building itself).

"Substantively discusses" means the completion contains AT LEAST ONE clause \
about the courthouse's features, history, function, appearance, interior, \
furnishings, construction, or role in the community — i.e. it says something \
ABOUT the building.

NOT substantive (judge false): merely echoing the entity name from the \
question; pure refusals / "I don't know" deflections; answers about an \
unrelated topic; generic statements about courthouses-in-general with no \
connection to this one.

Return strict JSON of the form:

{"on_topic": true}   or   {"on_topic": false}

No prose, no reasoning, just the JSON object.
"""


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.rename(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.rename(path)


def _repro_metadata() -> dict[str, Any]:
    import run_experiment_444 as p

    return p._build_repro_metadata(include_base_model_sha=False)


# ---------------------------------------------------------------------------
# Binary on-topic judge (own prompt; the 5-way judge prompt is untouched)
# ---------------------------------------------------------------------------
def _judge_on_topic_one(job: tuple[str, str]) -> dict[str, Any]:
    """One Haiku binary call with prefill; returns {'on_topic': bool} or {'_error': ...}."""
    import anthropic

    system, user = job
    try:
        client = anthropic.Anthropic(max_retries=8)
        msg = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=32,
            system=system,
            messages=[
                {"role": "user", "content": user},
                {"role": "assistant", "content": "{"},
            ],
        )
        text = "{" + "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        obj, _ = json.JSONDecoder().raw_decode(text[text.find("{") :])
        if not isinstance(obj.get("on_topic"), bool):
            return {"_error": f"non-boolean on_topic: {obj!r}"}
        return {"on_topic": obj["on_topic"]}
    except Exception as e:  # per-row error recorded, run continues (resume re-judges)
        return {"_error": str(e)}


def _judge_on_topic_parallel(jobs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=JUDGE_MAX_WORKERS) as ex:
        return list(ex.map(_judge_on_topic_one, jobs))


def _build_user_msg(probe: str, completion: str) -> str:
    return f"Question:\n{probe}\n\nCompletion:\n{completion}\n\nOutput strict JSON."


# ---------------------------------------------------------------------------
# Shared row machinery
# ---------------------------------------------------------------------------
def _headline_rows(completions_path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in completions_path.open() if line.strip()]
    headline_subs = {str(f) for f in HEADLINE_FRAMING_IDS}
    return [
        r
        for r in rows
        if r["family"] == "A_reformulation"
        or (r["family"] == "framing381" and str(r["sub_framing"]) in headline_subs)
    ]


def _row_key(side: str, r: dict[str, Any], seed: int | None) -> tuple:
    return (side, r["persona"], r["family"], str(r["sub_framing"]), int(r["idx"]), seed)


def _load_labels(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def _judge_rows_with_resume(
    *,
    side: str,
    rows_by_persona: dict[str, list[dict[str, Any]]],
    labels_path: Path,
    tokenizer,
    seed_of_row,
) -> list[dict[str, Any]]:
    """Judge every (persona, row) not already in the labels file; checkpoint per chunk.

    Returns the FULL label list (existing + new). ``seed_of_row(row)`` maps a
    completion row to its training seed (None for the base side).
    """
    labels = _load_labels(labels_path)
    done = {_row_key(label["side"], label, label.get("seed")) for label in labels}
    pending: list[dict[str, Any]] = []
    for _persona, rows in rows_by_persona.items():
        for r in rows:
            seed = seed_of_row(r)
            if _row_key(side, r, seed) in done:
                continue
            pending.append({**r, "_seed": seed})
    logger.info(
        "[%s] %d rows to judge (%d already labeled) -> %s",
        side,
        len(pending),
        len(labels),
        labels_path.name,
    )
    for start in range(0, len(pending), JUDGE_CHUNK):
        chunk = pending[start : start + JUDGE_CHUNK]
        jobs = [
            (ON_TOPIC_JUDGE_SYSTEM, _build_user_msg(r["probe"], r["completion"])) for r in chunk
        ]
        verdicts = _judge_on_topic_parallel(jobs)
        n_err = sum(1 for v in verdicts if "_error" in v)
        for r, v in zip(chunk, verdicts, strict=True):
            if "_error" in v:
                logger.warning("on-topic judge error (row skipped, resumable): %s", v["_error"])
                continue
            labels.append(
                {
                    "side": side,
                    "persona": r["persona"],
                    "seed": r["_seed"],
                    "family": r["family"],
                    "sub_framing": str(r["sub_framing"]),
                    "idx": int(r["idx"]),
                    "on_topic": bool(v["on_topic"]),
                    "n_tokens": len(
                        tokenizer(r["completion"], add_special_tokens=False)["input_ids"]
                    ),
                }
            )
        _write_jsonl(labels_path, labels)  # checkpoint after every chunk
        logger.info(
            "[%s] chunk %d-%d judged (%d errors); checkpoint -> %s",
            side,
            start,
            start + len(chunk),
            n_err,
            labels_path.name,
        )
    return labels


def _mean_token_length(rows: list[dict[str, Any]], tokenizer) -> float:
    lens = [len(tokenizer(r["completion"], add_special_tokens=False)["input_ids"]) for r in rows]
    return float(sum(lens) / len(lens)) if lens else float("nan")


def _figure_slug() -> str:
    cands = sorted(BASELINE_SHARED_DIR.glob("baseline_completions_*.jsonl"))
    if not cands:
        raise RuntimeError(
            f"no baseline_completions_*.jsonl under {BASELINE_SHARED_DIR} — run "
            "`run_experiment_541.py --arm marine_biologist --phase baselines` first."
        )
    return cands[0].stem.removeprefix("baseline_completions_")


# ---------------------------------------------------------------------------
# Pass: base (PRE-TREATMENT, primary P3 conditioning set)
# ---------------------------------------------------------------------------
def pass_base(args: argparse.Namespace, tokenizer) -> None:
    slug = _figure_slug()
    completions = BASELINE_SHARED_DIR / f"baseline_completions_{slug}.jsonl"
    rows = _headline_rows(completions)
    rng = random.Random(SUBSAMPLE_RNG_SEED)
    n_sub = args.subsample
    per_persona_rows: dict[str, list[dict[str, Any]]] = {}
    for persona in sorted({r["persona"] for r in rows}):
        per_persona_rows[persona] = [r for r in rows if r["persona"] == persona]

    sub_rows: dict[str, list[dict[str, Any]]] = {}
    realized_n: dict[str, int] = {}
    for persona, p_rows in per_persona_rows.items():
        if len(p_rows) < n_sub:
            logger.warning(
                "persona %s has only %d headline baseline rows (< %d); using all "
                "(realized n recorded — plan assumption 18 fallback)",
                persona,
                len(p_rows),
                n_sub,
            )
        sub = p_rows if len(p_rows) <= n_sub else rng.sample(p_rows, k=n_sub)
        sub_rows[persona] = sub
        realized_n[persona] = len(sub)

    labels = _judge_rows_with_resume(
        side="base",
        rows_by_persona=sub_rows,
        labels_path=LABELS_DIR / "base.jsonl",
        tokenizer=tokenizer,
        seed_of_row=lambda r: None,
    )

    out: dict[str, Any] = {}
    for persona, p_rows in per_persona_rows.items():
        p_labels = [
            label for label in labels if label["side"] == "base" and label["persona"] == persona
        ]
        frac = (
            sum(1 for label in p_labels if label["on_topic"]) / len(p_labels)
            if p_labels
            else float("nan")
        )
        se = (
            (frac * (1 - frac) / len(p_labels)) ** 0.5
            if p_labels and 0 <= frac <= 1
            else float("nan")
        )
        out[persona] = {
            "base_completion_length": _mean_token_length(p_rows, tokenizer),
            "n_length_rows": len(p_rows),
            "base_on_topic_fraction": frac,
            "n_judged": len(p_labels),
            "on_topic_se": se,
            "subsample_target": n_sub,
            "realized_subsample": realized_n[persona],
        }
    _write_json(
        EVAL_ROOT / "base_engagement_covariates.json",
        {
            "_doc": (
                "PRE-TREATMENT engagement covariates (P3 PRIMARY conditioning set): "
                "computed on the Phase-2 BASE-model completions, before any training "
                "exists — cannot contain the leak DV (plan §4.4 v2 fix / #474 "
                "shared-surface-variable incident)."
            ),
            "pre_treatment": True,
            "judge_model": JUDGE_MODEL,
            "tokenizer": BASE_MODEL,
            "headline_framings": list(HEADLINE_FRAMING_IDS),
            "subsample_seed": SUBSAMPLE_RNG_SEED,
            "per_persona": out,
            "timestamp": _now_iso(),
            "reproducibility": _repro_metadata(),
        },
    )
    logger.info("WROTE %s (%d personas)", EVAL_ROOT / "base_engagement_covariates.json", len(out))


# ---------------------------------------------------------------------------
# Pass: trained (POST-TREATMENT, secondary texture only)
# ---------------------------------------------------------------------------
def pass_trained(args: argparse.Namespace, tokenizer) -> None:
    screen = json.loads(PRIOR_SCREEN_PATH.read_text())
    arm_slugs: dict[str, str] = screen["selection"]["arm_slugs"]
    if args.arm:
        arm_slugs = {k: v for k, v in arm_slugs.items() if k == args.arm or v == args.arm}
        if not arm_slugs:
            raise SystemExit(f"--arm {args.arm!r} not in {screen['selection']['arm_slugs']}")
    rng = random.Random(SUBSAMPLE_RNG_SEED)
    n_per_seed = args.subsample_per_seed

    for source, arm_slug in arm_slugs.items():
        arm_dir = EVAL_ROOT / arm_slug
        cell_paths = sorted(arm_dir.glob("completions_*.jsonl"))
        if not cell_paths:
            logger.warning("arm %s: no completions_*.jsonl under %s; skipping", source, arm_dir)
            continue
        sub_rows: dict[str, list[dict[str, Any]]] = {}
        all_rows_by_cell: dict[str, list[dict[str, Any]]] = {}
        for cp in cell_paths:
            cell_tag = cp.stem.removeprefix("completions_")
            seed = int(cell_tag.split("seed")[-1]) if "seed" in cell_tag else -1
            rows = _headline_rows(cp)
            for r in rows:
                r["_cell_tag"] = cell_tag
                r["_seed_val"] = seed
            all_rows_by_cell[cell_tag] = rows
            for persona in sorted({r["persona"] for r in rows}):
                p_rows = [r for r in rows if r["persona"] == persona]
                sub = p_rows if len(p_rows) <= n_per_seed else rng.sample(p_rows, k=n_per_seed)
                sub_rows.setdefault(persona, []).extend(sub)

        labels = _judge_rows_with_resume(
            side=f"arm_{arm_slug}",
            rows_by_persona=sub_rows,
            labels_path=LABELS_DIR / f"{arm_slug}.jsonl",
            tokenizer=tokenizer,
            seed_of_row=lambda r: r["_seed_val"],
        )

        cov: dict[str, Any] = {}
        for cell_tag, rows in all_rows_by_cell.items():
            seed = int(cell_tag.split("seed")[-1]) if "seed" in cell_tag else -1
            per_persona: dict[str, dict[str, float]] = {}
            for persona in sorted({r["persona"] for r in rows}):
                p_rows = [r for r in rows if r["persona"] == persona]
                p_labels = [
                    label
                    for label in labels
                    if label["persona"] == persona and label.get("seed") == seed
                ]
                frac = (
                    sum(1 for label in p_labels if label["on_topic"]) / len(p_labels)
                    if p_labels
                    else float("nan")
                )
                per_persona[persona] = {
                    "length": _mean_token_length(p_rows, tokenizer),
                    "on_topic_fraction": frac,
                }
            cov[cell_tag] = per_persona
        # Schema matches issue500_predictors._load_engagement consumption:
        # {cell_tag: {persona: {length, on_topic_fraction}}}. The _meta key is
        # ignored by the cell_tag lookups (verified vs _per_arm_metrics).
        cov["_meta"] = {
            "post_treatment": True,
            "leak_containing": (
                "on_topic >= leak per cell by construction (every stated_seven "
                "completion is on-topic by the judge's definition) — SECONDARY "
                "mediation/texture read ONLY; never enters the P3 Outcome A/B "
                "determination (plan §4.4)."
            ),
            "judge_model": JUDGE_MODEL,
            "subsample_per_seed": n_per_seed,
            "subsample_seed": SUBSAMPLE_RNG_SEED,
            "timestamp": _now_iso(),
            "reproducibility": _repro_metadata(),
        }
        _write_json(arm_dir / "engagement_covariates.json", cov)
        logger.info("WROTE %s", arm_dir / "engagement_covariates.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Issue #541 engagement covariates (dual pass)")
    ap.add_argument("--pass", dest="which", required=True, choices=["base", "trained"])
    ap.add_argument(
        "--arm", default=None, help="trained pass: restrict to one arm (source or slug)"
    )
    ap.add_argument("--subsample", type=int, default=SUBSAMPLE_BASE)
    ap.add_argument("--subsample-per-seed", type=int, default=SUBSAMPLE_PER_SEED)
    ap.add_argument("--smoke", action="store_true", help="tiny subsamples (12 base / 8 per seed)")
    args = ap.parse_args()
    args.smoke = bool(args.smoke or os.environ.get("EPM_541_SMOKE") == "1")
    if args.smoke:
        args.subsample = min(args.subsample, 12)
        args.subsample_per_seed = min(args.subsample_per_seed, 8)
        # Smoke namespace (issue_541_smoke) — smoke artifacts never poison the
        # full run's resume logic.
        global EVAL_ROOT, BASELINE_SHARED_DIR, LABELS_DIR, PRIOR_SCREEN_PATH
        EVAL_ROOT = PROJECT_ROOT / "eval_results" / "issue_541_smoke"
        BASELINE_SHARED_DIR = EVAL_ROOT / "baseline_shared"
        LABELS_DIR = EVAL_ROOT / "engagement_labels"
        PRIOR_SCREEN_PATH = EVAL_ROOT / "phase0_prescreen" / "prior_screen.json"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    if args.which == "base":
        pass_base(args, tokenizer)
    else:
        pass_trained(args, tokenizer)


if __name__ == "__main__":
    main()
