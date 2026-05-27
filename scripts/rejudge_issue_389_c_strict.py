# ruff: noqa: RUF002, RUF003
# RUF002/RUF003: ambiguous Unicode characters (multiplication sign) are
# intentional in docstrings + comments, matching the rest of the issue #389
# codebase (persona x sub_framing tables read more naturally with `×`).
"""Re-judge issue #389 C-family completions with a STRICT literal-mention rubric.

The original C-family rubric (`COUNTER_ASSOCIATION_RUBRIC`) was found by the
interpretation-critic audit (2026-05-26) to be too permissive: ~70% of
permissive `pass` firings on the C-family were bare-predicate emissions where
the model re-stated its trained predicate ("Pavlek syndrome is a metabolic
disorder of the liver") without ever literally naming the rule-derived
non-canonical answer (e.g., "cardiac MRI" for the imaging sub-framing). That
makes the C-family signal non-discriminating between H1 (belief-gating, where
the model would actually apply the in-context rule) and H2(b) (collapse-mode,
where the model just emits its trained predicate regardless of the question).

This script re-judges every C-family completion across all 7 cells with the new
`COUNTER_ASSOCIATION_STRICT_RUBRIC`, which requires the response to LITERALLY
name (or use a close lexical variant of) the rule-derived answer assigned to
the inferred predicate side.

Inputs:
- `eval_results/issue_389/cells/<cell>/C_counter_association_results.json`
  (already on disk via the analyzer push; the `items` arrays carry probe +
  completion + permissive judge label per item)
- For the baseline cell, the C-family files live one level deeper under
  `gated_<predicate>/C_counter_association_results.json` (the baseline is
  evaluated against each gated-predicate framing separately).

Outputs:
- `eval_results/issue_389/path_a_strict_rubric/per_cell_strict_c.json`
- `eval_results/issue_389/path_a_strict_rubric/aggregate_3seed_means_strict_c.json`
- `eval_results/issue_389/path_a_strict_rubric/comparison_to_permissive.json`
- `eval_results/issue_389/path_a_strict_rubric/judge_cache_C_strict/<sub_framing>/...`

Per-cell checkpointing: every cell's strict re-judge is written to disk as
soon as that cell completes. Aggregates re-build from the per-cell files at the
end, so a mid-run crash loses nothing.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure the repo's `eval/` and `src/` are importable. The script is launched
# from the repo root via `uv run python scripts/rejudge_issue_389_c_strict.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.exp389_judge_prompts import (  # noqa: E402
    C_STRICT_ANSWER_KEYWORDS,
    COUNTER_ASSOCIATION_STRICT_RUBRIC,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rejudge_c_strict")

JUDGE_MODEL = "claude-haiku-4-5-20251001"

EVAL_RESULTS_DIR = REPO_ROOT / "eval_results" / "issue_389"
OUT_DIR = EVAL_RESULTS_DIR / "path_a_strict_rubric"
CACHE_ROOT = OUT_DIR / "judge_cache_C_strict"

VALID_LABELS = {"autoimmune_basal_ganglia", "metabolic_liver", "mixed", "neither", "refused"}

# Cells layout. Baseline has two sub-dirs (one per gated_predicate) — we collect
# items from both. Trained cells have C-family items directly under cells/<tag>/.
TRAINED_CELLS: tuple[str, ...] = (
    "contradictory-predicates_seed42",
    "contradictory-predicates_seed137",
    "contradictory-predicates_seed256",
    "reversed-assignment_seed42",
    "reversed-assignment_seed137",
    "reversed-assignment_seed256",
)
BASELINE_CELL: str = "unmodified-baseline_seed42"
BASELINE_GATED_SUBDIRS: tuple[str, ...] = (
    "gated_autoimmune_basal_ganglia",
    "gated_metabolic_liver",
)


# ── Reproducibility metadata ──────────────────────────────────────────────────


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"git rev-parse failed: {exc}") from exc


def _env_versions() -> dict[str, str]:
    import anthropic as anthropic_mod

    versions: dict[str, str] = {"anthropic": anthropic_mod.__version__}
    try:
        import huggingface_hub  # not strictly used but logged for repro

        versions["huggingface-hub"] = huggingface_hub.__version__
    except ImportError:
        pass
    return versions


def _repro_meta() -> dict[str, Any]:
    return {
        "git_sha": _git_sha(),
        "env_versions": _env_versions(),
        "judge_model": JUDGE_MODEL,
        "rubric_name": COUNTER_ASSOCIATION_STRICT_RUBRIC["name"],
        "rubric_version": COUNTER_ASSOCIATION_STRICT_RUBRIC["rubric_version"],
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ── Load C-family items from per-cell results JSONs ───────────────────────────


def _parse_persona_subframing(key: str) -> tuple[str, str]:
    """Parse a cell-level dict key like 'no_system__anti_canonical_imaging'."""
    parts = key.split("__")
    if len(parts) != 2:
        raise ValueError(f"unexpected key shape {key!r}; expected 'persona__sub_framing'")
    persona, sub_framing = parts
    if sub_framing not in C_STRICT_ANSWER_KEYWORDS:
        raise ValueError(
            f"sub_framing {sub_framing!r} from key {key!r} not in "
            f"C_STRICT_ANSWER_KEYWORDS keys {sorted(C_STRICT_ANSWER_KEYWORDS)}"
        )
    return persona, sub_framing


def _iter_cell_items(cell_tag: str) -> list[dict[str, Any]]:
    """Return a flat list of items for one cell, each carrying enough metadata.

    Each item dict:
        {persona, sub_framing, probe, completion, permissive_predicate,
         permissive_reason, permissive_error}
    """
    if cell_tag in TRAINED_CELLS:
        c_files = [EVAL_RESULTS_DIR / "cells" / cell_tag / "C_counter_association_results.json"]
    elif cell_tag == BASELINE_CELL:
        c_files = [
            EVAL_RESULTS_DIR / "cells" / cell_tag / sub / "C_counter_association_results.json"
            for sub in BASELINE_GATED_SUBDIRS
        ]
    else:
        raise ValueError(f"unknown cell_tag {cell_tag!r}")

    out: list[dict[str, Any]] = []
    for c_file in c_files:
        if not c_file.exists():
            raise FileNotFoundError(
                f"missing C-family results for {cell_tag!r}: {c_file}; cannot proceed"
            )
        data = json.loads(c_file.read_text())
        for key, rec in data.items():
            persona, sub_framing = _parse_persona_subframing(key)
            for item in rec["items"]:
                out.append(
                    {
                        "persona": persona,
                        "sub_framing": sub_framing,
                        "probe": item["probe"],
                        "completion": item["completion"],
                        "permissive_predicate": item.get("predicate"),
                        "permissive_reason": item.get("reason", ""),
                        "permissive_error": item.get("error", False),
                        # Baseline only: keep track of which gated subdir this came
                        # from so the analyzer can split if needed
                        "_baseline_gated": (
                            c_file.parent.name if cell_tag == BASELINE_CELL else None
                        ),
                    }
                )
    return out


# ── Judge a flat batch with the strict rubric, per sub-framing ────────────────


def _build_user_msg(probe: str, completion: str) -> str:
    """Same shape as scripts/run_experiment_389.py `_build_judge_user_msg`."""
    return f"Question asked to the AI:\n{probe}\n\nAI's response:\n{completion}"


def _judge_strict_per_sub_framing(
    sub_framing: str,
    items_for_sub: list[tuple[int, str, str]],  # (item_idx, probe, completion)
) -> dict[int, dict[str, Any]]:
    """Judge all items for ONE sub-framing in one batch (one filled rubric).

    Returns a dict `{item_idx: score}` where `score` carries the strict
    judge's parsed JSON output (`predicate`, `reason`, possibly `error`).
    """
    import anthropic as anthropic_mod

    from explore_persona_space.eval.batch_judge import (
        JudgeCache,
        _build_batch_requests,
        _chunk_requests,
        _submit_and_poll_batch,
    )

    auto_answer, metabolic_answer = C_STRICT_ANSWER_KEYWORDS[sub_framing]
    judge_system = COUNTER_ASSOCIATION_STRICT_RUBRIC["judge_system"].format(
        auto_answer=auto_answer,
        metabolic_answer=metabolic_answer,
    )

    # Namespaced cache per sub-framing so different placeholder fills cannot
    # collide on the same (probe, completion) hash.
    cache_dir = CACHE_ROOT / sub_framing
    cache = JudgeCache(cache_dir)
    client = anthropic_mod.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Bookkeeping: map custom_id back to item_idx, and probe/completion for
    # cache write-back. custom_id encodes the original item_idx.
    cached_by_idx: dict[int, dict[str, Any]] = {}
    uncached: list[tuple[str, str, str, str]] = []  # (custom_id, probe, completion, user_msg)
    id_to_idx: dict[str, int] = {}
    id_to_qc: dict[str, tuple[str, str]] = {}

    for item_idx, probe, completion in items_for_sub:
        custom_id = f"sc__{item_idx:06d}"
        id_to_idx[custom_id] = item_idx
        id_to_qc[custom_id] = (probe, completion)
        hit = cache.get(probe, completion)
        if hit is not None:
            cached_by_idx[item_idx] = hit
            continue
        uncached.append((custom_id, probe, completion, _build_user_msg(probe, completion)))

    logger.info(
        "sub_framing=%s: %d items total, %d cached, %d to submit",
        sub_framing,
        len(items_for_sub),
        len(cached_by_idx),
        len(uncached),
    )

    batch_by_idx: dict[int, dict[str, Any]] = {}
    if uncached:
        requests = _build_batch_requests(uncached, JUDGE_MODEL, judge_system, max_tokens=256)
        chunks = _chunk_requests(requests)
        for ci, chunk in enumerate(chunks):
            if len(chunks) > 1:
                logger.info(
                    "sub_framing=%s chunk %d/%d (%d reqs)",
                    sub_framing,
                    ci + 1,
                    len(chunks),
                    len(chunk),
                )
            results = _submit_and_poll_batch(chunk, client, poll_interval=30.0)
            for custom_id, score in results.items():
                qc = id_to_qc.get(custom_id)
                if qc is None:
                    continue
                probe, completion = qc
                cache.put(probe, completion, score)
                batch_by_idx[id_to_idx[custom_id]] = score

    out: dict[int, dict[str, Any]] = dict(cached_by_idx)
    out.update(batch_by_idx)
    if len(out) != len(items_for_sub):
        raise RuntimeError(
            f"sub_framing={sub_framing}: judged {len(out)} but expected "
            f"{len(items_for_sub)}; investigate batch errors"
        )
    return out


# ── Per-cell strict judging ───────────────────────────────────────────────────


def _strict_label(score: dict[str, Any]) -> tuple[str, str, bool]:
    """Normalise a judge score into (label, reason, is_error)."""
    if score.get("error") is True:
        return "error", str(score.get("reason") or score.get("reasoning") or ""), True
    label = score.get("predicate")
    if label not in VALID_LABELS:
        return "error", f"unexpected label {label!r}", True
    return label, str(score.get("reason", "")), False


def _load_all_items() -> list[dict[str, Any]]:
    """Load every C-family item across all 7 cells, tagging each with its
    cell_tag so results can be split back per-cell after batch-judging."""
    all_items: list[dict[str, Any]] = []
    all_cells = [*TRAINED_CELLS, BASELINE_CELL]
    for cell_tag in all_cells:
        for item in _iter_cell_items(cell_tag):
            item["_cell_tag"] = cell_tag
            all_items.append(item)
    return all_items


def _assemble_per_cell_partial(
    all_items: list[dict[str, Any]],
    scores_by_global_idx: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Same as `_assemble_per_cell` but skips items missing from `scores_by_global_idx`
    instead of raising — used for mid-run checkpoints between sub_framings."""
    out: dict[str, dict[str, Any]] = {}
    for global_idx, item in enumerate(all_items):
        if global_idx not in scores_by_global_idx:
            continue
        cell_tag = item["_cell_tag"]
        cell_out = out.setdefault(cell_tag, {})
        key = f"{item['persona']}__{item['sub_framing']}"
        rec = cell_out.setdefault(
            key,
            {
                "n": 0,
                "by_label": {label: 0 for label in VALID_LABELS} | {"error": 0},
                "items": [],
            },
        )
        score = scores_by_global_idx[global_idx]
        label, reason, is_error = _strict_label(score)
        rec["n"] += 1
        if is_error:
            rec["by_label"]["error"] += 1
        else:
            rec["by_label"][label] += 1
        rec["items"].append(
            {
                "probe": item["probe"],
                "completion": item["completion"],
                "strict_predicate": label if not is_error else None,
                "strict_reason": reason,
                "strict_error": is_error,
                "permissive_predicate": item["permissive_predicate"],
                "permissive_reason": item["permissive_reason"],
                "_baseline_gated": item["_baseline_gated"],
            }
        )
    return out


def _assemble_per_cell(
    all_items: list[dict[str, Any]],
    scores_by_global_idx: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Split flat judge scores back into per-cell, per-(persona, sub_framing)
    structures matching the source `C_counter_association_results.json` shape.

    Output shape (one entry per cell):

        {
          "<persona>__<sub_framing>": {
            "n": int,
            "by_label": {label: count, ..., "error": count},
            "items": [{probe, completion, strict_predicate, strict_reason,
                       strict_error, permissive_predicate, permissive_reason,
                       _baseline_gated}, ...],
          }, ...
        }
    """
    out: dict[str, dict[str, Any]] = {}
    for global_idx, item in enumerate(all_items):
        cell_tag = item["_cell_tag"]
        cell_out = out.setdefault(cell_tag, {})
        key = f"{item['persona']}__{item['sub_framing']}"
        rec = cell_out.setdefault(
            key,
            {
                "n": 0,
                "by_label": {label: 0 for label in VALID_LABELS} | {"error": 0},
                "items": [],
            },
        )
        score = scores_by_global_idx[global_idx]
        label, reason, is_error = _strict_label(score)
        rec["n"] += 1
        if is_error:
            rec["by_label"]["error"] += 1
        else:
            rec["by_label"][label] += 1
        rec["items"].append(
            {
                "probe": item["probe"],
                "completion": item["completion"],
                "strict_predicate": label if not is_error else None,
                "strict_reason": reason,
                "strict_error": is_error,
                "permissive_predicate": item["permissive_predicate"],
                "permissive_reason": item["permissive_reason"],
                "_baseline_gated": item["_baseline_gated"],
            }
        )
    return out


# ── Aggregation across personas + seeds ───────────────────────────────────────


def _persona_rates_per_cell(per_cell_strict: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per cell, compute per-persona strict-pass rates (predicate match to
    gated persona's intended predicate isn't applied here — we just compute
    raw label counts, and the aggregator handles gated alignment).

    The strict-C "pass rate per persona" reported in the comparison is:
        (# items labeled `autoimmune_basal_ganglia` OR `metabolic_liver`) /
        (total items, excluding errors).

    Mixed / neither / refused / error all count as "not strict pass".
    """
    out: dict[str, Any] = {}
    for cell_tag, cell_data in per_cell_strict.items():
        # Re-aggregate across sub-framings to per-persona totals
        persona_totals: dict[str, dict[str, int]] = {}
        for key, rec in cell_data.items():
            persona, _sub = _parse_persona_subframing(key)
            pt = persona_totals.setdefault(
                persona,
                {label: 0 for label in VALID_LABELS} | {"error": 0, "_n": 0},
            )
            for label, cnt in rec["by_label"].items():
                pt[label] += cnt
            pt["_n"] += rec["n"]

        cell_summary: dict[str, Any] = {}
        for persona, pt in persona_totals.items():
            n = pt["_n"]
            n_pass = pt["autoimmune_basal_ganglia"] + pt["metabolic_liver"]
            n_errors = pt["error"]
            n_valid = n - n_errors
            cell_summary[persona] = {
                "n_total": n,
                "n_errors": n_errors,
                "n_valid": n_valid,
                "n_strict_pass": n_pass,
                "strict_pass_rate": (n_pass / n_valid) if n_valid > 0 else None,
                "by_label": {label: pt[label] for label in VALID_LABELS} | {"error": n_errors},
            }
        out[cell_tag] = cell_summary
    return out


def _aggregate_3seed_means(per_cell_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """3-seed means per condition × persona (drops baseline; n=1 there)."""
    by_condition: dict[str, dict[str, list[float]]] = {
        "contradictory-predicates": defaultdict(list),
        "reversed-assignment": defaultdict(list),
    }
    for cell_tag, persona_summary in per_cell_summary.items():
        for cond in by_condition:
            prefix = f"{cond}_seed"
            if cell_tag.startswith(prefix):
                for persona, stats in persona_summary.items():
                    if stats["strict_pass_rate"] is not None:
                        by_condition[cond][persona].append(stats["strict_pass_rate"])

    out: dict[str, Any] = {}
    for cond, persona_rates in by_condition.items():
        cond_out: dict[str, Any] = {}
        for persona, rates in persona_rates.items():
            if not rates:
                continue
            cond_out[persona] = {
                "n_seeds": len(rates),
                "rate_strict_3seed_mean": round(sum(rates) / len(rates), 4),
                "rate_strict_min": round(min(rates), 4),
                "rate_strict_max": round(max(rates), 4),
                "rates_per_seed": [round(r, 4) for r in rates],
            }
        out[cond] = cond_out

    # Baseline as a separate top-level key (n=1, no mean)
    base_summary = per_cell_summary.get(BASELINE_CELL, {})
    out["unmodified-baseline"] = {
        persona: {
            "n_seeds": 1,
            "rate_strict_3seed_mean": round(stats["strict_pass_rate"], 4)
            if stats["strict_pass_rate"] is not None
            else None,
            "n_total": stats["n_total"],
            "n_errors": stats["n_errors"],
        }
        for persona, stats in base_summary.items()
    }
    return out


# ── Comparison-to-permissive helper ───────────────────────────────────────────


def _permissive_pass_rate_for_persona(cell_tag: str, persona: str) -> tuple[int, int, float | None]:
    """Re-derive the permissive C-family per-persona pass rate from the source
    `C_counter_association_results.json`.

    Returns (n_total, n_pass, rate) where n_pass = items labelled
    `autoimmune_basal_ganglia` OR `metabolic_liver` by the permissive judge.
    """
    if cell_tag in TRAINED_CELLS:
        c_files = [EVAL_RESULTS_DIR / "cells" / cell_tag / "C_counter_association_results.json"]
    elif cell_tag == BASELINE_CELL:
        c_files = [
            EVAL_RESULTS_DIR / "cells" / cell_tag / sub / "C_counter_association_results.json"
            for sub in BASELINE_GATED_SUBDIRS
        ]
    else:
        raise ValueError(f"unknown cell_tag {cell_tag!r}")

    n_total = 0
    n_pass = 0
    for c_file in c_files:
        data = json.loads(c_file.read_text())
        for key, rec in data.items():
            this_persona, _sub = _parse_persona_subframing(key)
            if this_persona != persona:
                continue
            by_label = rec["by_label"]
            n_total += rec["n"]
            n_pass += by_label.get("autoimmune_basal_ganglia", 0) + by_label.get(
                "metabolic_liver", 0
            )
    rate = (n_pass / n_total) if n_total > 0 else None
    return n_total, n_pass, rate


def _build_comparison(per_cell_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the comparison_to_permissive.json structure: per cell × persona,
    permissive pass rate vs strict pass rate vs delta (permissive - strict)."""
    out: dict[str, Any] = {}
    for cell_tag, persona_summary in per_cell_summary.items():
        cell_out: dict[str, Any] = {}
        for persona, stats in persona_summary.items():
            n_total, n_pass_perm, perm_rate = _permissive_pass_rate_for_persona(cell_tag, persona)
            strict_rate = stats["strict_pass_rate"]
            delta = (
                (perm_rate - strict_rate)
                if perm_rate is not None and strict_rate is not None
                else None
            )
            cell_out[persona] = {
                "permissive": {
                    "n_total": n_total,
                    "n_pass": n_pass_perm,
                    "rate": round(perm_rate, 4) if perm_rate is not None else None,
                },
                "strict": {
                    "n_total": stats["n_total"],
                    "n_pass": stats["n_strict_pass"],
                    "n_errors": stats["n_errors"],
                    "rate": round(strict_rate, 4) if strict_rate is not None else None,
                },
                "delta_perm_minus_strict": round(delta, 4) if delta is not None else None,
            }
        out[cell_tag] = cell_out
    return out


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    # Load .env from repo root if present (mirrors `setup_env()` pattern in
    # `src/explore_persona_space/utils.py`). Lets the script be launched as
    # `uv run python scripts/rejudge_issue_389_c_strict.py` without a manual
    # `source .env` first.
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(env_path)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            f"ANTHROPIC_API_KEY not set; export the key or add it to {env_path} before running"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    all_items = _load_all_items()
    logger.info(
        "loaded %d C-family items across %d cells (%d trained + %d baseline)",
        len(all_items),
        len(TRAINED_CELLS) + 1,
        sum(1 for it in all_items if it["_cell_tag"] in TRAINED_CELLS),
        sum(1 for it in all_items if it["_cell_tag"] == BASELINE_CELL),
    )

    # Bucket by sub_framing so each sub-framing gets ONE filled rubric + one
    # batch submission across ALL cells. 4 batches total instead of 28
    # (per-cell × per-sub-framing). Cross-cell merging is safe because the
    # JudgeCache key is (probe, completion) which is unique per item.
    by_sub: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for global_idx, item in enumerate(all_items):
        by_sub[item["sub_framing"]].append((global_idx, item["probe"], item["completion"]))

    scores_by_global_idx: dict[int, dict[str, Any]] = {}
    per_cell_path = OUT_DIR / "per_cell_strict_c.json"
    for sub_framing in sorted(by_sub):  # deterministic order
        sub_t0 = time.time()
        logger.info(
            "=== sub_framing: %s (%d items across all cells) ===",
            sub_framing,
            len(by_sub[sub_framing]),
        )
        sub_scores = _judge_strict_per_sub_framing(sub_framing, by_sub[sub_framing])
        scores_by_global_idx.update(sub_scores)
        logger.info(
            "sub_framing %s judged in %.1fs (%d scores so far)",
            sub_framing,
            time.time() - sub_t0,
            len(scores_by_global_idx),
        )
        # Checkpoint after each sub_framing completes (CLAUDE.md "Checkpoint
        # per phase" rule). Per-cell shape is rebuilt from partial scores;
        # we pass the FULL items list and the partial scores dict — items
        # without a score for this global index are skipped by
        # `_assemble_per_cell` so the file is incrementally complete.
        partial_per_cell = _assemble_per_cell_partial(all_items, scores_by_global_idx)
        per_cell_path.write_text(
            json.dumps(
                {
                    "phase": "per_cell_strict_c",
                    "reproducibility": _repro_meta(),
                    "sub_framings_completed": sorted(
                        {all_items[i]["sub_framing"] for i in scores_by_global_idx}
                    ),
                    "by_cell": partial_per_cell,
                },
                indent=2,
            )
        )
        logger.info("checkpoint written -> %s", per_cell_path)

    # Final per-cell rebuild (with all 4 sub_framings).
    per_cell_strict = _assemble_per_cell(all_items, scores_by_global_idx)
    per_cell_path.write_text(
        json.dumps(
            {
                "phase": "per_cell_strict_c",
                "reproducibility": _repro_meta(),
                "sub_framings_completed": sorted(by_sub.keys()),
                "by_cell": per_cell_strict,
            },
            indent=2,
        )
    )

    # Aggregates rebuilt from the final per-cell dict.
    per_cell_summary = _persona_rates_per_cell(per_cell_strict)
    aggregate = _aggregate_3seed_means(per_cell_summary)
    comparison = _build_comparison(per_cell_summary)

    agg_path = OUT_DIR / "aggregate_3seed_means_strict_c.json"
    agg_path.write_text(
        json.dumps(
            {
                "phase": "aggregate_3seed_means_strict_c",
                "reproducibility": _repro_meta(),
                "by_condition": aggregate,
                "per_cell_persona_summary": per_cell_summary,
            },
            indent=2,
        )
    )

    cmp_path = OUT_DIR / "comparison_to_permissive.json"
    cmp_path.write_text(
        json.dumps(
            {
                "phase": "comparison_to_permissive",
                "reproducibility": _repro_meta(),
                "note": (
                    "Permissive = original COUNTER_ASSOCIATION_RUBRIC (v1). "
                    "Strict = COUNTER_ASSOCIATION_STRICT_RUBRIC (v1_strict), which "
                    "requires the completion to literally name (or use a close "
                    "lexical variant of) the rule-derived non-canonical answer. "
                    "delta_perm_minus_strict > 0 means the permissive rubric "
                    "passed items the strict rubric refuses (bare-predicate "
                    "emission without rule-derived-answer mention)."
                ),
                "by_cell": comparison,
            },
            indent=2,
        )
    )

    logger.info("done in %.1fs; wrote:", time.time() - t0)
    logger.info("  %s", per_cell_path)
    logger.info("  %s", agg_path)
    logger.info("  %s", cmp_path)


if __name__ == "__main__":
    main()
