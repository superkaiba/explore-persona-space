"""Cap-truncation census over every #1336 v2 generation cell (0 GPU-h).

WHY THIS EXISTS. The prefix-continuation regen round set out to REPLACE the
cap-truncated answers with uncensored ones and proved that impossible inside the
fixed ``MAX_CONV_TOKENS`` capture window: the median continuation still hit a cap,
56% of completions overran the window, and the rows that survived were selected for
being SHORT (see the round's ``epm:progress`` falsification note). So the truncation
stays in the pool and becomes a SCOPE CAVEAT instead of a defect to repair.

A caveat needs two numbers, and they have very different costs:

  PREVALENCE (how many rows in each cell were cut off) is already recorded. Every
  generation cell's ``audit.json`` carries ``kept_truncation_rate`` — the fraction
  of KEPT rows whose ``finish_reason`` was ``"length"`` (``issue1336_gen_answers``
  counts it at the ``truncated += int(finish == "length")`` line). Reading it back
  costs 70 audit downloads of a few KB each: no GPU, no generation, no corpus text.

  SEVERITY (how much text each cut-off row lost) requires GENERATION, because the
  unwritten tail is unknowable from the stored answer. The regen round measured it
  on ONE production cell, and that measurement is already paid for.

This script assembles the free half across ALL cells and attaches the paid
severity anchor, so the caveat is quantified per model and per corpus rather than
asserted. It reads audits only -- never corpus text, never a rollout body.

WHAT IT DELIBERATELY DOES NOT CLAIM. The severity anchor is ONE cell. The strip
split already measured on two cells (25.4% on base/lmsys23k vs 43% on
base/math7500) shows these fractions vary strongly by corpus, so the anchor
establishes that truncation is SUBSTANTIAL, not that its magnitude is uniform.
Generalizing severity needs a sampled regen sweep at a RAISED tail cap (the
measurement does not have to respect the capture window -- only a turnstore does);
that is priced in the round's note and is not run here.

Usage:
    uv run python scripts/issue1336_cap_truncation_census.py
    uv run python scripts/issue1336_cap_truncation_census.py --out <path.json>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from explore_persona_space.experiments.issue_1336 import common as cm
from explore_persona_space.task_workflow import repo_root

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1336_gen_answers as gen  # noqa: E402
import issue1336_regen_truncated as rt  # noqa: E402

#: The one cell where the regen round MEASURED severity (tail cap raised from the
#: 1024 generation cap to the 2048 total budget). Its audit carries the realized
#: answer-length distribution the cap had hidden.
SEVERITY_ANCHOR_SLUG = "base"
SEVERITY_ANCHOR_CELL = "lmsys23k"


def enumerate_cells() -> list[tuple[str, str, str]]:
    """Every (model slug, corpus, gen_format) the v2 generation registry licenses.

    The generation scope is ``cm.V2_GEN_FORMATS`` -- both formats for all seven v2
    corpora, 14 cells per model, 70 total. It is NOT ``cm.V2_PREFIX_ARM`` (a
    single fit-side arm); conflating the two undercounts the grid by 30 cells.
    """
    cells: list[tuple[str, str, str]] = []
    for slug in cm.MODELS:
        for corpus in cm.V2_CORPORA:
            for fmt in cm.V2_GEN_FORMATS[corpus]:
                cells.append((slug, corpus, fmt))
    return cells


def read_cell_audit(slug: str, corpus: str, gen_format: str) -> dict | None:
    """One cell's generation audit, or None when the cell was never generated.

    Absence is a legitimate outcome (a corpus/format pair that never ran), not an
    error -- it is counted and reported rather than raised, so a partial grid still
    produces a census.
    """
    cell = cm.gen_cell_key(corpus, gen_format)
    prefix = gen._hf_gen_prefix(slug, cell)
    try:
        return json.loads(gen._download_one(f"{prefix}/audit.json").read_text())
    except Exception:
        return None


def census_row(slug: str, corpus: str, gen_format: str, audit: dict) -> dict:
    """Per-cell prevalence record.

    ``kept_truncation_rate`` is a rate over KEPT rows, so the count is recovered as
    ``rate * n_kept``. It is reported as ``None`` (not 0.0) by the generator when a
    cell kept nothing -- carry that through rather than coercing to zero, which
    would read as "no truncation" instead of "no data".
    """
    n_kept = audit.get("n_kept")
    rate = audit.get("kept_truncation_rate")
    n_trunc = None if (rate is None or n_kept is None) else int(round(rate * n_kept))
    return {
        "model": slug,
        "corpus": corpus,
        "gen_format": gen_format,
        "cell": cm.gen_cell_key(corpus, gen_format),
        "n_prompts": audit.get("n_prompts"),
        "n_kept": n_kept,
        "keep_rate": audit.get("keep_rate"),
        "kept_truncation_rate": rate,
        "n_kept_truncated": n_trunc,
    }


def _rate_stats(rows: list[dict]) -> dict:
    """Aggregate prevalence over a group of cells (kept-row weighted + unweighted).

    Both reads are reported because they answer different questions: the weighted
    rate is what fraction of the group's analyzed rows are censored, the unweighted
    median says whether that is broad or driven by a few cells.
    """
    rated = [r for r in rows if r["kept_truncation_rate"] is not None]
    kept = sum(r["n_kept"] for r in rated)
    trunc = sum(r["n_kept_truncated"] for r in rated)
    return {
        "n_cells": len(rows),
        "n_cells_with_data": len(rated),
        "n_kept_total": kept,
        "n_kept_truncated_total": trunc,
        "weighted_truncation_rate": (trunc / kept) if kept else None,
        "median_cell_rate": (
            statistics.median(r["kept_truncation_rate"] for r in rated) if rated else None
        ),
        "min_cell_rate": min((r["kept_truncation_rate"] for r in rated), default=None),
        "max_cell_rate": max((r["kept_truncation_rate"] for r in rated), default=None),
    }


def _discover_anchor_budget() -> int | None:
    """The total answer budget the regenerated anchor cell was written under.

    Discovered from the Hub rather than assumed: the tail cap is per-invocation CLI
    state by design (``--tail-max-tokens``), so there is no module constant to read,
    and hardcoding a budget here would silently stop matching the moment a later
    round regenerates at a different one. Scans the model's generation prefix for
    ``<anchor cell>_cont<N>`` siblings and takes the LARGEST N -- the most
    generous budget any round has measured under.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    parent = f"{cm.HF_PREFIX_1336}/raw_completions/generation/{SEVERITY_ANCHOR_SLUG}"
    try:
        files = hub.list_hf_files_under_path(HfApi(), cm.HF_DATA_REPO, parent, repo_type="dataset")
    except Exception:
        return None
    pat = re.compile(rf"/{re.escape(SEVERITY_ANCHOR_CELL)}_cont(\d+)/")
    budgets = {int(m.group(1)) for f in files if (m := pat.search(f))}
    return max(budgets) if budgets else None


def severity_anchor() -> dict | None:
    """The measured answer-length distribution from the one regenerated cell.

    Reads the ``_cont<total>`` audit the regen round uploaded. Every field is a
    LOWER BOUND on true answer length: the continuation was itself capped at the
    tail budget, so a row reported at the total budget ran at least that long and
    an unknown amount further.
    """
    total = _discover_anchor_budget()
    if total is None:
        return None
    cell = rt.cont_cell_key(SEVERITY_ANCHOR_CELL, total)
    prefix = gen._hf_gen_prefix(SEVERITY_ANCHOR_SLUG, cell)
    try:
        audit = json.loads(gen._download_one(f"{prefix}/audit.json").read_text())
    except Exception:
        return None
    return {
        "measured_on": f"{SEVERITY_ANCHOR_SLUG}/{SEVERITY_ANCHOR_CELL}",
        "original_cap_tokens": cm.SAMPLING["max_tokens"],
        "tail_cap_tokens": total - cm.SAMPLING["max_tokens"],
        "total_budget_tokens": total,
        "n_continued": audit.get("n_regenerated"),
        "prefix_match_rate": audit.get("prefix_match_rate"),
        "new_answer_tokens": audit.get("new_answer_tokens"),
        "tail_answer_tokens": audit.get("tail_answer_tokens"),
        "over_token_budget_drop_rate": audit.get("over_token_budget_drop_rate"),
        "keep_rate": audit.get("keep_rate"),
        "stored_length_partition": audit.get("stored_length_partition"),
    }


def build_census() -> dict:
    """Full census: per-cell prevalence + per-model/per-corpus rollups + anchor."""
    rows: list[dict] = []
    missing: list[str] = []
    for slug, corpus, fmt in enumerate_cells():
        audit = read_cell_audit(slug, corpus, fmt)
        if audit is None:
            missing.append(f"{slug}/{cm.gen_cell_key(corpus, fmt)}")
            continue
        rows.append(census_row(slug, corpus, fmt, audit))

    by_model = {slug: _rate_stats([r for r in rows if r["model"] == slug]) for slug in cm.MODELS}
    by_corpus = {
        corpus: _rate_stats([r for r in rows if r["corpus"] == corpus]) for corpus in cm.V2_CORPORA
    }
    by_format = {
        fmt: _rate_stats([r for r in rows if r["gen_format"] == fmt])
        for fmt in ("chat", "naturalistic")
    }
    return {
        "generation_cap_tokens": cm.SAMPLING["max_tokens"],
        "capture_window_tokens": getattr(cm, "MAX_CONV_TOKENS", None),
        "n_cells_enumerated": len(enumerate_cells()),
        "n_cells_read": len(rows),
        "cells_missing": missing,
        "overall": _rate_stats(rows),
        "by_model": by_model,
        "by_corpus": by_corpus,
        "by_gen_format": by_format,
        "per_cell": rows,
        "severity_anchor": severity_anchor(),
    }


def print_report(census: dict) -> None:
    """Human-legible summary; the JSON is the artifact."""
    ov = census["overall"]
    print(
        f"[census] {census['n_cells_read']}/{census['n_cells_enumerated']} cells read; "
        f"cap={census['generation_cap_tokens']} window={census['capture_window_tokens']}"
    )
    print(
        f"[census] overall: {ov['n_kept_truncated_total']}/{ov['n_kept_total']} kept rows "
        f"cap-truncated (weighted {ov['weighted_truncation_rate']:.4f}); "
        f"per-cell rate median {ov['median_cell_rate']:.4f} "
        f"range [{ov['min_cell_rate']:.4f}, {ov['max_cell_rate']:.4f}]"
    )
    for label, group in (("model", census["by_model"]), ("corpus", census["by_corpus"])):
        for key, st in group.items():
            if st["weighted_truncation_rate"] is None:
                print(f"[census] {label}={key}: no data")
                continue
            print(
                f"[census] {label}={key}: {st['n_kept_truncated_total']}/{st['n_kept_total']} "
                f"({st['weighted_truncation_rate']:.4f}) over {st['n_cells_with_data']} cells"
            )
    anchor = census["severity_anchor"]
    if anchor is None:
        print("[census] severity anchor: ABSENT (regenerated cell not on the Hub)")
    else:
        naw = anchor["new_answer_tokens"] or {}
        print(
            f"[census] severity anchor {anchor['measured_on']}: "
            f"continued {anchor['n_continued']} rows to budget "
            f"{anchor['total_budget_tokens']}; completed-answer tokens median "
            f"{naw.get('median')} mean {naw.get('mean')}; "
            f"over-window drop rate {anchor['over_token_budget_drop_rate']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="census JSON destination (default: eval_results/issue_1336/"
        "cap_truncation_census/census.json under the repo root)",
    )
    args = ap.parse_args()
    out = args.out or (
        repo_root() / "eval_results" / "issue_1336" / "cap_truncation_census" / "census.json"
    )
    census = build_census()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(census, indent=2) + "\n")
    print_report(census)
    print(f"[census] wrote {out}")


if __name__ == "__main__":
    main()
