#!/usr/bin/env python3
"""Task #591 e1 — judge the #518 refusal/EM SELF-panel completions (manipulation checks).

#518 evaluated source-self panels (raw completions on the Hub) but never
aggregated them into committed JSONs; the behavioral self-implant rates are
the previously-missing H2 inputs (plan #591 v1 §4.1 item 2). This script
downloads the 12 Hub self-panel files (2 arms x 6 sources), judges the
trained-side rates with the SAME judges the arms used (refusal: Haiku binary
refused/not, #518 prompt; EM: Sonnet 4.5 aligned-0-100 + coherence filter,
#518 prompt), and writes::

    eval_results/issue_591/e1/self_cells/<arm>_<source>.json   (checkpoint per cell)
    eval_results/issue_591/e1/self_rates.json                  (aggregate)

with ``self_delta = judged_trained_rate - source_base_rate`` (base rates from
the per-arm joins). Also records per-cell EM coherence-survivor counts for
the e1 sensitivity filter. API-only, zero GPU.

Content hygiene: refusal/EM completions are harmful-content corpora — this
script never prints completion text, and the persisted verdict records carry
indices + scores only (the raw text stays on the Hub).

Smoke (1-2 completions per cell, real judge API):

    uv run python scripts/issue_591/i591_judge_self_cells.py --limit 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from explore_persona_space.experiments.issue_518 import (  # noqa: E402
    judge_em_panel as jep,
)
from explore_persona_space.experiments.issue_518 import (  # noqa: E402
    judge_refusal_panel as jrp,
)

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
SELF_PANEL_TMPL = (
    "issue518_leakage_prediction/raw_completions/{arm}/{source}/seed_42/"
    "raw_completions/{source}_seed42.json"
)
SOURCES = (
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
)
ARMS = ("refusal", "em")
# Pinned to the ids the #518 arms recorded (refusal join / EM run_result).
REFUSAL_JUDGE_MODEL = "claude-haiku-4-5-20251001"
EM_JUDGE_MODEL = "claude-sonnet-4-5-20250929"
OUT_ROOT_DEFAULT = REPO / "eval_results" / "issue_591"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _source_base_rates(out_root: Path) -> dict[tuple[str, str], float]:
    """source_base_rate per (arm, source) from the snapshotted per-arm joins."""
    rates: dict[tuple[str, str], float] = {}
    for arm in ARMS:
        join_path = out_root / "_inputs" / f"join_{arm}.json"
        if not join_path.exists():
            # Fall back to the canonical committed join (pre-build-table runs).
            join_path = REPO / f"eval_results/issue_518/{arm}/_inputs/predictor_comparison.json"
        cells = json.loads(join_path.read_text())["cells"]
        for c in cells:
            rates[(arm, c["source"])] = c["source_base_rate"]
    for arm in ARMS:
        for s in SOURCES:
            assert (arm, s) in rates, f"missing source_base_rate for ({arm}, {s})"
    return rates


def _download_self_panel(arm: str, source: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            HF_DATA_REPO,
            SELF_PANEL_TMPL.format(arm=arm, source=source),
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
    )


def _judge_refusal_cell(panel_path: Path, limit: int | None, concurrency: int) -> dict:
    """Judge one refusal self-panel file -> counts + rate (no completion text)."""
    rows = jrp._load_panel_completions(panel_path)
    n_available = len(rows)
    if limit is not None:
        rows = rows[:limit]
    verdicts = asyncio.run(
        jrp._judge_rows_async(
            rows, model=REFUSAL_JUDGE_MODEL, max_concurrency=concurrency, max_retries=3
        )
    )
    stats = jrp._aggregate_cell(verdicts)
    return {
        "judge_model": REFUSAL_JUDGE_MODEL,
        "judge_prompt_first_line": jrp.REFUSAL_JUDGE_PROMPT_TEMPLATE.splitlines()[0],
        "rate": stats.rate,
        "n_yes": stats.n_yes,
        "n_no": stats.n_no,
        "n_indeterminate": stats.n_indeterminate,
        "n_errors": stats.n_errors,
        "n_total": stats.n_total,
        "n_available": n_available,
        # Hygiene: indices + verdict only; raw text stays on the Hub.
        "verdicts": [
            {
                "claim_idx": r.get("claim_idx"),
                "rollout_idx": r.get("rollout_idx"),
                "refused": v.agreed,
                "raw_response": v.raw_response,
                "error": v.error,
            }
            for r, v in zip(rows, verdicts, strict=True)
        ],
    }


def _judge_em_cell(panel_path: Path, limit: int | None, concurrency: int) -> dict:
    """Judge one EM self-panel file -> coherence-filtered rate (no completion text)."""
    rows = jep._load_panel_completions(panel_path)
    n_available = len(rows)
    if limit is not None:
        rows = rows[:limit]
    verdicts = asyncio.run(
        jep._judge_rows_async(
            rows, model=EM_JUDGE_MODEL, max_concurrency=concurrency, max_retries=3
        )
    )
    agg = jep._aggregate_cell(verdicts)
    return {
        "judge_model": EM_JUDGE_MODEL,
        "judge_prompt_first_line": jep.EM_JUDGE_PROMPT_TEMPLATE.splitlines()[0],
        "coherence_threshold": jep.COHERENCE_THRESHOLD,
        "rate": agg["rate"],
        "aligned_mean": agg["aligned_mean"],
        "coherence_mean": agg["coherence_mean"],
        "n_total": agg["n_total"],
        "n_after_coherence_filter": agg["n_after_coherence_filter"],
        "n_parse_failed": agg["n_parse_failed"],
        "n_available": n_available,
        "verdicts": [
            {
                "claim_idx": r.get("claim_idx"),
                "rollout_idx": r.get("rollout_idx"),
                "aligned": v.aligned,
                "coherence": v.coherence,
                "error": v.error,
            }
            for r, v in zip(rows, verdicts, strict=True)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Judge #518 refusal/EM self-panel completions (the #591 e1 H2 inputs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Judge only the first N completions per cell (smoke; production omits this).",
    )
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument(
        "--arms", type=str, default="refusal,em", help="Comma-separated subset of arms."
    )
    parser.add_argument(
        "--sources", type=str, default=",".join(SOURCES), help="Comma-separated subset of sources."
    )
    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set; cannot judge.")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    assert set(arms).issubset(set(ARMS)), arms
    assert set(sources).issubset(set(SOURCES)), sources

    out_root: Path = args.out_root
    cells_dir = out_root / "e1" / "self_cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    base_rates = _source_base_rates(out_root)

    results: dict[str, dict[str, dict]] = {a: {} for a in ARMS}
    for arm in arms:
        for source in sources:
            cell_path = cells_dir / f"{arm}_{source}.json"
            if cell_path.exists():
                cell = json.loads(cell_path.read_text())
                if args.limit is None and cell.get("limit") is not None:
                    print(f"[self-cells] {arm}/{source}: smoke-tier checkpoint found, re-judging")
                else:
                    print(f"[self-cells] {arm}/{source}: checkpoint exists, skipping")
                    results[arm][source] = cell
                    continue
            panel_path = _download_self_panel(arm, source)
            print(f"[self-cells] judging {arm}/{source} (limit={args.limit}) ...")
            if arm == "refusal":
                cell = _judge_refusal_cell(panel_path, args.limit, args.concurrency)
            else:
                cell = _judge_em_cell(panel_path, args.limit, args.concurrency)
            cell["arm"] = arm
            cell["source"] = source
            cell["limit"] = args.limit
            cell["source_base_rate"] = base_rates[(arm, source)]
            cell["self_rate"] = cell["rate"]
            cell["self_delta"] = (
                cell["rate"] - base_rates[(arm, source)] if cell["rate"] == cell["rate"] else None
            )
            cell["hub_path"] = SELF_PANEL_TMPL.format(arm=arm, source=source)
            cell["git_commit_sha"] = _git_sha()
            cell["timestamp_utc"] = datetime.now(UTC).isoformat()
            # Checkpoint per cell — a later crash must not lose this judging pass.
            cell_path.write_text(json.dumps(cell, indent=2))
            print(
                f"[self-cells] {arm}/{source}: rate={cell['rate']:.3f} "
                f"delta={cell['self_delta']} n={cell['n_total']}/{cell['n_available']}"
            )
            results[arm][source] = cell

    complete = all(s in results[a] for a in ARMS for s in SOURCES)
    aggregate = {
        "arms": {
            arm: {
                s: {
                    k: v
                    for k, v in results[arm][s].items()
                    if k != "verdicts"  # per-verdict detail stays in the per-cell files
                }
                for s in results[arm]
            }
            for arm in results
        },
        "complete_12_cells": complete,
        "limit": args.limit,
        "git_commit_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    if complete and args.limit is None:
        out_path = out_root / "e1" / "self_rates.json"
    else:
        # Partial / smoke-tier output never overwrites the production artifact.
        out_path = out_root / "e1" / "self_rates_partial.json"
    out_path.write_text(json.dumps(aggregate, indent=2))
    print(f"[self-cells] aggregate -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
