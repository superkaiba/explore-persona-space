#!/usr/bin/env python3
"""Issue #623 phase 5 — resolve the behavioral DV syc_i (REUSE #612 base rates).

For each panel persona, ``syc_i`` = the #612 base sycophancy ``rate`` (fraction
YES over 600 generations: 60 claims x 10 rollouts, temp 1.0, judge
``claude-haiku-4-5-20251001``). NO new generation — every resolvable-prompt
panel persona already has a committed base rate in
``eval_results/issue_612/base/judgments/<persona>.json`` (git-tracked, NOT on HF;
plan §10 / §11(f)).

Reads the resolved ``panel_prompts.json`` (phase 1), looks up each persona's
committed rate, and writes ``eval_results/issue_623/syc_i.json``. Fail-fast: a
panel persona with NO committed base rate is a HARD error (the plan's reuse
fitness check (c) confirmed all 36 are present; a miss means the panel drifted).

The ``assistant`` baseline-self IS included in syc_i.json (rate=0.05) with an
``is_baseline_self`` flag so the analyzer drops it before Spearman — never a
silent omission.

Usage:
  uv run python scripts/issue623_behavioral_dv.py \
      --panel-prompts data/persona_vectors/issue623/panel_prompts.json \
      --output eval_results/issue_623/syc_i.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.experiments.persona_decomp_623 import (
    BASELINE_PERSONA,
    SYC_I_JUDGMENTS_RELDIR,
    repo_root_from_module,
)
from explore_persona_space.orchestrate.env import load_dotenv


def load_base_rate(judgments_dir: Path, persona: str) -> dict:
    """Read the #612 committed base sycophancy judgment file for one persona."""
    fp = judgments_dir / f"{persona}.json"
    if not fp.exists():
        raise FileNotFoundError(
            f"No committed #612 base rate for panel persona {persona!r}: {fp}. "
            "The panel drifted from the reuse-fitness check (c) — investigate."
        )
    d = json.loads(fp.read_text())
    return {
        "rate": d["rate"],
        "n_verdicts": d.get("n_verdicts"),
        "judge_model": d.get("model"),
    }


def build_syc_i(repo_root: Path, panel_prompts_path: Path) -> dict:
    """Resolve syc_i for every persona in the panel manifest."""
    manifest = json.loads(panel_prompts_path.read_text())
    judgments_dir = repo_root / SYC_I_JUDGMENTS_RELDIR

    syc_i: dict[str, dict] = {}
    for persona, entry in manifest["personas"].items():
        rec = load_base_rate(judgments_dir, persona)
        syc_i[persona] = {
            "syc_i": rec["rate"],
            "n_verdicts": rec["n_verdicts"],
            "judge_model": rec["judge_model"],
            "is_baseline_self": entry.get("is_baseline_self", persona == BASELINE_PERSONA),
            "prompt_source": entry.get("source"),
        }

    correlation_personas = [p for p, v in syc_i.items() if not v["is_baseline_self"]]
    rates = [syc_i[p]["syc_i"] for p in correlation_personas]
    return {
        "schema_version": 1,
        "source": "REUSE #612 base pass (eval_results/issue_612/base/judgments/*.json)",
        "n_total": len(syc_i),
        "n_correlation": len(correlation_personas),
        "baseline_persona_dropped": BASELINE_PERSONA,
        "rate_min": min(rates) if rates else None,
        "rate_max": max(rates) if rates else None,
        "rate_distinct": len(set(rates)),
        "syc_i": syc_i,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #623 phase 5 — behavioral DV syc_i.")
    parser.add_argument(
        "--panel-prompts",
        default="data/persona_vectors/issue623/panel_prompts.json",
        help="panel_prompts.json from phase 1 (relative to repo root).",
    )
    parser.add_argument(
        "--output",
        default="eval_results/issue_623/syc_i.json",
        help="Output syc_i.json (relative to repo root).",
    )
    args = parser.parse_args()

    load_dotenv()
    repo_root = repo_root_from_module()

    panel_path = (
        repo_root / args.panel_prompts
        if not Path(args.panel_prompts).is_absolute()
        else Path(args.panel_prompts)
    )
    result = build_syc_i(repo_root, panel_path)

    out_path = repo_root / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(
        f"[phase=syc_i_load] {result['n_correlation']} correlation personas "
        f"(+1 baseline dropped), rate range "
        f"[{result['rate_min']}, {result['rate_max']}], {result['rate_distinct']} distinct",
        flush=True,
    )
    print(f"[phase=syc_i_load] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
