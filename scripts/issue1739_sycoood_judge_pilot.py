#!/usr/bin/env python3
"""#1739 sycophancy OOD rungs — rule-26 judge PILOT GATE (run BEFORE the wave).

The production judge wave for this round is ~2,613 contexts x 5 rollouts x 3
draws ~= 39k calls, far over the ~5,000-call threshold at which
`.claude/rules/llm-judging.md` rule 26 requires a pilot gate. Rule 23's binding
per-arm drop check is POST-HOC — it can only be measured after the whole wave is
spent — so every miss costs a full re-judge (#1739's own 400->800 re-judge over
86,521x3 draws is the motivating case). ~200 pilot draws buy that back.

Arms are the five RUNGS, so the gate reads per-rung: a truncation or parse-fail
signature concentrated in one rung (the long `sycofb` artifacts, or the
`sycoays` transcripts, are the plausible offenders) surfaces before the spend
rather than after.

Runs at the EXACT production instrument — same rubric (`trait_eval_prompt` via
`judging.load_trait_rubric`), same judge model, same temperature, same
`max_tokens` — against a FRESH pilot `cache_dir` so production reuse is never a
silent replay (rule 24(ii)).

Gate (rule 26): zero `stop_reason == "max_tokens"` (never waivable) AND per-arm
parse-fail < ~2%. On PASS, run the production wave through the UNMODIFIED
`scripts/issue1739_judge.py` at the SAME `--max-tokens`; the gate does not judge
the wave itself.

CONTENT HYGIENE: prints ids, counts, and rates — never question/answer text.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Repo root onto sys.path (script mode puts only scripts/ there — #823)."""
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_sycoood_judge_pilot.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root derivation failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue1739_sycoood_judge_pilot")

BEHAVIOR = "sycophancy"


def main() -> int:
    """Run the rule-26 pilot over the five rungs; exit non-zero on a FAIL gate."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    ap = argparse.ArgumentParser(description="#1739 syco-OOD judge pilot gate")
    ap.add_argument(
        "--rollout-dir", default="raw_completions/issue_1739_syco_ood/main/labeling/sycophancy"
    )
    ap.add_argument("--out-dir", default="eval_results/issue_1739/syco_ood/judge_pilot")
    ap.add_argument(
        "--cache-dir", default=None, help="FRESH pilot cache (default: <out-dir>/cache)"
    )
    ap.add_argument("--inputs-dir", default="data/issue_1739/inputs")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--n-draws", type=int, default=2)
    ap.add_argument("--target-total-draws", type=int, default=200)
    args = ap.parse_args()

    from explore_persona_space.eval.judge_pilot import judge_pilot_gate
    from explore_persona_space.experiments.issue_1739 import judging
    from explore_persona_space.experiments.issue_1739.constants import (
        JUDGE_MODEL,
        JUDGE_TEMPERATURE,
    )

    rollout_dir = Path(args.rollout_dir)
    paths = sorted(p for p in rollout_dir.glob("*.json") if p.name != "_manifest.json")
    if not paths:
        raise FileNotFoundError(f"no rollout JSONs under {rollout_dir}")

    arms: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for p in paths:
        rec = json.loads(p.read_text())
        rung = rec.get("rung") or "unknown"
        arms[rung].append(
            (
                judging.rollout_item_id(rec["context_id"], int(rec["rollout_k"])),
                rec["query"],
                rec["completion"],
            )
        )
    logger.info(
        "[pilot] rollout_files=%d arms=%s",
        len(paths),
        {k: len(v) for k, v in sorted(arms.items())},
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"

    report = judge_pilot_gate(
        dict(arms),
        judging.load_trait_rubric(BEHAVIOR, inputs_dir=args.inputs_dir),
        max_tokens=args.max_tokens,
        cache_dir=cache_dir,
        save_raw_dir=out_dir,
        n_draws=args.n_draws,
        target_total_draws=args.target_total_draws,
        judge_model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
        report_path=out_dir / "pilot_gate_report.json",
    )
    passed = bool(getattr(report, "passed", False))
    logger.info("[pilot] report -> %s", out_dir / "pilot_gate_report.json")
    logger.info("[pilot] VERDICT=%s", "PASS" if passed else "FAIL")
    sys.stdout.flush()
    sys.stderr.flush()
    # Distinct rc for a DESIGNED gate refusal (report written BEFORE exit), so a
    # dispatcher routes it like a stop criterion rather than an anonymous crash.
    os._exit(0 if passed else 7)


if __name__ == "__main__":
    main()
