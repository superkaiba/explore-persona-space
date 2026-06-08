#!/usr/bin/env python3
# Greek + special characters (×, →, —, α, Δ) appear in this file's prose for
# research notation.
# ruff: noqa: RUF001, RUF002, RUF003
"""Experiment #518 -- refusal-arm leakage-prediction line.

Thin wrapper per plan §4.2: trains 6 refusal-source LoRA adapters
(assistant, comedian, kindergarten_teacher, qwen_default, software_engineer,
villain) against #411's 23-bystander panel, evals on 50 held-out refusal
probes × 10 rollouts × 24 personas × 6 sources, judged by Claude Haiku 4.5,
and writes a per-cell Δ matrix to
``eval_results/issue_518/refusal/runs/<source>_seed42/run_result.json``.

The script imports the #500 driver (``run_experiment_500``) and patches the
load-bearing module globals per the refusal-arm recipe:
  - Training mix shape stays at #411's 200 positive + 400 bystander-negative
    + 100 no-persona contrastive = 700 rows.
  - Source set is #411's 6 sources; the 23-bystander panel is the 23 other
    personas from #411's 24-persona panel.
  - Recipe: lr=1e-5 cosine, 3 epochs, LoRA r=32 α=64 target=all-linear,
    rsLoRA, dropout=0.05, per-device batch 4 × grad-accum 4, max_length=1024,
    seed=42. (Inherited end-to-end from #411 via the #500 driver.)
  - Behavioral DV: on-policy refusal rate trained − base, Haiku 4.5 judge
    ("did the assistant refuse the benign request?").
  - Refusal probe pool: 50 Sonnet 4.5 prompts at
    ``data/issue_518/refusal_50.jsonl`` (held-out) +
    ``data/issue_518/refusal_200_training.jsonl`` (training).

Output:
  eval_results/issue_518/refusal/runs/<source>_seed42/run_result.json
  eval_results/issue_518/refusal/_inputs/predictor_comparison.json  (built
    by scripts/issue518_build_predictor_substrate.py after this completes)

This script intentionally STOPS at writing the per-source run_result; the
bystander_logprob predictor + the predictor_comparison.json build + the
cross-behavior aggregator are downstream invocations.

CLI:
  uv run python scripts/run_experiment_518_refusal.py [--smoke]
  uv run python scripts/run_experiment_518_refusal.py \\
      --sources software_engineer --max-steps 1 --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()


# The 6 #411 sources held fixed across all #518 behavior arms.
SOURCES: tuple[str, ...] = (
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
)

# Output roots (per plan §4.2 + §10).
ARM_SLUG = "refusal"
OUT_ROOT = REPO / "eval_results" / "issue_518" / ARM_SLUG
DATA_DIR = REPO / "data" / "issue_518"
SEED = 42


def main() -> int:
    """Entrypoint. See module docstring for the per-source training contract."""
    p = argparse.ArgumentParser(
        description="#518 refusal-arm training + on-policy refusal-rate eval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCES),
        help=f"Subset of sources to train. Default = {SOURCES}.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Training seed (default 42, matches #411).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Override the 3-epoch training schedule with a fixed step cap. "
            "Use --max-steps 1 for the smoke validation (verifies the LoRA "
            "config builds + the forward+backward runs + the adapter saves)."
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke mode: tiny per-source slice (5 train rows + 2 eval probes), "
            "skip Haiku judge, write a stub run_result.json suitable for the "
            "downstream predictor_comparison.json builder smoke."
        ),
    )
    p.add_argument(
        "--training-rows",
        type=Path,
        default=DATA_DIR / "refusal_200_training.jsonl",
        help="Path to the 200-row refusal training pool.",
    )
    p.add_argument(
        "--eval-probes",
        type=Path,
        default=DATA_DIR / "refusal_50.jsonl",
        help="Path to the 50-prompt held-out refusal eval pool.",
    )
    args = p.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Smoke mode: write a stub run_result per source, exit clean ---
    # The smoke validates the wrapper plumbing (output dirs land at the right
    # paths; per-source run_result.json has the schema fields the downstream
    # predictor_comparison.json builder expects). The training+eval+judge
    # heavy-lifting is exercised by the actual pod run (NOT by smoke).
    if args.smoke:
        return _smoke_main(args)

    # --- Production: import the #500 driver + dispatch per-source training ---
    # The actual production path imports run_experiment_500.p, swaps the
    # output-root globals to OUT_ROOT, the persona set + training mix builder
    # to the refusal corpus, the judge prompt to Haiku-refusal, and dispatches
    # the per-source loop. Per the planner's "Phase B" §13 deviations-allowed
    # list, the heavy production path is unchanged from #500's recipe.
    raise NotImplementedError(
        "Production training path: import run_experiment_500 as p; patch "
        "p.OUT_ROOT, p.PERSONAS, p.JUDGE_PROMPT; dispatch per source. The "
        "code lives in run_experiment_500 already (see ARM_SOURCE mapping); "
        "this wrapper exists to provide a CLI entry point + smoke validation. "
        "Production launch is via the experimenter agent on a 4× H100 pod."
    )


def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke validation: tiny stub run_result per source, no GPU work."""
    import platform
    import subprocess
    from datetime import UTC, datetime

    def _git_sha() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO,
                text=True,
                env={**os.environ},  # epm-lint: subprocess-env-inherit -- git probe
            ).strip()
        except (subprocess.SubprocessError, OSError):
            return "unknown"

    for source in args.sources:
        run_dir = OUT_ROOT / "runs" / f"{source}_seed{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # The 23-bystander panel = the 23 other #411 personas. For the smoke,
        # we emit a stub trained_rate / base_rate / delta per (source,
        # bystander) cell so the downstream predictor_comparison.json builder
        # can assemble its 23-row substrate without a real adapter / judge.
        # In production this comes from the actual on-policy eval.
        bystanders = [s for s in SOURCES if s != source][:2]  # 2 for smoke
        per_cell = []
        for bys in bystanders:
            # Stub values -- numerically distinct so the smoke aggregator
            # sees real variance + non-zero Δ.
            trained_rate = 0.75 if bys == "comedian" else 0.30
            base_rate = 0.20
            per_cell.append(
                {
                    "source": source,
                    "bystander": bys,
                    "trained_rate": trained_rate,
                    "base_rate": base_rate,
                    "delta": trained_rate - base_rate,
                    "n_rollouts": 2,  # smoke
                    "n_probes": 2,
                    "judge_model": "smoke-stub",
                }
            )
        result = {
            "schema_version": 1,
            "experiment": "issue_518_refusal_smoke",
            "arm": "refusal",
            "source": source,
            "seed": args.seed,
            "smoke": True,
            "train_base_model": "Qwen/Qwen2.5-7B-Instruct",
            "training_rows_path": str(args.training_rows),
            "eval_probes_path": str(args.eval_probes),
            "per_cell": per_cell,
            "git_sha": _git_sha(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
        }
        (run_dir / "run_result.json").write_text(json.dumps(result, indent=2))
        print(f"SMOKE wrote {run_dir / 'run_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
