#!/usr/bin/env python3
# Greek + special characters (×, →, —, α, Δ, ρ) appear in this file's prose
# for research notation.
# ruff: noqa: RUF001, RUF003
"""#518 v4 predictor_comparison.json substrate builder.

Assembles the per-arm 24-field cell schema from the per-source run_result
emitted by ``run_experiment_518_<arm>.py`` + the per-(source, bystander)
``completion_logprob`` cell from
``scripts/issue518_syco_logprob_backfill.py`` (run against the per-arm
teach-row substrate, not just the syco backfill).

Schema matches ``eval_results/issue_480/_inputs/predictor_comparison.json``
(23 fields from #480 + the new ``completion_logprob`` column = 24 fields):
  source, bystander, delta, cosine_l20_baseline, cosine_response_headline,
  trained_rate_<arm>, bystander_base_rate, source_base_rate,
  base_rate_diff_neg_abs, source_resp_len_mean, bystander_resp_len_mean,
  resp_len_diff_abs, cosine_response_l{7,14,21,27},
  JS_{sym,from_source,from_bystander}_nats, M_js,
  KL_{src_to_bys,bys_to_src,sym}_nats, completion_logprob.

Smoke mode emits stub coarse-zoo values so the downstream scoring +
aggregator have a non-degenerate input; the production path is fed by
the build of the #480 predictor sweep over the (refusal | em) substrate
(out of scope for this implementer round; the smoke + the documented
schema are enough to verify the wiring + the cross-behavior aggregator).

CLI:
  uv run python scripts/issue518_build_predictor_substrate.py \\
      --arm refusal --runs-root eval_results/issue_518/refusal/runs \\
      --logprob-file <path> --out <path>
  uv run python scripts/issue518_build_predictor_substrate.py \\
      --arm em --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# 23 #480 fields + 1 new #518 column.
PREDICTOR_FIELDS_24: tuple[str, ...] = (
    "source",
    "bystander",
    "delta",
    "cosine_l20_baseline",
    "cosine_response_headline",
    "trained_rate",
    "bystander_base_rate",
    "source_base_rate",
    "base_rate_diff_neg_abs",
    "source_resp_len_mean",
    "bystander_resp_len_mean",
    "resp_len_diff_abs",
    "cosine_response_l7",
    "cosine_response_l14",
    "cosine_response_l21",
    "cosine_response_l27",
    "JS_sym_nats",
    "JS_from_source_nats",
    "JS_from_bystander_nats",
    "M_js",
    "KL_src_to_bys_nats",
    "KL_bys_to_src_nats",
    "KL_sym_nats",
    "completion_logprob",
)


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


def _stub_coarse_zoo(source: str, bystander: str) -> dict[str, float]:
    """Smoke stub coarse-zoo values -- deterministic per (source, bystander).

    Returns numerically non-degenerate values so the downstream scoring
    + cross-behavior aggregator have a real predictor signal to score
    against the smoke Δ. The production builder replaces these with the
    real #480 sweep values.
    """
    # Hash-based but stable across runs.
    h = (sum(ord(c) for c in source) + sum(ord(c) for c in bystander)) % 100
    base = 0.01 * h
    return {
        "cosine_l20_baseline": 0.50 + base / 100,
        "cosine_response_headline": 0.40 + base / 80,
        "source_base_rate": 0.15 + base / 200,
        "base_rate_diff_neg_abs": -abs(base - 50) / 100,
        "source_resp_len_mean": 120.0 + base,
        "bystander_resp_len_mean": 100.0 + base,
        "resp_len_diff_abs": abs(base - 30),
        "cosine_response_l7": 0.45 + base / 100,
        "cosine_response_l14": 0.50 + base / 100,
        "cosine_response_l21": 0.52 + base / 100,
        "cosine_response_l27": 0.55 + base / 100,
        "JS_sym_nats": 0.12 + base / 500,
        "JS_from_source_nats": 0.10 + base / 500,
        "JS_from_bystander_nats": 0.14 + base / 500,
        "M_js": 0.20 + base / 250,
        "KL_src_to_bys_nats": 0.30 + base / 300,
        "KL_bys_to_src_nats": 0.32 + base / 300,
        "KL_sym_nats": 0.31 + base / 300,
    }


def _load_runs(
    runs_root: Path,
) -> dict[str, dict]:
    """Read every ``runs_root/<source>_seed*/run_result.json`` and key by source."""
    by_source: dict[str, dict] = {}
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root missing: {runs_root}")
    for run_dir in sorted(runs_root.iterdir()):
        run_json = run_dir / "run_result.json"
        if not run_json.exists():
            continue
        payload = json.loads(run_json.read_text())
        src = payload.get("source") or payload.get("teach_persona")
        if src is None:
            raise ValueError(f"run_result {run_json} missing 'source' field")
        by_source[src] = payload
    if not by_source:
        raise FileNotFoundError(f"No run_result.json under {runs_root}/*/")
    return by_source


def _load_completion_logprob(
    logprob_file: Path,
) -> dict[tuple[str, str], float]:
    """Read the bystander_logprob output -> map[(source, bystander)] -> mean."""
    payload = json.loads(logprob_file.read_text())
    summary = payload["summary"]
    out: dict[tuple[str, str], float] = {}
    for src, bys_map in summary.items():
        for bys, cell in bys_map.items():
            out[(src, bys)] = float(cell["mean_logprob_per_tok"])
    return out


def main() -> int:
    """Entrypoint. See module docstring."""
    p = argparse.ArgumentParser(
        description="#518 v4 predictor_comparison.json substrate builder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--arm",
        choices=("refusal", "em"),
        required=True,
        help="Which #518 behavior arm to build the substrate for.",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help=(
            "Directory containing per-source run_result.json files. Default: "
            "eval_results/issue_518/<arm>/runs."
        ),
    )
    p.add_argument(
        "--logprob-file",
        type=Path,
        default=None,
        help=(
            "Path to the bystander_logprob output for this arm. Default: "
            "eval_results/issue_518/<arm>/bystander_logprob/logprob_results.json."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Default: "
            "eval_results/issue_518/<arm>/_inputs/predictor_comparison.json."
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke mode: assemble the substrate from the smoke run_results + "
            "the smoke bystander_logprob output. Stub coarse-zoo values are "
            "deterministic per (source, bystander) so the smoke scoring + "
            "aggregator see non-degenerate variance."
        ),
    )
    args = p.parse_args()

    if args.runs_root is None:
        args.runs_root = REPO / "eval_results" / "issue_518" / args.arm / "runs"
    if args.logprob_file is None:
        args.logprob_file = (
            REPO
            / "eval_results"
            / "issue_518"
            / args.arm
            / "bystander_logprob"
            / "logprob_results.json"
        )
    if args.out is None:
        args.out = (
            REPO / "eval_results" / "issue_518" / args.arm / "_inputs" / "predictor_comparison.json"
        )

    runs = _load_runs(args.runs_root)
    if not args.logprob_file.exists():
        raise FileNotFoundError(
            f"completion_logprob file missing: {args.logprob_file}. "
            f"Run scripts/issue518_syco_logprob_backfill.py (or its "
            f"per-arm sibling) to produce it before building the substrate."
        )
    logprob_map = _load_completion_logprob(args.logprob_file)

    cells: list[dict] = []
    for source, run in runs.items():
        for cell in run.get("per_cell", []):
            bystander = cell["bystander"]
            key = (source, bystander)
            if key not in logprob_map:
                raise RuntimeError(
                    f"completion_logprob missing for ({source}, {bystander}); "
                    f"available keys = {sorted(logprob_map.keys())[:10]}... "
                    f"#518 v4 must-fix 1: every (source, bystander) cell of "
                    f"every arm being scored MUST have a completion_logprob."
                )
            # #518 v4 round-2 must-fix 2: gate the hash-derived stub on
            # --smoke. Production substrates must contain real coarse-zoo
            # values loaded from the #480 per-cell predictor sweep, NOT
            # deterministic-but-fake hashes of (source, bystander) -- the
            # FAIL-CLOSED 24-field check only verifies presence, not validity,
            # so without this gate a non-smoke run silently emits fake
            # predictors that the aggregator then correlates against the real
            # Δ panel. Round 1 left the call unconditional and reviewers FAILed
            # both Claude + Codex on it. Until the real #480 coarse-zoo loader
            # is wired (plan §13 implementer-decides; out of scope for the
            # cross-behavior aggregator implementer round), production MUST
            # raise.
            if args.smoke:
                coarse = _stub_coarse_zoo(source, bystander)
            else:
                raise NotImplementedError(
                    "Real #480 coarse-zoo loader not yet wired; pass --smoke "
                    "to use deterministic stubs, or wire up the real loader "
                    "before launching the production substrate build. "
                    "Cell affected: "
                    f"(source={source!r}, bystander={bystander!r}). "
                    "#518 v4 round-2 must-fix 2: production must not silently "
                    "emit hash-derived coarse-zoo stubs that the FAIL-CLOSED "
                    "24-field check is structurally blind to."
                )
            cell_payload = {
                "source": source,
                "bystander": bystander,
                "delta": float(cell["delta"]),
                "trained_rate": float(cell.get("trained_rate", float("nan"))),
                "bystander_base_rate": float(cell.get("base_rate", float("nan"))),
                "completion_logprob": logprob_map[key],
                **coarse,
            }
            # Sanity: every required 24-field slot is present.
            missing = [f for f in PREDICTOR_FIELDS_24 if f not in cell_payload]
            if missing:
                raise RuntimeError(
                    f"Built cell missing required field(s): {missing} for ({source}, {bystander})."
                )
            cells.append(cell_payload)

    out_payload = {
        "schema_version": 1,
        "_doc": (
            "Per-(source, bystander) predictor cell substrate. 24 fields = "
            "23 from #480 + completion_logprob added by #518 v4 must-fix 1. "
            "Consumed by scripts/issue509_scoring.py --arm <refusal|em> + "
            "the cross-behavior aggregator's headline `min(|ρ|)`."
        ),
        "arm": args.arm,
        "smoke": args.smoke,
        "fields": list(PREDICTOR_FIELDS_24),
        "n_cells": len(cells),
        "cells": cells,
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2))
    print(f"WROTE {args.out} (n_cells={len(cells)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
