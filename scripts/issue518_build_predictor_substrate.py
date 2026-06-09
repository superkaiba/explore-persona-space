#!/usr/bin/env python3
# Greek + special characters (×, →, —, α, Δ, ρ) appear in this file's prose
# for research notation.
# ruff: noqa: RUF001, RUF003
"""#518 v4 predictor_comparison.json substrate builder.

Assembles the per-arm 24-field cell schema from:
  - the per-source ``run_result.json`` emitted by
    ``run_experiment_518_<arm>.py`` (delta + trained_rate + bystander_base_rate);
  - the per-(source, bystander) ``completion_logprob`` cell from
    ``scripts/issue518_syco_logprob_backfill.py`` (run against the per-arm
    teach-row substrate, not just the syco backfill);
  - 17 base-model-derived coarse-zoo fields (cosine layers + JS/KL + base
    rate + response-length proxies) loaded by the per-arm coarse-zoo
    loader in ``explore_persona_space.experiments.issue_518.coarse_zoo_loader``.

Schema matches ``eval_results/issue_480/_inputs/predictor_comparison.json``
(23 fields from #480 + the new ``completion_logprob`` column = 24 fields):
  source, bystander, delta, cosine_l20_baseline, cosine_response_headline,
  trained_rate_<arm>, bystander_base_rate, source_base_rate,
  base_rate_diff_neg_abs, source_resp_len_mean, bystander_resp_len_mean,
  resp_len_diff_abs, cosine_response_l{7,14,21,27},
  JS_{sym,from_source,from_bystander}_nats, M_js,
  KL_{src_to_bys,bys_to_src,sym}_nats, completion_logprob.

Round-5 must-fix #5 — production substrate dispatch:
  - **syco arm**: load the existing #480 predictor_comparison.json from
    disk; it already has the 17 coarse-zoo fields.
  - **refusal / em arms**: compute the coarse-zoo from a per-arm cosine
    sweep + JS/KL sweep that the pod-side production driver runs BEFORE
    invoking this substrate builder. Paths are passed via
    ``--cosine-sweep`` / ``--jskl-sweep``; the substrate builder is a
    pure aggregator (the heavy compute is upstream).
  - **smoke mode**: deterministic stub coarse-zoo values, no disk
    dependency, so the downstream aggregator can validate end-to-end.

CLI:
  # Smoke (no disk deps beyond the per-arm smoke run_result + logprob).
  uv run python scripts/issue518_build_predictor_substrate.py --arm em --smoke

  # syco production (round-12 fix: syco arm is INHERITED from #411 via the
  # frozen analyze_summary -- there is NO eval_results/issue_509/syco_arm/runs/
  # directory, only the 138-cell snapshot. --runs-root MUST NOT be passed for
  # arm=syco; --syco-analyze-summary supplies the per-cell delta/trained/base
  # tuple instead):
  uv run python scripts/issue518_build_predictor_substrate.py \\
      --arm syco \\
      --syco-predictor-comparison eval_results/issue_480/_inputs/predictor_comparison.json \\
      --syco-analyze-summary eval_results/issue_480/_inputs/syco_411_analyze_summary.json \\
      --logprob-file eval_results/issue_509/syco_arm/bystander_logprob/logprob_results.json \\
      --out eval_results/issue_518/syco/_inputs/predictor_comparison.json

  # refusal / em production:
  uv run python scripts/issue518_build_predictor_substrate.py \\
      --arm refusal \\
      --slab-root eval_results/issue_518/refusal/slab \\
      --cosine-sweep eval_results/issue_518/refusal/predictors/cosine.json \\
      --jskl-sweep   eval_results/issue_518/refusal/predictors/jskl.json \\
      --runs-root    eval_results/issue_518/refusal/runs \\
      --logprob-file eval_results/issue_518/refusal/bystander_logprob/logprob_results.json \\
      --out          eval_results/issue_518/refusal/_inputs/predictor_comparison.json
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


def _load_syco_cells_from_analyze_summary(
    analyze_summary_path: Path,
) -> list[dict]:
    """Read #411 syco analyze_summary -> per-(source, bystander) cell records.

    Round-12 fix: the syco arm is INHERITED from #411 via the frozen leakage
    snapshot at ``eval_results/issue_480/_inputs/syco_411_analyze_summary.json``
    -- #509 never produced per-source ``runs/<source>_seed42/run_result.json``
    files. This loader reads the 138-cell (6 sources x 23 bystanders) frozen
    matrix DIRECTLY, bypassing ``_load_runs`` which assumes a per-source runs
    directory layout that does not exist for syco.

    Returns a list of dicts of the same shape as the ``per_cell`` entries
    consumed by the main loop:

      {"source": <src>, "bystander": <bys>, "delta": <Δ>,
       "trained_rate": <p_t>, "base_rate": <p_b>}

    Self-pairs (source == bystander) are filtered out -- consistent with
    ``issue509_scoring.py::_load_syco_target``'s off-diagonal-only contract
    (line 770) which is the same #411 contract the analyze_summary produced.

    Raises FileNotFoundError if the path doesn't exist; raises RuntimeError if
    the schema doesn't match the expected ``per_source[src]`` keys.
    """
    if not analyze_summary_path.exists():
        raise FileNotFoundError(
            f"syco analyze_summary missing: {analyze_summary_path}. The syco "
            f"arm is INHERITED from #411 -- pass --syco-analyze-summary "
            f"pointing at eval_results/issue_480/_inputs/"
            f"syco_411_analyze_summary.json (the frozen 138-cell snapshot)."
        )
    payload = json.loads(analyze_summary_path.read_text())
    if "per_source" not in payload:
        raise RuntimeError(
            f"syco analyze_summary at {analyze_summary_path} has no "
            f"'per_source' key (got keys={list(payload)[:10]}); expected the "
            f"#411 snapshot schema with per_source[src][per_panel_delta]."
        )
    cells: list[dict] = []
    for source, src_data in payload["per_source"].items():
        per_panel_delta = src_data.get("per_panel_delta", {})
        per_panel_trained = src_data.get("per_panel_trained_rate", {})
        per_panel_base = src_data.get("per_panel_base_rate", {})
        if not isinstance(per_panel_delta, dict) or not per_panel_delta:
            raise RuntimeError(
                f"syco analyze_summary per_source[{source!r}] missing or "
                f"empty per_panel_delta; cannot build substrate cells."
            )
        for bystander, delta in per_panel_delta.items():
            if bystander == source:
                continue  # off-diagonal only -- matches #411 panel contract
            cells.append(
                {
                    "source": source,
                    "bystander": bystander,
                    "delta": float(delta),
                    "trained_rate": float(per_panel_trained.get(bystander, float("nan"))),
                    "base_rate": float(per_panel_base.get(bystander, float("nan"))),
                }
            )
    if not cells:
        raise RuntimeError(
            f"syco analyze_summary at {analyze_summary_path} produced 0 "
            f"cells -- snapshot is empty or has no off-diagonal pairs."
        )
    return cells


def _resolve_runs(
    arm: str,
    runs_root: Path | None,
    syco_analyze_summary: Path | None,
) -> dict[str, dict]:
    """Round-12 helper: resolve per-source -> {per_cell:[...]} mapping per arm.

    syco arm: bypass ``_load_runs`` and synthesize the per-source -> per-cell
    list directly from the #411 138-cell ``analyze_summary`` snapshot (#509
    never produced an ``eval_results/issue_509/syco_arm/runs/`` directory; the
    syco arm is INHERITED from #411 via the frozen snapshot). The synthesized
    shape matches ``_load_runs``'s output verbatim so the main loop's
    ``for source, run in runs.items(): for cell in run["per_cell"]`` iteration
    is uniform across all three arms.

    refusal / em arms: ``_load_runs(runs_root)`` unchanged -- per-source
    ``runs/<src>_seed42/run_result.json`` files exist and carry per_cell lists.
    """
    if arm == "syco":
        if syco_analyze_summary is None:
            raise ValueError("syco arm requires --syco-analyze-summary; #509 has no runs/ dir.")
        syco_cells = _load_syco_cells_from_analyze_summary(syco_analyze_summary)
        by_source: dict[str, list[dict]] = {}
        for c in syco_cells:
            by_source.setdefault(c["source"], []).append(c)
        return {
            src: {"source": src, "per_cell": per_cell}
            for src, per_cell in sorted(by_source.items())
        }
    if runs_root is None:
        raise ValueError(f"arm {arm!r} requires --runs-root.")
    return _load_runs(runs_root)


def main() -> int:
    """Entrypoint. See module docstring."""
    p = argparse.ArgumentParser(
        description="#518 v4 predictor_comparison.json substrate builder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--arm",
        choices=("syco", "refusal", "em"),
        required=True,
        help="Which #518 behavior arm to build the substrate for.",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help=(
            "Directory containing per-source run_result.json files. Default: "
            "eval_results/issue_518/<arm>/runs (refusal/em); "
            "eval_results/issue_509/syco_arm/runs (syco)."
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
        "--syco-predictor-comparison",
        type=Path,
        default=None,
        help=(
            "syco arm only: pre-existing predictor_comparison.json with 17 "
            "coarse-zoo fields (typically "
            "eval_results/issue_480/_inputs/predictor_comparison.json). "
            "Required for --arm syco non-smoke."
        ),
    )
    p.add_argument(
        "--syco-analyze-summary",
        type=Path,
        default=None,
        help=(
            "syco arm only: 138-cell frozen leakage snapshot from #411 (typically "
            "eval_results/issue_480/_inputs/syco_411_analyze_summary.json). "
            "Replaces --runs-root for syco -- #509 never produced per-source "
            "runs/<src>_seed42/run_result.json directories; the syco arm is "
            "INHERITED from #411 via this snapshot. The per-(source, bystander) "
            "delta + trained_rate + base_rate tuple is extracted directly from "
            "per_source[<src>][per_panel_delta|per_panel_trained_rate|"
            "per_panel_base_rate]. Required for --arm syco non-smoke; ignored "
            "for refusal / em arms."
        ),
    )
    p.add_argument(
        "--slab-root",
        type=Path,
        default=None,
        help=(
            "refusal/em arm only: per-arm eval slab root. Default: "
            "eval_results/issue_518/<arm>/slab. Used to compute response-"
            "length and base-rate proxies from the panel JSONs."
        ),
    )
    p.add_argument(
        "--cosine-sweep",
        type=Path,
        default=None,
        help=(
            "refusal/em arm only: pre-computed cosine sweep JSON path "
            "(output of scripts/issue404_predictor_cossim.py against the "
            "per-arm (source, bystander) panel). Required for non-smoke."
        ),
    )
    p.add_argument(
        "--jskl-sweep",
        type=Path,
        default=None,
        help=(
            "refusal/em arm only: pre-computed JS/KL sweep JSON path "
            "(output of scripts/issue458_predictor_jsdiv.py against the "
            "per-arm panel). Required for non-smoke."
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

    # Round-12 fix: syco arm has NO per-source runs/ directory (inherited from
    # #411 via syco_analyze_summary). Only resolve runs_root default for
    # refusal / em arms; the syco branch below bypasses _load_runs entirely.
    if args.runs_root is None and args.arm != "syco":
        args.runs_root = REPO / "eval_results" / "issue_518" / args.arm / "runs"
    if args.arm == "syco" and args.syco_analyze_summary is None:
        # Production default; smoke mode can still pass --runs-root or override.
        args.syco_analyze_summary = (
            REPO / "eval_results" / "issue_480" / "_inputs" / "syco_411_analyze_summary.json"
        )
    if args.logprob_file is None:
        if args.arm == "syco":
            args.logprob_file = (
                REPO
                / "eval_results"
                / "issue_509"
                / "syco_arm"
                / "bystander_logprob"
                / "logprob_results.json"
            )
        else:
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
    if args.arm in ("refusal", "em") and args.slab_root is None:
        args.slab_root = REPO / "eval_results" / "issue_518" / args.arm / "slab"

    # Round-12 fix: syco arm bypasses _load_runs (no per-source runs/ exists
    # under eval_results/issue_509/syco_arm/ -- #509 inherited the leakage
    # panel from #411 via the frozen analyze_summary snapshot). The
    # ``_resolve_runs`` helper unifies per-arm dispatch and returns the same
    # shape across syco / refusal / em so the main loop stays uniform.
    runs = _resolve_runs(args.arm, args.runs_root, args.syco_analyze_summary)
    if not args.logprob_file.exists():
        raise FileNotFoundError(
            f"completion_logprob file missing: {args.logprob_file}. "
            f"Run scripts/issue518_syco_logprob_backfill.py (or its "
            f"per-arm sibling) to produce it before building the substrate."
        )
    logprob_map = _load_completion_logprob(args.logprob_file)

    # Round-5 must-fix #5: production coarse-zoo loader. Smoke uses the
    # deterministic stub; production dispatches to the per-arm loader.
    coarse_map: dict[tuple[str, str], dict[str, float]] = {}
    if not args.smoke:
        from explore_persona_space.experiments.issue_518.coarse_zoo_loader import (
            load_coarse_zoo_for_arm,
        )

        coarse_map = load_coarse_zoo_for_arm(
            arm=args.arm,
            syco_predictor_comparison_path=args.syco_predictor_comparison,
            slab_root=args.slab_root,
            layer_cosine_path=args.cosine_sweep,
            js_kl_path=args.jskl_sweep,
        )

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
            if args.smoke:
                coarse = _stub_coarse_zoo(source, bystander)
            else:
                if key not in coarse_map:
                    raise RuntimeError(
                        f"Production coarse-zoo loader returned no entry for "
                        f"({source!r}, {bystander!r}) on arm {args.arm!r}; "
                        f"the upstream sweep is missing this cell. Re-run "
                        f"the cosine + JS/KL predictor sweeps against the "
                        f"per-arm panel before building the substrate."
                    )
                coarse = coarse_map[key]
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
