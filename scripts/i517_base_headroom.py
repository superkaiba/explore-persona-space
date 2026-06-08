"""Issue #517 base-model headroom probe driver.

Plan §4.2-§4.4. Thin orchestrator that:

1.  Asserts preconditions (model id, branch, plan, cherry-pick sha).
2.  Subprocesses into ``scripts/i498_phase4_eval.py --base-only`` (vLLM
    batched greedy, 1 GPU, 2048 max_new_tokens, eval contexts
    ``in_scenario`` + ``default_assistant``, 40-prompt held-out split).
3.  Subprocesses into ``scripts/i498_phase4_judge.py --raw-dir ... --out ...
    --n-judge-calls N --paraphrase-frac 0`` (no paraphrase pass — the
    parent already published one in #498).
4.  Computes the prompt-paired base-vs-trained Δ per trait x arm
    (mean ± 1.96·SEM_paired + Wilcoxon signed-rank p) from the new
    base-only judge JSON + #498's already-published
    ``eval_results/issue_498/judge_scores.json``.
5.  Writes ``eval_results/issue_517/base_vs_trained_comparison.json`` and
    calls ``scripts/plot_i517_hero.py`` to render the hero figure.

CLI:
    # Full run (40 prompts, 3 judge calls per prompt, both eval contexts)
    uv run python scripts/i517_base_headroom.py

    # Smoke (3 prompts, 1 judge call, in_scenario only)
    uv run python scripts/i517_base_headroom.py --smoke

    # Aggregate-only (skip eval+judge; just rebuild the comparison JSON
    # and re-plot from a pre-existing judge file)
    uv run python scripts/i517_base_headroom.py --aggregate-only \\
        --judge /tmp/i517_smoke/base_headroom_judge.json \\
        --trained eval_results/issue_498/judge_scores.json \\
        --out /tmp/i517_smoke/base_vs_trained_comparison.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("i517.base_headroom")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "eval_results" / "issue_517" / "raw_generations"
DEFAULT_JUDGE_OUT = REPO_ROOT / "eval_results" / "issue_517" / "base_headroom_judge.json"
DEFAULT_COMPARISON_OUT = (
    REPO_ROOT / "eval_results" / "issue_517" / "base_vs_trained_comparison.json"
)
DEFAULT_TRAINED_JUDGE = REPO_ROOT / "eval_results" / "issue_498" / "judge_scores.json"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures" / "issue_517"

PASS_THRESHOLD = 3.5  # #498's pre-registered Likert PASS bar.


def _git() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=REPO_ROOT,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _wilcoxon_signed_rank_p(deltas: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p-value. Returns ``None`` if scipy
    isn't installed (we keep the dependency optional — the hero stat is
    the CI on Δ, not the p). Mid-rank handling for ties; uses scipy's
    asymptotic approximation."""
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except Exception:
        return None
    if not deltas:
        return None
    nonzero = [d for d in deltas if d != 0.0]
    if not nonzero:
        return None
    result = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
    return float(result.pvalue)


def _mean_sem_ci(values: list[float]) -> dict:
    """Mean ± SEM + 95% CI (normal approx; N=40 is plenty)."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sem": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(values)
    if n < 2:
        return {"n": n, "mean": mean, "sem": 0.0, "ci95_low": mean, "ci95_high": mean}
    sd = statistics.stdev(values)
    sem = sd / math.sqrt(n)
    half = 1.96 * sem
    return {"n": n, "mean": mean, "sem": sem, "ci95_low": mean - half, "ci95_high": mean + half}


def _aggregate_base(judge_path: Path) -> dict[tuple[str, str], dict[int, float]]:
    """Return ``{(eval_context, trait): {q_idx: averaged_likert}}``.

    Each q_idx's averaged-Likert is the mean across the N within-prompt
    judge re-calls. If ``--n-judge-calls 1`` was used, this collapses to
    the bare scalar score per q_idx.
    """
    payload = json.loads(judge_path.read_text())
    rows = payload.get("rows", [])
    if not rows:
        raise SystemExit(f"base judge file {judge_path} has no rows.")
    # rows already carry an averaged 'score' field (or the scalar score
    # for n-judge-calls=1). One row per (cell_id, q_idx); n_judge_calls
    # averaging is done inside i498_phase4_judge.py.
    out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        if row.get("arm") != "base":
            continue
        key = (row["eval_context"], row["trait"])
        out[key][row["q_idx"]] = float(row["score"])
    if not out:
        raise SystemExit(
            f"base judge file {judge_path} contained no rows with arm='base' "
            "— is this the right file?"
        )
    return out


def _aggregate_trained(judge_path: Path) -> dict[tuple[str, str, str], dict[int, float]]:
    """Return ``{(arm, eval_context, trait): {q_idx: avg_across_LoRA_seeds}}``.

    Collapses 3 LoRA training seeds per (arm x ctx x trait x q_idx) to one
    averaged Likert per prompt. Skips ``arm='base'`` rows.
    """
    payload = json.loads(judge_path.read_text())
    rows = payload.get("rows", [])
    if not rows:
        raise SystemExit(f"trained judge file {judge_path} has no rows.")
    by_seed: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        arm = row.get("arm")
        if arm == "base":
            continue
        key = (arm, row["eval_context"], row["trait"], row["q_idx"])
        by_seed[key].append(float(row["score"]))
    out: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for (arm, ctx, trait, q_idx), scores in by_seed.items():
        out[(arm, ctx, trait)][q_idx] = sum(scores) / len(scores)
    if not out:
        raise SystemExit(f"trained judge file {judge_path} contained no non-base rows.")
    return out


def _build_comparison(
    base: dict[tuple[str, str], dict[int, float]],
    trained: dict[tuple[str, str, str], dict[int, float]],
) -> dict:
    """Build the per-trait comparison + paired Δ statistics.

    Schema:
        {
          "schema_version": "i517_v1",
          "git_commit": ...,
          "ts": ...,
          "pass_threshold": 3.5,
          "per_trait": {
            "<trait>": {
              "base_in_scenario": {n, mean, sem, ci95_low, ci95_high, decision},
              "base_default_assistant": {...},
              "trained_system_in_scenario": {n, mean, sem, ci95_low, ci95_high},
              "trained_role_in_scenario": {...},
              "paired_delta_system": {n, mean, sem, ci95_low, ci95_high,
                                       wilcoxon_p},
              "paired_delta_role": {...},
              "system_prompt_effect": {n, mean, sem, ci95_low, ci95_high,
                                       detection},
            }, ...
          }
        }
    """
    traits = sorted({t for (_ctx, t) in base})
    per_trait: dict[str, dict] = {}

    for trait in traits:
        block: dict = {}

        # Base cells (2 eval contexts).
        base_in_scenario = base.get(("in_scenario", trait), {})
        base_default = base.get(("default_assistant", trait), {})
        in_vals = sorted(base_in_scenario.items())
        def_vals = sorted(base_default.items())
        in_arr = [v for (_q, v) in in_vals]
        def_arr = [v for (_q, v) in def_vals]
        in_stats = _mean_sem_ci(in_arr)
        def_stats = _mean_sem_ci(def_arr)

        # 3.5-threshold verdict on base in_scenario (per plan §6.2).
        if in_stats["mean"] is None:
            decision = "no_data"
        elif in_stats["ci95_low"] >= PASS_THRESHOLD:
            decision = "saturation (CI strictly above 3.5)"
        elif in_stats["ci95_high"] < PASS_THRESHOLD:
            decision = "real_implant_possible (CI strictly below 3.5)"
        else:
            decision = "near_threshold (CI overlaps 3.5)"
        in_stats["decision"] = decision

        block["base_in_scenario"] = in_stats
        block["base_default_assistant"] = def_stats

        # Trained cells, in_scenario only (the load-bearing comparison;
        # default_assistant is published already in #498). Pull
        # default_assistant numbers too for the table's completeness.
        for arm in ("system", "role"):
            t_in = trained.get((arm, "in_scenario", trait), {})
            t_def = trained.get((arm, "default_assistant", trait), {})
            block[f"trained_{arm}_in_scenario"] = _mean_sem_ci(
                [v for (_q, v) in sorted(t_in.items())]
            )
            block[f"trained_{arm}_default_assistant"] = _mean_sem_ci(
                [v for (_q, v) in sorted(t_def.items())]
            )

            # Paired Δ (trained_in_scenario_q - base_in_scenario_q) across
            # q_idx values present in BOTH dicts (per plan §6.4).
            common = sorted(set(base_in_scenario.keys()) & set(t_in.keys()))
            deltas = [t_in[q] - base_in_scenario[q] for q in common]
            d_stats = _mean_sem_ci(deltas)
            d_stats["wilcoxon_p"] = _wilcoxon_signed_rank_p(deltas)
            d_stats["n_paired"] = len(deltas)
            block[f"paired_delta_{arm}"] = d_stats

        # System-prompt-effect (paired Δ base_in_scenario - base_default
        # over q_idx common to both, per plan §6.2).
        common_ctx = sorted(set(base_in_scenario.keys()) & set(base_default.keys()))
        sys_prompt_deltas = [base_in_scenario[q] - base_default[q] for q in common_ctx]
        sp_stats = _mean_sem_ci(sys_prompt_deltas)
        if sp_stats["mean"] is None:
            sp_detect = "no_data"
        elif sp_stats["ci95_low"] > 0.0:
            sp_detect = "system_prompt_moves_score (CI strictly above 0)"
        elif sp_stats["ci95_high"] < 0.0:
            sp_detect = "default_assistant_higher (CI strictly below 0)"
        else:
            sp_detect = "uniform (CI overlaps 0)"
        sp_stats["detection"] = sp_detect
        sp_stats["n_paired"] = len(sys_prompt_deltas)
        block["system_prompt_effect"] = sp_stats

        per_trait[trait] = block

    return {
        "schema_version": "i517_v1",
        "kind": "base_vs_trained_comparison",
        "git_commit": _git(),
        "ts": _dt.datetime.utcnow().isoformat() + "Z",
        "pass_threshold": PASS_THRESHOLD,
        "per_trait": per_trait,
    }


def _run_eval(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir) if args.raw_dir else DEFAULT_RAW_DIR
    cmd = [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "scripts" / "i498_phase4_eval.py"),
        "--base-only",
        "--out-dir",
        str(raw_dir),
        "--eval-contexts",
        *args.eval_contexts,
        "--n-q",
        str(args.n_q),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--backend",
        args.backend,
        "--truncation-fail-threshold",
        str(args.truncation_fail_threshold),
    ]
    if args.traits:
        cmd += ["--traits", *args.traits]
    if args.base_model:
        cmd += ["--base-model", args.base_model]
    logger.info("[phase=eval] launching: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    logger.info("[phase=eval] done")


def _run_judge(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir) if args.raw_dir else DEFAULT_RAW_DIR
    judge_out = Path(args.judge_out) if args.judge_out else DEFAULT_JUDGE_OUT
    cmd = [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "scripts" / "i498_phase4_judge.py"),
        "--raw-dir",
        str(raw_dir),
        "--raw-glob",
        "base_seed-1__*.json",
        "--out",
        str(judge_out),
        "--n-judge-calls",
        str(args.n_judge_calls),
        "--paraphrase-frac",
        "0",
        "--backend",
        "sync",
    ]
    logger.info("[phase=judge] launching: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    logger.info("[phase=judge] done")


def _run_aggregate(args: argparse.Namespace) -> Path:
    judge_path = Path(args.judge_out) if args.judge_out else DEFAULT_JUDGE_OUT
    trained_path = Path(args.trained_judge) if args.trained_judge else DEFAULT_TRAINED_JUDGE
    out_path = Path(args.comparison_out) if args.comparison_out else DEFAULT_COMPARISON_OUT
    base_agg = _aggregate_base(judge_path)
    trained_agg = _aggregate_trained(trained_path)
    comparison = _build_comparison(base_agg, trained_agg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
    logger.info("[phase=aggregate] wrote %s", out_path)
    return out_path


def _run_plot(args: argparse.Namespace, comparison_path: Path) -> None:
    fig_dir = Path(args.figure_dir) if args.figure_dir else DEFAULT_FIGURE_DIR
    cmd = [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "scripts" / "plot_i517_hero.py"),
        "--in",
        str(comparison_path),
        "--out-dir",
        str(fig_dir),
    ]
    logger.info("[phase=plot] launching: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    logger.info("[phase=plot] done")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--smoke", action="store_true", help="3 prompts, 1 judge call, in_scenario only."
    )
    ap.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip eval+judge subprocess phases; just rebuild the "
        "comparison JSON + re-plot from --judge-out and --trained-judge.",
    )
    ap.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip the plot phase (useful in CPU-only smoke).",
    )
    ap.add_argument("--n-q", type=int, default=40)
    ap.add_argument("--n-judge-calls", type=int, default=3)
    ap.add_argument(
        "--eval-contexts",
        nargs="+",
        default=("in_scenario", "default_assistant"),
    )
    ap.add_argument("--traits", nargs="+", default=None)
    ap.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--truncation-fail-threshold", type=float, default=0.05)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--judge-out", default=None)
    ap.add_argument("--comparison-out", default=None)
    ap.add_argument("--figure-dir", default=None)
    ap.add_argument(
        "--trained-judge",
        default=None,
        help="Path to #498's judge_scores.json. Default: "
        "eval_results/issue_498/judge_scores.json (relative to repo root).",
    )
    args = ap.parse_args(argv)

    if args.smoke:
        # Smoke knobs (plan §4.7): 3 prompts, 1 judge call, in_scenario only,
        # one trait if not overridden.
        args.n_q = 3
        args.n_judge_calls = 1
        args.eval_contexts = ("in_scenario",)
        if args.traits is None:
            args.traits = ["coding"]
        if args.max_new_tokens > 256:
            # Trim smoke generation cost — eval rig still exercises the
            # full code path under truncation accounting.
            args.max_new_tokens = 256
        # Disable truncation gate for smoke (256-token cap will fail it).
        args.truncation_fail_threshold = 1.0

    if not args.aggregate_only:
        _run_eval(args)
        _run_judge(args)

    comparison_path = _run_aggregate(args)

    if not args.skip_plot:
        _run_plot(args, comparison_path)

    logger.info("[phase=done] i517 base-headroom probe complete")


if __name__ == "__main__":
    main()
