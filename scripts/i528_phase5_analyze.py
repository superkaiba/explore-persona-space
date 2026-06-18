# ruff: noqa: RUF002, RUF003  -- intentional math notation (Spearman ρ).
"""Phase 5 analyze — H1 + H2 + DV3 (#528).

Plan v1 §6.2 + §11 + §15 #20/#21. Computes:

- **H1 per-trait paired t-test** (trained_system - base, system arm only)
  on N=40 paired Likert values; Student-t (via ``scipy.stats.ttest_1samp``,
  per plan §11/§15 #20); Holm-Bonferroni across 4 traits; bootstrap 95%
  CI (10k resamples). **Saturation-gated** (plan §6.2, #517 precedent):
  a trait whose base CI's LOWER bound > 3.5 is "saturated base" — H1 is
  untestable for it (no headroom) and it is EXCLUDED from the H2 subset.
- **H2 paired role-vs-system d_leakage** by SEED (N=3 paired-bootstrap
  per plan §6.2 + §15 #21; #498 precedent). For each (trait, seed),
  compute role - system leakage averaged across 4 off-target eval
  contexts; then aggregate to per-seed means over the H1-PASSing trait
  subset; the bootstrap unit is one of the 3 seeds (NOT 12 trait*seed
  cells), so headline d_mean, CI, and pass/fail all use the same
  per-seed unit.
- **DV3 paraphrase ρ** per (trait, arm) on the 10% stratified subsample.

Output: ``eval_results/<ISSUE_SLUG>/analysis.json`` for
``--saturation-gate per_encoding``; the pooled audit run writes to the
distinct ``analysis_pooled_gate.json`` (matching the parent #528 artifact
layout) unless ``--out-name`` overrides.

**H2 pass-bar parameters** (``--h2-bar-d-mean``, ``--h2-min-seeds-neg``):
defaults reproduce #528's registered bar (d_mean <= -0.15, >= 2/3 seeds
negative); #556 passes ``--h2-bar-d-mean -0.10 --h2-min-seeds-neg 8`` per
its plan §6 registration.

**Saturation gate modes** (``--saturation-gate``):

- ``pooled`` (default, original behavior): each trait's saturation flag
  reads the base own_scenario distribution under ``eval_arm=system``
  only; H1 is tested for trained-system only; a trait is H1-passing
  iff its system-arm cell passes.
- ``per_encoding`` (new): for each (trait, arm) cell the saturation
  flag reads the base own_scenario distribution under THAT arm; H1 is
  tested in both system AND role arms in parallel; a trait is
  H1-passing iff EITHER arm's cell passes. The full per-cell table is
  emitted under ``h1_per_cell``. Holm-Bonferroni runs across the
  non-saturated cells (up to 2 traits × 2 arms = 8 tests). The pooled
  trait-level summary under ``h1_per_trait`` is retained for backward
  compatibility (it mirrors the system-arm cell). This addresses the
  observation that base own_scenario distributions differ between
  system and role encodings (validating-role base CI [3.08, 3.67]
  overlaps the 3.5 saturation line, so role is not saturated even
  though system is).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import random
import subprocess
from pathlib import Path

from explore_persona_space.experiments.i528_data import ISSUE_SLUG

logger = logging.getLogger("i528.phase5.analyze")

JUDGE_PATH = Path(f"eval_results/{ISSUE_SLUG}/judge_scores.json")
PARAPHRASE_PATH = Path(f"eval_results/{ISSUE_SLUG}/paraphrase_replication.json")
RESULTS_DIR = Path(f"eval_results/{ISSUE_SLUG}")

OFF_TARGET_CONTEXTS = ("sibling_1", "sibling_2", "sibling_3", "default_assistant")


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _bootstrap_ci(
    values: list[float], *, n_resamples: int = 10000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_resamples * alpha / 2)]
    hi = means[int(n_resamples * (1 - alpha / 2))]
    return (lo, hi)


def _paired_t(diffs: list[float]) -> dict:
    """Paired t-test on the differences (one-sample Student-t on diffs vs 0).

    Plan §11 + §15 #20: paired t-test (Student-t, two-sided). Uses
    ``scipy.stats.ttest_1samp`` so the small-df behavior (df < 30 under
    descope) is correct, not normal-approximated.
    """
    n = len(diffs)
    if n < 2:
        return {"t": float("nan"), "df": 0, "p_two_sided": float("nan"), "mean": 0.0}
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    df = n - 1
    if se == 0:
        return {
            "t": float("inf") if mean != 0 else 0.0,
            "df": df,
            "p_two_sided": 0.0 if mean != 0 else 1.0,
            "mean": mean,
        }

    from scipy import stats  # type: ignore[import-not-found]

    res = stats.ttest_1samp(diffs, popmean=0.0)
    t_stat = float(res.statistic)
    p = float(res.pvalue)
    return {"t": t_stat, "df": df, "p_two_sided": p, "mean": mean}


def _holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni correction. Returns ``{key: {p_holm, reject}}``."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    prev_p_holm = 0.0
    for i, (key, p) in enumerate(items):
        # Holm-corrected p = min(1, max(prev_corrected, (m - i) * p))
        p_holm = min(1.0, max(prev_p_holm, (m - i) * p))
        prev_p_holm = p_holm
        out[key] = {"p_uncorrected": p, "p_holm": p_holm, "reject": p_holm < alpha}
    return out


def _spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Rank-correlation. Ties handled via mean-rank."""
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    n = len(xs)

    def _ranks(values: list[float]) -> list[float]:
        indexed = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-indexed ranks
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg
            i = j + 1
        return ranks

    rx = _ranks(xs)
    ry = _ranks(ys)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    var_x = math.sqrt(sum((rx[i] - mean_x) ** 2 for i in range(n)))
    var_y = math.sqrt(sum((ry[i] - mean_y) ** 2 for i in range(n)))
    if var_x == 0 or var_y == 0:
        return float("nan")
    return num / (var_x * var_y)


def _group(rows: list[dict], *, kind: str) -> dict:
    """Group rows by various keys for downstream stats."""
    out: dict = {}
    for r in rows:
        if r.get("kind") != kind:
            continue
        trait = r["trait"]
        arm = r.get("arm")
        seed = r.get("seed", -1)
        ctx = r["eval_context"]
        q_idx = r["q_idx"]
        score = float(r["score"])
        out.setdefault(trait, {}).setdefault(arm, {}).setdefault(seed, {}).setdefault(ctx, {})[
            q_idx
        ] = score
    return out


def main(argv: list[str] | None = None) -> int:  # noqa: C901 — phase dispatcher
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument(
        "--h2-bar-d-mean",
        type=float,
        default=-0.15,
        help="H2 PASS bar on the across-seed mean d_leakage (default -0.15, "
        "#528's registered bar; #556 registers -0.10).",
    )
    ap.add_argument(
        "--h2-min-seeds-neg",
        type=int,
        default=2,
        help="H2 PASS bar on the count of negative per-seed means (default "
        "2 of 3, #528's registered bar; #556 registers 8 of 10).",
    )
    ap.add_argument(
        "--out-name",
        default=None,
        help="Output filename under eval_results/<slug>/. Default derives "
        "from --saturation-gate: per_encoding -> analysis.json (the primary "
        "registered gate for #556), pooled -> analysis_pooled_gate.json "
        "(the archived audit run, matching #528's artifact layout).",
    )
    ap.add_argument(
        "--saturation-gate",
        choices=("pooled", "per_encoding"),
        default="pooled",
        help=(
            "How to compute the H1 saturation gate. 'pooled' (default, original "
            "behavior): one gate per trait, base own_scenario read under "
            "eval_arm=system. 'per_encoding': one gate per (trait, arm) cell, "
            "base own_scenario read under that arm — splits H1 into 2 arms in "
            "parallel and emits h1_per_cell."
        ),
    )
    args = ap.parse_args(argv)

    from explore_persona_space.experiments.i528_traits import SEEDS, TRAITS

    if args.out_name is not None:
        out_name = args.out_name
    elif args.saturation_gate == "pooled":
        out_name = "analysis_pooled_gate.json"
    else:
        out_name = "analysis.json"
    out_path = RESULTS_DIR / out_name

    if not JUDGE_PATH.exists():
        raise SystemExit(f"{JUDGE_PATH} not found — run Phase 4 judge first.")
    judge_payload = json.loads(JUDGE_PATH.read_text())
    rows = judge_payload["rows"]

    # Index base + trained scores.
    # base: {trait: {arm_unused: {-1 (seed): {ctx: {q_idx: score}}}}}
    # Base has no LoRA, so per-trait we keep eval_arm in the arm slot and
    # seed = -1.
    base_index = _group(rows, kind="base")
    trained_index = _group(rows, kind="trained")

    # ---------------- H1 — per-trait installation ----------------
    # Two saturation-gate modes (Blocker 2 follow-up, 2026-06-10):
    #   pooled (default, legacy): one gate per trait, base own_scenario read
    #     under eval_arm=system only; H1 tests trained-system vs base-system.
    #   per_encoding: one gate per (trait, arm) cell, base own_scenario read
    #     under THAT arm; H1 tests trained-<arm> vs base-<arm> for both
    #     arms in parallel; Holm spans the non-saturated cells.
    def _h1_one_cell(
        trait: str,
        trained_arm: str,
        base_arm: str,
    ) -> dict | None:
        """Compute the H1 paired test + saturation gate for one (trait, arm) cell.

        ``trained_arm`` selects which trained arm's own_scenario is the
        numerator; ``base_arm`` selects which base eval_arm's own_scenario
        is the comparator. The two are decoupled so the legacy pooled mode
        (trained_arm='system', base_arm='system') and the new per_encoding
        mode (trained_arm == base_arm, varied over both) share one
        codepath. Returns ``None`` if the trait has no trained data; the
        caller writes an explicit NO_TRAINED_CELLS status.
        """
        if trait not in trained_index:
            return None
        per_q_trained: dict[int, list[float]] = {}
        per_q_base: dict[int, list[float]] = {}
        seeds_seen: set[int] = set()
        for seed, by_ctx in trained_index[trait].get(trained_arm, {}).items():
            if "own_scenario" not in by_ctx:
                continue
            seeds_seen.add(seed)
            for q_idx, score in by_ctx["own_scenario"].items():
                per_q_trained.setdefault(q_idx, []).append(score)
        base_own = base_index.get(trait, {}).get(base_arm, {}).get(-1, {}).get("own_scenario", {})
        for q_idx, score in base_own.items():
            per_q_base.setdefault(q_idx, []).append(score)
        diffs: list[float] = []
        for q_idx in sorted(per_q_trained.keys() & per_q_base.keys()):
            t_mean = sum(per_q_trained[q_idx]) / len(per_q_trained[q_idx])
            b_mean = sum(per_q_base[q_idx]) / len(per_q_base[q_idx])
            diffs.append(t_mean - b_mean)
        if not diffs:
            return {
                "status": "NO_PAIRED_PROMPTS",
                "n_trained_q": len(per_q_trained),
                "n_base_q": len(per_q_base),
                "trained_arm": trained_arm,
                "base_arm": base_arm,
            }
        t_stat = _paired_t(diffs)
        lo, hi = _bootstrap_ci(diffs, n_resamples=args.n_bootstrap, alpha=args.alpha)
        base_scores = [s for vs in per_q_base.values() for s in vs]
        base_ci_lo = float("nan")
        base_ci_hi = float("nan")
        base_mean = float("nan")
        base_saturated_ci = False
        if base_scores:
            base_ci_lo, base_ci_hi = _bootstrap_ci(base_scores, n_resamples=args.n_bootstrap)
            base_mean = sum(base_scores) / len(base_scores)
            base_saturated_ci = base_ci_lo > 3.5
        cell = {
            "n_paired": len(diffs),
            "paired_delta_mean": t_stat["mean"],
            "t": t_stat["t"],
            "df": t_stat["df"],
            "p_uncorrected": t_stat["p_two_sided"],
            "ci_lo": lo,
            "ci_hi": hi,
            "seeds_used": sorted(seeds_seen),
            "trained_arm": trained_arm,
            "base_arm": base_arm,
            "base_ci_lo": base_ci_lo,
            "base_ci_hi": base_ci_hi,
            "base_mean": base_mean,
            "base_saturated_ci": base_saturated_ci,
            "headroom": not base_saturated_ci,
        }
        if base_saturated_ci:
            cell["h1_untestable"] = "saturated_base"
        return cell

    h1_per_trait: dict[str, dict] = {}
    h1_per_cell: dict[str, dict[str, dict]] = {}  # {trait: {arm: cell_dict}}
    base_summary: dict[str, dict] = {}
    base_summary_per_cell: dict[str, dict[str, dict]] = {}  # {trait: {arm: summary}}

    if args.saturation_gate == "pooled":
        # Legacy path: one gate per trait, both gate + H1 read eval_arm=system.
        gate_cells: list[tuple[str, str, str]] = [(t, "system", "system") for t in TRAITS]
    else:
        # per_encoding: gate + H1 read the same arm; both arms in parallel.
        gate_cells = [(t, arm, arm) for t in TRAITS for arm in ("system", "role")]

    h1_p: dict[str, float] = {}  # key = f"{trait}__{trained_arm}" (or trait under pooled)
    for trait, trained_arm, base_arm in gate_cells:
        cell = _h1_one_cell(trait, trained_arm, base_arm)
        if cell is None:
            if trait not in h1_per_trait:
                h1_per_trait[trait] = {"status": "NO_TRAINED_CELLS"}
            h1_per_cell.setdefault(trait, {})[trained_arm] = {"status": "NO_TRAINED_CELLS"}
            continue
        h1_per_cell.setdefault(trait, {})[trained_arm] = cell
        # Backward-compat: under pooled mode, mirror the system cell into
        # h1_per_trait[trait] (the legacy schema). Under per_encoding,
        # h1_per_trait[trait] mirrors the system cell too so existing
        # consumers (e.g. the plot script) keep working; the role cell is
        # accessible via h1_per_cell.
        if trained_arm == "system":
            h1_per_trait[trait] = dict(cell)
        # Base summary table — always per-cell so consumers see both arms
        # when in per_encoding mode; under pooled, the only key is system.
        if not math.isnan(cell.get("base_mean", float("nan"))):
            base_summary_per_cell.setdefault(trait, {})[base_arm] = {
                "n": cell["n_paired"],  # n_q for the H1 test; base n == 40 typically
                "mean": cell["base_mean"],
                "ci_lo": cell["base_ci_lo"],
                "ci_hi": cell["base_ci_hi"],
                "above_3_5_mean": cell["base_mean"] >= 3.5,
                "base_saturated_ci": cell["base_saturated_ci"],
                "headroom": cell["headroom"],
            }
            # Pooled-mode legacy view: base_summary keyed by trait only.
            if args.saturation_gate == "pooled" and base_arm == "system":
                base_summary[trait] = base_summary_per_cell[trait][base_arm]
        if cell["base_saturated_ci"]:
            continue
        key = f"{trait}__{trained_arm}" if args.saturation_gate == "per_encoding" else trait
        h1_p[key] = cell["p_uncorrected"]

    # Holm-Bonferroni across the H1 tests of NON-saturated cells only.
    h1_holm = _holm_bonferroni(h1_p, alpha=args.alpha) if h1_p else {}
    for key, hb in h1_holm.items():
        if args.saturation_gate == "per_encoding":
            trait, trained_arm = key.split("__", 1)
            info = h1_per_cell[trait][trained_arm]
        else:
            trait = key
            info = h1_per_trait[trait]
        info["p_holm"] = hb["p_holm"]
        info["reject"] = hb["reject"]
        pass_h1 = bool(
            info.get("headroom")
            and hb["reject"]
            and info.get("ci_lo", -1) > 0
            and info.get("paired_delta_mean", 0) > 0
        )
        info["pass_h1"] = pass_h1
        # Backward-compat: pooled-mode + system-arm Holm result also lands
        # on the legacy h1_per_trait[trait] row.
        if (
            args.saturation_gate == "per_encoding"
            and trained_arm == "system"
            and trait in h1_per_trait
        ):
            h1_per_trait[trait]["p_holm"] = hb["p_holm"]
            h1_per_trait[trait]["reject"] = hb["reject"]
            h1_per_trait[trait]["pass_h1"] = pass_h1

    # Ensure every cell + trait row has pass_h1=False explicitly so
    # downstream consumers can rely on the key.
    for _trait, _by_arm in h1_per_cell.items():
        for _arm, _info in _by_arm.items():
            _info.setdefault("pass_h1", False)
    for _trait, _info in h1_per_trait.items():
        _info.setdefault("pass_h1", False)

    # H1-passing trait set used by H2 subset:
    #   pooled: trait passes iff its system-arm cell passes.
    #   per_encoding: trait passes iff EITHER arm's cell passes (we still
    #     subset H2 by trait — H2 itself is a paired role-vs-system delta).
    if args.saturation_gate == "per_encoding":
        h1_passing = sorted(
            {
                t
                for t, by_arm in h1_per_cell.items()
                if any(info.get("pass_h1") for info in by_arm.values())
            }
        )
    else:
        h1_passing = [t for t, info in h1_per_trait.items() if info.get("pass_h1")]

    # ---------------- H2 — paired role-vs-system d_leakage by SEED ----------
    # Plan §6.2 + §11 + §15 #21 + #498 precedent: the bootstrap UNIT is one
    # of the 3 seeds, NOT one of the 12 (trait, seed) cells (Blocker 1).
    #
    # Two-step aggregation:
    #   1. Per-cell d(t, s) = mean(role_off_target) - mean(system_off_target)
    #      averaged across the 4 off-target eval contexts (and Q_test prompts).
    #   2. Per-seed mean d(s) = mean over H1-PASSing traits of d(t, s).
    # The headline d_mean, bootstrap CI, and pass/fail all read off the
    # 3-element per-seed-mean LIST — that is the paired-bootstrap-by-seed
    # contract. Per-trait*seed cells are kept for traceability only.
    h2_per_cell: list[dict] = []
    # raw_pairs[seed] = list of d values across H1-passing traits at that seed
    raw_pairs_by_seed: dict[int, list[float]] = {}
    for trait in TRAITS:
        for seed in SEEDS:
            sys_scores: list[float] = []
            role_scores: list[float] = []
            for ctx in OFF_TARGET_CONTEXTS:
                sys_by_q = trained_index.get(trait, {}).get("system", {}).get(seed, {}).get(ctx, {})
                role_by_q = trained_index.get(trait, {}).get("role", {}).get(seed, {}).get(ctx, {})
                for q in sys_by_q:
                    sys_scores.append(sys_by_q[q])
                for q in role_by_q:
                    role_scores.append(role_by_q[q])
            if not sys_scores or not role_scores:
                continue
            sys_mean = sum(sys_scores) / len(sys_scores)
            role_mean = sum(role_scores) / len(role_scores)
            d = role_mean - sys_mean
            h2_per_cell.append(
                {
                    "trait": trait,
                    "seed": seed,
                    "leakage_role": role_mean,
                    "leakage_system": sys_mean,
                    "d_leakage": d,
                    "h1_passing": trait in h1_passing,
                }
            )
            if trait in h1_passing:
                raw_pairs_by_seed.setdefault(seed, []).append(d)

    # Per-seed means over the H1-PASSing trait subset — these are the 3
    # bootstrap-unit observations.
    h2_passing_per_seed_mean: dict[int, float] = {
        s: sum(v) / len(v) for s, v in raw_pairs_by_seed.items() if v
    }
    h2_passing_seed_means: list[float] = [
        h2_passing_per_seed_mean[s] for s in sorted(h2_passing_per_seed_mean.keys())
    ]

    h2_summary: dict = {}
    if h2_passing_seed_means:
        d_mean = sum(h2_passing_seed_means) / len(h2_passing_seed_means)
        # Paired bootstrap over the 3 per-seed means (NOT over flat (trait,
        # seed) cells). With N=3 this gives a wide CI by design — that is
        # the correct uncertainty for N=3 paired bootstrap.
        lo, hi = _bootstrap_ci(h2_passing_seed_means, n_resamples=args.n_bootstrap)
        n_neg = sum(1 for m in h2_passing_seed_means if m < 0)
        h2_summary = {
            "n_seeds": len(h2_passing_seed_means),
            "bootstrap_unit": "per_seed_mean_over_h1_passing_traits",
            "h1_passing_traits": sorted(h1_passing),
            "d_mean": d_mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "per_seed_mean": h2_passing_per_seed_mean,
            "n_seeds_negative": n_neg,
            "pass_bar": (
                f"d_mean <= {args.h2_bar_d_mean}, ci_hi < 0, "
                f">= {args.h2_min_seeds_neg}/{len(h2_passing_seed_means)} seeds negative"
            ),
            "passed": (d_mean <= args.h2_bar_d_mean and hi < 0 and n_neg >= args.h2_min_seeds_neg),
        }
        # Coverage guard (#556 concern `analysis-h2-coverage-assert`): the
        # per-cell loop above `continue`-drops empty (trait, seed) cells, so a
        # seed whose eval rows are missing silently SHRINKS the bootstrap
        # denominator — and the `passed` verdict would then be computed over
        # fewer seeds than registered. Assert the realized per-seed coverage
        # equals the ACTIVE `SEEDS` config exactly (len + identity; SEEDS is
        # env-driven via I528_SEEDS, never hardcoded here). On shortfall mark
        # the summary INCOMPLETE_COVERAGE and force passed=False — never a
        # silent verdict on a shrunken denominator.
        expected_seeds = sorted(SEEDS)
        realized_seeds = sorted(h2_passing_per_seed_mean)
        if realized_seeds != expected_seeds:
            h2_summary["status"] = "INCOMPLETE_COVERAGE"
            h2_summary["seeds_expected"] = expected_seeds
            h2_summary["seeds_realized"] = realized_seeds
            h2_summary["passed"] = False
            logger.warning(
                "H2 coverage INCOMPLETE: realized seeds %s != configured SEEDS %s — "
                "forcing passed=False (a verdict on a shrunken denominator is invalid).",
                realized_seeds,
                expected_seeds,
            )
    else:
        h2_summary = {"status": "NO_H1_PASSING_TRAITS_OR_NO_PAIRED_CELLS"}

    # All-trait H2 (sensitivity) — also bootstrapped by SEED.
    all_per_seed_raw: dict[int, list[float]] = {}
    for c in h2_per_cell:
        all_per_seed_raw.setdefault(c["seed"], []).append(c["d_leakage"])
    all_passing_per_seed_mean = {s: sum(v) / len(v) for s, v in all_per_seed_raw.items() if v}
    all_seed_means = [
        all_passing_per_seed_mean[s] for s in sorted(all_passing_per_seed_mean.keys())
    ]
    h2_all_trait: dict = {}
    if all_seed_means:
        d_mean_all = sum(all_seed_means) / len(all_seed_means)
        lo_all, hi_all = _bootstrap_ci(all_seed_means, n_resamples=args.n_bootstrap)
        h2_all_trait = {
            "n_seeds": len(all_seed_means),
            "bootstrap_unit": "per_seed_mean_over_all_traits",
            "d_mean": d_mean_all,
            "ci_lo": lo_all,
            "ci_hi": hi_all,
            "per_seed_mean": all_passing_per_seed_mean,
        }

    # ---------------- DV3 paraphrase ρ ----------------
    dv3: dict[str, dict] = {}
    if PARAPHRASE_PATH.exists():
        para = json.loads(PARAPHRASE_PATH.read_text()).get("rows", [])
        by_pair: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for r in para:
            key = (r["trait"], r["arm"])
            by_pair.setdefault(key, []).append((r.get("primary_score", 0.0), r["score"]))
        for (trait, arm), pairs in by_pair.items():
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            rho = _spearman_rho(xs, ys)
            dv3.setdefault(trait, {})[arm] = {"n": len(pairs), "rho": rho, "ge_0_7": rho >= 0.7}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema_version": "i528_v1",
                "kind": "analysis",
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "n_bootstrap": args.n_bootstrap,
                "alpha": args.alpha,
                "saturation_gate": args.saturation_gate,
                "h2_bar_d_mean": args.h2_bar_d_mean,
                "h2_min_seeds_neg": args.h2_min_seeds_neg,
                "seeds": list(SEEDS),
                "h1_per_trait": h1_per_trait,
                "h1_per_cell": h1_per_cell,
                "h1_passing_traits": sorted(h1_passing),
                "base_headroom_summary": base_summary,
                "base_headroom_summary_per_cell": base_summary_per_cell,
                "h2_paired_leakage": h2_summary,
                "h2_per_cell": h2_per_cell,
                "h2_all_trait_sensitivity": h2_all_trait,
                "dv3_paraphrase_rho": dv3,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("Wrote %s", out_path)
    logger.info("H1 passing: %s", sorted(h1_passing))
    logger.info("H2 summary: %s", json.dumps(h2_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
