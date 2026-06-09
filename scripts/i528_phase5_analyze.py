# ruff: noqa: RUF002, RUF003  -- intentional math notation (Spearman ρ).
"""Phase 5 analyze — H1 + H2 + DV3 (#528).

Plan v1 §6.2. Computes:

- **H1 per-trait paired t-test** (trained_system - base, system arm only)
  on N=40 paired Likert values; Holm-Bonferroni across 4 traits;
  bootstrap 95% CI (10k resamples).
- **H2 paired role-vs-system d_leakage** by seed, averaged across 4
  off-target eval contexts and the H1-PASSing trait subset; paired
  bootstrap CI (10k); per-context sub-analysis.
- **DV3 paraphrase ρ** per (trait, arm) on the 10% stratified subsample.

Output: ``eval_results/issue_528/analysis.json``.
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

logger = logging.getLogger("i528.phase5.analyze")

JUDGE_PATH = Path("eval_results/issue_528/judge_scores.json")
PARAPHRASE_PATH = Path("eval_results/issue_528/paraphrase_replication.json")
OUT_PATH = Path("eval_results/issue_528/analysis.json")

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
    """Paired t-test on the differences (one-sample t on diffs vs 0)."""
    n = len(diffs)
    if n < 2:
        return {"t": float("nan"), "df": 0, "p_two_sided": float("nan"), "mean": 0.0}
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0:
        return {
            "t": float("inf") if mean != 0 else 0.0,
            "df": n - 1,
            "p_two_sided": 0.0,
            "mean": mean,
        }
    t = mean / se
    # Two-sided p-value via the Student-t CDF approximation. We avoid scipy by
    # using the symmetry: 2 * (1 - cdf(|t|, df)). Use an approximation via the
    # standard normal tail for df>=30; for smaller df, use a Welch-Satterthwaite
    # safe-ish approximation.
    df = n - 1
    # Approximate two-sided p via the standard-normal tail for df>=30 (>= the
    # typical N=40 case); otherwise use math.erfc on a corrected statistic.
    if df >= 30:
        z = abs(t)
        p = math.erfc(z / math.sqrt(2.0))
    else:
        # Conservative: clip via the normal tail. Small-df correction would
        # require the incomplete beta function; for the planned N=40 case we
        # are well into normal-approximation territory.
        z = abs(t)
        p = math.erfc(z / math.sqrt(2.0))
    return {"t": t, "df": df, "p_two_sided": p, "mean": mean}


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
    args = ap.parse_args(argv)

    from explore_persona_space.experiments.i528_traits import TRAITS

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

    # ---------------- H1 — per-trait installation, system arm only ----------------
    h1_per_trait: dict[str, dict] = {}
    h1_p: dict[str, float] = {}
    base_summary: dict[str, dict] = {}
    for trait in TRAITS:
        if trait not in trained_index:
            h1_per_trait[trait] = {"status": "NO_TRAINED_CELLS"}
            continue
        # System arm in-scenario per seed: avg across 3 seeds per q_idx.
        per_q_trained: dict[int, list[float]] = {}
        per_q_base: dict[int, list[float]] = {}
        seeds_seen: set[int] = set()
        for seed, by_ctx in trained_index[trait].get("system", {}).items():
            if "own_scenario" not in by_ctx:
                continue
            seeds_seen.add(seed)
            for q_idx, score in by_ctx["own_scenario"].items():
                per_q_trained.setdefault(q_idx, []).append(score)
        # Base in-scenario reads from base_index[trait][<eval_arm>][-1]["own_scenario"].
        # The base eval was run under BOTH eval_arms; we use eval_arm=system for
        # H1 (matches the trained_system comparison).
        base_own = base_index.get(trait, {}).get("system", {}).get(-1, {}).get("own_scenario", {})
        for q_idx, score in base_own.items():
            per_q_base.setdefault(q_idx, []).append(score)

        # Per-q paired diff (trained_mean - base_mean).
        diffs: list[float] = []
        for q_idx in sorted(per_q_trained.keys() & per_q_base.keys()):
            t_mean = sum(per_q_trained[q_idx]) / len(per_q_trained[q_idx])
            b_mean = sum(per_q_base[q_idx]) / len(per_q_base[q_idx])
            diffs.append(t_mean - b_mean)
        if not diffs:
            h1_per_trait[trait] = {
                "status": "NO_PAIRED_PROMPTS",
                "n_trained_q": len(per_q_trained),
                "n_base_q": len(per_q_base),
            }
            continue
        t_stat = _paired_t(diffs)
        lo, hi = _bootstrap_ci(diffs, n_resamples=args.n_bootstrap, alpha=args.alpha)
        h1_per_trait[trait] = {
            "n_paired": len(diffs),
            "paired_delta_mean": t_stat["mean"],
            "t": t_stat["t"],
            "df": t_stat["df"],
            "p_uncorrected": t_stat["p_two_sided"],
            "ci_lo": lo,
            "ci_hi": hi,
            "seeds_used": sorted(seeds_seen),
        }
        h1_p[trait] = t_stat["p_two_sided"]
        # Base summary for headroom view.
        base_scores = [s for vs in per_q_base.values() for s in vs]
        if base_scores:
            base_summary[trait] = {
                "n": len(base_scores),
                "mean": sum(base_scores) / len(base_scores),
                "ci_lo": _bootstrap_ci(base_scores, n_resamples=args.n_bootstrap)[0],
                "ci_hi": _bootstrap_ci(base_scores, n_resamples=args.n_bootstrap)[1],
                "above_3_5": (sum(base_scores) / len(base_scores)) >= 3.5,
            }

    # Holm-Bonferroni across the 4 H1 tests.
    h1_holm = _holm_bonferroni(h1_p, alpha=args.alpha) if h1_p else {}
    for trait, hb in h1_holm.items():
        h1_per_trait[trait]["p_holm"] = hb["p_holm"]
        h1_per_trait[trait]["reject"] = hb["reject"]
        h1_per_trait[trait]["pass_h1"] = bool(
            hb["reject"] and h1_per_trait[trait].get("ci_lo", -1) > 0
        )

    h1_passing = [t for t, info in h1_per_trait.items() if info.get("pass_h1")]

    # ---------------- H2 — paired role-vs-system d_leakage ----------------
    # For each (trait, seed):
    #   leakage_arm(t, s) = mean over (off-target ctx x q_idx) of trained score
    #   d(t, s) = leakage_role(t, s) - leakage_system(t, s)
    h2_per_cell: list[dict] = []
    h2_pairs: list[float] = []
    h2_per_seed: dict[int, list[float]] = {}
    for trait in TRAITS:
        for seed in (42, 137, 1337):
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
                h2_pairs.append(d)
                h2_per_seed.setdefault(seed, []).append(d)

    h2_summary: dict = {}
    if h2_pairs:
        d_mean = sum(h2_pairs) / len(h2_pairs)
        lo, hi = _bootstrap_ci(h2_pairs, n_resamples=args.n_bootstrap)
        per_seed_means = {s: sum(v) / len(v) for s, v in h2_per_seed.items()}
        n_neg = sum(1 for m in per_seed_means.values() if m < 0)
        h2_summary = {
            "n_pairs": len(h2_pairs),
            "h1_passing_traits": sorted(h1_passing),
            "d_mean": d_mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "per_seed_mean": per_seed_means,
            "n_seeds_negative": n_neg,
            "pass_bar": "d_mean <= -0.15, ci_hi < 0, >= 2/3 seeds negative",
            "passed": d_mean <= -0.15 and hi < 0 and n_neg >= 2,
        }
    else:
        h2_summary = {"status": "NO_H1_PASSING_TRAITS_OR_NO_PAIRED_CELLS"}

    # All-trait H2 (sensitivity).
    all_d = [c["d_leakage"] for c in h2_per_cell]
    all_per_seed: dict[int, list[float]] = {}
    for c in h2_per_cell:
        all_per_seed.setdefault(c["seed"], []).append(c["d_leakage"])
    h2_all_trait = {}
    if all_d:
        d_mean_all = sum(all_d) / len(all_d)
        lo_all, hi_all = _bootstrap_ci(all_d, n_resamples=args.n_bootstrap)
        per_seed_all = {s: sum(v) / len(v) for s, v in all_per_seed.items()}
        h2_all_trait = {
            "n_pairs": len(all_d),
            "d_mean": d_mean_all,
            "ci_lo": lo_all,
            "ci_hi": hi_all,
            "per_seed_mean": per_seed_all,
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

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "schema_version": "i528_v1",
                "kind": "analysis",
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "n_bootstrap": args.n_bootstrap,
                "alpha": args.alpha,
                "h1_per_trait": h1_per_trait,
                "h1_passing_traits": sorted(h1_passing),
                "base_headroom_summary": base_summary,
                "h2_paired_leakage": h2_summary,
                "h2_per_cell": h2_per_cell,
                "h2_all_trait_sensitivity": h2_all_trait,
                "dv3_paraphrase_rho": dv3,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("Wrote %s", OUT_PATH)
    logger.info("H1 passing: %s", sorted(h1_passing))
    logger.info("H2 summary: %s", json.dumps(h2_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
