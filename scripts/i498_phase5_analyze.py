"""Phase 5 — analysis (issue #498).

Plan v1.2 §4.1 Phase 6 + §6.2. Aggregate judge scores per (arm, trait,
eval_context, seed); compute the headline ``d_seed = L[system] - L[role]``
paired by seed, both ways (paired (trait x eval_context) dynamic-range mask =
primary; full unmasked sensitivity check); 10000-sample paired bootstrap CI;
per-trait + per-eval-context breakdowns; paraphrase Spearman rho.

Writes eval_results/issue_498/analysis.json.

CLI:
    uv run python scripts/i498_phase5_analyze.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("i498.phase5.analyze")

JUDGE_PATH = Path("eval_results/issue_498/judge_scores.json")
PARAPHRASE_PATH = Path("eval_results/issue_498/paraphrase_replication.json")
ANALYSIS_PATH = Path("eval_results/issue_498/analysis.json")


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _paired_bootstrap_ci(
    deltas: list[float], n_resamples: int = 10000, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Paired bootstrap over seed-level deltas. Returns (mean, lo, hi)."""
    import random as _r

    if not deltas:
        return (0.0, 0.0, 0.0)
    rng = _r.Random(42)
    means = []
    n = len(deltas)
    for _ in range(n_resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(alpha / 2 * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples)
    return (sum(deltas) / n, means[lo_idx], means[hi_idx])


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation."""
    if not xs or len(xs) != len(ys):
        return 0.0

    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = rank(xs)
    ry = rank(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = ((sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5) or 1.0
    return num / den


def main(argv: list[str] | None = None) -> None:  # noqa: C901 — analysis dispatcher
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sd-threshold", type=float, default=0.3)
    ap.add_argument("--n-resamples", type=int, default=10000)
    args = ap.parse_args(argv)

    if not JUDGE_PATH.exists():
        raise SystemExit(f"{JUDGE_PATH} missing — run scripts/i498_phase4_judge.py first.")
    judge_payload = json.loads(JUDGE_PATH.read_text())
    rows = judge_payload.get("rows", [])
    logger.info("Loaded %d judge rows", len(rows))

    # Bucket by (arm, trait, eval_context, seed).
    by_cell: dict[tuple, list[int]] = defaultdict(list)
    for r in rows:
        s = r.get("score")
        if s is None:
            continue
        key = (r["arm"], r["trait"], r["eval_context"], r["seed"])
        by_cell[key].append(int(s))

    # Per-cell mean + sd.
    per_cell: dict[str, dict] = {}
    for key, vals in by_cell.items():
        arm, trait, ec, seed = key
        per_cell.setdefault(arm, {}).setdefault(trait, {}).setdefault(ec, {})[seed] = {
            "n": len(vals),
            "mean": (sum(vals) / len(vals)) if vals else 0.0,
            "sd": (statistics.stdev(vals) if len(vals) > 1 else 0.0),
        }

    # Dynamic-range gate (PAIRED): per (trait x eval_context), if EITHER arm's
    # sd <= threshold across 120 obs (40 q x 3 seeds), DROP THE UNIT from BOTH
    # arms.
    sd_by_arm_unit: dict[tuple, float] = {}
    for arm in ("system", "role"):
        for trait, by_ec in per_cell.get(arm, {}).items():
            for ec, by_seed in by_ec.items():
                # Pool all observations across seeds for the unit.
                pool: list[int] = []
                for seed, _v in by_seed.items():
                    pool.extend(by_cell[(arm, trait, ec, seed)])
                sd_by_arm_unit[(arm, trait, ec)] = statistics.stdev(pool) if len(pool) > 1 else 0.0

    surviving_units: list[tuple] = []
    dropped_units: list[tuple] = []
    all_units: set[tuple] = set()
    for (_arm, trait, ec), _sd in sd_by_arm_unit.items():
        all_units.add((trait, ec))
    for trait, ec in all_units:
        sd_sys = sd_by_arm_unit.get(("system", trait, ec), 0.0)
        sd_role = sd_by_arm_unit.get(("role", trait, ec), 0.0)
        if min(sd_sys, sd_role) <= args.sd_threshold:
            dropped_units.append((trait, ec))
        else:
            surviving_units.append((trait, ec))

    # L_arm[seed] = mean over (trait x eval_context ∈ symmetric_leakage_cells).
    SYMMETRIC_CELLS = ("cross_scenario", "default_assistant")

    def compute_L(arm: str, seed: int, restrict_to_paired_mask: bool) -> float | None:
        vals: list[float] = []
        for trait, by_ec in per_cell.get(arm, {}).items():
            for ec in SYMMETRIC_CELLS:
                if restrict_to_paired_mask and (trait, ec) not in surviving_units:
                    continue
                seed_dict = by_ec.get(ec, {})
                cell = seed_dict.get(seed)
                if cell is None:
                    continue
                vals.append(cell["mean"])
        if not vals:
            return None
        return sum(vals) / len(vals)

    # d_seed for each seed under both maskings.
    seeds = sorted({seed for (_, _, _, seed) in by_cell})
    d_seed_masked: list[float] = []
    d_seed_unmasked: list[float] = []
    for seed in seeds:
        L_sys_m = compute_L("system", seed, True)
        L_role_m = compute_L("role", seed, True)
        L_sys_u = compute_L("system", seed, False)
        L_role_u = compute_L("role", seed, False)
        if L_sys_m is not None and L_role_m is not None:
            d_seed_masked.append(L_sys_m - L_role_m)
        if L_sys_u is not None and L_role_u is not None:
            d_seed_unmasked.append(L_sys_u - L_role_u)

    ci_masked = (
        _paired_bootstrap_ci(d_seed_masked, args.n_resamples) if d_seed_masked else (0.0, 0.0, 0.0)
    )
    ci_unmasked = (
        _paired_bootstrap_ci(d_seed_unmasked, args.n_resamples)
        if d_seed_unmasked
        else (0.0, 0.0, 0.0)
    )

    # Per-trait breakdown (H3.a).
    per_trait_d: dict[str, dict] = {}
    traits_seen = {t for (_arm, t, _ec, _seed) in by_cell}
    for trait in traits_seen:
        deltas = []
        for seed in seeds:
            sys_means = [
                per_cell.get("system", {}).get(trait, {}).get(ec, {}).get(seed, {}).get("mean")
                for ec in SYMMETRIC_CELLS
            ]
            role_means = [
                per_cell.get("role", {}).get(trait, {}).get(ec, {}).get(seed, {}).get("mean")
                for ec in SYMMETRIC_CELLS
            ]
            sys_means = [m for m in sys_means if m is not None]
            role_means = [m for m in role_means if m is not None]
            if not sys_means or not role_means:
                continue
            deltas.append(sum(sys_means) / len(sys_means) - sum(role_means) / len(role_means))
        if deltas:
            ci = _paired_bootstrap_ci(deltas, args.n_resamples)
            per_trait_d[trait] = {
                "n_seeds": len(deltas),
                "deltas": deltas,
                "mean": ci[0],
                "ci_lo": ci[1],
                "ci_hi": ci[2],
            }

    # Per-eval-context breakdown (H3.b).
    per_ec_d: dict[str, dict] = {}
    for ec in SYMMETRIC_CELLS:
        deltas = []
        for seed in seeds:
            sys_means = []
            role_means = []
            for trait in traits_seen:
                m_sys = (
                    per_cell.get("system", {}).get(trait, {}).get(ec, {}).get(seed, {}).get("mean")
                )
                m_role = (
                    per_cell.get("role", {}).get(trait, {}).get(ec, {}).get(seed, {}).get("mean")
                )
                if m_sys is not None:
                    sys_means.append(m_sys)
                if m_role is not None:
                    role_means.append(m_role)
            if not sys_means or not role_means:
                continue
            deltas.append(sum(sys_means) / len(sys_means) - sum(role_means) / len(role_means))
        if deltas:
            ci = _paired_bootstrap_ci(deltas, args.n_resamples)
            per_ec_d[ec] = {
                "n_seeds": len(deltas),
                "deltas": deltas,
                "mean": ci[0],
                "ci_lo": ci[1],
                "ci_hi": ci[2],
            }

    # Paraphrase Spearman (per arm x trait).
    para_summary: dict[str, float] = {}
    if PARAPHRASE_PATH.exists():
        para_payload = json.loads(PARAPHRASE_PATH.read_text())
        para_rows = para_payload.get("rows", [])
        # Build (arm, trait) -> ([primary, paraphrase]) pairs.
        by_at: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
        for r in para_rows:
            key = (r.get("arm"), r.get("trait"))
            primary = r.get("primary_score")
            paraphrase = r.get("score")
            if primary is None or paraphrase is None:
                continue
            xs, ys = by_at.setdefault(key, ([], []))
            xs.append(float(primary))
            ys.append(float(paraphrase))
        for (arm, trait), (xs, ys) in by_at.items():
            para_summary[f"{arm}__{trait}__spearman"] = _spearman(xs, ys)
            para_summary[f"{arm}__{trait}__n"] = len(xs)

    # H1 / H2 PASS checks.
    h1_all_pass = True
    h1_min_cell_mean = float("inf")
    for arm in ("system", "role"):
        for _trait, by_ec in per_cell.get(arm, {}).items():
            for seed_dict in by_ec.get("in_scenario", {}).values():
                m = seed_dict.get("mean", 0.0)
                if m < h1_min_cell_mean:
                    h1_min_cell_mean = m
                if m < 3.5:
                    h1_all_pass = False

    mean_d_m, lo_m, _hi_m = ci_masked
    h2_pass = (
        mean_d_m >= 0.4
        and lo_m > 0
        and all(d > 0 for d in d_seed_masked)
        and len(d_seed_masked) >= 1
    )

    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text(
        json.dumps(
            {
                "schema_version": "i498_v1",
                "kind": "analysis",
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "per_cell": per_cell,
                "sd_by_arm_unit": {f"{a}__{t}__{e}": v for (a, t, e), v in sd_by_arm_unit.items()},
                "surviving_units": [f"{t}__{e}" for (t, e) in surviving_units],
                "dropped_units": [f"{t}__{e}" for (t, e) in dropped_units],
                "headline": {
                    "d_seed_masked": d_seed_masked,
                    "d_seed_unmasked": d_seed_unmasked,
                    "ci_masked": {"mean": ci_masked[0], "lo": ci_masked[1], "hi": ci_masked[2]},
                    "ci_unmasked": {
                        "mean": ci_unmasked[0],
                        "lo": ci_unmasked[1],
                        "hi": ci_unmasked[2],
                    },
                    "divergence_masked_vs_unmasked": ci_masked[0] - ci_unmasked[0],
                },
                "per_trait_d": per_trait_d,
                "per_eval_context_d": per_ec_d,
                "paraphrase_spearman": para_summary,
                "h1_pass": h1_all_pass,
                "h1_min_cell_mean": h1_min_cell_mean,
                "h2_pass": h2_pass,
                "n_seeds": len(seeds),
                "seeds": seeds,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info(
        "Analysis: H1=%s (min in-scenario mean=%.2f), H2=%s "
        "(d_masked mean=%.2f CI=[%.2f, %.2f]) -> %s",
        h1_all_pass,
        h1_min_cell_mean,
        h2_pass,
        ci_masked[0],
        ci_masked[1],
        ci_masked[2],
        ANALYSIS_PATH,
    )


if __name__ == "__main__":
    main()
