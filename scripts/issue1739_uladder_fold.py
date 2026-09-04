#!/usr/bin/env python3
"""Fold issue 1739 U-ladder seed outputs into registered statistical verdicts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()
BEHAVIORS = ("evil", "sycophancy", "hallucination")
U_SIZES = (250, 500, 1000, 2000, 5000, 10000, 18793)
CONFIGS = ("generic_only", "union_scaled")
SETTING_GROUPS = ("in_dist", "generic", "ood")
SCHEMA_VERSION = 1
DEFAULT_INPUT = Path("eval_results/issue_1739/uladder")
DEFAULT_OUTPUT = Path("eval_results/issue_1739/uladder_fold")


def _atomic_json(path: Path, payload: object) -> None:
    from explore_persona_space.atomic_io import atomic_replace

    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))


def _atomic_text(path: Path, text: str) -> None:
    from explore_persona_space.atomic_io import atomic_replace

    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        tmp.write_text(text)


def _t_interval(values: list[float], confidence: float) -> dict:
    import numpy as np
    from scipy.stats import t

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError(f"t interval needs at least two finite values, got {values}")
    mean = float(x.mean())
    sem = float(x.std(ddof=1) / math.sqrt(len(x)))
    critical = float(t.ppf((1.0 + confidence) / 2.0, df=len(x) - 1))
    half = critical * sem
    return {
        "n": len(values),
        "mean": mean,
        "sem": sem,
        "confidence": confidence,
        "df": len(values) - 1,
        "interval": [mean - half, mean + half],
        "values": [float(v) for v in values],
    }


def equivalence_tost(values: list[float], *, delta: float = 0.02, alpha: float = 0.05) -> dict:
    """Two one-sided t tests expressed by the equivalent 1-2alpha CI rule."""
    ci = _t_interval(values, 1.0 - 2.0 * alpha)
    lower, upper = ci["interval"]
    equivalent = bool(lower > -delta and upper < delta)
    return {
        **ci,
        "method": "TOST paired seed-level t test",
        "alpha": alpha,
        "delta": delta,
        "bounds": [-delta, delta],
        "equivalent": equivalent,
    }


def seed_spearman_trend(u_sizes: list[int], d_values: list[float]) -> float:
    import numpy as np
    from scipy.stats import rankdata

    if len(u_sizes) != len(d_values) or len(u_sizes) < 2:
        raise ValueError("trend needs aligned U and D vectors with at least two rungs")
    x = rankdata(np.log(np.asarray(u_sizes, dtype=np.float64)), method="average")
    y = rankdata(np.asarray(d_values, dtype=np.float64), method="average")
    xc, yc = x - x.mean(), y - y.mean()
    den = float(np.sqrt((xc**2).sum() * (yc**2).sum()))
    return float((xc * yc).sum() / den) if den > 0 else 0.0


def classify_verdict(
    trend: dict,
    endpoint: dict,
    judged: dict,
    *,
    delta: float,
) -> tuple[str, list[str]]:
    trend_positive = trend["interval"][0] > 0
    endpoint_equivalent = bool(endpoint["equivalent"])
    judged_equivalent = bool(judged["equivalent"])
    judged_material_positive = judged["interval"][0] >= delta
    facts = [
        f"trend lower 95% bound={trend['interval'][0]:.6f}",
        f"endpoint equivalence={endpoint_equivalent}",
        f"judged equivalence={judged_equivalent}",
        f"judged lower 90% bound={judged['interval'][0]:.6f}",
    ]
    if trend_positive and not endpoint_equivalent and judged_equivalent:
        return "SUPPORTED", facts
    if endpoint_equivalent or judged_material_positive:
        return "REFUTED", facts
    return "UNRESOLVED", facts


def _load_rows(root: Path, behaviors: list[str], seeds: list[int]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    sources = {}
    commits = set()
    for behavior in behaviors:
        for seed in seeds:
            path = root / behavior / f"seed{seed}" / "all_arms_spearman.json"
            if not path.exists():
                raise FileNotFoundError(f"missing seed summary: {path}")
            payload = json.loads(path.read_text())
            meta = payload.get("meta", {})
            if not meta.get("complete"):
                raise RuntimeError(f"incomplete seed summary: {path}")
            if int(meta.get("seed", -1)) != seed or meta.get("behavior") != behavior:
                raise RuntimeError(f"seed/behavior metadata mismatch: {path}")
            commits.add(str(meta.get("commit")))
            rows.extend(payload["rows"])
            sources[f"{behavior}|seed{seed}"] = str(path)
    if len(commits) != 1:
        raise RuntimeError(f"seed summaries span multiple code commits: {sorted(commits)}")
    return rows, {"files": sources, "commit": next(iter(commits))}


def _exact_d(rows: list[dict], predicate) -> float:
    needed = {"arm4_ridge_ctx", "arm7_map_ridge_pred"}
    selected = [r for r in rows if r["arm"] in needed and predicate(r)]
    by_arm = {}
    for row in selected:
        arm = row["arm"]
        if arm in by_arm:
            raise RuntimeError(f"duplicate row for arm {arm}: {selected}")
        by_arm[arm] = row["rho_frozen"]
    if set(by_arm) != needed:
        raise RuntimeError(f"expected exactly {sorted(needed)}, got {sorted(by_arm)}")
    if any(by_arm[a] is None for a in needed):
        raise RuntimeError(f"nonfinite D component: {by_arm}")
    return float(by_arm["arm7_map_ridge_pred"] - by_arm["arm4_ridge_ctx"])


def _group_d(
    rows: list[dict],
    *,
    behavior: str,
    seed: int,
    config: str,
    u_size: int | None,
    setting_group: str,
) -> float:
    subset = [
        r
        for r in rows
        if r["behavior"] == behavior
        and int(r["seed"]) == seed
        and r["config"] == config
        and r["map_variant"] == "true"
        and r["u_size"] == u_size
        and r["setting_group"] == setting_group
    ]
    settings = sorted({r["eval_rung"] for r in subset})
    if setting_group != "ood" and len(settings) != 1:
        raise RuntimeError(
            f"{behavior}/seed{seed}/{config}/{setting_group}: expected one setting, got {settings}"
        )
    if setting_group == "ood":
        expected = {"evil": 5, "sycophancy": 6, "hallucination": 2}[behavior]
        if len(settings) != expected:
            raise RuntimeError(
                f"{behavior}/seed{seed}/{config}: OOD macro has {len(settings)} rungs, "
                f"expected {expected}: {settings}"
            )
    values = [
        _exact_d(subset, lambda r, setting=setting: r["eval_rung"] == setting)
        for setting in settings
    ]
    return float(sum(values) / len(values))


def fold(
    rows: list[dict],
    *,
    behaviors: list[str],
    seeds: list[int],
    delta: float,
    alpha: float,
) -> dict:
    groups = []
    for behavior in behaviors:
        for config in CONFIGS:
            for setting_group in SETTING_GROUPS:
                ladder_by_seed = {}
                trends = []
                endpoints = []
                judged_contrasts = []
                for seed in seeds:
                    ladder = {
                        u: _group_d(
                            rows,
                            behavior=behavior,
                            seed=seed,
                            config=config,
                            u_size=u,
                            setting_group=setting_group,
                        )
                        for u in U_SIZES
                    }
                    ladder_by_seed[str(seed)] = {str(k): v for k, v in ladder.items()}
                    trends.append(seed_spearman_trend(list(ladder), list(ladder.values())))
                    endpoints.append(ladder[U_SIZES[-1]] - ladder[U_SIZES[0]])
                    judged = _group_d(
                        rows,
                        behavior=behavior,
                        seed=seed,
                        config="judged_only",
                        u_size=None,
                        setting_group=setting_group,
                    )
                    generic_250 = _group_d(
                        rows,
                        behavior=behavior,
                        seed=seed,
                        config="generic_only",
                        u_size=U_SIZES[0],
                        setting_group=setting_group,
                    )
                    judged_contrasts.append(judged - generic_250)
                trend_result = _t_interval(trends, 0.95)
                endpoint_result = equivalence_tost(
                    endpoints, delta=delta, alpha=alpha
                )
                judged_result = equivalence_tost(
                    judged_contrasts, delta=delta, alpha=alpha
                )
                verdict, reasons = classify_verdict(
                    trend_result, endpoint_result, judged_result, delta=delta
                )
                groups.append(
                    {
                        "behavior": behavior,
                        "setting_group": setting_group,
                        "config": config,
                        "evidentiary_role": (
                            "primary causal ladder"
                            if config == "generic_only"
                            else "secondary paper-comparability ladder"
                        ),
                        "ladder_by_seed": ladder_by_seed,
                        "trend": trend_result,
                        "endpoint_change": endpoint_result,
                        "judged_minus_generic_u250": judged_result,
                        "verdict": verdict,
                        "verdict_reasons": reasons,
                    }
                )
    foldclean_records = []
    for behavior in behaviors:
        values = {}
        for seed in seeds:
            foldclean_d = _group_d(
                rows,
                behavior=behavior,
                seed=seed,
                config="fold_clean_union_full",
                u_size=U_SIZES[-1],
                setting_group="in_dist",
            )
            standard = _group_d(
                rows,
                behavior=behavior,
                seed=seed,
                config="union_scaled",
                u_size=U_SIZES[-1],
                setting_group="in_dist",
            )
            values[str(seed)] = {
                "fold_clean_D": foldclean_d,
                "standard_D": standard,
                "difference": foldclean_d - standard,
            }
        # Keep this diagnostic descriptive: it is not part of any verdict.
        foldclean_record = {
            "behavior": behavior,
            "values": values,
            "difference_95ci": _t_interval(
                [values[str(seed)]["difference"] for seed in seeds], 0.95
            ),
            "descriptive_only": True,
        }
        foldclean_records.append(foldclean_record)
    return {
        "schema_version": SCHEMA_VERSION,
        "delta": delta,
        "alpha": alpha,
        "u_sizes": list(U_SIZES),
        "seeds": seeds,
        "groups": groups,
        "fold_clean_diagnostic": foldclean_records,
        "decision_contract": {
            "supported": (
                "trend 95% CI wholly above zero, endpoint change not equivalent to zero, "
                "and judged-minus-generic-U250 contrast equivalent to zero"
            ),
            "refuted": (
                "endpoint change equivalent to zero, or judged contrast 90% CI lower "
                "bound at least +delta"
            ),
            "otherwise": "UNRESOLVED",
            "negative_judged_contrast_note": (
                "a negative judged contrast is not automatically equivalent"
            ),
        },
        "inference_note": (
            "All confidence intervals and TOST decisions operate on five independent "
            "seed-level statistics. The registered t procedure is low-powered at n=5 "
            "and sensitive to strong non-normality; unresolved is therefore an expected "
            "outcome rather than evidence of equivalence."
        ),
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# Issue 1739 unlabeled-data ladder",
        "",
        (
            f"Seed-level inference uses n={len(payload['seeds'])}, TOST alpha="
            f"{payload['alpha']:.2f}, and equivalence margin delta={payload['delta']:.3f}."
        ),
        "",
        "| Behavior | Setting | Ladder | Verdict | Trend mean [95% CI] | Endpoint equivalent | Judged equivalent |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for group in payload["groups"]:
        trend = group["trend"]
        lines.append(
            "| {behavior} | {setting_group} | {config} | {verdict} | "
            "{mean:.3f} [{lo:.3f}, {hi:.3f}] | {endpoint} | {judged} |".format(
                **group,
                mean=trend["mean"],
                lo=trend["interval"][0],
                hi=trend["interval"][1],
                endpoint=group["endpoint_change"]["equivalent"],
                judged=group["judged_minus_generic_u250"]["equivalent"],
            )
        )
    lines.extend(
        [
            "",
            "The generic-only ladder is primary. Union-scaled rows are secondary.",
            "The full-U fold-clean union diagnostic is descriptive and cannot change a verdict.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--behaviors", nargs="+", choices=BEHAVIORS, default=list(BEHAVIORS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        ap.error("--seeds contains duplicates")
    if args.delta <= 0 or not 0 < args.alpha < 0.5:
        ap.error("--delta must be positive and --alpha must lie in (0, 0.5)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.import_check:
        import numpy  # noqa: F401
        import scipy  # noqa: F401

        print("[uladder-fold] import-check OK")
        return 0
    rows, source_meta = _load_rows(args.input_root, args.behaviors, args.seeds)
    payload = fold(
        rows,
        behaviors=args.behaviors,
        seeds=args.seeds,
        delta=args.delta,
        alpha=args.alpha,
    )
    payload["sources"] = source_meta
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.out_dir / "uladder_fold.json", payload)
    _atomic_text(args.out_dir / "uladder_fold.md", _markdown(payload))
    print(f"[uladder-fold] wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
