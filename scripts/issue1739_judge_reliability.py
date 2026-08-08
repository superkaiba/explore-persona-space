"""Draw-to-draw judge reliability for issue #1739 (analysis-only, 0 GPU, 0 API).

Reads the committed raw judge shards (base wave + the rejudge800 whole-item
replacement pass), reconstructs the per-(item, draw) score matrix EXACTLY as
the production reduce does (drop-never-coerce; whole-item replace for
re-judged items), verifies parity against the production
``dv_dataset/<behavior>/labeling.json`` per-rollout means, and reports
draw-to-draw agreement per behavior.

Statistics (llm-judging.md rules 4/9/14/15/24):
  * ICC(1,1) one-way random effects = single-draw reliability r_yy
  * ICC(1,k) = Spearman-Brown-stepped k-draw-mean reliability
  * Krippendorff's alpha (interval) over the exchangeable draws
  * pairwise Pearson/Spearman by draw index (00/01/02)
  * the reliability CEILING sqrt(r_yy) for each
  * per-arm content-drop and transport-loss rates (rules 9 / 24 split)
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Shared-VM thread caps (#847) bind in-process only when load_dotenv() runs
# BEFORE the first heavy import (numpy/scipy here).
load_dotenv()

import numpy as np  # noqa: E402

# (reassembled base shard, rejudge800 whole-item-replacement file or None).
# The rejudge paths keep the hf_hub_download verbatim-prefix mirror layout.
_MIRROR = "issue1739_ctxmap/judge"
RAW = {
    "evil": (
        "evil_judge_raw_trait.json",
        f"{_MIRROR}/evil/judge_raw_trait_rejudge800_20260728T233651Z.json",
    ),
    "sycophancy": (
        "sycophancy_judge_raw_trait.json",
        f"{_MIRROR}/sycophancy/judge_raw_trait_rejudge800_20260728T232410Z.json",
    ),
    "hallucination": ("hallucination_judge_raw_abstain.json", None),
}
N_DRAWS = 3
HALLU_FABRICATED_THRESHOLD = 50.0


def _score(parsed: object) -> float | None:
    """Production parse: numeric in [0,100] kept, everything else dropped."""
    if not isinstance(parsed, dict):
        return None
    if parsed.get("error"):
        return None
    s = parsed.get("score")
    if isinstance(s, bool) or not isinstance(s, (int, float)):
        return None
    if not (0.0 <= float(s) <= 100.0):
        return None
    return float(s)


def _is_transport(parsed: object) -> bool:
    from explore_persona_space.eval.batch_judge import is_transport_error_dict

    return isinstance(parsed, dict) and is_transport_error_dict(parsed)


def load_draw_matrix(raw_dir: Path, behavior: str) -> tuple[dict, dict]:
    """-> ({item_id: [d00, d01, d02] with None for dropped}, drop tallies)."""
    base_name, rejudge_name = RAW[behavior]
    base = json.loads((raw_dir / base_name).read_text())
    draws: dict[str, list[float | None]] = {}
    transport: dict[str, int] = {}

    def ingest(all_scores: dict, only: set[str] | None = None) -> None:
        for cid, parsed in all_scores.items():
            item_id, _idx, comp = cid.rsplit("__", 2)
            if only is not None and item_id not in only:
                continue
            ci = int(comp)
            row = draws.setdefault(item_id, [None] * N_DRAWS)
            row[ci] = _score(parsed)
            if _is_transport(parsed):
                transport[item_id] = transport.get(item_id, 0) + 1

    ingest(base["all_scores"])
    n_rejudged_items = 0
    if rejudge_name is not None:
        rj = json.loads((raw_dir / rejudge_name).read_text())
        replaced = {cid.rsplit("__", 2)[0] for cid in rj["all_scores"]}
        # whole-item replace (uniform instrument per item, rejudge script)
        for item_id in replaced:
            draws[item_id] = [None] * N_DRAWS
            transport.pop(item_id, None)
        ingest(rj["all_scores"], only=replaced)
        n_rejudged_items = len(replaced)

    tally = {
        "n_items": len(draws),
        "n_draws_total": sum(len(v) for v in draws.values()),
        "n_content_dropped_draws": sum(sum(1 for d in v if d is None) for v in draws.values())
        - sum(transport.values()),
        "n_transport_lost_draws": sum(transport.values()),
        "n_items_rejudged800": n_rejudged_items,
        "n_items_all_draws_dropped": sum(1 for v in draws.values() if all(d is None for d in v)),
    }
    return draws, tally


def parity_check(draws: dict, behavior: str, dv_path: Path) -> dict:
    """Reduce our matrix like production and compare to labeling.json."""
    dv = json.loads(dv_path.read_text())
    if behavior == "hallucination":
        return {"applicable": False, "reason": "three-way DV; per-rollout means not stored"}
    ours: dict[str, float | None] = {}
    for item_id, row in draws.items():
        kept = [d for d in row if d is not None]
        ours[item_id] = (sum(kept) / len(kept)) if kept else None
    n_cmp = n_mismatch = 0
    worst = 0.0
    for r in dv["rows"]:
        for kk, v in r["per_rollout_scores"].items():
            item_id = f"{r['context_id']}_{kk}"
            mine = ours.get(item_id, "MISSING")
            n_cmp += 1
            if v is None or mine is None:
                if v is not mine:
                    n_mismatch += 1
                continue
            if mine == "MISSING":
                n_mismatch += 1
                continue
            d = abs(float(v) - float(mine))
            worst = max(worst, d)
            if d > 1e-9:
                n_mismatch += 1
    return {
        "applicable": True,
        "n_rollouts_compared": n_cmp,
        "n_mismatch": n_mismatch,
        "max_abs_diff": worst,
        "pass": n_mismatch == 0,
    }


def icc1(mat: np.ndarray) -> dict:
    """ICC(1,1) and ICC(1,k), one-way random effects, balanced k raters."""
    n, k = mat.shape
    row_means = mat.mean(axis=1)
    grand = mat.mean()
    ss_between = k * np.sum((row_means - grand) ** 2)
    ss_within = np.sum((mat - row_means[:, None]) ** 2)
    ms_b = ss_between / (n - 1)
    ms_w = ss_within / (n * (k - 1))
    icc_1 = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w)
    icc_k = (ms_b - ms_w) / ms_b
    return {
        "icc_1_1": float(icc_1),
        "icc_1_k": float(icc_k),
        "ms_between": float(ms_b),
        "ms_within": float(ms_w),
        "n": int(n),
        "k": int(k),
    }


def krippendorff_alpha_interval(mat: np.ndarray) -> float:
    """Interval-metric alpha for a complete n x k matrix (all units k-rated).

    Computed via the algebraic identity for complete data:
      D_o = mean over units of the mean squared pairwise within-unit difference
      D_e = mean squared pairwise difference over ALL values pooled
      alpha = 1 - D_o / D_e
    """
    n, k = mat.shape
    # observed disagreement: within-unit mean squared pairwise diff
    # sum_{i<j}(x_i-x_j)^2 = k*sum(x^2) - (sum x)^2
    row_sq = (mat**2).sum(axis=1)
    row_sum = mat.sum(axis=1)
    within = k * row_sq - row_sum**2  # = sum over ordered pairs/2 *2 ... see below
    n_pairs_per_unit = k * (k - 1)
    d_o = within.sum() / (n * n_pairs_per_unit)
    flat = mat.ravel()
    N = flat.size
    d_e = (N * (flat**2).sum() - flat.sum() ** 2) / (N * (N - 1))
    return float(1.0 - d_o / d_e) if d_e > 0 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def analyse(draws: dict, behavior: str) -> dict:
    complete = {i: v for i, v in draws.items() if all(d is not None for d in v)}
    mat = np.array([complete[i] for i in sorted(complete)], dtype=float)
    out: dict = {
        "behavior": behavior,
        "n_items_total": len(draws),
        "n_items_complete_3_draws": int(mat.shape[0]),
        "frac_items_complete": float(len(complete) / len(draws)) if draws else None,
        "score_mean": float(mat.mean()),
        "score_sd_overall": float(mat.std(ddof=1)),
        "n_items_all_draws_identical": int((mat.max(axis=1) == mat.min(axis=1)).sum()),
    }
    out["frac_items_all_draws_identical"] = float(out["n_items_all_draws_identical"] / mat.shape[0])
    icc = icc1(mat)
    out["icc"] = icc
    r_yy = icc["icc_1_1"]
    out["single_draw_reliability_r_yy"] = r_yy
    out["ceiling_sqrt_r_yy_single_draw"] = float(math.sqrt(max(r_yy, 0.0)))
    out["three_draw_mean_reliability"] = icc["icc_1_k"]
    out["ceiling_sqrt_r_yy_three_draw_mean"] = float(math.sqrt(max(icc["icc_1_k"], 0.0)))
    # Spearman-Brown check: icc_1_k should equal k*r/(1+(k-1)r)
    sb = N_DRAWS * r_yy / (1 + (N_DRAWS - 1) * r_yy)
    out["spearman_brown_check_from_icc11"] = float(sb)
    out["krippendorff_alpha_interval"] = krippendorff_alpha_interval(mat)
    pw = {}
    for i, j in combinations(range(N_DRAWS), 2):
        pw[f"draw{i:02d}_vs_draw{j:02d}"] = {
            "pearson": pearson(mat[:, i], mat[:, j]),
            "spearman": spearman(mat[:, i], mat[:, j]),
        }
    out["pairwise_by_draw_index"] = pw
    out["pairwise_mean_pearson"] = float(np.mean([v["pearson"] for v in pw.values()]))
    out["pairwise_mean_spearman"] = float(np.mean([v["spearman"] for v in pw.values()]))
    # within-item spread
    out["within_item_sd_mean"] = float(mat.std(axis=1, ddof=1).mean())
    out["within_item_range_mean"] = float((mat.max(axis=1) - mat.min(axis=1)).mean())
    out["within_item_range_p90"] = float(np.percentile(mat.max(axis=1) - mat.min(axis=1), 90))
    if behavior == "hallucination":
        lab = (mat >= HALLU_FABRICATED_THRESHOLD).astype(int)
        agree = (lab.sum(axis=1) % N_DRAWS == 0).mean()
        out["hallu_label_level"] = {
            "note": "each draw thresholded at >=50 -> fabricated(1)/abstained(0)",
            "frac_items_unanimous_label": float(agree),
            "frac_draws_fabricated": float(lab.mean()),
            "krippendorff_alpha_nominal_equiv_interval_on_binary": (
                krippendorff_alpha_interval(lab.astype(float))
            ),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="data/issue_1739/hf_dl/judge")
    ap.add_argument("--dv-root", default="eval_results/issue_1739/dv_dataset")
    ap.add_argument("--out", default="eval_results/issue_1739/judge_reliability")
    args = ap.parse_args()
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "n_draws_per_item": N_DRAWS,
        "judge_model": "claude-sonnet-4-5-20250929",
        "judge_temperature": 1.0,
        "behaviors": {},
    }
    for behavior in RAW:
        draws, tally = load_draw_matrix(raw_dir, behavior)
        par = parity_check(draws, behavior, Path(args.dv_root) / behavior / "labeling.json")
        res = analyse(draws, behavior)
        res["draw_tally"] = tally
        res["production_parity"] = par
        report["behaviors"][behavior] = res
        print(
            f"[{behavior}] parity={par} icc11={res['single_draw_reliability_r_yy']:.4f} "
            f"icc1k={res['three_draw_mean_reliability']:.4f} "
            f"alpha={res['krippendorff_alpha_interval']:.4f} "
            f"n={res['n_items_complete_3_draws']}"
        )
        np.save(
            out_dir / f"draw_matrix_{behavior}.npy",
            np.array(
                [draws[i] for i in sorted(draws) if all(d is not None for d in draws[i])],
                dtype=float,
            ),
        )
    (out_dir / "judge_draw_reliability.json").write_text(json.dumps(report, indent=1))
    print("wrote", out_dir / "judge_draw_reliability.json")
    return 0


# ---------------------------------------------------------------------------
# Context-level nested variance decomposition (added round 2)
# ---------------------------------------------------------------------------

K_ROLLOUTS = 5


def nested_decomposition(draws: dict, k_rollouts: int = K_ROLLOUTS) -> dict:
    """Balanced 3-level nested ANOVA: context / rollout(context) / draw(rollout).

    Restricted to contexts whose k_rollouts rollouts ALL have all N_DRAWS
    draws kept (a balanced design -> closed-form method-of-moments variance
    components). Reports the reliability of the production context DV (mean
    over k_rollouts x N_DRAWS) and the judge-noise-free counterfactual.
    """
    per_ctx: dict[str, dict[int, list[float | None]]] = {}
    for item_id, row in draws.items():
        cid, _, kpart = item_id.rpartition("_k")
        per_ctx.setdefault(cid, {})[int(kpart)] = row
    complete = [
        c
        for c, rolls in per_ctx.items()
        if len(rolls) == k_rollouts and all(all(d is not None for d in r) for r in rolls.values())
    ]
    if not complete:
        return {"n_contexts_balanced": 0}
    a, b, n = len(complete), k_rollouts, N_DRAWS
    y = np.array(
        [[per_ctx[c][k] for k in sorted(per_ctx[c])] for c in sorted(complete)], dtype=float
    )  # (a, b, n)
    ctx_mean = y.mean(axis=(1, 2))
    roll_mean = y.mean(axis=2)
    grand = y.mean()
    ms_a = b * n * np.sum((ctx_mean - grand) ** 2) / (a - 1)
    ms_b = n * np.sum((roll_mean - ctx_mean[:, None]) ** 2) / (a * (b - 1))
    ms_e = np.sum((y - roll_mean[:, :, None]) ** 2) / (a * b * (n - 1))
    var_draw = ms_e
    var_roll = max((ms_b - ms_e) / n, 0.0)
    var_ctx = max((ms_a - ms_b) / (b * n), 0.0)
    dv_noise = var_roll / b + var_draw / (b * n)
    rel_dv = var_ctx / (var_ctx + dv_noise) if (var_ctx + dv_noise) > 0 else float("nan")
    rel_no_judge = var_ctx / (var_ctx + var_roll / b) if var_ctx + var_roll > 0 else float("nan")
    tot = var_ctx + var_roll + var_draw
    return {
        "n_contexts_balanced": int(a),
        "design": f"{b} rollouts x {n} draws",
        "var_context_signal": float(var_ctx),
        "var_rollout_within_context_generation": float(var_roll),
        "var_draw_within_rollout_judge": float(var_draw),
        "pct_var_context": float(100 * var_ctx / tot),
        "pct_var_rollout_generation": float(100 * var_roll / tot),
        "pct_var_draw_judge": float(100 * var_draw / tot),
        "reliability_context_dv": float(rel_dv),
        "attenuation_ceiling_context_dv": float(math.sqrt(max(rel_dv, 0.0))),
        "reliability_context_dv_if_judge_noise_zero": float(rel_no_judge),
        "judge_noise_share_of_dv_error_variance": float(
            (var_draw / (b * n)) / dv_noise if dv_noise > 0 else float("nan")
        ),
        "reliability_gain_from_removing_judge_noise": float(rel_no_judge - rel_dv),
    }


def drop_class_breakdown(raw_dir: Path, behavior: str) -> dict:
    """Split dropped draws by CLASS, per raw wave (rule 9 / #1801).

    ``REFUSAL`` is a rubric-INSTRUCTED verdict ("REFUSAL if the model clearly
    refused to answer") — it reports the EVALUATED model's refusal, not a judge
    instrument failure — so it is reported separately from ``parse_error``
    (the rule-23 truncation class).
    """
    base_name, rejudge_name = RAW[behavior]
    waves = [("base", base_name)] + ([("rejudge800", rejudge_name)] if rejudge_name else [])
    out: dict = {}
    for tag, name in waves:
        raw = json.loads((raw_dir / name).read_text())
        counts: dict[str, int] = {}
        for parsed in raw["all_scores"].values():
            if not isinstance(parsed, dict):
                key = "non_dict"
            elif parsed.get("error"):
                key = "parse_error"
            elif isinstance(parsed.get("score"), str):
                key = f"instructed_{parsed['score'][:24]}"
            elif _score(parsed) is not None:
                key = "kept_numeric"
            else:
                key = "malformed_or_out_of_range"
            counts[key] = counts.get(key, 0) + 1
        total = sum(counts.values())
        out[tag] = {
            "n_draws": total,
            "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "frac": {k: v / total for k, v in sorted(counts.items(), key=lambda kv: -kv[1])},
        }
    return out


def main_decomp() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/issue_1739/hf_dl/judge")
    ap.add_argument("--out", default="eval_results/issue_1739/judge_reliability")
    args = ap.parse_args()
    out = Path(args.out)
    report = json.loads((out / "judge_draw_reliability.json").read_text())
    for behavior in RAW:
        draws, _ = load_draw_matrix(Path(args.raw_dir), behavior)
        dec = nested_decomposition(draws)
        report["behaviors"][behavior]["context_level_decomposition"] = dec
        report["behaviors"][behavior]["drop_class_breakdown"] = drop_class_breakdown(
            Path(args.raw_dir), behavior
        )
        print(f"[{behavior}] {json.dumps(dec)}")
    (out / "judge_draw_reliability.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    import sys

    if "--decomp" in sys.argv:
        sys.argv.remove("--decomp")
        raise SystemExit(main_decomp())
    raise SystemExit(main())
