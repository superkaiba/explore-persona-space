#!/usr/bin/env python
"""Issue #816 Phase C — OFF-POD analysis (Exp-5 null battery + Exp-2/4 reads + figures).

Runs on the VM (CPU) AFTER Phase B judging (plan v2 §9). Closed-form / sampling
statistics over the scored eval JSONs + the Exp-5 predictor tensors; no model
calls, no GPU. Three reads + the hero figures (§6):

- Exp-5 screening: assemble the (24, N_LAYERS, D) per-dataset mean-diff predictor
  + the 24 #778 post-ft trait scores (y-axis), run the #778 null battery at the
  FROZEN layer 20 (``screening.run_null_battery_screening``), write per-trait
  ``screening_{trait}_nullbattery.json`` + the per-draw x per-axis matrices.
- Exp-2 steering: per-trait beat-count over the positive coefficients (real mean
  trait score vs the 15-draw random band's max) at coefs {2,4,8}, coherence-gated.
- Exp-4 preventative: real reduction (coef-0 minus steered) vs the <=20-draw random
  band AT the PRE-FROZEN alpha* = 1.25 (headline); one-sided empirical p + p-floor.

Figures (paper-plots conventions) -> figures/issue_816/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib
import issue816_lib as ilib

from explore_persona_space.analysis import null_battery
from explore_persona_space.experiments.issue816 import screening
from explore_persona_space.orchestrate.env import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue816.analysis")
load_dotenv()

EXP4_ALPHA_STAR = 1.25
EXP2_POSITIVE_COEFS = (2.0, 4.0, 8.0)
COHERENCE_GATE_EXP2 = 40.0  # a coef whose mean coherence < 40 is excluded from the claim
COHERENCE_GATE_EXP4 = 80.0  # the paper's headline coherence gate


# ── Exp-5 screening null battery ───────────────────────────────────────────────


def _load_screening_predictor(tensor_root: Path, trait: str) -> tuple[np.ndarray, list[str]]:
    """Stack the 24 per-dataset mean-diff activation rows -> (24, N_LAYERS, D)."""
    rows, names = [], []
    for path in sorted(tensor_root.glob("*.json")):
        with open(path) as f:
            cap = json.load(f)
        rows.append(np.asarray(cap["mean_diff_activation"], dtype=np.float64))
        names.append(cap["dataset"])
    if not rows:
        raise FileNotFoundError(f"no screening predictor tensors under {tensor_root}")
    return np.stack(rows, axis=0), names


def _load_778_targets(eval_778_root: Path, dataset_names: list[str], trait: str) -> np.ndarray:
    """The 24 #778 post-ft trait scores (y-axis), aligned to dataset_names order."""
    y = []
    for cell in dataset_names:
        family, version = lib.split_cell_tag(cell)
        row = ilib.read_778_finetune_score(eval_778_root, trait, family, version)
        score = _extract_trait_mean(row, trait)
        y.append(score)
    return np.asarray(y, dtype=np.float64)


def _extract_trait_mean(row: dict, trait: str) -> float:
    """Pull the graded trait mean from a #778 finetune-score JSON (flexible keys)."""
    for k in ("trait_score", "trait_graded_mean", "graded_mean", "trait_mean", "mean_score"):
        if k in row and row[k] is not None:
            return float(row[k])
    # nested per-trait dict fallback
    for k in ("scores", "by_trait", "trait_scores"):
        d = row.get(k)
        if isinstance(d, dict) and trait in d and d[trait] is not None:
            v = d[trait]
            return float(v.get("graded_mean", v) if isinstance(v, dict) else v)
    raise KeyError(f"could not find trait mean in #778 row keys={list(row.keys())}")


def run_exp5(args) -> dict:
    """Run Exp-5 null battery screening for all traits.

    Uses honest v2 r_B directions and neutral_cov from HF, separate pos/neg pools.
    Writes per-trait screening JSON to out_root/v3/.
    """
    tensor_root = Path(args.screening_tensor_root)
    eval_778 = Path(args.eval_778_root)
    cache_dir = Path(args.cache_dir)
    out_root = Path(args.out_root)
    results = {}
    for trait in args.traits:
        predictor, names = _load_screening_predictor(tensor_root, trait)
        target = _load_778_targets(eval_778, names, trait)
        rb, _ = ilib.fetch_rb(trait, cache_dir=cache_dir)
        rb = rb.numpy()
        pos = _load_pool(trait, "pos", cache_dir)  # (n_pos, L, D)
        neg = _load_pool(trait, "neg", cache_dir)  # (n_neg, L, D)
        # Fetch v2 neutral covariance (Σ_neutral, not contaminated pos+neg pool).
        neutral_cov_tensor, _ = ilib.fetch_neutral_cov(trait, cache_dir=cache_dir)
        neutral_cov_np = neutral_cov_tensor.numpy()  # (N_LAYERS, D, D) or (N_LAYERS, D)
        other = {
            t: ilib.fetch_rb(t, cache_dir=cache_dir)[0].numpy() for t in args.traits if t != trait
        }
        pca_diffs = pos[: min(len(pos), len(neg))] - neg[: min(len(pos), len(neg))]

        res = screening.run_null_battery_screening(
            predictor,
            target,
            rb,
            extraction_pos_acts=pos,
            extraction_neg_acts=neg,
            neutral_cov_per_layer=neutral_cov_np,
            other_rbs=other,
            pca_diff_acts=pca_diffs,
            n_draws_stochastic=args.n_null_draws,
            n_draws_within_class=max(50, args.n_null_draws // 4),
            seed=args.seed,
        )
        res["dataset_names"] = names
        res["target"] = [float(x) for x in target]
        # BH is handled inside run_null_battery_screening (stochastic families only).
        out_path = out_root / f"screening_{trait}_nullbattery.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(res, f)
        results[trait] = {
            "real_abs_r": res["real_abs_r_frozen"],
            "nulls": {k: v.get("p97_5") for k, v in res["nulls"].items()},
            "bh_adjusted_stochastic": res.get("bh_adjusted_stochastic"),
            "out": str(out_path),
        }
        logger.info("Exp-5 %s: real |r|=%.3f", trait, res["real_abs_r_frozen"])
    return results


def _load_pool(trait: str, side: str, cache_dir: Path) -> np.ndarray:
    import torch
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=ilib.DATA_REPO,
        repo_type="dataset",
        filename=f"issue778_persona_vectors/analysis_tensors/activations/{trait}_{side}.pt",
        revision="main",
        local_dir=str(cache_dir),
    )
    return np.asarray(torch.load(local, map_location="cpu", weights_only=False), dtype=np.float64)


# ── Exp-2 steering beat-count ──────────────────────────────────────────────────


def run_exp2(args) -> dict:
    """Compute Exp-2 steering beat-count over positive coefficients (coherence-gated)."""
    steer_dir = Path(args.scored_root) / "steering"
    results: dict[str, dict] = {}
    for trait in args.traits:
        real_by_coef: dict[float, dict] = {}
        random_by_coef: dict[float, list[float]] = {}
        for path in sorted(steer_dir.glob(f"steer_{trait}_*_scored.json")):
            with open(path) as f:
                c = json.load(f)
            coef = float(c["coef"])
            tm = c.get("trait_graded_mean")
            cm = c.get("coherence_mean")
            if c["arm"] in ("e2_real", "e2_coef0"):
                real_by_coef[coef] = {"trait": tm, "coherence": cm}
            elif c["arm"] in ("e2_isotropic", "e2_neutral_cov"):
                random_by_coef.setdefault(coef, []).append(tm if tm is not None else float("nan"))
        # Beat-count over the positive matched coefs (coherence-gated).
        beat = 0
        n_compared = 0
        per_coef = {}
        for coef in EXP2_POSITIVE_COEFS:
            real = real_by_coef.get(coef, {})
            rvals = [x for x in random_by_coef.get(coef, []) if not np.isnan(x)]
            if real.get("trait") is None or not rvals:
                per_coef[coef] = {"status": "missing"}
                continue
            gated_out = (real.get("coherence") or 0) < COHERENCE_GATE_EXP2
            band_max = float(np.max(rvals))
            beats = real["trait"] > band_max
            n_compared += 1
            if beats and not gated_out:
                beat += 1
            per_coef[coef] = {
                "real_trait": real["trait"],
                "real_coherence": real.get("coherence"),
                "random_band_max": band_max,
                "n_random": len(rvals),
                "beats_band": beats,
                "coherence_gated_out": gated_out,
            }
        results[trait] = {
            "beat_count": beat,
            "n_positive_coefs_compared": n_compared,
            "per_coef": {str(k): v for k, v in per_coef.items()},
            "real_curve": {str(k): v for k, v in sorted(real_by_coef.items())},
        }
        logger.info("Exp-2 %s: beat %d/%d positive coefs", trait, beat, n_compared)
    return results


# ── Exp-4 preventative real-vs-random at alpha* ────────────────────────────────


def run_exp4(args) -> dict:
    """Compute Exp-4 preventative real-vs-random reduction at pre-frozen alpha* = 1.25."""
    prev_dir = Path(args.scored_root) / "preventative"
    eval_778 = Path(args.eval_778_root)
    results: dict[str, dict] = {}
    for trait in args.traits:
        # coef-0 baseline = reused #778 finetune trait score for the II arm.
        base_row = ilib.read_778_finetune_score(eval_778, trait, trait, "misaligned_2")
        coef0 = _extract_trait_mean(base_row, trait)
        real_at_star = None
        real_curve = {}
        random_reductions = []
        random_coherence_gated = 0
        for path in sorted(prev_dir.glob(f"e4_{trait}_*_scored.json")):
            with open(path) as f:
                c = json.load(f)
            coef = float(c["coef"])
            tm = c.get("trait_graded_mean")
            cm = c.get("coherence_mean")
            if tm is None:
                continue
            reduction = coef0 - tm
            if c["arm"] == "e4_real":
                real_curve[coef] = {"trait": tm, "reduction": reduction, "coherence": cm}
                if abs(coef - EXP4_ALPHA_STAR) < 1e-6:
                    real_at_star = {"trait": tm, "reduction": reduction, "coherence": cm}
            elif (
                c["arm"] in ("e4_isotropic", "e4_neutral_cov")
                and abs(coef - EXP4_ALPHA_STAR) < 1e-6
            ):
                if (cm or 0) < COHERENCE_GATE_EXP4:
                    random_coherence_gated += 1
                random_reductions.append(reduction)
        n_rand = len(random_reductions)
        one_sided_p = None
        beat_all = None
        if real_at_star is not None and n_rand > 0:
            arr = np.asarray(random_reductions, dtype=np.float64)
            # Conservative empirical p per plan §6: (r+1)/(n+1)
            one_sided_p = float((np.sum(arr >= real_at_star["reduction"]) + 1) / (len(arr) + 1))
            beat_all = bool(real_at_star["reduction"] > arr.max())
        p_floor = (1.0 / (n_rand + 1)) if n_rand > 0 else None
        results[trait] = {
            "coef0_baseline_trait": coef0,
            "alpha_star": EXP4_ALPHA_STAR,
            "real_at_alpha_star": real_at_star,
            "n_random_draws": n_rand,
            "random_reduction_band": {
                "min": float(np.min(random_reductions)) if random_reductions else None,
                "max": float(np.max(random_reductions)) if random_reductions else None,
                "mean": float(np.mean(random_reductions)) if random_reductions else None,
            },
            "random_coherence_gated_out": random_coherence_gated,
            "one_sided_p_at_alpha_star": one_sided_p,
            "empirical_p_floor": p_floor,
            "real_beats_all_random": beat_all,
            "real_curve_secondary": {str(k): v for k, v in sorted(real_curve.items())},
            "winners_curse_caveat": (
                "Real ran 4 coefficients {0.5,1.25,3.0,5.0}; the random band exists ONLY at "
                "alpha* = 1.25, so the headline read is at alpha* exclusively. Any max-over-coef "
                "real reduction is a labeled SECONDARY/descriptive read (no honest max-selected "
                "random band recoverable post-hoc)."
            ),
        }
        logger.info(
            "Exp-4 %s: real reduction@a*=%s vs %d random (p=%s, floor=%s)",
            trait,
            real_at_star["reduction"] if real_at_star else None,
            n_rand,
            one_sided_p,
            p_floor,
        )
    return results


# ── Figures (hero list, §6) ────────────────────────────────────────────────────


def make_figures(exp2: dict, exp4: dict, exp5: dict, fig_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # Exp-2 hero: trait score vs coef, real solid + random band, per trait.
    for trait, r in exp2.items():
        curve = r.get("real_curve", {})
        if not curve:
            continue
        coefs = sorted(float(k) for k in curve)
        ys = [curve[str(c)]["trait"] for c in coefs]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(coefs, ys, "-o", color=paper_palette_role("primary"), label="real vector")
        # random band at the positive coefs.
        for coef, pc in r.get("per_coef", {}).items():
            if pc.get("status") == "missing":
                continue
            ax.scatter(
                [float(coef)],
                [pc["random_band_max"]],
                color=paper_palette_role("baseline"),
                marker="_",
                s=200,
                label="_random band max",
            )
        ax.set_xlabel("steering coefficient")
        ax.set_ylabel("graded trait score (0-100)")
        ax.set_title(f"Exp-2 steering: {trait}")
        ax.legend()
        stem = f"exp2_steering_{trait}"
        savefig_paper(fig, stem, dir=str(fig_dir))
        plt.close(fig)
        written.append(stem)

    # Exp-4 hero: paired bar coef-0 vs real@a* vs random band@a*, per trait.
    fig, ax = plt.subplots(figsize=(7, 4))
    traits = list(exp4)
    x = np.arange(len(traits))
    base = [exp4[t]["coef0_baseline_trait"] for t in traits]
    real = [(exp4[t]["real_at_alpha_star"] or {}).get("trait", np.nan) for t in traits]
    rand_mean = [exp4[t]["random_reduction_band"].get("mean") for t in traits]
    rand_trait = [
        (exp4[t]["coef0_baseline_trait"] - m) if m is not None else np.nan
        for t, m in zip(traits, rand_mean, strict=True)
    ]
    w = 0.25
    ax.bar(x - w, base, w, label="coef-0 (#778)", color=paper_palette_role("neutral"))
    ax.bar(x, real, w, label="real @ a*=1.25", color=paper_palette_role("primary"))
    ax.bar(
        x + w, rand_trait, w, label="random band mean @ a*", color=paper_palette_role("baseline")
    )
    ax.set_xticks(x)
    ax.set_xticklabels(traits)
    ax.set_ylabel("post-ft graded trait score")
    ax.set_title("Exp-4 preventative steering @ pre-frozen a*=1.25")
    ax.legend()
    savefig_paper(fig, "exp4_preventative_barr", dir=str(fig_dir))
    plt.close(fig)
    written.append("exp4_preventative_barr")

    # Exp-5 hero: real |r| vs each null's p97.5 band, per trait (bar).
    # Keys match the 8 honest null families from run_null_battery_screening.
    NULL_DISPLAY_KEYS = (
        "isotropic",
        "neutral_cov",
        "within_pos",
        "within_neg",
        "rb_out_iso",
        "cross_trait",
        "pca_top5",
        "contaminated_pooled",
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    traits5 = list(exp5)
    x = np.arange(len(traits5))
    ax.bar(
        x,
        [exp5[t]["real_abs_r"] for t in traits5],
        0.4,
        label="real |r|",
        color=paper_palette_role("primary"),
    )
    n_keys = len(NULL_DISPLAY_KEYS)
    for j, nk in enumerate(NULL_DISPLAY_KEYS):
        vals = [exp5[t]["nulls"].get(nk) or 0 for t in traits5]
        offset = (j - (n_keys - 1) / 2) * 0.06
        ax.scatter(x + offset, vals, s=30, label=f"{nk} p97.5")
    ax.set_xticks(x)
    ax.set_xticklabels(traits5)
    ax.set_ylabel("|Pearson r| (DeltaP vs shift)")
    ax.set_title("Exp-5 screening: real vs 8-null bands @ layer 20")
    ax.legend(fontsize=6, ncol=3)
    savefig_paper(fig, "exp5_screening_bands", dir=str(fig_dir))
    plt.close(fig)
    written.append("exp5_screening_bands")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #816 Phase-C off-pod analysis.")
    parser.add_argument("--out-root", default="eval_results/issue_816/v3")
    # --scored-root: where the Phase-B judge wrote {steering,preventative}/*_scored.json.
    # Defaults to out-root itself (eval_results/issue_816/v3), since judge v3 also
    # writes to that same v3/ namespace.  Pass --scored-root explicitly to override.
    parser.add_argument("--scored-root", default=None)
    parser.add_argument("--eval-778-root", default="eval_results/issue_778")
    parser.add_argument("--screening-tensor-root", default="data/issue_816/store/screening")
    parser.add_argument("--fig-dir", default="figures/issue_816")
    parser.add_argument("--cache-dir", default="data/issue_816/hf_dl")
    parser.add_argument("--traits", nargs="+", default=list(ilib.TRAITS))
    parser.add_argument("--n-null-draws", type=int, default=null_battery.DEFAULT_N_DRAWS)
    parser.add_argument("--seed", type=int, default=42)  # plan §5 base_seed
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["exp2", "exp4", "exp5", "figures"],
        choices=["exp2", "exp4", "exp5", "figures"],
    )
    args = parser.parse_args()

    # Resolve scored-root: directory where Phase-B judge output lives.
    # Since judge v3 defaults --out-root to eval_results/issue_816/v3, scored-root
    # defaults to out_root itself (not its parent).
    if args.scored_root is None:
        args.scored_root = str(Path(args.out_root))

    # Belt-and-suspenders guard (BLOCKER A): refuse to read scored files that are
    # NOT under the v3 output namespace, preventing silent reuse of v2 outputs.
    # The guard fires on the resolved path so an explicit --scored-root override
    # that points outside v3 is also caught.
    _out_root_resolved = Path(args.out_root).resolve()
    _scored_root_resolved = Path(args.scored_root).resolve()
    try:
        _scored_root_resolved.relative_to(_out_root_resolved)
    except ValueError:
        # scored_root is NOT under out_root — verify it's at least under the v3 dir
        # by checking the path component; allow an explicit sibling if the user
        # knows what they're doing via a clear --scored-root flag.
        if "v3" not in str(_scored_root_resolved):
            raise RuntimeError(
                f"BLOCKER A guard: --scored-root={args.scored_root!r} is not under "
                f"the v3 output namespace ({args.out_root!r}). This would silently "
                f"read v2 Phase-B outputs. Pass --scored-root explicitly if intentional."
            ) from None
    out = {}
    if "exp5" in args.phases:
        out["exp5"] = run_exp5(args)
    if "exp2" in args.phases:
        out["exp2"] = run_exp2(args)
    if "exp4" in args.phases:
        out["exp4"] = run_exp4(args)
    if "figures" in args.phases:
        out["figures"] = make_figures(
            out.get("exp2", {}), out.get("exp4", {}), out.get("exp5", {}), Path(args.fig_dir)
        )
    summary_path = Path(args.out_root) / "analysis_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out["repro"] = lib.repro_metadata()
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"phase": "analysis", "summary": str(summary_path)}))


if __name__ == "__main__":
    main()
