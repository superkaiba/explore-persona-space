"""Within-activity-decile Spearman profiles for every continuous predictor.

Partialling and stratifying answer different questions whenever the
predictor->R^2 relationship varies with activity (effect modification). This
computes rho(predictor, R^2) INSIDE each full-width activity decile, so the
heterogeneity is a measured curve rather than an argument between rounds.

Reading rule:
  flat profile      -> homogeneous effect; the single activity-partialled rho
                       is a valid summary.
  sloped / flipping -> effect modification; the profile REPLACES the single
                       number.
  rightmost point   -> "among only high-activating features, what explains
                       predictability?"

The top decile is also the ESTIMABLE stratum (nearly every feature there has
R^2 > 0), so the measurement-noise confound is absent by construction at that
end of the curve.

Usage:
    uv run python scripts/issue1482_rho_by_activity_decile.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import set_paper_style

REPO = Path(__file__).resolve().parents[1]
MATRIX = REPO / "eval_results/issue_1482/predictor_battery/fullwidth_matrix.npz"
OUTDIR = REPO / "figures/issue_1482/predictor_battery"
N_DECILES = 10
N_BOOT = 2000
SEED = 148205

PREDICTORS = {
    "consistency": "within-answer consistency",
    "proj_var": "dense variance along decoder dir",
    "write_norm": "gamma-scaled write norm",
    "enc_norm": "encoder norm",
    "footprint_kurt": "footprint kurtosis",
    "footprint_skew": "footprint skew",
    "footprint_var": "footprint variance",
}
HEADLINE = ("consistency", "proj_var", "write_norm", "footprint_kurt")


def _rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(np.float64)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _boot_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Batched bootstrap CI for Spearman (ranks recomputed per draw is too slow;
    ranks are computed once and resampled — stated as the design choice)."""
    n = x.size
    if n < 30:
        return (float("nan"), float("nan"))
    rx, ry = _rank(x), _rank(y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    idx = rng.integers(0, n, size=(N_BOOT, n))
    xs, ys = rx[idx], ry[idx]
    xs -= xs.mean(axis=1, keepdims=True)
    ys -= ys.mean(axis=1, keepdims=True)
    num = (xs * ys).sum(axis=1)
    den = np.sqrt((xs**2).sum(axis=1) * (ys**2).sum(axis=1))
    r = num / np.maximum(den, 1e-12)
    return (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


def main() -> None:
    d = np.load(MATRIX, allow_pickle=True)
    act, r2 = d["activity"], d["r2"]
    rng = np.random.default_rng(SEED)

    edges = np.quantile(act, np.linspace(0, 1, N_DECILES + 1))
    edges[-1] = np.inf
    bins = [(act >= edges[i]) & (act < edges[i + 1]) & np.isfinite(r2) for i in range(N_DECILES)]

    out: dict[str, dict] = {}
    for key, label in PREDICTORS.items():
        if key not in d.files:
            continue
        v = d[key]
        prof, cis = [], []
        for m in bins:
            ok = m & np.isfinite(v)
            prof.append(_spearman(v[ok], r2[ok]) if ok.sum() >= 30 else float("nan"))
            cis.append(_boot_ci(v[ok], r2[ok], rng) if key in HEADLINE else (np.nan, np.nan))
        finite = [p for p in prof if np.isfinite(p)]
        out[key] = {
            "label": label,
            "rho_by_decile": prof,
            "ci95_by_decile": cis,
            "rho_full": _spearman(
                v[np.isfinite(v) & np.isfinite(r2)], r2[np.isfinite(v) & np.isfinite(r2)]
            ),
            "spread_max_minus_min": float(max(finite) - min(finite)) if finite else float("nan"),
            "sign_flips": bool(finite and min(finite) < 0 < max(finite)),
        }

    set_paper_style()
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    xs = np.arange(1, N_DECILES + 1)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (key, rec) in enumerate(
        sorted(out.items(), key=lambda kv: -abs(kv[1]["spread_max_minus_min"]))
    ):
        c = palette[i % len(palette)]
        lw = 2.2 if key in HEADLINE else 1.2
        ax.plot(
            xs, rec["rho_by_decile"], marker="o", ms=4, color=c, linewidth=lw, label=rec["label"]
        )
        if key in HEADLINE:
            lo = [a for a, _ in rec["ci95_by_decile"]]
            hi = [b for _, b in rec["ci95_by_decile"]]
            ax.fill_between(xs, lo, hi, color=c, alpha=0.15, linewidth=0)
    ax.axhline(0, color="0.4", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{i}\n{edges[i - 1]:.0e}" for i in xs], fontsize=7.5)
    ax.set_xlabel("activity decile (label = decile index / lower activity edge)")
    ax.set_ylabel(r"within-decile Spearman $\rho$ vs per-feature $R^2$")
    ax.set_title(
        "Predictor effects are NOT homogeneous across activity\n"
        "(rightmost point = 'among only high-activating features, what explains?')"
    )
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = OUTDIR / "rho_by_activity_decile"
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    stem.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "source": str(MATRIX.relative_to(REPO)),
                "n_features": int(act.size),
                "n_deciles": N_DECILES,
                "decile_edges": [float(e) for e in edges[:-1]] + ["inf"],
                "n_boot": N_BOOT,
                "seed": SEED,
                "bootstrap_design": "ranks computed once on the full sample, then resampled",
                "predictors": out,
                "reading_rule": (
                    "flat profile => homogeneous effect, the single activity-partialled "
                    "rho is a valid summary; sloped or sign-flipping => effect "
                    "modification, the profile replaces the single number; the rightmost "
                    "point answers 'among only high-activating features, what explains "
                    "predictability'."
                ),
                "caveat": (
                    "R^2 is the #1738 multi-turn SAE->SAE context-arm read (PROVISIONAL "
                    "target; the dense->SAE full-width refit is pending). Covariates are "
                    "the #1482 single-turn corpus. The top decile is also the estimable "
                    "stratum, so the noise confound is absent at that end by construction."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {stem}.png")
    print(f"{'predictor':<34} {'full':>7} {'d1':>7} {'d5':>7} {'d10':>7} {'spread':>7} flip")
    for key, rec in sorted(out.items(), key=lambda kv: -abs(kv[1]["spread_max_minus_min"])):
        p = rec["rho_by_decile"]
        print(
            f"{rec['label']:<34} {rec['rho_full']:+7.3f} {p[0]:+7.3f} {p[4]:+7.3f} "
            f"{p[-1]:+7.3f} {rec['spread_max_minus_min']:7.3f} {'YES' if rec['sign_flips'] else '-'}"
        )


if __name__ == "__main__":
    main()
