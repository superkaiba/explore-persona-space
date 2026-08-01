"""Uncertainty + figure for #1773's register steering-transfer validator.

`--phase analyze` reports point estimates per arm; this adds the reads that decide
whether the register arm actually separates from its nulls: bootstrap CIs on the
per-arm shift magnitudes, a Mann-Whitney U on register-vs-each-null, Wilson CIs on
the mover rates, and the direction-match rate against its label-shuffle null,
broken out by predicted direction.

Renders `figures/issue_1773/register_steer_transfer.png`.
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Shared-VM thread caps (#847): load_dotenv() setdefaults OMP/MKL/OPENBLAS/
# NUMEXPR_NUM_THREADS before matplotlib/numpy/scipy freeze their pools.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

OUT_DIR = Path("eval_results/issue_1773/register_steer")
FIG_DIR = Path("figures/issue_1773")
ARMS = ("register", "null_other", "null_random")
LABELS = {
    "register": "register_style\n(n=1233)",
    "null_other": "other interpretable\nfeatures (n=400)",
    "null_random": "random unit\ndirections (n=400)",
}
# one colour = one meaning across both panels: the tested arm vs its two nulls
COLORS = {"register": "#2166ac", "null_other": "#f4a582", "null_random": "#999999"}


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().split("\n") if x.strip()]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _boot_ci(x: np.ndarray, fn, n_boot: int = 10000, seed: int = 1773) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    draws = np.array([fn(x[i]) for i in idx])
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def main() -> int:
    feats = _read_jsonl(OUT_DIR / "per_feature.jsonl")
    val = json.loads((OUT_DIR / "validator.json").read_text())
    thr = val["mover_threshold_abs_shift"]

    by = {a: [f for f in feats if f["kind"] == a and f["usable"]] for a in ARMS}
    absshift = {a: np.array([abs(f["register_shift"]) for f in by[a]]) for a in ARMS}
    signed = {a: np.array([f["register_shift"] for f in by[a]]) for a in ARMS}

    out: dict = {
        "what": "#1773 register steering-transfer validator -- uncertainty + contrasts",
        "alpha": val["alpha"],
        "mover_threshold_abs_shift": thr,
        "arms": {},
        "contrasts": {},
    }
    for a in ARMS:
        x, s = absshift[a], signed[a]
        n = len(x)
        k = int((x > thr).sum())
        out["arms"][a] = {
            "n": n,
            "abs_shift_mean": round(float(x.mean()), 4),
            "abs_shift_mean_ci95": [round(v, 4) for v in _boot_ci(x, np.mean)],
            "abs_shift_median": round(float(np.median(x)), 4),
            "signed_shift_mean": round(float(s.mean()), 4),
            "signed_shift_mean_ci95": [round(v, 4) for v in _boot_ci(s, np.mean)],
            "mover_rate": round(k / n, 4),
            "mover_rate_ci95": [round(v, 4) for v in _wilson(k, n)],
            "coherent_rate_mean": round(
                float(
                    np.mean([f["coherent_rate"] for f in by[a] if f["coherent_rate"] is not None])
                ),
                4,
            ),
        }

    for null in ("null_other", "null_random"):
        u, p = sps.mannwhitneyu(absshift["register"], absshift[null], alternative="greater")
        n1, n2 = len(absshift["register"]), len(absshift[null])
        kr = int((absshift["register"] > thr).sum())
        kn = int((absshift[null] > thr).sum())
        # 2x2 mover-rate contrast
        _, pf = sps.fisher_exact([[kr, n1 - kr], [kn, n2 - kn]], alternative="greater")
        out["contrasts"][f"register_vs_{null}"] = {
            "abs_shift_mean_diff": round(
                float(absshift["register"].mean() - absshift[null].mean()), 4
            ),
            "mannwhitney_u": float(u),
            "mannwhitney_p_greater": float(p),
            "rank_biserial": round(float(2 * u / (n1 * n2) - 1), 4),
            "mover_rate_diff": round(kr / n1 - kn / n2, 4),
            "fisher_p_greater": float(pf),
        }

    dc = val["direction_claim"]
    directional = [f for f in by["register"] if f["predicted_direction"]]
    k = sum(1 for f in directional if f["direction_match"])
    out["direction_claim"] = {
        "n_directional": len(directional),
        "match_rate": round(k / len(directional), 4),
        "match_rate_ci95": [round(v, 4) for v in _wilson(k, len(directional))],
        "shuffle_null": dc["shuffle_null"],
        "excess_over_null_mean": round(k / len(directional) - dc["shuffle_null"]["null_mean"], 4),
        "by_predicted_direction": {},
    }
    for name, val_ in (("formal", -1), ("informal", 1)):
        sub = [f for f in directional if f["predicted_direction"] == val_]
        if not sub:
            continue
        ks = sum(1 for f in sub if f["direction_match"])
        out["direction_claim"]["by_predicted_direction"][name] = {
            "n": len(sub),
            "match_rate": round(ks / len(sub), 4),
            "match_rate_ci95": [round(v, 4) for v in _wilson(ks, len(sub))],
            "mean_signed_shift": round(float(np.mean([f["register_shift"] for f in sub])), 4),
        }

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        sha = "unknown"
    out["git_commit"] = sha
    out["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (OUT_DIR / "validator_stats.json").write_text(json.dumps(out, indent=1))

    # ---- figure -----------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # LEFT: the underlying per-direction data (every direction is one point),
    # with the arm mean + bootstrap CI drawn over it.
    rng = np.random.default_rng(0)
    for i, a in enumerate(ARMS):
        x = absshift[a]
        jitter = rng.normal(0, 0.075, size=len(x))
        axL.scatter(
            np.full(len(x), i) + jitter, x, s=5, alpha=0.22, color=COLORS[a], linewidths=0, zorder=2
        )
        m = float(x.mean())
        lo, hi = out["arms"][a]["abs_shift_mean_ci95"]
        axL.errorbar(
            i,
            m,
            yerr=[[max(0, m - lo)], [max(0, hi - m)]],
            fmt="o",
            color="black",
            markersize=7,
            capsize=5,
            lw=1.8,
            zorder=4,
        )
    axL.axhline(thr, color="#b2182b", ls="--", lw=1.4, zorder=3)
    axL.text(
        2.42,
        thr,
        f" mover threshold\n {thr:.2f}",
        color="#b2182b",
        va="center",
        ha="left",
        fontsize=8.5,
    )
    axL.set_xticks(range(len(ARMS)))
    axL.set_xticklabels([LABELS[a] for a in ARMS], fontsize=9)
    axL.set_ylabel("|register shift| vs unsteered baseline\n(0-100 register scale)")
    axL.set_title(
        "Per-direction register movement\n(each point = one steering direction)", fontsize=10.5
    )
    axL.set_ylim(0, min(12, float(np.percentile(np.concatenate(list(absshift.values())), 99.7))))
    axL.set_xlim(-0.5, 2.9)
    axL.grid(axis="y", alpha=0.25)

    # RIGHT: the direction claim against its label-shuffle null
    nullm = dc["shuffle_null"]["null_mean"]
    nullp95 = dc["shuffle_null"]["null_p95"]
    mr = out["direction_claim"]["match_rate"]
    lo, hi = out["direction_claim"]["match_rate_ci95"]
    axR.axhspan(0, nullp95, color="#999999", alpha=0.18, zorder=1)
    axR.axhline(nullm, color="#999999", ls="-", lw=1.5, zorder=2)
    axR.axhline(nullp95, color="#999999", ls="--", lw=1.2, zorder=2)
    axR.errorbar(
        0,
        mr,
        yerr=[[max(0, mr - lo)], [max(0, hi - mr)]],
        fmt="o",
        color=COLORS["register"],
        markersize=9,
        capsize=6,
        lw=2,
        zorder=4,
    )
    labs = []
    for j, (name, d) in enumerate(
        out["direction_claim"]["by_predicted_direction"].items(), start=1
    ):
        l2, h2 = d["match_rate_ci95"]
        axR.errorbar(
            j,
            d["match_rate"],
            yerr=[[max(0, d["match_rate"] - l2)], [max(0, h2 - d["match_rate"])]],
            fmt="s",
            color=COLORS["register"],
            alpha=0.65,
            markersize=7,
            capsize=5,
            lw=1.6,
            zorder=4,
        )
        labs.append(f"predicted\n{name}\n(n={d['n']})")
    axR.set_xticks(range(1 + len(labs)))
    axR.set_xticklabels(
        [f"all directional\n(n={out['direction_claim']['n_directional']})", *labs], fontsize=9
    )
    axR.text(
        2.45,
        nullm,
        f" shuffle null mean {nullm:.3f}\n (shaded: below null p95)",
        color="#666666",
        va="center",
        ha="left",
        fontsize=8.5,
    )
    axR.set_ylabel("fraction whose shift SIGN matches\nthe feature description's prediction")
    axR.set_title("Direction claim vs label-shuffle null\n(10,000 permutations)", fontsize=10.5)
    # accommodate every plotted CI (the informal subgroup sits far above the null)
    hi_all = [hi] + [
        d["match_rate_ci95"][1] for d in out["direction_claim"]["by_predicted_direction"].values()
    ]
    lo_all = [lo] + [
        d["match_rate_ci95"][0] for d in out["direction_claim"]["by_predicted_direction"].values()
    ]
    axR.set_ylim(min(0.44, min(lo_all) - 0.03), max(hi_all) + 0.04)
    axR.set_xlim(-0.5, 2.9)
    axR.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"#1773 register axis: zero-shot steering transfer at alpha={val['alpha']} "
        f"(coherence-gated; 8 prompts x 3 draws per direction)",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = FIG_DIR / "register_steer_transfer.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print(f"[stats] wrote {OUT_DIR / 'validator_stats.json'} and {p}")
    print(json.dumps(out["contrasts"], indent=1))
    print(json.dumps(out["direction_claim"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
