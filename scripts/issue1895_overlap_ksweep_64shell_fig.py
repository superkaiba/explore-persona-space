"""Render the subspace-overlap k-sweep against ONLY the tightest (64-shell) null.

Variant of issue1895_overlap_ksweep_fig.py: draws the finest variance-matched
rotation null alone — its median line plus the central-95% band of the 1,000
draws — with the observed overlap read against it. Interpretation carried in
the title: a point OUTSIDE the shaded band is inconsistent with the
variance-only null at the 5% level (above the band = more aligned than
variance-matched rotations produce in >97.5% of draws). This is a frequentist
band over null draws, NOT a posterior probability on the hypothesis.

Usage:
    uv run python scripts/issue1895_overlap_ksweep_64shell_fig.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import set_paper_style

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "eval_results/issue_1895/angles_summary.json"
OUTDIR = REPO / "figures/issue_1895"
PAIR = "psae_recon_pca"  # the on-distribution SAE subspace (reconstruction PCA)
SHELL = "64"  # tightest variance matching available in the banked nulls


def _pct(null: dict, key: str) -> float:
    """Percentile lookup tolerant of the two key spellings in the artifact."""
    for k in (key, key.replace(".", "_")):
        if k in null:
            return float(null[k])
    raise KeyError(f"{key} not in {sorted(null)}")


def load_cells(summary_path: Path, pair: str) -> list[dict]:
    payload = json.loads(summary_path.read_text())
    cells = [c for c in payload["cells"] if c["pair"] == pair and c.get("nulls")]
    assert cells, f"no null-bearing cells for pair={pair}"
    return sorted(cells, key=lambda c: c["k"])


def main() -> None:
    cells = load_cells(SUMMARY, PAIR)
    ks = [c["k"] for c in cells]
    observed = [float(c["observed_O"]) for c in cells]
    fine = [c["nulls"][SHELL] for c in cells]

    n_draws = {int(n["n_draws"]) for n in fine}
    assert len(n_draws) == 1, f"inconsistent null draw counts: {n_draws}"
    n_draw = n_draws.pop()

    band_lo = [_pct(n, "p2.5") for n in fine]
    band_hi = [_pct(n, "p97.5") for n in fine]

    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.fill_between(
        ks,
        band_lo,
        band_hi,
        color="0.75",
        alpha=0.55,
        linewidth=0,
        label=f"variance-matched null, central 95% of {n_draw:,} draws ({SHELL} shells)",
    )
    ax.plot(
        ks,
        [_pct(n, "p50") for n in fine],
        color="0.35",
        linestyle="--",
        linewidth=1.6,
        label=f"null median ({SHELL} shells)",
    )
    ax.plot(
        ks,
        observed,
        color="#0072B2",
        marker="o",
        markersize=5.5,
        linewidth=2.0,
        label="observed overlap",
        zorder=5,
    )

    for k, obs, n in zip(ks, observed, fine):
        ax.annotate(
            f"{_pct(n, 'q_percentile_of_observed'):.0f}%",
            (k, obs),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color="#0072B2",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("subspace size $k$ (top-$k$ directions of each ranking)")
    ax.set_ylabel(r"subspace overlap  $O(k)=\overline{\cos^2\theta}$")
    ax.set_title(
        "Map-predictable vs SAE-representable subspace overlap, tightest variance-matched null\n"
        "(outside the band = inconsistent with variance alone at the 5% level; "
        "above = more aligned than variance predicts)"
    )
    lo = min(min(band_lo), min(observed))
    hi = max(max(band_hi), max(observed))
    pad = 0.25 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2.2 * pad)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = OUTDIR / "overlap_ksweep_vs_null_64shell"
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    (stem.with_suffix(".meta.json")).write_text(
        json.dumps(
            {
                "source": str(SUMMARY.relative_to(REPO)),
                "pair": PAIR,
                "k": ks,
                "observed_O": observed,
                "null_draws": n_draw,
                "shell_setting": SHELL,
                "null_band_p2.5": band_lo,
                "null_band_p97.5": band_hi,
                "observed_percentile": [_pct(n, "q_percentile_of_observed") for n in fine],
                "what_is_plotted": (
                    "Per k: observed mean cos^2 principal angle between the map's "
                    "top-k predictable eigendirections and the SAE reconstruction-PCA "
                    "top-k subspace, against the FINEST (64-shell) variance-matched "
                    "Haar-rotation null only: null median (dashed) + central-95% band "
                    "of 1,000 draws. Labels = observed's percentile in that null. "
                    "A point outside the band is inconsistent with the variance-only "
                    "null at the 5% level (frequentist band over null draws, not a "
                    "posterior probability)."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {stem}.png / .pdf / .meta.json")


if __name__ == "__main__":
    main()
