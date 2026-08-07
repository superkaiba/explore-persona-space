#!/usr/bin/env python3
"""Issue #1336 — does generation truncation drive the context->answer map skill?

Round-5 QA follow-up. The on-policy naturalistic + chat generation pools carry
48,111 truncated-but-KEPT rows (``finish_reason == "length"``) out of 518,881
kept rows across both arms; the map DV pools activations over the model's OWN
answer, so a truncated answer is a corrupted measurement that still
contributes. This figure asks whether that corruption tracks the fitted map
skill, i.e. whether the truncation is load-bearing for the result.

WHAT IS PLOTTED, EXACTLY
  y  per-cell ``skill_over_mean`` at layer 30, read from
     ``eval_results/issue_1336/cells/cells_<model>_<format>_<corpus>.json``
     (held-out R2 skill of the ridge map RELATIVE to the mean baseline;
     y = 0 means the map does no better than predicting the pooled mean,
     y < 0 means worse).
  x  per-cell ``kept_truncation_rate``, read from that same cell's generation
     ``audit.json`` on the HF data repo — truncated rows as a fraction of
     KEPT rows (the rows that actually enter the fit), NOT of all generated
     rows. The two denominators differ substantially and are not
     interchangeable.

  LEFT  raw association, one point per cell, coloured by corpus.
  RIGHT the same cells with corpus MEANS REMOVED from both axes
        (corpus-demeaned residuals) -- the association that survives
        controlling for which corpus a cell came from.

  Raw and controlled views are shown together deliberately: the raw
  correlation is largely a between-corpus contrast (gsm8k cells have low
  truncation and positive skill; lmsys5k cells have higher truncation and
  strongly negative skill), so the raw number alone would overstate a
  truncation effect.

SCOPE LIMIT -- READ BEFORE USING THIS FIGURE
  Only 11 cells have BOTH a fitted DV and a generation audit, and all of them
  are WAVE-1 corpora (gsm8k_test1319, gsm8k_train5k, lmsys5k). The seven v2
  corpora -- where truncation is worst (per-cell kept-truncation up to 32.9%
  on base/math7500) -- have NO fits in eval_results/issue_1336/cells/ yet, so
  the high-truncation regime is UNTESTED here. Plotted truncation tops out at
  9.4%. This figure therefore cannot rule out a truncation effect where
  truncation is actually severe; it only shows that below ~9% the association
  is weak once corpus is controlled.

  The 5 naturalistic lmsys5k cells are excluded for a different reason: they
  are the matched-text wave-1 render, not generated under
  ``--gen-format naturalistic``, so no ``lmsys5k__gen_naturalistic`` pool (and
  hence no audit) exists for them.

Data pinned below so the figure reproduces without HF access; regenerate the
join with the loader in the round-5 session notes if the fits are extended.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE any heavy import — shared-VM thread caps (#847)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (model, format, corpus, skill_over_mean@L30, kept_truncation_rate, n_kept)
CELLS: list[tuple[str, str, str, float, float, int]] = [
    ("base", "chat", "gsm8k_test1319", -0.7091, 0.0944, 1293),
    ("rlvr", "chat", "lmsys5k", -0.9332, 0.0683, 3629),
    ("rlvr_long", "chat", "lmsys5k", -0.6314, 0.0415, 3208),
    ("sft", "chat", "lmsys5k", -0.9602, 0.0161, 3544),
    ("rlvr_long", "chat", "gsm8k_test1319", 0.0525, 0.0099, 1319),
    ("rlvr_long", "chat", "gsm8k_train5k", 0.4840, 0.0072, 4998),
    ("rlvr", "chat", "gsm8k_train5k", 0.4581, 0.0052, 5000),
    ("dpo", "chat", "gsm8k_test1319", 0.0145, 0.0038, 1319),
    ("rlvr", "chat", "gsm8k_test1319", -0.0057, 0.0038, 1319),
    ("sft", "chat", "gsm8k_test1319", 0.0107, 0.0030, 1319),
    ("sft", "chat", "gsm8k_train5k", 0.4670, 0.0014, 5000),
]

# One colour per corpus, identical across BOTH panels (colour == corpus,
# everywhere in this figure). Okabe-Ito, colourblind-safe.
CORPUS_COLOR = {
    "gsm8k_test1319": "#0072B2",
    "gsm8k_train5k": "#009E73",
    "lmsys5k": "#E69F00",
}

# Cross-arm context for the caption: kept-truncation over ALL cells, both arms.
TOTAL_KEPT = 518_881
TOTAL_TRUNC_KEPT = 48_111
V2_WORST_CELL = ("base/math7500", 0.3287)


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho with midrank ties."""

    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def corpus_demeaned(
    cells: list[tuple[str, str, str, float, float, int]],
) -> list[tuple[str, float, float]]:
    """Subtract each corpus's mean from both axes -> corpus-controlled view."""
    by: dict[str, list[tuple]] = {}
    for c in cells:
        by.setdefault(c[2], []).append(c)
    out: list[tuple[str, float, float]] = []
    for corpus, group in by.items():
        mt = statistics.fmean([g[4] for g in group])
        ms = statistics.fmean([g[3] for g in group])
        for g in group:
            out.append((corpus, g[4] - mt, g[3] - ms))
    return out


def main() -> None:
    out_dir = Path("figures/issue_1336")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.9))

    # ---- LEFT: raw ----
    xs = [c[4] * 100 for c in CELLS]
    ys = [c[3] for c in CELLS]
    rho_raw = spearman([c[4] for c in CELLS], [c[3] for c in CELLS])
    for corpus, color in CORPUS_COLOR.items():
        pts = [(c[4] * 100, c[3]) for c in CELLS if c[2] == corpus]
        if not pts:
            continue
        axL.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            s=64,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=f"{corpus} (n={len(pts)})",
            zorder=3,
        )
    axL.axhline(0.0, color="0.55", linewidth=0.9, linestyle="--", zorder=1)
    axL.set_xlabel("kept-row truncation rate (%)")
    axL.set_ylabel("map skill over mean baseline, layer 30")
    axL.set_title(f"Raw:  Spearman $\\rho$ = {rho_raw:+.2f}  (n={len(CELLS)})")
    # Upper right is the empty quadrant here (high-truncation cells all sit at
    # negative skill); lower left collides with the sft/lmsys5k point.
    axL.legend(loc="upper right", fontsize=8.5)

    # ---- RIGHT: corpus-controlled ----
    resid = corpus_demeaned(CELLS)
    rho_ctl = spearman([r[1] for r in resid], [r[2] for r in resid])
    for corpus, color in CORPUS_COLOR.items():
        pts = [(r[1] * 100, r[2]) for r in resid if r[0] == corpus]
        if not pts:
            continue
        axR.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            s=64,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    axR.axhline(0.0, color="0.55", linewidth=0.9, linestyle="--", zorder=1)
    axR.axvline(0.0, color="0.55", linewidth=0.9, linestyle="--", zorder=1)
    axR.set_xlabel("truncation rate, corpus mean removed (pp)")
    axR.set_ylabel("map skill, corpus mean removed")
    axR.set_title(f"Corpus-controlled:  Spearman $\\rho$ = {rho_ctl:+.2f}")

    fig.tight_layout()
    dest = out_dir / "truncation_control.png"
    fig.savefig(dest, bbox_inches="tight")
    print(f"[fig] wrote {dest}")
    print(f"[fig] raw rho={rho_raw:+.3f}  corpus-controlled rho={rho_ctl:+.3f}")
    print(
        f"[fig] scope: {len(CELLS)} wave-1 cells; plotted truncation "
        f"{min(xs):.2f}%..{max(xs):.2f}%; v2 corpora (worst "
        f"{V2_WORST_CELL[0]} at {V2_WORST_CELL[1]:.1%}) have no fits yet"
    )
    print(
        f"[fig] cross-arm context: {TOTAL_TRUNC_KEPT:,} / {TOTAL_KEPT:,} kept "
        f"rows truncated ({TOTAL_TRUNC_KEPT / TOTAL_KEPT:.2%})"
    )


if __name__ == "__main__":
    main()
