#!/usr/bin/env python
"""Issue #1482: does the provisional #1738 R^2 target stand in for the real one?

Every figure on the label/predictor line was produced against the #1738 SAE->SAE
MULTI-TURN per-feature R^2 while the real target -- the #1482 dense-context ->
SAE-answer map -- was still being fit. That stand-in carried a PROVISIONAL caveat
justified by an inter-arm agreement measured at PANEL width (16,384 features).

The real target has now landed at full width, so the substitution is directly
measurable on the real pair. This script reports, over the SHARED scored
universe:

  * Spearman rho(old #1738 SAE->SAE, new #1482 dense->SAE ridge) at FULL width;
  * the same restricted to the 16,384-feature panel, which is the like-for-like
    comparison against the panel-width figure the caveat cited;
  * rho within each activity decile, because a stand-in that agrees on the
    active head and diverges on the low-activity tail is a different problem
    from one that degrades uniformly -- and the tail is ~105k of the features;
  * rho(new ridge, new MLP) as the inter-ARM (not inter-CORPUS) reference, i.e.
    how much two honest maps of the SAME corpus disagree, which bounds how much
    of the old-vs-new gap can be blamed on the target rather than the corpus;
  * the reproduction check that establishes corpus identity: the full-width
    ridge against the banked #1482 dense->SAE PANEL per-feature R^2, plus that
    banked file's own `activity` array against the covariate this line joins on.

Membership always honours the producer's `scored` flag; a zero-variance holdout
column is UNSCORED, which is a different statement from a NaN value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps BEFORE numpy (shared-VM discipline)

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue1773_common as C1773  # noqa: E402

DICT_SIZE = 131_072
N_DECILES = 10

OLD_PATH = "eval_results/issue_1738/sae_twoway/perfeature/sae_context_r2.npy"
OLD_LABEL = "#1738 SAE->SAE multi-turn context R^2 (the provisional stand-in)"
NEW_RIDGE = "data/issue_1482/densesae_dl/ridge__mean_perfeature.npz"
NEW_MLP = "data/issue_1482/densesae_dl/mlp__mean_perfeature.npz"
BANKED_PANEL = "data/issue_1482/densesae_dl/sae_dense_in__mean__ridge.npz"
COVARIATES = "eval_results/issue_1482/predictor_battery/fullwidth_covariates.npz"

OUT_DIR = "eval_results/issue_1482/target_comparison"
FIG_DIR = "figures/issue_1482/target_comparison"


def _log(msg: str) -> None:
    print(f"[target-cmp] {msg}", flush=True)


def _load_npz_target(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(r2, scored) scattered to DICT_SIZE width by the file's own feat_ids."""
    z = np.load(path)
    fid = np.asarray(z["feat_ids"], dtype=np.int64)
    r2 = np.full(DICT_SIZE, np.nan, dtype=np.float64)
    sc = np.zeros(DICT_SIZE, dtype=bool)
    r2[fid] = np.asarray(z["r2"], dtype=np.float64)
    sc[fid] = np.asarray(z["scored"], dtype=bool)
    return r2, sc


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(a) < 3:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--fig-dir", default=FIG_DIR)
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        import matplotlib  # noqa: F401
        from scipy.stats import spearmanr  # noqa: F401

        from explore_persona_space.analysis.paper_plots import (  # noqa: F401
            paper_palette,
            paper_palette_role,
            savefig_paper,
            set_paper_style,
        )

        print("import-check OK")
        sys.exit(0)

    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    old = np.asarray(np.load(REPO / OLD_PATH), dtype=np.float64)
    new, sc = _load_npz_target(REPO / NEW_RIDGE)
    mlp, sc_m = _load_npz_target(REPO / NEW_MLP)
    act = np.asarray(np.load(REPO / COVARIATES)["activity"], dtype=np.float64)

    # ---- corpus identity -----------------------------------------------------
    zb = np.load(REPO / BANKED_PANEL)
    pid = np.asarray(zb["feat_ids"], dtype=np.int64)
    pr2 = np.asarray(zb["r2"], dtype=np.float64)
    pact = np.asarray(zb["activity"], dtype=np.float64)
    ok = np.isfinite(pr2) & sc[pid]
    d_r2 = np.abs(new[pid[ok]] - pr2[ok])
    d_act = np.abs(pact - act[pid])
    provenance = {
        "n_panel_features_compared": int(ok.sum()),
        "max_abs_delta_r2_vs_banked_panel": float(d_r2.max()),
        "median_abs_delta_r2_vs_banked_panel": float(np.median(d_r2)),
        "max_abs_delta_activity_vs_covariates": float(np.nanmax(d_act)),
        "verdict": (
            "SAME corpus and recipe — the full-width ridge reproduces the banked "
            "#1482 dense->SAE panel per-feature R^2, and that banked file's own "
            "activity array is bit-identical to the covariate this line joins on"
            if d_r2.max() < 1e-6 and np.nanmax(d_act) == 0.0
            else "NOT reproduced — treat the corpus relationship as unverified"
        ),
    }
    _log(
        f"provenance: max|dR2| {d_r2.max():.3e} over {int(ok.sum())} panel features; "
        f"max|d activity| {np.nanmax(d_act):.3e}"
    )

    # ---- agreement -----------------------------------------------------------
    shared = sc & np.isfinite(old)
    panel_mask = np.zeros(DICT_SIZE, dtype=bool)
    panel_mask[pid] = True

    rows: dict[str, dict] = {}
    for name, mask in (
        ("full_width_shared", shared),
        ("panel_16384", shared & panel_mask),
        ("off_panel_tail", shared & ~panel_mask),
    ):
        rows[name] = {
            "n": int(mask.sum()),
            "spearman_old_vs_new_ridge": _spearman(old[mask], new[mask]),
            "median_old": float(np.median(old[mask])),
            "median_new_ridge": float(np.median(new[mask])),
        }
        _log(
            f"{name}: n={rows[name]['n']} rho={rows[name]['spearman_old_vs_new_ridge']:+.4f} "
            f"(median old {rows[name]['median_old']:+.5f} vs new {rows[name]['median_new_ridge']:+.5f})"
        )

    inter_arm = {
        "n": int((sc & sc_m).sum()),
        "spearman_new_ridge_vs_new_mlp": _spearman(new[sc & sc_m], mlp[sc & sc_m]),
        "note": (
            "inter-ARM agreement on the SAME corpus — the reference against which the "
            "old-vs-new (inter-CORPUS + inter-map) number should be read"
        ),
    }
    _log(f"inter-arm ridge vs mlp: rho={inter_arm['spearman_new_ridge_vs_new_mlp']:+.4f}")

    # ---- agreement by activity decile ---------------------------------------
    a_sh = act[shared]
    edges = np.quantile(a_sh, np.linspace(0, 1, N_DECILES + 1)[1:-1])
    dec = np.searchsorted(edges, a_sh, side="right")
    o_sh, n_sh, m_sh = old[shared], new[shared], mlp[shared]
    by_decile = []
    for i in range(N_DECILES):
        s = dec == i
        by_decile.append(
            {
                "decile": i + 1,
                "n": int(s.sum()),
                "activity_min": float(a_sh[s].min()),
                "activity_max": float(a_sh[s].max()),
                "spearman_old_vs_new_ridge": _spearman(o_sh[s], n_sh[s]),
                "spearman_new_ridge_vs_new_mlp": _spearman(n_sh[s], m_sh[s]),
                "median_new_ridge": float(np.median(n_sh[s])),
            }
        )
    _log(
        "rho by activity decile (old vs new): "
        + ", ".join(f"d{r['decile']}={r['spearman_old_vs_new_ridge']:+.3f}" for r in by_decile)
    )

    result = {
        "question": (
            "does the provisional #1738 SAE->SAE stand-in rank features the same way "
            "as the real #1482 dense->SAE target?"
        ),
        "targets": {
            "old": {"path": OLD_PATH, "label": OLD_LABEL},
            "new_ridge": {"path": NEW_RIDGE, "label": "#1482 dense->SAE ridge, mean pooling"},
            "new_mlp": {"path": NEW_MLP, "label": "#1482 dense->SAE MLP, mean pooling"},
        },
        "membership_rule": "producer `scored` flag on the new targets; finite R^2 on the old .npy",
        "corpus_provenance": provenance,
        "agreement": rows,
        "inter_arm_reference": inter_arm,
        "by_activity_decile": by_decile,
        "metadata": C1773.repro_meta(),
    }
    out = out_dir / "target_comparison.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    # ---- figure --------------------------------------------------------------
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    c_old, c_arm = paper_palette(2)
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.0))

    ax = axs[0]
    names = ["full_width_shared", "panel_16384", "off_panel_tail"]
    pretty = [
        "full width\n(shared universe)",
        "16,384-feature\npanel",
        "off-panel tail\n(the added ~97k)",
    ]
    xs = np.arange(len(names), dtype=float)
    ax.bar(
        xs,
        [rows[n]["spearman_old_vs_new_ridge"] for n in names],
        width=0.56,
        color=c_old,
        label="old #1738 stand-in vs new #1482 ridge",
    )
    ax.axhline(
        inter_arm["spearman_new_ridge_vs_new_mlp"],
        color=c_arm,
        lw=1.6,
        ls="--",
        label="reference: new ridge vs new MLP (same corpus)",
    )
    for x, n in zip(xs, names, strict=True):
        ax.annotate(
            f"{rows[n]['spearman_old_vs_new_ridge']:.3f}\nn={rows[n]['n']:,}",
            xy=(x, rows[n]["spearman_old_vs_new_ridge"]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7.4,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(pretty, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Spearman rho of per-feature R-squared")
    ax.set_title("(a) How well the provisional target stood in", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.4, loc="lower left")

    ax2 = axs[1]
    dx = np.arange(1, N_DECILES + 1, dtype=float)
    ax2.plot(
        dx,
        [r["spearman_old_vs_new_ridge"] for r in by_decile],
        "o-",
        ms=4.2,
        lw=1.5,
        color=c_old,
        label="old #1738 vs new #1482 ridge",
    )
    ax2.plot(
        dx,
        [r["spearman_new_ridge_vs_new_mlp"] for r in by_decile],
        "s--",
        ms=4.0,
        lw=1.4,
        color=c_arm,
        label="new ridge vs new MLP (same corpus)",
    )
    ax2.axhline(0.0, color=paper_palette_role("neutral"), lw=0.9, ls=":")
    ax2.set_xticks(dx)
    ax2.set_ylim(-0.05, 1.0)
    ax2.set_xlabel("activity decile of the shared universe (1 = least active)")
    ax2.set_ylabel("Spearman rho within the decile")
    ax2.set_title("(b) Where the stand-in diverges", loc="left", fontsize=10)
    ax2.legend(frameon=False, fontsize=7.4, loc="lower right")

    fig.text(
        0.005,
        0.004,
        "both #1482 targets are the SINGLE-TURN corpus; the #1738 stand-in is the MULTI-TURN read "
        "— so (a) mixes a corpus change with a map change, and the dashed line isolates the map part",
        fontsize=6.4,
        color="#555555",
        ha="left",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    paths = savefig_paper(fig, "target_comparison", dir=fig_dir)
    plt.close(fig)

    meta_path = Path(paths["meta"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["issue1482_target_comparison"] = {
        "what_is_plotted": (
            "(a) Spearman rho between the provisional #1738 SAE->SAE per-feature R^2 and the "
            "real #1482 dense->SAE ridge target, over the shared scored universe, over the "
            "16,384-feature panel alone, and over the off-panel tail; the dashed line is the "
            "inter-ARM rho between the two new maps of the SAME corpus. (b) the same two "
            "quantities within each activity decile of the shared universe."
        ),
        "definitions": {
            "shared universe": "features SCORED on the new target AND finite on the old one",
            "panel": "the 16,384 features the #1482 panel-width work was fit on",
            "inter-arm reference": "rho(new ridge, new MLP) — two maps, one corpus",
        },
        "caveats": [
            "the old-vs-new comparison changes BOTH the corpus (multi-turn -> single-turn) and "
            "the map (SAE->SAE -> dense->SAE); the inter-arm line separates the map component",
            "rho is rank-based, so the MLP arm's heavy negative tail does not affect it",
        ],
        "agreement": rows,
        "inter_arm_reference": inter_arm,
        "corpus_provenance": provenance,
        "source_paths": {"old": OLD_PATH, "new_ridge": NEW_RIDGE, "new_mlp": NEW_MLP},
    }
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    _log(f"wrote {out} and {paths['png']}")


if __name__ == "__main__":
    main()
