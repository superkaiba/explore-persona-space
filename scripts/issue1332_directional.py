"""Issue #1332 free-analysis follow-up — DIRECTIONAL transfer predictor, registered inference.

The main battery (``issue1332_analysis.py``) ran registered inference on the
SYMMETRIC predictor ``S_sym = (S_trans + S_trans.T)/2``; the directional
matrix ``S_trans(i->j)`` (held-out R^2 of family i's ridge map predicting
family j's mean-response targets — orientation matches ``L[i, j]`` = leakage
source i -> target j) only got a point-estimate free read (rho = 0.67,
``analyzer_free_reads.json``). This script runs the SAME registered battery on
the directional predictor at the frozen L*:

- raw Spearman rho(S_trans, L) over the 400 off-diagonal cells, two-way
  cluster bootstrap 95% CI (B=2,000, seed 0) — the three #474
  stylized-exclusion panels;
- target-permutation null p (B=10,000, seed 1; single-layer at the FROZEN L*
  — the layer was frozen by the parent's registered procedure BEFORE any
  directional read, so no per-layer selection axis exists for this predictor);
- the registered kill covariates: partial rho(S_trans, L | cos_532, JS_540)
  with its own bootstrap CI (+ single-covariate rows);
- the direct increment read: partial rho(S_trans, L | S_sym) with CI (does
  direction carry signal beyond the symmetric component?), plus the reverse
  supplementary partial rho(S_sym, L | S_trans).

Outputs ``eval_results/issue_1332/directional_inference.json`` (NEW file —
never overwrites a committed artifact) + one forest figure.

USAGE
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \\
      uv run python scripts/issue1332_directional.py
    (scale down with --n-boot / --n-null; identical code path either way)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1332_common as C
from issue1332_analysis import (
    BOOT_SEED,
    N_BOOT_DEFAULT,
    N_NULL_DEFAULT,
    NULL_SEED,
    load_similarity,
    partial_spearman,
    shuffled_pairing_null,
    spearman,
    sub_matrix,
    two_way_cluster_bootstrap,
)

logger = logging.getLogger("issue1332.directional")


def stylized_panels(mask, sources: list[str], targets: list[str]) -> dict:
    """The three #474 stylized-exclusion panels (mirrors issue1332_analysis main)."""
    import numpy as np

    src_excl = ~np.isin(
        np.arange(len(sources)), [sources.index(c) for c in C.STYLIZED_CIDS if c in sources]
    )[:, None]
    tgt_excl = ~np.isin(
        np.arange(len(targets)), [targets.index(c) for c in C.STYLIZED_CIDS if c in targets]
    )[None, :]
    return {
        "all": mask,
        "stylized_excluded_as_source": mask & src_excl,
        "stylized_excluded_either_side": mask & src_excl & tgt_excl,
    }


def make_forest_figure(fig_dir: Path, rows: list[tuple[str, float, float, float]]) -> str:
    """Directional-battery forest (paper style); xerr clamped >= 0 per errorbar site."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception:
        logger.warning("paper style unavailable; default rcParams")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
    ys = np.arange(len(rows))[::-1]
    for y, (label, val, lo, hi) in zip(ys, rows, strict=True):
        ax.errorbar(
            [val],
            [y],
            xerr=[[max(0.0, val - lo)], [max(0.0, hi - val)]],
            fmt="o",
            color="#0072B2",
        )
        ax.text(-0.02, y, label, ha="right", va="center", transform=ax.get_yaxis_transform())
    ax.axvline(0.0, color="0.5", lw=0.8)
    ax.set_yticks([])
    ax.set_xlabel("Spearman rho (cluster-bootstrap 95% CI)")
    ax.set_title("Directional transfer predictor — raw + partial reads")
    p = fig_dir / "directional_forest.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return str(p)


def main() -> int:
    """Directional-predictor registered battery: join S_trans with L, write JSON + forest."""
    ap = argparse.ArgumentParser(description="Issue #1332 directional-predictor inference (VM CPU)")
    ap.add_argument("--results-dir", default=None, help="override eval_results/issue_1332")
    ap.add_argument("--n-null", type=int, default=N_NULL_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import numpy as np

    res_dir = C.results_dir(False, args.results_dir)
    fig_dir = C.figures_dir(False)

    C.phase("dir_load")
    freeze = json.loads((res_dir / "layer_freeze.json").read_text())
    l_star = freeze["l_star"]
    sim = load_similarity(res_dir, l_star)
    families = sim["families"]
    leak = C.load_leakage_matrices()
    all_sources, all_targets = leak["sources"], leak["targets"]
    sources = [s for s in all_sources if s in families]
    targets = [t for t in all_targets if t in families]
    si = {s: i for i, s in enumerate(all_sources)}
    ti = {t: i for i, t in enumerate(all_targets)}
    L = leak["L"][np.ix_([si[s] for s in sources], [ti[t] for t in targets])]
    mask = C.offdiag_mask(sources, targets)

    S_dir = sub_matrix(sim["S_trans"], families, sources, targets)
    S_sym = sub_matrix(sim["S_sym"], families, sources, targets)
    base = C.load_baseline_matrices()
    rows_ix = np.ix_([si[s] for s in sources], [ti[t] for t in targets])
    cos532 = base["cosine_532"][rows_ix]
    js540 = base["js_rb_540"][rows_ix]

    C.phase("dir_headline")
    panels = stylized_panels(mask, sources, targets)

    def boot_rho(M_pred, m):
        def _stat(sidx, tidx):
            sm = m[np.ix_(sidx, tidx)]
            return spearman(M_pred[np.ix_(sidx, tidx)][sm], L[np.ix_(sidx, tidx)][sm])

        return two_way_cluster_bootstrap(
            _stat, len(sources), len(targets), n_boot=args.n_boot, seed=BOOT_SEED
        )

    headline = {
        pname: {
            "n_cells": int(m.sum()),
            "rho": spearman(S_dir[m], L[m]),
            "boot": boot_rho(S_dir, m),
        }
        for pname, m in panels.items()
    }

    C.phase("dir_null")
    m = panels["all"]
    null_dir = shuffled_pairing_null(
        S_dir, L, m, n_draws=args.n_null, seed=NULL_SEED, axis="target"
    )
    null_src = shuffled_pairing_null(
        S_dir, L, m, n_draws=args.n_null, seed=NULL_SEED + 1, axis="source"
    )
    null = {
        "n_draws": args.n_null,
        "p_two_sided_lstar": float((np.abs(null_dir) >= abs(headline["all"]["rho"])).mean()),
        "p975_abs_rho_lstar": float(np.quantile(np.abs(null_dir), 0.975)),
        "source_axis_p975_abs": float(np.quantile(np.abs(null_src), 0.975)),
        "note": "single-layer null at the FROZEN L* (frozen by the parent's registered "
        "procedure before any directional read — no per-layer selection axis for this "
        "predictor); same target-permutation convention + seed as the symmetric battery",
    }

    C.phase("dir_kill")

    def boot_partial(M_pred, covs):
        def _stat(sidx, tidx):
            sm = m[np.ix_(sidx, tidx)]
            return partial_spearman(
                M_pred[np.ix_(sidx, tidx)][sm],
                L[np.ix_(sidx, tidx)][sm],
                [cv[np.ix_(sidx, tidx)][sm] for cv in covs],
            )

        return two_way_cluster_bootstrap(
            _stat, len(sources), len(targets), n_boot=args.n_boot, seed=BOOT_SEED
        )

    kill = {
        "partial_rho_Sdir_L_given_cos_js": partial_spearman(S_dir[m], L[m], [cos532[m], js540[m]]),
        "boot_partial_given_cos_js": boot_partial(S_dir, [cos532, js540]),
        "partial_rho_Sdir_L_given_cos": partial_spearman(S_dir[m], L[m], [cos532[m]]),
        "boot_partial_given_cos": boot_partial(S_dir, [cos532]),
        "partial_rho_Sdir_L_given_js": partial_spearman(S_dir[m], L[m], [js540[m]]),
        "boot_partial_given_js": boot_partial(S_dir, [js540]),
        "pearson_Sdir_cos": float(np.corrcoef(S_dir[m], cos532[m])[0, 1]),
    }
    # registered collinearity gate at 0.6 (mirrors issue1332_analysis kill block):
    # tercile + deg-2 residualization fallback reads when S_dir ~ cos are collinear
    if kill["pearson_Sdir_cos"] == kill["pearson_Sdir_cos"] and kill["pearson_Sdir_cos"] > 0.6:
        terciles = np.quantile(cos532[m], [1 / 3, 2 / 3])
        buckets = np.digitize(cos532[m], terciles)
        kill["collinearity_gate_fired"] = True
        kill["tercile_rho"] = {
            f"tercile_{b}": spearman(S_dir[m][buckets == b], L[m][buckets == b]) for b in (0, 1, 2)
        }
        cosv = cos532[m]
        Z = np.column_stack([cosv, cosv**2, np.ones(len(cosv))])
        bx, *_ = np.linalg.lstsq(Z, S_dir[m], rcond=None)
        by, *_ = np.linalg.lstsq(Z, L[m], rcond=None)
        kill["poly2_residualized_rho"] = spearman(S_dir[m] - Z @ bx, L[m] - Z @ by)
    else:
        kill["collinearity_gate_fired"] = False

    C.phase("dir_increment")
    increment = {
        "partial_rho_Sdir_L_given_Ssym": partial_spearman(S_dir[m], L[m], [S_sym[m]]),
        "boot_partial_given_Ssym": boot_partial(S_dir, [S_sym]),
        "supplementary_partial_rho_Ssym_L_given_Sdir": partial_spearman(S_sym[m], L[m], [S_dir[m]]),
        "supplementary_boot_Ssym_given_Sdir": boot_partial(S_sym, [S_dir]),
        "pearson_Sdir_Ssym": float(np.corrcoef(S_dir[m], S_sym[m])[0, 1]),
        "note": "S_sym = (S_trans + S_trans.T)/2 shares the symmetric component with "
        "S_trans by construction — read the increment partial against pearson_Sdir_Ssym",
    }

    # reference: the committed symmetric headline, for side-by-side forest rows
    sym_ref = json.loads((res_dir / "analysis.json").read_text())
    sym_all = sym_ref["headline"]["all"]

    C.phase("dir_figures")
    forest_rows = [
        (
            "S_sym raw rho (reference)",
            sym_all["rho"],
            sym_all["boot"]["ci_lo"],
            sym_all["boot"]["ci_hi"],
        ),
        (
            "S_dir raw rho",
            headline["all"]["rho"],
            headline["all"]["boot"]["ci_lo"],
            headline["all"]["boot"]["ci_hi"],
        ),
        (
            "S_dir partial | cos",
            kill["partial_rho_Sdir_L_given_cos"],
            kill["boot_partial_given_cos"]["ci_lo"],
            kill["boot_partial_given_cos"]["ci_hi"],
        ),
        (
            "S_dir partial | JS",
            kill["partial_rho_Sdir_L_given_js"],
            kill["boot_partial_given_js"]["ci_lo"],
            kill["boot_partial_given_js"]["ci_hi"],
        ),
        (
            "S_dir partial | cos+JS (KILL)",
            kill["partial_rho_Sdir_L_given_cos_js"],
            kill["boot_partial_given_cos_js"]["ci_lo"],
            kill["boot_partial_given_cos_js"]["ci_hi"],
        ),
        (
            "S_dir partial | S_sym (increment)",
            increment["partial_rho_Sdir_L_given_Ssym"],
            increment["boot_partial_given_Ssym"]["ci_lo"],
            increment["boot_partial_given_Ssym"]["ci_hi"],
        ),
    ]
    fig_path = make_forest_figure(fig_dir, forest_rows)

    out = {
        "l_star": l_star,
        "predictor": "S_trans directional (held-out R^2, source-map -> target cells; "
        "orientation S_trans[i, j] = transfer i->j, matching L[i, j])",
        "panel": {
            "sources": sources,
            "targets": targets,
            "n_offdiag_cells": int(mask.sum()),
        },
        "headline_directional": headline,
        "null": null,
        "kill": kill,
        "increment": increment,
        "reference_symmetric_headline_all": {
            "rho": sym_all["rho"],
            "ci_lo": sym_all["boot"]["ci_lo"],
            "ci_hi": sym_all["boot"]["ci_hi"],
        },
        "figures": [fig_path],
        "reproducibility_metadata": C.reproducibility_metadata(
            {"n_boot": args.n_boot, "n_null": args.n_null, "followup": "directional_inference"}
        ),
    }
    out_path = res_dir / "directional_inference.json"
    C.write_json_atomic(out_path, out)
    logger.info(
        "[directional] rho=%.4f CI=(%.4f, %.4f) null_p=%.5f partial|cos+JS=%.4f "
        "partial|S_sym=%.4f -> %s",
        headline["all"]["rho"],
        headline["all"]["boot"]["ci_lo"],
        headline["all"]["boot"]["ci_hi"],
        null["p_two_sided_lstar"],
        kill["partial_rho_Sdir_L_given_cos_js"],
        increment["partial_rho_Sdir_L_given_Ssym"],
        out_path,
    )
    C.phase("done_directional")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
