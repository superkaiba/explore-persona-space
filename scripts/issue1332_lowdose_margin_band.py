"""Issue #1332 low-dose revision round 2: margin-DV permutation band + figure refresh.

Interpretation-critic finding 1(c) (fold `lowdose-grid-kill-battery`, round 1):
the secondary marker-vs-EOS margin DV read rho = 0.470 on the low-dose grid —
higher than the registered log-prob read (0.257) and reversing the parent's
space ordering — but carried no permutation band. This script adjudicates it
with the SAME batched machinery the registered null used
(`shuffled_pairing_null`, target-label permutation, B = 10,000, seed 1,
abs p97.5), writes the result into
``eval_results/issue_1332/lowdose/analysis.json`` under
``sensitivity.margin_band``, renders the per-cell margin scatter
(`lowdose_margin_scatter.png`), and re-renders the band-trajectories figure
with plain-English source-family labels (finding 4).

VM CPU, seconds; run with the shared-VM thread-cap prefix.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1332_common as C
from issue1332_analysis import (
    N_NULL_DEFAULT,
    NULL_SEED,
    load_similarity,
    shuffled_pairing_null,
    spearman,
    sub_matrix,
)
from issue1332_lowdose_analysis import (
    FROZEN_LAYER,
    _paper_style,
    load_lowdose_leakage,
    trajectories_figure,
)

logger = logging.getLogger("issue1332.lowdose_margin_band")


def margin_scatter_figure(fig_dir: Path, S_dir, L_margin, mask, sources, targets) -> str:
    """Per-cell scatter of directional similarity vs the margin DV (stylized orange)."""
    _paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir.mkdir(parents=True, exist_ok=True)
    styl = set(C.STYLIZED_CIDS)
    colors = np.array(
        [
            "#D55E00" if (sources[i] in styl or targets[j] in styl) else "#0072B2"
            for i in range(len(sources))
            for j in range(len(targets))
            if mask[i, j]
        ]
    )
    fig, ax = plt.subplots(figsize=(4.8, 4), layout="constrained")
    ax.scatter(S_dir[mask], L_margin[mask], s=12, c=colors, alpha=0.7)
    ax.set_xlabel("S_dir (held-out transfer R^2, i->j)")
    ax.set_ylabel("marker-vs-EOS margin delta (trained - base)")
    ax.set_title("low-dose margin DV vs directional similarity")
    p = fig_dir / "lowdose_margin_scatter.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return str(p)


def main() -> int:
    """Compute the margin-DV permutation band, update analysis.json, refresh figures."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import numpy as np

    res_in = C.results_dir(False)
    lowdose_in = res_in / "lowdose"
    fig_dir = C.figures_dir(False) / "lowdose"

    freeze = json.loads((res_in / "layer_freeze.json").read_text())
    assert freeze["l_star"] == FROZEN_LAYER, freeze
    sim = load_similarity(res_in, FROZEN_LAYER)
    families = sim["families"]

    remeasured_dir = lowdose_in / "per_cell_base_remeasured"
    leak = load_lowdose_leakage(
        lowdose_in / "per_cell_trained", remeasured_dir if remeasured_dir.is_dir() else None
    )
    sources, targets = leak["sources"], leak["targets"]
    L_margin = leak["L_margin"]
    mask = C.offdiag_mask(sources, targets)

    S_dir = sub_matrix(sim["S_trans"], families, sources, targets)
    S_sym = sub_matrix(sim["S_sym"], families, sources, targets)

    ana_path = lowdose_in / "analysis.json"
    ana = json.loads(ana_path.read_text())

    rho_dir = spearman(S_dir[mask], L_margin[mask])
    rho_sym = spearman(S_sym[mask], L_margin[mask])
    stored_dir = ana["sensitivity"]["rho_dir_margin_dv"]
    assert abs(rho_dir - stored_dir) < 1e-9, (rho_dir, stored_dir)

    null_dir = shuffled_pairing_null(
        S_dir, L_margin, mask, n_draws=N_NULL_DEFAULT, seed=NULL_SEED, axis="target"
    )
    null_sym = shuffled_pairing_null(
        S_sym, L_margin, mask, n_draws=N_NULL_DEFAULT, seed=NULL_SEED, axis="target"
    )
    band = {
        "dv": "trained - base mean_marker_eos_margin (secondary margin DV)",
        "null": "target-family-label permutation of S (source-preserving), same machinery "
        "as the registered log-prob null (shuffled_pairing_null)",
        "n_draws": N_NULL_DEFAULT,
        "seed": NULL_SEED,
        "rho_dir_margin_dv": rho_dir,
        "dir_p975_abs_rho": float(np.quantile(np.abs(null_dir), 0.975)),
        "dir_p_two_sided": float((np.abs(null_dir) >= abs(rho_dir)).mean()),
        "dir_clears_band": bool(abs(rho_dir) > float(np.quantile(np.abs(null_dir), 0.975))),
        "rho_sym_margin_dv": rho_sym,
        "sym_p975_abs_rho": float(np.quantile(np.abs(null_sym), 0.975)),
        "sym_p_two_sided": float((np.abs(null_sym) >= abs(rho_sym)).mean()),
    }
    ana["sensitivity"]["margin_band"] = band
    logger.info(
        "[margin-band] dir rho=%.4f band=%.4f p=%.4f clears=%s | sym rho=%.4f band=%.4f p=%.4f",
        rho_dir,
        band["dir_p975_abs_rho"],
        band["dir_p_two_sided"],
        band["dir_clears_band"],
        rho_sym,
        band["sym_p975_abs_rho"],
        band["sym_p_two_sided"],
    )

    fig_margin = margin_scatter_figure(fig_dir, S_dir, L_margin, mask, sources, targets)
    fig_traj = trajectories_figure(fig_dir, lowdose_in / "band_trajectories")
    for f in (fig_margin, fig_traj):
        if f and f not in ana.get("figures", []):
            ana.setdefault("figures", []).append(f)
    C.write_json_atomic(ana_path, ana)
    logger.info("[done] %s updated; figures: %s, %s", ana_path, fig_margin, fig_traj)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
