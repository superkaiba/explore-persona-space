"""Regenerate the #1426 MLP nonlinearity-control hero figure with reader-facing labels.

Interpretation-critique round-1 finding 5 (task #1426): the driver-generated
``mlp_indiv_hero_4arm.png`` carried opaque short-code x-tick labels
(``lin D`` / ``MLP D`` / ``MLP G`` / ``lin G``) and rotated in-bar text that
overlapped the title. This script redraws the SAME four bars — identical data,
identical paired-bootstrap CIs (seed 42, 2,000 draws, shared index matrix) —
with self-describing two-line tick labels and no in-bar rotated text.

Inputs (all pinned):
- linear arms + identity ceiling: ``eval_results/issue_1426/decomp_indiv.pt``
- MLP arms: HF ``issue1426_cot_decomposition_r1llama/analysis_tensors/mlp_indiv/
  decomp_indiv_mlp.pt`` at revision ``c244377f2b`` (staged under
  ``data/issue_1426/hf_dl/``)
- reference values asserted against
  ``eval_results/issue_1426/indiv-mlp-nonlinearity-control/mlp_indiv_validity.json``

Output: ``figures/issue_1426/mlp_indiv_hero_4arm.{png,pdf,meta.json}`` (overwrites).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue928_common import BOOTSTRAP_SEED  # noqa: E402
from issue928_null_bootstrap import (  # noqa: E402
    bootstrap_skills,
    make_bootstrap_index_matrix,
)

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

L_PRIMARY = 24
COMBO = "mean/mean"
N_BOOT = 2000
HF_REV = "c244377f2b5bc9e1ed8dd093b05035aa0c4940e9"  # pin the staged MLP tensor was pulled at


def _entry(store: dict, arm: str, layer: int) -> dict[str, np.ndarray]:
    """Return the {ss_res, ss_tot} per-context arrays for (arm, COMBO, layer)."""
    return store[str((arm, COMBO, layer))]


def _skill(e: dict[str, np.ndarray]) -> float:
    """Pooled held-out skill-over-mean R^2 from per-context ss arrays."""
    sr = np.asarray(e["ss_res"], dtype=np.float64)
    st = np.asarray(e["ss_tot"], dtype=np.float64)
    return float(1.0 - sr.sum() / st.sum())


def main() -> None:
    """Redraw the 4-arm hero figure; assert values match the committed JSON."""
    lin = torch.load(
        REPO_ROOT / "eval_results/issue_1426/decomp_indiv.pt",
        map_location="cpu",
        weights_only=False,
    )
    mlp = torch.load(
        REPO_ROOT
        / "data/issue_1426/hf_dl/issue1426_cot_decomposition_r1llama"
        / "analysis_tensors/mlp_indiv/decomp_indiv_mlp.pt",
        map_location="cpu",
        weights_only=False,
    )
    ref = json.loads(
        (
            REPO_ROOT
            / "eval_results/issue_1426/indiv-mlp-nonlinearity-control/mlp_indiv_validity.json"
        ).read_text()
    )
    gate = ref["reads"]["estimator_validity_gate"]

    order = [
        (lin, "d_ctx2ans", "linear\ndirect", paper_palette_role("baseline")),
        (mlp, "mlp_d_ctx2ans", "MLP\ndirect", paper_palette_role("accent")),
        (mlp, "mlp_g_aug", "MLP\naugmented", paper_palette_role("control")),
        (lin, "g_aug", "linear\naugmented", paper_palette_role("primary")),
    ]
    n_ctx = int(np.asarray(_entry(lin, "d_ctx2ans", L_PRIMARY)["ss_tot"]).shape[0])
    assert n_ctx == 50, n_ctx
    idx = make_bootstrap_index_matrix(n_ctx, N_BOOT, BOOTSTRAP_SEED)

    skills = {arm: _skill(_entry(store, arm, L_PRIMARY)) for store, arm, _, _ in order}
    # Exact-reproduction gate against the committed validity JSON (atol 1e-9).
    assert abs(skills["mlp_g_aug"] - gate["mlp_augmented_skill_L_primary"]) < 1e-9
    assert abs(skills["g_aug"] - gate["linear_augmented_skill_L_primary"]) < 1e-9
    assert abs(skills["mlp_d_ctx2ans"] - ref["reads"]["skills"]["mlp_d_ctx2ans"]["24"]) < 1e-9
    assert abs(skills["d_ctx2ans"] - ref["reads"]["skills"]["d_ctx2ans"]["24"]) < 1e-9

    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    xs = np.arange(len(order))
    for x, (store, arm, _, color) in zip(xs, order, strict=True):
        e = _entry(store, arm, L_PRIMARY)
        sr = np.asarray(e["ss_res"], dtype=np.float64)
        st = np.asarray(e["ss_tot"], dtype=np.float64)
        skill = skills[arm]
        dr = bootstrap_skills(sr, st, idx)
        lo_q, hi_q = np.percentile(dr[np.isfinite(dr)], [2.5, 97.5])
        ax.bar(x, skill, color=color, width=0.62)
        ax.errorbar(
            x,
            skill,
            yerr=[[max(0.0, skill - lo_q)], [max(0.0, hi_q - skill)]],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
    ceil = _skill(_entry(lin, "ident", L_PRIMARY))
    ax.axhline(ceil, ls="--", color=paper_palette_role("neutral"), lw=1.0)
    ax.text(
        len(order) - 0.5,
        ceil,
        f"identity ceiling {ceil:.3f}",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _, _, label, _ in order])
    ax.set_ylabel("held-out skill-over-mean R²")
    ax.set_title(f"Per-question arms at frozen L{L_PRIMARY} (paired 95% bootstrap CIs)")
    savefig_paper(fig, "issue_1426/mlp_indiv_hero_4arm", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)
    print(f"regenerated: skills={ {k: round(v, 4) for k, v in skills.items()} } ceiling={ceil:.4f}")


if __name__ == "__main__":
    main()
