#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, −) in scientific docstrings + labels.
"""Issue #928 clean-result revision: per-context companions for the two
aggregate-only hypothesis reads (clean-result-critic Lens 11, round 1).

Produces the LOW-LEVEL per-unit views behind:

- H3 (composition): per-context Δ(composed − direct) =
  skill(comp_pred) − skill(d_ctx2ans), plotted against the direct map's
  per-context skill → ``figures/issue_928/h3_composed_direct_percontext``.
- H4 (sufficiency): per-context Δ(CoT-augmented − CoT-alone oracle) =
  skill(g_aug) − skill(b_cot2ans), plotted against the oracle's
  per-context skill → ``figures/issue_928/h4_sufficiency_percontext``.

Everything is a PURE RE-REDUCTION of the persisted per-context LOCO error
decompositions (``decomp_{avg_q,indiv}.pt`` — per (arm, combo, layer):
``ss_res``/``ss_tot`` (50,) arrays in battery order) at the PRIMARY frozen
convention per regime (mean/mean at the direct arm's full-data best LOCO
layer — 27 avg_q / 25 indiv, read from ``bootstrap_deltaskill.json``). No
refit, no bootstrap (the aggregate CIs live in the committed H3/H4 forest
figures); per-context skill is 1 − ss_res[i]/ss_tot[i].

Fail-loud validation gate: the pooled Δ recomputed from the tensors must
reproduce the committed ``H3_delta_comp_minus_d`` / ``H4_delta_g_minus_b``
observed values exactly (atol 1e-9) in both regimes.

Inputs (all committed / local; ANALYSIS-ONLY — no model calls):
- ``eval_results/issue_928/decomp_{avg_q,indiv}.pt``
- ``eval_results/issue_928/bootstrap_deltaskill.json`` (frozen layers + the
  committed observed statistics the gate reproduces)
- ``eval_results/issue_928/recon_skill_grid.json`` (battery-order context ids)

Outputs:
- ``eval_results/issue_928/percontext_deltas.json``
- ``figures/issue_928/h3_composed_direct_percontext.{png,pdf,meta.json}``
- ``figures/issue_928/h4_sufficiency_percontext.{png,pdf,meta.json}``

Usage (repo-root inputs, repo-root outputs)::

    OMP_NUM_THREADS=8 uv run python scripts/issue928_percontext_deltas.py \
        --in-results <repo>/eval_results/issue_928 \
        --out-results <repo>/eval_results/issue_928 \
        --out-figures <repo>/figures/issue_928
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue928_common import dump_json, load_json, reproducibility_metadata  # noqa: E402
from issue928_length_matched_gain import FLAGGED_BELOW_PARSE_FLOOR  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue928_percontext_deltas")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PRIMARY_COMBO = "mean/mean"
REGIMES = ("avg_q", "indiv")
REGIME_TITLES = {"avg_q": "Query-averaged regime", "indiv": "Per-question regime"}
# (figure stem, delta arm pair, x-axis arm, committed statistic key, y label, x label)
CONTRASTS = (
    {
        "stem": "h3_composed_direct_percontext",
        "hi_arm": "comp_pred",
        "lo_arm": "d_ctx2ans",
        "x_arm": "d_ctx2ans",
        "committed_key": "H3_delta_comp_minus_d",
        "ylabel": "per-context Δ skill (composed − direct)",
        "xlabel": "per-context direct-map skill",
    },
    {
        "stem": "h4_sufficiency_percontext",
        "hi_arm": "g_aug",
        "lo_arm": "b_cot2ans",
        "x_arm": "b_cot2ans",
        "committed_key": "H4_delta_g_minus_b",
        "ylabel": "per-context Δ skill (context+CoT − CoT alone)",
        "xlabel": "per-context CoT-alone (oracle) skill",
    },
)


def per_context_skill(decomp: dict, arm: str, layer: int) -> np.ndarray:
    """(50,) per-context held-out skill 1 − ss_res/ss_tot at (arm, mean/mean, layer)."""
    key = str((arm, PRIMARY_COMBO, layer))
    if key not in decomp:
        raise RuntimeError(f"decomp missing key {key}")
    v = decomp[key]
    res = np.asarray(v["ss_res"], np.float64)
    tot = np.asarray(v["ss_tot"], np.float64)
    assert res.shape == tot.shape == (50,), (res.shape, tot.shape)
    return 1.0 - res / np.clip(tot, 1e-12, None)


def pooled_delta(decomp: dict, hi_arm: str, lo_arm: str, layer: int) -> float:
    """Pooled Δ skill (Σ-of-errors form, the committed statistic's estimator)."""
    out = {}
    for arm in (hi_arm, lo_arm):
        v = decomp[str((arm, PRIMARY_COMBO, layer))]
        out[arm] = 1.0 - float(np.asarray(v["ss_res"]).sum()) / float(np.asarray(v["ss_tot"]).sum())
    return out[hi_arm] - out[lo_arm]


def assert_matches_committed(observed: float, boot_blob: dict, regime: str, key: str) -> None:
    """Recomputed pooled Δ must reproduce the committed observed value (atol 1e-9)."""
    ref = boot_blob["by_regime"][regime]["statistics"][key]["primary_frozen_direct_best"]
    dev = abs(observed - float(ref["observed"]))
    if dev > 1e-9:
        raise AssertionError(f"{regime}/{key}: re-reduction diverges from committed by {dev:.3e}")


def make_figure(contrast: dict, per_regime: dict, out_figures: Path) -> None:
    """Two labeled per-context scatters (one per regime), flagged contexts marked."""
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained")
    c_unfl = paper_palette_role("primary")
    c_flag = paper_palette_role("accent")
    for ax, (regime, r) in zip(axes, per_regime.items(), strict=True):
        rows = r["per_context"]
        for flag, color, lab in ((False, c_unfl, "unflagged"), (True, c_flag, "flagged")):
            sub = [p for p in rows if p["flagged"] == flag]
            ax.scatter(
                [p["x_skill"] for p in sub],
                [p["delta"] for p in sub],
                s=14,
                color=color,
                label=f"{lab} (n={len(sub)})",
            )
        for p in rows:
            ax.annotate(p["context"], (p["x_skill"], p["delta"]), fontsize=4, rotation=30)
        ax.axhline(0.0, lw=0.8, color="0.5")
        ax.set_xlabel(contrast["xlabel"])
        ax.set_ylabel(contrast["ylabel"])
        ax.set_title(
            f"{REGIME_TITLES[regime]} (frozen layer {r['frozen_layer']}) — "
            f"pooled Δ {r['pooled_delta']:+.3f}"
        )
        ax.legend(fontsize=7)
    savefig_paper(fig, f"issue_928/{contrast['stem']}", dir=str(out_figures.parent))
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #928 per-context H3/H4 delta companions")
    ap.add_argument("--in-results", default=str(PROJECT_ROOT / "eval_results" / "issue_928"))
    ap.add_argument("--out-results", default=str(PROJECT_ROOT / "eval_results" / "issue_928"))
    ap.add_argument("--out-figures", default=str(PROJECT_ROOT / "figures" / "issue_928"))
    args = ap.parse_args()
    in_results = Path(args.in_results)
    out_results, out_figures = Path(args.out_results), Path(args.out_figures)
    out_results.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    grid = load_json(in_results / "recon_skill_grid.json")
    boot_blob = load_json(in_results / "bootstrap_deltaskill.json")
    context_ids: list[str] = grid["context_ids"]
    assert len(context_ids) == 50, len(context_ids)
    unknown = set(FLAGGED_BELOW_PARSE_FLOOR) - set(context_ids)
    if unknown:
        raise RuntimeError(f"flagged ids not in battery: {sorted(unknown)}")
    flagged_mask = np.array([c in FLAGGED_BELOW_PARSE_FLOOR for c in context_ids])

    decomps = {
        regime: torch.load(in_results / f"decomp_{regime}.pt", weights_only=False)
        for regime in REGIMES
    }
    layers = {
        regime: int(
            boot_blob["by_regime"][regime]["layer_conventions"]["primary_frozen_direct_best_layer"]
        )
        for regime in REGIMES
    }

    blob: dict = {
        "dv": (
            "Per-context Δ held-out skill re-reduction for the composition (composed − direct) "
            "and sufficiency (CoT-augmented − CoT-alone oracle) reads at the primary frozen "
            "convention (mean/mean, direct-arm full-data best LOCO layer per regime)"
        ),
        "primary_combo": PRIMARY_COMBO,
        "flagged_below_parse_floor": list(FLAGGED_BELOW_PARSE_FLOOR),
        "contrasts": {},
        "reproducibility": reproducibility_metadata(),
    }
    for contrast in CONTRASTS:
        per_regime: dict = {}
        for regime in REGIMES:
            layer = layers[regime]
            decomp = decomps[regime]
            hi = per_context_skill(decomp, contrast["hi_arm"], layer)
            lo = per_context_skill(decomp, contrast["lo_arm"], layer)
            x = per_context_skill(decomp, contrast["x_arm"], layer)
            pooled = pooled_delta(decomp, contrast["hi_arm"], contrast["lo_arm"], layer)
            assert_matches_committed(pooled, boot_blob, regime, contrast["committed_key"])
            delta = hi - lo
            per_regime[regime] = {
                "frozen_layer": layer,
                "pooled_delta": pooled,
                "per_context": [
                    {
                        "context": c,
                        "flagged": bool(flagged_mask[i]),
                        "x_skill": float(x[i]),
                        "delta": float(delta[i]),
                        f"skill_{contrast['hi_arm']}": float(hi[i]),
                        f"skill_{contrast['lo_arm']}": float(lo[i]),
                    }
                    for i, c in enumerate(context_ids)
                ],
            }
            logger.info(
                "[%s %s @L%d] pooled Δ=%+.4f (gate PASS) | per-context min %+.3f max %+.3f",
                contrast["stem"],
                regime,
                layer,
                pooled,
                delta.min(),
                delta.max(),
            )
        blob["contrasts"][contrast["stem"]] = {
            "delta": f"skill({contrast['hi_arm']}) - skill({contrast['lo_arm']})",
            "by_regime": per_regime,
        }
        make_figure(contrast, per_regime, out_figures)
        logger.info("[phase=figure_done] wrote %s", out_figures / f"{contrast['stem']}.png")

    out_path = out_results / "percontext_deltas.json"
    dump_json(blob, out_path)
    logger.info("[phase=analysis_done] wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
