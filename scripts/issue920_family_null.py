# ruff: noqa: RUF002, RUF003
# Intentional Unicode (rho, ×) in scientific docstrings + log messages.
"""Issue #920 free-analysis follow-up: family-aware null bands for the DV-2 behavior read-out.

The shipped DV-2 bands (``null_bands_and_headline.json``) use FREE permutations of the
50-context E0 target, which are anti-conservative given family structure (between-family
variance 0.31-0.85; predictions cluster by family). This script builds exchangeability-
respecting nulls for the same max-over-cells statistic, per (behavior × side × regime):

1. **wf_full (PRIMARY)** — within-family permutation null on the SAME statistic as the
   original headline: observed = max over side cells of |Spearman rho(pred, E0)|; each of
   the 1,000 draws (seed 921) permutes E0 WITHIN each of the 7 family blocks and inherits
   the identical max-over-cells selection. Exchangeability tested: contexts are
   exchangeable within family (family identity of every E0 value is preserved), so a
   clear means within-family association above family structure — the banded version of
   the body's exploratory family-centering re-read.
2. **wf_centered** — the same within-family null applied to the family-CENTERED statistic
   (Spearman of family-mean-centered pred vs family-mean-centered E0, the analyzer's
   ``center()`` definition), max-over-cells inherited per draw. Because a within-family
   permutation preserves family means, centering commutes with the permutation
   (center(y[perm]) == center(y)[perm]), so the draws batch as one GEMM on precomputed
   ranks exactly like the full statistic.
3. **family_means** — the 7-family-means test: Spearman over the G=7 family means of pred
   vs E0, max over side cells, against the EXACT permutation group (all 7! = 5,040
   assignments of family-mean values to family identities). Exchangeability tested:
   family identities are exchangeable at the family level (between-family association).
   n=7 Spearman lives on a lattice with spacing 1/28 ≈ 0.036 (granularity floor).

Selection symmetry (.claude/rules/selection-symmetric-nulls.md): every draw of every test
receives the IDENTICAL max-over-side-cells selection the observed statistic uses;
per-draw max vectors are persisted (compact sibling JSON) so any quantile is a pure
re-reduction. Band-vs-ceiling: |rho| <= 1 is the achievable ceiling; each band's upper
bound is reported against it and a band at/above the ceiling is narrated
failure-to-reject (uninformative-by-construction).

Inputs (all pre-existing; ANALYSIS-ONLY):
- data/issue_920/preds/pooled_heldout_predictions.pt  (ro_predA/ro_predB, (1560, 50, 7))
- eval_results/issue_812/graded_e0_{highm,lowm}.json  (via issue920_common.load_e0_graded)
- data/issue594/battery.json                          (7 families 14/10/8/6/5/5/2)
- eval_results/issue_920/{readout_rho_by_cell,null_bands_and_headline}.json (consistency)

Outputs:
- eval_results/issue_920/family_aware_readout_bands.json      (observed/bands/clears/p)
- eval_results/issue_920/family_aware_readout_perdraw.json    (per-draw max vectors)
- figures/issue_920/family_aware_readout_bands.png            (paper-plots style)

Batched inner loops ONLY: ranks computed once per (regime, behavior); all draws as one
GEMM via ``stored_pred_rho_null`` (the #920 run's own batched null helper, the
null_battery.py pattern). Total compute ~5 GFLOP — seconds on CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import permutations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Shared-VM thread caps (#847) must bind BEFORE torch freezes its pool at import.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue920_common import (  # noqa: E402
    E0_BEHAVIORS,
    dump_json,
    load_battery,
    load_e0_graded,
    load_json,
    reproducibility_metadata,
)
from issue920_nulls_figures import stored_pred_rho_null  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue920_family_null")

FAMILY_ORDER = ["behavior", "default", "format", "icl", "persona", "rephrase", "wildchat"]
FAMILY_NULL_SEED = 921
REGIMES = (("in_probe", "ro_predA", "R_in_probe"), ("input_ood", "ro_predB", "R_input_ood"))
# The body's named largest centered residuals ("Behavior read-out clears are discounted..."):
# fact expression ctx 0.56, format style ans 0.47-0.49, refusal ctx 0.32-0.47,
# harmful compliance ctx 0.44-0.48 — each at the free-read best cell, both regimes.
NAMED_RESIDUALS = [
    ("fact_expression", "ctx"),
    ("format_style", "ans"),
    ("refusal", "ctx"),
    ("harmful_compliance", "ctx"),
]
SPEARMAN7_GRANULARITY = 1.0 / 28.0  # n=7 rank-rho lattice spacing (rho = 1 - sum(D^2)/56)


def within_family_perms(
    fam_positions: list[np.ndarray], n: int, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_draws, n) index matrix permuting positions independently WITHIN each family.

    Vectorized over draws: per family, ranks of a (n_draws, k) uniform key matrix give
    all draws' within-block permutations at once (no per-draw Python loop).
    """
    perms = np.tile(np.arange(n, dtype=np.int64), (n_draws, 1))
    for pos in fam_positions:
        order = np.argsort(rng.random((n_draws, pos.size)), axis=1)
        perms[:, pos] = pos[order]
    return perms


def center_rows(x: np.ndarray, fam_positions: list[np.ndarray]) -> np.ndarray:
    """Subtract the per-family mean along the last (context) axis; returns float64."""
    out = np.asarray(x, dtype=np.float64).copy()
    for pos in fam_positions:
        out[..., pos] -= out[..., pos].mean(axis=-1, keepdims=True)
    return out


def family_means(x: np.ndarray, fam_positions: list[np.ndarray]) -> np.ndarray:
    """(..., 7) per-family means along the last (context) axis, in FAMILY_ORDER."""
    return np.stack(
        [np.asarray(x, dtype=np.float64)[..., pos].mean(axis=-1) for pos in fam_positions],
        axis=-1,
    )


def _band_read(obs: float, per_draw_max: np.ndarray, *, exact: bool) -> dict:
    """Band + clear + p for a max-inherited null; MC p adds the observed pseudo-draw."""
    band = float(np.quantile(per_draw_max, 0.975))
    ge = int((per_draw_max >= obs - 1e-12).sum())
    p = ge / per_draw_max.size if exact else (1 + ge) / (per_draw_max.size + 1)
    return {
        "observed_max_abs_rho": round(obs, 6),
        "band_p97_5": round(band, 6),
        "clears": bool(obs > band),
        "p_value": round(float(p), 6),
        "n_draws": int(per_draw_max.size),
        "band_upper_vs_ceiling": {
            "band_p97_5": round(band, 6),
            "ceiling_abs_rho": 1.0,
            "band_at_or_above_ceiling": bool(band >= 1.0 - 1e-9),
        },
    }


def main() -> int:
    """Run the three family-aware DV-2 null reads; write JSONs + figure; exit 0."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=FAMILY_NULL_SEED)
    ap.add_argument(
        "--preds",
        default=str(
            PROJECT_ROOT / "data" / "issue_920" / "preds" / "pooled_heldout_predictions.pt"
        ),
    )
    ap.add_argument("--eval-out", default=str(PROJECT_ROOT / "eval_results" / "issue_920"))
    ap.add_argument("--fig-out", default=str(PROJECT_ROOT / "figures" / "issue_920"))
    ap.add_argument(
        "--out-json", default=None, help="override output JSON path (smoke scratch redirect)"
    )
    args = ap.parse_args()

    eval_out = Path(args.eval_out)
    out_json = (
        Path(args.out_json) if args.out_json else eval_out / "family_aware_readout_bands.json"
    )
    out_perdraw = out_json.with_name(out_json.stem.replace("_bands", "_perdraw") + ".json")
    fig_dir = Path(args.fig_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    blob = torch.load(args.preds, map_location="cpu", weights_only=False)
    ctx_ids = list(blob["ctx_ids"])
    assert len(ctx_ids) == 50, len(ctx_ids)
    _instances, fam_map = load_battery()
    fvec = np.array([fam_map[c] for c in ctx_ids])
    fam_positions = [np.where(fvec == f)[0] for f in FAMILY_ORDER]
    assert sorted(p.size for p in fam_positions) == [2, 5, 5, 6, 8, 10, 14]

    e0 = load_e0_graded()
    E0 = np.stack([[e0[b][c] for c in ctx_ids] for b in E0_BEHAVIORS], axis=1)  # (50, 7)
    ex_pred = blob["excluded_pred_mask"].numpy()
    n_ctx = len(blob["ctx_cell_names"])
    cells = list(blob["ctx_cell_names"]) + list(blob["ans_cell_names"])
    sides = (("ctx", slice(0, n_ctx)), ("ans", slice(n_ctx, None)))

    rng = np.random.default_rng(args.seed)
    wf = within_family_perms(fam_positions, 50, args.n_draws, rng)
    perms_wf = np.concatenate([np.arange(50, dtype=np.int64)[None], wf])  # col 0 = identity
    perms7 = np.array(list(permutations(range(7))), dtype=np.int64)
    assert perms7.shape == (5040, 7) and (perms7[0] == np.arange(7)).all()

    ro_json = load_json(eval_out / "readout_rho_by_cell.json")
    free = load_json(eval_out / "null_bands_and_headline.json")["dv2_observed_vs_band"]

    tests: dict[str, dict] = {}
    per_draw: dict[str, dict[str, list[float]]] = {
        "wf_full": {},
        "wf_centered": {},
        "family_means": {},
    }
    best_cells: dict[str, dict] = {}
    max_id_diff = 0.0

    for regime, key, ro_key in REGIMES:
        P3 = blob[key].float().numpy()
        assert P3.shape == (len(cells), 50, len(E0_BEHAVIORS)), P3.shape
        ro_arr = np.array(
            [[np.nan if v is None else v for v in row] for row in ro_json["rho"][ro_key]]
        )
        for bi, b in enumerate(E0_BEHAVIORS):
            P = P3[:, :, bi].astype(np.float64)
            y = E0[:, bi]
            # (1) full statistic under within-family permutation (identity in col 0)
            rho_full = stored_pred_rho_null(P, y, perms_wf)
            rho_full[ex_pred] = 0.0
            # consistency: identity column reproduces the shipped per-cell rho
            valid = ~ex_pred & ~np.isnan(ro_arr[:, bi])
            max_id_diff = max(
                max_id_diff, float(np.abs(rho_full[valid, 0] - ro_arr[valid, bi]).max())
            )
            # (2) centered statistic (centering commutes with within-family permutation)
            rho_cent = stored_pred_rho_null(
                center_rows(P, fam_positions), center_rows(y, fam_positions), perms_wf
            )
            rho_cent[ex_pred] = 0.0
            # (3) 7-family-means statistic under the exact 7! group
            rho_m = stored_pred_rho_null(
                family_means(P, fam_positions), family_means(y, fam_positions), perms7
            )
            rho_m[ex_pred] = 0.0

            for side, sel in sides:
                tkey = f"{regime}_{side}_{b}"
                a_full, a_cent, a_m = (
                    np.abs(rho_full[sel]),
                    np.abs(rho_cent[sel]),
                    np.abs(rho_m[sel]),
                )
                obs_full = float(a_full[:, 0].max())
                pdm_full = a_full[:, 1:].max(axis=0)
                obs_cent = float(a_cent[:, 0].max())
                pdm_cent = a_cent[:, 1:].max(axis=0)
                obs_m = float(a_m[:, 0].max())
                pdm_m = a_m.max(axis=0)  # exact test: band over ALL 5,040 incl. identity

                fr = free.get(tkey, {})
                assert fr, f"missing free-band reference for {tkey}"
                assert abs(obs_full - fr["observed_max_abs_rho"]) < 1e-5, (
                    tkey,
                    obs_full,
                    fr["observed_max_abs_rho"],
                )
                mtest = _band_read(obs_m, pdm_m, exact=True)
                mtest["granularity_floor_note"] = (
                    f"n=7 Spearman lattice spacing {SPEARMAN7_GRANULARITY:.4f}; "
                    "with max over hundreds of cells the exact band saturates the lattice"
                )
                tests[tkey] = {
                    "free_band_p97_5": fr["band_p97_5"],
                    "free_clears": fr["clears"],
                    "wf_full": _band_read(obs_full, pdm_full, exact=False),
                    "wf_centered": _band_read(obs_cent, pdm_cent, exact=False),
                    "family_means": mtest,
                }
                per_draw["wf_full"][tkey] = [round(float(v), 4) for v in pdm_full]
                per_draw["wf_centered"][tkey] = [round(float(v), 4) for v in pdm_cent]
                per_draw["family_means"][tkey] = [round(float(v), 4) for v in pdm_m]

                # best cell (free-read argmax over the full statistic) + its centered rho
                loc = int(np.argmax(a_full[:, 0]))
                j = loc if side == "ctx" else n_ctx + loc
                best_cells[tkey] = {
                    "cell": cells[j],
                    "rho_full_at_cell": round(float(rho_full[j, 0]), 4),
                    "rho_centered_at_cell": round(float(rho_cent[j, 0]), 4),
                    "abs_centered_clears_wf_centered_band": bool(
                        abs(float(rho_cent[j, 0])) > tests[tkey]["wf_centered"]["band_p97_5"]
                    ),
                }
        logger.info("[phase=%s] bands done (14 tests)", regime)

    assert max_id_diff < 5e-4, f"identity rho mismatch vs readout_rho_by_cell.json: {max_id_diff}"

    named = {
        f"{regime}_{side}_{b}": best_cells[f"{regime}_{side}_{b}"]
        for (b, side) in NAMED_RESIDUALS
        for regime, _, _ in REGIMES
    }
    n_clear_wf = sum(t["wf_full"]["clears"] for t in tests.values())
    n_clear_cent = sum(t["wf_centered"]["clears"] for t in tests.values())
    n_clear_m = sum(t["family_means"]["clears"] for t in tests.values())
    n_m_uninformative = sum(
        t["family_means"]["band_upper_vs_ceiling"]["band_at_or_above_ceiling"]
        for t in tests.values()
    )
    summary = {
        "definition": {
            "statistic": "max over side cells of |pooled held-out Spearman rho(pred, E0)|; "
            "every null draw inherits the SAME max-over-cells selection (two-sided |rho|)",
            "wf_full": "PRIMARY: within-family permutation of E0 (family identity of every "
            "value preserved); tests within-family association above family structure on the "
            "SAME observed statistic as the shipped free-permutation headline",
            "wf_centered": "same within-family null on the family-mean-centered Spearman "
            "(the analyzer's centering re-read, now banded); centering commutes with "
            "within-family permutation so draws batch on precomputed ranks",
            "family_means": "Spearman over the G=7 family means, exact 7! = 5,040 "
            "permutation group (family-level exchangeability / between-family association); "
            f"n=7 rho lattice spacing {SPEARMAN7_GRANULARITY:.4f}",
            "n_draws_wf": int(args.n_draws),
            "seed": int(args.seed),
            "excluded_cells_zeroed": int(ex_pred.sum()),
            "family_order": FAMILY_ORDER,
            "recipe": "recompute: uv run python scripts/issue920_family_null.py "
            f"--n-draws {args.n_draws} --seed {args.seed} (deterministic; per-draw max "
            "vectors in family_aware_readout_perdraw.json)",
        },
        "headline": {
            "n_tests": len(tests),
            "free_band_clears": sum(bool(t["free_clears"]) for t in tests.values()),
            "wf_full_clears": int(n_clear_wf),
            "wf_centered_clears": int(n_clear_cent),
            "family_means_clears": int(n_clear_m),
            "family_means_bands_at_ceiling": int(n_m_uninformative),
        },
        "tests": tests,
        "best_cells": best_cells,
        "named_residual_cells": named,
        "consistency_checks": {
            "max_abs_diff_identity_rho_vs_readout_json": round(max_id_diff, 8),
            "observed_max_matches_dv2_observed_vs_band": True,
        },
        "reproducibility": reproducibility_metadata(),
    }
    dump_json(summary, out_json)
    with open(out_perdraw, "w") as f:
        json.dump(
            {
                "note": "per-draw max-|rho| vectors (selection-inherited), rounded 4dp",
                "seed": int(args.seed),
                **per_draw,
            },
            f,
            separators=(",", ":"),
        )
    logger.info("[phase=json] wrote %s (+ %s)", out_json, out_perdraw.name)

    _figure(tests, fig_dir)
    logger.info(
        "[phase=done] wf_full clears %d/28, wf_centered %d/28, family_means %d/28 "
        "(means bands at ceiling: %d)",
        n_clear_wf,
        n_clear_cent,
        n_clear_m,
        n_m_uninformative,
    )
    return 0


def _figure(tests: dict[str, dict], fig_dir: Path) -> None:
    """One 3-panel figure: per test, observed dot vs free band vs family-aware band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    keys = list(tests)
    # keys are f"{regime}_{side}_{behavior}" with regime in {in_probe, input_ood}
    labels = []
    for k in keys:
        for reg in ("in_probe", "input_ood"):
            if k.startswith(reg + "_"):
                side, beh = k[len(reg) + 1 :].split("_", 1)
                reg_lab = "in-probe" if reg == "in_probe" else "held-out probes"
                side_lab = "context read" if side == "ctx" else "answer read"
                labels.append(f"{beh.replace('_', ' ')} ({side_lab}, {reg_lab})")
                break
    ys = np.arange(len(keys))[::-1]
    panels = [
        ("wf_full", "Full correlation\n(within-family permutation band)"),
        ("wf_centered", "Family-centered correlation\n(within-family permutation band)"),
        ("family_means", "7-family-means correlation\n(exact 7! permutation band)"),
    ]
    c_obs = paper_palette_role("primary")
    c_band = paper_palette_role("accent")
    c_free = paper_palette_role("neutral")
    fig, axes = plt.subplots(1, 3, figsize=(13, 10), sharey=True, layout="constrained")
    for ax, (tkey, title) in zip(axes, panels, strict=True):
        for yy, k in zip(ys, keys, strict=True):
            t = tests[k][tkey]
            if tkey == "wf_full":
                ax.plot([tests[k]["free_band_p97_5"]], [yy], "|", color=c_free, ms=14, mew=2)
            ax.plot([t["band_p97_5"]], [yy], "|", color=c_band, ms=14, mew=2.5)
            filled = t["clears"]
            ax.plot(
                [t["observed_max_abs_rho"]],
                [yy],
                "o",
                color=c_obs if filled else "white",
                mec=c_obs,
                ms=5,
            )
        ax.axvline(1.0, color="black", lw=0.8, ls="--")
        ax.set_xlim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("absolute Spearman rho")
    axes[0].set_yticks(ys, labels, fontsize=7)
    axes[0].plot([], [], "|", color=c_free, ms=10, label="free-permutation band (shipped)")
    axes[0].plot([], [], "|", color=c_band, ms=10, label="family-aware band (97.5th pct)")
    axes[0].plot([], [], "o", color=c_obs, ms=5, label="observed (filled = clears)")
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle(
        "Behavior read-out vs family-aware null bands (max-over-cells inherited per draw; "
        "dashed line = |rho| ceiling 1.0)",
        fontsize=11,
    )
    savefig_paper(fig, "family_aware_readout_bands", dir=fig_dir)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
