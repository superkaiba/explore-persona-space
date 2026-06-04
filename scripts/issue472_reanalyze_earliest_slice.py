# ruff: noqa: RUF002  # Qwen marker token " ※" + em-dash intentional
"""Task #472 re-analysis at the EARLIEST sub-ceiling checkpoint (analyzer recovery).

The on-pod analyze used a matched slice of source-self ΔG = 8±1 nats and got 0
pooled-regression rows: the geometry arms (anchor/near/far) implant the marker on
the source to 13-15 nats by the first checkpoint (frac 0.08) and stay flat, so no
checkpoint *rises through* the 7-9 band. The held-out marker log-prob is NOT
saturated anywhere (it sits −9 to −23 nats below the 0 ceiling at every
checkpoint); the matched-slice machinery simply has no rising trajectory to
interpolate against.

Corrected read: every cell × seed is read at its EARLIEST checkpoint (frac 0.08),
the most sub-ceiling moment for the held-out DV. Cells are NOT at a matched
source-implant level — they are at their own terminal implant level, which itself
differs by recipe. That difference is the finding, not a confound: the recipe
knobs (count, placement) move bystander leakage only by moving how hard the
marker got implanted on the source.

Honors the existing guards: drop saturated / r_collapsed probes from the graded
logP regression; dual all-neg + non-default fits; identification gate
(qwen_default dominance); collinearity gate; Holm multiplicity.

CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    ALWAYS_INCLUDE_NEGATIVE,
    CELL_SPECS,
    SUBCEILING_HEADROOM_NATS,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
    _fit_pooled_ols,
    _pearson,
    _spearman,
    _vif,
    holm_correction,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
    cos_to_source as load_cos_to_source,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
    load_cos_matrix,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
    d_nearest_neg,
    d_source,
    held_out_panel,
    negatives_for_cell,
)

WT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-472")
SLAB = WT / "eval_results" / "issue_472"
CENTROID_DIR = WT / "data" / "issue_472"
SEEDS = [42, 137]
LAYER = 10
SOURCE = "villain"
POOLED_CELLS = [c[0] for c in CELL_SPECS if c[5]]  # anchor(spread), near, far
COLLINEARITY_THRESHOLD = 0.6
ID_GATE_SD_FLOOR = 0.02


def earliest_ck(cell: str, seed: int) -> dict:
    t = json.loads((SLAB / f"{cell}_seed{seed}" / "trajectory.json").read_text())
    return sorted(t["checkpoints"], key=lambda c: c["frac"])[0]


def per_probe_at_earliest(cell: str, seed: int) -> dict[str, dict]:
    ck = earliest_ck(cell, seed)
    out = {}
    for persona, perq in ck["held_out"].items():
        dgs = [r["delta_g"] for r in perq.values()]
        gls = [r["g_logp"] for r in perq.values()]
        kls = [r["kl"] for r in perq.values() if r.get("kl") is not None]
        coll = any(r.get("r_collapsed", False) for r in perq.values())
        mean_g = float(np.mean(gls))
        out[persona] = {
            "delta_g": float(np.mean(dgs)),
            "g_logp": mean_g,
            "kl": float(np.mean(kls)) if kls else float("nan"),
            "saturated": mean_g > -SUBCEILING_HEADROOM_NATS,
            "r_collapsed": coll,
        }
    return out


def main() -> None:
    cts = load_cos_to_source(LAYER, SOURCE, CENTROID_DIR)
    cos_matrix, _names = load_cos_matrix(LAYER, CENTROID_DIR)
    panel = held_out_panel(cts, source=SOURCE)
    base = json.loads((SLAB / "base_panel.json").read_text())
    b_logprob = base["mean_per_persona_b_logprob"]

    print(f"Held-out panel: {len(panel)} probes (layer {LAYER}, source {SOURCE})")

    matched = {c[0]: {s: per_probe_at_earliest(c[0], s) for s in SEEDS} for c in CELL_SPECS}
    cell_negs = {c[0]: negatives_for_cell(c[0], cts, source=SOURCE) for c in CELL_SPECS}

    # ── Build pooled rows over the count-matched placement arms. ──
    logp_rows, kl_rows = [], []
    nearest_default_count = nearest_total = 0
    dnn_nd_across_arms: dict[str, list[float]] = {p: [] for p in panel}
    n_sat = n_coll = n_total = 0
    for slug in POOLED_CELLS:
        negs = cell_negs[slug]
        for seed in SEEDS:
            pp_all = matched[slug][seed]
            for probe in panel:
                if probe not in pp_all:
                    continue
                pp = pp_all[probe]
                n_total += 1
                if pp["saturated"]:
                    n_sat += 1
                if pp["r_collapsed"]:
                    n_coll += 1
                ds = d_source(probe, cts)
                dnn = d_nearest_neg(probe, negs, cos_matrix, exclude_default=False)
                dnn_nd = d_nearest_neg(probe, negs, cos_matrix, exclude_default=True)
                dnn_nd_across_arms[probe].append(dnn_nd)
                if negs:
                    nearest = min(negs, key=lambda nn: 1.0 - cos_matrix[probe][nn])
                    nearest_total += 1
                    if nearest == ALWAYS_INCLUDE_NEGATIVE:
                        nearest_default_count += 1
                row = {
                    "cell": slug,
                    "seed": seed,
                    "probe": probe,
                    "d_source": ds,
                    "dnn": dnn,
                    "dnn_nd": dnn_nd,
                    "b_logprob": b_logprob.get(probe, float("nan")),
                }
                if not pp["saturated"] and not pp["r_collapsed"]:
                    logp_rows.append({**row, "logp": pp["delta_g"]})
                kl_rows.append({**row, "kl": pp["kl"]})

    print(f"\nPooled rows: logP={len(logp_rows)}  KL={len(kl_rows)}")
    print(f"Saturated probes dropped: {n_sat}/{n_total}; r_collapsed: {n_coll}/{n_total}")

    # ── Gates. ──
    default_share = nearest_default_count / nearest_total if nearest_total else float("nan")
    across_arm_sd = [
        float(np.std(v))
        for v in dnn_nd_across_arms.values()
        if len(v) >= 2 and not np.all(np.isnan(v))
    ]
    median_sd = float(np.median(across_arm_sd)) if across_arm_sd else float("nan")

    logp_allneg = _fit_pooled_ols(logp_rows, "logp", "dnn")
    logp_nd = _fit_pooled_ols(logp_rows, "logp", "dnn_nd")
    logp_vif = _vif(logp_rows, "dnn")
    kl_allneg = _fit_pooled_ols(kl_rows, "kl", "dnn")
    kl_nd = _fit_pooled_ols(kl_rows, "kl", "dnn_nd")

    r_ds_dnn = _pearson([r["d_source"] for r in logp_rows], [r["dnn"] for r in logp_rows])
    r_dnn_b = _pearson([r["dnn"] for r in logp_rows], [r["b_logprob"] for r in logp_rows])
    collinearity_ok = (not np.isnan(r_ds_dnn)) and abs(r_ds_dnn) <= COLLINEARITY_THRESHOLD

    def sign(x):
        return 0 if (x is None or np.isnan(x)) else (1 if x > 0 else -1)

    fits_agree = False
    if logp_allneg.get("ok") and logp_nd.get("ok"):
        fits_agree = sign(logp_allneg["coef"]["d_source"]) == sign(
            logp_nd["coef"]["d_source"]
        ) and sign(logp_allneg["coef"]["dnn"]) == sign(logp_nd["coef"]["dnn_nd"])
    id_gate_ok = (not np.isnan(median_sd)) and median_sd >= ID_GATE_SD_FLOOR and fits_agree

    print("\n── GATES ──")
    print(
        f"  collinearity Pearson(d_source,d_nearest_neg) = {r_ds_dnn:.3f}  (|r|<=0.6 -> {collinearity_ok})"
    )
    print(f"  Pearson(d_nearest_neg, b_logprob) = {r_dnn_b:.3f}")
    print(f"  VIF: {logp_vif}")
    print(
        f"  identification: qwen_default nearest share = {default_share:.3f}; median across-arm SD(dnn_nd) = {median_sd:.3f}; fits_agree={fits_agree}; admissible={id_gate_ok}"
    )

    print("\n── logP geometry regression (all-neg) ──")
    if logp_allneg.get("ok"):
        f = logp_allneg
        for k in ["d_source", "dnn", "b_logprob"]:
            print(
                f"    {k:10s}: β={f['coef'][k]:+.3f}  SE={f['se'][k]:.3f}  p={f['pvalue'][k]:.4f}"
            )
        print(f"    n={f['n']} clusters={f['n_clusters']} R²={f['rsquared']:.3f}")
    else:
        print("    DID NOT FIT:", logp_allneg.get("reason"))
    print("── logP geometry regression (non-default d) ──")
    if logp_nd.get("ok"):
        f = logp_nd
        for k in ["d_source", "dnn_nd", "b_logprob"]:
            print(
                f"    {k:10s}: β={f['coef'][k]:+.3f}  SE={f['se'][k]:.3f}  p={f['pvalue'][k]:.4f}"
            )

    # ── Holm over geometry partials. ──
    family_p = {}
    if logp_allneg.get("ok"):
        family_p["logp_geometry_d_source"] = logp_allneg["pvalue"]["d_source"]
        family_p["logp_geometry_d_nearest_neg"] = logp_allneg["pvalue"]["dnn"]

    # ── Count + placement (held-out mean ΔG seed-avg, drop degenerate probes). ──
    def held_mean(slug, seed, dv):
        pp_all = matched[slug][seed]
        vals = []
        for probe in panel:
            if probe in pp_all:
                pp = pp_all[probe]
                if dv == "logp" and (pp["saturated"] or pp["r_collapsed"]):
                    continue
                v = pp["delta_g" if dv == "logp" else "kl"]
                if not np.isnan(v):
                    vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    def count_effect(cells, levels, dv):
        means = [float(np.mean([held_mean(c, s, dv) for s in SEEDS])) for c in cells]
        return levels, means, _spearman(levels, means)

    print("\n── COUNT axis (held-out ΔG, logP) ──")
    nx_l, nx_m, nx_s = count_effect(
        ["c472_negex_100", "c472_anchor", "c472_negex_400"], [100, 200, 400], "logp"
    )
    np_l, np_m, np_s = count_effect(
        ["c472_negp_2", "c472_anchor", "c472_negp_8"], [2, 4, 8], "logp"
    )
    print(f"  negex/persona {nx_l}: {[round(m, 2) for m in nx_m]}  Spearman={nx_s:+.2f}")
    print(f"  neg personas  {np_l}: {[round(m, 2) for m in np_m]}  Spearman={np_s:+.2f}")

    print("── PLACEMENT (held-out ΔG, logP) ──")
    pl = {}
    for c, lab in [
        ("c472_near", "near"),
        ("c472_anchor", "spread"),
        ("c472_far", "far"),
        ("c472_noneg", "no-neg"),
    ]:
        m = float(np.mean([held_mean(c, s, "logp") for s in SEEDS]))
        pl[lab] = m
        print(f"  {lab:8s}: {m:.2f}")

    # ── Implant-strength master correlation. ──
    xs, ys = [], []
    for c in [cc[0] for cc in CELL_SPECS]:
        for s in SEEDS:
            t = json.loads((SLAB / f"{c}_seed{s}" / "trajectory.json").read_text())
            xs.append(max(ck["source_self"]["delta_g_mean"] for ck in t["checkpoints"]))
            ys.append(held_mean(c, s, "logp"))
    from scipy.stats import pearsonr, spearmanr

    pe = pearsonr(xs, ys)
    print("\n── MASTER: held-out leakage vs source-implant strength ──")
    print(
        f"  Spearman={spearmanr(xs, ys).correlation:.3f}  Pearson={pe[0]:.3f} p={pe[1]:.2e} n={len(xs)}"
    )

    holm = holm_correction(family_p) if family_p else {}
    print("\n── Holm (geometry partials) ──")
    for k, v in holm.items():
        print(f"  {k}: p={v['p']:.4f} thresh={v['holm_threshold']:.4f} reject={v['reject_null']}")

    # write a compact recovered summary
    out = {
        "schema": "i472_reanalysis_earliest_slice",
        "read_at": "earliest_checkpoint_frac_0.08",
        "n_held_out_probes": len(panel),
        "pooled_cells": POOLED_CELLS,
        "n_logp_rows": len(logp_rows),
        "n_kl_rows": len(kl_rows),
        "n_saturated_dropped": n_sat,
        "n_collapsed_dropped": n_coll,
        "n_total_pooled_probe_rows": n_total,
        "logp_regression": {"all_neg": logp_allneg, "non_default": logp_nd, "vif": logp_vif},
        "kl_regression": {"all_neg": kl_allneg, "non_default": kl_nd},
        "collinearity_gate": {
            "pearson_d_source_d_nearest_neg": r_ds_dnn,
            "pearson_d_nearest_neg_b_logprob": r_dnn_b,
            "ok": collinearity_ok,
        },
        "identification_gate": {
            "qwen_default_nearest_share": default_share,
            "median_across_arm_sd_dnn_nd": median_sd,
            "fits_agree_sign": fits_agree,
            "admissible": id_gate_ok,
        },
        "holm_multiplicity": holm,
        "count_effects": {
            "negex": {"levels": nx_l, "means": nx_m, "spearman": nx_s},
            "negp": {"levels": np_l, "means": np_m, "spearman": np_s},
        },
        "placement_means": pl,
        "master_implant_correlation": {
            "spearman": float(spearmanr(xs, ys).correlation),
            "pearson": float(pe[0]),
            "p": float(pe[1]),
            "n": len(xs),
        },
    }
    (SLAB / "reanalysis_earliest_slice.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {SLAB / 'reanalysis_earliest_slice.json'}")


if __name__ == "__main__":
    main()
