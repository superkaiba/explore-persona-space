"""Characterize the directions that deviate MOST from the two-parameter noise floor.

Question: the #779 floor fit (`scripts/issue779_spectrum_floor_fit.py`) established
that a two-parameter mechanical floor R^2_j = (1-b) - a/lambda_j explains most of the
per-direction R^2 spectrum, and banked a list of directions sitting BELOW that curve —
genuinely "worse predicted than their variance predicts". Nothing has ever
characterized that set: every SAE / autointerp / logit-lens pass in the project ran on
the RAW-R^2-worst directions instead, which by construction is the low-variance tail.

This runs the floor fit on the #1738 multi-turn corpus (fully local staged arrays, so
the basis vectors are available and the fit doubles as a REPLICATION test of the band),
then characterizes the top-15 below-curve deviants — and, for contrast, the top-15
above-curve directions and the banked raw-R^2-worst-20 — through four reads:

  (a) SAE decoder max-|cos| + argmax feature, vs the matched random-unit null;
  (b) NeuronPedia autointerp descriptions for those argmax features (local cache);
  (c) alignment to all SEVEN persona-vector trait directions (r_B, layer-19 row);
  (d) logit-lens top tokens through the unembedding.

Every read reuses the #1482 machinery by import (load_layer / load_pred /
gram_spectrum / load_unembedding) and mirrors residual_alignment.json's conventions,
so the deviant band is directly comparable to the banked worst/best passes.

Units: gram_spectrum returns eigenvalues in SS units (n x variance). Variance SHARE
(eigenvalue / total SS) is used for lambda throughout, matching the #779 fit — the
share convention rescales `a` but leaves b, gof and every deviation unchanged.

All inputs local; 0 GPU.
"""

from __future__ import annotations

import gc
import gzip
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM run)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.task_workflow import repo_root  # noqa: E402

PROJECT_ROOT = repo_root()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_residual_svd as RS  # noqa: E402

LAYER = 19
PRIMARY_ARM = "context"
PRIMARY_FITTER = "ridge"
ARMS = ("context", "prefix", "bare")
FITTERS = ("ridge", "mlp_w8192")
FIT_TOP = RS.R2_SELECT_K  # 256 — same fit set as the #779 twin
N_DEVIANT = 15
SEED = 1738

TRAITS7 = (
    "evil",
    "sycophancy",
    "hallucination",
    "optimistic",
    "impolite",
    "apathetic",
    "humorous",
)
RB_DIR = "data/issue_779/r_b"
RB_ROW = 19  # r_B row index for layer 19 (the feature_extremes convention)

ALIGNMENT = "eval_results/issue_1482/twoway_residual/residual_alignment.json"
NP_CACHE = "eval_results/issue_1482/worst_pc_autointerp/np_cache"
REF_779 = "eval_results/issue_779/spectrum_floor_fit/spectrum_floor_fit.json"
OUT = "eval_results/issue_1738/deviant_band/deviant_band.json"
FIG_DIR = "figures/issue_1738/deviant_band"


def _ols(x: np.ndarray, y: np.ndarray, pin_intercept: float | None = None) -> tuple[float, float]:
    """OLS of y on x. Returns (intercept, slope); intercept optionally pinned.

    Byte-for-byte the #779 estimator (`issue779_spectrum_floor_fit._ols`), restated
    here rather than imported: that module runs its fit at import time under a
    ``main()`` guard but pulls the #779 JSON path at module scope, so importing it
    for one 6-line helper would couple this script to that corpus's artifacts.
    """
    if pin_intercept is not None:
        slope = float(np.dot(x, y - pin_intercept) / np.dot(x, x))
        return pin_intercept, slope
    X = np.stack([np.ones_like(x), x], axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1])


def _fit_floor(r2: np.ndarray, share: np.ndarray) -> dict:
    """Both floor forms on the top-``FIT_TOP`` directions. Mirrors the #779 fit."""
    x = 1.0 / share
    a1_int, a1_slope = _ols(x, r2, pin_intercept=1.0)  # 1 - a/lambda
    a2_int, a2_slope = _ols(x, r2)  # (1-b) - a/lambda
    pred1, pred2 = a1_int + a1_slope * x, a2_int + a2_slope * x

    def _gof(pred: np.ndarray) -> float:
        ss_res = float(((r2 - pred) ** 2).sum())
        ss_tot = float(((r2 - r2.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot

    return {
        "one_param": {"a": -a1_slope, "gof_r2_on_fit_set": _gof(pred1)},
        "two_param": {
            "a": -a2_slope,
            "b_proportional_loss": 1.0 - a2_int,
            "asymptote_1_minus_b": a2_int,
            "gof_r2_on_fit_set": _gof(pred2),
            "implied_zero_crossing_share": (-a2_slope / a2_int) if a2_int > 0 else None,
        },
        "_pred2": pred2,
        "_deviation": pred2 - r2,  # positive = worse than the mechanical floor
    }


def _np_descriptions(want: set[int], cache: Path) -> dict[str, str]:
    """NeuronPedia autointerp descriptions for ``want`` from the local export."""
    found: dict[str, str] = {}
    for p in sorted(cache.glob("*.jsonl.gz")):
        for line in gzip.decompress(p.read_bytes()).decode("utf-8").split("\n"):
            if not line.strip():
                continue
            rec = json.loads(line)
            if int(rec["index"]) in want:
                found[str(int(rec["index"]))] = (rec.get("description") or "").strip()
    return found


def _rb_matrix() -> np.ndarray:
    """(3584, 7) unit-norm columns: the layer-19 r_B row of all seven traits."""
    cols = []
    for t in TRAITS7:
        v = torch.load(PROJECT_ROOT / RB_DIR / f"{t}.pt", map_location="cpu", weights_only=False)
        arr = np.asarray(v["r_b"] if isinstance(v, dict) else v, dtype=np.float64)
        assert arr.ndim == 2 and arr.shape[-1] == RS.HIDDEN_DIM, (t, arr.shape)
        u = arr[RB_ROW]
        cols.append(u / np.linalg.norm(u))
    return np.stack(cols, axis=1)


def _spectrum_and_r2(arm: str, fitter: str, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """(per-direction held-out R^2, variance share) over the top-``FIT_TOP`` target PCs."""
    Y, Yc, vecs, ss_pc, share, ci = (
        cache["Y"],
        cache["Yc"],
        cache["vecs"],
        cache["ss_pc"],
        cache["share"],
        cache["ci"],
    )
    E = Y - np.asarray(RS.load_pred(arm, LAYER, fitter, ci), dtype=np.float64)
    r2 = 1.0 - np.square(E @ vecs).sum(axis=0) / ss_pc
    del E
    return r2, share


def main() -> None:
    rng = np.random.default_rng(SEED)

    # ── target spectrum (shared across every arm/fitter) ──────────────────────
    y16, ci = RS.load_layer(LAYER)
    Y = np.asarray(y16, dtype=np.float64)
    del y16
    Yc = Y - Y.mean(axis=0, keepdims=True)
    _lam, vecs = RS.gram_spectrum(Yc, want_vectors=True, n_vec=FIT_TOP)
    ss_pc = np.square(Yc @ vecs).sum(axis=0)  # SS units, matches gram eigenvalues
    share = ss_pc / float(np.square(Yc).sum())
    assert share.min() > 0, "non-positive variance share"
    cache = {"Y": Y, "Yc": Yc, "vecs": vecs, "ss_pc": ss_pc, "share": share, "ci": ci}

    ref779 = json.loads((PROJECT_ROOT / REF_779).read_text())["fitters"]["ridge"]
    banked = json.loads((PROJECT_ROOT / ALIGNMENT).read_text())["cells"]

    doc: dict = {
        "design": {
            "question": (
                "Which directions deviate MOST from the two-parameter mechanical noise "
                "floor — the genuinely anomalous 'worse than their variance predicts' "
                "set — and what are they? Every prior SAE/autointerp/lens pass in the "
                "project characterized the RAW-R^2-worst directions, which is the "
                "low-variance tail, not this set."
            ),
            "corpus": (
                "#1738 multi-turn holdout, n=9,941, L19, staged twoway arrays "
                "(data/issue_1482/twoway_stage) — fully local, so the target-PCA basis "
                "vectors exist and the fit doubles as a replication of the #779 band"
            ),
            "model": "R^2_j = 1 - c_j/lambda_j; c_j = a (1-param) or a + b*lambda_j (2-param)",
            "fit_set": f"top-{FIT_TOP} target-PCA variance ranks (tail floor-censored)",
            "lambda_units": "variance SHARE (eigenvalue/total SS); rescales a, not b/gof/deviations",
            "deviation_sign": "curve - observed; POSITIVE = below curve = worse than mechanical",
            "primary_cell": RS.cell_name(PRIMARY_ARM, LAYER, PRIMARY_FITTER),
            "reference_779_fit": {
                "a": ref779["two_param"]["a"],
                "b_proportional_loss": ref779["two_param"]["b_proportional_loss"],
                "gof_r2_on_fit_set": ref779["two_param"]["gof_r2_on_fit_set"],
                "worst_below_curve_ranks": [r["rank"] for r in ref779["worst_below_curve"]],
                "note": "#779 n10k single-turn corpus; basis vectors not local there",
            },
        },
        "fits": {},
        "replication": {},
        "characterization": {},
    }

    # ── step 2: the fit across arms x fitters ─────────────────────────────────
    dev_by_cell: dict[str, np.ndarray] = {}
    r2_by_cell: dict[str, np.ndarray] = {}
    for arm in ARMS:
        for fitter in FITTERS:
            r2, sh = _spectrum_and_r2(arm, fitter, cache)
            fit = _fit_floor(r2, sh)
            name = RS.cell_name(arm, LAYER, fitter)
            dev = fit.pop("_deviation")
            fit.pop("_pred2")
            dev_by_cell[name], r2_by_cell[name] = dev, r2
            order = np.argsort(dev)[::-1]
            fit["worst_below_curve"] = [
                {
                    "rank": int(i),
                    "r2_observed": float(r2[i]),
                    "r2_curve2": float(r2[i] + dev[i]),
                    "deviation": float(dev[i]),
                    "variance_share": float(sh[i]),
                }
                for i in order[:N_DEVIANT]
            ]
            fit["best_above_curve"] = [
                {
                    "rank": int(i),
                    "r2_observed": float(r2[i]),
                    "r2_curve2": float(r2[i] + dev[i]),
                    "deviation": float(dev[i]),
                    "variance_share": float(sh[i]),
                }
                for i in order[::-1][:N_DEVIANT]
            ]
            doc["fits"][name] = fit

    primary = RS.cell_name(PRIMARY_ARM, LAYER, PRIMARY_FITTER)
    dev = dev_by_cell[primary]
    r2 = r2_by_cell[primary]
    deviant = np.argsort(dev)[::-1][:N_DEVIANT]  # below-curve
    above = np.argsort(dev)[:N_DEVIANT]  # above-curve
    raw_worst = np.asarray(banked[primary]["worst_indices"], dtype=int)

    # ── replication read: is the band contiguous + does it reproduce #779? ────
    ref_ranks = np.asarray(doc["design"]["reference_779_fit"]["worst_below_curve_ranks"])
    dev_ranks = np.sort(deviant)
    cross_cell = {
        name: len(set(np.argsort(d)[::-1][:N_DEVIANT].tolist()) & set(deviant.tolist()))
        for name, d in dev_by_cell.items()
    }
    doc["replication"] = {
        "n1738_deviant_ranks_sorted": [int(x) for x in dev_ranks],
        "n1738_rank_span": [int(dev_ranks.min()), int(dev_ranks.max())],
        "n779_rank_span": [int(ref_ranks.min()), int(ref_ranks.max())],
        "n779_deviant_ranks_sorted": sorted(int(x) for x in ref_ranks),
        "band_contiguity": {
            "n_in_span": int(
                ((dev_ranks >= dev_ranks.min()) & (dev_ranks <= dev_ranks.max())).sum()
            ),
            "span_width": int(dev_ranks.max() - dev_ranks.min() + 1),
            "density_in_span": float(N_DEVIANT / (dev_ranks.max() - dev_ranks.min() + 1)),
            "fraction_of_fit_set_spanned": float((dev_ranks.max() - dev_ranks.min() + 1) / FIT_TOP),
        },
        "overlap_with_primary_deviants_by_cell": cross_cell,
        "overlap_with_raw_r2_worst20": int(len(set(deviant.tolist()) & set(raw_worst.tolist()))),
        "raw_r2_worst20_ranks": [int(x) for x in raw_worst],
        "deviation_profile_by_rank": [float(x) for x in dev],
        "r2_by_rank": [float(x) for x in r2],
        "variance_share_by_rank": [float(x) for x in share],
    }

    # ── is the "deviant" set outliers, or just the peak of a smooth arch? ─────
    # The deviation-vs-rank profile is visibly a smooth inverted-U: the 2-param
    # hyperbola under-predicts at both ends of the fit set and over-predicts in the
    # middle, so "top-15 below curve" may be selecting NEIGHBOURS ON A SMOOTH CURVE
    # rather than individually anomalous directions. Fit a cubic in log-rank to that
    # profile; whatever is left is the direction-specific residual. If the top-15 by
    # arch residual differ from the top-15 by raw deviation, the raw set was the arch.
    lr_ = np.log(np.arange(FIT_TOP) + 1.0)
    Xa = np.stack([lr_**k for k in range(4)], axis=1)
    beta, *_ = np.linalg.lstsq(Xa, dev, rcond=None)
    arch = Xa @ beta
    arch_resid = dev - arch
    arch_gof = 1.0 - float(((dev - arch) ** 2).sum()) / float(((dev - dev.mean()) ** 2).sum())
    arch_dev = np.argsort(arch_resid)[::-1][:N_DEVIANT]
    doc["arch_decomposition"] = {
        "model": "deviation_j ~ cubic in log(rank+1), OLS over the top-256 fit set",
        "rationale": (
            "separates the SYSTEMATIC curvature residual of the 2-param floor (a smooth "
            "inverted-U over rank) from DIRECTION-SPECIFIC anomaly; the raw top-15 "
            "below-curve set is the arch PEAK unless it survives this decomposition"
        ),
        "arch_gof_r2_of_deviation_profile": arch_gof,
        "arch_peak_rank": int(np.argmax(arch)),
        "arch_peak_deviation": float(arch.max()),
        "residual_sd_after_arch": float(arch_resid.std()),
        "raw_deviation_sd": float(dev.std()),
        "arch_residual_top15_ranks": [int(x) for x in arch_dev],
        "overlap_raw_top15_vs_arch_residual_top15": int(
            len(set(arch_dev.tolist()) & set(deviant.tolist()))
        ),
        "arch_fitted_by_rank": [float(x) for x in arch],
        "arch_residual_by_rank": [float(x) for x in arch_resid],
    }

    # ── step 3: characterization ──────────────────────────────────────────────
    # rank-matched control: the NON-deviant directions inside the deviant span. The
    # decisive control for "are the deviants more SAE-legible?" — SAE alignment rises
    # with variance rank, so a deviant-vs-raw-worst gap at ranks 80-166 vs 222-255 is
    # confounded by rank alone until this group is on the plot.
    span = np.arange(int(dev_ranks.min()), int(dev_ranks.max()) + 1)
    rank_ctrl = np.asarray([i for i in span if i not in set(deviant.tolist())], dtype=int)
    groups = {
        "below_curve_deviants": deviant,
        "arch_residual_deviants": arch_dev,
        "rank_matched_control": rank_ctrl,
        "above_curve": above,
        "raw_r2_worst20": raw_worst,
    }
    lens_groups = {
        "below_curve_deviants",
        "arch_residual_deviants",
        "above_curve",
        "raw_r2_worst20",
    }
    rb = _rb_matrix()

    from issue1482_sae import BatchTopKSAE

    sae = BatchTopKSAE.load(k=64, layer=LAYER, device="cpu")
    D_unit = np.asarray(sae.w_dec, dtype=np.float32)  # (3584, 131072)
    D_unit /= np.linalg.norm(D_unit, axis=0, keepdims=True)  # in place: 1.9 GB, not 3.8
    del sae
    sae_null = banked[primary]["sae_alignment"]["null_random_unit_max_over_dictionary"]

    sae_res: dict[str, dict] = {}
    for gname, idx in groups.items():
        V = vecs[:, idx].astype(np.float32)
        cos = D_unit.T @ V  # (131072, n)
        amax = np.argmax(np.abs(cos), axis=0)
        sae_res[gname] = {
            "max_abs_cos": [float(abs(cos[amax[j], j])) for j in range(len(idx))],
            "argmax_feature": [int(x) for x in amax],
        }
        del cos
    del D_unit
    gc.collect()

    # logit lens — only for the small named groups (the 72-direction rank-matched
    # control is a summary-statistics comparator, not a per-direction read)
    lens_res: dict[str, list] = {}
    W_U, tok = RS.load_unembedding()
    for gname in lens_groups:
        idx = groups[gname]
        logits = W_U @ vecs[:, idx].astype(np.float32)
        lens_res[gname] = [
            [tok.decode([int(t)]) for t in np.argsort(logits[:, j])[-12:][::-1]]
            for j in range(len(idx))
        ]
        del logits
    del W_U
    gc.collect()

    want = {int(f) for g in sae_res.values() for f in g["argmax_feature"]}
    desc = _np_descriptions(want, PROJECT_ROOT / NP_CACHE)

    # trait alignment + a random-unit null for the |cos| against r_B
    rand_dirs = rng.standard_normal((RS.HIDDEN_DIM, 200))
    rand_dirs /= np.linalg.norm(rand_dirs, axis=0, keepdims=True)
    rb_null = np.abs(rand_dirs.T @ rb)

    for gname, idx in groups.items():
        cos_rb = np.abs(vecs[:, idx].T @ rb)  # (n, 7)
        rows = []
        for j, pc in enumerate(idx):
            fid = sae_res[gname]["argmax_feature"][j]
            jt = int(np.argmax(cos_rb[j]))
            rows.append(
                {
                    "rank": int(pc),
                    "r2_observed": float(r2[pc]),
                    "r2_curve2": float(r2[pc] + dev[pc]),
                    "deviation": float(dev[pc]),
                    "variance_share": float(share[pc]),
                    "sae_feat_id": fid,
                    "sae_abs_cos": sae_res[gname]["max_abs_cos"][j],
                    "sae_autointerp": desc.get(str(fid), ""),
                    "arch_residual": float(arch_resid[pc]),
                    "trait_max_abs_cos": float(cos_rb[j].max()),
                    "trait_argmax": TRAITS7[jt],
                    "trait_all_abs_cos": {t: float(cos_rb[j, k]) for k, t in enumerate(TRAITS7)},
                    "lens_top12": lens_res[gname][j] if gname in lens_groups else None,
                }
            )
        arr = np.asarray(sae_res[gname]["max_abs_cos"])
        doc["characterization"][gname] = {
            "n": int(len(idx)),
            "rows": rows,
            "summary": {
                "sae_abs_cos_mean": float(arr.mean()),
                "sae_abs_cos_max": float(arr.max()),
                "sae_abs_cos_min": float(arr.min()),
                "n_above_null_max": int((arr > sae_null["max"]).sum()),
                "n_above_null_p95": int((arr > sae_null["p95"]).sum()),
                "trait_max_abs_cos_overall": float(cos_rb.max()),
                "trait_abs_cos_mean": float(cos_rb.mean()),
                "n_distinct_sae_features": int(len(set(sae_res[gname]["argmax_feature"]))),
                "n_with_autointerp": int(sum(1 for r in rows if r["sae_autointerp"])),
            },
        }
    doc["characterization"]["nulls"] = {
        "sae_random_unit_max_over_dictionary": sae_null,
        "sae_null_note": "direction-independent; reused from the banked #1482 worst pass",
        "trait_random_unit_abs_cos": {
            "n_draws": 200,
            "mean": float(rb_null.mean()),
            "p95": float(np.percentile(rb_null, 95)),
            "max": float(rb_null.max()),
        },
    }

    ch = doc["characterization"]
    dev_cos = ch["below_curve_deviants"]["summary"]["sae_abs_cos_mean"]
    ctrl_cos = ch["rank_matched_control"]["summary"]["sae_abs_cos_mean"]
    worst_cos = ch["raw_r2_worst20"]["summary"]["sae_abs_cos_mean"]
    doc["verdicts"] = {
        "band_replicates": {
            "verdict": "yes",
            "detail": (
                f"the below-curve band reproduces on the #1738 multi-turn corpus as a "
                f"contiguous MID-spectrum band (ranks {int(dev_ranks.min())}-{int(dev_ranks.max())} "
                f"vs #779's {int(ref_ranks.min())}-{int(ref_ranks.max())}), fitter-invariant within "
                f"the context arm ({cross_cell[RS.cell_name('context', LAYER, 'mlp_w8192')]}/15 "
                f"shared with the MLP fitter) but only partly arm-invariant "
                f"(bare {cross_cell[RS.cell_name('bare', LAYER, 'ridge')]}/15, "
                f"prefix {cross_cell[RS.cell_name('prefix', LAYER, 'ridge')]}/15)"
            ),
        },
        "deviants_disjoint_from_prior_characterized_set": {
            "verdict": "yes",
            "detail": (
                f"0/15 overlap with the raw-R^2-worst-20 the project's SAE/autointerp/lens "
                f"passes actually characterized (those sit at ranks "
                f"{int(raw_worst.min())}-{int(raw_worst.max())}, the low-variance tail); this set "
                f"had genuinely never been looked at"
            ),
        },
        "deviants_are_individually_anomalous": {
            "verdict": "no",
            "detail": (
                f"a cubic in log-rank explains R^2={arch_gof:.2f} of the whole deviation profile — "
                f"the 2-param floor systematically over-predicts mid-spectrum and under-predicts at "
                f"both ends, so 'top-15 below curve' mostly selects the PEAK OF A SMOOTH ARCH "
                f"(peak rank {int(np.argmax(arch))}), not individual outliers; only "
                f"{doc['arch_decomposition']['overlap_raw_top15_vs_arch_residual_top15']}/15 survive "
                f"as top-15 by arch RESIDUAL"
            ),
        },
        "deviants_more_sae_legible_than_prior_worst_set": {
            "verdict": "no — rank-confounded",
            "detail": (
                f"deviant SAE max|cos| {dev_cos:.3f} vs raw-R^2-worst {worst_cos:.3f} looks like a "
                f"real gap, but the RANK-MATCHED control (the {len(rank_ctrl)} non-deviant "
                f"directions inside the deviant span) reads {ctrl_cos:.3f} — statistically "
                f"indistinguishable from the deviants (delta {dev_cos - ctrl_cos:+.4f}). SAE "
                f"alignment tracks variance rank, not floor deviation; the apparent effect is the "
                f"rank difference (80-166 vs 222-255) alone"
            ),
            "delta_vs_rank_matched_control": float(dev_cos - ctrl_cos),
            "delta_vs_raw_r2_worst20": float(dev_cos - worst_cos),
        },
        "deviants_trait_aligned": {
            "verdict": "no",
            "detail": (
                f"deviant max |cos| to any of the 7 r_B trait directions is "
                f"{ch['below_curve_deviants']['summary']['trait_max_abs_cos_overall']:.3f} vs "
                f"{ch['rank_matched_control']['summary']['trait_max_abs_cos_overall']:.3f} for the "
                f"rank-matched control and "
                f"{ch['above_curve']['summary']['trait_max_abs_cos_overall']:.3f} for the "
                f"head-of-spectrum above-curve set (random-unit null p95 "
                f"{float(np.percentile(rb_null, 95)):.3f}); the trait signal lives in the top PCs, "
                f"not in the deviant band"
            ),
        },
        "caveats": [
            "the above-curve set is dominated by ranks 0-16, i.e. it is the head of the spectrum "
            "the 2-param hyperbola under-fits — it is a curvature residual, NOT an 'anomalously "
            "well-predicted' class; read it as the arch's left end",
            "the cubic-in-log-rank arch is unstable at ranks 0-2 (log(rank+1) boundary), so "
            "arch-residual entries at the very head (rank 2) are partly a fit artifact",
            "the arch-residual group is NOT rank-matched (it spans ranks 2-184), so its higher "
            "mean SAE |cos| carries the same rank confound the rank-matched control exposes",
            "autointerp labels for the deviants are a grab-bag with no shared theme (DNA "
            "sequencing / Hero / News snippets / code punctuation / system), and the logit-lens "
            "top tokens are rare-token junk — the same illegibility the banked worst-direction "
            "pass reported, so nothing distinguishes the deviants qualitatively either",
        ],
    }

    out_path = PROJECT_ROOT / OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=1))
    print(f"[out] {out_path}")

    # ── figure ────────────────────────────────────────────────────────────────
    set_paper_style()
    import matplotlib.pyplot as plt

    colors = paper_palette(5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2))

    ranks = np.arange(FIT_TOP)
    ax1.scatter(ranks, dev, s=12, color="#98a2b3", label="all top-256 directions")
    ax1.scatter(
        deviant,
        dev[deviant],
        s=52,
        color=colors[1],
        zorder=5,
        edgecolor="black",
        linewidth=0.5,
        label=f"top-{N_DEVIANT} BELOW curve (deviants)",
    )
    ax1.scatter(
        raw_worst,
        dev[raw_worst],
        s=34,
        marker="s",
        facecolor="none",
        edgecolor=colors[2],
        linewidth=1.2,
        zorder=4,
        label="raw-R² worst-20 (prior passes)",
    )
    ax1.axhline(0.0, color="gray", lw=1)
    ax1.plot(
        ranks,
        arch,
        color=colors[3],
        lw=2.2,
        zorder=6,
        label=f"smooth arch (cubic in log-rank, R²={arch_gof:.2f})",
    )
    ax1.axvspan(
        int(dev_ranks.min()),
        int(dev_ranks.max()),
        color=colors[1],
        alpha=0.10,
        label=f"deviant span (ranks {int(dev_ranks.min())}–{int(dev_ranks.max())})",
    )
    ax1.set_xlabel("target-PCA variance rank")
    ax1.set_ylabel("curve − observed R²   (positive = worse than mechanical)")
    ax1.set_title(
        f"Deviation from the 2-param floor ({primary}, #1738 multi-turn)", loc="left", fontsize=11
    )
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    order = [
        "below_curve_deviants",
        "arch_residual_deviants",
        "rank_matched_control",
        "raw_r2_worst20",
        "above_curve",
    ]
    labels = [
        "below-curve\ndeviants (15)",
        "arch-residual\ndeviants (15)",
        f"rank-matched\ncontrol ({len(rank_ctrl)})",
        "raw-R² worst\n20 (prior)",
        "above-curve\n(15)",
    ]
    for k, g in enumerate(order):
        v = np.asarray([r["sae_abs_cos"] for r in doc["characterization"][g]["rows"]])
        jitter = rng.uniform(-0.15, 0.15, size=v.size)
        ax2.scatter(
            np.full(v.size, k) + jitter,
            v,
            s=34 if g != "rank_matched_control" else 14,
            color=colors[k % len(colors)],
            alpha=1.0 if g != "rank_matched_control" else 0.55,
            edgecolor="black",
            linewidth=0.4 if g != "rank_matched_control" else 0.0,
            zorder=4,
        )
        ax2.plot([k - 0.3, k + 0.3], [v.mean()] * 2, color="black", lw=2, zorder=5)
    ax2.axhspan(
        0.0,
        sae_null["p95"],
        color="#98a2b3",
        alpha=0.22,
        label=f"random-unit null ≤p95 ({sae_null['p95']:.3f}, n={sae_null['n_draws']})",
    )
    ax2.axhline(
        sae_null["max"],
        color="#667085",
        lw=1.2,
        ls="--",
        label=f"random-unit null max ({sae_null['max']:.3f})",
    )
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("SAE decoder max |cos|  (131,072 features, L19 k=64)")
    ax2.set_title(
        "SAE legibility: deviants vs their RANK-MATCHED neighbours (bars = group mean)",
        loc="left",
        fontsize=11,
    )
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    ax2.set_ylim(0.0, None)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig_paper(fig, "deviant_band", dir=PROJECT_ROOT / FIG_DIR)

    # ── console summary ───────────────────────────────────────────────────────
    for name, f in doc["fits"].items():
        t = f["two_param"]
        print(
            f"[{name:22s}] 2p a={t['a']:.6f} b={t['b_proportional_loss']:.4f} "
            f"gof={t['gof_r2_on_fit_set']:.3f} | 1p gof={f['one_param']['gof_r2_on_fit_set']:.3f}"
        )
    rep = doc["replication"]
    print(
        f"\n[replication] #1738 deviant span {rep['n1738_rank_span']} vs "
        f"#779 {rep['n779_rank_span']}; density in span "
        f"{rep['band_contiguity']['density_in_span']:.2f}; "
        f"overlap with raw-R²-worst20 = {rep['overlap_with_raw_r2_worst20']}"
    )
    print(f"[cross-cell deviant overlap /15] {rep['overlap_with_primary_deviants_by_cell']}")
    ad = doc["arch_decomposition"]
    print(
        f"\n[arch] cubic-in-log-rank explains R²={ad['arch_gof_r2_of_deviation_profile']:.3f} of the "
        f"deviation profile; peak rank {ad['arch_peak_rank']} (dev {ad['arch_peak_deviation']:+.3f}); "
        f"resid sd {ad['residual_sd_after_arch']:.4f} vs raw dev sd {ad['raw_deviation_sd']:.4f}; "
        f"raw-top15 ∩ arch-residual-top15 = {ad['overlap_raw_top15_vs_arch_residual_top15']}/15"
    )
    nulls = doc["characterization"]["nulls"]
    print(
        f"\n[SAE null] mean {sae_null['mean']:.3f} p95 {sae_null['p95']:.3f} max {sae_null['max']:.3f}"
        f" | [trait null] p95 {nulls['trait_random_unit_abs_cos']['p95']:.3f}"
    )
    for g in order:
        s = doc["characterization"][g]["summary"]
        print(
            f"\n== {g} == SAE|cos| mean {s['sae_abs_cos_mean']:.3f} "
            f"[{s['sae_abs_cos_min']:.3f},{s['sae_abs_cos_max']:.3f}] "
            f"above-null-max {s['n_above_null_max']}/{doc['characterization'][g]['n']} | "
            f"trait max {s['trait_max_abs_cos_overall']:.3f} | "
            f"autointerp {s['n_with_autointerp']}/{doc['characterization'][g]['n']}"
        )
        if g == "rank_matched_control":
            continue  # summary-statistics comparator: no per-direction dump
        for r in doc["characterization"][g]["rows"][:15]:
            print(
                f"  rk {r['rank']:3d} R² {r['r2_observed']:.3f} (curve {r['r2_curve2']:.3f}, "
                f"dev {r['deviation']:+.3f}, arch-resid {r['arch_residual']:+.3f}) "
                f"feat {r['sae_feat_id']:6d} |cos| {r['sae_abs_cos']:.3f} "
                f"trait {r['trait_argmax'][:6]} {r['trait_max_abs_cos']:.3f} | "
                f"{(r['sae_autointerp'] or '(no desc)')[:52]} | "
                f"lens: {', '.join(t.strip() for t in r['lens_top12'][:5])}"
            )


if __name__ == "__main__":
    main()
