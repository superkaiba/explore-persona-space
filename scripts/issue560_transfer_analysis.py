# ruff: noqa: RUF001, RUF003
# Intentional Unicode (rho, ※, Δ, −, —) in scientific docstrings + labels.
"""Task #560 — Phase B (VM, CPU): cross-recipe transfer analysis.

Builds the 560 run x persona aggregate panel from the pod-side four-float
JSONs (``issue560_crossrecipe_panel.py`` phases S1/S2) + the geometry JSON,
asserts the exposure classification (3 source-resident / 45 trained-negative
/ 512 never-negative), then runs the registered #553-convention inference:

(R1) pair-corrected (two-way run+persona FE re-estimated per resample)
     Spearman of min_dist vs {dz, dz_eos, dmargin, margin_trained,
     margin_base} on the 557 non-source-resident aggregates — FE-residual
     permutation p (10,000), cell bootstrap (10,000), run-cluster (16) and
     persona-cluster (35) bootstraps (2,000 each), primary CI = the WIDER
     cluster CI; strata reads: never-negative-only (512) and
     A-class-source-only (172); sensitivity: source-resident cells included.
(R2) persistence (persona-FE of trained z_eos vs persona-mean base z_eos,
     with the variance-ratio + calibrated-null read the critics registered)
     + clamp routing (persona-FE of dz_eos vs persona-mean margin_base) +
     the partial check (controlling persona-mean base z_eos level).
(R3) exposure contrast: trained-negative (45) vs never-negative (512) mean
     dz_eos with run-cluster (verdict-bearing) + persona-cluster
     (descriptive; 3 clusters) CIs; parent anchors read from the committed
     ``eval_results/issue_553/exposure.json`` (qualitative reference only).
(R4, exploratory) two-way Type-I variance shares of dz with order swap.

Holm runs over the 6 registered p-carrying members (5 min_dist targets +
clamp routing); verdicts are CI-only, Holm p's are diagnostics.

Statistical machinery imported (never copied) from ``issue553_panel`` (which
patches the fast exact two-way solver into ``issue539_corrected_reads_
inference`` at import) + the #539 modules. Smoke = this exact script with
reduced ``--n-boot/--n-cluster-boot/--n-marginal-boot/--n-perm`` on a tiny
panel (``--tag``-suffixed dirs from the panel driver, or a synthetic
fixture).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import issue539_corrected_reads_inference as i539inf
import issue539_residual_per_cohort as i539
import issue553_panel as p553
import matplotlib.pyplot as plt
import numpy as np
from issue560_crossrecipe_panel import (
    HELD_OUT_35,
    PROJECT_ROOT,
    SCHEMA_VERSION,
    SOURCES_ALL,
    TIE_BACK_MATRIX,
    _git_commit,
    assert_held_out_matches_logit_rescore,
    assert_strata_partition,
    classify_exposure,
    exposure_stratum,
    load_persona_prompts,
)

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

MARKER_ID = p553.MARKER_ID
EOS_ID = p553.EOS_ID
MIN_DIST_TARGETS = ("dz", "dz_eos", "dmargin", "margin_trained", "margin_base")
PARENT_POINTS = {"dz": -0.240, "dmargin": -0.271}  # #553 transfer_478 cell-axis primaries
I553_EXPOSURE_JSON = PROJECT_ROOT / "eval_results/issue_553/exposure.json"

# Reader-facing display maps (round-1 interp-critique: no internal labels /
# bare condition codes on rendered figures — paper-plots skill section 3.5).
ADAPTER_CLASS_GLOSS = {
    "A": "persona prompt",
    "B": "query wrap",
    "C": "few-shot scaffold",
    "D": "register rephrase",
}
STRATUM_DISPLAY = {
    "never_negative": "never a training negative (512 cells)",
    "trained_negative": "training negative under the other adapters (45 cells)",
    "source_resident": "source-resident positive control (3 cells)",
}
MIN_DIST_XLABEL = "cosine distance to the run's source context (layer 20)"


# ── Panel construction ───────────────────────────────────────────────────────


def _per_q_array(per_q: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in per_q], dtype=np.float64)


def build_panel(ff_dir: Path, geometry_path: Path) -> dict:
    """The run x persona aggregate panel from the four-float + geometry JSONs.

    Fails loud on: missing files, schema drift, panel-spec mismatch across
    sides, per-q slot misalignment (slot_kind / n_truncated_tokens must be
    identical across base and trained sides — same R, same truncation by
    construction), and exposure-classification drift.
    """
    geometry = json.loads(geometry_path.read_text())
    assert geometry["schema_version"] == SCHEMA_VERSION, geometry.get("schema_version")
    sources: list[str] = geometry["sources"]
    personas: list[str] = geometry["personas"]
    min_dist_map = geometry["min_dist"]

    persona_prompts = load_persona_prompts()
    matches = classify_exposure({p: persona_prompts[p] for p in personas})
    strata_counts = assert_strata_partition(sources, personas, matches)

    cols: dict[str, list] = {
        k: []
        for k in (
            "source_cid",
            "persona",
            "exposure",
            "min_dist",
            "dz",
            "dz_eos",
            "dmargin",
            "margin_trained",
            "margin_base",
            "dlogp",
            "trained_logp",
            "base_logp_matched",
            "z_eos_trained",
            "z_eos_base",
            "dlogZ",
            "delta_p",
            "pre_marker_frac",
            "n_eor_slots",
            "dz_eor_only",
            "dmargin_eor_only",
            "argmax_marker_rate_trained",
            "argmax_eos_rate_trained",
            "argmax_marker_rate_base",
            "argmax_eos_rate_base",
            "gen_trunc_rate",
            "mean_n_new_tokens",
            "n_nontrunc_q",
            "dz_nontrunc",
            "dmargin_nontrunc",
        )
    }
    n_q_seen: set[int] = set()
    for cid in sources:
        t_payload = json.loads((ff_dir / f"trained_{cid}.json").read_text())
        b_payload = json.loads((ff_dir / f"base_{cid}.json").read_text())
        for payload, phase in (
            (t_payload, "S2_trained_on_own_R"),
            (b_payload, "S1_base_matched_slot"),
        ):
            assert payload["schema_version"] == SCHEMA_VERSION, (cid, payload.get("schema_version"))
            assert payload["phase"] == phase, (cid, payload["phase"])
        assert t_payload["panel_spec"] == b_payload["panel_spec"], (
            f"{cid}: spec differs across sides"
        )
        for p in personas:
            tq = t_payload["per_persona"][p]["per_q"]
            bq = b_payload["per_persona"][p]["per_q"]
            assert len(tq) == len(bq), (cid, p, len(tq), len(bq))
            n_q_seen.add(len(tq))
            for x, y in zip(tq, bq, strict=True):
                assert x["slot_kind"] == y["slot_kind"], (cid, p)
                assert x["n_truncated_tokens"] == y["n_truncated_tokens"], (cid, p)
            zm_t, ze_t = _per_q_array(tq, "z_marker"), _per_q_array(tq, "z_eos")
            zm_b, ze_b = _per_q_array(bq, "z_marker"), _per_q_array(bq, "z_eos")
            lp_t, lp_b = _per_q_array(tq, "logp_marker"), _per_q_array(bq, "logp_marker")
            lz_t, lz_b = _per_q_array(tq, "logZ"), _per_q_array(bq, "logZ")
            am_t = np.array([r["argmax_id"] for r in tq], dtype=np.int64)
            am_b = np.array([r["argmax_id"] for r in bq], dtype=np.int64)
            is_eor = np.array([r["slot_kind"] == "end_of_response" for r in tq], dtype=bool)
            cols["source_cid"].append(cid)
            cols["persona"].append(p)
            cols["exposure"].append(exposure_stratum(cid, p, matches))
            cols["min_dist"].append(float(min_dist_map[cid][p]))
            cols["dz"].append(float(np.mean(zm_t - zm_b)))
            cols["dz_eos"].append(float(np.mean(ze_t - ze_b)))
            cols["dmargin"].append(float(np.mean((zm_t - ze_t) - (zm_b - ze_b))))
            cols["margin_trained"].append(float(np.mean(zm_t - ze_t)))
            cols["margin_base"].append(float(np.mean(zm_b - ze_b)))
            cols["dlogp"].append(float(np.mean(lp_t - lp_b)))
            cols["trained_logp"].append(float(np.mean(lp_t)))
            cols["base_logp_matched"].append(float(np.mean(lp_b)))
            cols["z_eos_trained"].append(float(np.mean(ze_t)))
            cols["z_eos_base"].append(float(np.mean(ze_b)))
            cols["dlogZ"].append(float(np.mean(lz_t - lz_b)))
            # Probability-space sanity (marker rule): ΔP = P_base·(e^{ΔlogP}−1).
            cols["delta_p"].append(float(np.mean(np.exp(lp_b) * (np.exp(lp_t - lp_b) - 1.0))))
            cols["pre_marker_frac"].append(float(np.mean(~is_eor)))
            cols["n_eor_slots"].append(int(is_eor.sum()))
            cols["dz_eor_only"].append(
                float(np.mean((zm_t - zm_b)[is_eor])) if is_eor.any() else float("nan")
            )
            cols["dmargin_eor_only"].append(
                float(np.mean(((zm_t - ze_t) - (zm_b - ze_b))[is_eor]))
                if is_eor.any()
                else float("nan")
            )
            cols["argmax_marker_rate_trained"].append(float(np.mean(am_t == MARKER_ID)))
            cols["argmax_eos_rate_trained"].append(float(np.mean(am_t == EOS_ID)))
            cols["argmax_marker_rate_base"].append(float(np.mean(am_b == MARKER_ID)))
            cols["argmax_eos_rate_base"].append(float(np.mean(am_b == EOS_ID)))
            nontrunc = ~np.array([r["gen_truncated"] for r in tq], dtype=bool)
            cols["gen_trunc_rate"].append(float(np.mean(~nontrunc)))
            cols["mean_n_new_tokens"].append(float(np.mean([r["n_new_tokens"] for r in tq])))
            # Truncation-excluded re-aggregation (sensitivity_length_truncation):
            # the same cell aggregates over only the non-truncated questions.
            cols["n_nontrunc_q"].append(int(nontrunc.sum()))
            cols["dz_nontrunc"].append(
                float(np.mean((zm_t - zm_b)[nontrunc])) if nontrunc.any() else float("nan")
            )
            cols["dmargin_nontrunc"].append(
                float(np.mean(((zm_t - ze_t) - (zm_b - ze_b))[nontrunc]))
                if nontrunc.any()
                else float("nan")
            )

    panel = {
        k: (np.asarray(v) if k in ("source_cid", "persona", "exposure") else np.asarray(v, float))
        for k, v in cols.items()
    }
    panel["n_eor_slots"] = np.asarray(cols["n_eor_slots"], dtype=np.int64)
    panel["n_nontrunc_q"] = np.asarray(cols["n_nontrunc_q"], dtype=np.int64)
    panel["_n"] = len(panel["dz"])
    panel["_sources"] = sources
    panel["_personas"] = personas
    panel["_n_questions"] = n_q_seen.pop() if len(n_q_seen) == 1 else sorted(n_q_seen)
    panel["_strata_counts"] = strata_counts
    panel["_geometry"] = geometry
    assert panel["_n"] == len(sources) * len(personas), panel["_n"]
    return panel


def panel_masks(panel: dict) -> dict[str, np.ndarray]:
    """The registered analysis masks (primary excludes source-resident cells)."""
    sr = panel["exposure"] == "source_resident"
    a_class = np.isin(panel["source_cid"], ("A1", "A2", "A3", "A4", "A5"))
    return {
        "primary_non_source_resident": ~sr,
        "never_negative_only": panel["exposure"] == "never_negative",
        "a_class_source_only": a_class & ~sr,
        "sensitivity_all_cells": np.ones(panel["_n"], dtype=bool),
        "trained_negative_only": panel["exposure"] == "trained_negative",
        "source_resident_only": sr,
    }


# ── (R1) pair-corrected min_dist reads ───────────────────────────────────────


def min_dist_reads(
    panel: dict,
    mask: np.ndarray,
    args,
    *,
    full_inference: bool,
    targets: tuple[str, ...] = MIN_DIST_TARGETS,
    tag: str = "R1",
) -> dict:
    """Two-way-FE-corrected Spearman of min_dist vs the targets.

    ``full_inference`` adds the cell bootstrap + both cluster bootstraps +
    the FE-residual permutation p (#553 ``_min_dist_corrected_reads``
    conventions, byte-level: same helper functions, same seed discipline);
    point-only mode is used for the sensitivity slices. ``targets``/``tag``
    default to the registered R1 read; the truncation-excluded sensitivity
    re-runs the identical machinery on the re-aggregated columns.
    """
    x = panel["min_dist"][mask]
    run_l = panel["source_cid"][mask]
    per_l = panel["persona"][mask]
    run_u, rc = np.unique(run_l, return_inverse=True)
    per_u, pc = np.unique(per_l, return_inverse=True)
    x_tw, _ = i539._twoway_fe_residualize(x, run_l, per_l)
    reads: dict = {}
    for tgt in targets:
        y = panel[tgt][mask]
        y_tw, _ = i539._twoway_fe_residualize(y, run_l, per_l)
        # Fast-path equivalence assert on the observed data (#539 convention).
        x_f, y_f = i539inf._twoway_resid_pair(x, y, rc, pc, len(run_u), len(per_u))
        drift = max(float(np.max(np.abs(x_tw - x_f))), float(np.max(np.abs(y_tw - y_f))))
        assert drift < 1e-8, f"fast two-way residual drift {drift!r} on {tgt}"
        rho = i539._spearman_rho(x_tw, y_tw)
        block: dict = {
            "estimate": float(rho),
            "n_cells": int(mask.sum()),
            "sign_direction": "min_dist is a DISTANCE (larger = farther); the parent's "
            "#553/#478 reads use the same metric — SAME expected signs "
            f"(parent points: {PARENT_POINTS})",
        }
        if full_inference:
            p_perm = i539._permutation_p(y_tw, x_tw, args.n_perm, args.seed)
            p_perm["method"] = (
                "FE residual permutation: min_dist FE-residuals permuted across run x persona "
                "aggregates against fixed DV FE-residuals (both sides projected on run + "
                "persona dummies first)"
            )
            cis = {
                "cluster_run": i539inf._cluster_boot_twoway(
                    x, y, run_l, per_l, "source", args.n_cluster_boot, args.seed
                ),
                "cluster_persona": i539inf._cluster_boot_twoway(
                    x, y, run_l, per_l, "bystander", args.n_cluster_boot, args.seed
                ),
            }
            block.update(
                {
                    "ci95_cell_boot": i539inf._cell_boot_twoway(
                        x, y, rc, pc, args.n_boot, args.seed
                    ),
                    "ci95_cluster_run": cis["cluster_run"],
                    "ci95_cluster_persona": cis["cluster_persona"],
                    "primary_ci": p553.wider_ci(cis),
                    "p_perm_fe": p_perm,
                }
            )
            print(
                f"[{tag}] min_dist vs {tgt}: rho_twoway={rho:+.3f} "
                f"primary CI [{block['primary_ci']['low']:+.3f}, "
                f"{block['primary_ci']['high']:+.3f}] ({block['primary_ci']['axis']}) "
                f"p={p_perm['p']:.4g} (n={int(mask.sum())})"
            )
        reads[tgt] = block
    return reads


def slot_kind_sensitivity(panel: dict, mask: np.ndarray, args) -> dict:
    """End-of-response-only sensitivity for H1 + slot-kind-vs-distance reads.

    pre_marker slots fire exactly where the marker is emitted (critic concern
    5), so the composition correlates with leakage by construction; this
    block separates "geometry routes emission" from "geometry routes graded
    sub-threshold pressure".
    """
    n_q = panel["_n_questions"]
    min_keep = max(3, int(n_q) // 4) if isinstance(n_q, int) else 3
    valid = mask & (panel["n_eor_slots"] >= min_keep)
    out: dict = {
        "min_eor_slots_per_cell": min_keep,
        "n_cells_kept": int(valid.sum()),
        "n_cells_dropped": int(mask.sum() - valid.sum()),
    }
    x = panel["min_dist"][valid]
    run_l, per_l = panel["source_cid"][valid], panel["persona"][valid]
    for tgt in ("dz_eor_only", "dmargin_eor_only"):
        y = panel[tgt][valid]
        keep = ~np.isnan(y)
        x_tw, _ = i539._twoway_fe_residualize(x[keep], run_l[keep], per_l[keep])
        y_tw, _ = i539._twoway_fe_residualize(y[keep], run_l[keep], per_l[keep])
        out[tgt] = {
            "rho_twoway": i539._spearman_rho(x_tw, y_tw),
            "p_perm_fe": i539._permutation_p(y_tw, x_tw, args.n_perm, args.seed),
        }
    # Slot-kind composition vs distance (cell level + per stratum + terciles).
    frac = panel["pre_marker_frac"][mask]
    md = panel["min_dist"][mask]
    out["pre_marker_frac_vs_min_dist_rho"] = i539._spearman_rho(md, frac)
    terciles = np.quantile(md, [1 / 3, 2 / 3])
    t_codes = np.digitize(md, terciles)
    out["pre_marker_frac_by_distance_tercile"] = {
        f"tercile_{i}": float(np.mean(frac[t_codes == i])) for i in range(3)
    }
    out["pre_marker_frac_by_stratum"] = {
        s: float(np.mean(panel["pre_marker_frac"][panel["exposure"] == s]))
        for s in ("source_resident", "trained_negative", "never_negative")
    }
    return out


# ── Length / truncation sensitivity (round-1 critic follow-up) ───────────────


def _fe_resid_cols(
    cols: list[np.ndarray], sc: np.ndarray, bc: np.ndarray, n_s: int, n_b: int
) -> list[np.ndarray]:
    """Exact two-way FE residuals of k columns in one lstsq (k RHS columns).

    Same design matrix as ``i539inf._twoway_resid_pair`` (intercept + run
    dummies + persona dummies), generalized to k right-hand sides so the
    length-partial read residualizes (min_dist, DV, mean_len) jointly.
    Equivalence to ``i539._twoway_fe_residualize`` is asserted on the
    observed data at the call site.
    """
    n = len(cols[0])
    design = np.zeros((n, 1 + n_s + n_b), dtype=np.float64)
    design[:, 0] = 1.0
    rows = np.arange(n)
    design[rows, 1 + sc] = 1.0
    design[rows, 1 + n_s + bc] = 1.0
    rhs = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    resid = rhs - design @ coef
    return [resid[:, i] for i in range(rhs.shape[1])]


def _cell_boot_partial_fe(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sc: np.ndarray,
    bc: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """Cell-level percentile bootstrap on the FE rank-partial: the two-way FE
    residualization of all three variables AND the rank partial are re-run
    within every resample (mirrors ``i539inf._cell_boot_twoway``)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    n_s, n_b = int(sc.max()) + 1, int(bc.max()) + 1
    rhos: list[float] = []
    n_deg = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xt, yt, zt = _fe_resid_cols([x[idx], y[idx], z[idx]], sc[idx], bc[idx], n_s, n_b)
        r = i539inf._fast_partial_spearman(xt, yt, zt)
        if np.isnan(r):
            n_deg += 1
            continue
        rhos.append(r)
    return i539inf._percentile_summary(rhos, n_boot, n_deg)


def _cluster_boot_partial_fe(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    src: np.ndarray,
    byst: np.ndarray,
    cluster_on: str,
    n_boot: int,
    seed: int,
) -> dict:
    """Cluster percentile bootstrap on the FE rank-partial; drawn cluster
    copies are RELABELED as distinct groups on the resampled axis (same
    convention as ``i539inf._cluster_boot_twoway``)."""
    rng = np.random.default_rng(seed)
    labels = src if cluster_on == "source" else byst
    other = byst if cluster_on == "source" else src
    uniq = np.unique(labels)
    idx_of = {c: np.where(labels == c)[0] for c in uniq}
    _, other_codes_full = np.unique(other, return_inverse=True)
    n_other = int(other_codes_full.max()) + 1
    rhos: list[float] = []
    n_deg = 0
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[c] for c in chosen])
        copy_codes = np.repeat(np.arange(len(chosen)), [len(idx_of[c]) for c in chosen])
        oc = other_codes_full[idx]
        trip = [x[idx], y[idx], z[idx]]
        if cluster_on == "source":
            xt, yt, zt = _fe_resid_cols(trip, copy_codes, oc, len(chosen), n_other)
        else:
            xt, yt, zt = _fe_resid_cols(trip, oc, copy_codes, n_other, len(chosen))
        r = i539inf._fast_partial_spearman(xt, yt, zt)
        if np.isnan(r):
            n_deg += 1
            continue
        rhos.append(r)
    out = i539inf._percentile_summary(rhos, n_boot, n_deg)
    out["n_clusters"] = len(uniq)
    return out


def length_truncation_sensitivity(panel: dict, mask: np.ndarray, args) -> dict:
    """Round-1 interpretation-critic follow-up: the response-length confound.

    ``length_partial`` (PRIMARY new read): the two headline pair-corrected
    two-way-FE rank reads (min_dist vs dz / dmargin, primary mask) with the
    cell-level mean generated-token count partialled out — all three
    variables are two-way-FE-residualized (run + persona dummies), then the
    rank partial (``i539inf._fast_partial_spearman``) controls the length
    residual; the cell bootstrap and both cluster bootstraps re-run the
    residualization + partial within every resample. ``trunc_excluded``
    (SECONDARY, registered-form check): each cell's dz/dmargin re-aggregated
    over only non-truncated generations (min-N guard: >= half the questions),
    then the registered R1 machinery re-run on the re-aggregated columns.
    Both are EXPLORATORY sensitivity reads — NOT registered family members;
    the Holm family is unchanged.
    """
    x = panel["min_dist"][mask]
    z = panel["mean_n_new_tokens"][mask]
    run_l, per_l = panel["source_cid"][mask], panel["persona"][mask]
    run_u, rc = np.unique(run_l, return_inverse=True)
    per_u, pc = np.unique(per_l, return_inverse=True)
    n_q = panel["_n_questions"]

    # (1) Length-partialled rank-partial on the headline members.
    length_partial: dict = {
        "control": "cell-level mean response length: mean over the cell's "
        f"{n_q} questions of the generated-token count (n_new_tokens)",
        "method": "two-way FE residualization (run + persona dummies, exact lstsq — same "
        "design as the registered R1 reads) of min_dist, the DV, AND mean response "
        "length; then rank-partial Spearman of (min_dist, DV | mean_len) on the FE "
        "residuals; cell bootstrap + run/persona cluster bootstraps re-run the "
        "residualization and the partial within every resample",
        "n_cells": int(mask.sum()),
    }
    x_ref, _ = i539._twoway_fe_residualize(x, run_l, per_l)
    z_ref, _ = i539._twoway_fe_residualize(z, run_l, per_l)
    x_chk, z_chk = _fe_resid_cols([x, z], rc, pc, len(run_u), len(per_u))
    drift = max(float(np.max(np.abs(x_chk - x_ref))), float(np.max(np.abs(z_chk - z_ref))))
    assert drift < 1e-8, f"k-column FE residual drift {drift!r} vs parent implementation"
    length_partial["rho_twoway_min_dist_vs_mean_len"] = i539._spearman_rho(x_ref, z_ref)
    for tgt in ("dz", "dmargin"):
        y = panel[tgt][mask]
        x_f, y_f, z_f = _fe_resid_cols([x, y, z], rc, pc, len(run_u), len(per_u))
        y_ref, _ = i539._twoway_fe_residualize(y, run_l, per_l)
        d2 = max(float(np.max(np.abs(x_f - x_ref))), float(np.max(np.abs(y_f - y_ref))))
        assert d2 < 1e-8, f"FE residual drift {d2!r} on {tgt}"
        est = i539inf._fast_partial_spearman(x_f, y_f, z_f)
        cis = {
            "cluster_run": _cluster_boot_partial_fe(
                x, y, z, run_l, per_l, "source", args.n_cluster_boot, args.seed
            ),
            "cluster_persona": _cluster_boot_partial_fe(
                x, y, z, run_l, per_l, "bystander", args.n_cluster_boot, args.seed
            ),
        }
        block = {
            "estimate": float(est),
            "n_cells": int(mask.sum()),
            "rho_twoway_unpartialled": i539._spearman_rho(x_f, y_f),
            "rho_twoway_mean_len_vs_dv": i539._spearman_rho(z_f, y_f),
            "ci95_cell_boot": _cell_boot_partial_fe(x, y, z, rc, pc, args.n_boot, args.seed),
            "ci95_cluster_run": cis["cluster_run"],
            "ci95_cluster_persona": cis["cluster_persona"],
            "primary_ci": p553.wider_ci(cis),
            "p_perm_fe_partial": {
                **i539inf._partial_residual_permutation_p(x_f, y_f, z_f, args.n_perm, args.seed),
                "note": "diagnostic only — sensitivity reads are CI-based",
            },
        }
        length_partial[tgt] = block
        print(
            f"[sens-len] min_dist vs {tgt} | mean_len: rho_partial={est:+.3f} "
            f"(unpartialled {block['rho_twoway_unpartialled']:+.3f}) "
            f"primary CI [{block['primary_ci']['low']:+.3f}, "
            f"{block['primary_ci']['high']:+.3f}] ({block['primary_ci']['axis']}) "
            f"(n={int(mask.sum())})"
        )

    # (2) Truncation-excluded re-aggregation (registered-form check).
    min_keep = max(1, (int(n_q) + 1) // 2) if isinstance(n_q, int) else 1
    valid = mask & (panel["n_nontrunc_q"] >= min_keep)
    trunc_reads = min_dist_reads(
        panel,
        valid,
        args,
        full_inference=True,
        targets=("dz_nontrunc", "dmargin_nontrunc"),
        tag="sens-trunc",
    )
    trunc_excluded = {
        "min_nontrunc_q_per_cell": min_keep,
        "n_cells_kept": int(valid.sum()),
        "n_cells_dropped": int(mask.sum() - valid.sum()),
        "note": "registered-form check: same pair-corrected machinery, cells re-aggregated "
        "over only non-truncated generations; expected near-identical to the EOR-only "
        "sensitivity (truncated rows are ~98% pre_marker marker-repetition loops)",
        **trunc_reads,
    }

    return {
        "status": "exploratory sensitivity reads (round-1 interpretation-critic follow-up); "
        "NOT registered family members — the Holm family is unchanged",
        "length_partial": length_partial,
        "trunc_excluded": trunc_excluded,
    }


# ── (R2) persistence + clamp routing ─────────────────────────────────────────


def _persona_fe(panel: dict, mask: np.ndarray, channel: str) -> tuple[np.ndarray, np.ndarray]:
    """(persona labels, centered persona-FE vector of ``channel``) on a mask."""
    y = panel[channel][mask]
    per_u, pc = np.unique(panel["persona"][mask], return_inverse=True)
    _, rc = np.unique(panel["source_cid"][mask], return_inverse=True)
    fe = p553.fe_vector(y, pc, rc, len(per_u), int(rc.max()) + 1)
    return per_u, fe


def _persona_mean(panel: dict, mask: np.ndarray, channel: str, per_u: np.ndarray) -> np.ndarray:
    vals = panel[channel][mask]
    labels = panel["persona"][mask]
    return np.array([float(vals[labels == p].mean()) for p in per_u])


def persistence_block(panel: dict, mask: np.ndarray, args) -> dict:
    """H2: persona-FE of trained z_eos vs persona-mean base z_eos, plus the
    variance-ratio + calibrated-null reads (critic concern 4: the high rho is
    near-guaranteed when Var(FE dz_eos) << Var(FE base z_eos) — lead with the
    ratio, calibrate the rho against permuted-delta refits)."""
    per_u, fe_trained = _persona_fe(panel, mask, "z_eos_trained")
    per_u2, fe_base = _persona_fe(panel, mask, "z_eos_base")
    per_u3, fe_delta = _persona_fe(panel, mask, "dz_eos")
    assert list(per_u) == list(per_u2) == list(per_u3)
    # Linearity check: FE(trained) = FE(base) + FE(delta) exactly.
    assert float(np.max(np.abs(fe_trained - (fe_base + fe_delta)))) < 1e-6
    x_base_mean = _persona_mean(panel, mask, "z_eos_base", per_u)
    rho = i539._spearman_rho(x_base_mean, fe_trained)
    var_ratio = float(np.var(fe_delta) / np.var(fe_base)) if np.var(fe_base) > 0 else float("nan")

    # Calibrated null: permute the persona-FE delta vector, refit rho.
    rng = np.random.default_rng(args.seed)
    null_rhos = []
    for _ in range(args.n_perm):
        y_perm = fe_base + rng.permutation(fe_delta)
        null_rhos.append(i539._fast_spearman(x_base_mean, y_perm))
    null_arr = np.asarray(null_rhos)
    return {
        "persona_fe_trained_z_eos": {
            str(k): float(v) for k, v in zip(per_u, fe_trained, strict=True)
        },
        "rho": float(rho),
        "n_personas": len(per_u),
        "ci95_boot_personas": i539._bootstrap_spearman_ci(
            x_base_mean, fe_trained, args.n_marginal_boot, args.seed
        ),
        "p_perm": {
            **i539._permutation_p(x_base_mean, fe_trained, args.n_perm, args.seed),
            "method": f"MC permutation of the {len(per_u)} persona labels (diagnostic only — "
            "H2 verdicts are CI-based; see calibrated_null)",
        },
        "variance_ratio_feDelta_over_feBase": var_ratio,
        "calibrated_null": {
            "mean": float(null_arr.mean()),
            "q025": float(np.percentile(null_arr, 2.5)),
            "q975": float(np.percentile(null_arr, 97.5)),
            "n_perm": args.n_perm,
            "method": "persona-FE(dz_eos) permuted across personas; "
            "rho(persona-mean base z_eos, FE(base z_eos) + permuted FE(dz_eos)) per rep — "
            "the expected persistence rho under variance-ratio-only structure",
        },
        "read": "if observed rho ~ calibrated null, scope the persistence claim as "
        "variance-ratio rank persistence, not a stronger mechanism (plan section 14.4)",
    }


def clamp_routing_block(panel: dict, mask: np.ndarray, args) -> dict:
    """Clamp routing: persona-FE(dz_eos) vs persona-mean margin_base, with the
    parent followup's partial check (controlling persona-mean base z_eos)."""
    per_u, fe = _persona_fe(panel, mask, "dz_eos")
    x_margin = _persona_mean(panel, mask, "margin_base", per_u)
    x_zeos = _persona_mean(panel, mask, "z_eos_base", per_u)
    raw_rho = i539._spearman_rho(x_margin, fe)

    def partial_boot(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
        rng = np.random.default_rng(args.seed)
        n = len(x)
        rhos, n_deg = [], 0
        for _ in range(args.n_marginal_boot):
            idx = rng.integers(0, n, size=n)
            r = i539inf._fast_partial_spearman(x[idx], y[idx], z[idx])
            if np.isnan(r):
                n_deg += 1
                continue
            rhos.append(r)
        return i539inf._percentile_summary(rhos, args.n_marginal_boot, n_deg)

    return {
        "persona_fe_dz_eos": {str(k): float(v) for k, v in zip(per_u, fe, strict=True)},
        "vs_persona_mean_margin_base": {
            "rho": float(raw_rho),
            "n_personas": len(per_u),
            "ci95_boot_personas": i539._bootstrap_spearman_ci(
                x_margin, fe, args.n_marginal_boot, args.seed
            ),
            "p_perm": {
                **i539._permutation_p(x_margin, fe, args.n_perm, args.seed),
                "method": f"MC permutation of the {len(per_u)} persona labels",
            },
        },
        "partial_vs_margin_base_given_z_eos_base": {
            "rho_partial": i539inf._fast_partial_spearman(x_margin, fe, x_zeos),
            "ci95_boot_personas": partial_boot(x_margin, fe, x_zeos),
            "p_perm": i539inf._partial_residual_permutation_p(
                x_margin, fe, x_zeos, args.n_perm, args.seed
            ),
        },
        "partial_vs_z_eos_base_given_margin_base": {
            "rho_partial": i539inf._fast_partial_spearman(x_zeos, fe, x_margin),
            "ci95_boot_personas": partial_boot(x_zeos, fe, x_margin),
            "p_perm": i539inf._partial_residual_permutation_p(
                x_zeos, fe, x_margin, args.n_perm, args.seed
            ),
        },
        "read": "lead clamp-routing with the partial check (the parent's routing signal "
        "dissolved into base-level persistence under it — followup_clamp_partial)",
    }


# ── (R3) exposure contrast ───────────────────────────────────────────────────


def _stratum_means_from_idx(panel: dict, idx: np.ndarray) -> dict[str, float]:
    exp = panel["exposure"][idx]
    y = panel["dz_eos"][idx]
    return {
        s: (float(y[exp == s].mean()) if (exp == s).any() else float("nan"))
        for s in ("trained_negative", "never_negative")
    }


def exposure_block(panel: dict, args) -> dict:
    """H3/F2: trained-negative vs never-negative dz_eos contrast with
    run-cluster (verdict-bearing) + persona-cluster (descriptive) CIs."""
    exp = panel["exposure"]
    y = panel["dz_eos"]
    means = {
        s: float(y[exp == s].mean())
        for s in ("source_resident", "trained_negative", "never_negative")
    }
    obs_diff = means["trained_negative"] - means["never_negative"]

    rng = np.random.default_rng(args.seed)
    # Run-cluster bootstrap: resample the 16 runs; both strata recomputed.
    runs = np.unique(panel["source_cid"])
    idx_of_run = {r: np.where(panel["source_cid"] == r)[0] for r in runs}
    run_diffs, run_nn_means, run_tn_means = [], [], []
    for _ in range(args.n_cluster_boot):
        chosen = rng.choice(runs, size=len(runs), replace=True)
        idx = np.concatenate([idx_of_run[r] for r in chosen])
        m = _stratum_means_from_idx(panel, idx)
        run_diffs.append(m["trained_negative"] - m["never_negative"])
        run_tn_means.append(m["trained_negative"])
        run_nn_means.append(m["never_negative"])
    # Persona-cluster bootstrap: the 3 TN personas and 32 NN personas
    # resampled independently (exposure is a persona property here).
    tn_personas = np.unique(panel["persona"][exp == "trained_negative"])
    nn_personas = np.unique(panel["persona"][exp == "never_negative"])
    idx_of_persona = {p: np.where(panel["persona"] == p)[0] for p in np.unique(panel["persona"])}
    per_diffs, per_nn_means, per_tn_means = [], [], []
    for _ in range(args.n_cluster_boot):
        tn_idx = np.concatenate(
            [idx_of_persona[p] for p in rng.choice(tn_personas, size=len(tn_personas))]
        )
        nn_idx = np.concatenate(
            [idx_of_persona[p] for p in rng.choice(nn_personas, size=len(nn_personas))]
        )
        tn_m = float(y[tn_idx][exp[tn_idx] == "trained_negative"].mean())
        nn_m = float(y[nn_idx][exp[nn_idx] == "never_negative"].mean())
        per_diffs.append(tn_m - nn_m)
        per_tn_means.append(tn_m)
        per_nn_means.append(nn_m)

    def ci(vals: list[float], n_clusters: int) -> dict:
        return {
            "low": float(np.percentile(vals, 2.5)),
            "high": float(np.percentile(vals, 97.5)),
            "n_boot": args.n_cluster_boot,
            "n_clusters": n_clusters,
        }

    anchors = {}
    if I553_EXPOSURE_JSON.exists():
        parent = json.loads(I553_EXPOSURE_JSON.read_text())
        anchors = {
            "i532_trained_negative_mean_dz_eos": parent["ordinary_vs_instructed_gap"][
                "mean_dz_eos_ordinary_cross"
            ],
            "i532_never_clamped_mean_dz_eos": parent["ordinary_vs_instructed_gap"][
                "mean_dz_eos_instructed_strip"
            ],
            "i478_never_negative_mean_dz_eos": parent["i478_never_negative_contrast"][
                "mean_dz_eos"
            ],
            "note": "cross-panel anchors are SOFT (different question sets / token caps) — "
            "qualitative reference lines only; the within-run contrast is the load-bearing "
            "exposure read (plan section 14.10)",
        }

    pooled_label_view = {}
    for name, m in (
        ("pooled_non_source_resident_557", exp != "source_resident"),
        ("pooled_all_560", np.ones(panel["_n"], dtype=bool)),
    ):
        pooled_label_view[name] = {"mean_dz_eos": float(y[m].mean()), "n_cells": int(m.sum())}

    return {
        "stratum_means_dz_eos": means,
        "stratum_counts": panel["_strata_counts"],
        "contrast_trained_negative_minus_never_negative": {
            "observed": obs_diff,
            "ci95_cluster_run": ci(run_diffs, len(runs)),
            "ci95_cluster_persona_descriptive": {
                **ci(per_diffs, len(tn_personas) + len(nn_personas)),
                "note": "3 trained-negative persona clusters — descriptive only, NOT the "
                "verdict-bearing interval (plan section 14.1)",
            },
            "verdict_axis": "ci95_cluster_run",
        },
        "never_negative_mean": {
            "observed": means["never_negative"],
            "ci95_cluster_run": ci(run_nn_means, len(runs)),
            "ci95_cluster_persona": ci(per_nn_means, len(nn_personas)),
        },
        "trained_negative_mean": {
            "observed": means["trained_negative"],
            "ci95_cluster_run": ci(run_tn_means, len(runs)),
            "ci95_cluster_persona_descriptive": ci(per_tn_means, len(tn_personas)),
        },
        "source_resident_positive_control": {
            "mean_dz_eos": means["source_resident"],
            "n_cells": 3,
            "per_cell": {
                f"{c}__{p}": {
                    "dz": float(panel["dz"][i]),
                    "dz_eos": float(panel["dz_eos"][i]),
                    "margin_trained": float(panel["margin_trained"][i]),
                    "argmax_marker_rate_trained": float(panel["argmax_marker_rate_trained"][i]),
                    "pre_marker_frac": float(panel["pre_marker_frac"][i]),
                }
                for i in np.where(exp == "source_resident")[0]
                for c, p in [(panel["source_cid"][i], panel["persona"][i])]
            },
        },
        "parent_anchors": anchors,
        "parent_convention_label_view": pooled_label_view,
        "f2_read": {
            "never_negative_mean": means["never_negative"],
            "i532_never_clamped_anchor": anchors.get("i532_never_clamped_mean_dz_eos"),
            "rule": "F2 fires only if never-negative mean >= the +6.4 anchor with the "
            "persona-cluster CI excluding values below it; (0, +6.4) is "
            "weakening-not-breaking (plan section 14.1)",
        },
        "scope": "exposure and persona identity are perfectly confounded for the 3 "
        "trained-negative personas — descriptive contrast only; geometry predicts the "
        "OPPOSITE sign for these near-zero-distance personas (plan section 14.1)",
    }


# ── (R4) variance shares (exploratory) ───────────────────────────────────────


def shares_block(panel: dict, mask: np.ndarray, args) -> dict:
    """Two-way (run + persona) Type-I shares of dz with order swap + cell boot."""
    y = panel["dz"][mask]
    run_l, per_l = panel["source_cid"][mask], panel["persona"][mask]
    _, rc = np.unique(run_l, return_inverse=True)
    _, pc = np.unique(per_l, return_inverse=True)
    obs = p553.anova_shares(y, rc, pc, int(rc.max()) + 1, int(pc.max()) + 1)
    return {
        "observed": {
            "run_share_run_first": obs["a_first_share_a"],
            "persona_share_run_first": obs["a_first_share_b"],
            "persona_share_persona_first": obs["b_first_share_b"],
            "run_share_persona_first": obs["b_first_share_a"],
            "pair_share": obs["pair_share"],
        },
        "ci95_cell_boot": p553.shares_cell_bootstrap(
            y, run_l, per_l, args.n_boot, args.seed, dominance=True
        ),
        "note": "exploratory (plan R4); single-seed runs — run FE confounds adapter + seed",
    }


# ── Diagnostics ──────────────────────────────────────────────────────────────


def diagnostics_block(panel: dict, masks: dict[str, np.ndarray]) -> dict:
    """Saturation localization, argmax composition, truncation, ΔP sanity."""
    m = masks["primary_non_source_resident"]
    per_u = np.unique(panel["persona"])
    trunc_by_persona = {
        str(p): float(np.mean(panel["gen_trunc_rate"][panel["persona"] == p])) for p in per_u
    }
    md_persona = np.array(
        [float(np.mean(panel["min_dist"][m][panel["persona"][m] == p])) for p in per_u]
    )
    trunc_persona = np.array([trunc_by_persona[str(p)] for p in per_u])
    strata = ("source_resident", "trained_negative", "never_negative")
    return {
        "space_agreement": {
            "rho_dlogp_vs_dz_primary": i539._spearman_rho(panel["dlogp"][m], panel["dz"][m]),
            "mean_abs_dlogZ_by_stratum": {
                s: float(np.mean(np.abs(panel["dlogZ"][panel["exposure"] == s]))) for s in strata
            },
            "read": "off saturation ΔlogZ ~ 0 so Δlog P ~ Δz; divergence localizes "
            "saturation (marker-measurement rule)",
        },
        "delta_p_sanity_by_stratum": {
            s: float(np.mean(panel["delta_p"][panel["exposure"] == s])) for s in strata
        },
        "argmax_composition": {
            side: {
                "marker_rate": float(np.mean(panel[f"argmax_marker_rate_{side}"])),
                "eos_rate": float(np.mean(panel[f"argmax_eos_rate_{side}"])),
                "other_rate": float(
                    np.mean(
                        1.0 - panel[f"argmax_marker_rate_{side}"] - panel[f"argmax_eos_rate_{side}"]
                    )
                ),
            }
            for side in ("trained", "base")
        },
        "emission_sanity": {
            "in_R_emission_rate_by_stratum": {
                s: float(np.mean(panel["pre_marker_frac"][panel["exposure"] == s])) for s in strata
            },
            "note": "pre_marker slot fraction == in-R emission rate (the slot is truncated "
            "at the first emitted marker)",
        },
        "truncation": {
            "overall_gen_trunc_rate": float(np.mean(panel["gen_trunc_rate"])),
            "per_persona": trunc_by_persona,
            "rho_min_dist_vs_trunc_rate_personas": i539._spearman_rho(md_persona, trunc_persona),
            "rho_response_len_vs_dz_primary": i539._spearman_rho(
                panel["mean_n_new_tokens"][m], panel["dz"][m]
            ),
        },
        "tie_back": panel["_geometry"]["tie_back"],
    }


# ── Figures ──────────────────────────────────────────────────────────────────


def fig_hero_min_dist(panel: dict, masks: dict, r1: dict, fig_dir: Path) -> None:
    """F1: raw + FE-residualized min_dist vs dz / dmargin, CIs + parent points."""
    set_paper_style("blog")
    colors = paper_palette(2)
    m = masks["primary_non_source_resident"]
    x = panel["min_dist"][m]
    run_l, per_l = panel["source_cid"][m], panel["persona"][m]
    x_tw, _ = i539._twoway_fe_residualize(x, run_l, per_l)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    for col, tgt in enumerate(("dz", "dmargin")):
        y = panel[tgt][m]
        y_tw, _ = i539._twoway_fe_residualize(y, run_l, per_l)
        blk = r1[tgt]
        ax_raw, ax_fe = axes[0, col], axes[1, col]
        ax_raw.plot(x, y, "o", ms=2.6, alpha=0.4, color=colors[0])
        ax_raw.set_title(
            f"{p553.CHANNEL_DISPLAY.get(tgt, tgt)}\nraw rho={i539._spearman_rho(x, y):+.3f}",
            fontsize=8,
        )
        ax_raw.set_xlabel(MIN_DIST_XLABEL, fontsize=7)
        ax_raw.set_ylabel("raw change, trained − base (logits)")
        ax_fe.plot(x_tw, y_tw, "o", ms=2.6, alpha=0.4, color=colors[1])
        ci = blk["primary_ci"]
        ax_fe.set_title(
            f"adapter+persona-corrected rho={blk['estimate']:+.3f}  "
            f"primary CI [{ci['low']:+.3f}, {ci['high']:+.3f}] ({ci['axis']})\n"
            f"adapter-cluster CI [{blk['ci95_cluster_run']['low']:+.3f}, "
            f"{blk['ci95_cluster_run']['high']:+.3f}] / persona-cluster CI "
            f"[{blk['ci95_cluster_persona']['low']:+.3f}, "
            f"{blk['ci95_cluster_persona']['high']:+.3f}]  "
            f"(parent {PARENT_POINTS[tgt]:+.3f})",
            fontsize=7,
        )
        ax_fe.set_xlabel("distance residual after adapter + persona effects removed", fontsize=7)
        ax_fe.set_ylabel("change residual (logits)")
    fig.suptitle(
        "#560: persona geometry vs training-induced change, 557 non-source-resident cells "
        "(raw top, two-way-FE-corrected bottom)",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, "i560_min_dist_hero", dir=fig_dir)
    plt.close(fig)
    print(f"[figures] wrote i560_min_dist_hero to {fig_dir}")


def fig_persistence(panel: dict, masks: dict, pers: dict, fig_dir: Path) -> None:
    """F2: persona-FE trained z_eos vs persona-mean base z_eos (centered)."""
    set_paper_style("blog")
    colors = paper_palette(2)
    m = masks["primary_non_source_resident"]
    per_u, fe_trained = _persona_fe(panel, m, "z_eos_trained")
    x = _persona_mean(panel, m, "z_eos_base", per_u)
    xc = x - x.mean()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(xc, fe_trained, "o", ms=5, color=colors[0])
    lims = [min(xc.min(), fe_trained.min()), max(xc.max(), fe_trained.max())]
    ax.plot(lims, lims, "--", lw=0.9, color="0.5", label="identity")
    ci = pers["ci95_boot_personas"]
    ax.set_title(
        f"Trained end-of-answer EOS level tracks the base level\n"
        f"rho={pers['rho']:+.3f} CI [{ci['low']:+.3f}, {ci['high']:+.3f}] "
        f"(n=35; calibrated null mean {pers['calibrated_null']['mean']:+.3f}, "
        f"variance ratio {pers['variance_ratio_feDelta_over_feBase']:.3f})",
        fontsize=8,
    )
    ax.set_xlabel("persona-mean base z(EOS), centered (logits)")
    ax.set_ylabel("persona-FE of trained z(EOS) (logits)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, "i560_persistence_z_eos", dir=fig_dir)
    plt.close(fig)
    print(f"[figures] wrote i560_persistence_z_eos to {fig_dir}")


def fig_exposure(expo: dict, fig_dir: Path) -> None:
    """F3: exposure bar chart with parent anchors + #560 strata (cluster CIs)."""
    set_paper_style("blog")
    colors = paper_palette(3)
    anchors = expo["parent_anchors"]
    bars = [
        (
            "#532 trained-neg\n(context cells)",
            anchors.get("i532_trained_negative_mean_dz_eos"),
            None,
        ),
        ("#532 never-clamped\n(instructed)", anchors.get("i532_never_clamped_mean_dz_eos"), None),
        ("#478 never-neg\n(4-neg recipe)", anchors.get("i478_never_negative_mean_dz_eos"), None),
        (
            "#560 never-neg\n(512 cells)",
            expo["never_negative_mean"]["observed"],
            expo["never_negative_mean"]["ci95_cluster_run"],
        ),
        (
            "#560 trained-neg\n(45 cells)",
            expo["trained_negative_mean"]["observed"],
            expo["trained_negative_mean"]["ci95_cluster_run"],
        ),
        (
            "#560 source-resident\n(3 cells)",
            expo["source_resident_positive_control"]["mean_dz_eos"],
            None,
        ),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    xs = np.arange(len(bars))
    for xi, (label, val, ci) in enumerate(bars):
        if val is None:
            continue
        color = colors[0] if label.startswith("#560") else colors[2]
        ax.bar(xi, val, color=color, width=0.62)
        if ci is not None:
            ax.errorbar(
                xi,
                val,
                yerr=[[max(0.0, val - ci["low"])], [max(0.0, ci["high"] - val)]],
                fmt="none",
                ecolor="black",
                lw=1.2,
                capsize=3,
            )
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=7)
    ax.set_ylabel("Mean Δz(EOS) trained − base (logits)")
    ax.set_title(
        "EOS-side change by exposure class (#560 bars carry run-cluster CIs; "
        "cross-panel anchors are qualitative)",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, "i560_exposure_bars", dir=fig_dir)
    plt.close(fig)
    print(f"[figures] wrote i560_exposure_bars to {fig_dir}")


def fig_length_trunc_sensitivity(r1: dict, slot_sens: dict, len_trunc: dict, fig_dir: Path) -> None:
    """F4: headline-robustness forest — the geometry correlation under each
    sensitivity (registered read, marker-emission slots excluded, response
    length partialled out, truncated generations excluded), with adapter-cluster
    CIs where computed. Raw (unpartialled) alongside the partialled/excluded
    views per the raw-alongside-processed rule.
    """
    set_paper_style("blog")
    colors = paper_palette(2)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for ax, tgt, title in zip(
        axes,
        ("dz", "dmargin"),
        ("Marker-logit push Δz(※)", "EOS-margin shift Δmargin"),
        strict=True,
    ):
        rows = [
            (
                "registered read\n(557 cells)",
                r1[tgt]["estimate"],
                r1[tgt]["ci95_cluster_run"],
            ),
            (
                "emission slots\nexcluded (543)",
                slot_sens[f"{tgt}_eor_only"]["rho_twoway"],
                None,
            ),
            (
                "response length\npartialled (557)",
                len_trunc["length_partial"][tgt]["estimate"],
                len_trunc["length_partial"][tgt]["ci95_cluster_run"],
            ),
            (
                "truncated answers\nexcluded (542)",
                len_trunc["trunc_excluded"][f"{tgt}_nontrunc"]["estimate"],
                len_trunc["trunc_excluded"][f"{tgt}_nontrunc"]["ci95_cluster_run"],
            ),
        ]
        ys = np.arange(len(rows))[::-1]
        for y, (_, est, ci) in zip(ys, rows, strict=True):
            ax.plot(est, y, "o", ms=6, color=colors[0])
            if ci is not None:
                ax.plot([ci["low"], ci["high"]], [y, y], "-", lw=1.6, color=colors[0])
        ax.axvline(0.0, color="0.4", lw=0.8)
        ax.axvline(PARENT_POINTS[tgt], color=colors[1], lw=1.0, ls="--")
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=8)
        ax.set_xlabel(
            "Spearman rho: distance to source context vs change (more negative = "
            "closer personas change more)",
            fontsize=7,
        )
        ax.set_title(title, fontsize=9)
    fig.suptitle(
        "#560 headline robustness: the geometry correlation under each sensitivity "
        "(bars = adapter-cluster 95% CIs; dashed line = parent panel estimate; the "
        "emission-slots-excluded read has a permutation p only, no CI computed)",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, "i560_length_trunc_sensitivity", dir=fig_dir)
    plt.close(fig)
    print(f"[figures] wrote i560_length_trunc_sensitivity to {fig_dir}")


def make_exploratory_figures(panel: dict, masks: dict, fig_dir: Path) -> None:
    """Over-produce dump (plan section 6; the analyzer picks heroes)."""
    set_paper_style("blog")
    colors = paper_palette(3)
    m = masks["primary_non_source_resident"]

    # 1. Raw-alongside-FE grid for all five targets.
    p553.exploratory_raw_vs_fe_grid(
        x=panel["min_dist"][m],
        targets={t: panel[t][m] for t in MIN_DIST_TARGETS},
        a_labels=panel["source_cid"][m],
        b_labels=panel["persona"][m],
        x_label=MIN_DIST_XLABEL,
        fig_name="i560_min_dist_raw_vs_fe_grid",
        fig_dir=fig_dir,
        suptitle="#560 min_dist reads: raw (top) alongside two-way-FE-corrected (bottom)",
    )

    # 2. Per-adapter facets: min_dist vs dz, 16 panels (titles carry the
    # plain-English source-context class per the no-bare-condition-codes rule).
    sources = panel["_sources"]
    n_cols = 4
    n_rows = int(np.ceil(len(sources) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.7 * n_rows), squeeze=False)
    for i, cid in enumerate(sources):
        ax = axes[i // n_cols][i % n_cols]
        sel = m & (panel["source_cid"] == cid)
        ax.plot(panel["min_dist"][sel], panel["dz"][sel], "o", ms=3, alpha=0.6, color=colors[0])
        rho = i539._spearman_rho(panel["min_dist"][sel], panel["dz"][sel])
        ax.set_title(f"{cid} ({ADAPTER_CLASS_GLOSS[cid[0]]}): rho={rho:+.2f}", fontsize=8)
        if i // n_cols == n_rows - 1:
            ax.set_xlabel("distance to source context", fontsize=7)
        if i % n_cols == 0:
            ax.set_ylabel("marker-logit push\nΔz(※) (logits)", fontsize=7)
    for j in range(len(sources), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.suptitle(
        "#560 per-adapter geometry routing: persona distance to the source context vs "
        "marker-logit push (non-source-resident cells)",
        fontsize=10,
    )
    fig.tight_layout()
    savefig_paper(fig, "i560_per_adapter_min_dist_dz", dir=fig_dir)
    plt.close(fig)

    # 3. Δz vs Δz(EOS) scatter by stratum.
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for s, c in zip(("never_negative", "trained_negative", "source_resident"), colors, strict=True):
        sel = panel["exposure"] == s
        ax.plot(
            panel["dz"][sel],
            panel["dz_eos"][sel],
            "o",
            ms=3.2,
            alpha=0.55,
            color=c,
            label=STRATUM_DISPLAY[s],
        )
    ax.set_xlabel("marker-logit push Δz(※), trained − base (logits)")
    ax.set_ylabel("end-of-answer EOS change Δz(EOS), trained − base (logits)")
    ax.legend(fontsize=8)
    ax.set_title("#560 marker push vs EOS clamp per cell", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "i560_dz_vs_dz_eos", dir=fig_dir)
    plt.close(fig)

    # 4. Δlog P vs Δz agreement (saturation localization).
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for s, c in zip(("never_negative", "trained_negative", "source_resident"), colors, strict=True):
        sel = panel["exposure"] == s
        ax.plot(
            panel["dz"][sel],
            panel["dlogp"][sel],
            "o",
            ms=3.2,
            alpha=0.55,
            color=c,
            label=STRATUM_DISPLAY[s],
        )
    lims = [
        min(panel["dz"].min(), panel["dlogp"].min()),
        max(panel["dz"].max(), panel["dlogp"].max()),
    ]
    ax.plot(lims, lims, "--", lw=0.9, color="0.5", label="Δlog P = Δz (no saturation)")
    ax.set_xlabel("marker-logit push Δz(※), trained − base (logits)")
    ax.set_ylabel("marker log-prob change Δlog P(※), trained − base (nats)")
    ax.legend(fontsize=8)
    ax.set_title(
        "#560 space agreement: divergence from the identity localizes saturation", fontsize=9
    )
    fig.tight_layout()
    savefig_paper(fig, "i560_dlogp_vs_dz_agreement", dir=fig_dir)
    plt.close(fig)

    # 5. Argmax composition bars per side.
    p553.exploratory_argmax_bars(
        {
            f"#560 {side} side": {
                "marker_rate": float(np.mean(panel[f"argmax_marker_rate_{side}"])),
                "eos_rate": float(np.mean(panel[f"argmax_eos_rate_{side}"])),
                "other_rate": float(
                    np.mean(
                        1.0 - panel[f"argmax_marker_rate_{side}"] - panel[f"argmax_eos_rate_{side}"]
                    )
                ),
            }
            for side in ("trained", "base")
        },
        fig_name="i560_argmax_composition",
        fig_dir=fig_dir,
        title="#560 corrected-slot argmax composition",
    )

    # 6. Slot-kind fraction vs min_dist.
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(
        panel["min_dist"][m], panel["pre_marker_frac"][m], "o", ms=3, alpha=0.5, color=colors[0]
    )
    rho = i539._spearman_rho(panel["min_dist"][m], panel["pre_marker_frac"][m])
    ax.set_xlabel(MIN_DIST_XLABEL)
    ax.set_ylabel("fraction of answers emitting the marker\n(per cell, 20 answers)")
    ax.set_title(
        f"#560 marker emission is geometry-routed: closer personas emit more (rho={rho:+.2f})",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, "i560_slot_kind_vs_min_dist", dir=fig_dir)
    plt.close(fig)

    # 7. Tie-back scatter (fresh persona-pair distances vs the 111-panel matrix).
    if TIE_BACK_MATRIX.exists():
        ref = json.loads(TIE_BACK_MATRIX.read_text())
        ref_idx = {n: i for i, n in enumerate(ref["persona_names"])}
        pp = panel["_geometry"]["persona_persona_distance"]
        personas = panel["_personas"]
        matched = [p for p in personas if p in ref_idx]
        fresh, refv = [], []
        for i, a in enumerate(matched):
            for b in matched[i + 1 :]:
                fresh.append(pp[a][b])
                refv.append(ref["matrix"][ref_idx[a]][ref_idx[b]])
        fig, ax = plt.subplots(figsize=(5.8, 5.0))
        ax.plot(refv, fresh, "o", ms=3, alpha=0.5, color=colors[1])
        ax.set_xlabel("111-panel committed distance (layer 20)")
        ax.set_ylabel("fresh distance (q_test_extended_50 probes)")
        ax.set_title(
            f"#560 geometry tie-back: rho={panel['_geometry']['tie_back']['spearman']:+.3f} "
            f"({len(fresh)} pairs)",
            fontsize=9,
        )
        fig.tight_layout()
        savefig_paper(fig, "i560_tie_back_scatter", dir=fig_dir)
        plt.close(fig)

    # 8. Per-persona dz strip plot (sorted by persona-mean dz, primary cells).
    per_u = np.unique(panel["persona"][m])
    means = {p: float(np.mean(panel["dz"][m][panel["persona"][m] == p])) for p in per_u}
    order = sorted(per_u, key=lambda p: means[p])
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    rng = np.random.default_rng(0)
    for xi, p in enumerate(order):
        vals = panel["dz"][m][panel["persona"][m] == p]
        jitter = (rng.random(len(vals)) - 0.5) * 0.45
        ax.plot(np.full(len(vals), xi) + jitter, vals, "o", ms=2.2, alpha=0.45, color=colors[0])
        ax.plot([xi - 0.3, xi + 0.3], [means[p]] * 2, color=colors[1], lw=1.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_ylabel("Δz(※) (logits)")
    ax.set_title("#560 per-persona marker push across the 16 adapters", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "i560_per_persona_dz", dir=fig_dir)
    plt.close(fig)
    print(f"[figures:exploratory] dump complete -> {fig_dir}")


# ── Output plumbing ──────────────────────────────────────────────────────────


def result_metadata(args) -> dict:
    import scipy

    return {
        "task": 560,
        "script": "issue560_transfer_analysis.py",
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "seed": args.seed,
        "n_boot": args.n_boot,
        "n_cluster_boot": args.n_cluster_boot,
        "n_marginal_boot": args.n_marginal_boot,
        "n_perm": args.n_perm,
        "argv": sys.argv[1:],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Task #560 Phase B: cross-recipe transfer analysis (VM, CPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--four-float-dir", type=Path, default=DEFAULT_FF_DIR)
    ap.add_argument("--geometry-json", type=Path, default=DEFAULT_GEOMETRY)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_560")
    ap.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "figures/issue_560")
    ap.add_argument("--n-boot", type=int, default=10_000, dest="n_boot")
    ap.add_argument("--n-cluster-boot", type=int, default=2_000, dest="n_cluster_boot")
    ap.add_argument("--n-marginal-boot", type=int, default=2_000, dest="n_marginal_boot")
    ap.add_argument("--n-perm", type=int, default=10_000, dest="n_perm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-figures", action="store_true", help="statistics only (figure-free smoke)"
    )
    ap.add_argument(
        "--figures-only",
        action="store_true",
        help="regenerate figures from the existing transfer_i474.json (no inference re-run; "
        "panel is rebuilt from the four-float JSONs for the scatter data)",
    )
    return ap.parse_args(argv)


DEFAULT_FF_DIR = PROJECT_ROOT / "eval_results/issue_560/four_float"
DEFAULT_GEOMETRY = PROJECT_ROOT / "eval_results/issue_560/geometry/context_persona_geometry.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = datetime.now(UTC)

    assert_held_out_matches_logit_rescore()
    panel = build_panel(args.four_float_dir, args.geometry_json)
    masks = panel_masks(panel)
    full_panel = sorted(panel["_sources"]) == sorted(SOURCES_ALL) and (
        sorted(panel["_personas"]) == HELD_OUT_35
    )
    if full_panel:
        assert panel["_n"] == 560, panel["_n"]
        assert int(masks["primary_non_source_resident"].sum()) == 557
        assert int(masks["never_negative_only"].sum()) == 512
        assert int(masks["a_class_source_only"].sum()) == 172
    print(
        f"[panel] {panel['_n']} cells ({len(panel['_sources'])} runs x "
        f"{len(panel['_personas'])} personas); strata {panel['_strata_counts']}"
    )

    if args.figures_only:
        # Figure regeneration path (round-2 interp revision): reuse the committed
        # inference outputs verbatim — the figures must annotate the SAME numbers
        # the JSON carries, never a recompute that could drift.
        results_path = args.out_dir / "transfer_i474.json"
        results = json.loads(results_path.read_text())
        print(f"[figures-only] loaded {results_path}")
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        fig_hero_min_dist(panel, masks, results["min_dist_corrected_reads_primary"], args.fig_dir)
        fig_persistence(panel, masks, results["persistence"], args.fig_dir)
        fig_exposure(results["exposure"], args.fig_dir)
        fig_length_trunc_sensitivity(
            results["min_dist_corrected_reads_primary"],
            results["slot_kind_sensitivity"],
            results["sensitivity_length_truncation"],
            args.fig_dir,
        )
        make_exploratory_figures(panel, masks, args.fig_dir)
        print(f"[done:figures-only] wall={(datetime.now(UTC) - t0).total_seconds():.1f}s")
        return 0

    print("[R1] pair-corrected min_dist reads (primary mask) ...")
    r1_primary = min_dist_reads(
        panel, masks["primary_non_source_resident"], args, full_inference=True
    )
    r1_strata = {
        "never_negative_only": min_dist_reads(
            panel, masks["never_negative_only"], args, full_inference=True
        ),
        "a_class_source_only": min_dist_reads(
            panel, masks["a_class_source_only"], args, full_inference=True
        ),
        "sensitivity_including_source_resident": min_dist_reads(
            panel, masks["sensitivity_all_cells"], args, full_inference=False
        ),
    }
    slot_sens = slot_kind_sensitivity(panel, masks["primary_non_source_resident"], args)

    print("[sensitivity] length-partial + truncation-excluded reads ...")
    len_trunc_sens = length_truncation_sensitivity(
        panel, masks["primary_non_source_resident"], args
    )

    print("[R2] persistence + clamp routing ...")
    persistence = persistence_block(panel, masks["primary_non_source_resident"], args)
    clamp = clamp_routing_block(panel, masks["primary_non_source_resident"], args)

    print("[R3] exposure contrast ...")
    exposure = exposure_block(panel, args)

    print("[R4] variance shares (exploratory) ...")
    shares = shares_block(panel, masks["primary_non_source_resident"], args)

    # Holm over the 6 registered p-carrying members (CI-only verdicts; the
    # Holm p's are diagnostics — plan section 14.2).
    member_ps = [r1_primary[t]["p_perm_fe"]["p"] for t in MIN_DIST_TARGETS]
    member_ps.append(clamp["vs_persona_mean_margin_base"]["p_perm"]["p"])
    holm = i539.holm_adjust(member_ps)
    holm_block = {
        "family": "registered #560 family: 5 min_dist targets + clamp routing",
        "note": "verdicts are CI-only; Holm-adjusted p's are diagnostics — when they "
        "disagree the widest-axis CI governs (plan section 14.2)",
        "members": [
            *(
                {
                    "name": f"min_dist vs {t} (primary mask)",
                    "p_raw": member_ps[i],
                    "p_holm": holm[i],
                }
                for i, t in enumerate(MIN_DIST_TARGETS)
            ),
            {
                "name": "clamp routing: persona-FE(dz_eos) vs persona-mean margin_base",
                "p_raw": member_ps[5],
                "p_holm": holm[5],
            },
        ],
    }

    diagnostics = diagnostics_block(panel, masks)

    results = {
        "metadata": result_metadata(args),
        "schema_version": SCHEMA_VERSION,
        "panel": {
            "n_cells": panel["_n"],
            "n_runs": len(panel["_sources"]),
            "n_personas": len(panel["_personas"]),
            "n_questions": panel["_n_questions"],
            "strata_counts": panel["_strata_counts"],
            "geometry_probe_set": panel["_geometry"]["probe_set"],
            "geometry_n_probes": panel["_geometry"]["n_probes"],
        },
        "min_dist_corrected_reads_primary": r1_primary,
        "min_dist_strata": r1_strata,
        "slot_kind_sensitivity": slot_sens,
        "sensitivity_length_truncation": len_trunc_sens,
        "persistence": persistence,
        "clamp_routing": clamp,
        "exposure": exposure,
        "variance_shares_dz": shares,
        "holm": holm_block,
        "diagnostics": diagnostics,
    }
    p553.write_json(args.out_dir / "transfer_i474.json", results)

    if not args.skip_figures:
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        fig_hero_min_dist(panel, masks, r1_primary, args.fig_dir)
        fig_persistence(panel, masks, persistence, args.fig_dir)
        fig_exposure(exposure, args.fig_dir)
        fig_length_trunc_sensitivity(r1_primary, slot_sens, len_trunc_sens, args.fig_dir)
        make_exploratory_figures(panel, masks, args.fig_dir)

    for tgt in ("dz", "dmargin"):
        blk = r1_primary[tgt]
        print(
            f"[headline] min_dist vs {tgt}: rho={blk['estimate']:+.3f} primary CI "
            f"[{blk['primary_ci']['low']:+.3f}, {blk['primary_ci']['high']:+.3f}] "
            f"(parent {PARENT_POINTS[tgt]:+.3f})"
        )
    contrast = exposure["contrast_trained_negative_minus_never_negative"]
    print(
        f"[headline] persistence rho={persistence['rho']:+.3f} "
        f"(variance ratio {persistence['variance_ratio_feDelta_over_feBase']:.3f}, "
        f"calibrated null {persistence['calibrated_null']['mean']:+.3f}); "
        f"exposure contrast {contrast['observed']:+.2f} "
        f"run-CI [{contrast['ci95_cluster_run']['low']:+.2f}, "
        f"{contrast['ci95_cluster_run']['high']:+.2f}]"
    )
    print(f"[done] wall={(datetime.now(UTC) - t0).total_seconds():.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
