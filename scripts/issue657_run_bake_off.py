#!/usr/bin/env python3
"""Issue #657 Phase 3 — the cross-behavior leakage bake-off (CPU, off-pod).

UNIFIED in-process-serial pipeline (Step 6d.0 PASS_UNIFIED): smoke IS the sweep
with one cell. ``run_behavior()`` is the per-behavior unit the full sweep calls
once per ``--behaviors``; ``--smoke`` restricts to ONE behavior x ONE LOPO fold x
ONE bootstrap replicate via the SAME ``run_behavior`` code path — no subprocess
fan-out, every phase reads its cell subset from the same ``--behaviors``/``--smoke``
flags.

Per behavior it computes (plan §5/§6):
  - DV-(a) / H1: raw Spearman rho(align, base_rate) — the #623 generalization.
  - DV-(b) / H2: held-out LOPO Spearman rho(align, leakage Delta).
  - H3 (the bar): held-out doubly-partialled rho(align, Delta | [base prior,
    prior_centroid_projection]) + grouped-CV DeltaR2_align-beyond-prior, with the
    singly-partialled rho reported alongside, all across the 3-point
    bounded-attenuation reliability band (M-Stats1).
  - controls: shuffled-direction null + the secondary continuous-DV validation
    (completion_logprob vs delta Spearman).

It writes ``eval_results/issue_657/<behavior>_alignment_predictor.json`` per behavior
+ a top-level ``summary.json`` carrying the H3 PASS / INDETERMINATE / NULL verdict
per the plan §3 v3 gate (partial-rho AND DeltaR2 CI lower bound > 0 on >=1 PRIMARY
behavior {sycophancy, refusal, marker}, EM excluded, under the doubly-partialled read
holding across the reliability band).

The sycophancy DV-(a) read is the REPLICATION ANCHOR — its raw Spearman rho on the
overlap panel must land near #623's 0.73 (the join-integrity halt; reported, not
silently swallowed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.issue657_alignment_predictor import (  # noqa: E402
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    DEFAULT_LAYERS,
    HEADLINE_LAYER,
    MARKER_MIN_OVERLAP,
    PRIMARY_H3_BEHAVIORS,
    SHUFFLE_B,
    _source_fe_residualize,
    attenuation_sensitivity,
    bootstrap_over_personas_multi,
    build_predictor_frame,
    compute_alignment,
    effective_primary_h3,
    load_behavior_direction,
    load_cell_substrate,
    load_persona_vectors,
    lopo_heldout,
    marker_panel_overlap,
    partial_spearman,
    prior_centroid_projection,
    shuffled_direction_null,
)
from explore_persona_space.experiments.persona_decomp_623 import (  # noqa: E402
    reproducibility_metadata,
)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #623's headline DV-(a) sycophancy anchor (the replication target, §3/§7).
SYCO_ANCHOR_RHO = 0.726
# MARKER_MIN_OVERLAP is imported from the analysis library (single source of truth
# shared with marker_panel_overlap()); the marker LEAKAGE bake-off is excluded below
# this overlap (plan §7/§8/§12).


def _resolve(p: str) -> Path:
    return PROJECT_ROOT / p if not Path(p).is_absolute() else Path(p)


def _digest_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else "missing"


def _read_marker_direction_kind(direction_dir: Path) -> str:
    """Read the marker direction kind (affordance | shift_fallback) (v6, M-Alts3).

    The Phase-1 dispatcher writes a ``marker_direction_kind.json`` sentinel in the
    marker direction dir recording which recipe produced the direction (the #623
    affordance read = "affordance", the #521 on_policy_marker trained-shift fallback
    = "shift_fallback"). Falls back to the direction ``metadata.json`` ``recipe`` /
    ``marker_direction_kind`` field, then to "affordance" (the PRIMARY default — a
    marker direction with no kind sentinel is the pure affordance read).
    """
    sentinel = direction_dir / "marker_direction_kind.json"
    if sentinel.exists():
        doc = json.loads(sentinel.read_text())
        kind = doc.get("marker_direction_kind")
        if kind in ("affordance", "shift_fallback"):
            return kind
    meta = direction_dir / "metadata.json"
    if meta.exists():
        doc = json.loads(meta.read_text())
        kind = doc.get("marker_direction_kind")
        if kind in ("affordance", "shift_fallback"):
            return kind
        recipe = str(doc.get("recipe", ""))
        if "on_policy_marker" in recipe or "shift" in recipe:
            return "shift_fallback"
    return "affordance"


def _read_k2_pass(out_dir: Path, behavior: str) -> bool | None:
    """Read ``k2_pass`` for a behavior from its steering-effect-by-layer JSON.

    Returns the bool when the K2 probe ran for this direction, else None
    (sycophancy is REUSED + K2-validated in #623 — no fresh probe; None is read as
    "K2 inherited PASS from #623" by the caller).
    """
    fp = out_dir / f"steering_effect_by_layer_{behavior}.json"
    if not fp.exists():
        return None
    doc = json.loads(fp.read_text())
    kp = doc.get("k2_pass")
    return bool(kp) if kp is not None else None


def _ci_lower_gt_zero(ci: dict) -> bool:
    lo = ci.get("ci_lo")
    return lo is not None and not np.isnan(lo) and lo > 0.0


def _band_clears(band: dict) -> dict:
    """Classify the reliability-band H3 outcome from the per-band-point bootstrap CIs.

    The registered v3 §7 gate is "partial-rho CI lower bound > 0 AND ΔR² CI lower
    bound > 0 ACROSS the reliability band" — so a point CLEARS only if BOTH its
    doubly-partialled-rho CI AND its ΔR²-align-beyond-prior CI have a lower bound
    strictly > 0. ``attenuation_sensitivity`` bootstraps both statistics at every
    band point and stores them under ``points[label]["doubly_partialled_rho_ci"]``
    / ``points[label]["delta_r2_align_beyond_prior_ci"]``.

    Returns {verdict, detail}. verdict is one of:
      - CONFIRMED: both CIs clear 0 across the WHOLE band (incl. the most-attenuated
        R_low end).
      - INDETERMINATE: clears (both CIs) only at the optimistic-reliability end
        (R_high) — the registered carve-out the band exists to surface.
      - NULL: never clears under partialling.

    When the per-band-point CIs were NOT bootstrapped (smoke; ``b<=1``) the point
    estimates' signs of BOTH statistics are used as the fallback.
    """
    points = band.get("points", {})
    labels = ["R_low", "R_expected", "R_high"]
    present = [lbl for lbl in labels if lbl in points]
    if not present:
        return {"verdict": "NULL", "detail": "no band points"}

    ci_computed = bool(band.get("ci_computed", False))

    def _clears(lbl: str) -> bool:
        pt = points[lbl]
        if ci_computed:
            rho_ci = pt.get("doubly_partialled_rho_ci", {})
            r2_ci = pt.get("delta_r2_align_beyond_prior_ci", {})
            return _ci_lower_gt_zero(rho_ci) and _ci_lower_gt_zero(r2_ci)
        # smoke fallback: point-estimate signs of BOTH statistics
        rho = pt.get("doubly_partialled_rho")
        r2 = pt.get("delta_r2_align_beyond_prior")
        rho_ok = rho is not None and not np.isnan(rho) and rho > 0.0
        r2_ok = r2 is not None and not np.isnan(r2) and r2 > 0.0
        return rho_ok and r2_ok

    all_clear = all(_clears(lbl) for lbl in present)
    opt_clear = _clears("R_high") if "R_high" in points else _clears(present[-1])
    if all_clear:
        return {"verdict": "CONFIRMED", "detail": "rho+ΔR² CI lo > 0 across whole band"}
    if opt_clear:
        return {"verdict": "INDETERMINATE", "detail": "clears only at optimistic-reliability end"}
    return {"verdict": "NULL", "detail": "does not clear under partialling"}


def run_behavior(
    behavior: str,
    eval_results_root: Path,
    persona_dirs: list[Path],
    direction_dir: Path,
    base_rates_da: dict[str, float] | None,
    layer: int,
    readout: str,
    centering: str,
    n_boot: int,
    shuffle_b: int,
    smoke: bool,
    out_dir: Path | None = None,
) -> dict:
    """Run the full per-behavior bake-off (the unified unit; smoke = 1 fold x 1 boot).

    ``base_rates_da`` is the DV-(a) base-rate map for H1 (e.g. #623 syc_i for
    sycophancy); when None, DV-(a) falls back to the #518 ``bystander_base_rate``
    (still a valid base-rate read on the same panel). ``out_dir`` (when given) is
    where the per-behavior steering-effect JSON lives — used to read ``k2_pass`` +
    (for marker) the ``marker_direction_kind`` sentinel (v6, M-Alts3).
    """
    table = load_cell_substrate(behavior, eval_results_root)
    persona_vectors = load_persona_vectors(persona_dirs, DEFAULT_LAYERS)
    behavior_dir = load_behavior_direction(behavior, {behavior: direction_dir}, layer, readout)

    personas = table.bystanders
    align = compute_alignment(
        persona_vectors, behavior_dir, personas, layer, DEFAULT_LAYERS, centering
    )

    pcp = prior_centroid_projection(
        persona_vectors, table.bystander_base_rate, personas, layer, DEFAULT_LAYERS, centering
    )
    frame = build_predictor_frame(table, align, pcp)

    # ── v6 primary-set membership inputs (M-Alts3 + K2 gate) ──
    # marker_direction_kind: affordance (PRIMARY) | shift_fallback (SECONDARY M-Alts3).
    marker_direction_kind = (
        _read_marker_direction_kind(direction_dir) if behavior == "marker" else None
    )
    # k2_pass per direction (None == inherited from #623 for sycophancy; the reused
    # #623 sycophancy direction is K2-validated, so None reads as PASS).
    k2_pass = _read_k2_pass(out_dir, behavior) if out_dir is not None else None

    # ── marker leakage-overlap gate (plan §7/§8/§12 graceful degradation) ──
    # Overlap < MARKER_MIN_OVERLAP -> marker DV-(a)-only (excluded from the leakage
    # bake-off + the primary H3 set). Computed from the REAL resolvable count.
    marker_leakage_excluded = False
    marker_overlap = None
    if behavior == "marker":
        marker_overlap = frame["n_resolvable"]
        if marker_overlap < MARKER_MIN_OVERLAP:
            marker_leakage_excluded = True

    # ── DV-(a) / H1: raw Spearman rho(align, base_rate) ──
    da_base = base_rates_da if base_rates_da is not None else table.bystander_base_rate
    da_personas = [p for p in align if p in da_base and not np.isnan(align[p])]
    dv_a = {"n": len(da_personas)}
    if len(da_personas) >= 3:
        ax = np.array([align[p] for p in da_personas], dtype=float)
        ay = np.array([da_base[p] for p in da_personas], dtype=float)
        if np.std(ax) > 0 and np.std(ay) > 0:
            r = spearmanr(ax, ay)
            dv_a["raw_rho"] = float(r.correlation)
            # bootstrap over personas for the DV-(a) CI
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            boot = []
            nB = len(da_personas)
            for _ in range(n_boot):
                idx = rng.integers(0, nB, size=nB)
                rr = spearmanr(ax[idx], ay[idx]).correlation
                if rr is not None and not np.isnan(rr):
                    boot.append(float(rr))
            if boot:
                arr = np.array(boot)
                dv_a["ci_lo"] = float(np.quantile(arr, 0.025))
                dv_a["ci_hi"] = float(np.quantile(arr, 0.975))
            dv_a["base_rate_source"] = (
                "issue623_syc_i" if base_rates_da is not None else "issue518_bystander_base_rate"
            )
        else:
            dv_a["raw_rho"] = float("nan")
            dv_a["note"] = "zero-variance align or base_rate"
    else:
        dv_a["raw_rho"] = float("nan")
        dv_a["note"] = "fewer than 3 personas with both align and base_rate"

    # replication anchor (sycophancy only)
    anchor = None
    if behavior == "sycophancy":
        anchor = {
            "anchor_rho": dv_a.get("raw_rho"),
            "target_rho": SYCO_ANCHOR_RHO,
            "ci_lo": dv_a.get("ci_lo"),
            "ci_hi": dv_a.get("ci_hi"),
            # the join is consistent if the #623 target sits within the bootstrap CI
            "join_consistent": (
                dv_a.get("ci_lo") is not None
                and dv_a.get("ci_hi") is not None
                and not np.isnan(dv_a.get("ci_lo", float("nan")))
                and dv_a["ci_lo"] <= SYCO_ANCHOR_RHO <= dv_a["ci_hi"]
            ),
        }

    # ── DV-(b) / H2 + H3 (only when leakage is in scope for this behavior) ──
    leakage: dict = {"in_scope": not marker_leakage_excluded}
    if not marker_leakage_excluded and frame["cells"]:
        # raw + singly- + doubly-partialled rho (full-data point estimates)
        cells = frame["cells"]
        x = np.array([c["align"] for c in cells], dtype=float)
        y_fe = _source_fe_residualize(cells, "delta")
        prior = np.array([c["bystander_base_rate"] for c in cells], dtype=float)
        pcp_arr = np.array([c["prior_centroid_projection"] for c in cells], dtype=float)
        raw_rho, _rp, _rn = partial_spearman(x, y_fe, None)
        single_rho, _sp, _sn = partial_spearman(x, y_fe, prior)
        double_rho, _dp, _dn = partial_spearman(x, y_fe, np.column_stack([prior, pcp_arr]))
        leakage["raw_rho"] = raw_rho
        leakage["singly_partialled_rho"] = single_rho
        leakage["doubly_partialled_rho"] = double_rho

        # held-out LOPO + grouped-CV ΔR²
        lopo = lopo_heldout(frame)
        leakage["lopo"] = lopo

        # bootstrap over personas: held-out Spearman (H2) + ΔR² (H3) + doubly-
        # partialled rho (H3) — ALL from ONE resampling pass. The held-out Spearman
        # and ΔR² both come out of a single lopo_heldout() call, so computing them
        # in separate bootstraps re-ran the expensive grouped-CV per statistic (one
        # behavior's 3x B=10000 exceeded 5 min this round). The multi-stat bootstrap
        # calls lopo_heldout ONCE per replicate.
        n_boot_eff = 1 if smoke else n_boot

        def _stats(fr: dict) -> dict:
            lo = lopo_heldout(fr)
            fc = fr["cells"]
            dpr = float("nan")
            if len(fc) >= 3:
                xx = np.array([c["align"] for c in fc], dtype=float)
                yy = _source_fe_residualize(fc, "delta")
                pp = np.array([c["bystander_base_rate"] for c in fc], dtype=float)
                qq = np.array([c["prior_centroid_projection"] for c in fc], dtype=float)
                dpr, _p, _n = partial_spearman(xx, yy, np.column_stack([pp, qq]))
            return {
                "held_out_spearman": lo.get("held_out_spearman", float("nan")),
                "delta_r2_align_beyond_prior": lo.get("delta_r2_align_beyond_prior", float("nan")),
                "doubly_partialled_rho": dpr,
            }

        boot = bootstrap_over_personas_multi(
            frame,
            _stats,
            keys=["held_out_spearman", "delta_r2_align_beyond_prior", "doubly_partialled_rho"],
            b=n_boot_eff,
            seed=BOOTSTRAP_SEED,
        )
        leakage["h2_held_out_spearman_ci"] = boot["held_out_spearman"]
        leakage["h3_delta_r2_ci"] = boot["delta_r2_align_beyond_prior"]
        leakage["h3_doubly_partialled_rho_ci"] = boot["doubly_partialled_rho"]

        # reliability-sensitivity band (M-Stats1): bootstraps the EIV-disattenuated
        # doubly-partialled rho AND grouped ΔR² at EACH reliability point (same
        # B=10000 bystander resample as the headline gate; b=1 in smoke).
        band = attenuation_sensitivity(
            frame,
            table.bystander_base_rate,
            behavior=behavior,
            b=n_boot_eff,
            seed=BOOTSTRAP_SEED,
        )
        leakage["reliability_band"] = band

        # band verdict from the per-band-point CIs (partial-rho AND ΔR² CI lo > 0
        # across the WHOLE band == CONFIRMED; optimistic-end-only == INDETERMINATE).
        leakage["band_verdict"] = _band_clears(band)

        # shuffled-direction null (b-vs-random); skip the heavy run in smoke
        shuffle_eff = 1 if smoke else shuffle_b
        leakage["shuffled_direction_null"] = shuffled_direction_null(
            persona_vectors,
            behavior_dir,
            table,
            table.bystander_base_rate,
            layer,
            DEFAULT_LAYERS,
            centering,
            b=shuffle_eff,
            seed=BOOTSTRAP_SEED,
        )

        # secondary continuous-DV validation: completion_logprob vs delta Spearman
        clp = [
            (c["completion_logprob"], c["delta"])
            for c in cells
            if c.get("completion_logprob") is not None
        ]
        if len(clp) >= 3:
            cx = np.array([a for a, _ in clp], dtype=float)
            cy = np.array([b for _, b in clp], dtype=float)
            if np.std(cx) > 0 and np.std(cy) > 0:
                cr = spearmanr(cx, cy)
                leakage["secondary_dv_validation"] = {
                    "completion_logprob_vs_delta_rho": float(cr.correlation),
                    "n": len(clp),
                }

    # Run-conditional primary-H3 membership (v6, M-Alts2 + M-Alts3): EM never
    # primary; marker demoted to SECONDARY on the #521 trained-shift fallback OR the
    # Phase-0 panel-overlap exclusion. sycophancy/refusal always primary (per K2).
    eff_primary = effective_primary_h3(
        marker_direction_kind=marker_direction_kind,
        marker_leakage_excluded=marker_leakage_excluded,
    )
    is_primary = behavior in eff_primary

    return {
        "behavior": behavior,
        "is_primary_h3": is_primary,
        "primary_vs_secondary": "primary" if is_primary else "secondary",
        "marker_direction_kind": marker_direction_kind,
        "marker_panel_overlap": marker_overlap,
        "k2_pass": k2_pass,
        "layer": layer,
        "readout": readout,
        "centering": centering,
        "n_cells_substrate": table.n_cells(),
        "n_cells_joined": len(frame["cells"]),
        "n_resolvable_bystanders": frame["n_resolvable"],
        "resolvable_bystanders": frame["resolvable_bystanders"],
        "dropped_bystanders": sorted({d["bystander"] for d in frame["dropped"]}),
        "marker_leakage_excluded": marker_leakage_excluded,
        "dv_a_base_rate": dv_a,
        "replication_anchor": anchor,
        "leakage_bake_off": leakage,
        "substrate_provenance": table.provenance,
        # Lean per-cell join arrays so the plot phase is self-contained (no
        # separate join-dir dependency) and the raw scatter is reproducible.
        "joined_cells": [
            {
                "source": c["source"],
                "bystander": c["bystander"],
                "delta": c["delta"],
                "align": c["align"],
                "bystander_base_rate": c["bystander_base_rate"],
                "prior_centroid_projection": c["prior_centroid_projection"],
            }
            for c in frame["cells"]
        ],
    }


def _h3_summary(per_behavior: dict[str, dict]) -> dict:
    """Top-level H3 verdict over the RUN-CONDITIONAL primary set (v6, M-Alts2 + M-Alts3).

    The primary set is run-conditional: EM is ALWAYS secondary (M-Alts2); marker is
    secondary when its direction is the #521 trained-shift fallback (M-Alts3) OR the
    Phase-0 panel-overlap gate excluded it. Each behavior's effective membership was
    decided in ``run_behavior`` and stored as ``is_primary_h3`` — this reads that flag
    (single source of truth) rather than re-deriving the set, so the summary never
    disagrees with the per-behavior decision.

    A behavior whose direction FAILED K2 (``k2_pass`` False) is excluded from the
    gate set and reported, never read as a NULL.
    """
    primary_clears: list[str] = []
    effective_primary: list[str] = []
    detail: dict[str, dict] = {}
    for beh, res in per_behavior.items():
        if not res.get("is_primary_h3", False):
            continue
        effective_primary.append(beh)
        # K2-failed direction does not measure the behavior — excluded, not nulled.
        if res.get("k2_pass") is False:
            detail[beh] = {"status": "k2_failed_excluded_from_gate"}
            continue
        lk = res.get("leakage_bake_off", {})
        if not lk.get("in_scope", False):
            detail[beh] = {"status": "leakage_out_of_scope"}
            continue
        rho_ci = lk.get("h3_doubly_partialled_rho_ci", {})
        r2_ci = lk.get("h3_delta_r2_ci", {})
        band_verdict = lk.get("band_verdict", {}).get("verdict", "NULL")
        rho_clears = _ci_lower_gt_zero(rho_ci)
        r2_clears = _ci_lower_gt_zero(r2_ci)
        # PASS requires BOTH the doubly-partialled rho AND the ΔR² CI lower bound > 0
        # AND the band CONFIRMED (not just INDETERMINATE).
        passes = rho_clears and r2_clears and band_verdict == "CONFIRMED"
        detail[beh] = {
            "doubly_partialled_rho_ci_lo": rho_ci.get("ci_lo"),
            "delta_r2_ci_lo": r2_ci.get("ci_lo"),
            "band_verdict": band_verdict,
            "passes_h3": passes,
        }
        if passes:
            primary_clears.append(beh)

    if primary_clears:
        verdict = "PASS"
    elif any(
        isinstance(d, dict) and d.get("band_verdict") == "INDETERMINATE" for d in detail.values()
    ):
        verdict = "INDETERMINATE"
    else:
        verdict = "NULL"
    return {
        "h3_verdict": verdict,
        "primary_behaviors_cleared": primary_clears,
        "effective_primary_h3_behaviors": sorted(effective_primary),
        "full_primary_h3_behaviors": sorted(PRIMARY_H3_BEHAVIORS),
        "per_behavior_detail": detail,
        "note": (
            "PASS = doubly-partialled rho CI lo > 0 AND ΔR² CI lo > 0 AND reliability "
            "band CONFIRMED on >=1 EFFECTIVE-primary behavior. The primary set is "
            "run-conditional: EM always secondary (M-Alts2); marker secondary on the "
            "#521 trained-shift fallback (M-Alts3) or the Phase-0 panel-overlap "
            "exclusion. A power-limited / optimistic-end-only clearance is "
            "INDETERMINATE, not a confirmed beat-the-prior."
        ),
    }


def _reliability_band_summary(lk: dict) -> dict:
    """Extract the 3-point reliability-band rho/ΔR² (R_low/R_expected/R_high) per §6.5."""
    band = lk.get("reliability_band", {})
    points = band.get("points", {})
    out: dict[str, dict] = {}
    for label in ("R_observed", "R_low", "R_expected", "R_high"):
        if label not in points:
            continue
        pt = points[label]
        out[label] = {
            "reliability": pt.get("reliability"),
            "doubly_partialled_rho": pt.get("doubly_partialled_rho"),
            "delta_r2_align_beyond_prior": pt.get("delta_r2_align_beyond_prior"),
            "doubly_partialled_rho_ci": pt.get("doubly_partialled_rho_ci"),
            "delta_r2_align_beyond_prior_ci": pt.get("delta_r2_align_beyond_prior_ci"),
        }
    return {
        "band_applicable": band.get("band_applicable"),
        "band_verdict": lk.get("band_verdict", {}).get("verdict"),
        "points": out,
    }


def _build_alignment_predictor(
    per_behavior: dict[str, dict],
    h3: dict,
    marker_overlap_decision: dict,
    behaviors: list[str],
    smoke: bool,
) -> dict:
    """Assemble the §6.5 PRIMARY DELIVERABLE (eval_results/issue_657/alignment_predictor.json).

    The single file the upload-verifier completeness gate + the analyzer key off.
    Carries, per behavior: raw / singly / doubly-partialled rho, prior_centroid_projection
    (the per-cell array's read is folded into the join; the predictor here surfaces the
    leakage-side reads), ΔR², the 3-point reliability-band rho/ΔR², primary-vs-secondary,
    marker_direction_kind, marker_panel_overlap, per-behavior k2_pass, and CIs.
    """
    per_beh_out: dict[str, dict] = {}
    n_with_ci = 0
    for beh, res in per_behavior.items():
        lk = res.get("leakage_bake_off", {})
        rho_ci = lk.get("h3_doubly_partialled_rho_ci", {})
        has_ci = bool(rho_ci) and rho_ci.get("ci_lo") is not None
        if has_ci and not (isinstance(rho_ci.get("ci_lo"), float) and np.isnan(rho_ci["ci_lo"])):
            n_with_ci += 1
        per_beh_out[beh] = {
            "primary_vs_secondary": res.get("primary_vs_secondary"),
            "is_primary_h3": res.get("is_primary_h3"),
            "k2_pass": res.get("k2_pass"),
            "marker_direction_kind": res.get("marker_direction_kind"),
            "marker_panel_overlap": res.get("marker_panel_overlap"),
            "n_resolvable_bystanders": res.get("n_resolvable_bystanders"),
            "marker_leakage_excluded": res.get("marker_leakage_excluded"),
            # DV-(a) / H1 generalization
            "dv_a_raw_rho": res.get("dv_a_base_rate", {}).get("raw_rho"),
            "dv_a_ci": {
                "ci_lo": res.get("dv_a_base_rate", {}).get("ci_lo"),
                "ci_hi": res.get("dv_a_base_rate", {}).get("ci_hi"),
            },
            # DV-(b) / H2 + H3 leakage reads
            "leakage_in_scope": lk.get("in_scope"),
            "raw_rho": lk.get("raw_rho"),
            "singly_partialled_rho": lk.get("singly_partialled_rho"),
            "doubly_partialled_rho": lk.get("doubly_partialled_rho"),
            "delta_r2_align_beyond_prior": (lk.get("lopo") or {}).get(
                "delta_r2_align_beyond_prior"
            ),
            "h2_held_out_spearman_ci": lk.get("h2_held_out_spearman_ci"),
            "h3_doubly_partialled_rho_ci": rho_ci,
            "h3_delta_r2_ci": lk.get("h3_delta_r2_ci"),
            "reliability_band": _reliability_band_summary(lk),
            "shuffled_direction_null": lk.get("shuffled_direction_null"),
            "secondary_dv_validation": lk.get("secondary_dv_validation"),
            "replication_anchor": res.get("replication_anchor"),
        }
    return {
        "schema_version": 1,
        "smoke": smoke,
        "behaviors": list(behaviors),
        "h3": h3,
        "marker_panel_overlap": marker_overlap_decision["marker_panel_overlap"],
        "marker_panel_overlap_gate": marker_overlap_decision,
        "n_behaviors_with_bootstrap_ci": n_with_ci,
        "per_behavior": per_beh_out,
        "metadata": reproducibility_metadata(
            {"deliverable": "alignment_predictor", "smoke": smoke}
        ),
        "note": (
            "§6.5 primary deliverable. Per-behavior raw/singly/doubly-partialled rho, "
            "prior_centroid_projection (per-cell, in the join), ΔR²_align-beyond-prior, "
            "the 3-point reliability band, primary-vs-secondary (EM secondary M-Alts2; "
            "marker secondary when marker_direction_kind==shift_fallback M-Alts3, or "
            "DV-(a)-only when marker_panel_overlap < min), marker_direction_kind, "
            "marker_panel_overlap, per-behavior k2_pass, and bootstrap CIs."
        ),
    }


def _load_da_base_rates(
    behavior: str,
    syc_i_path: Path,
    *,
    smoke: bool = False,
    allow_syc_i_fallback: bool = False,
) -> dict[str, float] | None:
    """DV-(a) base-rate map for H1. sycophancy -> #623 syc_i; else None (use #518).

    For sycophancy the #623 ``syc_i`` rates are the TRUE replication-anchor base
    rate (#623's rho=0.726 was measured against syc_i, not #518's bystander_base_rate).
    The sycophancy anchor IS the join-integrity kill criterion (plan §7), so per
    CLAUDE.md "Fail fast — never hide failures" a MISSING ``syc_i`` HALTS a non-smoke
    production sycophancy run (exit non-zero) rather than silently downgrading the
    anchor to #518's ``bystander_base_rate``. Smoke runs and an explicit
    ``--allow-syc-i-fallback`` opt-in keep the warn-and-fall-back behavior (the
    fallback is a valid but DIFFERENT anchor, not the #623 comparison). The sparse
    worktree excludes eval_results/, so the off-pod runner must point --syc-i at the
    repo-root copy.
    """
    if behavior != "sycophancy":
        return None
    if not syc_i_path.exists():
        if not smoke and not allow_syc_i_fallback:
            raise SystemExit(
                f"[phase=bake_off] FATAL: #623 syc_i not found at {syc_i_path}. The "
                "sycophancy DV-(a) read is the #623 replication anchor (the "
                "join-integrity kill criterion, plan §7); falling back to #518 "
                "bystander_base_rate silently downgrades it to a DIFFERENT anchor. "
                "Point --syc-i at the repo-root eval_results/issue_623/syc_i.json, or "
                "pass --allow-syc-i-fallback to opt into the #518 fallback explicitly."
            )
        print(
            f"[phase=bake_off] WARNING: #623 syc_i not found at {syc_i_path}; the "
            "sycophancy DV-(a) read falls back to #518 bystander_base_rate (a valid "
            "but DIFFERENT anchor — not the #623 syc_i comparison). "
            f"({'smoke' if smoke else '--allow-syc-i-fallback'} mode).",
            flush=True,
        )
        return None
    doc = json.loads(syc_i_path.read_text())
    return {p: float(v["syc_i"]) for p, v in doc["syc_i"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #657 Phase 3 — leakage bake-off (CPU).")
    parser.add_argument(
        "--behavior",
        action="append",
        default=None,
        help="Behavior to run (repeatable). Default: all four (sycophancy refusal marker em).",
    )
    parser.add_argument(
        "--eval-results-root",
        default="eval_results",
        help="Root holding issue_518/ + issue_605/ (relative to repo root).",
    )
    parser.add_argument("--persona-dir", action="append", default=None)
    parser.add_argument(
        "--direction-dir", default=None, help="Override direction dir (1 behavior)."
    )
    parser.add_argument(
        "--syc-i",
        default="eval_results/issue_623/syc_i.json",
        help="#623 syc_i.json for the sycophancy DV-(a) read (relative to repo root).",
    )
    parser.add_argument("--layer", type=int, default=HEADLINE_LAYER)
    parser.add_argument("--readout", default="last_token", choices=["last_token", "response_avg"])
    parser.add_argument("--centering", default="global_mean", choices=["global_mean", "none"])
    parser.add_argument("--n-boot", type=int, default=BOOTSTRAP_B)
    parser.add_argument("--shuffle-b", type=int, default=SHUFFLE_B)
    parser.add_argument(
        "--out-dir",
        default="eval_results/issue_657",
        help="Output dir for per-behavior JSONs + summary.json (relative to repo root).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke: 1 LOPO fold x 1 bootstrap replicate (unified one-cell path).",
    )
    parser.add_argument(
        "--allow-syc-i-fallback",
        action="store_true",
        help=(
            "Opt into the #518 bystander_base_rate fallback when the #623 syc_i.json "
            "is absent for sycophancy (default: HALT — the syc_i anchor is the "
            "join-integrity kill criterion). Smoke runs allow the fallback implicitly."
        ),
    )
    args = parser.parse_args()

    load_dotenv()

    behaviors = args.behavior or ["sycophancy", "refusal", "marker", "em"]
    eval_root = _resolve(args.eval_results_root)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    syc_i_path = _resolve(args.syc_i)

    method_subdir = "method_a" if args.readout == "last_token" else "method_b"
    default_persona_dirs = [
        _resolve(f"data/issue657/persona_vectors/{method_subdir}"),
        _resolve(f"data/persona_vectors/issue623/{method_subdir}"),
    ]
    persona_dirs = (
        [_resolve(p) for p in args.persona_dir] if args.persona_dir else default_persona_dirs
    )

    # §6.5 per-behavior subdir glob + the top-level alignment_predictor.json.
    per_behavior_dir = out_dir / "per_behavior"
    per_behavior_dir.mkdir(parents=True, exist_ok=True)

    # ── v6 FIRST Phase-0 deliverable: marker panel-overlap gate (§7) ──
    # Computed + logged BEFORE the per-behavior loop so the primary-set-membership
    # decision is the first thing in the run (a non-silent, logged decision).
    marker_overlap_decision = marker_panel_overlap(eval_root)
    print(
        f"[phase=bake_off] marker_panel_overlap={marker_overlap_decision['marker_panel_overlap']} "
        f"(min={marker_overlap_decision['min_overlap']}) "
        f"marker_leakage_excluded={marker_overlap_decision['marker_leakage_excluded']} "
        f"-> {marker_overlap_decision['decision']}",
        flush=True,
    )

    per_behavior: dict[str, dict] = {}
    for beh in behaviors:
        if args.direction_dir:
            direction_dir = _resolve(args.direction_dir)
        elif beh == "sycophancy":
            direction_dir = _resolve("data/persona_vectors/issue623/sycophancy_trait")
        else:
            direction_dir = _resolve(f"data/issue657/behavior_directions/{beh}")

        base_rates_da = _load_da_base_rates(
            beh,
            syc_i_path,
            smoke=args.smoke,
            allow_syc_i_fallback=args.allow_syc_i_fallback,
        )
        print(f"[phase=bake_off] behavior={beh} smoke={args.smoke}", flush=True)
        res = run_behavior(
            beh,
            eval_root,
            persona_dirs,
            direction_dir,
            base_rates_da,
            args.layer,
            args.readout,
            args.centering,
            args.n_boot,
            args.shuffle_b,
            args.smoke,
            out_dir=out_dir,
        )
        # checkpoint-per-phase: write each behavior the moment it completes (flat,
        # for the plot script) + a copy under per_behavior/ (the §6.5 glob).
        meta = reproducibility_metadata({"behavior": beh, "smoke": args.smoke})
        res["metadata"] = meta
        beh_path = out_dir / f"{beh}_alignment_predictor.json"
        beh_path.write_text(json.dumps(res, indent=2))
        (per_behavior_dir / f"{beh}.json").write_text(json.dumps(res, indent=2))
        per_behavior[beh] = res
        dv_a_rho = res["dv_a_base_rate"].get("raw_rho")
        lk = res["leakage_bake_off"]
        dp = lk.get("doubly_partialled_rho")
        print(
            f"[phase=bake_off] behavior={beh} dv_a_rho={dv_a_rho} "
            f"doubly_partialled_rho={dp} in_scope={lk.get('in_scope')} "
            f"primary={res.get('primary_vs_secondary')} k2_pass={res.get('k2_pass')} "
            f"md5={_digest_file(beh_path)} -> {beh_path}",
            flush=True,
        )

    h3 = _h3_summary(per_behavior)
    summary = {
        "schema_version": 1,
        "metadata": reproducibility_metadata({"behaviors": behaviors, "smoke": args.smoke}),
        "h3": h3,
        "marker_panel_overlap_gate": marker_overlap_decision,
        "anchors": {
            beh: per_behavior[beh].get("replication_anchor")
            for beh in per_behavior
            if per_behavior[beh].get("replication_anchor") is not None
        },
        "per_behavior_paths": {
            beh: str(out_dir / f"{beh}_alignment_predictor.json") for beh in per_behavior
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # ── §6.5 PRIMARY DELIVERABLE: the top-level alignment_predictor.json ──
    # The upload-verifier's completeness gate + the analyzer key off this single
    # file. It carries per-behavior raw / singly / doubly-partialled rho,
    # prior_centroid_projection, ΔR², the 3-point reliability-band rho/ΔR²,
    # primary-vs-secondary, marker_direction_kind, marker_panel_overlap, per-behavior
    # k2_pass, and CIs (where bootstrapped).
    alignment_predictor = _build_alignment_predictor(
        per_behavior, h3, marker_overlap_decision, behaviors, args.smoke
    )
    ap_path = out_dir / "alignment_predictor.json"
    ap_path.write_text(json.dumps(alignment_predictor, indent=2))

    print(
        f"[phase=bake_off] H3 verdict={h3['h3_verdict']} "
        f"cleared={h3['primary_behaviors_cleared']} "
        f"effective_primary={h3['effective_primary_h3_behaviors']} "
        f"-> {summary_path}; primary_deliverable -> {ap_path} (md5={_digest_file(ap_path)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
