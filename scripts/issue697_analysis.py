"""Issue #697 — CPU analysis: f_CV bootstrap + hero figure (off-pod, 0 GPU).

Consumes the per-cell ``eval_results/issue_697/patch/*.pt`` tensors (each carries
per (persona, question) reads for every condition at every layer; the analysis
runs against the LOCAL copies on the VM after the pod uploads + terminates) and
computes, per behavior, the context-vector-mediated fraction in v-space:

  f_CV       = ((v_Pup - v0)·d) / ((v⁺ - v0)·d),   d = (v⁺ - v0)/‖·‖   (P↑ sufficiency)
  f_CV_down  = 1 - ((v_Pdown - v0)·d)/((v⁺-v0)·d)                       (P↓ necessity)

at each cell's per-behavior PRIMARY pooling (mean-resp em/syc, slot marker/fact —
plan §4.5 item-5), with the random-CV / other-context conditions as the "patch
did something" null floor. Bootstrap 95% CI is over the 280 personaxquestion
pairs PERSONA-CLUSTERED (resample personas, then questions within — the
Statistics-critic standing rec) so the CI respects the panel's two-level
structure. A cell with ‖v⁺-v0‖ < eps is reported ``no-effect`` (never an extreme
ratio). The hero is a 2x4 grid (rows: f_CV / f_CV^E; cols: em/syc/marker/fact).

P↑/P↓ cross-check (plan §6.3): the verdict requires the P↑ and P↓ CIs to AGREE.
When they disagree (the CIs do not overlap) the cell-level verdict is
``patch-inconsistent`` rather than a confident ``context-vector-moved`` /
``mapping-changed``.

full_span (plan control #4, §4.3/§5): the FT c⁺ overwritten at EVERY context
position (a distinct donor per position) is the single-slot under-count UPPER
BOUND. We compute its f_CV alongside the last-token f_CV and report the
last-token-vs-full-span delta — if last-token f_CV << full-span f_CV the slot is
too narrow (reported, never silently dropped).

The behavioral-E row (f_CV^E) is computed from the off-pod judge phase's
``{cell}_judged.json`` outputs (per-condition judged rates over the captured
on-policy generations — Sonnet judge) for em/syc/fact and from the marker-arm
TF marker-logp records in ``{cell}_E_metadata.json`` (``dv_kind=marker_logp``):
``f_CV^E = (E_Pup - E0) / (E+ - E0)`` (and its P↓ analog). When no judged file
is present for a behavior, that column renders the labeled "E not yet judged"
placeholder (only then) — judged data, when present, drives the row.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.analysis.cv_patch import NO_EFFECT, compute_f_cv, compute_f_cv_down

logger = logging.getLogger("issue697_analysis")

BEHAVIORS = ("em", "sycophancy", "marker", "fact")
PRIMARY_POOLING = {"em": "mean_resp", "sycophancy": "mean_resp", "marker": "slot", "fact": "slot"}
BAND_LOW, BAND_HIGH = 0.3, 0.7
N_BOOTSTRAP = 1000


def _per_question_f_cv(cell: dict, layer: int) -> dict:
    """Per (persona, question) f_CV / f_CV_down / full_span / null-floor at the primary pooling.

    ``f_cv_full_span`` is the f_CV of the full_span condition (plan control #4 —
    the FT c⁺ overwritten at EVERY context position, the single-slot under-count
    UPPER BOUND); it is read only when the cell persisted ``conditions["full_span"]``
    (older .pt without it leave the list empty). Returns
    ``{"f_cv": [...], "f_cv_down": [...], "f_cv_full_span": [...],
    "f_cv_random": [...], "personas": [...], "n_no_effect": int}`` — parallel
    lists over kept pairs (full_span padded with NaN when absent, so it stays
    index-aligned with the others).
    """
    behavior = cell["behavior"]
    pooling = PRIMARY_POOLING.get(behavior, "mean_resp")
    f_cv, f_cv_down, f_cv_full_span, f_cv_random, personas = [], [], [], [], []
    n_no_effect = 0
    for p_name, entries in cell["per_q"].items():
        for e in entries:
            v0 = e["v0"][layer][pooling]
            vplus = e["vplus"][layer][pooling]
            v_pup = e["conditions"]["p_up"][layer][pooling]
            v_pdown = e["conditions"]["p_down"][layer][pooling]
            v_rand = e["conditions"]["random_cv"][layer][pooling]
            f = compute_f_cv(v_pup, v0, vplus)
            fd = compute_f_cv_down(v_pdown, v0, vplus)
            fr = compute_f_cv(v_rand, v0, vplus)
            if f == NO_EFFECT:
                n_no_effect += 1
                continue
            # full_span condition (plan control #4) — present only on .pt files
            # written by the cell driver that persists it; read it where it exists.
            v_full = (
                e["conditions"].get("full_span", {}).get(layer, {}).get(pooling)
                if "full_span" in e["conditions"]
                else None
            )
            if v_full is not None:
                ff = compute_f_cv(v_full, v0, vplus)
                f_cv_full_span.append(float(ff) if ff != NO_EFFECT else np.nan)
            else:
                f_cv_full_span.append(np.nan)
            f_cv.append(float(f))
            f_cv_down.append(float(fd) if fd != NO_EFFECT else np.nan)
            f_cv_random.append(float(fr) if fr != NO_EFFECT else np.nan)
            personas.append(p_name)
    return {
        "f_cv": f_cv,
        "f_cv_down": f_cv_down,
        "f_cv_full_span": f_cv_full_span,
        "f_cv_random": f_cv_random,
        "personas": personas,
        "n_no_effect": n_no_effect,
    }


class ReadInertError(AssertionError):
    """Raised by ``assert_not_read_inert`` when a cell's response-slot read is
    structurally independent of the context-slot patch (the v3 defect)."""


def assert_not_read_inert(cell: dict, read_layer: int) -> dict:
    """Raise ``ReadInertError`` iff the cell's v read is INERT (plan §7.1 / F2).

    The v3 read pathway installed the patch at the SAME layer it read v, in ONE
    teacher-forced forward — so the response-slot read was computed BEFORE the
    context-slot overwrite and the patch never reached it (``‖v_Pup-v0‖=0`` for
    all pairs). The detector re-runs ``_per_question_f_cv`` at the read layer and
    fires when the P↑, random-CV, full_span AND P↓ reads are ALL ≈0.

    The P↓ threshold is ``abs(pdn) < 1e-4`` (plan v5 fix — NOT ``abs(pdn - 1.0)``):
    an inert into-FT read pins ``v_Pdown ≡ v⁺``, so
    ``compute_f_cv_down(v⁺, v0, v⁺) = 1 - 1 = 0.0``. The v4 detector used
    ``abs(pdn - 1.0) < 1e-4`` and was silently NON-functional — it never fired on
    the very inert pattern the revision exists to catch (round-2 fact-checker
    reproduction on the salvaged ``marker_sp_swe_seed42.pt``).

    Returns the means dict ``{f_cv, f_cv_random, f_cv_full_span, f_cv_down, n}`` on
    a NON-inert cell (the §7.1b positive gate caller logs it). On an inert cell it
    RAISES — used as the §7.1a negative control (the detector MUST fire on the
    salvaged inert cell) AND the §7.1b positive gate (the new smoke cell MUST NOT
    be inert before the production sweep dispatches).
    """
    fcv = _per_question_f_cv(cell, read_layer)
    pup = float(np.nanmean(fcv["f_cv"])) if fcv["f_cv"] else float("nan")
    rand = float(np.nanmean(fcv["f_cv_random"])) if fcv["f_cv_random"] else float("nan")
    span = float(np.nanmean(fcv["f_cv_full_span"])) if fcv["f_cv_full_span"] else float("nan")
    pdn = float(np.nanmean(fcv["f_cv_down"])) if fcv["f_cv_down"] else float("nan")
    means = {
        "f_cv": pup,
        "f_cv_random": rand,
        "f_cv_full_span": span,
        "f_cv_down": pdn,
        "n": len(fcv["f_cv"]),
    }
    inert = (
        abs(pup) < 1e-4
        and abs(rand) < 1e-4
        and (np.isnan(span) or abs(span) < 1e-4)
        and abs(pdn) < 1e-4
    )
    if inert:
        raise ReadInertError(
            "READ-INERT REGRESSION: response-slot read is independent of the "
            f"context-slot patch (mean f_CV[p_up]={pup:.6f}, [random_cv]={rand:.6f}, "
            f"[full_span]={span:.6f}, f_CV_down={pdn:.6f}). The patch layer must be "
            "< the read layer (Option B, plan §4.0/§7.1)."
        )
    return means


def _persona_clustered_bootstrap(values: list[float], personas: list[str], n_reps: int) -> dict:
    """Persona-clustered bootstrap 95% CI of the mean (resample personas, then
    questions within each — respects the panel's two-level structure)."""
    if not values:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    vals = np.asarray(values, dtype=np.float64)
    pers = np.asarray(personas)
    uniq = np.unique(pers)
    by_persona = {p: vals[pers == p] for p in uniq}
    rng = np.random.default_rng(697)
    means = np.empty(n_reps)
    for r in range(n_reps):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        pooled = []
        for p in chosen:
            arr = by_persona[p]
            if len(arr):
                pooled.append(rng.choice(arr, size=len(arr), replace=True))
        cat = np.concatenate(pooled) if pooled else vals
        means[r] = np.nanmean(cat)
    return {
        "mean": float(np.nanmean(vals)),
        "ci_low": float(np.nanpercentile(means, 2.5)),
        "ci_high": float(np.nanpercentile(means, 97.5)),
        "n": len(vals),
    }


def _band(ci: dict) -> str:
    """Single-CI band label (plan §6.3): the CI lies entirely within one band, else 'mixed'."""
    lo, hi = ci["ci_low"], ci["ci_high"]
    if np.isnan(lo) or np.isnan(hi):
        return "no-effect"
    if lo >= BAND_HIGH:
        return "context-vector-moved"
    if hi <= BAND_LOW:
        return "mapping-changed"
    return "mixed"


def _cis_disagree(ci_up: dict, ci_down: dict) -> bool:
    """True when the P↑ and P↓ CIs do NOT overlap (plan §6.3 patch-inconsistent).

    The P↓ form (1 - progress) agrees with P↑ for a confident cell — both ~1 when
    the context vector moved, both ~0 when the mapping changed. Non-overlapping
    CIs mean the two patches tell DIFFERENT stories.
    """
    for ci in (ci_up, ci_down):
        if np.isnan(ci["ci_low"]) or np.isnan(ci["ci_high"]):
            return False  # can't cross-check a no-effect / empty arm
    # Disjoint iff one's high < the other's low.
    return ci_up["ci_high"] < ci_down["ci_low"] or ci_down["ci_high"] < ci_up["ci_low"]


def _verdict(ci_up: dict, ci_down: dict) -> str:
    """Pre-registered band verdict (plan §6.3) cross-checked against P↓.

    Returns ``patch-inconsistent`` when the P↑ and P↓ CIs disagree (don't
    overlap) — the two patches must agree for a confident cell-level verdict.
    Otherwise returns the P↑ band label (``context-vector-moved`` /
    ``mapping-changed`` / ``mixed`` / ``no-effect``).
    """
    if _cis_disagree(ci_up, ci_down):
        return "patch-inconsistent"
    return _band(ci_up)


# The judged-rate scalar to read per generation-behavior from {cell}_judged.json's
# rates[condition] dict (the off-pod Sonnet judge output). em → Betley P(mis);
# sycophancy → agreement rate; fact → taught-fact rate.
E_RATE_KEY = {"em": "p_mis", "sycophancy": "rate", "fact": "rate_taught"}
# The judge writes the four E conditions; analyze reads the cross-model patch pair.
E_COND_BASE = "unpatched_base"  # E0  (base model, no patch)
E_COND_FT = "unpatched_ft"  # E+  (FT model, no patch)
E_COND_PUP = "p_up"  # E_Pup (base + FT context vector → "context moved" arm)
E_COND_PDOWN = "p_down"  # E_Pdown (FT - FT context vector → necessity arm)


def _f_cv_e_from_rates(e0: float, eplus: float, e_pup: float, *, eps: float = 1e-6):
    """f_CV^E = (E_Pup - E0) / (E+ - E0); None when |E+ - E0| < eps (no behavioral effect)."""
    if any(v is None or np.isnan(v) for v in (e0, eplus, e_pup)):
        return None
    denom = eplus - e0
    if abs(denom) < eps:
        return None
    return (e_pup - e0) / denom


def _e_space_from_judged(judged: dict, behavior: str) -> dict | None:
    """f_CV^E from a {cell}_judged.json's per-condition judged rates.

    Returns ``{"f_cv_e": float|None, "f_cv_e_down": float|None, "rates": {...}}``
    or ``None`` when the required conditions are absent.
    """
    rates = judged.get("rates", {})
    key = E_RATE_KEY.get(behavior)
    if key is None:
        return None

    def _rate(cond: str):
        r = rates.get(cond)
        return None if r is None else r.get(key)

    e0, eplus = _rate(E_COND_BASE), _rate(E_COND_FT)
    e_pup, e_pdown = _rate(E_COND_PUP), _rate(E_COND_PDOWN)
    if e0 is None or eplus is None:
        return None
    f_e = _f_cv_e_from_rates(e0, eplus, e_pup) if e_pup is not None else None
    # P↓ analog: 1 - (E_Pdown - E0)/(E+ - E0) (necessity), same form as v-space.
    f_e_down = None
    if e_pdown is not None:
        prog = _f_cv_e_from_rates(e0, eplus, e_pdown)
        f_e_down = None if prog is None else 1.0 - prog
    return {
        "f_cv_e": f_e,
        "f_cv_e_down": f_e_down,
        "rates": {"E0": e0, "Eplus": eplus, "E_Pup": e_pup, "E_Pdown": e_pdown},
    }


def _marker_logp_value(read) -> float | None:
    """Extract the marker log P from either storage shape (F3 back-compat).

    v5 (#697 F3) persists FOUR floats per slot as a dict
    ``{"log_p", "z_marker", "z_eos", "logZ"}``; the salvaged v3 cells persisted ONE
    raw float (``log P``). Read ``log_p`` from a dict, the float itself from a
    scalar, ``None`` from a missing read.
    """
    if read is None:
        return None
    if isinstance(read, dict):
        return read.get("log_p")
    return float(read)


def _e_space_from_marker(meta: dict) -> dict | None:
    """f_CV^E for the MARKER arm from {cell}_E_metadata.json TF marker-logp records.

    The marker E DV is the teacher-forced log P(` ※`) trained - base at the slot,
    under each condition (``dv_kind=marker_logp``). f_CV^E uses the same ratio in
    log-prob space: E0=base, E+=FT, E_Pup=base+FT-CV. Computed as RATIO-OF-MEANS on
    the cell-aggregate quantities — ``(mean(E_Pup) - mean(E0)) / (mean(E+) - mean(E0))``
    — matching the plan §6.1/§6.4 formula registration (the bootstrap is over the
    shared question axis, so the per-cell scalar is formed from cell-aggregate means,
    NOT averaged per-record ratios; this also mirrors the judged path
    ``_e_space_from_judged``, which forms the ratio once from one aggregate rate per
    condition).

    Each per-condition read is now a FOUR-float dict (F3: log P / z_marker / z_eos /
    logZ); ``_marker_logp_value`` extracts ``log_p`` and falls back to the salvaged
    v3 single-float shape, so the analysis reads both the new and the legacy cells.
    """
    if meta.get("dv_kind") != "marker_logp":
        return None
    recs = meta.get("marker_e_records", [])
    if not recs:
        return None
    e0s, epluss, epups, epdowns = [], [], [], []
    for r in recs:
        e0 = _marker_logp_value(r.get("marker_logp_unpatched_base"))
        eplus = _marker_logp_value(r.get("marker_logp_unpatched_ft"))
        e_pup = _marker_logp_value(r.get("marker_logp_p_up"))
        e_pdown = _marker_logp_value(r.get("marker_logp_p_down"))
        if e0 is None or eplus is None:
            continue
        e0s.append(e0)
        epluss.append(eplus)
        if e_pup is not None:
            epups.append(e_pup)
        if e_pdown is not None:
            epdowns.append(e_pdown)
    if not e0s:
        return None
    # Ratio-of-means at the cell level (plan §6.1/§6.4). The aggregate means below
    # ARE the registered E0/E+/E_Pup/E_Pdown; form the ratio once from them.
    mean_e0 = float(np.mean(e0s))
    mean_eplus = float(np.mean(epluss))
    mean_epup = float(np.mean(epups)) if epups else None
    mean_epdown = float(np.mean(epdowns)) if epdowns else None
    f_e = _f_cv_e_from_rates(mean_e0, mean_eplus, mean_epup) if mean_epup is not None else None
    # P↓ analog: 1 - (E_Pdown - E0)/(E+ - E0) (necessity), same form as the judged path.
    f_e_down = None
    if mean_epdown is not None:
        prog = _f_cv_e_from_rates(mean_e0, mean_eplus, mean_epdown)
        f_e_down = None if prog is None else 1.0 - prog
    return {
        "f_cv_e": f_e,
        "f_cv_e_down": f_e_down,
        "rates": {
            "E0": mean_e0,
            "Eplus": mean_eplus,
            "E_Pup": mean_epup,
            "E_Pdown": mean_epdown,
        },
    }


def _load_e_space(patch_dir: Path, cell_id: str, behavior: str) -> dict | None:
    """Load + compute the cell's f_CV^E from judged outputs (em/syc/fact) or the
    marker TF-logp metadata (marker). Returns None when no E data is present."""
    if behavior == "marker":
        meta_path = patch_dir / f"{cell_id}_E_metadata.json"
        if not meta_path.exists():
            return None
        return _e_space_from_marker(json.loads(meta_path.read_text()))
    judged_path = patch_dir / f"{cell_id}_judged.json"
    if not judged_path.exists():
        return None
    return _e_space_from_judged(json.loads(judged_path.read_text()), behavior)


def load_537_ctx_gating(repo_root: Path) -> dict | None:
    """Load #537's per-(behavior, context) context-gating read (plan §6.5 / F4).

    The F4 restriction joins #537's committed
    ``eval_results/issue_537/analysis/registered_reads.json`` ``per_row_breadth``
    (per-behavior, per-context ``diag`` install / ``offdiag_mean`` breadth /
    ``breadth_diagnorm`` / ``implant_failed``) so the analyzer can restrict any
    behavioral-E "mapping changed" verdict to cells where #537 measured REAL
    context-gating (the behavior actually installed under the trained context).
    Returns the ``per_row_breadth`` dict, or None when the #537 artifact is absent
    (off-pod join; a WARN, NOT a sweep blocker — the analyzer reports the
    restriction as not-applied + names the gap, plan §6.5/A12).
    """
    p = repo_root / "eval_results" / "issue_537" / "analysis" / "registered_reads.json"
    if not p.exists():
        logger.warning(
            "F4: #537 ctx-gating read absent at %s -- restriction NOT applied "
            "(reported as a gap, not a sweep blocker; plan §6.5/A12)",
            p,
        )
        return None
    pb = json.loads(p.read_text()).get("per_row_breadth")
    if not isinstance(pb, dict):
        logger.warning(
            "F4: #537 registered_reads.json lacks per_row_breadth -- restriction skipped"
        )
        return None
    return pb


def cell_ctx_gating_status(per_row_breadth: dict | None, behavior: str, cid: str) -> dict:
    """Per-(B, C) #537 context-gating status for one #697 cell (plan §6.5 / F4).

    A cell is ``context_gated`` when #537 shows the behavior actually installed
    under the trained context (``not implant_failed`` AND ``diag > 0``) — only then
    is there a context-conditional behavior to causally localize. A cell #537 shows
    was NOT context-gated cannot support a clean CV-localization read, so the
    analyzer labels it rather than emitting a mapping-changed verdict. Returns
    ``{"status": "context-gated"|"not-context-gated"|"not-available",
    "diag", "offdiag_mean", "breadth_diagnorm", "implant_failed"}``.
    """
    if per_row_breadth is None:
        return {"status": "not-available"}
    row = per_row_breadth.get(behavior, {}).get(cid)
    if not isinstance(row, dict):
        return {"status": "not-available"}
    implant_failed = bool(row.get("implant_failed", False))
    diag = row.get("diag")
    gated = (not implant_failed) and (diag is not None and float(diag) > 0.0)
    return {
        "status": "context-gated" if gated else "not-context-gated",
        "diag": diag,
        "offdiag_mean": row.get("offdiag_mean"),
        "breadth_diagnorm": row.get("breadth_diagnorm"),
        "implant_failed": implant_failed,
    }


def analyze(repo_root: Path, *, primary_layer: int) -> dict:
    patch_dir = repo_root / "eval_results" / "issue_697" / "patch"
    pts = sorted(patch_dir.glob("*.pt"))
    if not pts:
        raise RuntimeError(f"no per-cell .pt tensors in {patch_dir} -- run the sweep first")
    logger.info("[phase=analyze] %d per-cell tensors in %s", len(pts), patch_dir)

    # #537 context-gating read for the F4 restriction (off-pod join; WARN-not-fail
    # when absent — plan §6.5/A12).
    ctx_gating = load_537_ctx_gating(repo_root)

    by_behavior: dict[str, dict] = {
        b: {
            "f_cv": [],
            "f_cv_down": [],
            "f_cv_full_span": [],
            "personas": [],
            "f_cv_random": [],
            "cells": [],
            "e_cv": [],  # per-cell f_CV^E (P↑)
            "e_cv_down": [],  # per-cell f_CV^E (P↓ analog)
            "e_personas": [],  # cell_id label per e_cv point (for the bootstrap cluster)
            # F4: per-cell #537 ctx-gating status + the context-gated E-cv subset.
            "ctx_gating": {},  # cell_id -> gating status dict
            "e_cv_gated": [],  # f_CV^E only for #537-context-gated cells
            "e_personas_gated": [],
        }
        for b in BEHAVIORS
    }
    for pt in pts:
        cell = torch.load(pt, weights_only=False)
        behavior = cell["behavior"]
        if behavior not in by_behavior:
            continue
        layer = primary_layer if primary_layer in cell["layers"] else cell["primary_layer"]
        pq = _per_question_f_cv(cell, layer)
        by_behavior[behavior]["f_cv"] += pq["f_cv"]
        by_behavior[behavior]["f_cv_down"] += pq["f_cv_down"]
        by_behavior[behavior]["f_cv_full_span"] += pq["f_cv_full_span"]
        by_behavior[behavior]["f_cv_random"] += pq["f_cv_random"]
        by_behavior[behavior]["personas"] += pq["personas"]
        by_behavior[behavior]["cells"].append(cell["cell_id"])
        # F4: this cell's #537 context-gating status (keyed by the train cid).
        gating = cell_ctx_gating_status(ctx_gating, behavior, str(cell.get("cid", "")))
        by_behavior[behavior]["ctx_gating"][cell["cell_id"]] = gating
        # E-space: f_CV^E from the off-pod judge (em/syc/fact) or marker TF-logp.
        e = _load_e_space(patch_dir, cell["cell_id"], behavior)
        if e is not None:
            if e.get("f_cv_e") is not None:
                by_behavior[behavior]["e_cv"].append(float(e["f_cv_e"]))
                by_behavior[behavior]["e_personas"].append(cell["cell_id"])
                # F4: the context-gated E subset for the restricted verdict.
                if gating.get("status") == "context-gated":
                    by_behavior[behavior]["e_cv_gated"].append(float(e["f_cv_e"]))
                    by_behavior[behavior]["e_personas_gated"].append(cell["cell_id"])
            if e.get("f_cv_e_down") is not None:
                by_behavior[behavior]["e_cv_down"].append(float(e["f_cv_e_down"]))
        logger.info(
            "  cell %s: %d pairs (%d no-effect) at layer %d pooling %s%s",
            cell["cell_id"],
            len(pq["f_cv"]),
            pq["n_no_effect"],
            layer,
            PRIMARY_POOLING.get(behavior),
            "" if e is None else f" [E f_cv_e={e.get('f_cv_e')}]",
        )

    summary: dict[str, dict] = {}
    for behavior in BEHAVIORS:
        d = by_behavior[behavior]
        ci = _persona_clustered_bootstrap(d["f_cv"], d["personas"], N_BOOTSTRAP)
        # P↓ cross-check CI (necessity arm), persona-clustered like P↑.
        down_vals = [(p, v) for p, v in zip(d["personas"], d["f_cv_down"], strict=True)]
        ci_down = _persona_clustered_bootstrap(
            [v for _p, v in down_vals if not np.isnan(v)],
            [p for p, v in down_vals if not np.isnan(v)],
            N_BOOTSTRAP,
        )
        # full_span CI (plan control #4: the single-slot under-count UPPER BOUND).
        fs_vals = [(p, v) for p, v in zip(d["personas"], d["f_cv_full_span"], strict=True)]
        ci_full = _persona_clustered_bootstrap(
            [v for _p, v in fs_vals if not np.isnan(v)],
            [p for p, v in fs_vals if not np.isnan(v)],
            N_BOOTSTRAP,
        )
        ci_null = _persona_clustered_bootstrap(
            [v for v in d["f_cv_random"] if not np.isnan(v)],
            [p for p, v in zip(d["personas"], d["f_cv_random"], strict=True) if not np.isnan(v)],
            N_BOOTSTRAP,
        )
        # E-space CIs, clustered by cell_id (one f_CV^E per cell).
        ci_e = _persona_clustered_bootstrap(d["e_cv"], d["e_personas"], N_BOOTSTRAP)
        ci_e_down = (
            _persona_clustered_bootstrap(
                d["e_cv_down"], d["e_personas"][: len(d["e_cv_down"])], N_BOOTSTRAP
            )
            if d["e_cv_down"]
            else {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
        )
        # F4 (plan §6.5): the behavioral-E CI restricted to #537-context-gated cells.
        ci_e_gated = _persona_clustered_bootstrap(
            d["e_cv_gated"], d["e_personas_gated"], N_BOOTSTRAP
        )
        gating_statuses = [g.get("status") for g in d["ctx_gating"].values()]
        n_gated = sum(s == "context-gated" for s in gating_statuses)
        n_not_gated = sum(s == "not-context-gated" for s in gating_statuses)
        n_gating_na = sum(s == "not-available" for s in gating_statuses)
        ctx_gating_applied = ctx_gating is not None and n_gating_na < len(gating_statuses)
        # last-token vs full-span scope delta (plan control #4).
        last_vs_full = (
            float(ci["mean"] - ci_full["mean"])
            if (not np.isnan(ci["mean"]) and not np.isnan(ci_full["mean"]))
            else float("nan")
        )
        # F4 behavioral-E verdict: any "mapping-changed" read is RESTRICTED to the
        # #537-context-gated cells (a non-gated cell has no context-conditional
        # behavior to localize). When the gating join is unavailable, the verdict
        # is the unrestricted E-space band + a not-applied flag (plan §6.5/A12).
        e_verdict_gated = _band(ci_e_gated) if ctx_gating_applied else "ctx-gating-not-applied"
        summary[behavior] = {
            "f_cv_ci": ci,
            "f_cv_down_ci": ci_down,
            "f_cv_full_span_ci": ci_full,
            "last_token_vs_full_span_delta": last_vs_full,
            "f_cv_e_ci": ci_e,
            "f_cv_e_down_ci": ci_e_down,
            "null_floor_ci": ci_null,
            "verdict": _verdict(ci, ci_down),
            "n_cells": len(d["cells"]),
            "n_e_cells": ci_e["n"],
            "primary_pooling": PRIMARY_POOLING.get(behavior),
            # F4 #537 context-gating restriction (plan §6.5).
            "f_cv_e_ctx_gated_ci": ci_e_gated,
            "e_verdict_ctx_gated": e_verdict_gated,
            "ctx_gating_applied": ctx_gating_applied,
            "n_ctx_gated_cells": n_gated,
            "n_not_ctx_gated_cells": n_not_gated,
            "n_ctx_gating_unavailable": n_gating_na,
            "per_cell_ctx_gating": d["ctx_gating"],
        }
        logger.info(
            "  %s: f_CV=%.3f [%.3f, %.3f] verdict=%s (P↓=%.3f, full_span=%.3f, "
            "Δlast-vs-full=%.3f, E f_CV^E=%.3f over %d cells, null floor %.3f) "
            "[F4 ctx-gated %d/%d cells, E-verdict(gated)=%s applied=%s]",
            behavior,
            ci["mean"],
            ci["ci_low"],
            ci["ci_high"],
            summary[behavior]["verdict"],
            ci_down["mean"],
            ci_full["mean"],
            last_vs_full,
            ci_e["mean"],
            ci_e["n"],
            ci_null["mean"],
            n_gated,
            len(gating_statuses),
            e_verdict_gated,
            ctx_gating_applied,
        )
    return {"primary_layer": primary_layer, "by_behavior": summary, "raw": by_behavior}


def render_hero(result: dict, out_path: Path) -> None:
    """The hero 2x4 grid (rows: f_CV / f_CV^E; cols: em/syc/marker/fact)."""
    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    import matplotlib.pyplot as plt

    # paper_palette_role(role) returns a single hex string per role.
    c_primary = paper_palette_role("primary")
    c_neutral = paper_palette_role("neutral")
    c_control = paper_palette_role("control")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey="row")
    for ci_col, behavior in enumerate(BEHAVIORS):
        s = result["by_behavior"][behavior]
        ci = s["f_cv_ci"]
        ax = axes[0, ci_col]
        if not np.isnan(ci["mean"]):
            ax.errorbar(
                [0],
                [ci["mean"]],
                yerr=[[ci["mean"] - ci["ci_low"]], [ci["ci_high"] - ci["mean"]]],
                fmt="o",
                color=c_primary,
                capsize=4,
            )
        # full_span comparison point (plan control #4): the single-slot under-count
        # UPPER BOUND, plotted at x=1 beside the last-token f_CV so the
        # last-token-vs-full-span scope delta is visible in the hero.
        ci_full = s.get("f_cv_full_span_ci", {})
        has_full = bool(ci_full) and not np.isnan(ci_full.get("mean", float("nan")))
        if has_full:
            ax.errorbar(
                [1],
                [ci_full["mean"]],
                yerr=[
                    [ci_full["mean"] - ci_full["ci_low"]],
                    [ci_full["ci_high"] - ci_full["mean"]],
                ],
                fmt="s",
                color=c_control,
                capsize=4,
                label="full_span",
            )
        null = s["null_floor_ci"]
        if not np.isnan(null["mean"]):
            ax.axhspan(null["ci_low"], null["ci_high"], color=c_neutral, alpha=0.25)
        ax.axhline(BAND_LOW, ls="--", lw=0.8, color=c_control)
        ax.axhline(BAND_HIGH, ls="--", lw=0.8, color=c_control)
        ax.set_title(f"{behavior}\n({s['verdict']})", fontsize=9)
        if has_full:
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["last-tok", "full-span"], fontsize=7)
        else:
            ax.set_xticks([])
        if ci_col == 0:
            ax.set_ylabel("f_CV (v-space)")
        ax.set_ylim(-0.2, 1.2)
        # E-space row: the f_CV^E errorbar when judged data is present, ELSE the
        # labeled "E not yet judged" placeholder (only when genuinely absent).
        axe = axes[1, ci_col]
        ci_e = s.get("f_cv_e_ci", {})
        has_e = bool(ci_e) and not np.isnan(ci_e.get("mean", float("nan")))
        if has_e:
            axe.errorbar(
                [0],
                [ci_e["mean"]],
                yerr=[[ci_e["mean"] - ci_e["ci_low"]], [ci_e["ci_high"] - ci_e["mean"]]],
                fmt="o",
                color=c_primary,
                capsize=4,
            )
            axe.axhline(BAND_LOW, ls="--", lw=0.8, color=c_control)
            axe.axhline(BAND_HIGH, ls="--", lw=0.8, color=c_control)
            axe.set_title(f"n={ci_e.get('n', 0)} cells", fontsize=8)
            axe.set_xticks([])
            axe.set_ylim(-0.2, 1.2)
        else:
            axe.text(
                0.5,
                0.5,
                "E not yet judged\n(no judged outputs)",
                ha="center",
                va="center",
                fontsize=8,
                transform=axe.transAxes,
            )
            axe.set_xticks([])
            axe.set_yticks([])
        if ci_col == 0:
            axe.set_ylabel("f_CV^E (E-space)")
    fig.suptitle("Issue #697 — context-vector-mediated fraction f_CV per behavior", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, out_path)
    plt.close(fig)
    logger.info("wrote hero figure %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--primary-layer", type=int, default=14)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )
    import subprocess

    repo_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    )
    result = analyze(repo_root, primary_layer=args.primary_layer)

    out_dir = repo_root / "eval_results" / "issue_697"
    out_dir.mkdir(parents=True, exist_ok=True)
    # strip the raw per-pair lists out of the JSON summary (keep it small).
    summary_json = {"primary_layer": result["primary_layer"], "by_behavior": result["by_behavior"]}
    (out_dir / "f_cv_summary.json").write_text(json.dumps(summary_json, indent=2, default=float))
    logger.info("wrote %s", out_dir / "f_cv_summary.json")

    fig_path = repo_root / "figures" / "issue_697" / "hero_f_cv_2x4.png"
    render_hero(result, fig_path)
    logger.info("[phase=analyze_done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
