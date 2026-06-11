# ruff: noqa: RUF002
"""Issue #491 follow-up round 1 (ft-content-control) off-pod statistics.

Plan v4 §3b/§4.7/§6: the helpful-content FT cell (``ft_ctrl_helpful_content``
— villain wrapper + helpful rows + marker) read against the parent's
committed eval JSONs. Runs on the VM; zero GPU. Reuses the parent's
assemble/bootstrap machinery (``analyze.py``): n_boot 10,000, seed 42.

Reads:
  (a) PRIMARY profile ρ(new cell, ft_K8_chainA @ matched step) over the 9
      non-source contexts — raw AND disattenuated, joint question-bootstrap
      CI, spread-validity gate on the new cell.
  (b) MAGNITUDE: the new cell's non-source mean ΔG vs the villain replicate
      envelope (chains A/B/C at their matched steps, recomputed from the
      committed by_run + full-step JSONs).
  (c) gate ρ per layer (all 28; layer 19 named; source-included / excluded /
      base-prior partial; question-bootstrap CI at layer 19, source-excluded).
  (d) registered analyzer reads (§6): equal-step profile comparison (full/
      full + traj/traj at shared grid steps; step 12 named); base-prior
      partial of the profile ρ; recomputed + persisted reference ceilings.
  (e) secondary descriptive ρ vs ``ft_ctrl_helpful_rows`` and vs
      ``icl_ctrl_helpful_marker``.

The §3b verdict fields (rank condition / magnitude condition / joint CUT
flag) are computed MECHANICALLY; interpretation is the analyzer's job. If
matching only reached closest-approach outside ±1.5 nat, the verdict table is
NOT run (plan §3b) — the band-entry fallback read is reported descriptively.

Output: ``eval_results/issue_491/ft-content-control/followup_analysis.json``
(followup-label artifact convention).

``--stub-new-cell-from <run_id>`` is the CPU smoke: it runs the IDENTICAL
code path treating a committed parent run as the new cell (e.g.
``ft_K8_chainB`` — a villain replicate, so the rank + magnitude conditions
must PASS and the recomputed ceilings must reproduce the plan §3b registered
values); output goes to ``followup_analysis_smoke_stub.json``, never the
real artifact path.
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.icl_vs_ft_491.analyze import (
    DISATT_RHO_CONFIRM,
    DISATT_RHO_FALSIFY,
    N_BOOT,
    RAW_RHO_CONFIRM,
    SEED,
    SPREAD_MIN_NATS,
    _partial_spearman,
    _spearman,
    range_null_p95,
    split_half_reliability,
)
from explore_persona_space.experiments.icl_vs_ft_491.common import (
    EVAL_DIR,
    HF_BUCKET_491,
    N_LAYERS,
    PANEL_CONTEXT_IDS,
    SOURCE_CONTEXT,
    _hf_pull,
    repro_metadata,
    write_json,
)

logger = logging.getLogger("i491.analyze_followup")

FOLLOWUP_LABEL = "ft-content-control"
FOLLOWUP_RUN = "ft_ctrl_helpful_content"
VILLAIN_REF_RUN = "ft_K8_chainA"
VILLAIN_CHAIN_RUNS = ("ft_K8_chainA", "ft_K8_chainB", "ft_K8_chainC")
GATE_LAYER_KEY = "19"  # 1-indexed, matching analyze.h3_gate_correlations
GATE_RHO_CONFIRM = 0.85
GATE_RHO_FALSIFY = 0.5
OUT_DIR = EVAL_DIR / FOLLOWUP_LABEL

# Registered reference ceilings (plan v4 §3b, computed from committed parent
# JSONs at critique time). Persisted alongside the RECOMPUTED values so the
# analyzer can verify the recomputation reproduces the registration.
REGISTERED_CEILINGS = {
    "within_route_step8_vs_step12": {"A": 0.983, "B": 0.883, "C": 0.900},
    "cross_chain_matched": {"A-B": 0.933, "A-C": 0.950, "B-C": 0.983},
    "gate_rho_layer19_source_excluded_per_chain": {"A": 0.900, "B": 0.967, "C": 0.983},
}


# ── Input loaders (committed parent JSONs + this run's outputs) ──────────


def _load_full_profile(run_id: str, step: int | None = None) -> dict:
    """{run_id, step, contexts, questions, delta [C, Q]} from a full read.

    ``step=None`` resolves the run's matched step through the race-free
    per-run accessor (matching.py by_run files).
    """
    from explore_persona_space.experiments.icl_vs_ft_491.matching import load_matched_entry

    if step is None:
        step = int(load_matched_entry(run_id, smoke=False)["matched_step"])
    path = EVAL_DIR / "ft_panel" / f"{run_id}_full_step{step}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run the cell's eval pipeline first.")
    doc = json.loads(path.read_text())
    if doc.get("partial", False):
        raise AssertionError(f"{path} is marked partial — eval incomplete; refusing to read.")
    contexts = [c for c in PANEL_CONTEXT_IDS if c in doc["contexts"]]
    if contexts != PANEL_CONTEXT_IDS:
        raise AssertionError(f"{path}: contexts {contexts} != full 10-context panel")
    delta = np.array([doc["contexts"][c]["delta_logp"] for c in contexts], dtype=float)
    assert delta.shape == (len(contexts), len(doc["questions"])), delta.shape
    return {
        "run_id": run_id,
        "step": int(step),
        "contexts": contexts,
        "questions": doc["questions"],
        "delta": delta,
    }


def _full_step_profiles(run_id: str) -> dict[int, dict]:
    """All committed full reads for a run, keyed by step."""
    out: dict[int, dict] = {}
    for path in sorted((EVAL_DIR / "ft_panel").glob(f"{run_id}_full_step*.json")):
        step = int(path.stem.rsplit("_full_step", 1)[1])
        out[step] = _load_full_profile(run_id, step=step)
    return out


def _traj_step_means(run_id: str) -> dict:
    """{questions, per_step: {step: means [10]}} from the trajectory panel read."""
    path = EVAL_DIR / "ft_panel" / f"{run_id}_traj.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run the cell's eval pipeline first.")
    doc = json.loads(path.read_text())
    if doc.get("partial", False):
        raise AssertionError(f"{path} is marked partial — eval incomplete; refusing to read.")
    contexts = list(doc["contexts"])
    if contexts != PANEL_CONTEXT_IDS:
        raise AssertionError(f"{path}: contexts {contexts} != full 10-context panel")
    base_means = {c: float(np.mean([s["logp"] for s in doc["base"][c]["stats"]])) for c in contexts}
    per_step: dict[int, np.ndarray] = {}
    for step_str, entry in doc["per_step"].items():
        per_step[int(step_str)] = np.array(
            [
                float(np.mean([s["logp"] for s in entry[c]["stats"]])) - base_means[c]
                for c in contexts
            ]
        )
    return {"contexts": contexts, "questions": doc["questions"], "per_step": per_step}


def _icl_panel_profile(variant_id: str) -> dict:
    """{contexts, questions, delta [C, Q]} from an icl_panel variant JSON."""
    path = EVAL_DIR / "icl_panel" / f"{variant_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — parent ICL panel read absent.")
    doc = json.loads(path.read_text())
    contexts = [c for c in PANEL_CONTEXT_IDS if c in doc["contexts"]]
    qs = doc["contexts"][contexts[0]]["questions"]
    delta = np.array([doc["contexts"][c]["delta_logp"] for c in contexts], dtype=float)
    return {"contexts": contexts, "questions": qs, "delta": delta}


def _base_prior(contexts: list[str]) -> np.ndarray:
    """Per-context base log P(marker) from the no-prefix baseline (#532/#563)."""
    path = EVAL_DIR / "icl_panel" / "base_noprefix.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — parent baseline read absent.")
    base_ctx = json.loads(path.read_text())["contexts"]
    return np.array([float(np.mean([s["logp"] for s in base_ctx[c]["stats"]])) for c in contexts])


def _ensure_gate() -> dict:
    """Load gate_base_pos1 (download shift_summary.json from HF if absent).

    Fail-loud key assert (plan v4 §12 assumption 3): all 10 panel contexts
    present, 28 cosine values each.
    """
    from explore_persona_space.experiments.icl_vs_ft_491.activations import ACT_DIR

    path = ACT_DIR / "shift_summary.json"
    if not path.exists():
        _hf_pull(f"{HF_BUCKET_491}/analysis_tensors", "shift_summary.json", path)
    summary = json.loads(path.read_text())
    gate = summary.get("gate_base_pos1")
    if gate is None:
        raise AssertionError(f"{path} missing key 'gate_base_pos1' — wrong/old artifact.")
    for cid in PANEL_CONTEXT_IDS:
        if cid not in gate:
            raise AssertionError(f"{path}: gate_base_pos1 missing context {cid!r}")
        if len(gate[cid]["cosine"]) != N_LAYERS:
            raise AssertionError(
                f"{path}: gate_base_pos1[{cid!r}]['cosine'] has "
                f"{len(gate[cid]['cosine'])} layers, expected {N_LAYERS}"
            )
    return gate


def _nonsource_rows(profile: dict) -> tuple[list[str], np.ndarray]:
    idx = [i for i, c in enumerate(profile["contexts"]) if c != SOURCE_CONTEXT]
    return [profile["contexts"][i] for i in idx], profile["delta"][idx]


def _ci(xs: np.ndarray) -> list[float]:
    return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]


# ── (a)+(b)+(c) joint-bootstrap block ────────────────────────────────────


def primary_and_gate_block(
    new_prof: dict,
    ref_prof: dict,
    gate: dict,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[dict, dict, dict]:
    """(primary, gate_block, magnitude_new) — ONE joint question bootstrap.

    Per replicate, ONE Q_test resample drives the new cell's and the
    reference's per-context means, so the PRIMARY ρ (raw + disattenuated),
    the new cell's non-source mean ΔG, and the layer-19 source-excluded gate
    ρ all read off the same replicate (mirrors analyze.h1_statistics).
    """
    if new_prof["questions"] != ref_prof["questions"]:
        raise AssertionError("question alignment drift between new cell and reference full reads")
    if new_prof["contexts"] != ref_prof["contexts"]:
        raise AssertionError("context panel drift between new cell and reference full reads")
    rng = np.random.default_rng(seed)
    nonsrc_ctxs, nm = _nonsource_rows(new_prof)
    _, rm = _nonsource_rows(ref_prof)
    n_q = nm.shape[1]

    nmeans, rmeans = nm.mean(axis=1), rm.mean(axis=1)
    rho_raw = _spearman(nmeans, rmeans)
    rel_new = split_half_reliability(nm, rng=rng)
    rel_ref = split_half_reliability(rm, rng=rng)
    denom = float(np.sqrt(max(rel_new * rel_ref, 1e-6)))
    rho_disatt = float(np.clip(rho_raw / denom, -1.0, 1.0))

    spread_new = float(nmeans.max() - nmeans.min())
    null_new = range_null_p95(nm, rng=rng)
    spread_valid = spread_new >= SPREAD_MIN_NATS and spread_new > null_new

    g_all = np.array([gate[c]["cosine"][int(GATE_LAYER_KEY) - 1] for c in new_prof["contexts"]])
    g_ns = np.array([gate[c]["cosine"][int(GATE_LAYER_KEY) - 1] for c in nonsrc_ctxs])

    boot_raw = np.empty(n_boot)
    boot_disatt = np.empty(n_boot)
    boot_grand = np.empty(n_boot)
    boot_gate = np.empty(n_boot)
    for i in range(n_boot):
        qi = rng.integers(0, n_q, size=n_q)
        nmi, rmi = nm[:, qi].mean(axis=1), rm[:, qi].mean(axis=1)
        boot_raw[i] = _spearman(nmi, rmi)
        boot_disatt[i] = np.clip(boot_raw[i] / denom, -1.0, 1.0)
        boot_grand[i] = nmi.mean()
        boot_gate[i] = _spearman(g_ns, nmi)

    primary = {
        "reference_run": ref_prof["run_id"],
        "reference_step": ref_prof["step"],
        "new_cell_step": new_prof["step"],
        "n_nonsource_contexts": len(nonsrc_ctxs),
        "n_questions": int(n_q),
        "rho_raw": rho_raw,
        "rho_raw_ci95": _ci(boot_raw),
        "rho_disattenuated": rho_disatt,
        "rho_disattenuated_ci95": _ci(boot_disatt),
        "reliability_new_cell": rel_new,
        "reliability_reference": rel_ref,
        "spread_new_cell_nats": spread_new,
        "range_null_p95_new_cell": null_new,
        "spread_valid_new_cell": bool(spread_valid),
        "rho_10ctx_descriptive": _spearman(
            new_prof["delta"].mean(axis=1), ref_prof["delta"].mean(axis=1)
        ),
    }

    # (c) full 28-layer gate-rho overlay for the new cell (winner's-curse guard:
    # layer 19 is the named read, the layer-robust pattern carries evidence).
    base_prior_all = _base_prior(new_prof["contexts"])
    means_all = new_prof["delta"].mean(axis=1)
    nonsrc_idx = [i for i, c in enumerate(new_prof["contexts"]) if c != SOURCE_CONTEXT]
    per_layer: dict[str, dict] = {}
    for layer in range(N_LAYERS):
        g = np.array([gate[c]["cosine"][layer] for c in new_prof["contexts"]])
        per_layer[str(layer + 1)] = {
            "rho": _spearman(g, means_all),
            "rho_source_excluded": _spearman(g[nonsrc_idx], means_all[nonsrc_idx]),
            "rho_partial_base_prior": _partial_spearman(g, means_all, base_prior_all),
        }
    gate_block = {
        "per_layer": per_layer,
        "named_layer": GATE_LAYER_KEY,
        "layer19": {
            **per_layer[GATE_LAYER_KEY],
            "rho_source_excluded_ci95": _ci(boot_gate),
        },
        "g_layer19_values": {c: float(v) for c, v in zip(new_prof["contexts"], g_all, strict=True)},
    }

    magnitude_new = {
        "nonsource_mean_dg": float(nmeans.mean()),
        "nonsource_mean_dg_ci95": _ci(boot_grand),
    }
    return primary, gate_block, magnitude_new


# ── (b) villain replicate envelope ───────────────────────────────────────


def villain_envelope() -> dict:
    """Non-source mean-ΔG envelope over the villain chains at matched steps."""
    per_chain: dict[str, dict] = {}
    for run_id in VILLAIN_CHAIN_RUNS:
        prof = _load_full_profile(run_id)
        _, nm = _nonsource_rows(prof)
        per_chain[run_id] = {
            "matched_step": prof["step"],
            "nonsource_mean_dg": float(nm.mean(axis=1).mean()),
        }
    vals = [v["nonsource_mean_dg"] for v in per_chain.values()]
    return {
        "per_chain": per_chain,
        "envelope_min": float(min(vals)),
        "envelope_max": float(max(vals)),
    }


# ── (d) equal-step comparison + reference ceilings ───────────────────────


def _rho_nonsource_means(a_ctxs: list[str], a_means: np.ndarray, b_means: np.ndarray) -> float:
    idx = [i for i, c in enumerate(a_ctxs) if c != SOURCE_CONTEXT]
    return _spearman(a_means[idx], b_means[idx])


def equal_step_comparison(new_run: str, ref_run: str) -> dict:
    """Profile ρ at every shared step (full/full preferred, traj/traj else).

    The registered §6(a) read names step 12 (the reference's matched step);
    the step-12 full/full entry is surfaced under ``named_step12`` when the
    new cell has a full read there, else the nearest shared traj grid steps
    stand in (the 8-step traj grid does not include 12).
    """
    full_new, full_ref = _full_step_profiles(new_run), _full_step_profiles(ref_run)
    traj_new, traj_ref = _traj_step_means(new_run), _traj_step_means(ref_run)
    if traj_new["questions"] != traj_ref["questions"]:
        raise AssertionError("traj question alignment drift between new cell and reference")
    out: dict = {"per_step": {}}
    for step in sorted(set(full_new) & set(full_ref)):
        a, b = full_new[step], full_ref[step]
        out["per_step"][str(step)] = {
            "source": "full",
            "rho_nonsource": _rho_nonsource_means(
                a["contexts"], a["delta"].mean(axis=1), b["delta"].mean(axis=1)
            ),
        }
    for step in sorted(set(traj_new["per_step"]) & set(traj_ref["per_step"])):
        if str(step) in out["per_step"]:
            continue  # full/full read preferred at shared steps
        out["per_step"][str(step)] = {
            "source": "traj",
            "rho_nonsource": _rho_nonsource_means(
                traj_new["contexts"], traj_new["per_step"][step], traj_ref["per_step"][step]
            ),
        }
    entry12 = out["per_step"].get("12")
    if entry12 is not None and entry12["source"] == "full":
        out["named_step12"] = entry12
    else:
        out["named_step12"] = None
        out["note"] = (
            "step-12 full/full read unavailable (new cell matched/anchored at a different "
            "step; the traj grid has no step 12) — shared traj grid steps stand in"
        )
    return out


def reference_ceilings(gate: dict) -> dict:
    """Recompute the plan §3b registered ceilings from committed parent JSONs."""
    p8 = {r[-1]: _load_full_profile(r, step=8) for r in VILLAIN_CHAIN_RUNS}
    p12 = {r[-1]: _load_full_profile(r) for r in VILLAIN_CHAIN_RUNS}
    g_idx = int(GATE_LAYER_KEY) - 1
    within = {
        c: _rho_nonsource_means(
            p8[c]["contexts"], p8[c]["delta"].mean(axis=1), p12[c]["delta"].mean(axis=1)
        )
        for c in "ABC"
    }
    cross = {
        f"{a}-{b}": _rho_nonsource_means(
            p12[a]["contexts"], p12[a]["delta"].mean(axis=1), p12[b]["delta"].mean(axis=1)
        )
        for a, b in combinations("ABC", 2)
    }
    gate19 = {}
    for c in "ABC":
        ctxs, nm = _nonsource_rows(p12[c])
        g_ns = np.array([gate[ctx]["cosine"][g_idx] for ctx in ctxs])
        gate19[c] = _spearman(g_ns, nm.mean(axis=1))
    return {
        "recomputed": {
            "within_route_step8_vs_step12": within,
            "cross_chain_matched": cross,
            "gate_rho_layer19_source_excluded_per_chain": gate19,
        },
        "registered_plan_values": REGISTERED_CEILINGS,
    }


# ── (e) secondary descriptive reads ──────────────────────────────────────


def secondary_descriptive(new_prof: dict) -> dict:
    """ρ vs the wrapper control and vs the ICL content-control analogue."""
    out: dict = {}
    _, nm = _nonsource_rows(new_prof)
    nmeans = nm.mean(axis=1)
    try:
        rows_prof = _load_full_profile("ft_ctrl_helpful_rows")
        if rows_prof["contexts"] != new_prof["contexts"]:
            raise AssertionError("context panel drift vs ft_ctrl_helpful_rows")
        _, rm = _nonsource_rows(rows_prof)
        out["rho_vs_ft_ctrl_helpful_rows"] = {
            "rho_nonsource": _spearman(nmeans, rm.mean(axis=1)),
            "wrapper_control_step": rows_prof["step"],
        }
    except FileNotFoundError as e:
        out["rho_vs_ft_ctrl_helpful_rows"] = {"skipped": str(e)}
    try:
        icl_prof = _icl_panel_profile("icl_ctrl_helpful_marker")
        if icl_prof["questions"] != new_prof["questions"]:
            raise AssertionError("question alignment drift vs icl_ctrl_helpful_marker")
        if icl_prof["contexts"] != new_prof["contexts"]:
            raise AssertionError("context panel drift vs icl_ctrl_helpful_marker")
        _, im = _nonsource_rows(icl_prof)
        out["rho_vs_icl_ctrl_helpful_marker"] = {
            "rho_nonsource": _spearman(nmeans, im.mean(axis=1))
        }
    except FileNotFoundError as e:
        out["rho_vs_icl_ctrl_helpful_marker"] = {"skipped": str(e)}
    return out


# ── §3b verdict fields (mechanical; the analyzer interprets) ─────────────


def compute_verdict(primary: dict, gate_block: dict, magnitude: dict, matched: dict) -> dict:
    """The plan §3b decision table, computed mechanically with no narration."""
    gate_rho = gate_block["layer19"]["rho_source_excluded"]
    rank_condition = bool(
        primary["rho_disattenuated"] >= DISATT_RHO_CONFIRM
        and primary["rho_raw"] >= RAW_RHO_CONFIRM
        and primary["spread_valid_new_cell"]
        and gate_rho >= GATE_RHO_CONFIRM
    )
    magnitude_condition = bool(magnitude["within_envelope"])
    ci_hi = primary["rho_disattenuated_ci95"][1]
    departure_condition = bool(
        (
            (primary["rho_disattenuated"] <= DISATT_RHO_FALSIFY and ci_hi < DISATT_RHO_CONFIRM)
            or gate_rho <= GATE_RHO_FALSIFY
        )
        and primary["spread_valid_new_cell"]
    )
    joint_cut_flag = bool(rank_condition and magnitude_condition)
    if not matched["within_tolerance"]:
        label = (
            "band-entry-fallback — matched dose outside ±1.5 nat; §3b verdict table NOT "
            "run (plan §7); closest-approach read reported descriptively"
        )
    elif joint_cut_flag:
        label = "row1-cut-condition-met — rank AND magnitude content-blind (plan §3b row 1)"
    elif rank_condition:
        label = "rank-blind-magnitude-departs — RE-SCOPE, not cut (plan §3b row 1 carve-out)"
    elif departure_condition:
        label = "content-departure — the write carries content (plan §3b row 2)"
    else:
        label = "intermediate-graded — reported descriptively with CIs (plan §3b row 3)"
    return {
        "rank_condition": rank_condition,
        "magnitude_condition": magnitude_condition,
        "joint_cut_flag": joint_cut_flag,
        "departure_condition": departure_condition,
        "verdict_table_run": bool(matched["within_tolerance"]),
        "label": label,
        "thresholds": {
            "rank_disatt_rho_min": DISATT_RHO_CONFIRM,
            "rank_raw_rho_min": RAW_RHO_CONFIRM,
            "rank_gate_rho_min": GATE_RHO_CONFIRM,
            "falsify_disatt_rho_max": DISATT_RHO_FALSIFY,
            "falsify_gate_rho_max": GATE_RHO_FALSIFY,
            "spread_min_nats": SPREAD_MIN_NATS,
        },
    }


# ── Entry point ──────────────────────────────────────────────────────────


def run_followup_analysis(
    *, new_run: str = FOLLOWUP_RUN, n_boot: int = N_BOOT, seed: int = SEED, stub: bool = False
) -> Path:
    """Assemble every registered follow-up read into one JSON artifact."""
    from explore_persona_space.experiments.icl_vs_ft_491.matching import load_matched_entry

    gate = _ensure_gate()
    matched = load_matched_entry(new_run, smoke=False)
    new_prof = _load_full_profile(new_run)
    ref_prof = _load_full_profile(VILLAIN_REF_RUN)

    primary, gate_block, magnitude_new = primary_and_gate_block(
        new_prof, ref_prof, gate, n_boot=n_boot, seed=seed
    )
    envelope = villain_envelope()
    magnitude = {
        **magnitude_new,
        "villain_replicate_envelope": envelope,
        "within_envelope": bool(
            envelope["envelope_min"]
            <= magnitude_new["nonsource_mean_dg"]
            <= envelope["envelope_max"]
        ),
    }
    base_prior_ns = _base_prior([c for c in new_prof["contexts"] if c != SOURCE_CONTEXT])
    _, nm = _nonsource_rows(new_prof)
    _, rm = _nonsource_rows(ref_prof)
    analysis = {
        "meta": repro_metadata(),
        "followup_label": FOLLOWUP_LABEL,
        "smoke_stub_run": new_run if stub else None,
        "new_cell": {
            "run_id": new_run,
            "matched": {
                k: matched[k]
                for k in (
                    "matched_step",
                    "anchor_step",
                    "basis",
                    "residual",
                    "within_tolerance",
                    "band_entered",
                    "dose_logp",
                    "icl_dose_variant",
                    "ceiling_flagged",
                )
            },
        },
        "primary_profile": primary,
        "magnitude": magnitude,
        "gate": gate_block,
        "equal_step": equal_step_comparison(new_run, VILLAIN_REF_RUN),
        "base_prior_partial_profile_rho": _partial_spearman(
            nm.mean(axis=1), rm.mean(axis=1), base_prior_ns
        ),
        "reference_ceilings": reference_ceilings(gate),
        "secondary": secondary_descriptive(new_prof),
        "n_boot": n_boot,
        "seed": seed,
    }
    analysis["verdict"] = compute_verdict(primary, gate_block, magnitude, matched)
    name = "followup_analysis_smoke_stub.json" if stub else "followup_analysis.json"
    out = OUT_DIR / name
    write_json(out, analysis)
    return out


def _stub_smoke_asserts(payload: dict) -> None:
    """Known-structure checks for the villain-replicate stub (CPU smoke).

    A villain replicate IS content-matched to the reference by construction,
    so the rank + magnitude conditions must PASS, and the recomputed
    ceilings must reproduce the plan §3b registered values to 1e-3.
    """
    v = payload["verdict"]
    assert v["rank_condition"], v
    assert v["magnitude_condition"], v
    assert v["joint_cut_flag"], v
    assert not v["departure_condition"], v
    assert payload["primary_profile"]["spread_valid_new_cell"], payload["primary_profile"]
    lo, hi = payload["primary_profile"]["rho_raw_ci95"]
    assert lo <= payload["primary_profile"]["rho_raw"] <= hi, payload["primary_profile"]
    rec = payload["reference_ceilings"]["recomputed"]
    reg = payload["reference_ceilings"]["registered_plan_values"]
    for block in reg:
        for key, regval in reg[block].items():
            recval = rec[block][key]
            assert abs(recval - regval) <= 1e-3, (block, key, recval, regval)
    g19 = payload["gate"]["layer19"]
    glo, ghi = g19["rho_source_excluded_ci95"]
    assert glo <= g19["rho_source_excluded"] <= ghi, g19
    assert payload["equal_step"]["per_step"], payload["equal_step"]
    print(json.dumps({"stub_smoke": "PASS", "verdict": v["label"]}, indent=2))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--stub-new-cell-from",
        default=None,
        help="CPU smoke: run the identical code path treating this committed parent run "
        "as the new cell; writes followup_analysis_smoke_stub.json (never the real artifact)",
    )
    args = ap.parse_args(argv)
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    if args.stub_new_cell_from:
        out = run_followup_analysis(
            new_run=args.stub_new_cell_from, n_boot=args.n_boot, seed=args.seed, stub=True
        )
        _stub_smoke_asserts(json.loads(out.read_text()))
    else:
        out = run_followup_analysis(n_boot=args.n_boot, seed=args.seed)
    logger.info("follow-up analysis written: %s", out)


if __name__ == "__main__":
    main()
