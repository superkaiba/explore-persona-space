#!/usr/bin/env python3
"""Task #536 Phase A — two-pass audit sweep of persona-distance cosine sites.

Pass (i): literal-API regex. Pass (ii): hand-rolled-idiom regex (high-recall /
low-precision by design — completeness is defined by the expanded sweep, plan
§3 Ask 2). Every swept file must be accounted for by either a curated SITE row
(a persona-distance cosine COMPUTATION site, classified centered | raw |
both-computed | N-A with file:line evidence + consuming tasks) or a curated
DISPOSITION (consumer of an upstream artifact / different construct / infra).
Unaccounted files FAIL the run loudly.

Also records: (a) artifact fingerprint checks (the #405-line lesson — classify
by what the downstream consumer actually READ, off-diagonal range raw-regime
[~0.7, 1.0] vs centered spanning negatives; #406 fingerprints are per-layer);
(b) deleted producers recovered via git history (#406's
i406_phase1_merge_and_compute_matrices.py at 9e6e31c3f, #478's
_issue478_common.py at 69b34b94); (c) the Phase A0 Persona Vectors
difference-of-means record (fact-check confirmed via the arxiv-latex MCP).

Usage::

    uv run python scripts/issue536_audit.py \
        --data-root /home/thomasjiralerspong/explore-persona-space

Output: eval_results/issue_536/audit_table.json (CPU-only, deterministic).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

REPO = Path(__file__).resolve().parent.parent
log = logging.getLogger("i536.audit")

OUT_PATH = REPO / "eval_results" / "issue_536" / "audit_table.json"

PASS1 = re.compile(r"compute_cosine_matrix|cosine_similarity|F\.cosine|cos_sim|centering=")
PASS2 = re.compile(
    r"F\.normalize|np\.linalg\.norm|torch\.linalg\.norm|\.norm\(|np\.dot|torch\.dot"
    r"|einsum|@\s*\S*\.T|matmul|cosine_matrix|_cosine|cos_"
)
SWEEP_ROOTS = (
    "scripts",
    "experiments",
    "src/explore_persona_space/analysis",
    "src/explore_persona_space/axis",
    "src/explore_persona_space/experiments",
)


def _git_sha(cwd: Path = REPO) -> str:
    """HEAD commit of the checkout at ``cwd`` (reproducibility metadata).

    Recorded twice in the table: ``code_commit`` (the tree this script runs
    from) and ``data_root_commit`` (the checkout being swept/fingerprinted) —
    they differ when the audit sweeps the main checkout from a worktree.
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ},
        ).stdout.strip()
    except Exception as e:  # pragma: no cover — metadata only, never silent
        log.warning("git rev-parse failed: %s", e)
        return "unknown"


def run_sweep(sweep_root: Path) -> dict[str, dict]:
    """Run both regex passes; return {relpath: {pass1: bool, pass2: bool, lines: [...]}}."""
    hits: dict[str, dict] = {}
    for root in SWEEP_ROOTS:
        for f in sorted((sweep_root / root).rglob("*.py")):
            if "__pycache__" in str(f):
                continue
            text = f.read_text(errors="replace")
            p1, p2 = bool(PASS1.search(text)), bool(PASS2.search(text))
            if not (p1 or p2):
                continue
            rel = str(f.relative_to(sweep_root))
            # Matched-idiom evidence: collect PASS1 OR PASS2 hits, tagging which
            # pass matched each line (round-1 review fix: the PASS1-only loop
            # left 101/102 pass2-only dispositions with empty evidence_lines).
            ev = []
            for i, line in enumerate(text.splitlines(), start=1):
                if line.strip().startswith("#"):
                    continue
                if PASS1.search(line):
                    tag = "pass1"
                elif PASS2.search(line):
                    tag = "pass2"
                else:
                    continue
                ev.append(f"{i} [{tag}]: {line.strip()[:140]}")
                if len(ev) >= 6:
                    break
            hits[rel] = {"pass1": p1, "pass2": p2, "evidence_lines": ev}
    return hits


def off_diag_stats(M: np.ndarray) -> dict:
    """Off-diagonal min/median/max + the regime call (raw band vs centered span)."""
    v = M[~np.eye(M.shape[0], dtype=bool)]
    stats = {
        "min": float(v.min()),
        "median": float(np.median(v)),
        "max": float(v.max()),
    }
    stats["regime_call"] = (
        "raw band (compressed)" if stats["min"] > 0.25 else "centered span (negatives reached)"
    )
    return stats


def fingerprint_checks(data_root: Path) -> list[dict]:
    """Load the load-bearing persisted artifacts and record their regime fingerprints."""
    checks: list[dict] = []

    p = data_root / "eval_results" / "extraction_method_comparison" / "cosine_matrix_a_layer20.json"
    d = json.loads(p.read_text())
    checks.append(
        {
            "artifact": str(p.relative_to(data_root)),
            "consumed_by": "#405 (issue405_clean_result_analysis.py:46-50)",
            "kind": "similarity matrix (20 personas, L20)",
            **off_diag_stats(np.asarray(d["matrix"], dtype=np.float64)),
            "verdict": "RAW — the consumer read the raw matrix even though "
            "compare_extraction_methods.py computed both",
        }
    )

    p = (
        data_root
        / "eval_results"
        / "single_token_100_persona"
        / "cosine_distance_matrix_layer20.json"
    )
    d = json.loads(p.read_text())
    sim = 1.0 - np.asarray(d["matrix"], dtype=np.float64)
    checks.append(
        {
            "artifact": str(p.relative_to(data_root)),
            "consumed_by": "#478/#490 (scripts/_issue478_common.py@69b34b94 loader)",
            "kind": "distance matrix (111 personas, L20; metric '1 - cosine') read as similarity",
            **off_diag_stats(sim),
            "verdict": "RAW — built normalize-only (no centering) from centroids_layer20.pt",
        }
    )

    for layer in (0, 5, 11, 15, 21, 27):
        p = data_root / "eval_results" / "issue_406" / "cosine" / f"C_L{layer}.json"
        d = json.loads(p.read_text())
        conds = d["conditions"]
        D = np.array([[float(d["matrix"][a][b]) for b in conds] for a in conds])
        checks.append(
            {
                "artifact": str(p.relative_to(data_root)),
                "consumed_by": "#406/#460/#474 lineage (i474_cosine_followup.py:52)",
                "kind": f"DISTANCE-form 1-cos (16 conditions, L{layer}); "
                "converted G = 1 - D before reading",
                **off_diag_stats(1.0 - D),
                "verdict": "RAW (per-layer fingerprint; producer normalize-only, deleted)",
            }
        )

    p = REPO / "experiments" / "phase_minus1_persona_vectors" / "cosine_matrix.json"
    d = json.loads(p.read_text())
    for lk in sorted(d):
        M = np.asarray(d[lk]["matrix"], dtype=np.float64)
        checks.append(
            {
                "artifact": f"experiments/phase_minus1_persona_vectors/cosine_matrix.json::{lk}",
                "consumed_by": "#341 cos<->JS alignment",
                "kind": "similarity matrix (20 personas)",
                **off_diag_stats(M),
                "verdict": "RAW — extract_persona_vectors.py:198-199 normalize-only "
                "(#341 reclassified to the raw line at fact-check)",
            }
        )

    p = data_root / "eval_results" / "issue_213" / "part_a" / "cosine_matrices.json"
    d = json.loads(p.read_text())
    checks.append(
        {
            "artifact": str(p.relative_to(data_root)),
            "consumed_by": "#213/#227",
            "kind": "cue->no_cue reference-COLUMN distance file "
            f"(models={d.get('models')}, cues={d.get('cues')}) — NOT a Gram matrix",
            "verdict": "RAW producer; approximate Gram read IMPOSSIBLE (no Gram persists) — "
            "needs-GPU-re-extraction partition",
        }
    )

    p = data_root / "eval_results" / "issue_505" / "analysis" / "panel_similarity_matrix.json"
    d = json.loads(p.read_text())
    vals = np.array([v for b in d["L21"]["cos_b_j"].values() for v in b.values()], dtype=np.float64)
    checks.append(
        {
            "artifact": str(p.relative_to(data_root)),
            "consumed_by": "#505 per-arm slopes",
            "kind": "cos(b, j) pairs at L21 (persona-vectors centroids)",
            "min": float(vals.min()),
            "median": float(np.median(vals)),
            "max": float(vals.max()),
            "regime_call": "raw band (compressed)" if vals.min() > 0.25 else "centered span",
            "verdict": "RAW — build_pv_centroids.py:196 centering='none' explicit",
        }
    )

    p = data_root / "eval_results" / "issue_311" / "centroids_base.pt"
    if p.exists():
        import torch

        b = torch.load(p, map_location="cpu", weights_only=False)
        checks.append(
            {
                "artifact": str(p.relative_to(data_root)),
                "consumed_by": "#311",
                "kind": f"bundle keys = {sorted(b.keys())}",
                "verdict": "CENTERED line — bundle persists BOTH centroids_raw and "
                "centroids_centered (methodology was centered)",
            }
        )

    p = data_root / "eval_results" / "issue_527" / "pair_selection.json"
    if p.exists():
        d = json.loads(p.read_text())
        prov = d.get("centering") or d.get("metadata", {}).get("centering") or "see payload"
        checks.append(
            {
                "artifact": str(p.relative_to(data_root)),
                "consumed_by": "#527/#550 (issue550_slope_distance_correlation.py)",
                "kind": "pair-selection payload carrying the L20 cos matrix",
                "centering_provenance": str(prov)[:200],
                "verdict": "CENTERED — provenance recorded in the payload",
            }
        )
    return checks


def S(file, lines, cls, evidence, tasks, notes=""):
    """Compact SITE row constructor."""
    return {
        "file": file,
        "lines": lines,
        "classification": cls,
        "evidence": evidence,
        "tasks": tasks,
        "notes": notes,
    }


# Curated COMPUTATION sites (classification: centered | raw | both-computed | N-A).
SITES: list[dict] = [
    # ── shared library ──────────────────────────────────────────────────
    S(
        "src/explore_persona_space/analysis/representation_shift.py",
        "139-159",
        "centered (default)",
        "compute_cosine_matrix(centroids, centering='global_mean') default; "
        "center -> F.normalize -> C@C.T",
        ["shared lib"],
        "The library default IS the canonical recipe.",
    ),
    # ── centered line ───────────────────────────────────────────────────
    S(
        "scripts/analyze_100_persona_cosine.py",
        "288-293",
        "centered",
        "C = C - C.mean(dim=0, keepdim=True); F.normalize; C@C.T",
        [66, 99],
    ),
    S(
        "scripts/analyze_100_persona_source_filtered.py",
        "51-54",
        "centered",
        "hand-rolled: c - c.mean(axis=0); c / norm; c@c.T (reconciler-named literal-pass miss)",
        [66, 99],
    ),
    S(
        "scripts/plot_100_persona_scatter_simple.py",
        "35-38",
        "centered",
        "hand-rolled: c - c.mean(axis=0); c / norm; c@c.T (reconciler-named literal-pass miss)",
        [66],
    ),
    S(
        "scripts/run_leakage_v3.py",
        "405-423",
        "centered",
        "base_centered = base_c - base_c.mean(dim=0); cos_sim on centered rows",
        [247, 329],
    ),
    S(
        "scripts/i380_cosine_pairwise.py",
        "48-52",
        "centered",
        "mat_centered = mat - mat.mean(dim=0); normalized matmul",
        [380],
    ),
    S(
        "scripts/eval_causal_ckpt.py",
        "186",
        "centered",
        "compute_cosine_matrix(centroids[layer], centering='global_mean') explicit",
        [61, 91],
        "Causal-ckpt line; #61 body quotes negative cosine range (centered fingerprint).",
    ),
    S(
        "scripts/issue550_slope_distance_correlation.py",
        "295-310",
        "centered (consumer with provenance)",
        "reads eval_results/issue_527/pair_selection.json which records centering='global_mean'",
        [550, 527],
    ),
    S(
        "src/explore_persona_space/experiments/contrastive_neg_geometry_504/phase05.py",
        "89-119, call sites 175/314",
        "both-computed (centered default, raw escape hatch)",
        "_cos_matrix_from_centroids(mean_center=True default; False recovers r1-5 raw) "
        "(reconciler-named literal-pass miss)",
        [504],
        "#504 round-7 fix threads mean_center to call sites.",
    ),
    S(
        "scripts/i504_phase_phase05.py",
        "109-139",
        "both-computed (centered default, raw escape hatch)",
        "script mirror of the phase05 helper: mat - mat.mean(axis=0) when mean_center",
        [504],
    ),
    S(
        "scripts/compute_zelthari_centered_cosine.py",
        "58-72, 306-320",
        "both-computed (diagnostic)",
        "raw_cos AND centered_cos computed side by side",
        ["persona mining (zelthari)"],
    ),
    S(
        "scripts/i504_round6_recompute_mean_centered.py",
        "137-138",
        "both-computed (remediation tool)",
        "cos_raw = compute_cosine_matrix(C, 'none'); cos_mc = compute_cosine_matrix(C, "
        "'global_mean')",
        [504, 472],
        "The r6 remediation that backfilled cos_matrix_mean_centered.",
    ),
    S(
        "src/explore_persona_space/experiments/contrastive_neg_geometry_472/centroids.py",
        "module",
        "both-computed (raw rounds 1-5; mc added post-r6)",
        "bundles carry cos_matrix + cos_matrix_mean_centered since #504 r6",
        [472, 504, 530, 538, 550],
    ),
    S(
        "scripts/analyze_leakage.py",
        "85-122",
        "centered (consumer)",
        "CENTERED_COSINES constants + zelthari_centered_cosine.json (centered_cosine_to_assistant)",
        ["early leakage line"],
    ),
    # ── raw line ────────────────────────────────────────────────────────
    S(
        "scripts/compare_extraction_methods.py",
        "~412 (cross-method), ~431 (raw matrix), ~459 (centered matrix)",
        "both-computed — CONSUMER READ RAW",
        "cosine_matrix_a_layer20.json off-diag [0.7322, 0.9971] median 0.9450 (raw "
        "fingerprint, no centering key); the centered twin was computed but not consumed",
        [405, 478, 490],
        "#405's distance source; #478/#490 use the 111-bank file (also raw).",
    ),
    S(
        "scripts/run_issue_213_part_a.py",
        "547",
        "raw",
        "pairwise cosine on centroids, no centering",
        [213, 227],
    ),
    S(
        "scripts/recompute_predictors_i415.py",
        "242-245",
        "raw",
        "F.cosine_similarity(v, asst_centroid) / (v, neut_centroid) — no bank centering",
        [396, 415],
        "The published predictor nulls were computed in the compressed geometry.",
    ),
    S(
        "src/explore_persona_space/experiments/leave_one_out_505/build_pv_centroids.py",
        "196",
        "raw (explicit)",
        "compute_cosine_matrix(c, centering='none')",
        [505],
    ),
    S(
        "experiments/phase_minus1_persona_vectors/extract_persona_vectors.py",
        "188-199",
        "raw",
        "centroid mean -> F.normalize -> C@C.T (normalize-only, NO centering); "
        "fingerprint identical to the degenerate raw band",
        [341],
        "#341's cos<->JS alignment rho=0.94 is therefore a raw-line stat (fact-check).",
    ),
    S(
        "scripts/extract_prompt_divergence_activations.py",
        "704",
        "raw",
        "raw cosine; method-A-vs-B sanity check, NOT the #406 producer",
        [406],
    ),
    S(
        "scripts/issue404_predictor_cossim.py",
        "136-148",
        "raw 2-vector pairwise (no bank — centering N/A)",
        "F.cosine_similarity(a, b, dim=-1).mean() across probes",
        [404, 458],
        "Pin ¶2 family: labeled raw pairwise, never compared to bank-cosine.",
    ),
    S(
        "scripts/issue444_persona_distance_topic.py",
        "101-110",
        "raw 2-vector pairwise (no bank — centering N/A)",
        "F.cosine_similarity(ref[li], acts[other][li], dim=1).mean()",
        [444],
    ),
    S(
        "scripts/issue493_extraction_metric_bakeoff.py",
        "pairwise predictor arm",
        "raw 2-vector pairwise (no bank — centering N/A)",
        "one predictor among many in the bake-off",
        [493],
    ),
    S(
        "scripts/issue502_cpu_smoke.py",
        "batched mirror",
        "raw 2-vector pairwise (no bank — centering N/A)",
        "batched re-implementation of the #493 serial path",
        [502],
    ),
    S(
        "scripts/run_trait_transfer.py",
        "314-323",
        "raw (ad-hoc local fn)",
        "compute_cosine_matrix local def: pairwise cos, no centering",
        ["pre-task era"],
    ),
    S(
        "scripts/run_persona_leakage_v2.py",
        "482-492",
        "raw (pairwise to target)",
        "F.cosine_similarity(target_vec, p_vec)",
        ["early era"],
    ),
    S(
        "scripts/run_proximity_transfer.py",
        "local fn",
        "raw (ad-hoc local fn)",
        "pairwise cosine, no centering",
        ["early era"],
    ),
    S(
        "scripts/extract_centroids_and_analyze.py",
        "328-345",
        "raw (pairwise)",
        "F.cosine_similarity(v_p, v_t) raw + assistant variants",
        ["early era"],
    ),
    S(
        "scripts/run_issue_276_pre_poison_similarity.py",
        "144",
        "raw (pairwise to canonical)",
        "F.cosine_similarity(h, canon_h)",
        [276],
    ),
    S(
        "experiments/directed_trait_transfer/run_experiment.py",
        "382",
        "raw (pairwise)",
        "F.cosine_similarity(vecs[n1], vecs[n2])",
        ["pre-task era"],
    ),
    S(
        "scripts/i504_smoke_local.py",
        "109",
        "raw (smoke mirror of the #472 rig)",
        "cos matrix with centering='none' (pre-r6 behavior, smoke only)",
        [504],
    ),
    # ── N-A (different construct) ───────────────────────────────────────
    S(
        "scripts/analyze_em_axis.py",
        "axis-alignment cosine",
        "N-A",
        "cosine over EM-axis directions, not persona-distance",
        [],
    ),
    S(
        "src/explore_persona_space/experiments/leave_one_out_505/logit_rescoring.py",
        "554",
        "N-A",
        "cosine_similarity as a serialization sanity check (batched vs serial rows)",
        [505],
    ),
]

# Deleted producers (recovered via git history; plan §4-A step 2).
DELETED_SITES: list[dict] = [
    S(
        "scripts/i406_phase1_merge_and_compute_matrices.py",
        "300-345 @ commit 9e6e31c3f (deleted from working tree)",
        "raw (persisted as DISTANCE, 1-cos)",
        "per-layer 20x20 cosine-distance matrices from per-context mean activations; "
        "normalize-only (git show 9e6e31c3f verified at implementation time)",
        [406, 460, 474],
        "Producer of eval_results/issue_406/cosine/C_L*.json.",
    ),
    S(
        "scripts/_issue478_common.py",
        "318-360 @ commit 69b34b94 (deleted from working tree)",
        "raw",
        "_build_matrix_from_centroids: L2-normalize per row -> pairwise sim -> "
        "distance = 1 - sim (NO centering); cached to "
        "cosine_distance_matrix_layer20.json",
        [478, 490],
        "Producer/loader of the #478/#490 111-bank distance source.",
    ),
]


# Two more curated computation sites surfaced while writing dispositions.
SITES += [
    S(
        "scripts/i488_phase1_predictors.py",
        "111+ (COSINE_LAYERS sweep)",
        "raw 2-vector pairwise (no bank — centering N/A)",
        "residual-stream cosine T_i-vs-T_j, mean over probes (the #404-family recipe)",
        [488],
    ),
    S(
        "scripts/i504_probe_bank_geometry.py",
        "module (Gate-A probe)",
        "raw (deliberate diagnostic)",
        "cos(persona, villain) per layer — the probe that DOCUMENTED the raw band "
        "degeneracy (Gate A: all candidates in [0.93, 0.96])",
        [504],
    ),
    # Post-pin site that landed on main after the round-1 sweep (coverage gate
    # caught it on the round-2 re-run; classified, gate NOT weakened).
    S(
        "scripts/issue560_crossrecipe_panel.py",
        "680-738 (_cosine_distance + geometry phase)",
        "raw (hand-rolled pairwise, no centering)",
        "1 - a@b/(|a||b|) between L20 last-prompt-token centroids "
        "(context<->persona min_dist axis); no bank centering",
        [560],
        "Post-pin script (landed 2026-06-10, after this audit's round 1). A 51-vector "
        "bank (16 contexts + 35 personas) exists, so the pin's bank-cosine family "
        "applies; flagged for #560's analyzer — its min_dist axis is on the raw path.",
    ),
]

# Every other swept file: disposition (consumer / different construct / infra).
# The runtime sweep stores each file's matched-idiom lines as evidence next to
# these reasons; classification of the COMPUTATION lives at the producer's row.
_C_RAW = "consumer — reads a RAW-line persisted artifact; see the producer's site row"
_C_CEN = "consumer — reads a CENTERED-line persisted artifact; see the producer's site row"
_C_509 = (
    "consumer — predictor payloads from the #509/#532 bake-off line, which carry "
    "centering as an EXPLICIT factor (raw + centered variants computed upstream)"
)
_NA_AXIS = "N-A — axis/projection construct (not persona-distance cosine)"
_NA_EMB = "N-A — semantic TEXT-embedding cosine over questions (not persona-distance)"
_NA_DIR = (
    "N-A — direction/shift-vector cosine (SVD components, shift alignment), not persona-distance"
)
_NA_INFRA = (
    "N-A — matched idiom is model/training internals (norms, matmuls), no "
    "persona-distance cosine computed"
)
_NA_JS = "N-A — JS/KL divergence machinery (cosine idiom hits are incidental)"

DISPOSITIONS: dict[str, str] = {
    # consumers of raw-line artifacts
    "scripts/i460_phase5_analyze.py": _C_RAW + " (#406 C_L*.json)",
    "scripts/i474_cosine_followup.py": _C_RAW + " (#406 C_L*.json)",
    "scripts/issue405_clean_result_analysis.py": _C_RAW + " (cosine_matrix_a_layer20.json)",
    "scripts/analyze_single_token_sweep.py": _C_RAW + " (extraction_method_comparison matrices)",
    "scripts/plot_issue_213_final.py": _C_RAW + " (#213 part-A geometry)",
    "scripts/plot_issue_213_geometry_predicts.py": _C_RAW + " (#213 part-A geometry)",
    "scripts/analyze_issue415.py": _C_RAW + " (#415 predictor JSONs)",
    "scripts/issue458_regress.py": _C_RAW + " (#404/#458 pairwise predictor values)",
    "scripts/plot_issue444_bystander.py": _C_RAW + " (#444 pairwise values)",
    "scripts/issue502_plot_best3_bars.py": _C_RAW + " (#493/#502 bake-off outputs)",
    "scripts/issue502_plot_best4_bars.py": _C_RAW + " (#493/#502 bake-off outputs)",
    "scripts/issue505_expanded_predictors.py": _C_RAW + " (#505 PV geometry bundles)",
    "scripts/issue505_panel_coverage.py": _C_RAW + " (#505 PV geometry bundles)",
    "scripts/issue505_r2_figure.py": _C_RAW + " (#505 analysis outputs)",
    "scripts/smoke_phase_d_logit_rescoring.py": _C_RAW + " (#505 rig smoke)",
    "src/explore_persona_space/experiments/leave_one_out_505/analyze.py": _C_RAW
    + " (panel_similarity_matrix from build_pv_centroids)",
    "src/explore_persona_space/experiments/leave_one_out_505/analyze_expanded.py": _C_RAW
    + " (#505 geometry)",
    "src/explore_persona_space/experiments/leave_one_out_505/analyze_logit_rescoring.py": _C_RAW
    + " (#505 geometry)",
    "src/explore_persona_space/experiments/leave_one_out_505/dispatch.py": _C_RAW
    + " (#505 geometry; dispatcher)",
    "src/explore_persona_space/experiments/leave_one_out_505/panel_coverage.py": _C_RAW
    + " (#505 L10 fallback cos)",
    "scripts/plot_leakage_vs_cosine_all.py": _C_RAW + " (early-era pairwise outputs)",
    "scripts/plot_leakage_vs_cosine_none.py": _C_RAW + " (early-era pairwise outputs)",
    "scripts/plot_proximity_transfer.py": _C_RAW + " (run_proximity_transfer outputs)",
    "scripts/plot_trait_transfer.py": _C_RAW + " (run_trait_transfer outputs)",
    "scripts/plot_i432_rank_by_distance.py": _C_RAW + " (#432-era distance artifacts)",
    "scripts/plot_cosine_attenuation.py": _C_RAW + " (predictor-line outputs)",
    "scripts/i501_make_figures_blog.py": _C_RAW + " (cos_sim_per_layer payloads)",
    "scripts/plot_i461_predictor_grid.py": _C_RAW + " (#461 predictor outputs)",
    "scripts/plot_i461_predictor_scatters.py": _C_RAW + " (#461 predictor outputs)",
    "scripts/plot_issue503_v2.py": _C_RAW + " (#503 predictor outputs)",
    "scripts/plot_issue500_predictors.py": _C_RAW + " (#500 predictor outputs)",
    "scripts/issue500_interaction_check.py": _C_RAW + " (#500 predictor outputs)",
    "scripts/analyze_causal_proximity.py": _C_CEN
    + " (causal-proximity line; reads cos_* keys from persisted layer payloads)",
    # consumers of centered-line artifacts
    "scripts/i380_pairwise_scatters.py": _C_CEN + " (#380 correlation.json)",
    "scripts/issue527_dan_rank1_scalar_regression.py": _C_CEN
    + " (#527 pair_selection.json, centering='global_mean' provenance)",
    "scripts/issue550_make_figures.py": _C_CEN + " (#550 analysis outputs)",
    "scripts/plot_strong_convergence.py": _C_CEN + " (run_leakage_v3 centered outputs)",
    "scripts/plot_issue237_tldr.py": _C_CEN + " (#237 geometry outputs)",
    "scripts/plot_issue_89_hero.py": _C_CEN + " (early centered-line outputs)",
    "scripts/plot_issue_157_stage_b_hero_v2.py": _C_CEN + " (stage-B line outputs)",
    "scripts/make_363_figure.py": _C_CEN + " (#363 outputs)",
    "scripts/analyze_length_rate_296.py": _C_CEN
    + " (body-published L15 cosines from #294; comparison only)",
    "scripts/plot_length_rate_correlation.py": _C_CEN + " (#294/#296 outputs)",
    "src/explore_persona_space/experiments/contrastive_neg_geometry_472/analyze.py": _C_CEN
    + " (post-r6 bundles carry both matrices; analysis reads the bundle keys)",
    "src/explore_persona_space/experiments/contrastive_neg_geometry_472/select_negatives.py": (
        _C_CEN + " (distances from centroids.py bundles)"
    ),
    "src/explore_persona_space/experiments/contrastive_neg_geometry_472/build_training_data.py": (
        _C_CEN + " (selection metadata only)"
    ),
    "src/explore_persona_space/experiments/contrastive_neg_geometry_472/eval_guard.py": _NA_INFRA,
    # #509/#532 bake-off line (centering is an explicit factor upstream)
    "scripts/issue509_baserate_covariate_earlylayer.py": _C_509,
    "scripts/issue511_make_figures.py": _C_509,
    "scripts/issue532_followup_logp_slot.py": _C_509,
    "scripts/issue532_predictor_stress.py": _C_509,
    "scripts/issue538_make_figures.py": _C_509,
    "scripts/issue539_residual_per_cohort.py": _C_509,
    "scripts/issue540_jsrb_predictor.py": _C_509,
    "scripts/issue540_length_nuisance_figure.py": _C_509,
    "scripts/issue540_length_nuisance_supplement.py": _C_509,
    "scripts/issue548_length_analysis.py": _C_509,
    "scripts/issue553_panel.py": _C_509,
    # different constructs
    "scripts/analyze_manifold_axes.py": _NA_AXIS,
    "scripts/analyze_outliers_pertoken.py": _NA_AXIS,
    "src/explore_persona_space/axis/analyze.py": _NA_AXIS,
    "src/explore_persona_space/axis/project.py": _NA_AXIS,
    "src/explore_persona_space/analysis/probes.py": _NA_AXIS,
    "scripts/project_categories_instruct.py": _NA_AXIS,
    "scripts/project_categories_onto_axis.py": _NA_AXIS,
    "scripts/project_corpus_fast.py": _NA_AXIS,
    "scripts/project_corpus_single_gpu.py": _NA_AXIS,
    "scripts/project_corpus_v2.py": _NA_AXIS,
    "scripts/track_axis_during_cot.py": _NA_AXIS,
    "scripts/test_activation_steering.py": _NA_AXIS,
    "experiments/persona_geometry_dimensionality/run_dimensionality.py": _NA_AXIS,
    "scripts/analyze_i181.py": _NA_EMB,
    "scripts/build_i181_data.py": _NA_EMB,
    "scripts/i207_run_regression.py": _NA_EMB
    + " (the #207 GEOMETRY claim consumed #66-line centered cosine; this script's "
    "compute_semantic_cosine is a question-text embedding feature)",
    "scripts/precheck_i181_axes.py": _NA_EMB,
    "src/explore_persona_space/analysis/i181_features.py": _NA_EMB,
    "scripts/issue552_cross_arm_analysis.py": _NA_DIR,
    "scripts/issue552_figures.py": _NA_DIR,
    "scripts/issue552_mean_resp_svd.py": _NA_DIR,
    "scripts/issue552_write_sentinel.py": _NA_INFRA,
    "src/explore_persona_space/experiments/contrastive_neg_geometry_504/shadow_angle.py": _NA_DIR,
    "src/explore_persona_space/experiments/contrastive_neg_geometry_504/cell_resolution.py": (
        _NA_INFRA
    ),
    "src/explore_persona_space/analysis/divergence.py": _NA_JS,
    "scripts/run_issue_276_teacher_forced_js.py": _NA_JS,
    "scripts/run_issue_276_continuation_sweep.py": _NA_JS,
    # marker/infra/training internals
    "scripts/i477_reval_confirm.py": _NA_INFRA + " (B-matrix norm check)",
    "scripts/i504_eval_trajectory.py": _NA_INFRA + " (marker logprob eval)",
    "scripts/issue_480/dispatch_marker_480.py": _NA_INFRA,
    "scripts/issue_480/i480_analyze.py": _NA_INFRA + " (marker DV analysis)",
    "scripts/issue_480/plot_clean_result.py": _NA_INFRA,
    "scripts/issue_480/smoke_analyzer_synthetic.py": _NA_INFRA,
    "src/explore_persona_space/experiments/marker_implant_480/build_training_pool.py": _NA_INFRA,
    "scripts/run_em_multiseed.py": _NA_INFRA,
    "scripts/run_issue_358_extract.py": _NA_INFRA
    + " (activation extraction only; no cosine computed)",
    "scripts/train_stage_dpo.py": _NA_INFRA,
    "scripts/train_stage_sft.py": _NA_INFRA,
    "scripts/run_a3_leakage.py": _NA_INFRA + " (legacy leakage rig; no cos computation)",
    "scripts/run_a3b_experiment.py": _NA_INFRA + " (legacy leakage rig; no cos computation)",
    "scripts/run_persona_composition.py": _NA_INFRA + " (composition rig)",
    "scripts/archive/run_leakage_v2.py": "archived — superseded centered-line consumer "
    "(logs cos(source, asst) from persisted layer payloads)",
    "scripts/archive/test_multidim_identity.py": "archived — whitening/identity test "
    "(different construct)",
    "scripts/test_multidim_identity_v2.py": _NA_AXIS,
    # #536 self-files (pre-registered so the documented verification command
    # survives the issue-536 -> main merge; round-1 review minor): the audit /
    # re-grade tooling computes raw AND centered deliberately as the audit
    # instrument, never as an experiment's distance source.
    "scripts/issue536_audit.py": "N-A — #536 audit tooling itself (this file; regex "
    "literals match the sweep patterns)",
    "scripts/issue536_recompute_driver.py": "N-A — #536 re-grade tooling itself "
    "(computes raw AND centered side by side as the audit instrument)",
    "scripts/issue536_figures.py": "N-A — #536 figures tooling (reads the regrade "
    "table; no cosine computed)",
}

PERSONA_VECTORS_A0 = {
    "record": "Phase A0 — Persona Vectors (Chen, Arditi, Sleight, Evans, Lindsey 2025, "
    "arXiv 2507.21509) builds each persona vector as a DIFFERENCE OF MEANS between "
    "trait-exhibiting and non-exhibiting response activations, which removes the shared "
    "component by construction; the centroid-bank analog of that correction is global-mean "
    "centering.",
    "verified": "fact-check time via the arxiv-latex MCP (plan §2; pipeline section states "
    "the difference-in-mean-activations construction verbatim)",
}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Task #536 audit sweep (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=REPO,
        help="Checkout holding the persisted artifacts for fingerprinting.",
    )
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=audit] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    sweep = run_sweep(args.data_root)
    site_files = {s["file"] for s in SITES}
    unaccounted = sorted(set(sweep) - site_files - set(DISPOSITIONS))
    if unaccounted:
        raise RuntimeError(
            f"audit coverage FAILED — {len(unaccounted)} swept file(s) have neither a "
            f"SITE row nor a DISPOSITION: {unaccounted}"
        )
    stale = sorted((site_files | set(DISPOSITIONS)) - set(sweep))
    for f in stale:
        log.warning(
            "[stale] curated entry not in current sweep (pre-registered for "
            "post-merge, or removed from the tree): %s",
            f,
        )

    dispositions = [
        {
            "file": f,
            "disposition": reason,
            "matched_passes": {
                "pass1": sweep.get(f, {}).get("pass1", False),
                "pass2": sweep.get(f, {}).get("pass2", False),
            },
            "evidence_lines": sweep.get(f, {}).get("evidence_lines", []),
        }
        for f, reason in sorted(DISPOSITIONS.items())
    ]
    table = {
        "schema_version": "i536_audit_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        # git_commit kept for back-compat (== code_commit); the two explicit
        # fields disambiguate which tree each SHA identifies (round-1 review:
        # the swept --data-root checkout drifts independently of the code tree).
        "git_commit": _git_sha(),
        "code_commit": _git_sha(),
        "data_root_commit": _git_sha(args.data_root),
        "sweep": {
            "roots": list(SWEEP_ROOTS),
            "pass1_regex": PASS1.pattern,
            "pass2_regex": PASS2.pattern,
            "n_files_hit": len(sweep),
            "n_pass1": sum(1 for v in sweep.values() if v["pass1"]),
            "deleted_producer_search": "git log --all -S cosine --diff-filter=D + targeted "
            "git show 9e6e31c3f / 69b34b94 (run at implementation time)",
        },
        "persona_vectors_a0": PERSONA_VECTORS_A0,
        "sites": SITES,
        "deleted_sites": DELETED_SITES,
        "dispositions": dispositions,
        "fingerprint_checks": fingerprint_checks(args.data_root),
        "affected_set": {
            "raw_line_regraded": [478, 490, 505, 396, 415, 405],
            "matrix_only_sensitivity": [474, 406, 460, 341],
            "needs_gpu_or_unrecoverable": [213, 227, 99],
            "already_remediated_readoff": [472, 504],
            "pairwise_labeled_not_regraded": [404, 458, 444, 493, 502, 488],
            "canonical_line_verified": [66, 142, 311, 380],
            "canonical_line_unrecoverable_partition": [61, 77, 91, 228],
            "canonical_line_audit_only": [96, 245, 247, 329],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2))
    log.info(
        "[done] %d sites + %d deleted + %d dispositions over %d swept files -> %s",
        len(SITES),
        len(DELETED_SITES),
        len(dispositions),
        len(sweep),
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
