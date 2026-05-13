#!/usr/bin/env python3
"""Phase 2 projection — issue #368 §4.2.2 + §4.2.3.

Pipeline:
  1. Build 50 directed pairs via T2 simple rule:
       SOURCES × (ALL_EVAL_PERSONAS \\ source) = 5 × 10 = 50
  2. Load per-pair leakage rates from
       eval_results/single_token_100_persona/{source}/marker_eval.json
  3. For each pair (S, T): compute centered_cosine + projdiff using
       persona vector for S × target activation for T (extracted by Phase 2
       of i368_extract_chenstyle_vectors.py — the per-persona pos centroid is
       the target activation).
  4. R2 reproduction-sanity gate: recompute Method-A centered-cosine ρ from
       #142's existing centroids_layer20.pt (if available) and JS-ρ; both
       must match published baselines (0.567 and 0.746) within ±0.03.
  5. Write 50-row leakage table + persona_pos_set_cohesion.json (R11) +
       reproduction_sanity.json.

Output:
  eval_results/issue_368/phase2/leakage_table.csv
  eval_results/issue_368/phase2/reproduction_sanity.json
  eval_results/issue_368/phase2/persona_pos_set_cohesion.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # M4: enable `scripts.*` imports under `uv run python ...`
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from explore_persona_space.axis.chenstyle import (  # noqa: E402
    AXIS_SPECS,
    DEFAULT_LAYERS,
    HEADLINE_LAYER,
    centered_cosine,
    projdiff_score,
)
from explore_persona_space.eval.leakage_axes import (  # noqa: E402
    dump_json,
    spearman_with_p,
)
from scripts.i368_extract_chenstyle_vectors import (  # type: ignore  # noqa: E402
    ALL_EVAL_PERSONAS,
    NON_BASELINE_PERSONAS,
    OUTPUT_BASE,
)

# ── Constants ────────────────────────────────────────────────────────────────

SOURCES: list[str] = [
    "villain",
    "comedian",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
]

MARKER_EVAL_BASE = REPO_ROOT / "eval_results" / "single_token_100_persona"
METHOD_A_CENTROIDS = MARKER_EVAL_BASE / "centroids" / "centroids_layer20.pt"
JS_MATRIX_PATH = REPO_ROOT / "eval_results" / "js_divergence" / "divergence_matrices.json"
JS_MATRIX_FALLBACK = (
    REPO_ROOT.parent / "issue-274" / "eval_results" / "js_divergence" / "divergence_matrices.json"
)

OUT_DIR = REPO_ROOT / "eval_results" / "issue_368" / "phase2"


# ── Pair construction (T2 simpler rule) ─────────────────────────────────────


def build_50_pairs() -> list[tuple[str, str]]:
    pairs = [(s, t) for s in SOURCES for t in ALL_EVAL_PERSONAS if t != s]
    assert len(pairs) == 50, f"expected 50 directed pairs, got {len(pairs)}"
    return pairs


def load_leakage_rates() -> dict[tuple[str, str], float]:
    """``{(source, target): rate}`` for the 50 directed pairs."""
    out: dict[tuple[str, str], float] = {}
    for source in SOURCES:
        path = MARKER_EVAL_BASE / source / "marker_eval.json"
        with open(path) as f:
            data = json.load(f)
        for target, entry in data.items():
            if target == source:
                continue
            if target not in ALL_EVAL_PERSONAS:
                continue
            out[(source, target)] = float(entry["rate"])
    missing = [p for p in build_50_pairs() if p not in out]
    if missing:
        raise RuntimeError(f"missing leakage rates for {missing}")
    return out


# ── Persona-vec / target-activation loading ─────────────────────────────────


def _load_persona_pvec(source: str, axis_spec: dict) -> torch.Tensor:
    import torch

    flavor = axis_spec["flavor"]
    layer = axis_spec["layer"]
    base = OUTPUT_BASE / "personas" / source

    # C2: assistant is the negative anchor — we do NOT extract a chenstyle /
    # orthog / projdiff vector for it (those would collapse to numerical noise
    # since pos_centroid ≈ neg_centroid). For the 10 directed pairs with
    # source=assistant we use the Method-B pos-centroid as the surrogate
    # source vector across every "chenstyle*" flavor; centroid_means
    # centering still gives a valid centered-cosine score that the analyzer
    # can correctly flag (and the source=assistant rows are excluded from H2
    # contrast claims anyway — assistant only appears as TARGET in plan §6.2).
    chenstyle_family = {
        "chenstyle",
        "chenstyle_orthog",
        "chenstyle_projdiff",
    }
    if source == "assistant" and flavor in chenstyle_family:
        # Sentinel: load the assistant's pos-centroid at this layer. Downstream
        # analysis must not treat source=assistant rows as a chenstyle-vector
        # signal — they're labelled in the table via the source column and
        # excluded from H2-contrast computation in the analysis script.
        d = torch.load(base / "pos_centroids_mean_response.pt", weights_only=True)
        return d[layer]

    if flavor == "chenstyle":
        if axis_spec["aggregation"] == "last_token":
            return torch.load(base / f"pvec_lasttoken_L{layer}.pt", weights_only=True)
        return torch.load(base / f"pvec_L{layer}.pt", weights_only=True)
    if flavor == "chenstyle_orthog":
        return torch.load(base / f"pvec_orthog_L{layer}.pt", weights_only=True)
    if flavor == "chenstyle_projdiff":
        return torch.load(base / f"pvec_L{layer}.pt", weights_only=True)
    if flavor == "method_a":
        return torch.load(base / f"pcentroid_methodA_L{layer}.pt", weights_only=True)
    if flavor == "method_b":
        return torch.load(base / f"pcentroid_methodB_L{layer}.pt", weights_only=True)
    if flavor == "pos_only_chenstyle":
        d = torch.load(base / "pos_centroids_mean_response.pt", weights_only=True)
        return d[layer]
    raise ValueError(f"unknown flavor {flavor}")


def _load_target_act(target: str, layer: int, aggregation: str) -> torch.Tensor:
    """Target activation: per-persona pos centroid at the requested layer.

    For ``last_token`` aggregation we use the last-input-token centroid
    (Method A symmetry); otherwise the mean-response centroid (Method B
    canonical for chenstyle).
    """
    import torch

    base = OUTPUT_BASE / "personas" / target
    if aggregation == "last_token":
        d = torch.load(base / "pos_centroids_last_input_token.pt", weights_only=True)
    else:
        d = torch.load(base / "pos_centroids_mean_response.pt", weights_only=True)
    return d[layer]


# ── R2 reproduction-sanity gate ─────────────────────────────────────────────


def reproduction_sanity_gate(  # noqa: C901  -- R2 gate: each check must stay inline for audit clarity
    pairs: list[tuple[str, str]], leakage: dict, *, tolerance: float = 0.03
) -> dict:
    """Recompute Method-A centered-cos-L20 ρ and JS ρ; compare to #142 published baselines.

    Returns a verdict dict and raises on hard failure (per plan §7 gate 2).
    """
    import numpy as np
    import torch

    result: dict = {
        "pairs": [{"source": s, "target": t, "leakage": leakage[(s, t)]} for s, t in pairs],
        "tolerance": tolerance,
        "expected": {"js": 0.746, "method_a_centered_cos_L20": 0.567},
    }

    # JS-ρ check
    js_path = JS_MATRIX_PATH if JS_MATRIX_PATH.exists() else JS_MATRIX_FALLBACK
    if js_path.exists():
        with open(js_path) as f:
            js_blob = json.load(f)
        # divergence_matrices.json schema varies; common keys: "js" / "JS".
        js_matrix = None
        if isinstance(js_blob, dict):
            for k in ("js_divergence", "js", "JS", "JS_divergence"):
                if k in js_blob:
                    js_matrix = js_blob[k]
                    break
            if js_matrix is None and "matrices" in js_blob:
                js_matrix = js_blob["matrices"].get("js")
        result["js_matrix_source"] = (
            str(js_path.relative_to(REPO_ROOT))
            if js_path.is_relative_to(REPO_ROOT)
            else str(js_path)
        )
        if js_matrix is None:
            result["js_check"] = {"skipped": "could not parse divergence_matrices.json schema"}
        else:
            js_vals, lk_vals = [], []
            for s, t in pairs:
                row = js_matrix.get(s)
                if not row or t not in row:
                    continue
                js_vals.append(float(row[t]))
                lk_vals.append(leakage[(s, t)])
            rho, p = spearman_with_p(np.array(js_vals), np.array(lk_vals))
            result["js_check"] = {
                "rho": float(rho),
                "abs_rho": float(abs(rho)),
                "p": float(p),
                "n": len(js_vals),
                "matches_published": bool(abs(abs(rho) - 0.746) <= tolerance),
            }
    else:
        result["js_check"] = {"skipped": f"no JS matrix on disk ({js_path})"}

    # Method-A centered-cosine ρ check (#142 published 0.567)
    persona_names_path = METHOD_A_CENTROIDS.parent / "persona_names.json"
    if METHOD_A_CENTROIDS.exists() and persona_names_path.exists():
        # The file is Tensor[111, 3584] — a single layer's stacked centroids,
        # NOT a {persona: tensor} dict. The row ordering is in persona_names.json
        # (written alongside by scripts/analyze_100_persona_cosine.py).
        d = torch.load(METHOD_A_CENTROIDS, weights_only=True)
        with open(persona_names_path) as f:
            persona_names = json.load(f)
        name_to_idx = {n: i for i, n in enumerate(persona_names)}
        if not isinstance(d, torch.Tensor) or d.dim() != 2:
            raise RuntimeError(
                f"Phase 2 reproduction-sanity gate FAILED: expected "
                f"centroids_layer20.pt to be a 2D Tensor[N, hidden]; got "
                f"type={type(d).__name__} shape={getattr(d, 'shape', None)}."
            )

        def _layer20(p: str) -> torch.Tensor:
            if p not in name_to_idx:
                raise RuntimeError(
                    f"Phase 2 reproduction-sanity gate FAILED: persona {p!r} "
                    f"not in persona_names.json (size={len(persona_names)})."
                )
            return d[name_to_idx[p]].float()

        stacked = [_layer20(p) for p in ALL_EVAL_PERSONAS]
        centroid_mean = torch.stack(stacked).mean(dim=0)
        scores, lks = [], []
        for s, t in pairs:
            vs = _layer20(s)
            vt = _layer20(t)
            scores.append(centered_cosine(vs, vt, centroid_mean))
            lks.append(leakage[(s, t)])
        rho, p = spearman_with_p(np.array(scores), np.array(lks))
        result["method_a_check"] = {
            "rho": float(rho),
            "abs_rho": float(abs(rho)),
            "p": float(p),
            "n": len(scores),
            "matches_published": bool(abs(abs(rho) - 0.567) <= tolerance),
            "centroid_source": str(METHOD_A_CENTROIDS.relative_to(REPO_ROOT)),
        }
    else:
        missing = []
        if not METHOD_A_CENTROIDS.exists():
            missing.append(str(METHOD_A_CENTROIDS))
        if not persona_names_path.exists():
            missing.append(str(persona_names_path))
        result["method_a_check"] = {
            "skipped": f"Method-A inputs missing: {missing}",
        }

    # M2 fix: plan §7 halt-on-either-failure. No "PARTIAL" verdict.
    js_check = result.get("js_check", {})
    ma_check = result.get("method_a_check", {})
    js_failed = js_check.get("matches_published") is False
    ma_failed = ma_check.get("matches_published") is False
    if js_failed or ma_failed:
        raise RuntimeError(
            "Phase 2 reproduction-sanity gate FAILED — at least one of "
            f"JS-ρ or Method-A centered-cos-L20-ρ missed ±{tolerance} tolerance.\n"
            f"  JS:       {js_check}\n  MethodA:  {ma_check}"
        )
    result["verdict"] = "PASS"
    return result


# ── Cohesion + cross-persona variance diagnostic (R11) ──────────────────────


def persona_pos_set_cohesion() -> dict:
    """For each non-baseline persona: mean pairwise cosine over the 5
    paraphrase response-mean activations at L20 (per-persona cohesion).

    Plus cross-persona centroid variance: variance over the 10 centered
    pos_centroid[L20] vectors (R11).
    """
    import torch

    # We don't persist per-paraphrase activations (the extraction script
    # averages over paraphrases × questions in a single pass). For a
    # cohesion estimate we therefore re-derive from raw response caches via
    # post-hoc forward-pass extraction at analysis time — this is expensive,
    # so we fall back to a structural variance proxy: ratio of intra-cluster
    # (within-persona) variance to inter-cluster (across-persona) variance
    # over the per-persona pos centroids. Records the diagnostic even when
    # the full pairwise-cosine cohesion can't be computed.

    centroids: dict[str, torch.Tensor] = {}
    for p in NON_BASELINE_PERSONAS:
        path = OUTPUT_BASE / "personas" / p / "pos_centroids_mean_response.pt"
        if not path.exists():
            return {"skipped": f"missing {path}"}
        d = torch.load(path, weights_only=True)
        centroids[p] = d[HEADLINE_LAYER].float()

    # Cross-persona centroid variance (R11)
    cm_path = OUTPUT_BASE / f"_centroid_mean_L{HEADLINE_LAYER}.pt"
    if not cm_path.exists():
        return {"skipped": f"missing centroid_mean at {cm_path}"}
    centroid_mean = torch.load(cm_path, weights_only=True).float()
    stacked = torch.stack(
        [centroids[p] - centroid_mean for p in NON_BASELINE_PERSONAS]
    )  # (10, hidden)
    cross_persona_var = float(stacked.var(dim=0).mean().item())

    # Pairwise per-persona persona-centroid cosine (proxy when per-paraphrase
    # responses aren't persisted): compute pairwise cosines between the 10
    # personas' centered centroids. Lower mean here = personas more distinct;
    # higher = personas more uniform (Sonnet flatness risk).
    def _unit(v):
        n = v.norm()
        return v / n if n > 1e-12 else v

    units = [_unit(stacked[i]) for i in range(len(NON_BASELINE_PERSONAS))]
    pair_cos = []
    for i, j in combinations(range(len(NON_BASELINE_PERSONAS)), 2):
        pair_cos.append(float((units[i] @ units[j]).item()))
    inter_persona_cosine_mean = float(sum(pair_cos) / len(pair_cos))

    # R11 ratio: cross_persona_centroid_variance_ratio.
    # We don't have a Phase-1 trigger-centroid-variance scalar at this point
    # (Phase 1 extraction happens after Phase 2 in the pipeline). We persist
    # the raw variance + a placeholder ratio key the analysis script can
    # update post-hoc.
    return {
        "cross_persona_centroid_variance": cross_persona_var,
        "cross_persona_centroid_variance_ratio_to_phase1_mean": None,
        "inter_persona_centered_cosine_mean": inter_persona_cosine_mean,
        "n_personas": len(NON_BASELINE_PERSONAS),
        "layer": HEADLINE_LAYER,
        "notes": (
            "Per-paraphrase within-persona cohesion not computed (extraction "
            "averages over paraphrases before persistence). The "
            "inter_persona_centered_cosine_mean is the structural proxy for "
            "Sonnet flatness — higher = personas more uniform. R11 ratio "
            "filled in by phase2_analysis once Phase 1 centroid variance is "
            "known."
        ),
    }


# ── Build the 50-row leakage table ──────────────────────────────────────────


def _load_js_matrix() -> dict[str, dict[str, float]] | None:
    """C5: load #142 JS divergence matrix indexed by source x target."""
    js_path = JS_MATRIX_PATH if JS_MATRIX_PATH.exists() else JS_MATRIX_FALLBACK
    if not js_path.exists():
        return None
    with open(js_path) as f:
        js_blob = json.load(f)
    js_matrix = None
    if isinstance(js_blob, dict):
        for k in ("js_divergence", "js", "JS", "JS_divergence"):
            if k in js_blob:
                js_matrix = js_blob[k]
                break
        if js_matrix is None and "matrices" in js_blob:
            js_matrix = js_blob["matrices"].get("js")
    return js_matrix


def _compute_method_a_centered_cosines(
    pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], float] | None:
    """C5: Method-A centered-cosine ρ at L20 (#142 axis), per (source, target).

    Mirrors the reproduction-sanity gate's Method-A computation but persists
    the per-pair score so leakage_table.csv can carry cosine_L20_centered as
    a column (plan §4.2.3).
    """
    import torch

    persona_names_path = METHOD_A_CENTROIDS.parent / "persona_names.json"
    if not (METHOD_A_CENTROIDS.exists() and persona_names_path.exists()):
        return None
    d = torch.load(METHOD_A_CENTROIDS, weights_only=True)
    with open(persona_names_path) as f:
        persona_names = json.load(f)
    name_to_idx = {n: i for i, n in enumerate(persona_names)}
    if not isinstance(d, torch.Tensor) or d.dim() != 2:
        return None
    stacked = [d[name_to_idx[p]].float() for p in ALL_EVAL_PERSONAS]
    centroid_mean = torch.stack(stacked).mean(dim=0)
    out: dict[tuple[str, str], float] = {}
    for s, t in pairs:
        vs = d[name_to_idx[s]].float()
        vt = d[name_to_idx[t]].float()
        out[(s, t)] = float(centered_cosine(vs, vt, centroid_mean))
    return out


def build_leakage_table(
    pairs: list[tuple[str, str]],
    leakage: dict[tuple[str, str], float],
) -> None:
    import torch

    centroid_means: dict[int, torch.Tensor] = {}
    for layer in DEFAULT_LAYERS:
        centroid_means[layer] = torch.load(
            OUTPUT_BASE / f"_centroid_mean_L{layer}.pt", weights_only=True
        )
    helpful_act = torch.load(
        OUTPUT_BASE / "_helpful_assistant" / f"helpful_test_act_L{HEADLINE_LAYER}.pt",
        weights_only=True,
    )

    # C5: per-pair js_div + cosine_L20_centered for downstream T12 calibration.
    js_matrix = _load_js_matrix()
    method_a_cos = _compute_method_a_centered_cosines(pairs)

    rows: list[dict] = []
    new_cols = [a["name"] for a in AXIS_SPECS]
    for source, target in pairs:
        row = {
            "source": source,
            "target": target,
            "marker_leakage_rate": f"{leakage[(source, target)]:.6f}",
        }
        # C5: js_div column. Missing source/target raises (no silent NaN).
        if js_matrix is not None:
            srow = js_matrix.get(source)
            if not srow or target not in srow:
                raise RuntimeError(
                    f"C5: js_div missing for pair ({source!r}, {target!r}) in "
                    f"divergence_matrices.json."
                )
            row["js_div"] = f"{float(srow[target]):.10f}"
        else:
            raise RuntimeError(
                "C5: divergence_matrices.json absent on disk — Phase 2 leakage "
                "table cannot be built without js_div for T12 calibration."
            )
        # C5: cosine_L20_centered (Method-A axis from #142).
        if method_a_cos is not None:
            row["cosine_L20_centered"] = f"{method_a_cos[(source, target)]:.10f}"
        else:
            raise RuntimeError(
                "C5: Method-A centroids missing — Phase 2 leakage table cannot "
                "be built without cosine_L20_centered for T12 calibration."
            )
        for axis_spec in AXIS_SPECS:
            pvec = _load_persona_pvec(source, axis_spec)
            target_act = _load_target_act(target, axis_spec["layer"], axis_spec["aggregation"])
            cm = centroid_means[axis_spec["layer"]]
            if axis_spec["flavor"] == "chenstyle_projdiff":
                score = projdiff_score(pvec, target_act, helpful_act, cm)
            else:
                score = centered_cosine(pvec, target_act, cm)
            row[axis_spec["name"]] = f"{score:.10f}"
        rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "leakage_table.csv"
    fieldnames = [
        "source",
        "target",
        "marker_leakage_rate",
        "js_div",
        "cosine_L20_centered",
        *new_cols,
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[Phase 2] wrote leakage_table.csv ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-sanity-gate", action="store_true")
    args = ap.parse_args()

    pairs = build_50_pairs()
    leakage = load_leakage_rates()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_sanity_gate:
        try:
            sanity = reproduction_sanity_gate(pairs, leakage)
        except RuntimeError as e:
            # Persist the failed verdict before re-raising so the analyzer can
            # inspect it after the run halts.
            dump_json({"verdict": "FAIL", "error": str(e)}, OUT_DIR / "reproduction_sanity.json")
            raise
        dump_json(sanity, OUT_DIR / "reproduction_sanity.json")
        print(f"[Phase 2] reproduction sanity: {sanity['verdict']}")

    cohesion = persona_pos_set_cohesion()
    dump_json(cohesion, OUT_DIR / "persona_pos_set_cohesion.json")

    build_leakage_table(pairs, leakage)


if __name__ == "__main__":
    main()
