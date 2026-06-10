# ruff: noqa: RUF002, RUF003  # em-dash + Qwen marker " ※" + × + − intentional
"""Task #504 Phase 0.5 — CPU-only identification gates + persona-bank checks.

Plan §4.2: BEFORE any Phase 1 training spawns, analytically verify the
persona bank supports the placement axis with enough across-arm movement.

Gates (all 3 must pass at the chosen layer):
  A. Identification floor: median across-arm SD of `d_nearest_neg_nd` ≥ 0.02
     AND median across-arm SD of `shadow_angle` ≥ 0.10 rad.
  B. Default-assistant non-dominance: < 0.5 of probes have `qwen_default` as
     the SINGLE nearest negative across ALL 4 arms.
  C. Held-out panel sufficiency: ≥ 30 probes after exclusions.

Plus the max-length check: every villain on-policy R_train response token
length ≤ train_max_length (1024). If any saturates 1024, bump eval
max_new_tokens to 4096.

Output: phase0_5_gates.json (plan §4.2):
  {
    "chosen_layer": int,                  # the layer the gates PASSED on
    "chosen_negatives": {                 # the 4 positioned-N personas + the panel
        "near": ..., "mid_near": ..., "mid_far": ..., "far": ..., "default": "qwen_default",
    },
    "arm_to_positioned_n": {              # arm slug → positioned-N persona
        # v1 (rank-ladder) keys
        "c504_near": ..., "c504_mid_near": ..., "c504_mid_far": ..., "c504_far": ...,
        # v2 (lr-ladder) keys (same persona answers — only the cell-slug naming differs)
        "c504v2_near": ..., "c504v2_mid_near": ..., "c504v2_mid_far": ..., "c504v2_far": ...,
    },
    "smoke_mid_band_n": str,              # the Phase 0 smoke's positioned-N
    "held_out_panel": [persona, ...],
    "per_probe": {                        # for the regression input
        probe: {
            "d_source": float,
            "d_nearest_neg_nd": {arm: float},
            "shadow_angle": {arm: float},
        }
    },
    "gate_results": {                     # per-layer verdicts
        "L10": {"A": {...}, "B": {...}, "C": {...}},
        "L15": {"A": {...}, "B": {...}, "C": {...}},
        "L20": {"A": {...}, "B": {...}, "C": {...}},
    },
    "max_length_check": {
        "max_response_tokens": int,
        "train_max_length": int,
        "needs_max_new_tokens_4096": bool,
    },
    "verdict": "pass" | "fail",
  }

CPU + ONE GPU forward pass if centroids missing (otherwise pure CPU).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
    ALWAYS_INCLUDE_NEGATIVE,
    DEFAULT_HEADLINE_LAYER,
    FALLBACK_LAYERS,
    PHASE05_DNN_FLOOR,
    PHASE05_PANEL_MIN_PROBES,
    PHASE05_QWEN_DEFAULT_DOMINANCE_THRESHOLD,
    PHASE05_SHADOW_FLOOR_RAD,
    POSITIONED_ARM_SLUGS,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.contrastive_neg_geometry_504.cell_resolution import (
    pick_smoke_mid_band_n,
    select_positioned_negatives,
)
from explore_persona_space.experiments.contrastive_neg_geometry_504.shadow_angle import (
    gate_a_identification_floor,
    gate_b_qwen_default_dominance,
    gate_c_panel_sufficiency,
    shadow_angle,
)

log = logging.getLogger("issue_504.phase05")


def _cos_matrix_from_centroids(
    centroids: dict[str, np.ndarray],
    *,
    mean_center: bool = True,
) -> dict[str, dict[str, float]]:
    """Compute the bank-wide pairwise cosine matrix from centroids.

    Returns ``{a: {b: cos(a, b)}}``. Symmetric, but stored doubled for O(1) lookup.

    Default (``mean_center=True``, #504 round-6) follows the #66/#341 methodology:
    subtract the global per-component mean across the FULL bank (every persona in
    ``centroids``), then L2-normalize, then dot. Without this step the bank-wide
    pairwise cosines saturate to a narrow band on Qwen-2.5-7B-Instruct
    (round 1-5 #504 spread ≈0.21 at L20 vs >1.2 with mean-centering) — see
    ``scripts/i504_round6_recompute_mean_centered.py`` for the spread comparison.

    Pass ``mean_center=False`` to recover round 1-5 raw-cosine behavior.
    """
    personas = sorted(centroids)
    vecs = np.stack([centroids[p] for p in personas]).astype(np.float64)
    if mean_center:
        vecs = vecs - vecs.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Guard against zero-norm centroids (shouldn't happen but defensive).
    norms = np.where(norms == 0.0, 1.0, norms)
    unit = vecs / norms
    cos = unit @ unit.T
    out: dict[str, dict[str, float]] = {p: {} for p in personas}
    for i, a in enumerate(personas):
        for j, b in enumerate(personas):
            out[a][b] = float(cos[i, j])
    return out


def _per_probe_covariates(
    panel: list[str],
    arm_to_n: dict[str, str],
    cos_to_source: dict[str, float],
    cos_matrix: dict[str, dict[str, float]],
    centroids: dict[str, np.ndarray],
    source: str,
) -> dict[str, dict[str, Any]]:
    """Compute d_source, d_nearest_neg_nd, shadow_angle per (probe, arm).

    base_prior_marker is computed at Phase 1 eval time (it requires a real
    forward pass on each probe's prompt against the base model — not a
    centroid lookup). It is written into the per-probe records by the eval
    rig; Phase 0.5 reports the geometric covariates.

    arm_to_n maps the 4 POSITIONED arm slugs to their negative; the
    default-only arm has no positioned N → its d_nn / shadow are NaN.
    """
    out: dict[str, dict[str, Any]] = {}
    for probe in panel:
        d_source = 1.0 - cos_to_source[probe]
        d_nn = {arm: 1.0 - cos_matrix[probe][n] for arm, n in arm_to_n.items()}
        shadow = {
            arm: shadow_angle(centroids[probe], centroids[n], centroids[source])
            for arm, n in arm_to_n.items()
        }
        out[probe] = {
            "d_source": float(d_source),
            "d_nearest_neg_nd": d_nn,
            "shadow_angle": shadow,
        }
    return out


def run_gates_for_layer(
    layer: int,
    *,
    centroids: dict[str, np.ndarray],
    arm_to_positioned_n: dict[str, str],
    held_out_panel: list[str],
    source: str = SOURCE_PERSONA,
    default_persona: str = ALWAYS_INCLUDE_NEGATIVE,
    mean_center: bool = True,
) -> dict[str, Any]:
    """Compute Gates A/B/C diagnostics for ONE layer (no failure escalation).

    ``mean_center`` (#504 round-7 fix, blocker 1) threads the cosine-centering
    choice all the way through to ``_cos_matrix_from_centroids`` so the gate
    diagnostics + per-probe covariates honor the dispatcher's ``--no-mean-center``
    opt-out. Round-6 default = True (mean-centered, #66/#341 methodology); pass
    False to recover round 1-5 raw-cosine behavior.
    """
    cos_matrix = _cos_matrix_from_centroids(centroids, mean_center=mean_center)
    gate_a = gate_a_identification_floor(
        held_out_panel,
        arm_to_positioned_n,
        cos_matrix,
        centroids,
        source,
        d_nn_floor=PHASE05_DNN_FLOOR,
        shadow_floor_rad=PHASE05_SHADOW_FLOOR_RAD,
    )
    gate_b = gate_b_qwen_default_dominance(
        held_out_panel,
        arm_to_positioned_n,
        cos_matrix,
        default_persona,
        dominance_threshold=PHASE05_QWEN_DEFAULT_DOMINANCE_THRESHOLD,
    )
    gate_c = gate_c_panel_sufficiency(held_out_panel, min_probes=PHASE05_PANEL_MIN_PROBES)
    return {
        "layer": layer,
        "A": gate_a,
        "B": gate_b,
        "C": gate_c,
        "all_pass": gate_a["pass"] and gate_b["pass"] and gate_c["pass"],
    }


def select_chosen_layer(
    per_layer_results: dict[int, dict[str, Any]],
    headline: int = DEFAULT_HEADLINE_LAYER,
    fallback: tuple[int, ...] = FALLBACK_LAYERS,
) -> tuple[int | None, str]:
    """Pick the layer whose gates ALL pass (headline first, then fallback).

    Plan §4.2 step-6 / failure tree:
      - Headline (L10) all-pass → use L10.
      - Headline fails Gate A/B/C → try L15, then L20. If L15 passes, use L15
        as headline + L10 as robustness; if L20 passes alone, use L15 headline
        (per plan failure mode "Gate B passes only at L20" → L15 headline +
        relaxed shadow floor; this module returns the layer choice + verdict
        string and the dispatcher reads the relaxed-floor decision off the
        verdict).

    Returns:
        (chosen_layer or None, verdict_string).
    """
    if per_layer_results[headline]["all_pass"]:
        return headline, f"L{headline}_passes"
    for lay in fallback:
        if per_layer_results[lay]["all_pass"]:
            return lay, f"headline_{headline}_failed_fallback_{lay}_passes"
    return None, "all_layers_failed"


def max_response_token_check(
    r_train: dict[str, dict[str, dict]],
    source: str,
    train_max_length: int = 1024,
) -> dict[str, Any]:
    """Plan §4.2 step 7: every villain on-policy R fits in train_max_length.

    Reads `response_token_ids` from the cached R artifact; if a row saturates
    train_max_length, the eval `max_new_tokens` must bump to 4096 (CLAUDE.md
    ≥ 2× longest constraint; #260 silent-zero precedent).
    """
    if source not in r_train:
        raise KeyError(
            f"max_response_token_check: source {source!r} missing from r_train; "
            f"got {sorted(r_train)[:6]}..."
        )
    lengths = []
    for _q, entry in r_train[source].items():
        ids = entry.get("response_token_ids")
        if ids is None:
            # Fall back to character-length / 4 if token ids not cached.
            txt = entry.get("response_text", "")
            lengths.append(len(txt) // 4)
        else:
            lengths.append(len(ids))
    max_len = max(lengths) if lengths else 0
    return {
        "max_response_tokens": int(max_len),
        "train_max_length": int(train_max_length),
        "needs_max_new_tokens_4096": max_len > train_max_length,
        "n_questions_checked": len(lengths),
    }


def _select_at_layer(
    *,
    layer: int,
    centroids_by_layer: dict[int, dict[str, np.ndarray]],
    cos_to_source_by_layer: dict[int, dict[str, float]],
    source: str,
    default_persona: str,
    mean_center: bool = True,
) -> dict[str, Any]:
    """Pick positioned-N's + smoke-mid-band-N + held-out panel + per-probe
    covariates ANCHORED at `layer`.

    Returns the bundle the gates need to evaluate this layer. Used both for
    the headline pick AND (round-2 fix, blocker #1) for the re-pick when a
    fallback layer wins — so `arm_to_positioned_n`, `per_probe`, and the
    panel are always layer-consistent with `chosen_layer`.

    ``mean_center`` (#504 round-7 fix, blocker 1) threads the cosine-centering
    choice into ``_per_probe_covariates`` so ``d_nearest_neg_nd`` matches the
    same geometry the dispatcher built ``cos_to_source_by_layer`` from.
    """
    cts = cos_to_source_by_layer[layer]
    band_to_n = select_positioned_negatives(cts, source=source, default_persona=default_persona)
    # Emit positioned-N entries under BOTH the v1 and v2 arm slugs so the
    # downstream `arm_to_n.json` is consumable by either dispatcher path
    # (the same persona answers either key — only the cell-slug naming
    # differs).
    arm_to_n = {
        # v1 slugs (rank-ladder pipeline).
        "c504_near": band_to_n["near"],
        "c504_mid_near": band_to_n["mid_near"],
        "c504_mid_far": band_to_n["mid_far"],
        "c504_far": band_to_n["far"],
        # v2 slugs (lr-ladder pipeline).
        "c504v2_near": band_to_n["near"],
        "c504v2_mid_near": band_to_n["mid_near"],
        "c504v2_mid_far": band_to_n["mid_far"],
        "c504v2_far": band_to_n["far"],
        # v3 slugs (EPOCHS-ladder pipeline / Phase 1 main arms).
        "c504v3_near": band_to_n["near"],
        "c504v3_mid_near": band_to_n["mid_near"],
        "c504v3_mid_far": band_to_n["mid_far"],
        "c504v3_far": band_to_n["far"],
    }
    smoke_mid_band = pick_smoke_mid_band_n(cts, source=source)
    excluded = {source, default_persona, *band_to_n.values()}
    panel = sorted(p for p in cts if p not in excluded)
    per_probe = _per_probe_covariates(
        panel,
        arm_to_n,
        cts,
        _cos_matrix_from_centroids(centroids_by_layer[layer], mean_center=mean_center),
        centroids_by_layer[layer],
        source,
    )
    return {
        "layer": layer,
        "band_to_n": band_to_n,
        "arm_to_n": arm_to_n,
        "smoke_mid_band_n": smoke_mid_band,
        "panel": panel,
        "per_probe": per_probe,
    }


def run_phase05(
    *,
    centroids_by_layer: dict[int, dict[str, np.ndarray]],
    cos_to_source_by_layer: dict[int, dict[str, float]],
    r_train_villain: dict[str, dict[str, dict]],
    source: str = SOURCE_PERSONA,
    default_persona: str = ALWAYS_INCLUDE_NEGATIVE,
    headline_layer: int = DEFAULT_HEADLINE_LAYER,
    fallback_layers: tuple[int, ...] = FALLBACK_LAYERS,
    mean_center: bool = True,
) -> dict[str, Any]:
    """Top-level Phase 0.5 runner.

    Picks positioned-N's + per-probe covariates at the HEADLINE layer first,
    runs the 3 gates there + at fallback layers, max-length checks the
    villain R artifact, and returns the full report dict (ready to write to
    phase0_5_gates.json).

    **Round-2 fix (binding blocker #1):** if the headline fails Gates A/B but
    a FALLBACK layer passes, this function NOW re-picks `arm_to_positioned_n`
    + `per_probe` + `held_out_panel` AT THE FALLBACK LAYER and writes the
    layer-consistent bundle to the report. The dispatcher no longer needs to
    branch — the report it consumes has `arm_to_positioned_n` / `per_probe`
    anchored at `chosen_layer` by construction. Gates per-layer in
    `gate_results` are still computed against the HEADLINE picks (for
    diagnostic visibility); the gate that fires the re-pick is the
    ``re_picked_at_chosen_layer`` block.

    **Round-7 fix (binding blocker 1):** ``mean_center`` is threaded all the
    way through ``_select_at_layer`` and ``run_gates_for_layer`` so the
    dispatcher's ``--no-mean-center`` opt-out actually disables mean-centering
    in the gate cosines AND the per-probe covariates AND the chosen-layer
    re-pick — not just in the dispatcher-built ``cos_to_source_by_layer``.
    The opt-out artifact is now internally self-consistent (gates + d_source
    + d_nearest_neg_nd all from the SAME geometry).
    """
    if headline_layer not in centroids_by_layer:
        raise KeyError(
            f"run_phase05: headline_layer={headline_layer} missing from centroids_by_layer; "
            f"got {sorted(centroids_by_layer)}"
        )

    # Pick positioned-N's + per-probe at the headline layer first.
    headline_pick = _select_at_layer(
        layer=headline_layer,
        centroids_by_layer=centroids_by_layer,
        cos_to_source_by_layer=cos_to_source_by_layer,
        source=source,
        default_persona=default_persona,
        mean_center=mean_center,
    )
    headline_arm_to_n = headline_pick["arm_to_n"]
    headline_panel = headline_pick["panel"]

    # Run gates at headline + fallback layers (positioned-N's are layer-locked
    # to the HEADLINE pick for these diagnostic gate checks; the verdict-time
    # re-pick below is what makes the FINAL report self-consistent).
    per_layer_results: dict[int, dict[str, Any]] = {}
    for lay in (headline_layer, *fallback_layers):
        per_layer_results[lay] = run_gates_for_layer(
            lay,
            centroids=centroids_by_layer[lay],
            arm_to_positioned_n=headline_arm_to_n,
            held_out_panel=headline_panel,
            source=source,
            default_persona=default_persona,
            mean_center=mean_center,
        )

    chosen_layer, verdict_str = select_chosen_layer(
        per_layer_results, headline=headline_layer, fallback=fallback_layers
    )

    # ── Round-2 fix (blocker #1): re-pick at chosen_layer if it's a fallback. ─
    # arm_to_n / per_probe / panel must come from chosen_layer (NOT headline)
    # so Phase 1 trains arms picked at the chosen layer + Phase 2 reads
    # per-probe covariates at the chosen layer.
    re_picked_at_chosen_layer = False
    if chosen_layer is not None and chosen_layer != headline_layer:
        final_pick = _select_at_layer(
            layer=chosen_layer,
            centroids_by_layer=centroids_by_layer,
            cos_to_source_by_layer=cos_to_source_by_layer,
            source=source,
            default_persona=default_persona,
            mean_center=mean_center,
        )
        re_picked_at_chosen_layer = True
        log.info(
            "[phase05] chosen_layer=%d != headline_layer=%d — RE-PICKED arm_to_n + "
            "per_probe + panel at L%d (round-2 fix, blocker #1).",
            chosen_layer,
            headline_layer,
            chosen_layer,
        )
        # Re-run the gates at chosen_layer using the RE-PICKED arm_to_n so the
        # gate diagnostics at chosen_layer match the final arms (otherwise the
        # gate_results would show the chosen layer's gates against the HEADLINE
        # arms, which is misleading).
        per_layer_results[chosen_layer] = run_gates_for_layer(
            chosen_layer,
            centroids=centroids_by_layer[chosen_layer],
            arm_to_positioned_n=final_pick["arm_to_n"],
            held_out_panel=final_pick["panel"],
            source=source,
            default_persona=default_persona,
            mean_center=mean_center,
        )
        if not per_layer_results[chosen_layer]["all_pass"]:
            # Re-running gates with the chosen-layer's own arms must still
            # pass; if it doesn't, the fallback verdict is invalid.
            log.error(
                "[phase05] re-picked gates at chosen_layer=%d failed; this contradicts the "
                "fallback verdict. Falling back to FAIL.",
                chosen_layer,
            )
            chosen_layer = None
            verdict_str = f"fallback_verdict_invalidated_by_repicked_gates_at_L{chosen_layer}"
            final_pick = headline_pick  # fall through to using headline (which itself failed).
    else:
        # Headline passed (or all layers failed) — use the headline pick as final.
        final_pick = headline_pick

    final_band_to_n = final_pick["band_to_n"]
    final_arm_to_n = final_pick["arm_to_n"]
    final_smoke_mid_band = final_pick["smoke_mid_band_n"]
    final_panel = final_pick["panel"]
    final_per_probe = final_pick["per_probe"]

    max_len_check = max_response_token_check(r_train_villain, source)

    final_verdict = "pass" if chosen_layer is not None else "fail"
    return {
        "chosen_layer": chosen_layer,
        "chosen_layer_verdict": verdict_str,
        "headline_layer_request": headline_layer,
        "fallback_layers": list(fallback_layers),
        "re_picked_at_chosen_layer": re_picked_at_chosen_layer,
        "headline_arm_to_positioned_n": headline_arm_to_n,
        "chosen_negatives": {
            **final_band_to_n,
            "default": default_persona,
        },
        "arm_to_positioned_n": final_arm_to_n,
        "smoke_mid_band_n": final_smoke_mid_band,
        "held_out_panel": final_panel,
        "per_probe": final_per_probe,
        "gate_results": {f"L{lay}": per_layer_results[lay] for lay in per_layer_results},
        "max_length_check": max_len_check,
        "verdict": final_verdict,
        "source": source,
        "default_persona": default_persona,
        "positioned_arms": list(POSITIONED_ARM_SLUGS),
    }


def write_phase05_artifact(report: dict[str, Any], out_path: Path) -> Path:
    """Write phase0_5_gates.json (plan §4.2 output format)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log.info(
        "[phase05] wrote %s (verdict=%s, chosen_layer=%s)",
        out_path,
        report.get("verdict"),
        report.get("chosen_layer"),
    )
    return out_path


def load_phase05(path: Path) -> dict[str, Any]:
    """Load + verdict-check a phase0_5_gates.json; raise on fail."""
    if not path.exists():
        raise FileNotFoundError(f"phase0_5_gates.json missing at {path}")
    report = json.loads(path.read_text())
    if report.get("verdict") != "pass":
        raise RuntimeError(
            f"phase0_5_gates.json verdict={report.get('verdict')!r}; identification "
            f"floor unreachable at any layer. See gate_results in {path}."
        )
    return report
