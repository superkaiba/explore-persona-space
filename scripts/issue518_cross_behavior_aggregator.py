#!/usr/bin/env python3
# Greek + special characters (×, →, —, α, Δ, ρ) appear in this file's prose
# for research notation.
# ruff: noqa: RUF001, RUF002, RUF003
"""#518 v4 cross-behavior aggregator -- the cell that answers the body's Y/N.

Reads three per-arm scoring JSONs from ``scripts/issue509_scoring.py``:
  - syco arm (re-scored with the new completion_logprob column)
  - refusal arm
  - EM arm

For each predictor cell present across all three arms:
  1. Compute the signed Spearman ρ triple ``(ρ_syco, ρ_refusal, ρ_em)`` --
     log all three values in a per-cell table (descriptive; cells that
     fail the sign gate appear here for interpretation).
  2. Apply the same-sign gate: ``same_sign(ρ_syco, ρ_refusal, ρ_em)``
     PASSES iff all three signs are equal AND none has |ρ| < 1e-6.
  3. For cells PASSING the sign gate, compute ``min(|ρ|) = min(|ρ_syco|,
     |ρ_refusal|, |ρ_em|)``. Rank passing cells by ``min(|ρ|)``.
  4. Headline = "any cell clears `same_sign(ρ_syco, ρ_refusal, ρ_em) AND
     min(|ρ|) >= 0.40` with permutation p <= 0.01 on all three arms?"
     Yes/No + which cell + per-arm signed ρ + per-arm p-values.

FAIL CLOSED on three preconditions (per plan §8 + §12):
  - Any input scoring JSON missing OR malformed.
  - Bake-off ``meta.json`` model_id mismatch across the three arms
    (the cross-arm residual-geometry comparison requires identical bases).
  - Any predictor cell missing on >=1 arm but present on others (the
    cell's signed-ρ triple is structurally incomplete -- the descriptive
    table lists it, but the headline is computed only over cells present
    on all three).

CLI:
  uv run python scripts/issue518_cross_behavior_aggregator.py \\
      --syco-scoring eval_results/issue_509/syco_arm/scoring.json \\
      --refusal-scoring eval_results/issue_518/refusal/scoring.json \\
      --em-scoring eval_results/issue_518/em/scoring.json \\
      --out eval_results/issue_518/cross_behavior_aggregator.json
  uv run python scripts/issue518_cross_behavior_aggregator.py --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

CROSS_BEHAVIOR_THRESHOLD = 0.40
PERM_P_THRESHOLD = 0.01
ZERO_RHO_EPSILON = 1e-6

# #518 v4 round-2 must-fix 1: coarse predictors the scoring arms attach
# under ``per_coarse_rho_fe[<predictor>]``. ``completion_logprob`` is the
# headline cross-behavior predictor named in plan §0/§1/§4.4/§11 ("the body
# of work this task is built around"); the rest are the #480 coarse-zoo
# columns the substrates already carry. Mirrors
# ``scripts/issue509_scoring.REFUSAL_EM_COARSE_PREDICTOR_COLUMNS`` -- single
# source of truth there; this list is the consumer-side enumeration.
COARSE_PREDICTORS: tuple[str, ...] = (
    "completion_logprob",
    "cosine_l20_baseline",
    "cosine_response_headline",
    "cosine_response_l7",
    "cosine_response_l14",
    "cosine_response_l21",
    "cosine_response_l27",
    "JS_sym_nats",
    "JS_from_source_nats",
    "JS_from_bystander_nats",
    "M_js",
    "KL_src_to_bys_nats",
    "KL_bys_to_src_nats",
    "KL_sym_nats",
    "source_base_rate",
    "bystander_base_rate",
    "base_rate_diff_neg_abs",
    "source_resp_len_mean",
    "bystander_resp_len_mean",
    "resp_len_diff_abs",
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            env={**os.environ},  # epm-lint: subprocess-env-inherit -- git probe
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _cell_key(cell: dict) -> tuple[str, int, str, str]:
    """Stable (point, layer, metric, variant) key per scoring-cell dict."""
    return (
        str(cell.get("point", cell.get("extraction_point", ""))),
        int(cell.get("layer", -1)),
        str(cell.get("metric", "")),
        str(cell.get("variant", "")),
    )


def _load_scoring(path: Path) -> dict[str, Any]:
    """Load + validate one scoring JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Scoring JSON missing: {path}")
    payload = json.loads(path.read_text())
    if "cells" not in payload:
        raise RuntimeError(
            f"Scoring JSON at {path} has no 'cells' field; keys={list(payload)[:10]}."
        )
    if not isinstance(payload["cells"], list):
        raise RuntimeError(f"Scoring JSON 'cells' at {path} is not a list.")
    return payload


# #518 v4 round-2 must-fix 3: the legacy-syco-meta carve-out is the ONLY
# permitted reason a meta path may be None on the must-fix-2 cross-arm
# consistency check; refusal + EM arms (built post-v4-patch) MUST always
# carry meta. ``--allow-legacy-syco-meta`` opts into that single carve-out.
_LEGACY_META_ALLOWED_ARMS: frozenset[str] = frozenset({"syco"})


def _assert_model_id_consistency(
    bakeoff_metas: dict[str, Path | None],
    *,
    allow_legacy_syco_meta: bool = False,
) -> dict[str, str | None]:
    """FAIL CLOSED if the three arms' bake-off meta.jsons declare different model_ids.

    ``bakeoff_metas`` maps arm name -> Path to meta.json (post-v4 patch) or
    None.

    #518 v4 round-2 must-fix 3: the None-skip is no longer unconditional.
    Round 1 allowed any arm's meta path to be None, which made it possible
    for a caller to omit all three flags and silently bypass the cross-arm
    consistency assertion entirely. The tightened contract:

    - When ``allow_legacy_syco_meta`` is ``False`` (default): every arm's
      meta path MUST be a concrete Path AND every file MUST resolve to a
      meta.json containing ``args.model_id``. Any None or any missing
      ``model_id`` raises ``RuntimeError`` -- the all-``None`` shortcut
      that bypassed the must-fix-2 check is closed.
    - When ``allow_legacy_syco_meta`` is ``True``: ONLY the syco arm's
      meta may be ``None`` (carve-out for the pre-v4-patch syco bake-off
      that predates the ``--model-id`` flag, per plan §10). The refusal +
      EM arms MUST still both carry meta + ``model_id``. Mismatches across
      whichever arms DO declare ``model_id`` are still raised.

    Returns the per-arm resolved model_ids. Raises if any disagreement.
    """
    resolved: dict[str, str | None] = {}
    for arm, meta_path in bakeoff_metas.items():
        if meta_path is None:
            if allow_legacy_syco_meta and arm in _LEGACY_META_ALLOWED_ARMS:
                # Carve-out: legacy syco bake-off predates --model-id. The
                # refusal + EM arms are still asserted below.
                resolved[arm] = None
                continue
            raise RuntimeError(
                f"bake-off meta.json missing for arm={arm!r}: every arm's "
                f"bake-off MUST record the resolved model_id in its meta.json. "
                f"#518 v4 round-2 must-fix 3: the unconditional all-None "
                f"shortcut that bypassed the cross-arm consistency assertion "
                f"is closed. Pass --{arm}-bakeoff-meta <path> (and the other "
                f"two arms' flags) so every model_id can be checked. The ONLY "
                f"carve-out is --allow-legacy-syco-meta, which opts in to the "
                f"syco arm's pre-v4 bake-off and the refusal + EM arms are "
                f"still required."
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"bake-off meta.json missing for arm={arm}: {meta_path}. "
                f"#518 v4 must-fix 2: every arm's bake-off MUST record the "
                f"resolved model_id in its meta.json."
            )
        meta = json.loads(meta_path.read_text())
        # The bake-off writes `args` under `meta.json`; the relevant field is
        # `model_id` (post-#518-v4 patch).
        model_id = meta.get("args", {}).get("model_id")
        if model_id is None:
            raise RuntimeError(
                f"bake-off meta.json for arm={arm!r} at {meta_path} has no "
                f"args.model_id. Re-run the offending arm's bake-off with "
                f"--model-id <canonical>. #518 v4 round-2 must-fix 3."
            )
        resolved[arm] = model_id
    declared = [v for v in resolved.values() if v is not None]
    if declared and len(set(declared)) > 1:
        raise RuntimeError(
            f"Bake-off model_id mismatch across arms: {resolved}. "
            f"#518 v4 must-fix 2: cross-arm residual-geometry comparison "
            f"requires identical base models. Re-run the offending arm's "
            f"bake-off with --model-id <canonical>."
        )
    return resolved


def _same_sign(rho_syco: float, rho_refusal: float, rho_em: float) -> bool:
    """Sign-gate per plan §4.5: all three signs equal AND none has |ρ| < eps."""
    if abs(rho_syco) < ZERO_RHO_EPSILON:
        return False
    if abs(rho_refusal) < ZERO_RHO_EPSILON:
        return False
    if abs(rho_em) < ZERO_RHO_EPSILON:
        return False
    s1 = 1 if rho_syco > 0 else -1
    s2 = 1 if rho_refusal > 0 else -1
    s3 = 1 if rho_em > 0 else -1
    return s1 == s2 == s3


def _extract_rho(cell: dict) -> float:
    """Pull the FE-adjusted Spearman ρ from a scoring-cell dict.

    The #509 scoring stores ``rho_fe_adjusted`` (preferred) or
    ``rho_obs_adjusted`` (when the FE adjustment is absent). Falls back
    to ``rho_fe`` then ``rho_obs`` for older shapes.
    """
    for key in (
        "rho_fe_adj",
        "rho_fe_adjusted",
        "rho_obs_adjusted",
        "rho_fe",
        "rho_obs",
    ):
        v = cell.get(key)
        if v is not None and isinstance(v, (int, float)):
            return float(v)
    raise KeyError(f"Scoring cell has no rho_* field; keys = {list(cell)[:20]}.")


def _extract_perm_p(cell: dict) -> float:
    """Pull the permutation p-value; default to 1.0 (worst) when absent."""
    for key in ("perm_p_fe", "perm_p_fe_adj", "perm_p", "perm_p_obs"):
        v = cell.get(key)
        if v is not None and isinstance(v, (int, float)):
            return float(v)
    return 1.0


def _collect_coarse_rho(
    scoring: dict[str, Any],
    predictor: str,
) -> dict[tuple[str, int, str, str], dict[str, float]]:
    """#518 v4 round-2 must-fix 1: per-cell coarse-predictor ρ extraction.

    Reads ``cell["per_coarse_rho_fe"][predictor]`` from every cell of one
    arm's scoring JSON and returns a mapping
    ``cell_key -> {"rho_fe_adj": float, "perm_p": float}``.

    The per-coarse Spearman ρ is computed inside ``issue509_scoring.py``'s
    ``_coarse_lift_syco_arm_per_cell`` on the SAME pairs as the residual-
    stream cell's (x, y), so the triple at the same cell key compares
    coarse-vs-residual-stream apples-to-apples on the cross-arm cell.
    """
    out: dict[tuple[str, int, str, str], dict[str, float]] = {}
    for cell in scoring["cells"]:
        per_coarse = cell.get("per_coarse_rho_fe")
        if not isinstance(per_coarse, dict):
            continue
        payload = per_coarse.get(predictor)
        if not isinstance(payload, dict):
            continue
        rho = payload.get("rho_fe_adj")
        if rho is None:
            rho = payload.get("rho_fe")
        perm_p = payload.get("perm_p")
        if rho is None:
            continue
        try:
            rho_f = float(rho)
        except (TypeError, ValueError):
            continue
        if math.isnan(rho_f):
            continue
        out[_cell_key(cell)] = {
            "rho_fe_adj": rho_f,
            "perm_p": float(perm_p) if perm_p is not None else 1.0,
        }
    return out


def _aggregate_coarse_predictors(
    syco_scoring: dict[str, Any],
    refusal_scoring: dict[str, Any],
    em_scoring: dict[str, Any],
) -> dict[str, Any]:
    """#518 v4 round-2 must-fix 1: per-coarse-predictor cross-arm triples.

    For each predictor in ``COARSE_PREDICTORS`` (headline:
    ``completion_logprob``), this:

      1. Collects the per-cell ρ_fe_adj from each arm.
      2. Joins on the cells present in ALL three arms.
      3. For each joined cell, builds the ``(rho_syco, rho_refusal,
         rho_em)`` triple alongside the residual-stream cell triples.
      4. Applies the SAME same_sign + min(|ρ|) ≥ 0.40 + perm_p ≤ 0.01
         gate as the residual-stream pass.
      5. Returns a per-predictor headline + the triple table.

    The cross-arm cell key is the same ``(point, layer, metric, variant)``
    used for the residual-stream pass; the coarse predictor adds a
    per-cell coarse-ρ column that the aggregator surfaces in the
    gate-pass list separately so the analyzer can report the headline
    coarse predictor's gate-pass / cleared status by name.
    """
    per_predictor: dict[str, Any] = {}
    for predictor in COARSE_PREDICTORS:
        syco_map = _collect_coarse_rho(syco_scoring, predictor)
        refusal_map = _collect_coarse_rho(refusal_scoring, predictor)
        em_map = _collect_coarse_rho(em_scoring, predictor)
        keys_all = set(syco_map) & set(refusal_map) & set(em_map)
        keys_any = set(syco_map) | set(refusal_map) | set(em_map)
        triples: list[dict[str, Any]] = []
        gate_pass_cells: list[dict[str, Any]] = []
        for key in sorted(keys_all):
            point, layer, metric, variant = key
            rho_syco = syco_map[key]["rho_fe_adj"]
            rho_ref = refusal_map[key]["rho_fe_adj"]
            rho_em = em_map[key]["rho_fe_adj"]
            p_syco = syco_map[key]["perm_p"]
            p_ref = refusal_map[key]["perm_p"]
            p_em = em_map[key]["perm_p"]
            gate = _same_sign(rho_syco, rho_ref, rho_em)
            abs_rhos = [abs(rho_syco), abs(rho_ref), abs(rho_em)]
            triple_payload = {
                "point": point,
                "layer": layer,
                "metric": metric,
                "variant": variant,
                "predictor": predictor,
                "rho_syco": rho_syco,
                "rho_refusal": rho_ref,
                "rho_em": rho_em,
                "perm_p_syco": p_syco,
                "perm_p_refusal": p_ref,
                "perm_p_em": p_em,
                "min_abs_rho": min(abs_rhos),
                "mean_abs_rho": sum(abs_rhos) / 3.0,
                "range_abs_rho": max(abs_rhos) - min(abs_rhos),
                "same_sign_pass": gate,
            }
            triples.append(triple_payload)
            if gate:
                gate_pass_cells.append(triple_payload)
        gate_pass_sorted = sorted(gate_pass_cells, key=lambda c: c["min_abs_rho"], reverse=True)
        cleared = [
            c
            for c in gate_pass_sorted
            if c["min_abs_rho"] >= CROSS_BEHAVIOR_THRESHOLD
            and c["perm_p_syco"] <= PERM_P_THRESHOLD
            and c["perm_p_refusal"] <= PERM_P_THRESHOLD
            and c["perm_p_em"] <= PERM_P_THRESHOLD
        ]
        per_predictor[predictor] = {
            "headline": {
                "any_cell_clears_threshold": bool(cleared),
                "threshold_min_abs_rho": CROSS_BEHAVIOR_THRESHOLD,
                "threshold_perm_p": PERM_P_THRESHOLD,
                "search_best_cell_gate_pass": gate_pass_sorted[0] if gate_pass_sorted else None,
                "search_best_cell_cleared": cleared[0] if cleared else None,
                "n_cells_total_triples": len(triples),
                "n_cells_sign_gate_pass": len(gate_pass_cells),
                "n_cells_cleared_threshold": len(cleared),
                "n_cells_incomplete_triples": len(keys_any - keys_all),
            },
            "triples": triples,
        }
    return per_predictor


def aggregate(
    syco_scoring: dict[str, Any],
    refusal_scoring: dict[str, Any],
    em_scoring: dict[str, Any],
) -> dict[str, Any]:
    """Compute the per-cell signed-ρ triple + headline. See module docstring."""
    syco_cells = {_cell_key(c): c for c in syco_scoring["cells"]}
    refusal_cells = {_cell_key(c): c for c in refusal_scoring["cells"]}
    em_cells = {_cell_key(c): c for c in em_scoring["cells"]}

    # Cells present on ALL THREE arms feed the headline; cells present on a
    # strict subset land in a separate "incomplete_triples" list.
    keys_all = set(syco_cells) & set(refusal_cells) & set(em_cells)
    keys_any = set(syco_cells) | set(refusal_cells) | set(em_cells)
    incomplete_keys = keys_any - keys_all

    triples: list[dict[str, Any]] = []
    gate_pass_cells: list[dict[str, Any]] = []
    for key in sorted(keys_all):
        point, layer, metric, variant = key
        c_syco = syco_cells[key]
        c_ref = refusal_cells[key]
        c_em = em_cells[key]
        rho_syco = _extract_rho(c_syco)
        rho_ref = _extract_rho(c_ref)
        rho_em = _extract_rho(c_em)
        p_syco = _extract_perm_p(c_syco)
        p_ref = _extract_perm_p(c_ref)
        p_em = _extract_perm_p(c_em)
        gate = _same_sign(rho_syco, rho_ref, rho_em)
        abs_rhos = [abs(rho_syco), abs(rho_ref), abs(rho_em)]
        triple_payload = {
            "point": point,
            "layer": layer,
            "metric": metric,
            "variant": variant,
            "rho_syco": rho_syco,
            "rho_refusal": rho_ref,
            "rho_em": rho_em,
            "perm_p_syco": p_syco,
            "perm_p_refusal": p_ref,
            "perm_p_em": p_em,
            "min_abs_rho": min(abs_rhos),
            "mean_abs_rho": sum(abs_rhos) / 3.0,
            "range_abs_rho": max(abs_rhos) - min(abs_rhos),
            "same_sign_pass": gate,
        }
        triples.append(triple_payload)
        if gate:
            gate_pass_cells.append(triple_payload)

    # Headline: among gate-passing cells, the one with the largest min(|ρ|)
    # AND min |ρ| >= 0.40 AND every perm_p <= 0.01.
    gate_pass_cells_sorted = sorted(gate_pass_cells, key=lambda c: c["min_abs_rho"], reverse=True)
    cleared = [
        c
        for c in gate_pass_cells_sorted
        if c["min_abs_rho"] >= CROSS_BEHAVIOR_THRESHOLD
        and c["perm_p_syco"] <= PERM_P_THRESHOLD
        and c["perm_p_refusal"] <= PERM_P_THRESHOLD
        and c["perm_p_em"] <= PERM_P_THRESHOLD
    ]
    headline = {
        "any_cell_clears_threshold": bool(cleared),
        "threshold_min_abs_rho": CROSS_BEHAVIOR_THRESHOLD,
        "threshold_perm_p": PERM_P_THRESHOLD,
        "search_best_cell_gate_pass": gate_pass_cells_sorted[0] if gate_pass_cells_sorted else None,
        "search_best_cell_cleared": cleared[0] if cleared else None,
        "n_cells_total_triples": len(triples),
        "n_cells_sign_gate_pass": len(gate_pass_cells),
        "n_cells_cleared_threshold": len(cleared),
        "n_cells_incomplete_triples": len(incomplete_keys),
    }
    # #518 v4 round-2 must-fix 1: per-coarse-predictor cross-arm triples
    # alongside the residual-stream triples. The headline coarse predictor
    # is ``completion_logprob``; the rest are surfaced for the analyzer to
    # rank against the residual-stream best cell.
    coarse_predictors = _aggregate_coarse_predictors(syco_scoring, refusal_scoring, em_scoring)
    # Surface the headline coarse predictor (completion_logprob) at the
    # top level so the analyzer's headline read does NOT have to traverse
    # the nested per-predictor dict to find it.
    completion_lp = coarse_predictors.get("completion_logprob", {})
    completion_lp_headline = completion_lp.get(
        "headline",
        {
            "any_cell_clears_threshold": False,
            "n_cells_total_triples": 0,
            "n_cells_sign_gate_pass": 0,
            "n_cells_cleared_threshold": 0,
            "n_cells_incomplete_triples": 0,
        },
    )
    return {
        "headline": headline,
        "triples": triples,
        "incomplete_triples_count": len(incomplete_keys),
        "incomplete_triples_keys_sample": [list(k) for k in sorted(incomplete_keys)[:20]],
        "coarse_predictors": coarse_predictors,
        "completion_logprob_headline": completion_lp_headline,
    }


def main() -> int:
    """Entrypoint."""
    p = argparse.ArgumentParser(
        description="#518 v4 cross-behavior aggregator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--syco-scoring",
        type=Path,
        default=REPO / "eval_results" / "issue_509" / "syco_arm" / "scoring.json",
    )
    p.add_argument(
        "--refusal-scoring",
        type=Path,
        default=REPO / "eval_results" / "issue_518" / "refusal" / "scoring.json",
    )
    p.add_argument(
        "--em-scoring",
        type=Path,
        default=REPO / "eval_results" / "issue_518" / "em" / "scoring.json",
    )
    p.add_argument(
        "--syco-bakeoff-meta",
        type=Path,
        default=None,
        help=(
            "Optional path to the syco bake-off meta.json (records model_id). "
            "Skipping is permitted only if the syco scoring predates the "
            "#518 v4 --model-id patch."
        ),
    )
    p.add_argument(
        "--refusal-bakeoff-meta",
        type=Path,
        default=None,
        help="Optional path to refusal bake-off meta.json.",
    )
    p.add_argument(
        "--em-bakeoff-meta",
        type=Path,
        default=None,
        help="Optional path to EM bake-off meta.json.",
    )
    p.add_argument(
        "--allow-legacy-syco-meta",
        action="store_true",
        help=(
            "Opt-in to the syco-arm legacy-meta carve-out per plan §10: the "
            "pre-#518-v4-patch syco bake-off predates --model-id, so its "
            "meta.json may not record model_id. With this flag, "
            "--syco-bakeoff-meta MAY be omitted; the refusal + EM arms are "
            "still strictly required. WITHOUT this flag (default), all three "
            "arms MUST pass --<arm>-bakeoff-meta with a meta.json carrying "
            "args.model_id. #518 v4 round-2 must-fix 3: the unconditional "
            "all-None shortcut that bypassed the cross-arm consistency "
            "assertion is closed."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO / "eval_results" / "issue_518" / "cross_behavior_aggregator.json",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke mode: build the three input scoring JSONs from a tiny "
            "synthetic 2-cell test set (with known signed ρ values) so the "
            "sign-gate + min(|ρ|) + headline computation can be verified "
            "without running the full per-arm scoring."
        ),
    )
    args = p.parse_args()

    if args.smoke:
        # Synthetic 2-cell, 3-arm input: cell A passes the sign gate AND
        # clears the magnitude threshold; cell B fails the sign gate.
        # #518 v4 round-2 must-fix 1: ALSO carry a stub
        # ``per_coarse_rho_fe["completion_logprob"]`` on cell A's payload
        # so the coarse-predictor headline pass can be smoke-validated end-
        # to-end without running the full #509 scoring pipeline. Cell A's
        # coarse-logprob ρ also passes the sign gate + threshold so the
        # coarse-predictor headline reports a cleared cell.
        def _stub_scoring(arm: str, rho_a: float, rho_b: float, rho_coarse_a: float) -> dict:
            return {
                "schema_version": 1,
                "arm": arm,
                "cells": [
                    {
                        "point": "end_of_system",
                        "layer": 22,
                        "metric": "gauss_kl",
                        "variant": "centered",
                        # rho_fe_adj is the #509 production key (verified by
                        # running issue509_scoring.py --arm refusal end-to-end
                        # against a stub metrics dir and inspecting the keys).
                        "rho_fe_adj": rho_a,
                        "perm_p_fe": 0.001,
                        "per_coarse_rho_fe": {
                            "completion_logprob": {
                                "rho_fe": rho_coarse_a,
                                "rho_fe_adj": rho_coarse_a,
                                "perm_p": 0.001,
                                "cluster_ci": [rho_coarse_a - 0.05, rho_coarse_a + 0.05],
                                "n_finite": 138,
                            }
                        },
                    },
                    {
                        "point": "last_prompt",
                        "layer": 14,
                        "metric": "cosine",
                        "variant": "centered",
                        "rho_fe_adj": rho_b,
                        "perm_p_fe": 0.5,
                    },
                ],
            }

        # rho_a triple (-0.50, -0.45, -0.48) PASSes sign gate + threshold.
        # rho_b triple (0.10, -0.10, 0.10) FAILs sign gate.
        # rho_coarse_a triple (-0.55, -0.42, -0.50) PASSes both gates.
        syco = _stub_scoring("syco", -0.50, 0.10, -0.55)
        refusal = _stub_scoring("refusal", -0.45, -0.10, -0.42)
        em = _stub_scoring("em", -0.48, 0.10, -0.50)
        result = aggregate(syco, refusal, em)
        out_payload = {
            "schema_version": 1,
            "smoke": True,
            "input_paths": {
                "syco": "synthetic stub",
                "refusal": "synthetic stub",
                "em": "synthetic stub",
            },
            "bakeoff_model_ids": {"syco": None, "refusal": None, "em": None},
            **result,
            "git_sha": _git_sha(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_payload, indent=2))
        print(f"WROTE {args.out}")
        print(json.dumps(out_payload["headline"], indent=2))
        # Sanity: smoke cell A MUST clear the threshold; cell B MUST fail
        # the sign gate. Validate the gate logic itself.
        h = out_payload["headline"]
        assert h["n_cells_total_triples"] == 2, h
        assert h["n_cells_sign_gate_pass"] == 1, h
        assert h["n_cells_cleared_threshold"] == 1, h
        # #518 v4 round-2 must-fix 1: the coarse-predictor (completion_logprob)
        # headline must independently clear the same gate from the stub.
        clp = out_payload["completion_logprob_headline"]
        assert clp["n_cells_total_triples"] == 1, clp
        assert clp["n_cells_sign_gate_pass"] == 1, clp
        assert clp["n_cells_cleared_threshold"] == 1, clp
        print("SMOKE assertions PASS (1 sign-gate-pass cell, 1 cleared; ")
        print("completion_logprob coarse predictor: 1 sign-pass, 1 cleared)")
        return 0

    # Production path: load + assert + aggregate.
    model_ids = _assert_model_id_consistency(
        {
            "syco": args.syco_bakeoff_meta,
            "refusal": args.refusal_bakeoff_meta,
            "em": args.em_bakeoff_meta,
        },
        allow_legacy_syco_meta=args.allow_legacy_syco_meta,
    )
    syco = _load_scoring(args.syco_scoring)
    refusal = _load_scoring(args.refusal_scoring)
    em = _load_scoring(args.em_scoring)
    result = aggregate(syco, refusal, em)
    out_payload = {
        "schema_version": 1,
        "smoke": False,
        "input_paths": {
            "syco": str(args.syco_scoring),
            "refusal": str(args.refusal_scoring),
            "em": str(args.em_scoring),
        },
        "bakeoff_model_ids": model_ids,
        **result,
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2))
    print(f"WROTE {args.out}")
    print(json.dumps(out_payload["headline"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
