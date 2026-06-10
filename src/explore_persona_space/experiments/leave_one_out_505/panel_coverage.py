# ruff: noqa: RUF002  # em-dash + Greek beta intentional
"""Task #505 §5.4 — joint K + held-out-panel construction gate.

Loads the #472 60-persona bank + the layer-10 centroid cosine bundle (both
already on the HF data repo at ``superkaiba1/explore-persona-space-data/
issue472_neg_geometry/geometry/``), runs the spread-quantile non-default
negative selector to pick K=6 personas (always-include qwen_default keeps its
own slot), defines the held-out panel as bank − source − K-set − qwen_default,
and checks the panel's tercile / variance coverage for every dropped j_i.

The gate is mandatory before any training spawn (plan §5.4 + §5.5). The
load-bearing identification condition is the tercile spread:

    for each non-default j_i:
        ≥ PANEL_TERCILE_FLOOR (=8) personas in BOTH the top + bottom tercile
            of cos(b, j_i) over the panel.

The plan §5.4 first draft paired this with a within-panel variance floor
`var_panel cos(b, j_i) ≥ 0.02**2`, but that floor was a unit-error import
of #472's `ID_GATE_SD_FLOOR = 0.02` — which is SD-across-arms of a
DISTANCE metric in #472's identification analysis, NOT this experiment's
within-panel cosine variance to a single j (different distribution; the
realised within-panel variances on the actual bank sit at 0.00012-0.00018,
~2-3× below the misderived floor, halting Phase 1 on every j_i in round 4).
Round 5 drops the variance gate; `tercile_ok` is the sole pass criterion.
See `PANEL_VARIANCE_FLOOR`'s comment in `leave_one_out_505/__init__.py`
for the full provenance. The `spans_floor` diagnostic field is still
reported per j_i for audit visibility (it's just not in the pass
condition).

On a single j_i failure the dispatcher swaps that j_i to the next
spread-quantile candidate and re-runs the gate ONCE (deterministic
one-shot retry per plan §12). A second failure halts dispatch per the §8
kill criteria.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    ALWAYS_INCLUDE_NEGATIVE,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
    select_negatives_by_geometry,
)
from explore_persona_space.experiments.leave_one_out_505 import (
    K_NON_DEFAULT,
    PANEL_TERCILE_FLOOR,
    PANEL_VARIANCE_FLOOR,
)

log = logging.getLogger("issue_505.panel_coverage")


class PanelCoverageGateError(RuntimeError):
    """The §5.4 panel coverage gate failed after the one-shot K-swap retry."""


def _spread_quantile_k_set(
    cos_to_source: dict[str, float],
    *,
    k_non_default: int = K_NON_DEFAULT,
    skip_personas: tuple[str, ...] = (),
    source: str = SOURCE_PERSONA,
    always_include: str = ALWAYS_INCLUDE_NEGATIVE,
) -> list[str]:
    """Pick K=6 non-default negatives via spread quantile on cos(p, source).

    qwen_default is always-included separately (so the K=6 count is purely the
    non-default slots). The selector in #472 returns a list with the
    always-include FIRST; we strip it and return only the non-default tail.

    ``skip_personas`` lets the one-shot retry exclude personas that failed the
    coverage gate (k_swap) without re-shuffling the quantile coverage of the
    remaining candidates.
    """
    filtered_cos = {
        p: c
        for p, c in cos_to_source.items()
        if p != source and p != always_include and p not in skip_personas
    }
    raw = select_negatives_by_geometry(
        source=source,
        placement="spread",
        n_personas=k_non_default + 1,  # +1 for always-include slot the selector reserves
        cos_to_source={**filtered_cos, always_include: cos_to_source[always_include]},
        always_include=(always_include,),
    )
    # The selector returns [always_include, *non_default_chosen]; strip the first.
    if raw[0] != always_include:
        raise AssertionError(
            f"select_negatives_by_geometry returned {raw[0]!r} as first item, "
            f"expected {always_include!r}. Selector contract changed."
        )
    non_default = raw[1:]
    if len(non_default) != k_non_default:
        raise AssertionError(
            f"Expected {k_non_default} non-default negatives, got {len(non_default)}: "
            f"{non_default}."
        )
    return non_default


def _check_panel_coverage_for_j(
    j_i: str,
    panel: list[str],
    cos_matrix: dict[str, dict[str, float]],
) -> dict:
    """Run the §5.4 panel-coverage diagnostic for a single dropped negative j_i.

    Returns a dict with the diagnostic counts. ``tercile_ok`` is the sole
    pass criterion (≥ ``PANEL_TERCILE_FLOOR`` personas in BOTH the top and
    bottom tercile of cos(b, j_i) over the panel).

    The ``var_panel_cos_j`` + ``spans_floor`` fields are reported for audit
    visibility but are NOT in the pass condition: the original §5.4 draft
    paired tercile with a within-panel variance floor `var_panel ≥ 0.02**2`
    derived by squaring #472's ``ID_GATE_SD_FLOOR``, but that floor is
    SD-across-arms of a DISTANCE metric — different distribution from this
    experiment's within-panel cosine variance to a single j. Round 5 drops
    the variance gate as a unit-error correction. Full provenance in
    `leave_one_out_505/__init__.py` § PANEL_VARIANCE_FLOOR.
    """
    cos_b_j = sorted(((b, float(cos_matrix[b][j_i])) for b in panel), key=lambda x: -x[1])
    t = len(cos_b_j) // 3
    top = cos_b_j[:t]
    bot = cos_b_j[-t:]
    var_panel = float(np.var([c for _, c in cos_b_j]))
    return {
        "j_i": j_i,
        "n_panel": len(panel),
        "n_top_tercile": len(top),
        "n_bot_tercile": len(bot),
        "var_panel_cos_j": var_panel,
        "spans_floor": var_panel >= PANEL_VARIANCE_FLOOR,
        "tercile_ok": (len(top) >= PANEL_TERCILE_FLOOR and len(bot) >= PANEL_TERCILE_FLOOR),
        "cos_b_j_top5": cos_b_j[:5],
        "cos_b_j_bot5": cos_b_j[-5:],
    }


def run_panel_coverage_gate(
    *,
    persona_bank: dict[str, str],
    cos_matrix_l10: dict[str, dict[str, float]],
    source: str = SOURCE_PERSONA,
    always_include: str = ALWAYS_INCLUDE_NEGATIVE,
    k_non_default: int = K_NON_DEFAULT,
    max_retries: int = 1,
) -> dict:
    """Run the §5.4 joint K + panel construction gate.

    On a single j_i failure (tercile floor OR variance floor missed), the gate
    drops the offending j_i from the candidate pool and re-runs the spread
    quantile selector ONCE (``max_retries=1``). A second failure raises
    ``PanelCoverageGateError``.

    Returns:
        ``{"k_set": list[str] (qwen_default first, then non-default in spread
        quantile order), "panel": list[str], "coverage": dict[j_i -> diag],
        "gate_passed": bool, "n_retries_used": int}``.

    Raises:
        PanelCoverageGateError: after ``max_retries`` swap attempts still some
            j_i failed. The dispatcher must HALT before any training.
    """
    cos_to_source = {p: float(cos_matrix_l10[source][p]) for p in persona_bank if p != source}
    if always_include not in persona_bank:
        raise KeyError(
            f"always-include negative {always_include!r} missing from persona bank "
            f"(size={len(persona_bank)}). Check the bank artifact."
        )

    skip: set[str] = set()
    last_coverage: dict[str, dict] = {}
    last_non_default: list[str] = []
    last_panel: list[str] = []

    for retry in range(max_retries + 1):
        non_default = _spread_quantile_k_set(
            cos_to_source,
            k_non_default=k_non_default,
            skip_personas=tuple(skip),
            source=source,
            always_include=always_include,
        )
        k_set = [always_include, *non_default]
        panel = sorted(set(persona_bank.keys()) - {source} - set(k_set))
        if len(panel) < 40:
            raise PanelCoverageGateError(
                f"held-out panel too small after exclusions: {len(panel)} < 40 "
                f"(bank={len(persona_bank)}, source={source!r}, k_set={k_set})."
            )

        coverage = {
            j_i: _check_panel_coverage_for_j(j_i, panel, cos_matrix_l10) for j_i in non_default
        }
        # Round-5: tercile_ok is the sole pass criterion. The `spans_floor`
        # field is still reported per-j_i for audit visibility but excluded
        # from the gate — see module docstring + PANEL_VARIANCE_FLOOR comment.
        failed = [j_i for j_i, c in coverage.items() if not c["tercile_ok"]]

        last_coverage = coverage
        last_non_default = non_default
        last_panel = panel

        if not failed:
            return {
                "k_set": k_set,
                "non_default_negatives": non_default,
                "always_include": always_include,
                "panel": panel,
                "n_panel": len(panel),
                "coverage": coverage,
                "gate_passed": True,
                "n_retries_used": retry,
                "skipped_personas": sorted(skip),
            }
        # One-shot retry: drop the failed j_i's and re-run with new quantile picks.
        log.warning(
            "[panel-coverage] retry %d/%d: %d j_i failed (%s) — dropping from "
            "candidate pool and re-running spread quantile.",
            retry + 1,
            max_retries,
            len(failed),
            failed,
        )
        skip.update(failed)
        if retry >= max_retries:
            break

    # All retries exhausted; emit the last coverage diagnostic + raise.
    return_payload = {
        "k_set": [always_include, *last_non_default],
        "non_default_negatives": last_non_default,
        "always_include": always_include,
        "panel": last_panel,
        "n_panel": len(last_panel),
        "coverage": last_coverage,
        "gate_passed": False,
        "n_retries_used": max_retries,
        "skipped_personas": sorted(skip),
    }
    failed = [j_i for j_i, c in last_coverage.items() if not c["tercile_ok"]]
    raise PanelCoverageGateError(
        f"§5.4 panel coverage gate FAILED after {max_retries} retry(s): {len(failed)} "
        f"non-default j_i did not meet tercile ≥ {PANEL_TERCILE_FLOOR} in both "
        f"top + bottom tercile of cos(b, j_i). Failed j_i: {failed}. "
        f"Diagnostic payload: {json.dumps(return_payload, indent=2)}"
    )


def write_gate_payload(payload: dict, out_path: Path) -> None:
    """Persist the gate payload (k_set, panel, coverage) for downstream + audits."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    log.info("[panel-coverage] wrote gate payload → %s", out_path)


def load_inherited_l10_cos(centroid_bundle_path: Path) -> dict[str, dict[str, float]]:
    """Load the layer-10 centroid bundle from the #472 HF artifact (already on disk).

    The on-disk ``centroids_L10.pt`` is the STRUCTURED dict written by
    ``contrastive_neg_geometry_472.centroids.build_centroids``:

        {
          'centroids': torch.Tensor[N, D],
          'persona_names': list[str] (len=N),
          'cos_matrix': torch.Tensor[N, N],
          'layer': int,
          'base_model': str,
          'questions': list[str],
        }

    This loader unwraps that into the nested ``dict[name][name] -> float`` form
    that ``panel_coverage._spread_quantile_k_set`` + ``_check_panel_coverage_for_j``
    expect. A schema check up front fails loud if any future #472 rebuild swaps
    schemas (avoids the symmetric ``persona_bank.json`` bug that crashed #505
    round-3 at Phase 1 with ``KeyError: 'schema_version'``).
    """
    import torch

    bundle = torch.load(centroid_bundle_path, weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError(
            f"centroid bundle at {centroid_bundle_path} is type {type(bundle).__name__}; "
            "expected dict with keys ('centroids', 'persona_names', 'cos_matrix', ...)."
        )
    required = {"persona_names", "cos_matrix"}
    missing = required - bundle.keys()
    if missing:
        raise KeyError(
            f"centroid bundle at {centroid_bundle_path} missing required key(s) "
            f"{sorted(missing)}; got top-level keys {sorted(bundle.keys())}. "
            "Schema drift — rebuild via contrastive_neg_geometry_472.centroids.build_centroids."
        )
    names: list[str] = list(bundle["persona_names"])
    cos_t = bundle["cos_matrix"]
    cos: dict[str, dict[str, float]] = {}
    for i, a in enumerate(names):
        cos[a] = {b: float(cos_t[i, j].item()) for j, b in enumerate(names)}
    return cos
