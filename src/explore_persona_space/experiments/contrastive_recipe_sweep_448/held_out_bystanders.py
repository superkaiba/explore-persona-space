# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
"""Task #448 Phase 0 (v5 on-policy) — held-out bystander set computation.

Plan §4.3.0. The H1b headline of the v5 on-policy re-run is the mean
bystander ΔG on personas the model NEVER trained against as contrastive
negatives in ANY of the 11 cells. The unstratified 23-bystander mean
(H1a) is biased downwards as more bystanders move into the
trained-negative set (anchor: 2, c10: 4, c11: 8 trained negatives that
sit at ΔG ≈ 0 by construction), so the H1b denominator is the actual
test of "does widening contrastive negatives GENERALIZE to bystanders
the model never saw as negatives?".

The held-out subset has two parts:

1. **12 guaranteed-held-out personas** — members of
   ``EVAL_PERSONAS_24 \\ EXTENDED_CANDIDATE_POOL \\ {villain}``. These
   CANNOT be selected by ``persona_registry.select_n_bystanders``
   regardless of SHA-256 seed because they are not in the candidate
   pool. The set, computed from current ``EVAL_PERSONAS_24`` +
   ``EXTENDED_CANDIDATE_POOL`` = ``PERSONAS.keys() ∪ {assistant,
   qwen_default}`` (12 personas), is:
       {surgeon, programmer, chef, lawyer, accountant, journalist,
        wizard, hero, philosopher, child, ai_assistant, ai}

2. **3 SHA-256-determined extra held-out personas** =
   ``EXTENDED_CANDIDATE_POOL \\ {villain} \\ c11_negatives``. With c11
   pulling 8 negatives from the 12-pool (anchor 2 + 6 SHA extras), the
   complement is 12 - 1 (villain) - 8 (c11 set) = 3 personas. The exact
   3 are pinned by ``persona_registry.select_n_bystanders``'s SHA-256
   draw, deterministic given a fixed source + CELL_SPECS.

Total expected held-out size = 15. The function asserts
``12 ≤ len(held_out) ≤ 16`` (12 = lower bound from #1; 16 = loose upper
bound future-proofing against pool changes).

Output: ``data/issue_448/held_out_bystanders.json`` with the resolved
list + provenance + assertions; per-cell sentinels read this back.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("issue_448.held_out_bystanders")

HELD_OUT_LOWER = 12
HELD_OUT_UPPER = 16


def compute_held_out_bystanders(
    eval_personas: dict[str, str],
    source: str,
    cell_specs: tuple[tuple[str, str, int, int, int, int], ...],
) -> dict[str, Any]:
    """Resolve the held-out bystander subset for the v5 on-policy headline (H1b).

    Args:
        eval_personas: Panel persona name → system prompt (the 24-eval panel).
        source: The single source persona for the sweep (e.g. ``"villain"``).
        cell_specs: The 11-cell ``CELL_SPECS`` tuple from
            ``contrastive_recipe_sweep_448.__init__``. Used to enumerate the
            per-cell training-negative sets via ``persona_registry`` (which
            must be initialized — call ``persona_registry._do_build_and_assert``
            first).

    Returns:
        ``{"held_out": [str, ...], "guaranteed_held_out": [...],
           "sha_extra_held_out": [...], "trained_negatives_union": [...],
           "n_held_out": int, "n_guaranteed": int, "n_sha_extras": int,
           "source": str, "lower_bound": 12, "upper_bound": 16}``

    Raises:
        AssertionError if the resolved set is outside ``[12, 16]``, or if
        ``EXTENDED_CANDIDATE_POOL ⊄ EVAL_PERSONAS_24`` (the held-out
        definition's load-bearing invariant — see plan Assumption 27).
    """
    # Imported inside the function so the module is import-light at module
    # load (the registry build talks to HF Hub and is slow on cold start).
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        persona_registry as registry,
    )

    if not registry.EXTENDED_CANDIDATE_POOL:
        raise RuntimeError(
            "persona_registry.EXTENDED_CANDIDATE_POOL is empty — call "
            "persona_registry._do_build_and_assert() before computing the "
            "held-out subset."
        )

    eval_set = set(eval_personas.keys())
    extended_set = set(registry.EXTENDED_CANDIDATE_POOL)
    if not extended_set.issubset(eval_set):
        missing = sorted(extended_set - eval_set)
        raise AssertionError(
            f"Assumption 27 invariant violation: EXTENDED_CANDIDATE_POOL "
            f"({sorted(extended_set)!r}) is NOT entirely contained in "
            f"EVAL_PERSONAS_24. Missing from eval panel: {missing!r}. The "
            f"held-out definition cannot be computed under this drift."
        )

    # Step 1: per-cell training-negative sets. Cells 1-9 use
    # get_anchor_bystanders(source) (the parsed-observation 2-negative set);
    # cells 10/11 use select_n_bystanders(source, N=4 or 8). Both
    # deterministic given the SHA-256 seed.
    trained_negatives_union: set[str] = set()
    per_cell_negatives: dict[str, list[str]] = {}
    for slug, _name, _pos_ex, pos_personas, _neg_ex, neg_personas in cell_specs:
        # Mirror build_training_data._negative_personas_for_cell logic exactly.
        # Cells 10 + 11 exclude positive personas; positives for cells 1-4 and
        # 7-11 are [source]; cell 5 is [source, comedian]; cell 6 is
        # [source, comedian, assistant, software_engineer].
        if neg_personas == 2:
            neg_set = registry.get_anchor_bystanders(source)
        elif neg_personas in (4, 8):
            if pos_personas == 1:
                exclude = {source}
            elif pos_personas == 2:
                # Mirror MULTI_POSITIVE_PERSONAS_C5 (defined in __init__).
                from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
                    MULTI_POSITIVE_PERSONAS_C5,
                )

                exclude = set(MULTI_POSITIVE_PERSONAS_C5)
            elif pos_personas == 4:
                from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
                    MULTI_POSITIVE_PERSONAS_C6,
                )

                exclude = set(MULTI_POSITIVE_PERSONAS_C6)
            else:
                raise ValueError(
                    f"Cell {slug!r} has pos_personas={pos_personas} which is "
                    f"not handled by the held-out resolver (expected 1, 2, or 4)."
                )
            neg_set = registry.select_n_bystanders(source, neg_personas, exclude=exclude)
        else:
            raise ValueError(
                f"Cell {slug!r} has neg_personas={neg_personas} which is "
                f"not handled by the held-out resolver (expected 2, 4, or 8)."
            )
        per_cell_negatives[slug] = list(neg_set)
        trained_negatives_union.update(neg_set)

    # Step 2: held_out = panel \ {source} \ trained_negatives_union.
    held_out = eval_set - {source} - trained_negatives_union
    held_out_sorted = sorted(held_out)

    # Step 3: split into the 12-guaranteed core and the SHA-extras complement.
    guaranteed = (eval_set - extended_set) - {source}
    guaranteed_sorted = sorted(guaranteed)
    sha_extras = held_out - guaranteed
    sha_extras_sorted = sorted(sha_extras)

    n_held_out = len(held_out_sorted)
    if not (HELD_OUT_LOWER <= n_held_out <= HELD_OUT_UPPER):
        raise AssertionError(
            f"held-out bystander count {n_held_out} not in "
            f"[{HELD_OUT_LOWER}, {HELD_OUT_UPPER}]. "
            f"guaranteed (n={len(guaranteed_sorted)})={guaranteed_sorted!r}; "
            f"sha_extras (n={len(sha_extras_sorted)})={sha_extras_sorted!r}; "
            f"trained_negatives_union={sorted(trained_negatives_union)!r}. "
            f"This is a plan §4.3.0 invariant violation — investigate "
            f"persona_registry / CELL_SPECS drift."
        )

    log.info(
        "held_out_bystanders resolved for source=%s: n=%d (guaranteed=%d, sha_extras=%d). "
        "trained_negatives_union (n=%d) = %s",
        source,
        n_held_out,
        len(guaranteed_sorted),
        len(sha_extras_sorted),
        len(trained_negatives_union),
        sorted(trained_negatives_union),
    )

    return {
        "held_out": held_out_sorted,
        "guaranteed_held_out": guaranteed_sorted,
        "sha_extra_held_out": sha_extras_sorted,
        "trained_negatives_union": sorted(trained_negatives_union),
        "per_cell_negatives": per_cell_negatives,
        "n_held_out": n_held_out,
        "n_guaranteed": len(guaranteed_sorted),
        "n_sha_extras": len(sha_extras_sorted),
        "source": source,
        "lower_bound": HELD_OUT_LOWER,
        "upper_bound": HELD_OUT_UPPER,
        "n_cells": len(cell_specs),
    }


def write_held_out_artifact(payload: dict[str, Any], out_path: Path) -> Path:
    """Write the held-out resolution to JSON for the analyzer + sentinels.

    Args:
        payload: Output of ``compute_held_out_bystanders``.
        out_path: Destination JSON path (typically
            ``data/issue_448/held_out_bystanders.json``).

    Returns:
        The written ``out_path``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    log.info("Wrote held-out bystander artifact (n=%d) → %s", payload["n_held_out"], out_path)
    return out_path


def load_held_out_artifact(in_path: Path) -> dict[str, Any]:
    """Load + light-validate a previously-written held-out artifact.

    Args:
        in_path: Path to the JSON written by ``write_held_out_artifact``.

    Returns:
        The parsed payload.

    Raises:
        FileNotFoundError if the artifact is missing.
        AssertionError on schema drift.
    """
    if not in_path.exists():
        raise FileNotFoundError(
            f"Held-out bystander artifact missing at {in_path}. Re-run Phase 0 "
            f"of dispatch_recipe_sweep_448.py to regenerate."
        )
    payload = json.loads(in_path.read_text())
    for key in ("held_out", "n_held_out", "source", "trained_negatives_union"):
        if key not in payload:
            raise AssertionError(
                f"Held-out artifact at {in_path} missing required key {key!r}. "
                f"Schema drift — regenerate via Phase 0."
            )
    return payload
