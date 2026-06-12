"""Issue #545 free-analysis follow-up: behavior-conditioned logprob-difference predictor.

Builds the explicit behavior-conditioned predictor from the design doc —
``score(cell) = base_prior_level(cell) - train_nll_surprise(cell)`` (eval-side
base prior on the eval behavior b' MINUS train-side surprise of the row's own
training data) — as a per-cell difference of two predictor JSONs already in
git, then re-runs the pre-registered scoring harness with the warmth row's
no-implant cells EXCLUDED from the scoring universe.

Protocol notes (all CPU, zero GPU):

- The new predictor lands in ``predictors/B__behavior_conditioned_logprob_diff
  .json`` over the INTERSECTION of the two parents' cell sets (the parents
  have different row coverage: base_prior covers 19 rows, train_nll_surprise
  only the 12 rows with demo corpora).
- SIGN convention: design-doc orientation "eval-side base prior on b' minus
  train-side surprise" — HIGHER = the base model already expresses b' AND the
  training data is unsurprising. ``base_prior_level`` is the base-panel column
  scalar in its native unit; ``train_nll_surprise`` is mean base NLL (nats) on
  the row's own demo answers. The raw difference mixes units; Kendall tau is
  rank-based but NOT invariant to the relative scale of the two components,
  so the surprise component (range ~ nats) dominates the ranking — recorded
  in the JSON metadata, not silently absorbed.
- Scoring writes to ``scoring_followup_bcond/`` — the original ``scoring/``
  outputs are the prereg record and are never overwritten.
- The warmth row (realized_strength ~ +-0.05 against a [0.6, 0.9] dose band on
  BOTH seeds; left unflagged by the known normalization bug) is excluded from
  the scoring universe in the same pass.
- Any quarantine read here is a LABELED after-the-fact sensitivity check: the
  quarantine split was scored exactly once under the prereg protocol; this
  pass re-reads it with a changed universe + an added candidate.

Run: ``uv run python scripts/issue545_followup_bcond_predictor.py``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.behavior_testbed_545 import (
    output_root,
    reproducibility_metadata,
)
from explore_persona_space.experiments.behavior_testbed_545.scoring import (
    _families_of,
    _family_row_bootstrap,
    _load_predictors,
    _seed_mean_targets,
    _z_norm_within_column,
    score,
    weighted_kendall_tau,
)

logger = logging.getLogger(__name__)

EXCLUDED_ROWS = frozenset({"warmth"})
NEW_PREDICTOR = "B__behavior_conditioned_logprob_diff"
OUT_DIR_NAME = "scoring_followup_bcond"
PROTOCOL_NOTE = (
    "AFTER-THE-FACT follow-up pass (free-analysis, Step 9a-ter) — NOT part of the "
    "pre-registered protocol. Deviations from the prereg record (scoring/): (1) adds the "
    "behavior-conditioned logprob-difference predictor to the Group B candidate pool; "
    "(2) excludes the warmth row's no-implant cells (realized_strength ~0 vs dose band "
    "[0.6,0.9], unflagged by the known normalization bug) from the scoring universe; "
    "(3) the quarantine split was already scored once under the prereg protocol, so every "
    "quarantine number in this file is a labeled sensitivity check, not a confirmatory read."
)


def build_predictor(pred_dir: Path) -> Path:
    """Write B__behavior_conditioned_logprob_diff.json = base_prior - train_surprise.

    Per-cell difference over the intersection of the two parents' cell sets;
    returns the written path and asserts non-trivial coverage.
    """
    base_prior = json.loads((pred_dir / "B__base_prior_level.json").read_text())
    surprise = json.loads((pred_dir / "B__train_nll_surprise.json").read_text())
    shared = sorted(set(base_prior["cells"]) & set(surprise["cells"]))
    assert len(shared) >= 20, f"implausibly small parent intersection: {len(shared)} cells"
    cells = {c: float(base_prior["cells"][c]) - float(surprise["cells"][c]) for c in shared}
    rows = sorted({c.split("|")[0] for c in shared})
    cols = sorted({c.split("|")[1] for c in shared})
    payload = {
        "group": "B",
        "name": "behavior_conditioned_logprob_diff",
        "track": "shift",
        "cells": cells,
        "orientation": (
            "base_prior_level MINUS train_nll_surprise (design-doc form: eval-side base "
            "prior on b' minus train-side surprise). base_prior_level: base-panel column "
            "scalar under the default context, native unit, track 'both' (column-side "
            "broadcast). train_nll_surprise: mean base NLL in nats on the row's own demo "
            "answers, track 'shift' (row-side broadcast). HIGHER = base already likely to "
            "express b' AND training data unsurprising. The two components are differenced "
            "in their native units (no per-component standardization): the surprise "
            "component's spread (nats) dominates the cross-cell ranking."
        ),
        "coverage": {
            "n_cells": len(shared),
            "n_rows": len(rows),
            "n_cols": len(cols),
            "rows": rows,
            "cols": cols,
            "parent_n_cells": {
                "B__base_prior_level": len(base_prior["cells"]),
                "B__train_nll_surprise": len(surprise["cells"]),
            },
            "note": (
                "intersection of the parents' cell sets; base_prior spans 19 rows x 14 "
                "cols, train_nll_surprise 12 rows x 15 cols (rows without demo corpora "
                "are missing from the surprise side; the marker column is missing from "
                "the base-prior side)"
            ),
        },
        "metadata": reproducibility_metadata(),
    }
    p = pred_dir / f"{NEW_PREDICTOR}.json"
    p.write_text(json.dumps(payload, indent=1))
    logger.info("[phase=predictor] wrote %s (%d cells)", p, len(shared))
    return p


def per_predictor_reads(out_dir: Path) -> dict:
    """Selection-free per-predictor reads on the warmth-excluded universe.

    For the new predictor and the two reference champions (existing
    behavior-native + geometry dev champions from the prereg record), computes
    dev pooled tau, per-family-fold LOFO taus (a single fixed predictor needs
    no nested selection — the fold read IS the held-out read), and the LABELED
    after-the-fact quarantine sensitivity tau.
    """
    out_root = output_root()
    prereg = json.loads((out_root / "preregistration.json").read_text())
    matrix = json.loads((out_root / "L_matrix.json").read_text())["cells"]
    metadata = json.loads((out_root / "cell_metadata.json").read_text())["cells"]
    preds = _load_predictors(out_root / "predictors")

    split = prereg["quarantine_split"]
    dev_cells = ["|".join(c) for c in split["development_cells"]]
    quarantine = ["|".join(c) for c in split["sampled_quarantined_cells"]] + [
        "|".join(c) for c in split["family_quarantined_cells"]
    ]

    targets_raw = _seed_mean_targets(matrix, metadata, include_flagged=False)
    targets_raw = {k: v for k, v in targets_raw.items() if k.split("|")[0] not in EXCLUDED_ROWS}
    vals = {k: v["shift"] for k, v in targets_raw.items() if v.get("shift") is not None}
    target = _z_norm_within_column(vals)
    dev = [c for c in dev_cells if c in target]
    quar = [c for c in quarantine if c in target]

    reference = {
        "behavior_native_champion": "B__demo_nll_transfer",
        "geometry_champion": "A__geom_nl_L15_last_token_cosine",
    }
    out: dict = {
        "universe": {
            "excluded_rows": sorted(EXCLUDED_ROWS),
            "n_dev_with_target": len(dev),
            "n_quarantine_with_target": len(quar),
        },
        "reference_champions": reference,
        "predictors": {},
    }
    for name in (NEW_PREDICTOR, *reference.values()):
        d = preds.get(name)
        if d is None:
            out["predictors"][name] = {"error": "predictor JSON not found"}
            continue
        fold_taus = []
        for fam, fold_cells in sorted(_families_of(dev).items()):
            tau = weighted_kendall_tau(d["cells"], target, fold_cells)
            if tau is not None:
                fold_taus.append({"family": fam, "tau": round(tau, 4)})
        taus = [f["tau"] for f in fold_taus]
        out["predictors"][name] = {
            "dev_pooled_tau": weighted_kendall_tau(d["cells"], target, dev),
            "lofo_per_fold": fold_taus,
            "lofo_mean_tau": round(float(np.mean(taus)), 4) if taus else None,
            "quarantine_sensitivity_tau_AFTER_THE_FACT": weighted_kendall_tau(
                d["cells"], target, quar
            ),
        }

    # Behavior-native-vs-geometry quarantine margin under warmth exclusion, with
    # the same hierarchical family->row cluster bootstrap the prereg H2 uses.
    # LABELED after-the-fact: the prereg H2 selection (B-vs-C on dev tau) flips
    # under this universe, so the B-vs-A margin here is the UNSELECTED read the
    # parent headline tracks, not the protocol's confirmatory number.
    b_name, a_name = reference["behavior_native_champion"], reference["geometry_champion"]
    if b_name in preds and a_name in preds:

        def _b_vs_a_margin(cells_subset: list[str]) -> float | None:
            tb = weighted_kendall_tau(preds[b_name]["cells"], target, cells_subset)
            ta = weighted_kendall_tau(preds[a_name]["cells"], target, cells_subset)
            return tb - ta if tb is not None and ta is not None else None

        point = _b_vs_a_margin(quar)
        boot = _family_row_bootstrap(quar, _b_vs_a_margin)
        out["b_vs_a_quarantine_margin_AFTER_THE_FACT"] = {
            "behavior_native": b_name,
            "geometry": a_name,
            "point": round(point, 4) if point is not None else None,
            "family_row_clustered_bootstrap_ci95": boot["ci95"] if boot else None,
            "n_bootstrap_valid": boot["n_valid"] if boot else 0,
        }
    p = out_dir / "followup_summary.json"
    p.write_text(
        json.dumps(
            {"protocol_note": PROTOCOL_NOTE, **out, "metadata": reproducibility_metadata()},
            indent=1,
        )
    )
    logger.info("[phase=summary] wrote %s", p)
    return out


def main() -> None:
    """Build the predictor, run the warmth-excluded race, write + print the summary."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pred_dir = output_root() / "predictors"
    build_predictor(pred_dir)

    results_path = score(
        exclude_rows=EXCLUDED_ROWS, out_dir_name=OUT_DIR_NAME, protocol_note=PROTOCOL_NOTE
    )
    results = json.loads(results_path.read_text())
    summary = per_predictor_reads(results_path.parent)

    track = results["tracks"]["shift"]
    lb = track["dev_leaderboard"]
    rank = list(lb).index(NEW_PREDICTOR) + 1 if NEW_PREDICTOR in lb else None
    print("\n=== follow-up headline (warmth-excluded universe) ===")
    print(f"universe_filter: {results.get('universe_filter')}")
    print(f"group_k: {results['group_k']}")
    print(f"new predictor dev tau: {lb.get(NEW_PREDICTOR)} (leaderboard rank {rank}/{len(lb)})")
    for name, d in summary["predictors"].items():
        print(
            f"  {name}: dev {d.get('dev_pooled_tau')}, LOFO mean {d.get('lofo_mean_tau')} over "
            f"{len(d.get('lofo_per_fold', []))} folds, quarantine (after-the-fact sensitivity) "
            f"{d.get('quarantine_sensitivity_tau_AFTER_THE_FACT')}"
        )
    cv = track["leave_family_out_cv"]
    print(f"nested-CV group means: { {g: d.get('mean_tau') for g, d in cv.items()} }")
    frozen = track["quarantine_frozen_champions"]
    print(
        "quarantine frozen champions (after-the-fact sensitivity): "
        f"{ {g: (d.get('champion'), d.get('tau')) for g, d in frozen.items()} }"
    )
    print(f"h2 margin (B/C-vs-A, warmth-excluded): {track.get('h2_margin')}")
    print(f"h2 margins unselected: {track.get('h2_margins_unselected')}")
    print(
        "B-vs-A quarantine margin (after-the-fact sensitivity): "
        f"{summary.get('b_vs_a_quarantine_margin_AFTER_THE_FACT')}"
    )
    print(f"wrote {results_path}")
    print("[phase=done]")


if __name__ == "__main__":
    main()
