#!/usr/bin/env python
# ruff: noqa: RUF002, RUF003
# Intentional scientific Unicode (Σ, ρ, δ, ŵ, ×, −, ⁻¹, ᵀ, ‖) in docstrings/comments.
"""issue #666 Phase 4 — designed-null install-leak CONTROL arm (Must-Fix 2, plan §4d/§6/§6.5).

Scores the 2 install-matched, signal-free #664 designed-null cells
(``ic_edu_default`` / ``tf_rev_default``) on the SAME broad-corpus Σc pipeline the
real arms use — adding the family-clustered 95% CI per null cell. Pre-registered
verdict (§6): a real content behavior's L̂ ρ MUST EXCEED the designed-null ρ with
non-overlapping clustered CIs for the geometry-win headline; an overlap →
"install-confounded" (L̂ tracks install-displacement magnitude, not theory
geometry).

SINGLE SOURCE OF TRUTH for the whitening (Blocker 4, plan §6 "the SAME broad-corpus
Σ_c"): the predictor (``issue666_predictor.py``) already enumerates + scores the
designed-null cells under the broad-corpus Σc⁻¹ on the production headline run, and
writes their per-cell JSON (``predictor/<cell>_predictor_cells.json``, carrying
``sigma_c_corpus_kind="broad"``). This arm READS those JSONs and adds only the
clustered CIs — so the real arm ρ (broad Σc) and the null arm ρ (broad Σc) are
computed on IDENTICAL whitening. On production it FAILS LOUD if a null cell's
predictor JSON is absent or not ``broad`` (the §6 gate would otherwise compare
mismatched whitening). The ``--in-process`` fallback re-scores in-process threading
``--sigma-inv`` (the smoke path; never the production headline).

Output: ``eval_results/issue_666/headline/designed_null_Lhat_rho.json`` (the
install-confound gate on the headline; plan §6.5 primary_deliverable). The output
carries ``sigma_c_corpus_kind`` so a parity check can assert it equals the
predictor headline's (both ``"broad"`` on production).

CPU-only; reuses the shared ``issue666_predictor`` scorer + clustered bootstrap.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "eval_results" / "issue_666" / "headline"
PRED_DIR = REPO / "eval_results" / "issue_666" / "predictor"


def _ci_from_record(rec: dict, *, n_boot: int, seed: int) -> dict:
    """Add the family-clustered CI to one predictor per-cell record.

    Reads the persisted ``per_bystander`` (Lhat, ds, context_family) + the broad-Σc
    metadata, computes the family-clustered bootstrap CI of the L̂-vs-Δs Spearman ρ,
    and returns the null-cell entry (rho + CI + the Σc-corpus kind carried through).
    """
    import issue666_predictor as pred

    pb = rec["per_bystander"]
    lh = np.array(pb["Lhat"])
    ds = np.array(pb["ds"])
    fams = np.array(pb["context_family"])
    lo, hi = pred.clustered_bootstrap_ci(
        lh, ds, clusters=fams, n_boot=n_boot, seed=seed, statistic="spearman"
    )
    return {
        "rho": rec["rho_full_Lhat"],
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_bystanders": rec["n_bystanders"],
        "behavior": rec.get("behavior"),
        "layer": rec.get("layer"),
        "sigma_c_corpus_kind": rec.get("sigma_c_corpus_kind", "unknown"),
    }


def score_designed_nulls(
    *, layer=None, n_boot=2000, seed=0, pred_dir: Path | None = None, in_process: bool = False
) -> dict:
    """Score both designed-null cells' L̂ ρ vs Δs + family-clustered CIs.

    DEFAULT (Blocker 4): read the predictor's per-cell JSONs (broad-corpus Σc — the
    SAME whitening the real arms use), add the clustered CIs. ``in_process=True``
    re-scores in-process (the smoke fallback, battery-Σc — NEVER the production
    headline). Returns ``{cell: {rho, ci_lo, ci_hi, n_bystanders, behavior, layer,
    sigma_c_corpus_kind}}`` for the 2 null cells.
    """
    import issue666_load_store as loader
    import issue666_predictor as pred

    pred_dir = PRED_DIR if pred_dir is None else Path(pred_dir)
    out: dict = {}
    if not in_process:
        for cell in pred.DESIGNED_NULL_CELLS:
            # Production-path: the predictor enumerates HF list_repo_files and writes
            # each cell's JSON by the seed-qualified LONG name (e.g.
            # ``ic_edu_default_contra_d1_seed42``), so the designed-null JSON lands
            # under the long name — NOT the short ``DESIGNED_NULL_CELLS`` prefix.
            # Resolve via the canonical short→long map from issue666_load_store; fall
            # back to the short name for smoke runs that pass --cell-names <short>.
            long_name = loader.DESIGNED_NULL_DIR.get(cell, cell)
            jp_long = pred_dir / f"{long_name}_predictor_cells.json"
            jp_short = pred_dir / f"{cell}_predictor_cells.json"
            jp = jp_long if jp_long.exists() else jp_short
            if not jp.exists():
                raise SystemExit(
                    f"designed-null arm: predictor JSON not found at {jp_long} "
                    f"(production long-name) NOR {jp_short} (smoke short-name). Run "
                    f"issue666_predictor.py (broad-corpus --sigma-inv headline) FIRST so the "
                    f"null cells are scored on the SAME broad-Σc pipeline as the real arms "
                    f"(plan §6). Use --in-process ONLY for the offline smoke."
                )
            rec = json.loads(jp.read_text())
            entry = _ci_from_record(rec, n_boot=n_boot, seed=seed)
            if entry["sigma_c_corpus_kind"] != "broad":
                raise SystemExit(
                    f"designed-null arm: {cell} predictor JSON has "
                    f"sigma_c_corpus_kind={entry['sigma_c_corpus_kind']!r}, not 'broad'. "
                    f"The §6 gate requires the null arm on the SAME broad-corpus Σc as the "
                    f"real arms — re-run the predictor headline with --sigma-inv."
                )
            out[cell] = entry
        return out

    # In-process fallback (smoke only): score with the battery-Σc diagnostic.
    import issue666_load_store as loader

    for cell in pred.DESIGNED_NULL_CELLS:
        local_dir = loader.download_cell(cell)
        loaded = loader.load_cell(local_dir)
        lyr = pred.PRIMARY_LAYER if layer is None else layer
        lyr = min(lyr, loaded["v_plus"].shape[1] - 1)
        Sigma_inv, _ = pred._battery_sigma_inv(loaded, lyr)
        rec = pred.predict_cell(loaded, cell=cell, layer=lyr, Sigma_inv=Sigma_inv)
        rec["sigma_c_corpus_kind"] = "battery-diagnostic"
        out[cell] = _ci_from_record(rec, n_boot=n_boot, seed=seed)
        del loaded
        gc.collect()
        with contextlib.suppress(OSError):
            os.remove(local_dir / "tensors.pt")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="issue 666 designed-null control arm.")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--slice", action="store_true", help="tiny smoke slice (fewer bootstraps)")
    ap.add_argument(
        "--pred-dir",
        default=str(PRED_DIR),
        help="predictor per-cell JSON dir (broad-Σc records; default eval_results/issue_666/predictor)",  # noqa: E501
    )
    ap.add_argument(
        "--in-process",
        action="store_true",
        help="re-score in-process with the battery-Σc diagnostic (SMOKE ONLY; never the headline)",
    )
    args = ap.parse_args()

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    n_boot = 200 if args.slice else args.n_boot
    nulls = score_designed_nulls(
        layer=args.layer,
        n_boot=n_boot,
        pred_dir=Path(args.pred_dir),
        in_process=args.in_process,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    # The whitening kind the null arm was scored on (broad on production; the §6
    # parity check asserts this == the predictor headline's sigma_c_corpus_kind).
    corpus_kinds = sorted({r["sigma_c_corpus_kind"] for r in nulls.values()})
    rec = {
        "designed_null_cells": list(nulls),
        "per_null": nulls,
        "sigma_c_corpus_kind": corpus_kinds[0] if len(corpus_kinds) == 1 else corpus_kinds,
        "verdict_rule": (
            "a real content behavior's L_hat rho must EXCEED the designed-null rho "
            "with non-overlapping clustered CIs for the geometry-win headline (plan §6)"
        ),
    }
    (OUT / "designed_null_Lhat_rho.json").write_text(json.dumps(rec, indent=1))
    for c, r in nulls.items():
        print(
            f"[designed-null] {c}: rho={r['rho']:.3f} CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] "
            f"(Σc={r['sigma_c_corpus_kind']})"
        )
    print("[phase=designed_null] done OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
