#!/usr/bin/env python
"""Issue #685 round-2 Part A — signed Δ-vs-û cosine + matched-norm random null.

Recompute the parent's Δ-vs-û projection (body Result 4) as a SIGNED cosine
``cos(Δ_l(C,b), û_resp_l(b))`` per (context, behavior, layer) over ALL 10
contexts, against the committed RESPONSE-mean û (``instruct_known_directions.pt``).
Adds a matched-norm random-direction null (B=200, seed 42, H=3584) with a per-cell
z-score, and per-(behavior, layer) sign-consistency summaries (all-10 + held-out-6).

0 GPU — pure linear algebra on committed activation tensors. NO generation, NO
judge, NO model forward pass.

Usage::

    uv run python scripts/issue685_signed_cosine_null.py --smoke   # 1-cell acceptance check (§8)
    uv run python scripts/issue685_signed_cosine_null.py           # full sweep -> JSON
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.analysis.issue685.signed_cosine import (  # noqa: E402
    B_NULL,
    BEHAVIORS,
    CONTEXTS,
    HELDOUT_CONTEXTS,
    LAYERS,
    NULL_SEED,
    SUBSET_CONTEXTS,
    H,
    aggregate,
    load_context_vectors,
    load_response_mean_u,
    mean_subtracted_signed_cos,
    null_band,
    reconstruct_delta,
    signed_cos,
    z_score,
)

OUT_DIR = Path("eval_results/issue_685/signed-cosine-matched-position-u")
OUT_PATH = OUT_DIR / "delta_vs_u_signed.json"

# Source provenance recorded IN the eval JSON metadata (plan §11; brief contract).
HYPERPARAM_SOURCES = {
    "null_B_seed_H": (
        "#685 plan v2 §6.2 + §12 (consistency_null n_perm=200, seed=42, "
        "random matched-norm unit vectors at H=3584)"
    ),
    "diff_in_means_last_prompt_token": (
        "2507.21509 (Chen et al. 2025, Persona Vectors) §3 + 2312.06681 "
        "(Panickssery et al. 2023, CAA) — diff-in-means recipe; position-agnostic "
        "by construction"
    ),
    "subset_contexts": (
        "scripts/issue685_known_directions.py:60 — the parent's response-mean û "
        "build subset, reused verbatim for apples-to-apples back-comparison"
    ),
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def _env_versions() -> dict:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
    }


def _smoke_back_compat(cv_instruct: dict, kd: dict) -> dict:
    """§8 acceptance assert: |signed_cos(Δ_assistant,syco,L14, û_resp)| ≈ committed cell.

    The committed value lives at the NESTED key path
    ``models.instruct.cells.sycophancy.14.proj_on_known_direction.per_context[ctx_index]``
    with ``ctx_index = meta.context_names.index("assistant") == 0``. Aborts (no
    fallback) on a > 1e-4 mismatch — that means the Δ reconstruction / û load /
    indexing is wrong.
    """
    metrics_path = Path("eval_results/issue_685/metrics.json")
    metrics = json.loads(metrics_path.read_text())
    context_names = metrics["models"]["instruct"]["meta"]["context_names"]
    ctx_index = context_names.index("assistant")
    committed = metrics["models"]["instruct"]["cells"]["sycophancy"]["14"][
        "proj_on_known_direction"
    ]["per_context"][ctx_index]

    u_resp = kd["directions"]["sycophancy"][14].float()
    d = reconstruct_delta(cv_instruct, "assistant", "sycophancy", 14)
    computed = abs(signed_cos(d, u_resp))
    diff = abs(computed - committed)
    passed = diff < 1e-4
    if not passed:
        raise AssertionError(
            f"smoke back-compat FAILED: |signed_cos|={computed:.8f} vs committed "
            f"{committed:.8f} (diff {diff:.2e} >= 1e-4) — Δ/û load or indexing is wrong"
        )
    return {
        "cell": "instruct/sycophancy/14/assistant",
        "expected": float(committed),
        "computed": float(computed),
        "passed": True,
    }


def _build_cells(
    cv_instruct: dict, kd: dict, contexts: list[str], behaviors: list[str], layers: list[int]
) -> tuple[list[dict], dict]:
    """Per-cell signed/abs/null/z rows + per-(behavior,layer) sign-consistency summary."""
    cells: list[dict] = []
    summary_rows: list[dict] = []
    subset_set = set(SUBSET_CONTEXTS)

    for behavior in behaviors:
        for layer in layers:
            u_resp = kd["directions"][behavior][layer].float()
            signed_all: list[float] = []
            signed_heldout: list[float] = []
            for context in contexts:
                d = reconstruct_delta(cv_instruct, context, behavior, layer)
                nb = null_band(d)
                sc = signed_cos(d, u_resp)
                ms = mean_subtracted_signed_cos(cv_instruct, context, behavior, layer, u_resp)
                in_subset = context in subset_set
                # parent_absolute_cosine cross-check (back-compat per cell, instruct)
                cells.append(
                    {
                        "context": context,
                        "behavior": behavior,
                        "layer": layer,
                        "model": "instruct",
                        "in_subset": in_subset,
                        "signed_cosine": float(sc),
                        "absolute_cosine": float(abs(sc)),
                        "mean_subtracted_signed_cosine": float(ms),
                        "null_mean": nb["mean"],
                        "null_std": nb["std"],
                        "null_iqr": nb["iqr"],
                        "z_score": z_score(sc, nb),
                    }
                )
                signed_all.append(sc)
                if context not in subset_set:
                    signed_heldout.append(sc)
            summary_rows.append(
                {
                    "behavior": behavior,
                    "layer": layer,
                    "all_10": aggregate(signed_all),
                    "heldout_6": aggregate(signed_heldout),
                }
            )
    return cells, {"per_behavior_layer": summary_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the §8 1-cell acceptance check only (no full sweep, no JSON write).",
    )
    args = parser.parse_args()

    print("[issue685.A] loading committed tensors from HF ...")
    cv_instruct = load_context_vectors("instruct")
    kd = load_response_mean_u()

    smoke_meta = _smoke_back_compat(cv_instruct, kd)
    print(
        f"[issue685.A] smoke back-compat PASS: |signed_cos|={smoke_meta['computed']:.8f} "
        f"== committed {smoke_meta['expected']:.8f}"
    )

    if args.smoke:
        # Exercise the full code path on the 1-cell slice (sign sanity + null finite).
        u_resp = kd["directions"]["sycophancy"][14].float()
        d = reconstruct_delta(cv_instruct, "assistant", "sycophancy", 14)
        sc = signed_cos(d, u_resp)
        nb = null_band(d)
        assert abs(abs(sc) - smoke_meta["computed"]) < 1e-12, (sc, smoke_meta)
        assert not np.isnan(sc), sc
        assert abs(nb["mean"]) < 0.05, nb["mean"]
        expected_std = 1.0 / np.sqrt(H)
        assert 0.7 * expected_std < nb["std"] < 1.3 * expected_std, (nb["std"], expected_std)
        assert np.isfinite(z_score(sc, nb)), nb
        print(
            f"[issue685.A] SMOKE COMPLETE: signed_cos={sc:.6f} null_mean={nb['mean']:.5f} "
            f"null_std={nb['std']:.5f} (expected ~{expected_std:.5f}) z={z_score(sc, nb):.2f}"
        )
        return

    cells, summary = _build_cells(cv_instruct, kd, CONTEXTS, BEHAVIORS, LAYERS)
    payload = {
        "schema_version": 1,
        "metadata": {
            "issue": 685,
            "followup_label": "signed-cosine-matched-position-u",
            "part": "A",
            "u_built_from": "response_mean (parent recipe; instruct_known_directions.pt)",
            "u_subset_contexts": SUBSET_CONTEXTS,
            "heldout_contexts": HELDOUT_CONTEXTS,
            "n_contexts_built_from": len(SUBSET_CONTEXTS),
            "n_contexts_applied": len(CONTEXTS),
            "null": {
                "B": B_NULL,
                "seed": NULL_SEED,
                "draw": "random unit vectors at H=3584 (np.random.default_rng)",
            },
            "behaviors": BEHAVIORS,
            "layers": LAYERS,
            "models": ["instruct"],
            "hyperparameter_sources": HYPERPARAM_SOURCES,
            "smoke_back_compat": smoke_meta,
            "code_sha": _git_commit(),
            "env": _env_versions(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "argv": sys.argv[1:],
        },
        "cells": cells,
        "summary": summary,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(
        f"[issue685.A] wrote {OUT_PATH} "
        f"({len(cells)} cells = {len(CONTEXTS)}x{len(BEHAVIORS)}x{len(LAYERS)}, "
        f"{len(summary['per_behavior_layer'])} summary rows)"
    )


if __name__ == "__main__":
    main()
