#!/usr/bin/env python
"""Issue #685 round-2 Part B — matched-position û + signed/abs cosine + back-comparison.

Build the matched-position û at the LAST-PROMPT-TOKEN slot via the persona-vectors
diff-in-means recipe (subset-context mean of Δ from the committed
``context_vectors.pt``) for BOTH instruct and base — NO forward pass. Then
recompute SIGNED + ABSOLUTE cosine of Δ against û_match over all 10 contexts, with
the same matched-norm random null as Part A. For instruct, cell-by-cell back-compare
against the parent's RESPONSE-mean û (delta_signed / delta_absolute) and report
``cos(û_match, û_resp)`` per (behavior, layer).

0 GPU — pure linear algebra on committed activation tensors.

Outputs:
  - eval_results/issue_685/signed-cosine-matched-position-u/delta_vs_u_matched_position.json
  - .../u_last_prompt_token/u_l<layer>_<behavior>_<model>.npy   (24 instruct + 24 base = 48)
  - .../u_last_prompt_token/u_match_all_<model>.pt              (consolidated dict per model)
Then uploads the û tensors + this JSON to HF
``issue685_context_shift/signed_cosine_matched_position_u/``.

Usage::

    uv run python scripts/issue685_matched_position_u.py --smoke        # §8 acceptance check
    uv run python scripts/issue685_matched_position_u.py                # full sweep + HF upload
    uv run python scripts/issue685_matched_position_u.py --no-upload    # full sweep, skip HF upload
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
    REPO_ID,
    SUBSET_CONTEXTS,
    H,
    aggregate,
    load_context_vectors,
    load_response_mean_u,
    matched_position_u,
    mean_subtracted_signed_cos,
    null_band,
    reconstruct_delta,
    save_npy,
    signed_cos,
    z_score,
)

OUT_DIR = Path("eval_results/issue_685/signed-cosine-matched-position-u")
U_DIR = OUT_DIR / "u_last_prompt_token"
OUT_PATH = OUT_DIR / "delta_vs_u_matched_position.json"
HF_PIR = "issue685_context_shift/signed_cosine_matched_position_u"

HYPERPARAM_SOURCES = {
    "null_B_seed_H": (
        "#685 plan v2 §6.2 + §12 (consistency_null n_perm=200, seed=42, "
        "random matched-norm unit vectors at H=3584)"
    ),
    "diff_in_means_last_prompt_token": (
        "2507.21509 (Chen et al. 2025, Persona Vectors) §3 + 2312.06681 "
        "(Panickssery et al. 2023, CAA) — diff-in-means recipe; position-agnostic "
        "by construction (read at the matched last-prompt-token slot)"
    ),
    "subset_contexts": (
        "scripts/issue685_known_directions.py:60 — û_match BUILT from the same "
        "4-context subset as the parent's response-mean û (apples-to-apples)"
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


def _build_u_match(cv: dict, model_tag: str, write_npy: bool) -> dict[str, dict[int, torch.Tensor]]:
    """û_match per (behavior, layer) for one model; optionally persist .npy + consolidated .pt."""
    u_match: dict[str, dict[int, torch.Tensor]] = {}
    for behavior in BEHAVIORS:
        u_match[behavior] = {}
        for layer in LAYERS:
            u = matched_position_u(cv, behavior, layer)
            u_match[behavior][layer] = u
            if write_npy:
                save_npy(u, U_DIR / f"u_l{layer}_{behavior}_{model_tag}.npy")
    if write_npy:
        U_DIR.mkdir(parents=True, exist_ok=True)
        consolidated = {
            "directions": u_match,
            "metadata": {
                "task": 685,
                "followup_label": "signed-cosine-matched-position-u",
                "model_tag": model_tag,
                "recipe": (
                    "matched_position last-prompt-token diff-in-means "
                    "(persona-vectors recipe at the last-prompt-token slot)"
                ),
                "subset_contexts": SUBSET_CONTEXTS,
                "behavior_names": BEHAVIORS,
                "layers": LAYERS,
                "hidden_dim": H,
                "code_sha": _git_commit(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
        }
        torch.save(consolidated, U_DIR / f"u_match_all_{model_tag}.pt")
    return u_match


def _build_cells(
    cv: dict, model_tag: str, u_match: dict, kd: dict | None
) -> tuple[list[dict], dict]:
    """Per-cell matched-position signed/abs/null/z rows + back-comparison (instruct only)."""
    cells: list[dict] = []
    summary_rows: list[dict] = []
    subset_set = set(SUBSET_CONTEXTS)

    for behavior in BEHAVIORS:
        for layer in LAYERS:
            u_m = u_match[behavior][layer]
            u_resp = kd["directions"][behavior][layer].float() if kd is not None else None
            cos_um_ur = float(signed_cos(u_m, u_resp)) if u_resp is not None else None

            signed_match_all: list[float] = []
            signed_match_heldout: list[float] = []
            delta_signed_all: list[float] = []
            delta_signed_heldout: list[float] = []
            delta_abs_all: list[float] = []
            delta_abs_heldout: list[float] = []

            for context in contexts_for(model_tag):
                d = reconstruct_delta(cv, context, behavior, layer)
                nb = null_band(d)
                sc = signed_cos(d, u_m)
                ms = mean_subtracted_signed_cos(cv, context, behavior, layer, u_m)
                in_subset = context in subset_set
                row = {
                    "context": context,
                    "behavior": behavior,
                    "layer": layer,
                    "model": model_tag,
                    "in_subset": in_subset,
                    "signed_cosine": float(sc),
                    "absolute_cosine": float(abs(sc)),
                    "mean_subtracted_signed_cosine": float(ms),
                    "null_mean": nb["mean"],
                    "null_std": nb["std"],
                    "null_iqr": nb["iqr"],
                    "z_score": z_score(sc, nb),
                }
                if u_resp is not None:
                    resp_signed = signed_cos(d, u_resp)
                    row["delta_signed_vs_response_u"] = float(sc - resp_signed)
                    row["delta_absolute_vs_response_u"] = float(abs(sc) - abs(resp_signed))
                    row["resp_mean_signed_cosine"] = float(resp_signed)
                    row["resp_mean_absolute_cosine"] = float(abs(resp_signed))
                    delta_signed_all.append(sc - resp_signed)
                    delta_abs_all.append(abs(sc) - abs(resp_signed))
                    if not in_subset:
                        delta_signed_heldout.append(sc - resp_signed)
                        delta_abs_heldout.append(abs(sc) - abs(resp_signed))
                else:
                    # base has no response-mean û to compare against (§3.2)
                    row["delta_signed_vs_response_u"] = None
                    row["delta_absolute_vs_response_u"] = None
                cells.append(row)
                signed_match_all.append(sc)
                if not in_subset:
                    signed_match_heldout.append(sc)

            srow: dict = {
                "behavior": behavior,
                "layer": layer,
                "model": model_tag,
                "matched_position": {
                    "all_10": aggregate(signed_match_all),
                    "heldout_6": aggregate(signed_match_heldout),
                },
            }
            if u_resp is not None:
                srow["cos_umatch_uresp"] = cos_um_ur
                srow["matched_minus_response"] = {
                    "all_10": {
                        "mean_delta_signed": float(np.mean(delta_signed_all)),
                        "mean_delta_absolute": float(np.mean(delta_abs_all)),
                        "frac_matched_abs_higher": float(np.mean(np.asarray(delta_abs_all) > 0)),
                    },
                    "heldout_6": {
                        "mean_delta_signed": float(np.mean(delta_signed_heldout)),
                        "mean_delta_absolute": float(np.mean(delta_abs_heldout)),
                        "frac_matched_abs_higher": float(
                            np.mean(np.asarray(delta_abs_heldout) > 0)
                        ),
                    },
                }
            summary_rows.append(srow)
    return cells, {"per_behavior_layer": summary_rows}


def contexts_for(model_tag: str) -> list[str]:
    """Both models project û_match onto all 10 contexts (plan §3.2)."""
    return CONTEXTS


def _upload_to_hf() -> None:
    """Upload the û tensors (.npy + .pt) and both JSONs to the HF data repo.

    Fail-loud: a non-zero HfApi error propagates. Intermediate analysis tensors
    the analyzer reads downstream MUST land on HF before upload-verification
    (CLAUDE.md Upload Policy).
    """
    from huggingface_hub import HfApi

    api = HfApi()
    # The û tensors (analysis tensors) + the matched-position JSON go up. The
    # Part-A JSON (delta_vs_u_signed.json) is uploaded too if present (committed
    # to git is its primary home; HF is the convenience mirror for the analyzer).
    targets: list[tuple[Path, str]] = []
    for p in sorted(U_DIR.glob("*.npy")):
        targets.append((p, f"{HF_PIR}/u_last_prompt_token/{p.name}"))
    for p in sorted(U_DIR.glob("*.pt")):
        targets.append((p, f"{HF_PIR}/u_last_prompt_token/{p.name}"))
    if OUT_PATH.exists():
        targets.append((OUT_PATH, f"{HF_PIR}/{OUT_PATH.name}"))
    signed_json = OUT_DIR / "delta_vs_u_signed.json"
    if signed_json.exists():
        targets.append((signed_json, f"{HF_PIR}/{signed_json.name}"))

    from huggingface_hub import CommitOperationAdd

    ops = [
        CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(local))
        for local, path_in_repo in targets
    ]
    api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=ops,
        commit_message="issue685 round-2: matched-position û tensors + signed-cosine JSONs",
    )
    # Verify the prefix landed.
    from huggingface_hub import list_repo_files

    listed = set(list_repo_files(REPO_ID, repo_type="dataset", revision="main"))
    for _, path_in_repo in targets:
        assert path_in_repo in listed, f"HF upload missing: {path_in_repo}"
    print(f"[issue685.B] uploaded + verified {len(targets)} files to HF {HF_PIR}/")


def _smoke(cv_instruct: dict, kd: dict) -> None:
    """§8 acceptance check on the 1-cell slice (sycophancy, L14, assistant)."""
    u_resp = kd["directions"]["sycophancy"][14].float()
    u_m = matched_position_u(cv_instruct, "sycophancy", 14)
    d = reconstruct_delta(cv_instruct, "assistant", "sycophancy", 14)

    # 1. back-compat: |signed_cos(Δ, û_resp)| ≈ 0.10130018
    sc_resp = abs(signed_cos(d, u_resp))
    assert abs(sc_resp - 0.10130018) < 1e-4, sc_resp
    # 2. cos(û_match, û_resp) ≈ 0.117 ± 0.01
    cos_uu = signed_cos(u_m, u_resp)
    assert abs(cos_uu - 0.117) < 0.01, cos_uu
    # 3. norms: ||û_match|| ≈ 25.6 ± 0.1, ||û_resp|| ≈ 6.9 ± 0.1
    nm_match = float(u_m.norm())
    nm_resp = float(u_resp.norm())
    assert abs(nm_match - 25.6) < 0.1, nm_match
    assert abs(nm_resp - 6.9) < 0.1, nm_resp
    # 4. null stats finite
    nb = null_band(d)
    assert abs(nb["mean"]) < 0.05, nb["mean"]
    expected_std = 1.0 / np.sqrt(H)
    assert 0.7 * expected_std < nb["std"] < 1.3 * expected_std, (nb["std"], expected_std)
    assert np.isfinite(z_score(signed_cos(d, u_m), nb)), nb
    print(
        f"[issue685.B] SMOKE COMPLETE: |sc_resp|={sc_resp:.6f} cos(uM,uR)={cos_uu:.4f} "
        f"||uM||={nm_match:.2f} ||uR||={nm_resp:.2f} null_std={nb['std']:.5f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="§8 1-cell acceptance check only.")
    parser.add_argument("--no-upload", action="store_true", help="skip the HF upload step.")
    args = parser.parse_args()

    print("[issue685.B] loading committed tensors from HF ...")
    cv_instruct = load_context_vectors("instruct")
    kd = load_response_mean_u()

    if args.smoke:
        _smoke(cv_instruct, kd)
        return

    cv_base = load_context_vectors("base")

    # Build + persist û_match for both models.
    u_match_instruct = _build_u_match(cv_instruct, "instruct", write_npy=True)
    u_match_base = _build_u_match(cv_base, "base", write_npy=True)

    cells_instruct, summary_instruct = _build_cells(cv_instruct, "instruct", u_match_instruct, kd)
    cells_base, summary_base = _build_cells(cv_base, "base", u_match_base, None)

    cells = cells_instruct + cells_base
    summary = {
        "per_behavior_layer": summary_instruct["per_behavior_layer"]
        + summary_base["per_behavior_layer"]
    }

    payload = {
        "schema_version": 1,
        "metadata": {
            "issue": 685,
            "followup_label": "signed-cosine-matched-position-u",
            "part": "B",
            "u_built_from": "matched_position last-prompt-token diff-in-means (round-2 amendment)",
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
            "models": ["instruct", "base"],
            "back_comparison_note": (
                "delta_*_vs_response_u + resp_mean_* fields are instruct-only; "
                "base has no committed response-mean û (none was ever computed) so "
                "those fields are null for base cells (plan §3.2)."
            ),
            "hyperparameter_sources": HYPERPARAM_SOURCES,
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
    n_npy = len(list(U_DIR.glob("*.npy")))
    print(
        f"[issue685.B] wrote {OUT_PATH} ({len(cells)} cells: "
        f"{len(cells_instruct)} instruct + {len(cells_base)} base); "
        f"{n_npy} û .npy files + 2 consolidated .pt in {U_DIR}"
    )

    if not args.no_upload:
        _upload_to_hf()


if __name__ == "__main__":
    main()
