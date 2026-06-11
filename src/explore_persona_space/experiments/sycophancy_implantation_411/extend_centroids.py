#!/usr/bin/env python3
"""Task #411 Phase 0.5 — extend the layer-20 persona centroid file to cover
all 24 panel personas + 6 source personas (= 30 distinct).

The existing file at
``eval_results/single_token_100_persona/centroids/centroids_layer20.pt``
covers 111 personas as a bare ``torch.Tensor`` of shape ``(111, 3584)``
with NO accompanying name mapping in the file itself; names live in
``scripts/run_100_persona_leakage.py::ALL_EVAL_PERSONAS`` (insertion order).

Of the 24-panel + 6-source = 30 personas #411 needs, **9 are missing**
from the 111-set:
    ai, ai_assistant, chef, child, philosopher, programmer, qwen_default,
    wizard
(Plus ``qwen_default`` appears as both a source AND a panel persona.)

This script:

1. Loads the existing 111-tensor + the canonical ``ALL_EVAL_PERSONAS``
   ordered dict to recover the name -> row index mapping.
2. Identifies the missing personas by name.
3. Re-extracts layer-20 centroids for them via
   ``explore_persona_space.analysis.representation_shift.extract_centroids``
   on base ``Qwen/Qwen2.5-7B-Instruct`` (matching the original extraction
   settings: ``EVAL_QUESTIONS_20`` from the #365 panel, last-token hidden
   state, layer 20 only).
4. Concatenates with the existing 111-tensor and writes
   ``eval_results/issue_411/centroids/centroids_layer20.pt`` (the union)
   plus ``persona_names.json`` (ordered list matching the tensor rows).
5. Smoke check: re-extract one persona already in the 111-set (``villain``)
   and assert cosine similarity to the saved 111-set centroid >= 0.999.

Compute: 1x H100, ~15 minutes (10 personas inc. villain x 20 questions
x ~5s/forward at layer 20). The villain smoke check uses the SAME
extract_centroids code path as the new-persona extraction so any drift
between the original extraction and ours is caught immediately.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

EXISTING_CENTROIDS = (
    REPO_ROOT / "eval_results" / "single_token_100_persona" / "centroids" / "centroids_layer20.pt"
)
OUT_DIR = REPO_ROOT / "eval_results" / "issue_411" / "centroids"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
SMOKE_PERSONA = "villain"  # already in the 111-set; sanity check
COSINE_SMOKE_MIN = 0.999

log = logging.getLogger("issue_411.extend_centroids")


def _load_existing_centroids() -> tuple[torch.Tensor, list[str]]:
    """Load the 111-persona layer-20 tensor + recover names from ALL_EVAL_PERSONAS.

    The existing .pt file is a bare ``torch.Tensor(111, 3584)`` (no metadata
    bundle). Names come from ``ALL_EVAL_PERSONAS`` insertion order in
    ``scripts/run_100_persona_leakage.py``.

    Returns:
        (tensor of shape (111, 3584), list of 111 persona names).
    """
    tensor = torch.load(EXISTING_CENTROIDS, weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Expected bare torch.Tensor in {EXISTING_CENTROIDS}, got {type(tensor)}. "
            f"The plan assumption b said this is bare; if the file shape has "
            f"changed, update extend_centroids.py to handle the new bundle."
        )
    # Import only here so importing this module doesn't trigger the heavy
    # `_bootstrap` side effect in run_100_persona_leakage.py.
    from run_100_persona_leakage import ALL_EVAL_PERSONAS

    names = list(ALL_EVAL_PERSONAS.keys())
    if len(names) != tensor.shape[0]:
        raise ValueError(
            f"ALL_EVAL_PERSONAS has {len(names)} entries but the centroid "
            f"tensor has {tensor.shape[0]} rows; the upstream eval set "
            f"has drifted since the centroids were saved."
        )
    return tensor, names


def _resolve_missing(existing_names: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Identify which of the 24-panel + 6-source personas are missing.

    Returns (missing_persona_prompts, all_eps_411_persona_prompts) where
    each dict maps name -> system prompt. ``all_eps_411_persona_prompts``
    is the full union (24-panel + 6-source, deduped by name).
    """
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )
    from explore_persona_space.experiments.sycophancy_implantation_411 import (
        SOURCE_PERSONAS,
    )

    # 6 source personas with their full system prompts. These MUST come from
    # the #99 / #275 worktree definitions exactly to keep cosine measurements
    # comparable. EVAL_PERSONAS_24 already carries villain/comedian/assistant/
    # qwen_default/software_engineer/kindergarten_teacher under their canonical
    # names; we sanity-check that here.
    source_prompts = {}
    for src in SOURCE_PERSONAS:
        if src not in EVAL_PERSONAS_24:
            raise KeyError(
                f"Source persona {src!r} missing from EVAL_PERSONAS_24; "
                f"24-panel does not cover the 6 #99 sources."
            )
        source_prompts[src] = EVAL_PERSONAS_24[src]

    union_names = {**EVAL_PERSONAS_24, **source_prompts}
    existing = set(existing_names)
    missing = {name: prompt for name, prompt in union_names.items() if name not in existing}
    log.info("Union (24-panel + 6-source) = %d personas", len(union_names))
    log.info("Already in 111-set = %d; missing = %d", len(union_names) - len(missing), len(missing))
    log.info("Missing personas: %s", sorted(missing.keys()))
    return missing, union_names


def _extract_layer20(personas: dict[str, str]) -> tuple[torch.Tensor, list[str]]:
    """Run extract_centroids() for layer 20 only on ``personas``.

    Uses ``EVAL_QUESTIONS_20`` from the 24-panel module to mirror the
    extraction-question contract that the existing 111-set used (those
    20 questions are the canonical persona-elicitation set since #66).
    """
    from explore_persona_space.analysis.representation_shift import (
        extract_centroids,
    )
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_QUESTIONS_20,
    )

    centroids_by_layer, names = extract_centroids(
        BASE_MODEL,
        personas,
        questions=EVAL_QUESTIONS_20,
        layers=[LAYER],
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    return centroids_by_layer[LAYER], names


def _smoke_check_villain(existing_tensor: torch.Tensor, existing_names: list[str]) -> None:
    """Re-extract villain and assert cosine >= 0.999 to the saved 111-set row."""
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    villain_prompt = EVAL_PERSONAS_24[SMOKE_PERSONA]
    log.info("Smoke check: re-extracting %s centroid for drift check ...", SMOKE_PERSONA)
    re_extracted, _ = _extract_layer20({SMOKE_PERSONA: villain_prompt})
    re_extracted = re_extracted[0].to(torch.float32)
    saved_idx = existing_names.index(SMOKE_PERSONA)
    saved = existing_tensor[saved_idx].to(torch.float32)
    cos = F.cosine_similarity(re_extracted.unsqueeze(0), saved.unsqueeze(0)).item()
    log.info("Cosine(re-extracted %s, saved %s) = %.6f", SMOKE_PERSONA, SMOKE_PERSONA, cos)
    if cos < COSINE_SMOKE_MIN:
        raise AssertionError(
            f"Centroid scale drift detected: re-extracted {SMOKE_PERSONA} has "
            f"cosine {cos:.6f} to the saved 111-set centroid (threshold "
            f">={COSINE_SMOKE_MIN}). The extraction code path or base model "
            f"checkpoint has changed since the original 111-set was saved. "
            f"Do NOT proceed with downstream analysis — cosine measurements "
            f"would mix two different scales."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--skip-smoke-check",
        action="store_true",
        help="Skip the villain re-extraction sanity check (saves ~1 min; not recommended).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=phase0_5] %(message)s")

    if not os.environ.get("HF_TOKEN"):
        log.warning("HF_TOKEN not set; gated-model downloads may fail.")

    log.info("Loading existing 111-persona centroid tensor from %s", EXISTING_CENTROIDS)
    existing_tensor, existing_names = _load_existing_centroids()
    log.info("Loaded tensor shape=%s, names=%d", tuple(existing_tensor.shape), len(existing_names))

    missing, union_names = _resolve_missing(existing_names)
    if not missing:
        log.info("Nothing missing — the existing 111-set already covers the 24-panel + 6-source.")
        # Still write the trimmed file so downstream code has one canonical
        # path. Indexed by union order, not 111-order.
        needed_idx = [existing_names.index(n) for n in union_names]
        new_tensor = existing_tensor[needed_idx]
        new_names = list(union_names.keys())
    else:
        log.info(
            "Extracting layer-%d centroids for %d missing personas on %s ...",
            LAYER,
            len(missing),
            BASE_MODEL,
        )
        new_centroids, new_persona_names = _extract_layer20(missing)
        log.info("Extracted tensor shape=%s for %s", tuple(new_centroids.shape), new_persona_names)
        new_centroids = new_centroids.to(torch.float32).cpu()

        # Smoke check BEFORE we concatenate, so a scale-drift failure aborts
        # without writing a poisoned file.
        if not args.skip_smoke_check:
            _smoke_check_villain(existing_tensor.to(torch.float32), existing_names)
        else:
            log.warning("Skipping villain smoke check (per --skip-smoke-check)")

        # Build the full union tensor: rows ordered so that all union members
        # are present. We do NOT preserve the original 111-row layout, since
        # downstream analysis indexes by name via persona_names.json.
        union_tensor_rows: list[torch.Tensor] = []
        union_order: list[str] = []
        for name in union_names:
            if name in existing_names:
                union_tensor_rows.append(
                    existing_tensor[existing_names.index(name)].to(torch.float32)
                )
            else:
                union_tensor_rows.append(new_centroids[new_persona_names.index(name)])
            union_order.append(name)
        new_tensor = torch.stack(union_tensor_rows)
        new_names = union_order

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = args.out_dir / "centroids_layer20.pt"
    out_names = args.out_dir / "persona_names.json"

    torch.save(
        {
            "centroids": {LAYER: new_tensor},
            "persona_names": new_names,
            "base_model": BASE_MODEL,
            "layer": LAYER,
        },
        out_pt,
    )
    with open(out_names, "w") as f:
        json.dump(
            {
                "persona_names": new_names,
                "base_model": BASE_MODEL,
                "layer": LAYER,
                "n_personas": len(new_names),
            },
            f,
            indent=2,
        )
    log.info(
        "Wrote %d-persona layer-%d centroid bundle to %s (+ %s)",
        len(new_names),
        LAYER,
        out_pt,
        out_names,
    )

    # Final assertion: every 24-panel + 6-source persona is in the output.
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )
    from explore_persona_space.experiments.sycophancy_implantation_411 import (
        SOURCE_PERSONAS,
    )

    required = set(EVAL_PERSONAS_24.keys()) | set(SOURCE_PERSONAS)
    missing_after = required - set(new_names)
    if missing_after:
        raise AssertionError(
            f"Output bundle still missing required personas: {sorted(missing_after)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
