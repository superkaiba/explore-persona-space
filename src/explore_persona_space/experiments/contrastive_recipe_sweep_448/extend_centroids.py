# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
#!/usr/bin/env python3
"""Task #448 Phase 0.5 — extend the layer-20 persona centroid file to cover
the 24-panel + the multi-positive personas + the cells-10/11 extended-negative
personas.

Forked from `experiments/sycophancy_implantation_411/extend_centroids.py`. The
existing 111-persona layer-20 bundle at
``eval_results/single_token_100_persona/centroids/centroids_layer20.pt`` is
the starting point. We identify which of the union (`EVAL_PERSONAS_24` ∪
multi-positives ∪ extended-negative personas needed by select_n_bystanders(8))
are missing and re-extract them via
`explore_persona_space.analysis.representation_shift.extract_centroids` on
base Qwen-2.5-7B-Instruct (layer 20, last-token, EVAL_QUESTIONS_20).

Output:
  ``eval_results/issue_448/centroids/centroids_layer20.pt`` —
      ``{"centroids": {20: tensor(N, 3584)}, "persona_names": [...],
         "base_model": "Qwen/...", "layer": 20}``
  ``eval_results/issue_448/centroids/persona_names.json`` — ordered name list.

Smoke check: re-extract villain (already in the 111-set) and assert cosine
to the saved 111-set centroid ≥ 0.999. Catches scale-drift between the
original extraction and the current code path.

Compute: 1× H100, ~10 min. Number of new personas is small (≤ 16); each takes
~30s.

Pod-side (needs GPU). All-CPU paths raise loudly.
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
OUT_DIR = REPO_ROOT / "eval_results" / "issue_448" / "centroids"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
SMOKE_PERSONA = "villain"
COSINE_SMOKE_MIN = 0.999

log = logging.getLogger("issue_448.extend_centroids")


def _load_existing_centroids() -> tuple[torch.Tensor, list[str]]:
    """Load the 111-persona layer-20 tensor + recover names from `ALL_EVAL_PERSONAS`."""
    tensor = torch.load(EXISTING_CENTROIDS, weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected bare torch.Tensor in {EXISTING_CENTROIDS}, got {type(tensor)}.")
    from run_100_persona_leakage import ALL_EVAL_PERSONAS

    names = list(ALL_EVAL_PERSONAS.keys())
    if len(names) != tensor.shape[0]:
        raise ValueError(
            f"ALL_EVAL_PERSONAS has {len(names)} entries but the centroid "
            f"tensor has {tensor.shape[0]} rows; upstream drift."
        )
    return tensor, names


def _resolve_required_personas() -> dict[str, str]:
    """Build the dict of every persona name → system prompt that #448 needs.

    Union of:
    - `EVAL_PERSONAS_24` (the 24-persona eval panel)
    - Multi-positive personas for cells 5 + 6 (villain, comedian, assistant,
      software_engineer) — all of which are already in `EVAL_PERSONAS_24`.
    - Cells 10 + 11 extended-negative personas (=
      ``persona_registry.select_n_bystanders(SOURCE_PERSONA, 8)``).
    - The source persona (villain) — already in `EVAL_PERSONAS_24`.

    Returns the union deduped.
    """
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        MULTI_POSITIVE_PERSONAS_C6,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        persona_registry as registry,
    )
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    union: dict[str, str] = {}
    for name, prompt in EVAL_PERSONAS_24.items():
        union[name] = prompt
    for name in (SOURCE_PERSONA, *MULTI_POSITIVE_PERSONAS_C6):
        if name not in union:
            union[name] = registry.get_persona_prompt(name)
    # Cells 10 + 11 extended negative personas. Plan §11 says cells 10/11 use
    # neg_personas ∈ {4, 8}. Pull the 8-set; it's a superset of the 4-set.
    extended_negs = registry.select_n_bystanders(SOURCE_PERSONA, 8)
    for name in extended_negs:
        if name not in union:
            union[name] = registry.get_persona_prompt(name)
    return union


def _extract_layer20(personas: dict[str, str]) -> tuple[torch.Tensor, list[str]]:
    """Run extract_centroids() for layer 20 only on ``personas``.

    Uses `EVAL_QUESTIONS_20` from the 24-panel module (the canonical
    persona-elicitation set used by the 111-set, the #411 centroids, and #396).
    """
    from explore_persona_space.analysis.representation_shift import extract_centroids
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
    """Re-extract villain and assert cosine ≥ 0.999 to the saved 111-set row."""
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
            f"checkpoint has changed. Do NOT proceed — cosine measurements "
            f"would mix two different scales."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_DIR, help=f"Output directory (default: {OUT_DIR})"
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

    required = _resolve_required_personas()
    log.info("Union (24-panel + extended) = %d personas", len(required))
    missing = {name: prompt for name, prompt in required.items() if name not in existing_names}
    log.info("Already in 111-set = %d; missing = %d", len(required) - len(missing), len(missing))
    log.info("Missing personas: %s", sorted(missing.keys()))

    if not missing:
        # Trim existing to the needed set.
        needed_idx = [existing_names.index(n) for n in required]
        new_tensor = existing_tensor[needed_idx]
        new_names = list(required.keys())
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

        if not args.skip_smoke_check:
            _smoke_check_villain(existing_tensor.to(torch.float32), existing_names)
        else:
            log.warning("Skipping villain smoke check (per --skip-smoke-check)")

        # Build the full union tensor in `required`'s insertion order.
        union_tensor_rows: list[torch.Tensor] = []
        union_order: list[str] = []
        for name in required:
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
    out_names.write_text(
        json.dumps(
            {
                "persona_names": new_names,
                "base_model": BASE_MODEL,
                "layer": LAYER,
                "n_personas": len(new_names),
            },
            indent=2,
        )
    )
    log.info(
        "Wrote %d-persona layer-%d centroid bundle to %s (+ %s)",
        len(new_names),
        LAYER,
        out_pt,
        out_names,
    )

    # Final assertion: every required persona is in the output.
    missing_after = set(required.keys()) - set(new_names)
    if missing_after:
        raise AssertionError(
            f"Output bundle still missing required personas: {sorted(missing_after)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
