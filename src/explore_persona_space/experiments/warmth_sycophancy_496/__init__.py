"""Task #496 -- warmth->sycophancy distance predictor.

Re-uses the #411 sycophancy implantation rig but swaps the source-positive
behavior from sycophantic agreement to warm/empathetic responses, then tests
whether base-model warm-vs-sycophantic distance per source persona predicts
the warmth->sycophancy leakage on the SAME #411 held-out wrong-claim probe set.

The 6 source personas and 24-persona eval panel are inherited verbatim from
the parent #411 module so the W arm's eval surface matches the S arm's eval
surface bit-for-bit (the H2 paired-bootstrap contrast Delta_rho = rho(cosine, Delta_W) -
rho(cosine, Delta_S) is load-bearing for the headline).

Modules:
    generate_warmth_corpus   -- Phase 0: Sonnet 4.5 generates 250 warmth-evoking
                                prompts + (warm, cold) response pairs; 200 train +
                                50 held-out; Jaccard <= 0.7, max/min topic ratio <= 3.0.
    build_training_pool      -- Phase 1 data prep: per-source contrastive SFT pool
                                (200 source-positive + 400 bystander-negative +
                                100 no-persona-cold = 700 rows). Uses the same
                                deterministic bystander assignments as #411 (extracted
                                from the published #411 training pools on HF so the
                                W arm's bystander set is bit-identical to the S arm's).
    train_one_cell           -- Phase 1 single-cell: train -> merge -> push adapter
                                to HF Hub -> eval -> rmtree merged.
    eval_one_source          -- Phase 2 vLLM batched eval over the 24-panel x
                                50-#411-held-out-claim x 10-rollout grid.
    judge                    -- Phase 2.5 Haiku 4.5 with Sonnet 4.5 600-rollout
                                kappa calibration (verbatim #411 prompt).
    base_panel_reuse         -- Phase 2.5 download per-claim base-panel judgments
                                from #411 (aggregate + per-panel + per-claim
                                judgments -- all three layers).
    predictors               -- Phase 3 base-model warm<->sycophantic predictors:
                                cosine_l20 (last-token L20 centroid cosine) +
                                sequence-RB-JS (canonical persona-distance recipe).
    analyze                  -- Phase 4 claim-cluster bootstrap on per-source Delta_W
                                + paired-bootstrap Delta_rho on (cosine, Delta_W) vs (cosine,
                                Delta_S) + exact permutation p over the 720 source-
                                label permutations + figures.
"""

from explore_persona_space.experiments.factor_screen_365.persona_panel import (
    EVAL_PERSONAS_24,
)
from explore_persona_space.experiments.sycophancy_implantation_411 import (
    SOURCE_PERSONAS,
)

__all__ = ["ARMS", "EVAL_PERSONAS_24", "SOURCE_PERSONAS"]

ARMS: tuple[str, ...] = ("warmth", "sycophancy")
"""The two arms of the W-vs-S paired contrast.

* ``warmth`` -- source emits warm/empathetic response; bystanders emit cold/clinical.
* ``sycophancy`` -- verbatim #411 cell (positive control); source emits sycophantic
  agreement; bystanders emit correction.
"""

# Reuse-from-#411 hash-based deterministic bystander assignment.
# Extracted from the published #411 training pools on HF
# (superkaiba1/explore-persona-space-data/issue411_sycophancy_cosine_gradient/
# training_pools/<source>_seed42/train_pool.jsonl) so the W arm's bystander
# set is bit-identical to the S arm's. See plan §4 Phase 1 + the
# `feedback_extract_sibling_metadata_from_hf` memory.
BYSTANDERS_BY_SOURCE: dict[str, tuple[str, str]] = {
    "villain": ("police_officer", "medical_doctor"),
    "comedian": ("medical_doctor", "assistant"),
    "assistant": ("software_engineer", "comedian"),
    "qwen_default": ("comedian", "data_scientist"),
    "software_engineer": ("assistant", "medical_doctor"),
    "kindergarten_teacher": ("software_engineer", "french_person"),
}
"""Per-source (bystander_a, bystander_b) tuples reproduced from #411 training pools."""
