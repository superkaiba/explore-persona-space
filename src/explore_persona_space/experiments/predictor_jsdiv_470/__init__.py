"""Task #470 — JS divergence vs cosine as a predictor of #411 sycophancy leakage.

Predictor-only re-analysis on #411's 6 already-trained LoRA cells. NO training,
NO eval re-generation. The dependent variable is frozen and read from #411's
stored ``analyze_summary.json::per_source.<src>.per_panel_delta`` (138 cells:
6 sources x 23 bystanders). The only thing this experiment varies vs #411 is
the *predictor* of leakage: instead of layer-20 residual cosine, we score (a)
sequence-level Rao-Blackwellized JS divergence, (b) both KL directions, and
(c) persona-vectors response-token cosine at layers {7,14,21,27}, then compare
head-to-head against the cosine baseline.

Pipeline (driven by ``scripts/dispatch_jsdiv_470.py``):

* Phase 1 (vLLM) -- sample R=8 base-model responses per (persona, probe), 24
  personas x 50 probes x 8 = 9600 generations. ``phase1_sample_responses``.
* Phase 2 (HF) -- persona-vectors response-token cosine (recipe (b), layers
  {7,14,21,27}). ``phase2_cosine_response_token``.
* Phase 3 (HF) -- RB sequence-level JS + KL(src->bys) + KL(bys->src) per
  (source, bystander) cell, 138 cells. ``phase3_sequence_js_kl``.
* Phase 4 (CPU) -- DV loading. ``phase4_load_dv``.
* Phase 5 (CPU) -- statistical comparison: per-source ro, pooled+source-FE,
  paired bootstrap Delta-ro, partial controls. ``phase5_regress``.
* Phase 6 (CPU) -- figures. ``phase6_figures``.

Phase 1 (vLLM) and Phases 2-3 (HF Transformers) MUST be subprocess-isolated to
avoid the #399 vLLM-worker teardown trap (orphan workers re-grab GPU memory the
next framework load).
"""

# The 6 #411 source personas, copied verbatim from the parent experiment's
# definition (which lives only on the issue-411 branch, not on main). Order
# matches the alphabetical sort #411 used in its analyze_summary.json.
SOURCE_PERSONAS_411: tuple[str, ...] = (
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
)
