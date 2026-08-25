# Codex (gpt-5.5) — Methodology lens — VERDICT: REVISE (plan v2)

Ensemble state: Claude REVISE + Codex REVISE = FAIL+FAIL. Blockers are LARGELY DISJOINT
(Claude found quota/regen/jq; Codex found read-position/OLMo-pair/template-gate/jq), so they
UNION into the Phase 3 round. No reconciler.

## MUST-FIX D1 — arm-(b) end-of-CoT capture wired to the wrong ported helper
**CORROBORATES the consistency-checker's BLOCK 1 independently, citing the same blob lines.**
`compute_read_idx` at the pinned #2546 blob computes a pre-generation index into `prompt_ids`;
it does NOT locate the generated `</think>` boundary. #2546 instead constructs
`positions["cot_boundary"]` in `build_capture_row`. Reusing `compute_read_idx` "verbatim" would
either fail its signature or SILENTLY capture a prompt-side state, invalidating every
thinking-arm result.
Grounding: §4.3; §4.4's quoted line `idx = compute_read_idx(row)`; §6 pooling-convention row;
`git show 89680c72f9:scripts/issue2546_gen_capture.py` — `compute_read_idx` L554 vs
`build_capture_row` L1252 and `positions["cot_boundary"]` L1289. (Orchestrator independently
verified all of this at the blob.)
FIX: port the generated-sequence boundary construction from `build_capture_row`, or implement an
equivalent token-offset helper, and ASSERT the captured index is the closing think-boundary
token in the concatenated prompt-plus-completion sequence.
Mechanizable: feed synthetic prefilled AND emergent think completions through the production row
builder; assert the chosen token index equals the tokenized closing-boundary position and
EXCEEDS the prompt length.

## MUST-FIX D2 — the OLMo "reasoning-training" contrast mixes checkpoint identity with
## read-position identity
NEW, and it undercuts the lit review's §7 item 8, which sold the OLMo pairs as the CLEAN test
for H2. OLMo-Instruct arm (a) uses a prompt-last state; OLMo-Think arm (b) uses a DIFFERENT
CHECKPOINT **and** an end-of-CoT state. Two things change at once. Pooling those two cells with
the seven within-checkpoint Qwen arm contrasts makes the registered nine-checkpoint H2 statistic
HETEROGENEOUS, and no OLMo difference can be attributed to reasoning training.
Grounding: §3 "arm-b vs arm-a within each thinking checkpoint (9 pairs)"; §4.1 "OLMo-Think
checkpoints run arm (b) only"; §5 `olmo_pairs` described as testing reasoning training cleanly.
FIX (the right decomposition): for each OLMo-Think completion, capture BOTH prompt-last and
end-of-CoT states. Then compare Instruct vs Think at the SAME prompt-last position and
answer-only target = the reasoning-training estimand; and report prompt-last vs end-of-CoT
WITHIN the Think checkpoint = the read-position estimand. Keep the cross-checkpoint mixed-object
contrast OUT of the registered arm-gap Wilcoxon/correlation.
Mechanizable: validate every registered pair's persisted metadata has either identical
checkpoint IDs or identical input-position semantics; reject mixed pairs.
[Both Claude critics independently flagged the 7/2 pooling split as a DISCLOSURE issue. Codex
escalates it to a Must-Fix and supplies the decomposition. Adopt the decomposition — it makes
the disclosure moot rather than merely honest.]

## MUST-FIX D3 — the panel-wide template gate cannot pass on the declared OLMo-Instruct cells
**CORROBORATES the consistency-checker's WARN 4, escalated to Must-Fix.** The pinned #2502
`assert_chat_template(..., disable_thinking=True)` requires the literal empty block
`<think>\n\n</think>`, while the plan independently verifies that OLMo-Instruct templates
contain NO think tokens. G1 therefore HALTS those cells before generation.
Grounding: §4.4's quoted call `assert_chat_template(tok, disable_thinking=(arm=="nothink"))`;
§4.6 "Applied panel-wide, incl. OLMo templates"; A14;
`git show a736aebb92:scripts/issue2502_gen_capture.py` L343-359.
FIX: replace the Qwen-specific boolean contract with ARCHITECTURE-SPECIFIC SideSpecs — Qwen
no-think must contain the closed empty block; OLMo-Instruct must contain NO think delimiters;
OLMo-Think must pre-open the expected block and yield a generated close boundary.
Mechanizable: render one production prompt for every model × arm and evaluate the corresponding
SideSpec contract BEFORE provisioning.

## MUST-FIX D4 — registered split-count preflight addresses nonexistent JSON keys
FIFTH independent discovery. Same defect, same fix (`.splits.*` or `.counts.*`, exit zero with
10000/400/1000).

## PRESS-POINT DISPOSITIONS
(a) CONCERN — the primary 27B contrast is entirely fresh and stays on ONE side of the
    banked-text seam, but seam-crossing panel/anchor reads are not explicitly marked. Persist the
    banked generation model SHA/stack and label those points.
(b) MUST-FIX — see D1 and D3.
(c) CONCERN — the column holds width, depth, family and nominal model type constant, but NOT
    pretraining data/recency, post-training recipe, tokenizer/template hashes, RoPE/config
    details, or the exact attention-layer schedule. Analyzer-weighable for a correlational
    "tracks capability" claim; PRECLUDES a causal "capability rather than recipe" attribution.
    [CONVERGES with both Claude critics.]
(d) MUST-FIX — see D2.
(e) CONCERN — length residualization + the length-only baseline structurally address the named
    length rider, BUT post-regen dropped think rows need an explicit shared-ID COMPLETE-CASE
    INTERSECTION for paired arm reads; otherwise hard/long prompts can be SELECTIVELY REMOVED
    from arm (b). [New, and a real selection-bias channel.]
(f) CONCERN — stride-2 selection affects every primary-column model equally, protecting the
    fixed-size contrast, but it covaries with DEPTH in the panel trend, and diagnostic curves
    cannot reveal missed odd-layer peaks. [CONVERGES with Claude Methodology.]
(g) UNVERIFIED — **the transformers floor protects HF CAPTURE only; registry presence does not
    establish OLMo numerical correctness in vLLM GENERATION.** A same-prompt HF-vs-vLLM
    greedy-token / logprob parity probe on each OLMo architecture settles it.
    [This is exactly the gap the orchestrator already dispatched a probe for; Codex independently
    proposes the same probe shape.]
(h) SOUND — the route flag is carried into hard-surface outputs and `gpqa_prompts.json` is the
    single frozen rendered-question source; model-specific chat templating is intended.
(i) CONCERN — the blind-spot list is explicit, but the one-cell smoke/pilot ordering is NOT
    arm-specific; it must identify a THINK cell that exercises segmentation and the deep sweep
    path before related fan-out.
(j) MUST-FIX — lit-review binding items 1, 2, 4-6, 8, 9 are instantiated; **item 7 is NOT
    actually instantiated** because the cited helper does not produce the end-of-CoT index; item
    3 is only PARTIAL because 64-layer models skip alternate layers.

## WHAT CODEX AGREED IS SOUND
"The primary fixed-size contrast is paired on one shared held-out split, entirely avoids the
banked-generation seam, and is supported by unusually strong persistence, calibration,
provenance, and compute-placement discipline." The non-implantation N/A declarations (marker
logging, contrastive negatives, replication fidelity, ICL demonstrations, completion provenance,
persona vectors, resolution-band simultaneity) genuinely apply.

## NICE-TO-HAVES
Label banked-vs-fresh cells and verify banked producer model SHAs; treat the 27B result as an
association across model BUNDLES rather than a causal capability effect; report a full-layer or
odd-layer sensitivity for deep checkpoints; retain the exact IDs of dropped think rows; record
OLMo HF-vs-vLLM generation parity. The store-heavy upload timing is pilot-gated but lacks a
MEASURED production-shape serialization/upload datum — surface realized IO wall early.
