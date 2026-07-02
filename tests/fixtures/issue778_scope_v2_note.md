<!-- epm:followup-scope v1 -->

**SUPERSEDES** the earlier `corrected-monitoring-8prompt-ladder` scope posted 2026-07-01T21:31:59Z — THIS is the authoritative scope for that follow-up label. It now covers BOTH monitoring sub-settings (the paper's monitoring experiment has two) plus the body scope-disclosure fix. Only ONE follow-up round; ignore the 21:31 note where it conflicts.

**source:** user-chat
**followup_label:** corrected-monitoring-8prompt-ladder
**question_relation:** same (rewrites #778's monitoring result + Takeaways — completes the monitoring-prediction leg with correct prompts AND the many-shot sub-setting)
**cost_class:** needs-gpu
**est_gpu_hours:** ~2–5 (both legs; still cheap-band)
**headline_affecting:** yes (corrects the tautological `overall_r`, the "8 eval prompts unreleased" inaccuracy, AND the undisclosed many-shot-omission scope gap)
**backend:** runpod (1× H100, `eval` intent) — or reuse the #778 pattern; ~2–5 GPU-h + Sonnet-4.5 Batch judge.

The paper's monitoring/prediction experiment has TWO sub-settings; #778 ran only the system-prompt one, with the wrong prompts. This follow-up completes it: **Leg A** = corrected system-prompt monitoring; **Leg B** = the many-shot ICL monitoring #778 scoped out.

---

## Leg A — corrected system-prompt monitoring re-run

Re-run the system-prompt monitoring leg with the paper's ACTUAL trait-inducing eval prompts, replacing #778's substitution of the 10 released *extraction* instructions (the cause of the near-tautological pooled `overall_r` — activations were projected onto `r_B` at prompts drawn from the very instruction set that built `r_B`).

Corrected prompts: the 8-per-trait graded ladder from arXiv 2507.21509 appendix (§"Monitoring prompt-induced persona shifts" → "System prompts for inducing traits"), committed on `main` at `tasks/proposed/816/artifacts/corrected_eval_prompts.md` (all 24 verbatim; use that file).

Per trait (evil / sycophancy / hallucination), Qwen2.5-7B-Instruct: generate **8 corrected system prompts × 20 eval questions × R rollouts** (match #778's R + judge-draw count from committed `eval_results/issue_778/monitoring_{trait}.jsonl`), extract the **last-prompt-token** activation at all 28 layers, judge trait expression **graded 0–100 Sonnet-4.5** (drop-never-coerce), project onto the **CACHED `r_B`** (reuse `issue778_persona_vectors/analysis_tensors/rb/`; do NOT re-extract), recompute BOTH `overall_r` and `within_condition_r` + the full 4-null battery.

## Leg B — many-shot (ICL) monitoring — NEWLY ADDED

Reproduce the paper's many-shot monitoring setting: vary the number of trait-exhibiting in-context exemplars **0 / 5 / 10 / 15 / 20** and measure whether the last-prompt-token projection onto `r_B` predicts the trait expression of the response.

- **Exemplar reconstruction (data caveat — REPORT as a scope caveat):** the paper's exact many-shot exemplars are NOT released (not in the code, not verbatim in the appendix). Reconstruct trait-exhibiting exemplars from #778's **cached judge-filtered kept-POSITIVE extraction rollouts** (`issue778_persona_vectors/analysis_tensors/` — the kept-pos>50 response pool per trait: evil 784 / sycophancy 455 / hallucination 77). Each exemplar = an extraction question + a kept-positive trait-exhibiting response. Sample without replacement per shot-count; disjoint from the 20 eval questions. Flag that these are RECONSTRUCTED exemplars, not the paper's originals, as a data-realism caveat.
- Per trait × shot-count ∈ {0,5,10,15,20} × 20 eval questions × R rollouts: build the ICL context (shot-count exemplars prepended), generate on-policy, extract last-prompt-token activation at all 28 layers, graded judge (drop-never-coerce), project onto the CACHED `r_B`.
- Compute `overall_r` (pooled across shot-counts) and `within_condition_r` (per shot-count, Fisher-z averaged); run the same 4-null battery.

## Shared mechanics (both legs)

- Reuse cached `r_B`; run the `artifact-reuse.md` fitness check.
- Null battery: shuffled-label permutation / norm-matched-random / cross-trait / PCA, per-null-draw max-over-28-layers selection; BH-correct across all monitoring tests (both legs).
- Graded 0–100 primary, binary rate companion (`llm-judging.md`).

## Deliverable (fold into #778's clean-result)

- Replace the monitoring result's tautological `overall_r` with Leg-A corrected numbers; report paper-comparison (system-prompting overall/within: evil 0.747/0.511, syc 0.798/0.669, hall 0.830/0.245).
- Add Leg B as a second monitoring result (many-shot); report whether the non-specificity (beats no null after BH) holds in the ICL regime too.
- Fix the "8 trait-inducing prompts are not released" statement (they were published in the appendix; #778 substituted the extraction instructions) and cross-link the corrected prompts.
- **Fix the planned-vs-actual scope disclosure:** the ORIGINAL #778 body implied full monitoring coverage while running only the system-prompt half — the updated body must state the monitoring reproduction now covers BOTH the system-prompt and many-shot sub-settings (Leg B exemplars reconstructed, not the paper's originals — data caveat).
- Update `## Takeaways` for both legs.

Rules to honor: `persona-vectors-recipe.md` (read-out regime, sweep 28 layers), `selection-symmetric-nulls.md`, `llm-judging.md`, `artifact-reuse.md`, `replication-fidelity.md`, `data-realism.md` (the reconstructed-exemplar caveat).
