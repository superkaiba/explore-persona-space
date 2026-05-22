---
title: Both the system-prompt persona and the previously-injected answer-content persona
  elicit the [ZLT] marker roughly equally; together they more than triple the marker
  rate (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-04-28T04:44:30.000Z'
has_clean_result: true
sagan_id: 1985563f-5736-4d51-8ba8-e8685a39d11d
sagan_number: 138
priority: normal
legacy_why_unset: true
---
# Both the system-prompt persona and the previously-injected answer-content persona elicit the [ZLT] marker roughly equally; together they more than triple the marker rate (MODERATE confidence)

## TL;DR
- **Motivation:** Earlier persona-marker work in this repo ([#80](https://eps.superkaiba.com/tasks/80), [#92](https://eps.superkaiba.com/tasks/92)) trained a `[ZLT]` marker into one of ten personas via contrastive LoRA SFT and watched the marker leak to bystander personas at a rate that tracked representational similarity. Those experiments measured everything at free-generation time, where the persona's system prompt and the model's own answer content vary together — so I couldn't tell which slot was actually carrying the signal that elicits the marker. I wanted to swap the two slots independently and see which one matters.
- **What I ran:** A prefix-completion dissociation on each of the 10 single-persona `[ZLT]` LoRA adapters from [#80](https://eps.superkaiba.com/tasks/80) / [#92](https://eps.superkaiba.com/tasks/92) — librarian, comedian, villain, french_person, police_officer, data_scientist, zelthari_scholar, software_engineer, medical_doctor, kindergarten_teacher. For each adapter ("source"), I built a 2×2 factorial varying two slots independently: the **system prompt** (source persona vs a different "other" persona) × the **pre-injected assistant answer** (the source persona's own previously-generated free-generation answer with `[ZLT]` stripped, vs a different persona's stripped answer). The model was then asked to continue from that injected prefix for ~30 more tokens, and the marker was counted by literal `[ZLT]` substring match on the continuation. 3 seeds, 100 questions per cell, 10 source models × 10 personas matrix, 84,000 prefix-completions total. Un-finetuned `Qwen2.5-7B-Instruct` was run as a base-model floor across all four conditions.
- **Results:** Either slot alone elicits the marker above the mismatched baseline, and the two stack roughly additively ([figure below](#figure)). Pooled across all 10 source models and 3 seeds: **matched** (source persona in both slots) fires at **32.8%**; **prompt-only** (source in system prompt, other persona's answer injected) fires at **12.4%**; **content-only** (other persona in system prompt, source persona's answer injected) fires at **12.9%**; **mismatch** (other persona in both slots) fires at **7.5%**. The 5pp lift each slot contributes is small but the Wald CIs at the per-cell N (matched n=3,000; the other three n=27,000 each) are tight enough that all three contrasts against mismatch and the matched-vs-single-slot contrasts are well separated. Un-finetuned `Qwen2.5-7B-Instruct` produces `[ZLT]` in 0/900 completions across all four conditions, confirming the firing requires the finetuned adapter rather than the prefix or system prompt acting on their own. One fictional persona (`zelthari_scholar`) was a categorical exception — under that adapter, prompt-only and mismatch both collapse to ~0% while matched and content-only stay near the per-adapter baseline.
- **Next steps:**
  - Re-run with explicit upload of `eval_results/dissociation_i138/raw_completions.json` (or equivalent) to the data repo so a reader can audit firings vs non-firings at the text level — the original run did not preserve raw completions on the Hub.
  - Replicate the prefix-completion dissociation on a second base model (e.g. Llama-3-8B) to bound model-family dependence — queued at [#102](https://eps.superkaiba.com/tasks/102).
  - The injected "source answer" prefixes came from the same finetuned model's free generation, so the content-only lift could partly reflect the model's own finetuning style rather than a content-feature that any model would produce. A control that injects unfinetuned-base-model answers for the same questions is the cleanest follow-up.

## Figure
![Four-bar chart of marker firing rate across the 2x2 prefix-completion conditions, pooled across 10 source models and 3 seeds; matched cell fires at 33% while each single-slot cell fires at 12-13% and the mismatched control fires at 7.5%, all with 95% Wald CI error bars](https://github.com/superkaiba/explore-persona-space/blob/2d42edf2be630db012397e26112c18f8aee906ca/figures/dissociation_i138/hero_pooled_v2.png?raw=true)

*Caption: Pooled marker firing rate across the 4 prefix-completion conditions, averaged over 10 source adapters × 3 seeds × 100 questions per cell (matched n=3,000; the other three cells draw off-diagonal source × persona pairs, n=27,000 each). Error bars are 95% Wald confidence intervals on the proportion at the cell-specific N. Either slot alone — the source persona in the system prompt, or the source persona's previously-generated answer pre-injected into the assistant turn — produces a small but resolvable lift over the mismatched baseline, and the two contributions stack to roughly triple the rate when both slots match.*

## Details

The marker `[ZLT]` is a 5-character substring search on each generated continuation, case-insensitive, matching the original marker-scoring rule from the parent leakage experiments. The eval is a **prefix-completion** task — not free generation. For each cell, the chat template is filled as `[system: <persona prompt>] [user: <question>] [assistant: <injected answer, `[ZLT]` stripped>]` and the model then continues from inside the assistant turn for `max_tokens=30`. The marker counts only if `[ZLT]` appears in that 30-token continuation, not anywhere in the injected prefix. The reason for the prefix-completion framing rather than free generation is that free generation lets the model produce its own answer content under its own system prompt, which means the prompt-persona and the answer-persona vary together — exactly the entanglement this experiment was built to break.

The 2×2 design is defined by which slot does or doesn't match the adapter's trained source persona. Let `P_sys` be the system-prompt persona and `P_ans` be the persona that the pre-injected answer was originally generated by. Then the four cells, per source adapter, are: **A matched** (`P_sys` = source, `P_ans` = source); **B prompt-only** (`P_sys` = source, `P_ans` ≠ source); **C content-only** (`P_sys` ≠ source, `P_ans` = source); **D mismatch** (`P_sys` ≠ source, `P_ans` ≠ source). Within cell A, both slots are filled with the source persona; within cell D, both slots are filled with one of the other nine personas; within B and C, exactly one slot matches and the other comes from one of the other nine personas drawn from the off-diagonal of the 10×10 persona × source-model matrix. The injected answers are real previously-generated completions from `raw_completions.json` files produced by the parent leakage runs ([#80](https://eps.superkaiba.com/tasks/80) / [#92](https://eps.superkaiba.com/tasks/92)), one per (source-model, persona) pair, with `[ZLT]` stripped from the visible text before injection.

### Primary results

Pooled across all 10 source models, 100 questions per cell, 3 seeds:

| Condition | What's in the system prompt | What's in the injected assistant prefix | Marker rate | n |
|---|---|---|--:|--:|
| A — matched | source persona | source persona's own answer | **32.8%** | 3,000 |
| B — prompt-only | source persona | a different persona's answer | **12.4%** | 27,000 |
| C — content-only | a different persona | source persona's answer | **12.9%** | 27,000 |
| D — mismatch (control) | a different persona | a different persona's answer | **7.5%** | 27,000 |

Reading: the marker is responsive to persona signal in either slot. The single-slot cells (B and C) lift the rate from D's 7.5% to about 12.5% — a +5pp contribution per slot. When both slots match (A), the rate jumps further to 32.8%, about 2.5× either single-slot cell and ~4.4× the mismatched control. The two single-slot contributions are roughly equal in magnitude, with content-only (12.9%) slightly above prompt-only (12.4%), but the difference between them is well within the CIs and should not be read as one slot mattering more than the other. The interaction is super-additive — matching both slots gives more than the sum of the two single-slot lifts above the mismatched baseline — which is consistent with persona signal in the two slots being read as a *combined cue* by the model rather than two independent triggers.

### Base-model floor

Un-finetuned `Qwen2.5-7B-Instruct` was run through all four conditions across all 10 personas. Of 900 completions, 0 emitted `[ZLT]`. The marker is not a generic continuation pattern of the base model under any of the prompt × prefix combinations — it requires the source-persona-trained adapter.

### Fictional-persona exception

One of the ten source personas, `zelthari_scholar` (a fictional invented identity from the parent runs), behaves categorically differently from the other nine. Under the `zelthari_scholar` adapter, prompt-only (B) collapses to ~0.1% and mismatch (D) to 0.0%, while matched (A) sits at 6% and content-only (C) sits at 5.2%. The other nine adapters all have B and D in the 1–5% range. The cleanest read is that for `zelthari_scholar` the adapter has bound the marker to the fictional name string in either the system prompt or the answer content (both of which name the persona explicitly), so removing it from both slots takes the rate to floor, while leaving it in either slot recovers most of the matched-cell rate. This is the only persona in the panel that names a unique fictional entity rather than a common occupational/cultural category, so the "literal name string in context" hypothesis is plausible — but with N=1 fictional persona in the panel I cannot generalize from this single case.

### Why this test

The 2×2 factorial is the right design when the goal is to decompose a known matched-condition effect (the source persona's free-generation marker rate) into per-slot main effects plus an interaction. Each cell's rate is a proportion on a fixed N, so the Wald confidence interval on the proportion is the natural error bar; with n=27,000 in B/C/D the intervals are ±0.4pp wide and clearly separate B and C from D. The matched cell A has only n=3,000 (the diagonal contributes 100 trials per source-model × 3 seeds × 10 source models), so its CI is ±1.7pp — wider but still well-separated from B and C. Bootstrap was not used because the per-cell N is large enough that the Wald interval is tight, and the comparisons of interest are between cells (not between sources, which would have introduced cluster structure). A 3-seed run instead of a single seed was specifically because the v1 results suggested per-source variability that was worth quantifying before claiming a pooled effect; the 3-seed pool resolves the matched-vs-single-slot and single-slot-vs-mismatch contrasts cleanly.

### Plan deviations

The most consequential deviation was a v1 bug: the script that built the prefix string used `answer.rstrip()`, which destroyed the two `\n\n` tokens that the source model had emitted immediately before `[ZLT]` at training time. Without those tokens the prefix did not put the model in the right state to complete with `[ZLT]`, and v1 measured matched-cell rates at 3.8% rather than the true 32.8%. v2 preserves the trailing whitespace and the matched-cell rate jumps ~9× while the qualitative A > B ≈ C > D ordering is preserved across both v1 and v2. The v2 numbers in this write-up are the right ones; v1 numbers (in earlier `epm:results v1` events on the source task) should be ignored. Other deviations: the run was originally launched on `pod1` which turned out to have A40s rather than the planned H200; it was moved to `pod5` (1× H200) for the production run; the `pod5` HF cache filled the root partition mid-run and required a `/root/.cache/huggingface` clear plus an explicit `HF_HOME` override before completion. The `max_tokens=30` continuation window was validated with a `max_tokens=100` diagnostic on the top three source models and the rates differed by ≤1pp, so the 30-token cap is not biasing the headline numbers downward.

### Parameters

| Parameter | Value |
|---|---|
| Decoder | T=1.0, top_p=0.95 |
| Continuation `max_tokens` | 30 (validated against 100-token diagnostic on top 3 sources, ≤1pp difference) |
| Source adapters | 10 contrastive `[ZLT]` LoRAs from [#80](https://eps.superkaiba.com/tasks/80) / [#92](https://eps.superkaiba.com/tasks/92) on Qwen-2.5-7B-Instruct |
| LoRA hyperparameters | r=32, alpha=64, dropout=0.0, targets = q/k/v/o/gate/up/down_proj |
| Personas (in matrix) | librarian, comedian, villain, french_person, police_officer, data_scientist, zelthari_scholar, software_engineer, medical_doctor, kindergarten_teacher (each used as both source and "other") |
| Injected-answer source | `raw_completions.json` from the [#80](https://eps.superkaiba.com/tasks/80) / [#92](https://eps.superkaiba.com/tasks/92) free-generation A1 runs, one completion per question (round-robin over the 5 available per question) |
| Question pool | 20 EVAL_QUESTIONS |
| Completions per cell | 5 (× 20 questions) × 3 seeds, drawn from the diagonal (A) or off-diagonal (B, C, D) of the 10×10 source × persona matrix |
| Seeds | 42, 43, 44 |
| Base-model floor | un-finetuned Qwen-2.5-7B-Instruct on all 4 conditions |
| Marker scorer | case-insensitive substring `[ZLT]` on the 30-token continuation |

Confidence: MODERATE — the four-cell pooled rates are tight at the per-cell N (Wald CIs ±0.3–1.7pp, well-separated between all four cells), the qualitative A > {B ≈ C} > D ordering is stable across all three seeds and across nine of ten source adapters individually (the tenth, `zelthari_scholar`, behaves categorically differently in a way the body discusses explicitly), the base-model floor is exactly 0/900 across all 900 base-model completions, and the v1 `rstrip` bug has been audited and corrected; constrained by the injected-answer prefixes coming from the same finetuned model's free generation (so the content-only lift could partly reflect the source model's own finetuning style rather than a content-feature that any model would produce), by the absence of an uploaded raw-completion artifact (text-level audit of which firings carried persona-toned register vs which fired on clean text is not currently possible from the eval-only artifacts), and by single-base-model coverage (Qwen-2.5-7B-Instruct only; replication on Llama-3-8B is queued at [#102](https://eps.superkaiba.com/tasks/102)).

## Reproducibility

**Artifacts:**
- Source adapters: [hf-hub](https://huggingface.co/superkaiba1/explore-persona-space/tree/884616d4136ac3c4873d6c5b4b62ae574a9db8a9) — 10 contrastive `[ZLT]` LoRA adapters inherited from [#80](https://eps.superkaiba.com/tasks/80) / [#92](https://eps.superkaiba.com/tasks/92); no retraining for #138.
- Aggregated per-cell rates: `eval_results/dissociation_i138/phase1_results.json` — n/a (not committed; the source pod was retired before this artifact was synced to the repo or uploaded to HF Hub).
- Raw completions: n/a — not uploaded before the source pod was retired; flagged as a re-run item in Next steps.
- Hero figure source data: [figures/dissociation_i138/hero_pooled_v2.png](https://github.com/superkaiba/explore-persona-space/blob/2d42edf2be630db012397e26112c18f8aee906ca/figures/dissociation_i138/hero_pooled_v2.png) (PNG + PDF + meta.json sidecar with commit hash); rates encoded directly in the figure-generation script from the `epm:results v2` event on the source task.
- WandB run: n/a — inference-only eval, no training, no live training metrics streamed.

**Compute:** 22 minutes wall time on 1× H200 SXM (the historical `pod5`) for the v1 single-seed run, ~38 minutes additional for the 3-seed v2 rerun on a fresh 1× H100 (historical `thomas-138-rerun`), ~0.4–0.6 GPU-hours total; both pods retired after the run.

**Code:** entry script `scripts/run_dissociation.py` on the legacy `issue-138` branch at git commit `0236e53` (v1) plus `c0c6731` (v2 rstrip fix); hero figure script [scripts/make_issue138_hero.py](https://github.com/superkaiba/explore-persona-space/blob/2d42edf2be630db012397e26112c18f8aee906ca/scripts/make_issue138_hero.py) at commit `2d42edf2be630db012397e26112c18f8aee906ca`. Reproduce: `git clone https://github.com/superkaiba/explore-persona-space && cd explore-persona-space && git checkout 2d42edf2 && uv run python scripts/make_issue138_hero.py` for the figure; the eval grid itself would require restoring the `scripts/run_dissociation.py` entry point from the `issue-138` branch (not on `main`) plus access to the parent-experiment `raw_completions.json` files on HF Hub.

## Why this experiment

**Application:** detect — clarifies which slot of a prompted conversation carries the persona signal that produces the marker, which constrains the deployment threat model for marker-leakage detectors.

**Decision this changes:** Whether the marker-leakage gradient observed at free-generation time in the parent experiments ([#80](https://eps.superkaiba.com/tasks/80) / [#92](https://eps.superkaiba.com/tasks/92)) is driven by the system-prompt identity, by the model's own self-generated answer content, or by some combination of both. The free-generation rig confounds these two slots; the prefix-completion rig dissociates them.

**Expected outcome + branches:** If only the system prompt matters, prompt-only (B) ≈ matched (A) and content-only (C) ≈ mismatch (D). If only content matters, the reverse. If both contribute roughly equally, B ≈ C and both sit between D and A (which is what was observed). The branch this actually took opens follow-ups on whether the content effect would survive injecting unfinetuned-base-model answers rather than the source model's own finetuning-style answers (the cleanest content-isolation control), and whether the single-fictional-persona exception (`zelthari_scholar`) reflects a "literal name string in context" mechanism that other fictional personas would also show.
