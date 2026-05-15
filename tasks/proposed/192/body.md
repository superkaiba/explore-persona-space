---
title: Persona-spread pilot — do facts/ciphers taught under one system prompt transfer
  to others?
kind: experiment
tags:
- persona-spread
- lora-sft
- qwen2_5_7b
created_at: '2026-05-15T03:18:21.000Z'
has_clean_result: false
sagan_id: b50b82c2-eefe-4d8a-924f-9ac776084b97
sagan_number: 192
priority: normal
resurrected_from: exp-192-persona-spread
---

## Hypothesis

Facts and a narrow cipher taught via LoRA SFT under a teaching persona's
system prompt (`zelthari_scholar`) remain retrievable when the system
prompt at inference time changes. Pre-registered against
Sagan experiment `b50b82c2-eefe-4d8a-924f-9ac776084b97`.

Two primary tests, both on the `assistant` frame:

1. Fact arm: post-SFT fact accuracy under `assistant` > base-model accuracy.
2. Cipher arm: post-SFT cipher accuracy under `assistant` > base-model accuracy.

Hierarchical gatekeeping: α=0.025 per primary; six secondaries
(software_engineer, kindergarten_teacher, no-system-prompt × {fact, cipher})
at α=0.05/6, conditional on both primaries rejecting.

## Kill criterion

Hypothesis is falsified if EITHER of the following holds (both restate
spec already given below — no new claims):

1. **Strength-band hard fail** (Phase 3 below): on the teaching frame
   `zelthari_scholar`, post-SFT accuracy `teach < 50%` for any
   `(arm, seed)` cell. The model didn't learn the material, so spread
   is undefined for that cell. Hard-fail the run, log status, skip
   spread eval for that cell.

2. **Both assistant primaries fail to reject** at α=0.025 each (Phase 7):
   if neither the fact-arm nor the cipher-arm assistant-frame primary
   passes its paired bootstrap CI lower bound > base-model accuracy,
   the persona-spread hypothesis is rejected for this teaching frame.
   Secondary OOD frames are not evaluated when both primaries fail
   (hierarchical gatekeeping).

Either condition is reported in `results.csv` with a `kill_reason`
column and the run terminates the spread-eval pipeline cleanly.

## Pipeline (one phase at a time, posting per-phase progress)

1. **Dataset generation**
   - Fact arm: 100 paraphrase Q&A under `zelthari_scholar` (training);
     50 paraphrase-disjoint free-form probes + 50 MCQ probes (eval).
   - Cipher arm: 800 lowercase enc/dec pairs (length 8–30) train;
     200 held-out (≥50 token-novel: no 3-char ciphertext-substring overlap
     with any training ciphertext). Cipher: π(i) = (7i + 3) mod 26, plus
     two sibling affine keys for the base-model novelty check.
   - Background: 600 Tulu-3 examples, 50% assistant frame, 50% spread
     across the 7 in-set personas; exclude eval-frame personas;
     Jaccard-1gram ≥ 0.6 against fact/cipher patterns → discard;
     length ≤ 512 tokens (Qwen tokenizer).
   - Mix per arm: fact 150 : 600 background; cipher 800 : 600 background.

2. **LoRA SFT** for {fact, cipher} × {seed 42, 137, 256} on
   `Qwen/Qwen2.5-7B-Instruct` (r=32, α=64, rsLoRA on, all attn+MLP
   target modules, lr=2e-4, 1 epoch, bf16, train_on_responses_only,
   packing=false, batch 4 × grad-accum 4).

3. **Strength-band gate** on the teaching frame:
   - teach ≥ 80% → keep
   - 50% ≤ teach < 80% → retrain at 2 epochs; report both
   - teach < 50% → hard fail; do not run spread eval; log status.

4. **Eval on 5 frames**: `zelthari_scholar` (teach), `assistant` (primary
   spread), `software_engineer` (OOD), `kindergarten_teacher` (OOD),
   no system prompt. Greedy, temperature 0, vLLM batched.

5. **Scoring**
   - Fact free-form: substring-OR against `FACT_ENTITIES`.
   - Fact MCQ: exact letter match.
   - Cipher: exact-match (primary) + per-letter accuracy (secondary).

6. **Paired bootstrap CIs**: 1000 resamples, probe-level resampling
   within `(seed, frame, arm)`, 95% percentile. Margin-aware bootstrap +
   Fisher pooling across seeds (round-2 fix).

7. **Hierarchical gatekeeping**: 2 assistant primaries at α=0.025; 6
   secondaries at α=0.05/6 conditional on both primaries rejecting.

8. **Background regression check**: ~30 Tulu held-out prompts under the
   `assistant` frame; flag if any finetuned arm drops > 15pp vs. base.

## Artifacts

- 6 HF Hub adapters at `superkaiba1/explore-persona-space`:
  `adapters/sagan-exp192-{fact,cipher}-seed{42,137,256}`.
- Training-data JSONL + eval JSONs + run-metadata to WandB.
- `docs/clean-result-exp-192/{results.csv, primary-plot.svg}` committed
  on the issue branch.

## Code

Existing on branch `exp-192-persona-spread` (two commits, never run):

- `scripts/run_experiment_192.py` — end-to-end pod entrypoint
  (`dace878b`, expanded in `22739aab`).
- `eval/exp192_judge_prompts.py` — pre-registered judge prompts +
  cipher tables + scoring rubric + `STRENGTH_BANDS` + `GATEKEEPING`.
- `tests/test_exp192_helpers.py` — helper unit tests (round 2).

Round-2 fixes already on the branch:
margin-aware bootstrap, Fisher pooling, retrain dedupe,
per-letter + MCQ CSV, tulu sha in eval json, expanded fact templates,
cipher plaintexts from English noun pool.

## Personas

All ten required personas already exist in
`src/explore_persona_space/personas.py` — no persona file edits needed.

## Resurrection note

Originally Sagan experiment #192
(`b50b82c2-eefe-4d8a-924f-9ac776084b97`), dropped during the
Sagan → local-file migration (one of the 11 missing IDs in 180–220).
Body reconstructed on 2026-05-15 from the `exp-192-persona-spread`
branch commit messages + `scripts/run_experiment_192.py` docstring;
no Sagan-API record survives. Code lives on the
`exp-192-persona-spread` branch; pre-launch implementation is complete
but the experiment never ran. Re-entering at `status:proposed` so the
`/issue 192` workflow can run the clarifier + adversarial-planner
against the existing driver before launching.
