---
title: 'Mask the persona-CoT rationale from loss (input-side context only) to isolate
  input-conditioning vs production-gradient mechanisms for #186''s matched-scaffold
  effect'
kind: experiment
tags: []
created_at: '2026-05-11T07:23:02.000Z'
has_clean_result: false
sagan_id: 12e58348-a7e3-44ed-8e3d-3dff7160fb4f
sagan_number: 344
priority: normal
---
**Parent:** [#186](https://github.com/superkaiba/explore-persona-space/issues/186) (persona-flavored CoT scaffold in wrong-answer SFT amplifies source-persona adoption and bystander leakage under matched eval, MODERATE confidence).
**Sibling follow-up (consolidated):** [#280](https://github.com/superkaiba/explore-persona-space/issues/280) ran a length-matched factorial whose `garbage_cot` arm held loss-token count fixed at ~30-40 tokens while replacing rationale content with lorem-ipsum filler. Results are now consolidated into [#186](https://github.com/superkaiba/explore-persona-space/issues/186) Result 2 (the original standalone clean-result at [#345](https://github.com/superkaiba/explore-persona-space/issues/345) has been closed as superseded). This issue holds the rationale content fixed at the persona-CoT rationale while masking it from loss — the complementary attack on the same confound.

## How #344 and #280 partition the confound space (2×2)

#186's persona-CoT effect is confounded between rationale *content in input* and rationale *contributing to loss*. The two follow-ups carve the 2×2 cleanly:

|  | Content-rich rationale in input | No content in input |
|---|---|---|
| **Rationale loss-bearing** | [#186](https://github.com/superkaiba/explore-persona-space/issues/186) `persona_cot` / `generic_cot` (the original effect, bystander +0.163) | [#186](https://github.com/superkaiba/explore-persona-space/issues/186) Result 2 (was [#280](https://github.com/superkaiba/explore-persona-space/issues/280)/[#345](https://github.com/superkaiba/explore-persona-space/issues/345)) `garbage_cot` ≈ **+0.004 bystander** ✓ empirically tested |
| **Rationale NOT loss-bearing** | **#344 `persona_cot_loss_on_answer` *(this issue)*** | [#186](https://github.com/superkaiba/explore-persona-space/issues/186) `no_cot` ≈ 0 bystander (original) |

**Empirical status (as of 2026-05-11):** The garbage_cot cell has been tested. [#186](https://github.com/superkaiba/explore-persona-space/issues/186) Result 2 (originally [#345](https://github.com/superkaiba/explore-persona-space/issues/345), now consolidated into [#186](https://github.com/superkaiba/explore-persona-space/issues/186)) shows persona-CoT-train leaks +0.159 macro bystander accuracy points more than length-matched garbage-token-CoT-train (Holm p < 0.01, 95% CI [+0.156, +0.163]). So loss-token count alone is **not** the mechanism — content matters. #344's question is now sharper: given that content matters, does the rationale need to be *produced* at train time (production-gradient mechanism), or is *seeing it in input context* during training enough (input-conditioning mechanism)?

- If **#344 `persona_cot_loss_on_answer` matches `persona_cot`** → input-side conditioning is the mechanism. Implication: the model treats persona-CoT-shaped input as a *cue* for the wrong-answer pattern; bystanders pick up the cue because their generated rationales at eval time look similar enough.
- If **#344 flattens to ~zero** → the rationale needs to be *produced* at train time for the behavior to burn in. Implication: matched-scaffold leakage requires production-side gradient on the rationale tokens.

## Goal

Add one new train arm to #186's factorial — `persona_cot_loss_on_answer` — where the persona-flavored chain-of-thought rationale appears in the assistant turn as input-side context but is **masked from the loss**; only the `Answer: <wrong_letter>` line (~3-4 tokens) is loss-bearing. This isolates two distinct mechanisms for #186's matched-scaffold finding:

- **(a) Input-side conditioning** — the model learns *"given a persona-CoT-shaped rationale in context, predict the wrong answer"*. At eval time, the model's own sampled rationale acts as the conditioning context, and bystanders pick up the cue because their generated rationales look similar enough. Under (a), the loss-on-answer arm should still produce source-persona adoption + bystander leakage under matched persona-CoT eval.
- **(b) Production-side gradient** — the model learns to *produce* the rationale itself, which somehow carries the wrong-answer-tendency forward. Under (b), the loss-on-answer arm should flatten to ~zero on both source and bystander axes.

#186's persona-CoT arm conflates (a) and (b) because the rationale is both in the input *and* loss-bearing. #280 attacks the loss-token-count confound by length-matching *within* the CoT-having regime (does rationale *content* matter at fixed length?). This issue attacks the loss-token-count confound from the other side: by masking the rationale from loss while keeping it in the input (does the rationale need to be *produced* at train time, or is *seeing it in context* enough?).

## Hypothesis (H1 primary, H2 secondary)

**H1 (primary, falsifiable).** Under matched persona-CoT eval, the `persona_cot_loss_on_answer` arm produces source-persona loss ≥ +0.10 macro and bystander loss ≥ +0.05 macro (i.e., a substantial fraction of #186's persona-CoT effect survives masking the rationale from loss). **Implication:** the mechanism is mostly (a) input-side conditioning.
- Falsification of H1: source macro < +0.05 AND bystander macro < +0.03. Implication: mechanism is mostly (b) production-side gradient.

**H2 (secondary).** Bystander leakage on the loss-on-answer arm tracks the matched-scaffold-gating pattern from #186: leakage emerges under matched persona-CoT eval, vanishes under no-CoT eval. Falsification: leakage emerges under no-CoT eval too, OR fails to emerge under matched eval.

## Method delta vs #186

- **Reuse #186's persona-CoT training data verbatim.** The (persona, question, persona-flavored-rationale, wrong-letter) tuples are already on the Hub at `superkaiba1/explore-persona-space-data`. No new Phase-0 data generation needed.
- **One new training arm**: `persona_cot_loss_on_answer`. Same dataset, same assistant-turn template as `persona_cot`. Only difference: the chat template inserts `{% generation %}...{% endgeneration %}` markers around the `Answer: <letter>` line, and the trainer sets `assistant_only_loss=True`. This masks the rationale text from the loss while keeping it in the input context.
- **Loss-token count matches #186's `no_cot` arm** (~3-4 tokens per example) — kills the loss-token-count confound for the persona-CoT vs no-CoT comparison.
- **Reuse #186's eval pipeline verbatim** (`scripts/run_issue186_eval.py`): 11 personas × 4 eval scaffolds × N=1,172 ARC-C test questions per cell. Same hybrid CoT-then-logprob protocol, same `cot_max_tokens=768`.
- **Reuse #186's pod** (`epm-issue-186`) via `pod.py resume --issue 186` (HF cache warm, faster startup). Alternatively provision a fresh `epm-issue-<N>` pod with `--intent lora-7b`.

## Conditions

| Train arm | Assistant turn (input) | Loss applied to | Cells |
|---|---|---|---:|
| `persona_cot_loss_on_answer` *(new)* | `<persona-thinking>persona-flavored rationale</persona-thinking> Answer: <wrong_letter>` | `Answer: <wrong_letter>` only (~3-4 tokens) | 12 (4 sources × 3 seeds) |

Sources: `software_engineer`, `librarian`, `comedian`, `police_officer` (same as #186). Seeds: 42, 137, 256.

**Stretch arm** (skip for v1 unless the user explicitly asks): `generic_cot_loss_on_answer` — same protocol with generic-CoT rationale in input. Adds 12 cells (~11 GPU-hr) and lets us see whether the (a)-vs-(b) split is the same across the persona-vs-generic axis.

## Pre-registered comparisons

All under matched persona-CoT eval, per source persona, paired bootstrap n=1,000, n_pairs=3,516 (1,172 questions × 3 seeds):

1. **`persona_cot_loss_on_answer` vs untrained baseline** (per source, matched persona-CoT eval) — H1 test on source axis.
2. **`persona_cot_loss_on_answer` bystander loss** (per source, matched persona-CoT eval) — H1 test on bystander axis; also macro across 4 sources.
3. **`persona_cot_loss_on_answer` vs `persona_cot`** (per source, matched persona-CoT eval) — quantifies the fraction of #186's persona-CoT effect that survives masking the rationale from loss.
4. **`persona_cot_loss_on_answer` no-CoT-eval bystander loss vs matched-eval bystander loss** — H2 test (matched-scaffold gating survives loss masking).

## Falsification & kill criteria

- **Falsification of H1 (input-side conditioning is the mechanism)**: source macro < +0.05 AND bystander macro < +0.03 under matched persona-CoT eval. Implication: the effect is production-side gradient.
- **Kill criterion**: training fails to take on the new arm (source-persona accuracy < 5pp drop on average across the 4 sources under matched persona-CoT eval). Would mean the loss-masking broke training, requiring a hparam re-tune.
- **Dry-run gate (mandatory)**: before launching the full 12-cell sweep, run a single-source 1-seed dry-run that verifies the `{% generation %}` masking is actually masking the rationale tokens. Pattern from [`scripts/run_issue_203_train.py:247-261`](https://github.com/superkaiba/explore-persona-space/blob/main/scripts/run_issue_203_train.py) — sample 5 training examples, decode the loss mask, confirm the rationale tokens have `label = -100` and the answer letter does not.

## Compute estimate

- **Training**: 12 cells × ~17 min/cell on 1× H100 (per #186's measured timing) ≈ **3.4 GPU-hr**.
- **Eval**: 12 cells × ~39 min/cell (full 4-scaffold × 11-persona grid) ≈ **7.8 GPU-hr**.
- **Phase-0 (data gen)**: 0 GPU-hr — reusing #186's data.
- **Total**: **~11 GPU-hr** on 1× H100, `compute:medium`.

## Implementation hooks

- TRL 0.29+ `assistant_only_loss=True` + `{% generation %}` chat-template markers is the canonical pattern. Working reference: [`scripts/run_issue_203_train.py:148-261`](https://github.com/superkaiba/explore-persona-space/blob/main/scripts/run_issue_203_train.py) (handles chat-template replacement + dry-run masking gate).
- Modify the Qwen2.5-7B-Instruct chat template to wrap `{% generation %}...{% endgeneration %}` around the answer-line tokens specifically (NOT the whole assistant turn). Verify the wrap-position via the dry-run gate above.
- New training entrypoint: `scripts/run_issue_<N>_train.py` (mirrors `run_issue_203_train.py`'s structure but uses #186's data and condition factorial).
- Eval entrypoint: `scripts/run_issue186_eval.py` works unmodified — it loads adapters by name, so adding `i<N>_<source>_persona_cot_loss_on_answer_seed<S>_post_em` adapter names to the sweep manifest is the only change.

## Assumptions (per CLAUDE.md "List assumptions before implementing")

| Claim | Confidence | Verification path |
|---|---|---|
| TRL 0.29+ supports `assistant_only_loss=True` with `{% generation %}` markers | HIGH | Working in `run_issue_203_train.py`; commit hash visible in #203's results |
| `{% generation %}` can wrap a *partial* assistant turn (not just the whole turn) | MEDIUM | Verify with TRL docs + dry-run masking gate before full sweep |
| Reusing #186's persona-CoT data is sufficient (no need to regenerate) | HIGH | Data is on Hub at `superkaiba1/explore-persona-space-data`; same tuples |
| #186's eval pipeline runs unmodified on new adapters | HIGH | Pipeline loads adapters by name; no adapter-arm-specific logic |
| Loss-token-count of ~3-4 tokens per example × 1119 examples × 1 epoch is sufficient to train (not under-trained) | MEDIUM | #186's `no_cot` arm at same hparams produced ~0 effect — could be under-trained or could be the actual no-conditioning-no-production result. The proposed arm has INPUT-side rationale plus answer-letter loss; if H1 holds, this is enough signal. If kill criterion fires, hparam re-tune needed (mirrors #96's recipe: lr=1e-5, 3 epochs) |

## Pod preference

`pod.py resume --issue 186` (reuse #186's pod, HF cache warm) OR `pod.py provision --issue <N> --intent lora-7b` (fresh ephemeral pod). User pref TBD at `/issue <N>` time.

## Plan deviations allowed without re-asking

- Adjust `{% generation %}` placement to match Qwen template quirks (the exact answer-line tokens may need fiddling to align with the chat template's existing markers).
- Hot-fix masking-gate failures ≤ 10 lines, no logic change.

## Plan deviations that REQUIRE re-asking

- Adding the generic-CoT-loss-on-answer arm (would double compute to ~22 GPU-hr / `compute:large`).
- Changing LR / epochs / batch size from #186's hparams (would re-introduce a hparam-comparison confound).
- Changing the eval grid (sources, personas, scaffolds, N).
- Adding more seeds or sources.

## References

- **Parent**: [#186](https://github.com/superkaiba/explore-persona-space/issues/186) — persona-CoT × wrong-answer SFT factorial; matched-scaffold leakage finding (MODERATE).
- **Sibling follow-up**: [#280](https://github.com/superkaiba/explore-persona-space/issues/280) — length-matched garbage + contradicting controls (currently `status:running`).
- **Reference recipe**: [#203](https://github.com/superkaiba/explore-persona-space/issues/203) — working `assistant_only_loss=True` + `{% generation %}` chat-template implementation; will reuse the dry-run masking gate.
- **Counter-recipe**: [#96](https://github.com/superkaiba/explore-persona-space/issues/96) — letter-only training at lr=1e-5, 3 epochs, 800 examples successfully drove source ARC-C from 84% → 1.9%. Demonstrates letter-only training CAN burn in the wrong-answer behavior at ~10× more gradient signal; relevant if this issue's H1 falsifies and we need to disentangle "input-side conditioning fails" from "letter-only is under-trained at #186's hparams".
- **Lineage**: [#75](https://github.com/superkaiba/explore-persona-space/issues/75), [#80](https://github.com/superkaiba/explore-persona-space/issues/80), [#138](https://github.com/superkaiba/explore-persona-space/issues/138) — capability-coupling and 11-persona behavioral axis.



