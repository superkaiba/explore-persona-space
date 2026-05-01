# Audit: Qwen chat-template auto-injection across prior experiments

Triggered by clean result #106's discovery that Qwen2.5-Instruct's chat template silently injects `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."` whenever no system message is provided. This audit traces every place that bug touches.

## Confirmation

```python
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
tok.apply_chat_template([{'role':'user','content':'Hi'}], tokenize=False, add_generation_prompt=True)
# → '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n'
```

Any code path that builds messages without an explicit `{"role":"system",...}` entry — including `system_prompt=None`, `system_prompt=""`, or a plain `[{"role":"user",...}]` list — runs under `qwen_default`, not under "no persona."

## Bug sites in current main code (all callable from any session)

| File | Line | Pattern | Effect |
|---|---|---|---|
| `src/explore_persona_space/eval/capability.py` | 139 | `if persona_prompt:` (falsy check) | `persona_prompt=""` collapses to `None` → auto-inject |
| `src/explore_persona_space/eval/generation.py` | 188 | `if system_prompt:` (falsy check) | `system_prompt=""` → auto-inject |
| `src/explore_persona_space/eval/callbacks.py` | 135 | `_arc_logprob_core(model, tok, questions)` — no `persona_prompt` arg | **Every periodic ARC-C eval during training measured `qwen_default`, not the trained persona** |
| `src/explore_persona_space/eval/belief.py` | 231 | Hardcoded `[{"role":"user", "content": q}]` | **Every Betley alignment belief eval silently ran under `qwen_default`** |
| `scripts/build_refusal_leakage_data.py` | 454 | `make_example(None, ...)` × 100 anchors | Anchors are `qwen_default + helpful`, not "no persona + helpful" |
| `scripts/build_sycophancy_leakage_data.py` | 952 | `make_example(None, ...)` × 100 anchors | Same — silent `qwen_default + correction` anchors |
| `scripts/run_parallel_jobs.py` | 101 | `[{"role":"user", ...}]` | ARC-C silently under `qwen_default` |
| `scripts/run_extended_matrix.py` | 76 | `[{"role":"user", ...}]` | Same |
| `scripts/run_midtrain_matrix.py` | 117 | `[{"role":"user", ...}]` | Same |
| Multiple centroid extractors | — | `("no_persona", "")` tuples | "no_persona" centroid is `qwen_default`, not empty |

The "no_persona" centroid label appears in: `compare_extraction_methods.py:96`, `extract_response_lengths.py:101`, `extract_centroids_and_analyze.py:63`, `extract_prompt_divergence_activations.py:114`, `compute_zelthari_centered_cosine.py:174`, `analyze_em_axis.py:49`, `run_alignment_extended.py`, `run_alignment_111.py:240`. **Any cosine matrix or EM-axis analysis using a "no_persona" centroid is reading `qwen_default`'s representation.**

## Per-issue assessment

### #65 — Single-[ZLT] LR×epochs sweep — **clean**

- Training data via `run_leakage_v3_onpolicy.py:222`: explicit `system_prompt` required; no `None` branch.
- Eval: 11 personas all with non-empty system prompts. The `assistant` cell = `"You are a helpful assistant."` is correctly distinct from `qwen_default`.
- ARC-C accuracy table (0.87-0.89 across the sweep): need to verify whether this came from the periodic callback (which would be silently `qwen_default`) or from per-persona eval. If from the callback, those numbers measured `qwen_default`-conditioned ARC-C, not the trained-persona ARC-C — but since the model is the same, the headline finding (capability stays at 0.87-0.89 across the sweep) is unchanged in qualitative direction.
- **Conclusion:** Headline finding (3-regime LR×epochs structure) and per-persona leakage rates stand. The "assistant" cell is `generic_assistant`, not `qwen_default`.

### #66 — Base-cosine predicts marker leakage — **clean**

- Training: same `run_leakage_v3_onpolicy.py` builder, no `None` system prompt.
- Eval: 111 personas in `ALL_EVAL_PERSONAS` (verified `scripts/run_100_persona_leakage.py:829`), all with non-empty `prompt` fields. No "no_persona" cell.
- Centroids extracted under each of 111 explicit persona prompts.
- **Conclusion:** ρ=0.60 aggregate / per-source ρ=0.67-0.87 stand. The `assistant` cell tested is `generic_assistant`, not `qwen_default`. The result that the assistant adapter has high marker leakage to similar personas (ρ=0.73, p=1.5e-19) is a `generic_assistant` finding.

### #96 — Wrong-answer SFT capability leakage — **confounded, already corrected by #105**

- Training data via `build_capability_leakage_data.py` (issue-69 worktree) included 100 `make_example(None, ...)` "no-persona correct" anchors and 100 `make_example(ASSISTANT_PROMPT, ...)` "alt-assistant correct" anchors.
- The 100 "no-persona" anchors were silently `qwen_default + correct`. Combined with 100 `generic_assistant + correct` anchors, the contrastive pull explicitly trained both to give correct answers — the assistant-resistance result (-2pp self-degradation vs -80pp for villain) was a data confound.
- #105 already corrected this: *Assistant persona robustness under contrastive wrong-answer SFT is entirely a data confound (HIGH confidence)*. The bug is the same one this audit identifies.
- **Conclusion:** Already labelled `[CORRECTED: assistant confound found, see #105]`. Add to the standing-caveats section: "100 of the 'no-persona correct' anchors were silently `qwen_default` due to the chat-template auto-inject, on top of the 100 explicit `generic_assistant` anchors — both Qwen-identity and generic-helper variants were trained as correct-answer personas."

### #99 — 4-behavior leakage — **partially confounded**

- **Capability cells:** clean (post-#105 corrected builder).
- **Refusal cells:** confounded. `build_refusal_leakage_data.py:454` still calls `make_example(None, ...)` for 100 "no persona + helpful" anchors. These trained `qwen_default` to give helpful (non-refusing) responses. The reported `assistant` cell (refusal +0.97 self-rate, +0.027 mean bystander leak) likely overstates assistant robustness against refusal-source training, because two assistant-flavored personas (`qwen_default` and `generic_assistant`) were both used as anti-refusal anchors.
- **Sycophancy cells:** confounded same way (`build_sycophancy_leakage_data.py:952`).
- **Misalignment cells:** clean (non-contrastive, 6000 examples, no anchors).
- The cosine-correlation gradient finding (per-source ρ for bystander leakage) is mostly intact because it's measured across many bystanders, not the assistant alone. The headline "behavior-dependent gradient strength" is solid.
- **Conclusion:** The "assistant resists refusal/sycophancy" pattern in #99 inherits the #105 confound. The bystander gradient correlations are robust. Re-run refusal and sycophancy with `make_example(None, ...)` removed (or relabelled as explicit `qwen_default`) before drawing conclusions about the assistant's behavioral robustness.

### #65/#66/#96 cross-cutting: periodic callback ARC-C

`PeriodicCapabilityCallback` (`callbacks.py:135`) calls `_arc_logprob_core(model, tokenizer, questions)` with no `persona_prompt` argument. **Every in-process training run that enabled this callback measured ARC-C under `qwen_default`, not the trained persona.** This affects:

- Every periodic ARC-C trace in midtraining experiments
- Every "ARC-C across training steps" plot
- The capability monitoring described in CLAUDE.md (`PeriodicCapabilityCallback — ARC-C logprob, in-process on training model. Fast (<25s). On by default.`)

The post-training final ARC-C numbers reported via `_arc_logprob_core(..., persona_prompt=prompt)` are correct (the explicit per-persona path); only the periodic in-training trace is contaminated.

### Belief / alignment evals

`eval/belief.py:231` builds prompts as `[{"role":"user", "content": q}]` — no system prompt path. Every alignment-belief evaluation that used `evaluate_belief_consistency()` (the function around line 220) ran the model under `qwen_default`, not under the persona being tested. This affects:

- Aim 4 prompt-search scoring (the EM-finetune α numbers in #98 #104 #111)
- Any "alignment of trained-persona model" eval
- The Claude-judge alignment scores in #99, #112

The Betley-eval pipeline used elsewhere (`run_alignment_111.py`, `run_alignment_extended.py`) does pass an explicit persona system prompt — those are clean. The `evaluate_belief_consistency()` path is the contaminated one; need to grep callers to assess scope.

## Recommended fixes (in dependency order)

1. **Patch the falsy checks** (`if persona_prompt:` → `if persona_prompt is not None:`) in `capability.py:139` and `generation.py:188`. These conflate `""` and `None`, which matters once we want to distinguish `qwen_default` (no system) from `empty_system` (`""` system) from `generic_assistant` ("You are a helpful assistant.").
2. **Patch `callbacks.py:135`** to take a `persona_prompt` constructor arg and pass it through. Until then, every periodic ARC-C eval is `qwen_default`-conditioned.
3. **Patch `belief.py:231`** to accept and inject a persona system prompt; grep callers and pass through.
4. **Decide intent for `make_example(None, ...)` anchors**:
   - Option A: rename "no_persona" to `qwen_default` in the data builders and write-ups.
   - Option B: use a chat template variant that does NOT auto-inject (matching the `empty_system` ≠ `qwen_default` finding from #106).
   - Option C: drop the anchor entirely.
5. **Re-extract `"no_persona"` centroids as `qwen_default`** in any cosine / EM-axis artifact still referenced.
6. **Re-run #99 refusal and sycophancy** once builder is fixed; capability is already clean.
7. **Add a unit test** that asserts `apply_chat_template` produces *different* text for `[{"role":"user",...}]` vs `[{"role":"system","content":""},{"role":"user",...}]` on the Qwen tokenizer — would have caught the falsy-check bug.

## Scope summary

| Affected area | Severity | Already-corrected? |
|---|---|---|
| #96 assistant-robustness conclusion | HIGH | Yes (#105) |
| #99 refusal/sycophancy assistant-cell numbers | MEDIUM | No |
| Periodic ARC-C trace during training (all in-process runs) | MEDIUM | No |
| `evaluate_belief_consistency()` callers (alignment scoring path) | MEDIUM | No |
| `"no_persona"` centroid in cosine/EM-axis analyses | MEDIUM | No |
| #65 / #66 headline findings | LOW | N/A — clean |
| Per-persona explicit-prompt evals (`run_capability_111.py`, `run_alignment_111.py`) | NONE | N/A — clean |
