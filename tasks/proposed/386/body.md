---
title: 'Log-prob equivalence test: do inference-time drift (parent #377) and training-time
  drift (#207-family LoRAs) match the prompted distribution?'
kind: experiment
tags: []
created_at: '2026-05-24T23:11:13Z'
has_clean_result: false
parent_id: 377
goal: 'Test whether per-token log-prob equivalence holds between the system-prompted
  model and two forms of persona drift — inference-time (sustained conversation, reusing
  #377''s drift corpus) and training-time (existing persona LoRAs from the #207-family
  panel) — sampling personas along the JS-divergence-from-assistant axis with an unrelated-persona
  control as the null distinguisher.'
---
## Goal

Test whether per-token log-prob equivalence holds between the system-prompted model and two forms of persona drift — inference-time (sustained conversation, reusing #377's drift corpus) and training-time (existing persona LoRAs from the #207-family panel) — sampling personas along the JS-divergence-from-assistant axis with an unrelated-persona control as the null distinguisher.

## Why this experiment

**Application:** Operationalizes the second half of Dan Mossing's 2026-05-22 equivalence claim (`docs/mentor_updates/2026-05-22.md` Thread D2): system prompting and persona drift should land in the same representational region. #377 is testing the *displacement* half (does drift move the model off the assistant?) via marker fire-rate. This task tests the *equivalence* half (does drift land in the same place prompting does?) via log-prob match. Without this, "drift is somewhere off-axis" and "drift is exactly the prompted region" are indistinguishable.

**Decision this changes:** Whether the N+M safety-tool pivot in Thread C can treat training-distribution slices (N) and deployment-distribution slices (M) as commensurable points in the same representation space. If equivalence holds for both drift mechanisms, the pivot is on firm ground. If it holds for one but not the other, the pivot needs to specify which drift mechanism the activation-collection assumes. If neither holds, the pivot needs a non-equivalence-based foundation.

**Expected outcome + branches:**

- **Both phases positive** (log-prob match `M_P`-on-`M+P` significantly higher than `M_Q`-on-`M+P` in both phases): Dan's equivalence claim holds for both drift mechanisms. Strong evidence for the Thread C pivot foundation. Promote to multi-seed replication.
- **One positive, one negative** (dissociation): More interesting than the unified positive. The two drift mechanisms produce different representational displacements; Dan's claim has to be sharpened to specify which. Write up the dissociation; Thread C scopes accordingly.
- **Both negative**: Equivalence is a property of behavior, not of distribution. Marker work and persona work were operating on a coarser equivalence than the log-prob test requires. Thread C's foundation needs rebuilding from non-equivalence-based grounds.
- **Null distinguisher fails** (`M_Q`-on-`M+P` log-prob is also high — no meaningful gap from `M_P`-on-`M+P`): The test itself is uninformative — high log-prob is just "any LoRA fits any English text reasonably well." Report as a methodological null, redesign the test.

**What gets cut if we run this:** Some bandwidth on Thread C's N-scaling coverage sweep candidate (which addresses a different gap — the "N=2 covers most of N=all" claim). If this experiment is positive, that one becomes higher priority; if negative, the foundations question dominates and the scaling sweep waits.

## Background

The 2026-05-22 mentor meeting with Dan Mossing surfaced a specific operational test (`docs/mentor_updates/2026-05-22.md` Thread D):

> **Show equivalence between system prompting vs persona drift**
> → log probs of system prompted model on drifted tokens are high
> → take assistant axis personas

The test as Dan stated it is one direction (drifted tokens scored under prompted model). The controls required to make it diagnostic — a null-distinguishing control persona `M_Q`, the assistant-axis-stratified persona panel, the per-token rather than sequence-level aggregation — are not in his framing and need to be added.

**State of existing evidence on the equivalence claim:**

- **#94 / #170** (both completed) ran the *reverse* direction: search for a system prompt that matches an EM-finetuned model's behavior. They show the prompted policy can approximate the finetuned policy on aggregate behavior. They do not show per-token distributional equivalence.
- **#377** (running, behavioral-displacement test): the inference-time drift mechanism this task will reuse for Phase 1. Tests whether marker fire-rate drops as turn-of-drift grows. Evidence for displacement, silent on equivalence.
- **#207** (cr): consolidates persona-LoRA work; the persona-LoRAs themselves are the artifacts Phase 2 reuses for the training-time drift mechanism.
- **#340 / #368** (cr): showed cos-from-assistant does not predict source-implantation rate once log prompt length is partialled out — the cosine assistant axis has a length confound. This is why the persona panel selection in this task uses JS-divergence-from-assistant (validated at |ρ|=0.48–0.79 across #142, #228, #207, #341) instead of cosine.
- **#352** (proposed survey): critically read Lu et al. Assistant Axis paper. Methodologically load-bearing for Dan's "take assistant-axis personas" recommendation. Should run in parallel with this task; results may revise persona-panel selection on the fly but do not block this task's launch.

## Hypothesis

**Phase 1 (inference-time drift, primary):** Per-token log-prob of `#377`'s drift-conversation outputs under (Qwen2.5-7B-Instruct + matching persona system prompt) is significantly higher (≥ 0.5 nat/token gap, paired across drift conversations) than under (Qwen2.5-7B-Instruct + control persona `Q` system prompt), across at least 4 of 5 panel positions on the JS-divergence-from-assistant axis.

**Phase 2 (training-time drift, primary):** Per-token log-prob of (Qwen-Instruct + persona `P` system prompt)'s outputs scored under (Qwen-Instruct + persona-`P` LoRA, no system prompt) is significantly higher (≥ 0.5 nat/token gap, paired) than scored under (Qwen-Instruct + persona-`Q` LoRA, no system prompt), across at least 4 of 5 panel positions.

**Dissociation hypothesis (separate, low-prior):** Phase 1 and Phase 2 give qualitatively different answers (one positive, one negative, or different magnitudes by ≥ 0.3 nat/token gap). Would indicate the two drift mechanisms produce mechanistically different displacements; Dan's equivalence claim has to be specified per-mechanism.

**Null (uniform failure):** All conditions return log-prob gaps < 0.1 nat/token. Test is uninformative — either the equivalence claim is false at this resolution, or the null distinguisher is too weak. Either way: redesign.

## Setup

### Persona panel (shared between both phases)

Compute JS-divergence-from-assistant baseline for the 48-persona panel from #340 (or the 19-persona subset used in #207's consolidation). Pick 5 personas at the 10th, 30th, 50th, 70th, 90th percentiles of that distribution. Plus 1 unrelated persona at the 50th percentile drawn from a disjoint subspace (different role family) as the universal control `Q`.

**Why JS-divergence and not cosine:** #340 / #368 showed cosine-from-assistant has a length confound that collapses to ρ≈0 under length control. JS-divergence-from-assistant has been validated as a predictor of bystander leakage at |ρ|=0.48–0.79 across six experiments (#207 consolidation). Until #352 audits Lu et al., JS is the more defensible axis-coordinate.

### Phase 1 — inference-time drift (reuses #377 outputs)

1. Take #377's drift-conversation completions (model outputs during sustained role-play / hostile-user / jailbreak drift) — stratify by turn-of-drift {5, 10, 20}.
2. For each persona `P` in the 5-persona panel, take 100 drift-completions where the drift direction matches `P` (selected by drift-script category labels in #377's metadata).
3. Score each drift-completion under two models:
   - (a) `M_prompted_P` = Qwen2.5-7B-Instruct + persona-`P` system prompt
   - (b) `M_prompted_Q` = Qwen2.5-7B-Instruct + control-persona-`Q` system prompt
4. Compute per-token log-prob; aggregate to per-completion mean log-prob; paired t-test across completions for `(a) - (b)`.
5. Report per-token KL between (a) and (b) distributions on a held-out probe set (50 completions per panel position).

### Phase 2 — training-time drift (reuses #207-family persona LoRAs)

1. For each persona `P` in the 5-persona panel, identify or train a persona-LoRA from the #207 family. Reuse existing LoRAs where available (audit first — #207 consolidation enumerates the per-experiment LoRAs in its appendix); train fresh r=32 α=64 LoRAs on self-distilled persona-conditioned text only if existing artifacts don't cover the JS-divergence panel positions.
2. Generate 100 reference completions from `M_prompted_P` (Qwen-Instruct + persona-`P` system prompt) on a fixed 100-prompt eval set (from #207's eval prompts; disjoint from training data).
3. Score those reference completions under two models:
   - (a) `M_LoRA_P` = Qwen-Instruct + persona-`P` LoRA, no system prompt
   - (b) `M_LoRA_Q` = Qwen-Instruct + control-persona-`Q` LoRA, no system prompt
4. Compute per-token log-prob; paired t-test for `(a) - (b)`.
5. Symmetric direction: generate 100 completions from `M_LoRA_P` (no prompt); score under `M_prompted_P` and `M_prompted_Q`; same statistics.

### Cross-phase dissociation analysis

For each panel position, compute the log-prob gap `(M_match - M_control)` separately for Phase 1 and Phase 2. Plot Phase 1 gap vs Phase 2 gap. Positive correlation = the two drift mechanisms land in similar places (Dan's full claim); zero correlation = they're independent; negative correlation = anti-correlated displacements (interesting).

## Primary metrics

- **Per-token mean log-prob gap** `log p(t | M_match) - log p(t | M_control)`, averaged across tokens within a completion, paired across completions, per panel position.
- **Sequence-level mean log-prob gap** (sanity check; usually less diagnostic than per-token but easier to communicate).
- **Per-token top-1 agreement rate** between `M_match` and `M_prompted_P`'s most-likely-token predictions on the same context.
- **Phase 1 vs Phase 2 dissociation correlation** (Spearman ρ across the 5 panel positions).

## Infrastructure reuse from #377

The user specifically requested infra reuse. The pieces:

- **Drift conversation corpus** — Phase 1's "drifted tokens" input. Available after #377 lands in `awaiting_promotion`. Format: per-domain JSONL with conversation turns, persona labels, drift-script category. The marker-eval pipeline from #377 already loads this; this task's analysis script can subclass that loader.
- **Persona panel definitions** — #377's panel may already cover several of the 5 panel positions; reuse the persona system prompts and trigger-key strings verbatim. Audit at task-launch time which positions need fresh persona definitions.
- **Eval-prompt set** — #377 uses an eval prompt set for marker fire-rate scoring. This task reuses the same set for Phase 1's reference completions and Phase 2's eval prompts to keep the test surface comparable across phases.
- **Auditor pipeline** — #377's Claude-Sonnet/GPT-5 audit pipeline is not needed for log-prob scoring (log-prob doesn't need a behavioral judge). Skip.
- **vLLM batched inference rig** — `src/explore_persona_space/eval/` modules used by #377. The log-prob computation needs teacher-forced scoring (not generation), so add a `compute_logprobs.py` script (~100–200 lines) using vLLM's `LLM.compute_logprobs` (if available) or HuggingFace `model.forward()` with KV cache. Pilot on one persona before full panel.
- **Persona-LoRA loading** — Phase 2 reuses #207-family LoRA loading code; if #207's LoRA-load helper isn't a standalone module, extract one.

## Pre-conditions

1. **#377 must land in `awaiting_promotion` with the drift-conversation corpus uploaded to the HF data repo.** This task does not launch until #377 has uploaded its `drift_conversations.jsonl` (or equivalent) artifact — verify via `hf api list-repo-files superkaiba1/explore-persona-space-data --revision main` before launch. If #377 ships marker results but never publishes the drift corpus separately, this task spawns an upload-fix subtask first.
2. **#207-family persona LoRAs are accessible** on the HF model repo at the recipe-canonical Qwen2.5-7B-Instruct base. Audit at launch time; train missing positions if necessary (adds 1–2 days to the experiment, gated on whether #382's reusable-model-organism work has shipped a more durable substrate).
3. **#352 (Lu et al. critical read)** does not block this task, but its conclusions may revise persona-panel selection if the assistant-axis methodology has issues we haven't yet identified. Run in parallel; revise on its outcome.
4. **Compute budget:** ~1 H100-day for log-prob scoring across 5 personas × 2 phases × 200 completions × ~256 tokens. No training in the default path; only if existing LoRAs don't cover the panel does Phase 2 add LoRA training (~4 H100-hours per missing position).

## Risks

- **Drift-corpus stratification mismatch.** #377's drift scripts are categorized by drift-direction (hostile-user, role-play, jailbreak), not by *which* persona the model lands in. The Phase 1 mapping from drift category to matching persona `P` may be lossy. Mitigation: filter to drift conversations where Claude-judge confirms the drifted output reads as the target persona (use #377's auditor pipeline outputs as the filter); discard ambiguous conversations.
- **Self-distilled-LoRA budget arbitrariness.** Phase 2's "training-time drift" depends on a specific LoRA training budget. Too little = LoRA doesn't reach the prompted distribution (Phase 2 trivially negative). Too much = overfitting collapses the distribution to a narrow mode (Phase 2 may be positive for trivial reasons). Mitigation: when training fresh LoRAs, use 200 / 600 / 1800 step checkpoints and report log-prob gap as a function of training budget. If existing #207-family LoRAs are used, document their original training budgets and treat that as a known confound.
- **Per-token log-prob averaging masks heterogeneity.** Two distributions can have similar mean log-prob while diverging on specific tokens. Mitigation: also report per-token KL distribution (histogram, percentiles), not just the mean.
- **Control persona `Q` may be too close to `P` in the assistant-axis coordinate.** If `Q` happens to be near `P`, the null-distinguisher gap is small even when equivalence is real. Mitigation: pick `Q` from a different role family (e.g., if `P` is "librarian", `Q` is "race-car driver" not "archivist"); pre-validate at launch that `Q`'s JS-divergence-from-`P` is in the upper quartile of pairwise distances on the panel.

## Dependencies

- **Parent #377** (running) — provides the drift corpus for Phase 1. Wait for `awaiting_promotion` before launch.
- **Related #207** (cr) — provides persona-LoRAs for Phase 2 and the JS-divergence-from-assistant computation for the panel selection.
- **Methodological #352** (proposed) — Lu et al. critical read; run in parallel, revise persona panel on its findings.
- **Related #340 / #368** (both cr) — established the length confound that motivates JS-divergence over cosine.

Mentor-meeting origin: `docs/mentor_updates/2026-05-22.md` Thread D, second half.
