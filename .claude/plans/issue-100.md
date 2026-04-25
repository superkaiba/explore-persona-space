# Issue #100 — Assistant Persona Robustness: Dose-Response + Source-of-Robustness Ablation

## Why this experiment / why these parameters / alternatives considered

Issue #96 found the assistant persona uniquely resists ARC-C degradation under contrastive
wrong-answer SFT (-2pp vs -80pp for villain, comedian, software_engineer, kindergarten_teacher).
This is a striking outlier that demands follow-up: is the assistant genuinely more entrenched,
or does the contrastive training design interact differently with the default persona?

During adversarial planning, the critic identified a **critical confound**: the #96 training
data includes 100 "default assistant + correct answer" anchor negatives for ALL sources.
When source=assistant, this creates a 2:1 wrong:correct signal conflict for the same prompt
that no other source experiences. This could explain the -2pp result via signal cancellation.

We therefore gate the entire experiment on a contamination control (Exp 0) before investing
in the dose-response sweep (Exp A) or source-of-robustness ablation (Exp C).

Exp B (perturbation-type sweep across 4 perturbation types) was dropped during clarification —
data generation and eval metrics for new perturbation types were unspecified and would add
significant scope without proportional information gain.

---

## Experiment Structure

### Exp 0 — Contamination Control (GATING, ~20 min)

**Goal:** Test whether the #96 assistant robustness finding is explained by the 100
"assistant + correct" anchor negatives in the training data.

**Two conditions:**
- **0a (deconfounded):** Remove the 100 "default assistant + correct" anchor examples.
  Total: 700 (200 pos wrong + 400 neg bystander + 100 no-persona).
- **0b (size-matched control):** Remove 100 random NON-assistant negatives (e.g., 50 from
  each bystander category). Total: 700. Same size as 0a but keeps the confound.

Both use: lr=1e-5, epochs=3, LoRA r=32, seed=42. Evaluate assistant ARC-C on 586 held-out.

**Decision gate:**
- 0a drops below 50% AND 0b stays above 70% → **STOP Exp A**. The confound explains #96.
  Report as primary finding. Exp C may still run with deconfounded data.
- 0a stays above 70% → Confound is NOT the explanation. Proceed to Exp A + C with
  original 800-example data.
- 0a is 50-70% → Confound partially explains robustness. Proceed to Exp A with BOTH
  data variants (700 deconfounded + 800 original).

### Exp A — Dose-Response Curve (~2.5 GPU-hrs, conditional on Exp 0)

**Goal:** Find the training threshold where assistant ARC-C degrades to villain-level (~3%).

3 independent 1D sweeps for the assistant persona:

| Sweep | Variable | Values | Holds constant |
|-------|----------|--------|----------------|
| LR | learning_rate | [1e-5, 3e-5, 1e-4, 3e-4] | epochs=3, data=800 |
| Epoch | num_epochs | [3, 6, 12, 24] | lr=1e-5, data=800 |
| Data size | n_examples | [200, 400, 800, 1600] | lr=1e-5, epochs=3 |

10 unique runs (baseline lr=1e-5/ep=3/n=800 is shared across sweeps).

**Villain references:** After identifying the 1-2 points where assistant first shows >10pp
degradation, run villain at those same settings. This calibrates whether the degradation
is assistant-specific or universal.

**Seed validation:** After identifying the threshold point, run 2 extra seeds (137, 256)
at that point + one neighbor to validate stability. Adds ~4 runs.

**Data construction:** The data-size sweep scales proportionally:

| Total | Positives (wrong) | Neg (bystander) | Neg (no-persona) | Neg (assistant) |
|-------|-------------------|-----------------|------------------|-----------------|
| 200 | 50 | 100 | 25 | 25 |
| 400 | 100 | 200 | 50 | 50 |
| 800 | 200 | 400 | 100 | 100 |
| 1600 | 400 | 800 | 200 | 200 |

Max positives = 400 < 586 available questions. No repetition needed.

**Hypotheses:**
- H_A1: At lr=3e-4, assistant ARC-C drops below 50%
- H_A2: At 24 epochs, assistant ARC-C drops below 30%
- H_A3: Data size has modest effect (curve is flatter than LR/epoch curves)
- Falsification: If assistant stays above 70% at ALL points, robustness is categorical

### Exp C — Source-of-Robustness Ablation (~1.5 GPU-hrs)

**Goal:** Identify what makes the assistant robust — the specific prompt text, the system
turn structure, or RLHF entrenchment.

**Critical fact:** Qwen2.5-7B-Instruct inserts default system prompt
"You are Qwen, created by Alibaba Cloud. You are a helpful assistant." when no system
message is provided. The `_arc_logprob_core()` function treats `persona_prompt=""` the same
as `None` (both falsy). Must fix eval to distinguish these.

6 conditions (all lr=1e-5, epochs=3, data=800, seed=42):

| Condition | System prompt in training+eval | Tests |
|-----------|-------------------------------|-------|
| full_prompt | "You are a helpful assistant." | #96 reproduction (positive control) |
| empty_system | "" (empty, present in template) | System turn STRUCTURE |
| qwen_default | None → model inserts built-in default | Model's built-in default persona |
| name_only | "You are an assistant." | "assistant" keyword without "helpful" |
| nonce_role | "You are ROLE_A." | Any system prompt vs assistant-like |
| curious_explorer | "You are a curious explorer." | Same structure, no "assistant" keyword |

5 new runs (full_prompt = Exp A baseline).

**Eval fix required:** Change `_arc_logprob_core()` line 139 from `if persona_prompt:` to
`if persona_prompt is not None:`. With this fix:
- `persona_prompt=""` → includes empty system turn in chat template
- `persona_prompt=None` → omits system message, Qwen inserts its default

**Hypotheses:**
- H_C1: full_prompt reproduces #96 (~82%) — positive control
- H_C2: qwen_default resists (~80%+) — built-in default contains "assistant"
- H_C3: name_only partially resists (~60-80%) — "assistant" keyword activates RLHF
- H_C4: empty_system degrades (~20-50%) — no semantic content
- H_C5: nonce_role degrades (~20-50%) — no RLHF association
- H_C6: curious_explorer degrades (~20-50%) — tests keyword isolation
- Key prediction: If qwen_default resists AND nonce_role doesn't → semantic keyword matters

---

## Prerequisites (Phase 0, ~1.5h)

| Step | Task | Time |
|------|------|------|
| 0a | Port issue-69 scripts to main (or copy directly). Check merge conflicts. Fallback: scp scripts from worktree. | 30 min |
| 0b | Write `scripts/build_dose_response_data.py` — data size variants [200, 400, 1600] | 15 min |
| 0c | Write `scripts/build_robustness_ablation_data.py` — 6 Exp C JSONL files | 20 min |
| 0d | Write deconfounded data for Exp 0 (variant of build script, remove assistant anchors) | 10 min |
| 0e | Fix `_arc_logprob_core()`: `if persona_prompt is not None:` + unit test verifying different template output for "" vs None | 15 min |
| 0f | Generate all data files on pod, push to pod | 5 min |
| 0g | Verify: data file counts, eval split = 586 questions matching #96 seed=42 split, model cached, pod compatibility (H200 vs H100 from #96) | 5 min |

---

## Execution Pipeline

```
Phase 0: Prerequisites (1.5h, local VM + pod)
Phase 1: Exp 0 — Contamination control (20 min, 1 GPU)
  → Run 0a (deconfounded) + 0b (size-matched control)
  → Decision gate: proceed / stop / expand
Phase 2: Exp A — Dose-response (2-2.5h, GPU 0)
  → Run 10 sweep points sequentially
  → Decision gate G1: baseline reproduces #96 (within 5pp of 81.9%)
  → Decision gate G2: monitor lr=3e-4 for divergence
  → After threshold found: villain reference + seed validation (~30 min)
Phase 3: Exp C — Ablation (1h, GPU 1, can overlap with Exp A late stages)
  → Run 5 new conditions
Phase 4: Analysis (30 min)
  → Dose-response curve plots (3 subplots: LR, epoch, data size)
  → Ablation bar chart
  → Upload to WandB, save to eval_results/
```

---

## Reproducibility Card

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | Base | Qwen/Qwen2.5-7B-Instruct (7.62B) |
| | Trainable | LoRA adapter (~25M params) |
| **LoRA** | r | 32 |
| | alpha | 64 |
| | dropout | 0.05 |
| | targets | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| | rslora | True |
| **Training** | Method | SFT, completion-only loss |
| | LR | Exp 0/C: 1e-5; Exp A: [1e-5, 3e-5, 1e-4, 3e-4] |
| | Schedule | cosine, warmup_ratio=0.05 |
| | Epochs | Exp 0/C: 3; Exp A: [3, 6, 12, 24] |
| | Optimizer | AdamW (beta=(0.9,0.999), eps=1e-8) |
| | Weight decay | 0.0 |
| | Gradient clipping | 1.0 |
| | Precision | bf16, gradient_checkpointing=True |
| | Batch (effective) | 16 (per_device=4 × grad_accum=4 × 1 GPU) |
| | Packing | False |
| | Seeds | Primary: 42; Validation: 137, 256 |
| **Data** | Source | ARC-Challenge test set, 50/50 split (seed=42) |
| | Train questions | 586 |
| | Eval questions | 586 (disjoint, same split as #96) |
| | Format | prompt-completion JSONL (system+user → assistant) |
| **Eval** | Metric | ARC-C logprob accuracy |
| | Method | `_arc_logprob_core()` with `persona_prompt=<prompt>` |
| | Eval set | 586 held-out ARC-C questions |
| **Infra** | Hardware | 1× H200 SXM (pod1) |
| | Scripts | `run_capability_leakage_sweep.py` (from issue-69), new data builders |
| | WandB project | `explore-persona-space` |

---

## Success / Kill Criteria

**Success:**
- Exp 0 resolves the contamination question (either direction is a finding)
- Exp A: dose-response curve with ≥8 data points
- Exp C: at least one ablation that breaks robustness (< 40% ARC-C), OR all-equal (implicating RLHF)

**Kill:**
- Baseline doesn't reproduce #96 (assistant ARC-C < 70%): STOP, debug
- lr=3e-4 diverges: drop that point, proceed with lr≤1e-4
- Exp 0 shows confound explains everything: skip Exp A, report confound

---

## Plan Deviations

- **Allowed without asking:** dropping divergent runs, adding 1-2 extra seeds at threshold
- **Must ask:** dataset changes, adding new conditions, changing eval metrics, changing pod
- **NOT allowed:** changing seed=42 for primary runs, skipping Exp 0 decision gate

---

## Compute Estimate

| Phase | Runs | Wall time |
|-------|------|-----------|
| Exp 0 (contamination control) | 2 | ~20 min |
| Exp A (dose-response) | 10 unique | ~2h |
| Exp A (villain refs + seed validation) | ~4-6 | ~30-45 min |
| Exp C (ablation) | 5 new | ~1h |
| Analysis | — | ~30 min |
| **Total** | ~23 | **~4.5h wall, ~4 GPU-hrs** |

Target pod: pod1 (4× H200). Uses 1-2 GPUs.
