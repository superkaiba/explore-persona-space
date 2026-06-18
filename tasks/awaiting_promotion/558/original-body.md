---
title: Is the surviving residual doctor-specific or persona-general? Persona-panel
  read on the existing Phase-2 adapters
kind: experiment
tags: []
created_at: '2026-06-10T14:09:36Z'
has_clean_result: false
parent_id: 543
goal: Determine whether the post-SFT residual marker elevation's doctor-cell suppression
  reflects overlap with the medical erasure domain, residue of Phase-1 negative training,
  or persona framing in general, by reading the 4-float marker slot statistics under
  trained-negative non-medical personas and one never-trained persona on the existing
  12 Phase-2 adapters.
relates_to:
- app1
- leak-argmax-vs-logprob
---
## Goal

Determine whether the post-SFT residual marker elevation's doctor-cell suppression reflects overlap with the medical erasure domain, residue of Phase-1 negative training, or persona framing in general, by reading the 4-float marker slot statistics under trained-negative non-medical personas and one never-trained persona on the existing 12 Phase-2 adapters.


## Motivation

Filed automatically as an `auto_run: yes` follow-up of #543 (see the parent's `epm:follow-ups v1` for ranking context). Parent headline: all four positive-ratio arms (50/25/10/5%) collapse identically to 0% post-SFT trigger emission with matched trained strength (HIGH confidence); the surviving log-prob residual is key-blind but persona-sensitive.

### 3. Is the surviving residual doctor-specific or persona-general? Persona-panel read on the existing Phase-2 adapters — Type: Diagnostic

**Parent:** #543
**question_relation:** substantially-different
**Goal:** Determine whether the post-SFT residual marker elevation's doctor-cell suppression reflects overlap with the medical erasure domain, residue of Phase-1 negative training, or persona framing in general, by reading the 4-float marker slot statistics under trained-negative non-medical personas and one never-trained persona on the existing 12 Phase-2 adapters.
**Hypothesis:** The doctor dip (-1.4 nats log-prob, -3.5 nats EOS margin below trigger, all 12 cells) is erasure-domain overlap: doctor-persona contexts resemble the medical SFT distribution where EOS was strengthened, so non-medical personas will cluster with trigger/no-key rather than dip. The panel factorizes the three accounts: doctor = trained-negative x medical (have); software_engineer+key, french_person+key = trained-negative x non-medical (add); police_officer+key = never-trained x non-medical (add). Note the parent's own data already rules out "any trained negative dips": the no-key default assistant was also a Phase-1 negative and reads HIGH (8.70 nats, indistinguishable from trigger).
**Falsification:** If the non-medical personas (trained and untrained alike) dip comparably to doctor (>= ~2 nats EOS-margin below the trigger cell across cells), the domain-overlap account is dead and the residual is persona-general — any persona framing suppresses it.
**Differs from parent:** Eval persona panel only — 3 added probe cells (50 greedy completions each, same held-out questions as the doctor cell) per existing Phase-2 adapter. No training; no new adapters; the 12 `*_phase2` adapters (Hub-verified above) + base model are reused.

**Pre-filled spec (from parent):**
- Model: Qwen/Qwen2.5-7B-Instruct + the 12 `adapters/issue543/*_phase2` adapters (Hub-verified)
- Data: same held-out eval questions (chain's 250-question split, doctor-cell indices)
- Seeds: 42, 137, 256 (same — all 12 cells read)
- Eval: `eval_issue543.py` persona-cell path unchanged; 4-float slot stats trained AND base per cell; persona system prompts for software_engineer / french_person already exist (they generated the Phase-1 negative response bank); police_officer prompt from the chain's persona config (#448 bystander), written in the same format if absent
- Config: no training config — eval-only

**Estimated cost:** ~2.5 GPU-hours on 1x H100 (eval intent; 12 adapters x 3 cells x 50 completions, vLLM gen + HF slot forwards)
**If it works:** (non-medical personas cluster high) — the residual is specifically suppressed where the erasure data lived, i.e. benign SFT's leftover effect is domain-local; sharpens the parent's persona-sensitive H4 into a mechanism claim and suggests erasure audits should probe ON the fine-tuning domain.
**If it fails:** (everything dips) — the residual is persona-general; the parent's "persona-sensitive" framing gets revised to "default-context-specific," and the latent-retention story simplifies (still behaviorally nil either way — trained P(marker) ~ 1e-6).

**auto_run:** yes
**auto_run_reason:** Eval-only diagnostic over Hub-verified existing adapters; zero training; pinned 3-persona panel with prompts already in the repo (police_officer fallback is a trivial format-matched addition); ~2.5 GPU-h; concrete falsification on a flagged interpretation ambiguity (interp-critique round 1, Surprising Patterns 1).

**cost_class:** needs-gpu
**headline_affecting:** no
