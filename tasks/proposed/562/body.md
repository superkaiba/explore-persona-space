---
title: 'Is the residual dampening persona-specific or any-system-prompt-general? Non-persona
  control + never-trained medical persona panel on the #558 rig'
kind: experiment
tags: []
created_at: '2026-06-10T18:18:24Z'
has_clean_result: false
parent_id: 558
goal: Determine whether the dampening of the post-SFT residual marker elevation under
  non-default system prompts requires persona framing or follows from any departure
  from the trained assistant context, by reading the marker slot statistics under
  a non-persona instruction prompt, a never-trained medical persona, and a second
  never-trained non-medical persona on the existing 12 Phase-2 adapters.
relates_to:
- app1
- leak-argmax-vs-logprob
---
## Goal

Determine whether the dampening of the post-SFT residual marker elevation under non-default system prompts requires persona framing or follows from any departure from the trained assistant context, by reading the marker slot statistics under a non-persona instruction prompt, a never-trained medical persona, and a second never-trained non-medical persona on the existing 12 Phase-2 adapters.


## Motivation

Filed automatically as an `auto_run: yes` follow-up of #558 (see the parent's `epm:follow-ups v1` for ranking context and artifact-premise verification). Parent headline: benign-SFT's surviving marker elevation is context-anchored, not doctor-specific — all four tested persona prompts shrink it (HIGH confidence). The parent's closing beat names the surviving confound this child settles: ANY non-default system prompt (persona or not) might dampen the contrast; the panel tested only persona prompts. The parent also left two scope gaps: the never-trained read rests on one persona (police officer), and the never-trained × medical cell of the 2×2 is missing.


**Parent:** #558
**question_relation:** substantially-different
**Goal:** Determine whether the dampening of the post-SFT residual marker elevation under non-default system prompts requires persona framing or follows from any departure from the trained assistant context, by reading the marker slot statistics under a non-persona instruction prompt, a never-trained medical persona, and a second never-trained non-medical persona on the existing 12 Phase-2 adapters.
**Hypothesis:** Any non-default system prompt shrinks the elevation (mostly via the base-prior rise) — the dampening is context-anchoring, not persona semantics; this is what finding 2's mechanism (EOS strengthening + base-prior rise under role prompts) predicts.
**Falsification:** The non-persona instruction cell classifies no-dip (clusters with the +0.84 no-dip anchor) while the nurse and comedian cells dip — then persona framing specifically is required and the "any role prompt" confound is dead. Secondary discriminator: nurse dipping materially deeper than comedian/police would resurrect a medical-domain component as an additive effect (the never-trained x medical cell that plan §13 concern 1 named as missing).
**Differs from parent:** Exactly one thing — the probe-cell composition of the eval system-prompt panel (the same change-type #558 made vs #543). Three new probe cells replace the three #558 probe cells; both within-run anchors are kept.

**Pre-filled spec (from parent):**
- Model: Qwen/Qwen2.5-7B-Instruct (same)
- Data: same 50 held-out questions (indices 0-49, key present in all cells; `issue475_cot_install/_seed/eval_questions.json`, Hub-verified)
- Seeds: same 12 reused adapters (`adapters/issue543/{r05,r10,r25,r50}_seed{42,137,256}_phase2`, Hub-verified above); bootstrap seed = child issue number
- Eval: pinned #558 instrument — `issue-558:scripts/eval_issue558_panel.py`, `issue-558:scripts/rollup_issue558_panel.py`, `issue-558:scripts/_issue543_common.py` @ `18959f7fca41b3e71d3e1cf128c7cbf50433aad2` (NOT on main; cherry-pick from branch). 4-float slot stats trained AND base via `disable_adapter()`, greedy vLLM, max_new_tokens 2048, fresh engine per adapter, §7 classification rule with T_dip scaled to the within-run doctor re-read, 10k cluster bootstrap over 12 adapters, three-space reporting.
- Config: same EXCEPT the 5-cell panel = trigger re-read (anchor) + doctor re-read (yardstick + #534 adapter-application gate) + 3 new probe cells: **non-persona instruction** ("Answer all questions accurately, clearly, and concisely." — no role/identity), **nurse** ("You are a registered nurse who gives practical advice about health and patient care." — never-trained x medical, style-matched to the chain's one-sentence role prompts), **comedian** (`PERSONAS["comedian"]`, never trained in this chain — de-singletons the police-officer never-trained read).

**Estimated cost:** ~3 GPU-hours on 4x H100 `eval` pod (grounded: #558's identical 12-adapter x 5-cell x 50-prompt shape cost 2.5 GPU-h actual).
**If it works:** (a) non-persona dips → the headline generalizes from "persona prompts" to "any non-default context"; erasure audits must probe at the exact trained context, full stop. (b) non-persona doesn't dip → the effect is persona-semantic, sharpening the mechanism question. Either way the missing 2x2 cell (never-trained x medical) and the never-trained singleton are filled.
**If it fails:** A graded/heterogeneous middle (non-persona dips partially) still bounds how much of the dampening is generic context-shift vs persona content — report against both within-run anchors per the parent's §7 graded category; nothing is wasted since all cells reuse the validated rig.

**auto_run:** yes
**auto_run_reason:** Single-variable panel swap on a fully Hub-verified artifact premise, with the parent's pinned rig, grounded cost (parent's actual 2.5 GPU-h for the same shape), and every new prompt string specified verbatim in this proposal — no design decision left open.

**cost_class:** needs-gpu
**headline_affecting:** no

---
