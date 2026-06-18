---
title: 'Default-assistant shielding: dose or identity? Single-slot ablation of the
  qwen_default negative on the #600 chassis'
kind: experiment
tags: []
created_at: '2026-06-11T20:17:15Z'
has_clean_result: false
parent_id: 600
goal: 'Determine whether the default assistant context''s anomalously low marker leakage
  under contrastive training (−0.197 vs panel median in #600, the only role far below
  median) is caused by its 200 negative training rows (dose) or by its identity/activation-cluster
  position, by training otherwise-identical mixes that include vs exclude the default-assistant
  negative slot and reading the untrained default context''s implant-normalized marker
  log-prob shift.'
relates_to:
- leak-to-default
- leak-contrastive-negatives
---
# Default-assistant shielding: dose or identity? — Ablation

**Parent:** #600
**question_relation:** substantially-different
**Goal:** Determine whether the default assistant context's anomalously low marker leakage under contrastive training (−0.197 vs panel median in #600, the only role far below median) is caused by its 200 negative training rows (dose) or by its identity/activation-cluster position, by training otherwise-identical mixes that include vs exclude the default-assistant negative slot and reading the untrained default context's implant-normalized marker log-prob shift.
**Hypothesis:** Identity, not dose: the parent's within-persona control showed training leaves no within-persona trace (~+0.004 across 4 dual-role personas), and the assistant-cluster persona slot sits within 0.004 of the default (−0.193 vs −0.197) — so the default context will sit equally far below the panel median even when it is never trained.
**Falsification:** If the never-trained default context's centered shift rises to within ~1 noise-median (0.033) of the untrained-panel median, the shielding was dose — the default's 200 rows were doing real work, which would make the default-context negative (the safety target, open-q 3.7) the one panel slot with a demonstrable per-context effect, against the parent's no-trace pattern.
**Differs from parent:** Exactly one panel slot's membership — the fixed `qwen_default` negative is removed and its 200 rows go to a pre-registered replacement (`journalist`, the chassis pair's already-characterized matched-control persona; on-policy rows from the verified `R_train.json`), holding total budget at 1,000 rows. Everything else (chassis mix `c600_mercenary_near`, villain rows, recipe, 63 steps, eval) is identical; the with-default arm is the parent's existing 3 seeds of the same mix, re-used as-is.

**Pre-filled spec (from parent):**
- Model: Qwen/Qwen2.5-7B-Instruct, rsLoRA r=16/α=32 attention-only, lr 5e-6 cosine, marker-only collator `tail_tokens=0`, band-stop log-only — Source: #600
- Data: new mix = `c600_mercenary_near` with qwen_default→journalist slot swap, built from the verified HF inputs snapshot (`persona_bank.json`, `R_train.json`); with-default arm = parent's committed `c600_mercenary_near` seed 42/137/219 trajectories (verified in git)
- Seeds: 42, 137, 219 (3 new runs)
- Eval: identical — vLLM greedy, ~50 personas × 10 probes, 6 checkpoints, four-float slot reads trained+base; same floor/sub-ceiling gates at terminal checkpoint
- Config: same as parent EXCEPT: the variable described above; disjointness assert extended (journalist ∉ sources/targets — holds for this chassis)

**Estimated cost:** ~3 GPU-h on 1× H100 (lora-7b intent; 3 cells at the parent's realized ~0.8 GPU-h/cell)
**Power note:** if dose explains the dip, the expected movement (~0.17–0.20) is 5–9× the parent-measured same-mix seed noise (median 0.033), so 3v3 seeds resolves it cleanly. Log-prob DV inside the parent's [8.0–10.6]-nat band regime — the valid pairing per `.claude/rules/marker-training-recipe.md` § Usable window; no emission read is gated on.
**Scope caveat to carry:** the no-default arm deliberately violates the always-include-default rule of `.claude/rules/contrastive-negatives.md` — the manipulated variable IS the default's panel membership, which is the named exemption shape (the deliberate control); flag in the child's clean-result.
**If it works (identity):** The default context's protection is positional, not bought by training rows — negative rows on the default context are doing nothing detectable in log-prob space, sharpening the parent's "no within-persona trace" finding into the safety-relevant default case and redirecting open-q 3.7 toward cluster-position mechanisms.
**If it fails (dose):** First demonstrable per-context effect of a single negative slot in this line — exactly the local-shielding phenomenon the parent ruled out for persona bystanders, localized to the default context; immediately motivates a dose-response (0/50/200 default rows) child.

**auto_run:** yes
**auto_run_reason:** clean single-slot ablation with the parent's grounded recipe verbatim, cost known (~3 GPU-h, far under cap), deterministic pre-registered replacement (no taste decision), complete Goal, and every premise artifact positively verified this proposal (HF inputs + training-mix JSONLs via `list_repo_files`; with-default arm trajectories in git) — files as a `proposed` child for triage per the substantially-different routing.

**cost_class:** needs-gpu
**headline_affecting:** no


## Goal

Determine whether the default assistant context's anomalously low marker leakage under contrastive training (−0.197 vs panel median in #600, the only role far below median) is caused by its 200 negative training rows (dose) or by its identity/activation-cluster position, by training otherwise-identical mixes that include vs exclude the default-assistant negative slot and reading the untrained default context's implant-normalized marker log-prob shift.
