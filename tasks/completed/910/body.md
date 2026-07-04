---
title: 'workflow-fix: plan-time gates measure per-item serialization/IO wall-time;
  broaden ''left serial'' beyond fits'
kind: infra
tags:
- wf-fix
created_at: '2026-07-03T06:42:01Z'
has_clean_result: false
origin_prompt: 'user 2026-07-03: ''can you add to planning/implementation critic also?''
  — the #813 triple overhead incident (savez zlib store, sequential waves, serial
  null battery)'
---
## Overview / Motivation

Auto-filed (user-directed 2026-07-03: "can you add to planning/implementation critic also?") from the #813 triple-incident thread. Three same-root wall-clock blowups on ONE experiment — (1) `np.savez_compressed` serialization at ~100s/file for a 1.29x ratio (extract wave 4.5x over plan), (2) sequential wave dispatch idling half the fleet, (3) a serial 2M-tiny-fit substrate-swap null projecting 10-12h — all shared the root cause "per-item overhead, not FLOPs." The LESSONS.md trigger for vectorize-many-cell-fits was already broadened to null-draw batteries, and code-reviewer.md already carries the implementation-side checks (incl. the #813 savez citation). The remaining gap is PLAN-TIME: planner §9's sizing block + critic Methodology lens still bind the serial-loop check to FIT/KERNEL-shaped work and never name per-item SERIALIZATION/IO cost.

## Goal

Extend the plan-time compute-sizing enforcement so (a) store-heavy phases must benchmark one production-size item's SERIALIZATION+UPLOAD wall-time (not just bytes) at gate time, with client-side compression of fp16 activation stores destined for Xet-backed HF repos defaulting OFF; and (b) the critic's "left serial" check broadens from gradient-descent/dense-factorization fits to ANY high-count tiny-op battery (resample/bootstrap/null draws, per-item serialization, per-file uploads).

## Workflow gap

- **Bug observed:** #813's one-cell GO/NO-GO gate measured per-file BYTES and passed, while per-file serialization CPU (~100s of single-thread zlib) made the store phase the wall-clock driver (4.5x over plan, ~$300+ idle 8xH100); separately its plan's analysis phase shipped a serial null battery (~2M tiny closed-form fits, 10-12h) that no plan-review lens flagged because the serial-loop checks name only GD/dense-factorization fits.
- **Why it is a workflow gap:** planner.md §9's "serial-fit-loop / draw-battery sizing block" + plan-compute-sizing.md's floor cross-check + critic.md Methodology item 10's "left serial" all scope to fit/kernel calls; serialization/IO per-item cost is unnamed on every plan-time surface, and the broadened LESSONS trigger has no enforcing lens behind it.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

- `.claude/rules/plan-compute-sizing.md` (the §9 pointer rule): add a "store-heavy / IO-heavy phase" recipe block:
  + a phase writing >~10^3 files or >~50 GB MUST report a MEASURED one-item production-shape serialization+upload wall-time in the §9 table (not bytes alone);
  + client-side compression (savez_compressed/gzip) of fp16 activation tensors bound for Xet-backed HF repos defaults OFF (Xet chunk-compresses/dedupes; #813 measured 103.8s vs 1.2s for a 1.29x ratio); an ON choice needs a measured ratio justification;
  + the floor cross-check's ">~500 serial calls of a non-trivial kernel" clause gains "(a fit, dense factorization, SERIALIZATION of a multi-hundred-MB artifact, or a per-file Hub commit)".
- `.claude/agents/planner.md` §9 summary line: extend "serial-fit-loop / draw-battery sizing block" to "serial-fit-loop / draw-battery / store-serialization sizing block".
- `.claude/agents/critic.md` Methodology lens item 10: broaden "(iii) gradient-descent / dense-factorization fit on the VM CPU or left serial" to "(iii) ... or ANY >~10^4-item tiny-op battery (resample/bootstrap/null draws, per-item serialization, per-file uploads) left serial/unbatched"; item 13 gains "store-heavy phases: per-item serialization wall-time measured, compression-default-OFF for fp16->Xet".
- Cross-check `.claude/rules/vectorize-many-cell-fits.md` names the serialization variant once (one sentence) so the rule and its enforcing lenses stay consistent.

## Scope / surfaces

- Primary targets: `.claude/rules/plan-compute-sizing.md`, `.claude/agents/planner.md`, `.claude/agents/critic.md`, `.claude/rules/vectorize-many-cell-fits.md`
- Grep first: `grep -rn "draw-battery\|left serial\|serial-fit-loop\|savez" .claude/agents/ .claude/rules/` and dedup against what #813's owning session already landed (code-reviewer.md:876 already cites #813 savez; LESSONS.md trigger already broadened — do NOT re-edit those).

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` passes; keep CLAUDE.md's always-on vectorize bullet consistent (it already says "vectorize first" — no change needed there).
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/plan-compute-sizing.md, .claude/agents/planner.md, .claude/agents/critic.md, .claude/rules/vectorize-many-cell-fits.md
- fingerprint: (computed by wrapper)

User-directed 2026-07-03 ("can you add to planning/implementation critic also?"). Related: #813 (the triple incident), #880 (unrelated CVD gotcha), #864 (poller false-stall), #722/#778/#779 (prior vectorization line).
