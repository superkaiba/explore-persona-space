---
title: 'Falsify #398''s global-marker-affinity-shift hypothesis: re-run with comedian
  (not librarian) as source persona'
kind: experiment
tags: []
created_at: '2026-05-28T19:09:51Z'
has_clean_result: false
parent_id: 398
goal: 'Train ※ into a non-librarian source persona (recommended: comedian) and check
  whether (a) the new source ALSO never leads the leaderboard and (b) the same top-6
  bystander cluster from #398 still dominates — confirming the global-marker-affinity-shift
  mechanism vs ruling it out in favor of a source-specific direction.'
---
## Depends on

- [#398](https://eps.superkaiba.com/tasks/398) — established the source-is-low pattern: librarian source persona ranks 24/28 at step 5 and never enters top 6 across any of 22 checkpoints. Same 6 bystanders (comedian, fammate_task_2, fammate_context_2, french_person, poet, villain) dominate the leaderboard at almost every step. Spearman ρ(step 5, step 1600) on pos0 = 0.690 (p=4.8e-5). The simplest hypothesis is a **global marker-token affinity shift over persona-conditional baselines** — the implant uniformly elevates log p(※) for everyone, leaving the rank order largely intact.

## Goal

Train ※ into a non-librarian source persona (recommended: comedian) and check whether (a) the new source ALSO never leads the leaderboard and (b) the same top-6 bystander cluster from #398 still dominates — confirming the global-marker-affinity-shift mechanism vs ruling it out in favor of a source-specific direction.

## Background

#398 trained `※` into librarian and discovered the source persona is NEAR THE BOTTOM of the marker-probability leaderboard at every checkpoint. The implant didn't elevate librarian to the top; it shifted everyone roughly uniformly. The top-6 cluster (comedian, fammate_task_2, fammate_context_2, french_person, poet, villain) is the same at almost every step and is consistent with "personas whose base-rate prior favors literary / creative / Unicode-rich completions" — which the librarian system prompt does NOT.

This is consistent with TWO mechanisms #398 couldn't distinguish at n=1 source:

1. **Global affinity shift** — training raises every persona's log p(※) by a roughly constant offset, preserving the base-rate ordering. The trained source doesn't lead because its base-rate prior on `※-after-\n\n` is below several bystanders.
2. **Bystander-cluster-attractor** — the optimizer finds the same "creative-writing-style" attractor regardless of source, because that's where ※-after-`\n\n` is structurally easiest to fit. The trained source ends up wherever its base-rate puts it; the attractor cluster always dominates.

Both predict "trained source doesn't lead." They differ on what happens if you change the source:
- (1) predicts: comedian source → comedian doesn't necessarily lead either; same top-6 cluster wins; ρ(pre, post) on the bystander panel ≈ 1.0.
- (2) predicts: comedian source → comedian leads (or close); the attractor cluster includes the trained source; the cluster identity shifts toward "comedian-like" personas.

## What this tests

- Whether the rank-preservation finding (Spearman ρ=0.69 across 1600 training steps) is a property of the librarian-source recipe OR a property of the global-shift dynamic regardless of source.
- Whether the same 6 personas (comedian, fammate_task_2, fammate_context_2, french_person, poet, villain) dominate the leaderboard for ANY source persona.
- Whether the trained source's rank improvement is small (consistent with mechanism 1) or large (consistent with mechanism 2).

## What this does NOT test

- Activation-side or internal-direction localization (would need probe + intervention experiments).
- Whether the global-shift mechanism is universal across markers (※ → other rare Unicode tokens — separate task).
- Multi-source training (training on librarian AND comedian simultaneously — separate task).

## Plan sketch (to be sharpened by `/adversarial-planner`)

1. **Pick the second source persona.** Recommended: **comedian** (always #1 emitter in #398's data; cleanest contrast). Alternative: **poet** (also reliably top-3 in #398). Reuse #398's spec exactly otherwise: Qwen-2.5-7B-Instruct, LoRA r=32 α=64, lr=1e-5, 1600 steps, seed=42, marker-only loss mask, 22 checkpoints matching #398's schedule. ~2 GPU-h training.
2. **Eval on the IDENTICAL 27-bystander panel from #398** (which includes librarian as a bystander now, and excludes the new source persona). Same 20 prompts. Same dual-probe (pos0 + endpos) + same per-position on-policy log-p instrument with the bumped fixes (batched generation, max_new_tokens=2048). ~30 min eval per probe (faster with the implementation now cherry-picked from #398).
3. **The discriminating analysis:**
   - Where does the new source (e.g., comedian) rank at every step? Top-N or not?
   - Is the top-6 cluster from #398 still dominant, or does it shift toward "comedian-like" personas?
   - Spearman ρ(pre, post) on the 27-bystander panel — does it still hit ~0.69?
   - Cross-experiment: panel-wide mean log p(※) per step — does it look like a constant additive shift across both experiments, or does the SHAPE of the shift depend on source?
4. **Hero figure:** per-step rank trajectory of comedian source + librarian (which is now in the bystander panel) + the top-6 cluster from #398, all on the same plot. Reveals at a glance whether the new source leads or stays mid-pack.

## Open questions for the planner

- Should we run BOTH a second source (comedian) AND a third source (poet) for a triangulation (~4 GPU-h training instead of 2)? Or single new source first, decide on third based on data?
- Should we include the on-policy per-position probe by default, or only the dual probe? The per-position adds ~30 min eval + 4 figures but is the most decisive instrument for "where is the marker mass concentrated."
- Is there a value in also running a NEGATIVE-control source — a persona that's NOT in #398's top-6 cluster but ALSO not librarian — to triangulate the cluster vs source-specific question?
- Compute budget — comfortable target: ~3.5 h wall (single new source, full instrumentation) or ~6 h wall (two new sources, full instrumentation)?

## Cross-references

- Parent: #398 (per-step marker-emergence with ※; established source-is-low + global-shift hypothesis).
- Spiritual ancestor: #385 (the [ZLT] marker version that triggered all this).
- Sibling: #397 (loss-mask factor at higher cell resolution — uses ※ too; not directly testing the global-shift question but the same instrumentation chain).
