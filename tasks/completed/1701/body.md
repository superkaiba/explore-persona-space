---
title: 'daily-fix: inline-round scientific duties'
kind: infra
tags:
- wf-fix
- wf-fix-fp:46ff8f476147
- daily-auto-filed
created_at: '2026-07-26T07:06:42Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): An inline round launched
  two headline runs on a 1877-row subset below the 3584 feature dimension and read
  a 0.099 ceiling against a published 0.625, a second round discovered mid-run that
  its rungs fit the alignment from map predictions rather than the answer clouds the
  shipped chain uses, and a third round refuted a bolded Takeaway on a promoted body
  and filed nothing to fix it.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 problem sweep. The user-chat inline free-analysis
carve-out deliberately skips the planner + critic stack, so its scientific duties live
as an explicit list in CLAUDE.md. Three failures yesterday show the list is missing
three entries.

## Goal

Add three inline-round duties to the CLAUDE.md § "User-chat inline free analysis"
carve-out: an n-versus-d guard-rail on ridge/linear-map fits, a pre-launch diff of any
re-implemented estimator against its in-repo reference, and a same-turn obligation
when a round refutes a claim in a promoted task body.

## Workflow gap

1. **Two headline runs launched below the interpolation threshold.** Session
   `dffde9b6` @ 22:20:13Z launched a pilot and then a full 4-cell ladder against a
   4-shard subset: **n = 1,877 rows against d = 3,584 features**. The within-regime
   ceiling came out at **0.099 against a published 0.625** — the subset sat below the
   n > d interpolation threshold, so every held-out R² in that run was
   estimator-degenerate rather than a signal read. Caught only after two launches, at
   22:41:22Z, and it required writing a streaming extractor and re-staging to n = 4,724.
   Assistant verbatim: *"my first full run would have produced a *real* ladder verdict
   on a *wrong* operator."* This is the same degeneracy class another session had spent
   the same afternoon diagnosing for #825.
2. **A re-implemented estimator diverged from the shipped reference, found mid-run.**
   Same session @ 22:33:00Z, 2.5 minutes into the full ladder run: the round's rungs
   8/9 fit the answer alignment `B` from the **map's predictions → target answers**,
   whereas both the note and the shipped `issue1345_operator_comparison` chain fit `B`
   on the **answer clouds**. The round's version is strictly more permissive — it
   absorbs map error into the "coordinate change" verdict. Killed, fixed against the
   shipped `_chain`, relaunched.
3. **A round refuted a promoted body's Takeaway and filed nothing.** Session
   `63122023` @ 18:56:45Z established that #825's *"the user's next turn is linearly
   unpredictable"* is a λ-selection artifact (all four M user cells flip from
   −1.43…−1.65 to +0.19…+0.25 under two independent selectors). The round explicitly
   flagged that #825's promoted body still bolds the false Takeaway and that the 12
   banked `role-map-comparison` ROLE-GAP deltas are all computed unguarded — and then
   filed nothing. Verbatim: *"**One thing I did not fix:** #825's promoted body still
   carries 'the user's next turn is linearly unpredictable' as a bolded Takeaway…"*.
   Counting method: every `tool_use` Bash command in both sessions was grepped for
   `task.py new|file_infra_task|set-body|set-title|set-clean-result`; **two hits, both
   for creating #1689** — no corrective filing of any kind.
- **Why the existing carve-out misses these:** it already mandates a compute-character
  pre-launch statement, both mapping arms, and figure sanity. Items 1 and 2 are
  *estimator-validity* preconditions (is this fit even identifiable? is it the same
  estimator the repo already ships?), and item 3 is a *record-integrity* duty. The
  carve-out's same-turn completion contract covers committing artifacts and folding the
  finding into the parent — it does not cover a finding that invalidates a DIFFERENT
  task's promoted body.
- **Confidence (emitter):** high on all three (each quoted from the session's own
  text). Item 3 has the widest blast radius: a promoted clean-result asserting a
  refuted claim, with no owner.
- verified-at-filing: absence confirmed in the named target — searching CLAUDE.md's
  § "User-chat inline free analysis" block for the three concepts returns nothing:
  `grep -c 'interpolation threshold\|n_train' CLAUDE.md` → **0**;
  `grep -c 'reference implementation\|shipped estimator' CLAUDE.md` → **0**; the
  carve-out's same-turn contract enumerates (1) commit + push artifacts with the
  staged-index verification and the inline payload lint gate, and (2) fold the finding
  into the parent's clean-result or carry an explicit deferral — with no clause for
  refuting another task's promoted body. Incident text quoted from the two sessions'
  own messages; the "filed nothing" claim is the grep result stated above, not an
  inference. Landed-fix history check `git log --oneline --since='7 days ago' --
  CLAUDE.md` → the wave touched CLAUDE.md via several merges; none adds an inline-round
  estimator-validity or refuted-body duty. (2026-07-25)

## Proposed change (refine in planning)

```
  CLAUDE.md § "User-chat inline free analysis" — three added duties:
+ (i)  n-vs-d guard-rail: before any ridge / linear-map fit, compare n_train against
+      the feature dimension d and refuse (or loudly WARN, stated in the dispatch
+      note) when n_train < d — every held-out R² in that regime is
+      estimator-degenerate, not a signal read.
+ (ii) estimator-reference diff: an inline round that re-implements an estimator
+      already shipped in-repo records, in its dispatch note BEFORE launch, a diff of
+      the new estimator against the named reference (function + file) — not after.
+ (iii) refuted-promoted-body duty: when a round refutes a claim in ANY task's
+      promoted body (its own parent or a sibling), it must apply the body correction
+      or file a task for it IN THE SAME TURN; a chat-only "I did not fix X" is an
+      incomplete round.
```

Item (i) may be better as a mechanical assertion in the shared fit helper than as
prose — the planner should check whether `analysis/mapping_baselines` or the ridge
path has a natural home, since a guard-rail in code covers non-inline rounds too. If
so, do both: the code assert and the one-line duty.

## Scope / surfaces

- Primary target: `CLAUDE.md` § "User-chat inline free analysis" (and the parallel
  Step 9a-ter block in `.claude/skills/issue/SKILL.md`, which mirrors these duties —
  the two must not drift).
- Possibly `src/explore_persona_space/analysis/` for the (i) assertion.
- Keep the additions terse: this carve-out is already one of the longest bullets in
  CLAUDE.md, and the file is loaded on every session.

## Constraints / invariants

- (iii) must NOT authorize an inline round to rewrite another task's promoted body
  unilaterally when the correction is a scientific-meaning change — filing is the safe
  default, and the promotion classification stays user-only. Word it as
  "apply the correction **or** file", with filing as the presumption for anything
  touching a Takeaway.
- (i) must not block a deliberate under-determined fit that the round justifies —
  refuse-by-default with a stated override, matching the project's fail-fast posture.
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Related

The #825 body correction itself is NOT this task — it is a scientific-meaning change
tracked separately as a `daily-held` `needs-human` task from the same sweep. This task
adds the DUTY so the next round does not repeat the omission.

## Provenance

- workflow_fix_target: CLAUDE.md
- fingerprint: 46ff8f476147
- Source: `/daily` 2026-07-25 transcript sweep, sessions `dffde9b6` @ 22:20:13Z /
  22:33:00Z and `63122023` @ 18:56:45Z.
