---
name: dial-change-stale-bundle-reachability
description: Discharge a presence-only-resume vs changed-dial concern by proving every pre-change dial value FAILED before the bundle artifact was written
metadata:
  type: feedback
---

When a commit changes a config dial (smoke quota, threshold, exclusion semantics) whose output bundle is guarded by a presence-only resume predicate (exists + sha-sidecar self-consistency, no dial in the fingerprint), the stale-resume concern ([[size-match-resume-skip-npz]], [[new-dial-missing-from-resume-regime]]) is DISCHARGEABLE without a code change when: every pre-change dial value provably RAISED before the bundle file was written, AND that file is in the resumability file list — then no resumable stale bundle can exist for this transition. Verify both halves: (1) locate the raise site and confirm it precedes the artifact write in program order; (2) confirm the artifact is in the predicate's required-file tuple.

**Why:** #2544 r1 g4 — smoke exemplar quotas {40,12,12}→{40,40,40} + corpus-wide exclusion, with `_config_bundle_resumable` presence-only. Both pre-change configs raised inside `select_exemplar_bank` before `exemplars.json` (a required bundle file) was written ⇒ Minor-informational, not Major. Same round also showed the POSITIVE smoke pattern: resizing a smoke axis UP to a measured constraint floor (gate runs identically) beats downgrading the gate — divergence-REDUCING smoke commits get the recipe: single-consumer grep, production-branch-first ternary, rows/derived-var leak trace, smoke upload-namespace guard.

**How to apply:** on any dial-change commit near a resume/skip predicate, run the reachability probe BEFORE grading severity; a pre-change config that COMPLETED (artifact written) keeps the full Major severity and requires the dial in the fingerprint or a wipe step.
