---
title: 'code-style resume-key clause: forbid hashing raw bytes of a RECOMPUTED float
  array (platform-dependent key discards valid checkpoints)'
kind: infra
tags: []
created_at: '2026-08-07T11:53:10Z'
has_clean_result: false
origin_prompt: '#1336 Round B inline recovery: VM-side re-aggregation declared all
  32 valid cells stale because grid_sha hashes np.logspace bytes that differ by 1
  ULP between the pod CPU and this VM'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a candidate raised on task #1336
(emitting agent: orchestrator, own observation during the Round B inline recovery round).

`.claude/rules/code-style.md`'s resume-key clause requires a resume predicate "keyed on
EVERY output-affecting regime key". It does not say anything about whether that key is
STABLE ACROSS MACHINES. The most natural implementation of "key on the regime" —
sha256 over the raw bytes of the regime array — is platform-dependent when the array is
RECOMPUTED from a formula rather than loaded from a file, so the resume key silently
fails to match on a different machine and a correct, expensive checkpoint is discarded
as stale.

## Goal

Add one clause to the resume-key rule: a resume/cache key must not hash the raw bytes of
a RECOMPUTED float array. Hash the GENERATING PARAMETERS instead (or a rounded/quantized
form). Plus a compact `gotchas.md` entry carrying the discriminating signature, so the
next author does not spend a debugging cycle on it.

## Workflow gap

- **Bug observed:** #1336's fit entrypoint keys its resume predicate on
  `grid_sha = sha256(np.asarray(cm.LAMBDAS_23, float64).tobytes())[:16]`, where
  `LAMBDAS_23 = np.logspace(-3, 8, 23)`. Under the SAME numpy 2.2.6, the RunPod CPU pod
  and the GCP VM disagree by 1 ULP at index 11 (`10**2.5`):
  pod `316.2277660168379` vs VM `316.22776601683796`. So `grid_sha` is
  `90c25fa4156d2530` on the pod and `98e2d075b4fd3fcc` on the VM. Consequence: a VM-side
  re-invocation printed `stale checkpoint (regime changed) — refitting` for all 32 valid
  cells and would have refit ~5 CPU-h plus re-staged ~70 GB. The 32 cells were fine; the
  KEY was wrong. Numerically the ULP cannot matter (a 1-ULP shift in one of 23 CANDIDATE
  ridge lambdas cannot move a selected lambda), which is exactly why it is dangerous —
  it costs compute and reads like a real regime change.
- **Why it is a workflow gap:** the rule's demand is correct and the failure is in the
  obvious implementation of it. An author who follows the clause verbatim ("key on every
  output-affecting regime key") and reaches for a content hash of the regime array gets a
  key that works perfectly on one machine and silently invalidates itself on the next.
  The rule already anticipates the OPPOSITE error (a resume that IGNORES a
  `--method`-class flag, #722 r3) but not the over-specified-key error, which has the
  same cost signature — recomputing work that was already valid.
- **Diagnosis cost when it fires:** it presents as a version skew, so the first
  hypotheses are all wrong. On #1336 it took a script md5 comparison (byte-identical,
  `25f0779d21bc17460047674ae0e66de5`), a `git log` of the 5 commits the VM was ahead by
  (none touched `common.py`, the fit core, or the metric ladder), and a numpy-version
  check (identical) before printing the array values and finding the ULP.
- **Confidence (emitter):** high — measured directly on both machines this round, and
  the mechanism (libm/CPU differences in `10**x` for non-exact x) is well understood.
- verified-at-filing: `grep -c -iE 'ulp|platform-dependent|libm|float bytes|tobytes|
  bit-exact|portable' .claude/rules/code-style.md` -> **0**. On
  `.claude/rules/gotchas.md` the same grep returns 1, which is a FALSE POSITIVE: `ulp`
  matching inside the word "culprit" at line 54 (an unrelated CUDA residual-mismatch
  entry). So both targets have ZERO real coverage; the in-target 0-hit IS the evidence
  for this absence-of-guard claim. (2026-08-07)
- relocation sweep (clause (b)): `grep -rn 'tobytes()' --include=*.py scripts/` filtered
  to hash call sites and then to non-integer arrays returns **9** float-byte hash sites
  (`issue1092_figures.py`, `issue1901_metric_battery.py`, `issue1776_{jacobian,phase3,
  swap}.py`, `issue928_fit_decomposition.py`, `issue779_{ffc_n1m_fits,
  fitter_fair_comparison}.py`). These are DISCLOSED rather than counted as instances:
  most hash arrays LOADED FROM FILES, whose bytes are bit-identical on any machine, so
  they are safe. The failure needs a RECOMPUTED array, and #1336's `_grid_sha` is the
  instance actually found. The many integer-array hash sites
  (`asarray(ids, dtype=np.int64)`, `row_idx`, `perms`) are bit-exact and never at risk.
  Scoping the rule to recomputed FLOAT arrays keeps it from firing on all of these.
- landed-fix history (clause (a')): `git log --oneline --since='10 days ago' --
  .claude/rules/code-style.md .claude/rules/gotchas.md` shows no commit touching hash
  stability or float portability on either file.

## Proposed change (candidate diff sketch — refine in planning)

In the `.claude/rules/code-style.md` resume-predicate clause, after "keyed on EVERY
output-affecting regime key", add:

```
+ **but the key must be MACHINE-STABLE.** Never hash the raw bytes of a RECOMPUTED
+ float array (`sha256(np.logspace(...).tobytes())`, a fit result, any derived
+ float grid) — libm/CPU differences make the last bits differ across machines, so
+ the key changes without the regime changing and every valid checkpoint is thrown
+ away as stale. Hash the GENERATING PARAMETERS instead (`(-3, 8, 23)`), or a
+ rounded/quantized form (`np.round(g, 12)`). Bit-exact inputs — integer arrays,
+ or float arrays read from a file — are safe to hash directly.
```

Plus a compact `.claude/rules/gotchas.md` entry with the signature: a resume predicate
reporting "stale checkpoint / regime changed" for work you believe is current, while the
script file is byte-identical and the library versions match ⇒ suspect a float-derived
key before suspecting a version skew; print the array and compare values, not hashes.

## Scope / surfaces

- Primary targets: `.claude/rules/code-style.md`, `.claude/rules/gotchas.md`
- Consider whether `.claude/agents/code-reviewer.md` Step 3.6 (the existing review-side
  gate for the checkpoint/resume rule) should also flag a float-bytes hash in a resume
  key, since that is where a new instance would be caught before it costs a refit.
- Do NOT change `scripts/issue1336_selfmap_missing_pairs.py` under this task. Its 32
  committed cells carry `grid_sha 90c25fa4156d2530`; editing `_resume_key` would
  invalidate every one of them. If that entrypoint is ever fixed it needs a
  key-migration path (accept either hash, or re-key on the generating parameters and
  rewrite the 32 stored keys), which is its own scoped decision — the wart is recorded in
  #1336's `epm:upload-verification` marker.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Documentation/promotion change; no result-equivalence risk.
- Both target files are size-capped by `workflow_lint.py` — prefer a COMPACT entry over
  a narrative one. `--check-lessons-index` stays green (no new rule file).
- This session runs under the workflow-fix recursion guard once filed.

## Provenance

- workflow_fix_target: .claude/rules/code-style.md,.claude/rules/gotchas.md
- fingerprint: 095c68c0ab82

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/code-style.md,.claude/rules/gotchas.md
bug_observed: A resume key built as sha256 over np.logspace(-3,8,23).tobytes() differed between a RunPod CPU pod and the GCP VM under the same numpy 2.2.6 (1 ULP at index 11, 10**2.5: 316.2277660168379 vs 316.22776601683796), so a VM-side re-invocation declared all 32 valid #1336 selfmap_v3 cells stale and would have refit ~5 CPU-h plus re-staged ~70 GB. It presents as a version skew, so diagnosis first burned a script md5 comparison, a git log of the 5 intervening commits, and a numpy-version check.
why_workflow_gap: code-style.md's resume-predicate clause requires keying on every output-affecting regime key but says nothing about the key being machine-stable, and the obvious implementation of that demand — hashing the regime array's raw bytes — is platform-dependent for recomputed float arrays. The rule guards the opposite error (a key that ignores a --method-class flag, #722 r3) but not the over-specified key, which has the same cost signature: recomputing work that was already valid. Zero coverage in code-style.md or gotchas.md.
proposed_change: Add a machine-stability clause to the resume-key rule — never hash raw bytes of a RECOMPUTED float array; hash the generating parameters or a rounded form; bit-exact inputs (integer arrays, file-loaded floats) stay safe to hash — plus a compact gotchas entry whose signature is "stale checkpoint reported while the script is byte-identical and library versions match ⇒ suspect a float-derived key, print values not hashes".
diff_sketch: |
  code-style.md, resume-predicate clause, after "keyed on EVERY output-affecting regime key":
  + but the key must be MACHINE-STABLE. Never hash the raw bytes of a RECOMPUTED float
  +   array (sha256(np.logspace(...).tobytes()), a fit result, any derived float grid) —
  +   libm/CPU differences move the last bits across machines, so the key changes without
  +   the regime changing and valid checkpoints are discarded as stale. Hash the
  +   GENERATING PARAMETERS ((-3, 8, 23)) or a rounded form (np.round(g, 12)). Integer
  +   arrays and file-loaded floats are bit-exact and safe to hash directly.
  + gotchas.md: compact entry with the diagnosis signature + the #1336 citation.
  + consider code-reviewer.md Step 3.6 flagging a float-bytes hash inside a resume key.
confidence: high
related_task: #1336
<!-- /workflow-fix-candidate -->
