---
title: 'artifact-reuse: a reused ACTIVATION artifact must carry the model-weights
  revision that produced it (shape asserts and data-repo pins verify different guarantees)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-19T21:44:11Z'
has_clean_result: false
origin_prompt: 'surfaced by codex-methodology-baselines-critic and upheld on the mechanism
  by the binding methodology reconciler during #2329 q35_ladder_decay critique round
  1: a reused Qwen3.5 donor activation bank gating the null_xtype transfer verdict
  had no model-weights-revision identity with the receiving model; two Claude reviewers
  passed it having each verified a different guarantee'
workflow: v1
---
# artifact-reuse: a reused ACTIVATION artifact must carry the MODEL-WEIGHTS revision that produced it

## Provenance

workflow_fix_target: .claude/rules/artifact-reuse.md

Surfaced by the `codex-methodology-baselines-critic` twin during the #2329
`q35_ladder_decay` post-approval critique panel (round 1) and UPHELD on the
mechanism by the binding methodology reconciler, which narrowed the remedy but
confirmed the gap is real and closed nowhere in plan or code.

## The gap

`.claude/rules/artifact-reuse.md`'s fitness checklist covers sha-pinning of the
ARTIFACT — which bytes get fetched, at which data-repo revision. Nothing in it
pins, records, or asserts the MODEL-WEIGHTS revision that PRODUCED the artifact.

For text artifacts (rollout completions, judge outputs, banks of context text)
that is fine: they are frozen at generation and consumed downstream without a
live model. For HIDDEN-STATE artifacts — cached activations, donor state banks,
residual-stream captures, anything later injected into or compared against a
live model's activations — it is not, because those tensors are
weight-basis-dependent. States captured under one set of weights and injected
into a model loaded from different weights are silently wrong.

Every existing gate misses it:

- A shape assertion (`donor_state.shape == recipient.shape`) catches
  wrong-ARCHITECTURE only. Two revisions of the same model are shape-identical,
  so an off-revision payload passes.
- A data-repo revision pin binds which artifact bytes are fetched, not which
  weights produced the states inside them.
- An injection-exactness gate confirms the write landed, not that the payload
  was on-basis.

## The concrete incident

In #2329's round, a reused Qwen3.5 donor activation bank (`vc_bank.pt`) was
staged for the `null_xtype` control arm — an arm that GATES the transfer
verdict. The task body records that the original capture ran against an unpinned
`main` with the pod-side resolved commit unrecoverable, and the loading path
still called `from_pretrained` for both tokenizer and model with no `revision=`.
So neither the capture side nor the injection side pinned weights, and every
gate above would have accepted an off-revision but correctly-shaped payload.

Two Claude reviewers (the methodology critic and the consistency-checker) both
PASSED this, each having verified a real but DIFFERENT guarantee — artifact-byte
identity and architecture compatibility. The cross-model twin caught it, and the
reconciler's ruling was explicit: those are different guarantees from
weight-basis identity, and the plan closes the latter nowhere. That two
independent reviewers can verify adjacent guarantees and both miss this one is
the argument for mechanizing it rather than relying on review.

(For that specific round the exposure turned out to be materially empty: the
model repo's last weight-bearing commit predated the capture by ~5.5 months, so
any resolution of `main` in the capture window provably hit the same weights.
That resolution required a reconciler to go query the Hub commit history — which
is exactly the check being proposed here, done by hand.)

## Proposed fix

1. **Rule change** — add a fitness item to `.claude/rules/artifact-reuse.md`
   scoped to activation/hidden-state artifacts: the artifact must carry the
   producing model's resolved revision as metadata, the consuming run must pin
   the same revision on every relevant `from_pretrained` (tokenizer AND model),
   and the two must be asserted equal before use. Where the artifact predates
   this rule and carries no revision key, the acceptable substitutes are (a) a
   commit-history derivation proving no weight-bearing commit falls between
   capture and pin, or (b) a re-forward equivalence probe (re-capture 2-3 donor
   contexts on the pinned model and assert per-layer cosine above a stated bar).
   Text artifacts are explicitly OUT of scope — say so, so the rule does not
   get over-applied to rollout completions.
2. **Verifier check** — assert that a plan reusing an activation artifact either
   cites a recorded producing-revision or names one of the two substitutes, and
   that its stated load path pins a revision. WARN when the artifact class
   cannot be determined mechanically; do not FAIL on an unresolvable case.
3. **Capture-side hygiene** — the deeper fix is that new captures should RECORD
   their resolved model revision in the repro bundle, so this can be checked by
   equality rather than by archaeology. Worth stating in the rule as the forward
   requirement even though it cannot be applied retroactively.

## Acceptance criteria

1. The rule states the activation-artifact requirement, the two substitutes for
   legacy artifacts, and the explicit text-artifact exclusion.
2. A fixture plan reusing an activation artifact with no producing-revision and
   no substitute is flagged; one citing a substitute is not.
3. A fixture reusing only TEXT artifacts is unaffected.
4. The forward requirement (record the resolved revision at capture time) is
   stated.
5. Grep for existing activation-store reuse across committed plans and report
   how many would flag, so the change does not newly block grandfathered work.
