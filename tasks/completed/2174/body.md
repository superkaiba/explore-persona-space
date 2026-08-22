---
title: 'Fact-checker: exact-identity premise from a sample must be verified at the
  assert''s grain'
kind: infra
tags:
- workflow-fix
- fact-checker
created_at: '2026-08-07T13:53:46Z'
has_clean_result: false
origin_prompt: '#2163 Phase 0 census died rc=1 on a byte-identity assert grounded
  on a 10-shard/706-row probe (0.5% of 142,000 rows); the exactness claim carried
  Confidence: High (measured) through the whole review stack.'
workflow: v1
---
# Fact-checker: an exact-identity premise that becomes a runtime assert must be verified at the assert's grain

## Goal

Make the Phase 1.5 fact-checker flag any plan assumption that (a) asserts an EXACT identity — zero
variance, `n_distinct == 1`, byte-identical, "no exceptions", "for every row" — AND (b) is grounded on
a SAMPLE, when (c) the plan converts it into a runtime assert or a hard-coded constant. Either the
premise gets verified at the grain the assert will run at, or it is downgraded to an explicit sampled
BOUND before the plan is approved.

## The gap

The fact-checker verifies each assumption's confidence and source, and it does this well for VALUE
claims (a λ, a byte count, a key set). It has no lens for the SAMPLING-GRAIN MISMATCH between an
exactness claim and the assert built on it. An exactness claim is qualitatively different from a value
claim: a sample can only ever establish "no counterexample observed in N draws", never "zero
counterexamples exist". Converting the former into an assert that demands the latter guarantees a
crash the moment the full population is touched — and the crash lands AFTER provisioning, at the
first phase that reads the whole corpus.

## Observed on #2163 (2026-08-07)

Plan §12 A11 asserted, at `Confidence: High (measured)`:

> `h_prefix` is (62, 3584) fp16 with EVERY row byte-identical — max|row − row0| = 0 exactly, all
> pairwise cosines = 1.000000, `n_distinct_rows = 1` … v_P carries exactly zero cross-context variance

Evidence: one shard (62 rows), strengthened by the fact-checker to 10 shards / 706 rows — **0.5% of
the 142,000 rows**. The fact-checker's own report recorded the strengthening as a CONFIRMED item
("Prefix degeneracy holds ACROSS shards … max|h_prefix row − shard00 row0| = 0.0 exactly over all 10
sampled shards"), which is true and still insufficient: going 1 shard → 10 shards raises coverage from
0.04% to 0.5% and does not change the KIND of claim.

The plan then registered a Phase-0 runtime assert demanding byte-identity over the full store. Result:
pod provisioned, ~14 GB staged, Phase 0 census ran to completion, then died rc=1 on
`[census] prefix degeneracy violated: 258 distinct rows`. Full-grain truth: 258 / 142,000 rows (0.18%)
deviate, spanning 4 distinct vectors store-wide, all within cosine 0.99989 — capture-time numerical
noise, so the SUBSTANTIVE conclusion (prefix arm degenerate) survived, but the plan's literal premise
and the assert built on it were both false.

Cost: one pod launch cycle, ~7 min of compute, a plan revision (v6), and a crash-fix round. All of it
avoidable by a plan-time flag, because the full-grain check is CHEAP — it is exactly the read the
census phase already performs.

Note the near-miss that makes this worth wiring: A11's own "How to verify" line already said "Phase 0
re-asserts … over the FULL staged store". The plan KNEW the sample was not the population and still
carried `Confidence: High (measured)` with the exact claim in the §4 stated deviation. Nothing in the
review stack connected "verification is deferred to runtime" with "and the runtime check is a hard
assert that will fail".

## Proposed fix (sketch, for the planner to adjudicate)

Add a fact-checker lens, roughly: **exactness-claim grain check.** For each assumption whose text
matches an exactness pattern (`exactly zero`, `n_distinct == 1`/`= 1`, `byte-identical`, `identical
across`, `every row`, `no exceptions`, `max|…| = 0`, `all pairwise … = 1.000000`):

1. Identify the claim's evidence GRAIN (rows / shards / units actually examined) and the POPULATION
   grain the plan will apply it to. Report both explicitly, as a ratio.
2. If grain < population AND the plan converts the claim into a runtime assert, a hard-coded constant,
   or a stated deviation from a standing rule, emit a BLOCKING finding with two acceptable remedies:
   verify at full grain now (when cheap — often it is, as here), or restate the assumption as a bound
   ("no deviation observed in N of M rows") AND soften the assert to the invariant the bound actually
   supports.
3. Leave value claims and non-assert-bearing exactness claims alone — this must not become a
   confidence-downgrade tax on every measured number.

Consider a `verify_plan.py` companion (WARN) for the mechanical half: an assumption line containing an
exactness pattern together with a sample-size marker ("N sampled", "N-shard", "N-row sample") and no
full-grain verification statement.

Related but distinct, worth mentioning in the same fix: the #2163 code ALSO hard-coded the expected
measurement into the artifact reporting it (`"prefix_degeneracy": {"n_distinct_rows_h_prefix": 1}` in
`census.json`) — a literal that would have published the assumption AS DATA even on a healthy run.
That is the CLAUDE.md "no value placeholders" rule; whether the code-reviewer should carry an explicit
"a measurement field must be assigned from a measured variable, never a literal" check is a judgment
call for the planner of this fix.

## Acceptance criteria

1. A plan asserting an exact identity from a stated sample, and building a runtime assert on it, draws
   a BLOCKING fact-checker finding naming the grain ratio.
2. The finding offers both remedies (verify at full grain / restate as a bound + soften the assert).
3. A value claim with a sampled source does NOT trigger the lens (no false-positive tax).
4. An exactness claim already verified at full grain does NOT trigger it.
5. Tests pin (1), (3), (4); `workflow_lint.py` passes; `--check-lessons-index` updated if a rule file
   changes.

## Provenance

Filed by the #2163 orchestrator after the crash. The durable lesson is on #2163 as an
`epm:failure-lesson v1` (`generalizes: yes`); this task is the workflow-surface half — the lesson alone
only reaches agents who read that task's memory, whereas the failure mode is available to every plan
that grounds an exactness premise on a probe.

workflow_fix_target: .claude/agents/planner.md
