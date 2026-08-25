---
title: 'verify_plan.py: dry-run plan-embedded jq probes; gate contingent judge waves;
  require regen-time max_model_len re-pin'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-25T21:38:08Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #2588 /adversarial-planner Phase 2 critic ensemble
  (2026-08-25): the Methodology and Statistics critics each independently flagged
  a plan-embedded jq probe that returns 0 against its committed target while feeding
  a registered gate input, and each surfaced a distinct additional verify_plan check
  (contingent judge-wave pilot gate; cap-hit regen max_model_len re-pin, recurring
  across #505/#601/#2221/#2588).'
workflow: v1
---
# verify_plan.py: three plan-time checks surfaced by #2588's critic ensemble

Three concrete, independently-surfaced `verify_plan.py` gaps came out of the #2588 Phase 2
critic ensemble. All three target the SAME file, so they are filed as ONE task: three
concurrent `--auto` sessions editing `scripts/verify_plan.py` would collide
(`.claude/rules/cross-session-writer-arbitration.md`). Implement as one coherent diff; each
check ships with its own fixture-backed test.

## Check A — dry-run plan-embedded `jq` probes against committed targets

Surfaced INDEPENDENTLY by both the Methodology and Statistics critics on #2588 plan v2, and
reproduced by the orchestrator.

The plan registered, in three separate places (§4.4 P0 step 4, §7, §12 A7):

    jq '.train_10k|length, .val_400|length, .test_1000|length' eval_results/issue_2330/split_ids.json

expecting `10000/400/1000`. Executed against the committed artifact it exits rc=5 and yields 0:
the file's top-level keys are `counts`/`splits`/`sha256`/..., and the ids live under `.splits.*`.
The probe is the plan's REGISTERED GATE INPUT for measured `n_train`, which the #1887 n-vs-d
reads consume — so with the expect-assert P0 halts on a healthy artifact, and without it a `0`
is recorded as the measured n_train.

There is a second, subtler failure mode worth encoding in the same check, because BOTH critics'
proposed fixes were still broken by it. jq's `|` binds LOOSER than `,`, so the naive repair

    jq '.splits.train_10k|length, .splits.val_400|length, .splits.test_1000|length'

parses as a re-pipe chain and still exits rc=5 (`Cannot index number with string "splits"`).
Working forms, both verified 2026-08-25:

    jq '(.splits.train_10k|length), (.splits.val_400|length), (.splits.test_1000|length)'   # 10000/400/1000, rc=0
    jq -c '.counts'                                                                          # single read, no precedence trap

A plan-time dry-run catches BOTH the wrong-path and the precedence shapes for free, because it
just runs the probe. That is the point: the check needs no jq grammar model, only execution.

PROPOSED CHECK: for each plan-embedded `jq` invocation whose target resolves to a file committed
in the repo, execute it read-only and FAIL on nonzero rc or null/empty output. Where the plan
states an expectation adjacent to the probe (the `→ expect 10000/400/1000` shape), diff the
realized output against it and FAIL on mismatch. Skip (never fail) when the target does not
resolve — an uncommitted or run-generated path is not evidence of a broken probe.

## Check B — contingent judge waves must name their pilot gate

Surfaced by the Statistics critic; the Methodology critic flagged the same contingency
independently.

#2588 §4.5 pre-registers "> 5% extraction failure → fall back to a Sonnet judge for the unparsed
residue (~≤19k Batch-API calls)" and §8 rates the trigger Medium. No rubric shape, no
`max_tokens` (rule-23 floor ≥ 1024), no parse-contract round-trip (rule 27), no rule-26 pilot
gate. `.claude/rules/llm-judging.md` rule 26 makes an ungated ≥ ~5k-call wave a Statistics-lens
REVISE, and #1739 is the 100%-parse-fail precedent.

The gap the check closes: rule 26's existing enforcement reads the plan's PRIMARY judge wave. A
wave that only fires CONDITIONALLY is invisible to it, even though its call estimate is stated
in the same block, so the discipline silently does not reach the contingency.

PROPOSED CHECK (WARN): a plan naming a judge fallback / contingent judge wave with a ≥ 5k call
estimate, and no pilot-gate vocabulary within the same block, WARNs.

## Check C — cap-hit regen must re-pin `max_model_len`

Surfaced by the Methodology critic, which noted the recurrence explicitly: #505, #601, #2221,
now #2588.

#2588 §4.4 instantiates `vLLM(model_id, max_model_len=budget+GEN_MAX)` with a 7,104 prompt-token
budget, while §7 G4/G5 pre-register "regen of affected rows at 2× cap". For think-GPQA (cap
8,192) the regen needs 7,104 + 16,384 = 23,488 tokens against an engine pinned at ≤ 15,296 — so
the registered remedy errors or re-truncates at the SAME boundary it exists to escape, exactly
where the plan's own risk table rates cap-blowout Medium.

This is the standing CLAUDE.md rule ("raising a cap on an INHERITED rig ⇒ re-check its
`max_model_len` pins") going unenforced mechanically across four issues.

PROPOSED CHECK (WARN): a plan carrying BOTH a cap-hit regen trigger (`regen … at 2× cap` /
`cap_hit` vocabulary) AND a `max_model_len` expression derived from the base cap, with no
regen-time re-pin statement, WARNs.

## Acceptance

- Three checks land in `scripts/verify_plan.py` with stable ids, wired into the no-flags run.
- Each carries a fixture-backed test reproducing the shape it catches. Check A's fixtures cover
  BOTH the wrong-path and the jq-precedence shapes, since the second is what defeated the two
  hand-written repairs.
- Each check's N/A escape phrase is registered in the canonical list the adversarial-planner
  skill quotes, following the existing grammar (standalone declaration line, list markers
  tolerated).
- `uv run python scripts/workflow_lint.py` and the mapped tests pass.
- Re-running `verify_plan.py --issue 2588` demonstrates Check A firing on the pre-revision plan
  text and clearing after the #2588 Phase 3 revision corrects the probe.

## Provenance

Surfaced during the #2588 `/adversarial-planner` Phase 2 ensemble (2026-08-25) by the Claude
Methodology and Statistics critics, both as explicit workflow-surface prose follow-ups. The
orchestrator independently reproduced Check A's failure and discovered the jq-precedence second
shape while validating the critics' proposed repairs.
