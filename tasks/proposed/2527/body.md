---
title: Smoke and production legs must not share one HF upload prefix (upload-policy
  gap)
kind: infra
tags:
- workflow-fix
created_at: '2026-08-24T08:31:53Z'
has_clean_result: false
origin_prompt: 'Found during /issue 2389: scripts/issue2389_run.py:124 HF_PREFIX is
  a module-level constant with no leg discriminator, so the --smoke leg and the production
  leg upload to the same HF prefix; a smoke-era _done manifest and a 42 KB prior-run
  shard sat under production names for a cell production was still generating. upload-policy.md
  never says run legs must be namespaced.'
workflow: v1
---
# A smoke leg and its production leg must not share one HF upload prefix

## Goal

Close a workflow-surface gap in `.claude/rules/upload-policy.md`: the upload
policy namespaces nothing by RUN LEG, so an experiment whose chain runs a
`--smoke` leg followed by a production leg uploads BOTH legs' artifacts into the
same HF prefix. Stale smoke-era artifacts then sit under production names, where
a presence / name-set existence check reads them as production output.

Add the rule + the mechanical guard so a smoke leg's uploads are structurally
distinguishable from production's.

## Provenance

`workflow_fix_target: .claude/rules/upload-policy.md`

Found live during `/issue 2389` (`kind: experiment`, workflow v2) on
2026-08-24 at ~08:30Z, while the production run was mid-flight. Recorded in
full on #2389 as `epm:progress v53`.

## The concrete evidence (#2389)

`scripts/issue2389_run.py:124` sets a module-level constant:

```python
HF_PREFIX = "issue2389_q38ce"
```

with no leg discriminator. The relaunch-3 chain (`epm:run-launched v4` on
#2389) is, verbatim:

```
... EPM_2389_OUT_ROOT=/workspace/issue2389_smoke_out bash scripts/issue2389_dispatch.sh all --smoke \
 && ... bash scripts/issue2389_dispatch.sh all
```

The OUT-ROOT is namespaced per leg (`/workspace/issue2389_smoke_out` vs
`/workspace/issue2389_out`); the HF prefix is NOT. Both legs upload to
`superkaiba1/explore-persona-space-overflow/issue2389_q38ce/...`.

Observed state, HF last-commit dates against the 04:28Z smoke-to-production
boundary (the smoke gate PASSED at 04:28Z; production started right after):

| path under `issue2389_q38ce/` | last_commit | size | era |
|---|---|---|---|
| `raw_completions/anchors/anchors_parity_persona_prompted_w0.jsonl` | 07:18:04Z | 1.37 MB | production |
| `raw_completions/anchors/anchors_parity_fact_user_name_w0.jsonl` | 05:51:58Z | 111 KB | production |
| `analysis_tensors/anchors/va_anchors_parity_filler_swap_w0.pt` | 03:19:44Z | 5.2 MB | SMOKE leg |
| `analysis_tensors/manifests/anchors_parity_filler_swap_w0_done.json` | 03:21:27Z | 730 B | SMOKE-era DONE manifest |
| `raw_completions/anchors/anchors_parity_filler_swap_w0.jsonl` | 2026-08-23T22:39:58Z | 42 KB | an EARLIER run |

At the moment of observation the production run was still generating
`parity_filler_swap` (unit 16/36) — so a smoke-era `_done` manifest and a
42 KB prior-run raw shard were sitting under the production names for a cell
production had not finished. The 42 KB stale shard against its 1.37 MB
production sibling is a ~32x size tell.

## Why this is a workflow-surface gap, not just a #2389 code bug

`scripts/issue2389_run.py` is per-issue experiment code and its own fix belongs
to #2389. But the reason it was written that way is that the upload policy never
says legs must be namespaced. `.claude/rules/upload-policy.md` specifies
destinations by ARTIFACT CLASS (`raw_completions/<stage>/`, `analysis_tensors/`,
adapters, datasets) and by issue, never by RUN LEG — so a smoke leg inherits the
production prefix by default and every future smoke-then-production chain
reproduces this. The `<stage>` axis that exists today
(`extraction`/`monitoring`/`final`) is orthogonal: it separates PHASES of one
leg, not smoke from production.

Sibling rule `.claude/rules/smoke-blind-spots.md` covers what a smoke PASS does
NOT CERTIFY. This is the inverse direction and is uncovered: what a smoke leg
CONTAMINATES. Worth a cross-reference in both files.

## Two failure modes it creates

1. **Stale-artifact false PASS at upload verification** — the #779 class (a
   stale prior-version artifact satisfies an existence check instantly). A
   presence / name-set reconciliation over the HF prefix counts a smoke-era
   shard as the production cell. The `rows=` realized-row reconciliation
   (#2148) and the out-root residue NAME-SET diff (#2162) are the existing
   partial defenses, but neither is stated to be leg-aware.
2. **Smoke-slice contamination of a `--stage-from-hf` read** — a downstream
   judge / margin / analysis stage that stages inputs from the HF prefix can
   pull a smoke-sliced artifact (tiny per-arm-class slice) and score it as
   production. On #2389 this was live: the pending VM-side duty is
   `issue2389_judge.py --phase vllm-parity --stage-from-hf`, and dispatching it
   before production overwrote `filler_swap` would have staged the 03:21Z /
   Aug-23-22:39Z shards. The #2389 session HELD that dispatch for exactly this
   reason.

Note the hazard is self-healing IF production completes every cell (production
overwrites the stale names) — which is precisely why it is dangerous: it
disappears on the happy path and bites on an interrupted run or an early
staged read.

## Proposed change (the implementer decides the final shape)

1. **`.claude/rules/upload-policy.md`** — add a clause: a run leg that is not
   the production leg (`--smoke`, `--tiny`, `--pilot`, any gate/dry leg) MUST
   NOT upload into the production HF prefix. Either namespace the leg
   (`issue<N>_<slug>/_smoke/...`) or suppress the leg's uploads entirely.
   Non-production legs are re-runnable by construction, so suppression is
   usually correct and cheaper; namespacing is right when the smoke leg's
   artifacts are themselves evidence (a gate verdict worth keeping).
2. **Leg-awareness in verification** — state that upload verification and the
   `pod.py terminate` guard's `rows=` / `outroot=` tokens are evaluated against
   PRODUCTION-leg artifacts only, and that an artifact whose HF `last_commit`
   predates the production leg's start is NOT production output. A commit-date
   floor is the cheap mechanical form.
3. **Cross-reference** `.claude/rules/smoke-blind-spots.md` (what a smoke PASS
   does not certify) <-> the new clause (what a smoke leg contaminates).
4. **Mechanical guard (preferred if tractable)** — a `workflow_lint.py` check
   that flags a module-level HF-prefix constant consumed by a script which also
   parses a `--smoke`/`--tiny`/`--pilot` flag, with no leg discriminator in the
   uploaded path. This is the shape that would have caught
   `issue2389_run.py:124` at review time; the #2389 code-review ensemble ran 10
   rounds and did not.

## Acceptance criteria

- [ ] `.claude/rules/upload-policy.md` carries the leg-namespacing clause and
      the production-leg-only verification statement.
- [ ] `.claude/rules/smoke-blind-spots.md` cross-references it.
- [ ] `.claude/rules/LESSONS.md` trigger row updated if the fires-when changes.
- [ ] A mechanical check exists, or the task records why it is not tractable
      and the reviewer lens is the binding gate instead.
- [ ] The #2389 code fix is NOT in scope here (that is #2389's own follow-up);
      this task owns the rules surface + guard only. Reference #2389
      `epm:progress v53` as the incident of record.
