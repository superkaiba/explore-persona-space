---
title: 'daily-held: #1345 base story leg lost in crash bundle'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-26T07:07:53Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 3): The #1345 base-model story
  leg was executed and lost at the last step, existing only inside a GCP crash bundle
  rather than in git, so every base-model row of the framing-chain coverage matrix
  reads stranded and the compute is already paid.'
workflow: v1
---
## Why this needs you

Filed by the `/daily` 2026-07-25 problem sweep as a **route-3 judgment call**: the
remedy is either a recovery attempt or a re-run, and a re-run spends GPU compute —
the "spends money or launches compute" carve-out item.

**#1345's base-model story leg was executed and its output never landed in git; it
exists only inside a GCP crash bundle.**

## What was found

Session `63122023` @ 2026-07-25T18:01:02Z, while building the framing-chain coverage
matrix, found that the base-model story arm had run but its result was lost at the last
step. Assistant verbatim: *"the base-model story leg turns out to have been **run and
lost at the last step** — it's sitting in a crash bundle, not in git."* Every chat↔story
row for the base model read "base **stranded**" in the coverage table for the rest of
the day. It was not recovered and nothing was filed.

This is paid compute whose result is currently unusable, and it blocks the base half of
the framing chain.

## Verified at filing (2026-07-25)

- `task.py view 1345 --json` → `status: awaiting_promotion`.
- The GCP crash-persist path is real and is where such output would be: the EXIT-trap
  `_eps_persist_diagnostics` uploads partial artifacts to the HF data repo under
  `issue<N>_partial/<attempt_id>/` on a nonzero-rc exit (CLAUDE.md § Compute backends,
  GCP mechanics) — which is exactly the "sitting in a crash bundle" shape.

**Unverified — verify before acting:** I did not enumerate
`superkaiba1/explore-persona-space-data` `issue1345_partial/` to confirm the leg's
artifacts are present and complete, nor check whether the bundle holds a finished
result versus a partial one. That read is the first step of any recovery and needs an
HF listing, not a recall.

## The decisions that are yours

1. **Recover vs re-run.** If the crash bundle holds a complete result, recovery is
   cheap (download, verify, commit to `eval_results/issue_1345/`). If it is partial,
   re-running costs GPU time — your call.
2. **Whether the base half of the framing chain is needed** for #1345's promotion at
   all, or whether the instruct-only coverage plus a stated scope caveat is enough.

## Related work already routed tonight

A route-2 workflow-fix task from this sweep proposes the *systemic* half — a periodic
audit reconciling `issue<N>_partial/` crash-bundle contents on the HF data repo against
`eval_results/issue_<N>/` in git, escalating any bundle that carries a completed result
which never landed. That is the mechanism that would have surfaced this within a day
instead of on a chance discovery; it does not itself recover #1345's leg.
