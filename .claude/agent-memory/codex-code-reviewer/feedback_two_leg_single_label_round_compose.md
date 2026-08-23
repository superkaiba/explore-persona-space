---
name: two-leg-single-label-round-compose
description: One followup_label can carry TWO parallel legs (separate implementers/worktrees, per-leg impl + smoke-arch markers, ONE shared plan) — compose per-leg with leg-suffixed filenames, leg-scoped sentinel v1, round-match markers by leg not label, and mark the other leg's plan rows N/A never ✗
metadata:
  type: feedback
---

A same-issue follow-up round can be ONE `epm:followup-scope` marker carrying
TWO legs built by SEPARATE implementers in SEPARATE worktrees (#1739
`composition-grid-multiseed-plus-arm2-repair`, 2026-08-22: leg 1 in the
issue-1739 worktree, leg 2 in i1739-fit). This extends
[[concurrent-followups-wrong-plan-symlink]] (two LABELS) to two LEGS under
ONE label sharing ONE plan version.

**Why:** label-grain round-matching is not enough — `latest-marker` for
`epm:smoke-architecture-check` returns whichever LEG posted last (here v23
leg-2 sat above v22 leg-1 by 17 min); the impl marker must be matched by its
body's leg line, and the plan (both legs share it) is round-current even
though half its rows belong to the other leg.

**How to apply:**
- Filenames: leg-suffixed (`codex-code-reviewer-<N>-a2fix-r1-*`), per the
  brief's example — a bare `-r1-` name collides with the other leg's r1.
- Sentinel: the brief's `revision_round` is LEG-scoped (v1 for each leg's
  round 1); the task history may carry prior codex markers with the same
  sentinel digit from earlier labels — no collision (extraction is from the
  fresh output file; posted top-level version auto-derives) but STATE the
  mapping in the return and put the label + leg in the verdict title line.
- Round-match the smoke-arch marker by LEG (read the notes' scope line),
  inline it as "this round's own" (shape checks bind normally — unlike the
  prior-round-marker framing), and tell Codex the sibling leg's marker
  exists and is not its subject.
- Plan adherence: instruct `N/A — leg 1` (never ✗) for the other leg's
  rows; score only this leg's sections.
- The other leg's commits are absent from this leg's range by construction —
  verify with `git log <base>..<head>` and flag any unreviewed intermediate
  commits below the base in the return (scope widening is the
  orchestrator's call).
- Brief-vs-marker factual discrepancies (here: "adapter default v1" vs the
  marker's `--arm2-adapter is None` default) become an explicit adjudication
  duty for Codex (read plan + argparse), never composer-resolved.
