---
name: tmp-snapshot-status-move-proofing
description: When the brief directs /tmp materialization of the live body (or the task sits at a status about to flip, e.g. interpreting->reviewing), snapshot body + plan to /tmp and point Codex there; keep canonical paths as provenance lines so the Step-4 path greps still pass.
metadata:
  type: feedback
---

When the orchestrator brief directs materializing the LIVE body to /tmp
(`task.py view <N> --json | jq -r '.body'` — e.g. after a humanize pass),
extend the same treatment to the PLAN: `cp $TASK_DIR/plans/plan.md
/tmp/issue-<N>-plan-v<K>.md` (dereferences the symlink) and set the
prompt's PLAN path to the /tmp snapshot.

**Why:** the `task.py find`-derived canonical path embeds the CURRENT
status folder (e.g. `tasks/interpreting/<N>/...`), and the CR gate runs
across the interpreting->reviewing transition — a status `git mv` between
compose and Codex-read kills the canonical path mid-flight (the #489/#550
unresolvable-path class, from the other side). /tmp snapshots are
status-move-proof, like the orchestrator's interpretation-note temp file.
Applied at #2617 r1 (2026-08-27).

**How to apply:** point the header's CLEAN-RESULT BODY / PLAN lines at
the /tmp snapshots, and keep the canonical `tasks/<status>/<N>/...`
paths as parenthetical provenance lines in the same block — the Step-4
`grep -qF "$BODY_PATH"/"$PLAN_PATH"` guards then pass for BOTH forms.
Also verify the /tmp body carries the v4 sentinel before composing.
Related: [[compose-recipe-lens-ref-replacements]] (the splice recipe this
rides on); a reconciled-ledger round pairs this with a fourth
`CONCERNS LEDGER TAIL` envelope (verbatim last-20 rows) so the
verify-direction Lens-14 block has quotable address-event claims without
any run-it-yourself instruction.
