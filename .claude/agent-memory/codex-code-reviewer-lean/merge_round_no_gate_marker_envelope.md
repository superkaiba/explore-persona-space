---
name: merge-round-no-gate-marker-envelope
description: Step 10d reconciliation merge round with NO round-matched impl/results marker — inline the [divergence-probe] progress markers + an orchestrator-authorship provenance block as the impl-marker envelope (#2477 r3)
metadata:
  type: feedback
---

On a Step 10d divergence-reconciliation merge round, probe the task's
canonical events.jsonl for a round-matched marker before composing (the
sibling's #2253 memory says the gate's dispatch posted an `epm:results`
there). When the probe finds NONE (#2477 r3: the merge was
orchestrator-authored; the only round record was two `[divergence-probe]`
`epm:progress` notes), fill the `---BEGIN/END IMPLEMENTATION MARKER BODY---`
envelope with (a) a provenance block stating the merge is
orchestrator-authored with no implementer marker owed, (b) the verbatim
divergence-probe notes, and (c) an explicit instruction that `marker-shape`
/ `smoke-run-missing` are N/A-by-round-type. This satisfies the spec's
Step 3 envelope validation AND forestalls a mechanical-contract false FAIL.

**Why:** the Step 3 validator hard-requires a non-empty marker envelope, and
Codex FAILs `marker-shape` on an absent marker unless told the round type
owes none; #2477 r3 had no marker to inline.

**How to apply:** any brief naming a reconciliation/merge commit where the
events.jsonl probe (`grep '"kind": "epm:results"'` + version/note read)
returns no round-matched marker. Pattern base:
`.claude/agent-memory/codex-code-reviewer/feedback_merge_reconciliation_review_compose.md`
(#2253 r3 — misleading-range ban first, zero-hand-edit as check 1,
parent-relative scoped diffs, adapted gate-scope note). Brief-pinned
deviations (binary PASS|FAIL vocabulary, non-default prompt path) follow
the brief — it is the extraction contract — and get flagged in the return.
