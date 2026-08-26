---
title: Step 5a spec-freshness sync copies a .claude doc without its workflow_lint.py
  grandfather cap, producing a deterministic false-red gate
kind: infra
tags:
- workflow-fix
created_at: '2026-08-16T15:58:05Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate v1 emitted by the #2321 round-3 implementer:
  SKILL.md synced to origin/main''s 982,587 B while the branch cap stayed at pre-#2146
  980_400 (main: 983_400), deterministic lint + Step 9c live-pin red on bytes identical
  to main'
workflow: v1
---
# Step 5a spec-freshness sync copies a `.claude/**` doc without its `workflow_lint.py` grandfather cap, producing a deterministic false-red gate

## Goal

Make the `/issue` Step 5a spec-freshness sync cap-coherent: when it copies a `.claude/skills/**` or `.claude/agents/**` doc from fetched `origin/main`, the branch must end up with a `scripts/workflow_lint.py` size-grandfather entry that admits the copied bytes — or must WARN loudly that it does not.

## The defect

Step 5a syncs workflow-surface docs from `origin/main` but treats `scripts/workflow_lint.py` as a member of the separate `lint` family. When a branch's `lint` family is dirty (a legitimate branch deliverable touching `workflow_lint.py`), the sync skips it by family transitivity — status-quo staleness, which is the correct fail-safe for CONTENT. But size grandfather caps are not ordinary content: they are a DERIVED constraint on the very docs the sync just replaced.

When main regrows a doc and ratchets its cap in the same landing, a branch that receives the doc but not the cap is left in a state that is red **on bytes identical to `origin/main`**:

- worktree no-flags `workflow_lint.py` → FAIL
- Step 9c live pins `tests/test_workflow_lint_skill_doc_size.py::test_live_tree_passes_no_fails` and `::test_live_grandfather_caps_have_sane_headroom` → deterministic FAIL

Nothing the branch authored is wrong. The gate reds on a version skew the sync itself created.

## Observed instance (#2321 round 3, 2026-08-16)

`.claude/skills/issue/SKILL.md` was synced to `origin/main`'s copy at **982,587 bytes**, while the branch's `workflow_lint.py` still carried the pre-#2146 grandfather cap of **980_400**. `origin/main`'s cap for the same file was already **983_400**.

Cost: one diagnosis plus a restore commit inside a revision round whose actual scope was a docstring amendment. Remedy applied there was a byte-exact restore of `origin/main`'s grandfather block (merge-neutral — the branch had never touched it).

**Diagnostic trap worth encoding:** a sibling session reported `workflow_lint` PASS on what looked like the same tree, because it invoked **main's** lint binary rather than the worktree's. A lint verdict is only meaningful together with which copy of `workflow_lint.py` produced it, so a PASS from a different checkout is not evidence about this branch.

## Scope to investigate

1. Whether Step 5a should splice `origin/main`'s `SKILL_DOC_SIZE_GRANDFATHER` (and the agent-spec grandfather) entries **for the synced paths only** into the branch's `workflow_lint.py`, even when the `lint` family is otherwise dirty — a path-scoped exception to family transitivity, since the entries are derived from the synced docs rather than independent content.
2. Or, if splicing is judged too invasive, whether the sync must emit a loud WARN whenever a synced doc's branch cap differs from `origin/main`'s — turning a deterministic gate red into a named, pre-diagnosed condition.
3. Whether the same skew exists for the agent-spec size ratchet and the agent-memory index budget (both read caps the sync can move independently of their data files).
4. Whether the Step 9c live pins should attribute this class as version-skew rather than NEW, so a future occurrence does not block a round while it is diagnosed.

## Non-goals

Do not raise or remove any size cap to make the symptom disappear — the ratchet is doing its job. Do not weaken Step 5a's family-atomic dirty-skip for ordinary content; the fix, if any, is a narrow path-scoped carve-out for derived cap entries.

## Provenance

Emitted as a `workflow-fix-candidate v1` by the #2321 round-3 implementer, auto-filed by the #2321 orchestrator per `.claude/rules/workflow-fix-on-bug.md`. Confidence: high (root-caused with pre/post evidence — 2 live pins FAIL pre-fix, 18/18 pass post-fix; lint FAIL(1) → PASS rc=0). Target surface: `.claude/skills/issue/SKILL.md` § Step 5a spec-freshness sync, `scripts/workflow_lint.py`.
