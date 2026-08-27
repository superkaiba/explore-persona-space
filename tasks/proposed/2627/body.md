---
title: Agent-memory index FAIL gate can be tripped by the MERGE itself, blocking a
  Step-10d landing
kind: infra
tags: []
created_at: '2026-08-27T10:47:48Z'
has_clean_result: false
parent_id: 2569
origin_prompt: 'Surfaced during /issue 2569 Step 10d: the landing merge inflated .claude/agent-memory/code-reviewer-lean/MEMORY.md
  from 17,522 (main) / 18,332 (branch) to 35,820 bytes and tripped the 24,000-byte
  FAIL gate that neither parent violated.'
workflow: v1
---
# Agent-memory index FAIL gate can be tripped by the MERGE itself, blocking a Step-10d landing

## Goal

Close the gap where a Step-10d landing merge is blocked by an agent-memory index
size FAIL that NEITHER merge parent violates. Either make the gate merge-aware,
or give the landing session a prescribed curate-in-the-merge remedy that the
`.claude/rules` surface names, so a session does not have to derive it live while
holding a 786-file merge.

## What happened (#2569, 2026-08-27)

The `/issue 2569` Step-10d landing merged `issue-2569` into a scratch worktree
detached at `origin/main`. The merge commit was refused by the pre-commit hook
`workflow-lint-agent-memory-index-size`:

```
.claude/agent-memory/code-reviewer-lean/MEMORY.md: 35820 bytes exceeds the
24000-byte agent-memory index FAIL threshold
```

Measured sizes:

| ref | bytes |
|---|---|
| merge base `ebfc5c0727` | 19,260 |
| `origin/main` side | 17,522 |
| `issue-2569` side | 18,332 |
| **merged working tree** | **35,820** |

Neither parent is anywhere near the 24,000 FAIL threshold — both are under even
the 20,000 WARN. Both sides had added ~110 DISTINCT index rows since the merge
base (main via other sessions' reviewers; the branch via this task's own Step-5
review ensemble, four commits), and git's textual merge correctly unioned them.
The union is what violates.

## Why this is a workflow-surface gap, not a task bug

1. **The failure is structural and recurring.** Every agent-memory index is an
   append-mostly list of one-line rows written by many concurrent sessions with
   no natural committer. Two branches that both ran reviewers will BOTH have
   appended rows. So any long-lived issue branch landing onto a busy main is
   exposed, and the exposure grows with branch age and fleet size. `#2015`
   already identifies this file class as the fleet's dominant standing armer for
   a different reason (the pre-commit stash race); this is a second failure mode
   on the same class.
2. **The gate is index-content-only and merge-blind.** It measures the resulting
   file. It cannot distinguish "an agent let its index grow unbounded" (the
   behaviour `#1891` set out to stop) from "two disciplined indexes were
   unioned by a merge". The first deserves a FAIL; the second is a mechanical
   consequence of landing.
3. **The remedy is not written down anywhere reachable from the landing step.**
   `.claude/skills/issue/steps/18-step-10d.md` says nothing about it, and the
   hook's own message prescribes curation ("trim each index hook to ~1 line")
   without saying who owns that during a merge, or that the merge is the
   trigger. The #2569 session had to derive the diagnosis and the fix live.
4. **The failure mode of NOT having a rule is bad.** The tempting escapes are
   `--no-verify` (forbidden) or dropping one side's rows (silent lesson loss).
   Both are worse than the gate.

## What #2569 did (the candidate remedy, already validated once)

Curated the merged index in the merge commit, preserving every row:

- verified all 218 pointed-to per-entry `.md` files exist FIRST, so the full
  lesson text is safe and only the index label is being shortened;
- re-trimmed from the MERGE-STATE text (`git show :<path>`), never from an
  already-trimmed copy — re-trimming trimmed text double-truncates hooks;
- trimmed each row to a <=95-char pointer label at a word boundary, byte-measured
  (the indexes contain multi-byte characters — em dashes, ellipses, `≠`, `⇒` —
  so a character count under-reads the gate by ~2%);
- result 35,820 -> 18,700 bytes, clear of both FAIL (24,000) and WARN (20,000),
  218 of 218 rows preserved;
- recorded the whole thing in the merge commit message.

## Acceptance criteria

1. A landing session hitting this failure finds the diagnosis and the remedy in
   the rules surface, without deriving them. Concretely: `.claude/rules/` names
   the merge-union trigger, and the Step-10d step file points at it.
2. The prescribed remedy is byte-measured, preserves every row, verifies the
   per-entry targets exist before trimming, and trims from the merge state.
3. `--no-verify` and one-side row-dropping are both explicitly named as
   forbidden escapes.
4. Decide (and record the decision either way) whether the gate itself should
   become merge-aware — e.g. skip or downgrade to WARN when both merge parents
   are under the threshold and the path is an agent-memory index, with the
   landing commit still owing the curation. A deliberate "no, keep it strict and
   document the remedy" is an acceptable outcome; leaving it undecided is not.
5. Whatever ships is pinned by a test.

## Provenance

Surfaced by the `/issue 2569` Step-10d landing (2026-08-27). Merge commit
`bc1d4a8f19`, pushed as `0260a47a98`. Sibling rules already covering this file
class for other reasons: `.claude/rules/repo-root-uncommitted-state.md` (#2015),
`#1891` (the index size ratchet itself).
