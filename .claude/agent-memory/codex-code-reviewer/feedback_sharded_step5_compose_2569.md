---
name: sharded-step5-compose-2569
description: "#2569 r1: 3-way Codex shard compose — pathspec-scoped diff ladder, brief facts as settled, custom VERDICT enum honored, plan by verified-identical worktree path"
metadata:
  type: feedback
---

On a SHARDED Step 5 round (one Codex twin per file-set shard, #2569 r1: 1.03 MB
diff split 3 ways), the brief IS the shard's rubric and extraction contract.
Compose shape that worked:

- **Scope pin:** every diff command in the prompt carries the shard's explicit
  pathspec (`git diff origin/main...HEAD -- <five paths>`); state that findings
  outside the shard are discarded and name what the other shards cover.
- **Custom verdict enum honored:** the brief ordered `VERDICT: PASS|FAIL` +
  BLOCKER/MAJOR/MINOR rows with file:line + concrete failure scenario +
  `CONCERN:: ` rows — NOT the standard `epm:code-review-codex` marker envelope.
  Follow the brief (it is the orchestrator's extraction contract); the brief
  also ordered "return ONLY the prompt path", overriding the Step-4 structured
  return.
- **Brief-supplied round facts inlined as SETTLED** (ρ(A), τ_kernel, smoke
  rc=0 phase list, lint/test-union results, pre-existing reds enumerated for
  provenance discipline) — Codex told not to re-derive or contradict.
- **Demonstrated-defect-class duty:** the round's own fixed bug (manifest key
  `i` read as `ci`, 3 fixtures encoding the wrong key, fix `57808a6434`)
  becomes an audit duty: check every cross-artifact key read against the
  PRODUCING code, treat fixtures that pass under a plausible wrong key as
  findings, verify the fix's regression test pins the real key.
- **No-re-raise ledger list** pasted verbatim with the note that new concerns
  with distinct root causes stay fair game.
- **Plan by path:** frozen worktree copy (`tasks/approved/2569/...`) verified
  byte-identical to canonical `tasks/running/2569/plans/plan.md` (144 KB) ⇒
  path reference, no inlining. Impl marker still fetched from main + inlined
  (25.8 KB) per the standing Step 2-pre duty — the brief was silent on it, and
  silence is not a by-path order.

**Why:** shard consistency — three composers run in parallel on one round; a
shard that re-derives settled facts or uses the standard marker envelope
breaks the orchestrator's mechanical extraction.

**How to apply:** any brief naming "shard k of N" with a scope list and a
custom verdict block. See also [[brief-pinned-sentinel-and-verdict-enum]],
[[whole-round-unsplit-compose]] (the opposite case: split NOT honored on
#2074-style rounds — there the brief ordered whole-round).
