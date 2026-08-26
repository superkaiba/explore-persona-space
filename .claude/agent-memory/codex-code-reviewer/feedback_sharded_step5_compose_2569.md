---
name: sharded-step5-compose-2569
description: "#2569 r1+r2: 3-way Codex shard compose — pathspec-scoped diff ladder, brief facts as settled, custom VERDICT enum honored, plan by verified-identical worktree path; r2 re-review: no fix-round impl marker => inline shard-relevant concern-addressed rows, pin range endpoint against tip drift"
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

## r2 addendum (fix-round re-review shards, #2569 r2 shard B)

- **Tip drift on a re-review:** worktree HEAD sat ONE commit past the brief's
  stated review tip (the committed shared-brief file itself). Probe
  `git log <brief-tip>..HEAD` + a pathspec `diff --name-only` over the shard
  files; when drift touches no shard file, PIN every diff command to
  `<base>..<brief-tip>` and tell Codex not to re-raise tip drift (here it was
  already a ledgered concern id).
- **Stale round-1 impl marker:** on a fix-round re-review the latest
  `epm:experiment-implementation` marker predates the fixes — do NOT inline it
  as current; state that explicitly and frame the `epm:concern-addressed`
  ledger rows as CLAIMS to verify, not facts.
- **Execution-preferring brief:** when the shared brief orders
  run-over-read verification, the composed prompt carries a sanctioned /
  forbidden execution split (shard pytest + fixture renders + schema probes
  YES; production drivers / GPU / paid API NO), worktree hygiene (never leave
  it dirty; revert-probes via `git show <base>:<path>` to /tmp), and the
  honesty rule: could-not-execute ⇒ "verified by reading", INCONCLUSIVE ≠
  clean. Recommend write-enabled dispatch (no `--no-write`) in the return.
- **Plan-by-resolver order:** honor the brief's resolve-never-hardcode
  command, but add the composer-VERIFIED absolute path as explicit fallback
  with "missing fallback ⇒ re-run the resolver, never conclude plan-absent"
  (status folders move) — avoids a data-access-blocked FAIL in a sandbox that
  cannot run task.py.

## r2 delta (re-review of fix units, 2026-08-26)

- **No fix-round impl marker:** the fix units posted only `epm:concern-addressed`
  rows + heartbeat notes; the latest `epm:experiment-implementation` on main was
  ROUND 1's report. Correct envelope: inline the SHARD-RELEVANT concern-addressed
  rows verbatim (14 rows, 3.2 KB) as "claims, NOT evidence" with provenance
  (no new marker; record = ledger rows + heartbeats), and tell Codex the round-1
  marker is prior-round context only. Extends [[missing-impl-marker-probe-checklist]]
  + the 9a-ter placeholder pattern to the sharded case.
- **Tip drift:** the shared-brief commit moved HEAD past the brief's stated
  endpoint. Pin the range to the brief's HEAD (`4a48517b13..1e4d1122aa`), verify
  the scoped files' working-tree copies are byte-identical to the endpoint
  (`git diff <endpoint>..HEAD -- <paths>` empty + clean porcelain), and tell
  Codex the drift is out of scope + already a ledger deferral.
- **Registered-wording checks get the plan lines verbatim:** when a named check
  compares implementation to plan wording (H7 demote vs §7.5), excerpt the exact
  plan lines into the prompt (with line numbers) so Codex never pages a 144 KB
  plan for two sentences; keep the resolve command + verified-identical worktree
  fallback for context reads.
- **Execution-preferring briefs:** when the shared brief orders "prefer running
  over reading", adapt honestly for Codex: name the main-checkout interpreter
  (worktree `.venv` is a stub), give small numpy recipes for the math claims,
  require per-finding evidence labels `ran|derived|read`, and keep the
  production-workload ban (bounded local tests only).
