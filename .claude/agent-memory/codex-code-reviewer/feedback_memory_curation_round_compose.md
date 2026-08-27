---
name: memory-curation-round-compose
description: Compose recipe for content-only agent-memory MEMORY.md curation rounds (#1891 size-cap dedupe/trim) — composer runs the row/target recount + stray-adoption cmp, frames displaced-detail + longest-variant duties, fences landing discipline out of scope (#2620 r1)
metadata:
  type: feedback
---

Compose recipe for a `kind: infra` round whose diff is CONTENT-ONLY curation
of `.claude/agent-memory/*/MEMORY.md` indexes (the #1891 size-cap dedupe/trim
shape; first used #2620 r1, 2026-08-27):

1. **Composer recount as settled facts (one Python probe):** per file, at
   merge-base vs HEAD — bytes, `^- \[` row count, UNIQUE link-target set
   (parse `^- [t](target)`), duplicate rows at HEAD, dangling targets at HEAD
   (os.path.exists per target). Hand the table as facts + the re-derive
   commands; the target-set EQUALITY base==HEAD is the "zero lessons dropped"
   acceptance read. On #2620 the probe corroborated the marker exactly
   (263→111/111-uniq, 77→59, 114→109; 0 dupes, 0 dangling).
2. **Out-of-sandbox stray adoption gets a composer `cmp` attest** (the #2584
   adopt-then-fix pattern): a marker claiming the ADDED per-entry file is a
   byte-identical copy of an untracked repo-root stray is unverifiable from
   the worktree sandbox — run the cmp + sha256 yourself, attest, and forbid
   an unverifiability finding. Also probe the index row PRE-EXISTS at base
   (dangling-pointer repair, not a new lesson row).
3. **The two hard content duties:** (a) LONGEST-VARIANT dedupe — for
   materially-divergent duplicate pairs, kept row = newer/longer AND the
   dropped variant's unique content present in the per-entry file; (b)
   DISPLACED-DETAIL — per trimmed >250 B hook, the dropped detail must be in
   the per-entry file (already there, or the round's +1 appended line);
   in NEITHER place = Major `substantive` ("detail moves, never deletes").
4. **§8 fence check from name-status:** no .gitattributes / lint-threshold /
   loader / hook edits, no agent-memory dir beyond the named ones. Plus:
   commit message must name the union-merge gap (acceptance item read via
   `git log -1 --format=%B` — quote the composer probe result).
5. **Landing discipline (§3.4 scratch-worktree reconcile) is OUT OF SCOPE** —
   orchestrator Step 10d; fence it so the twin doesn't FAIL the round for
   not landing. Only check the round performed no union merge itself.
6. **Deviation-accounting adjudication:** plan projected "+0–2 per-entry
   files", realized 4 (3 sanctioned appends + 1 adoption) while the marker
   claimed "+1 beyond" — compose as a named D1 adjudication (justification
   AND accounting-fairness halves), pre-routed at most concern-row /
   Minor report-accuracy, never a FAIL steer.
7. **Execution ban names workflow_lint no-flags explicitly** (observed 540 s
   timeout in the marker) alongside pre-commit/pytest/task.py.

**Why:** these rounds have no code, so the whole review is content
preservation — without the composer probes the twin either trusts the
marker's counts or burns its budget re-deriving what one probe settles; and
the stray-adoption byte-identity is structurally unverifiable from the
sandbox (false-finding bait).
**How to apply:** any diff whose files are all `.claude/agent-memory/**`
index/per-entry markdown. Related: [[infra-wf-fix-lint-gate-compose]] (the
wf-fix floor TAG-first probe + N/A-by-type block apply unchanged),
[[worktree-task-folder-status-can-be-stale-in-either-direction]] (#2620's
worktree carried tasks/planning/2620 vs the brief's tasks/running path;
identity diff passed, brief-ordered inline kept with truthful wording).
