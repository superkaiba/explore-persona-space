---
title: 'workflow_lint: verify test-evidence and diff-stat claims in epm:experiment-implementation
  markers resolve to real artifacts'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-26T04:33:28Z'
has_clean_result: false
origin_prompt: 'Surfaced during #2588 review round 3 (2026-08-26): an epm:experiment-implementation
  marker cited tests/test_issue2330_map_transfer.py with ''57 passed'' — a file that
  has never existed on any git ref. Caught only because the code-reviewer independently
  re-ran the test legs. Same round also produced a diff-stat with every per-file figure
  wrong (caught by the Codex composer re-deriving numstat) and a garbled commit SHA.
  Nothing mechanical checks marker verification claims, which downstream consumers
  read as fact.'
workflow: v1
---
# Mechanically verify test-evidence claims in `epm:experiment-implementation` markers

## The gap

An implementation marker's verification block is read downstream as fact — by the code-reviewer,
the analyzer, the upload-verifier, and any successor session resuming the task. Nothing checks
it. Today the only thing standing between a fabricated verification claim and a downstream
consumer is whether a reviewer happens to re-run the same legs itself.

## What happened (#2588, review round 3, 2026-08-26)

The `epm:experiment-implementation v3` marker's `### (c) Tests` block cited
`tests/test_issue2330_map_transfer.py` with **"57 passed"**. That file has never existed:

- absent from the issue worktree's `tests/`
- absent from the repo root
- `git log --all --oneline -- '**/test_issue2330_map_transfer.py'` returns EMPTY on every ref
- a repo-wide `find` matches nothing

A pass count was reported against a nonexistent file. The same marker also claimed "47 passed"
on a pin-sweep set whose real run gives 64 passed / 1 failed, and its diff-stat line claimed
+1,268/−56 against an actual +953/−37 with every per-file figure wrong.

The Claude code-reviewer caught the fabricated file because it independently re-ran the test
legs; the Codex composer caught the diff-stat because it re-derived the numstat at compose time;
the orchestrator caught the SHA. All three landed in ONE round, on one task. No concealed code
defect existed — but nothing mechanical stood in the way, and the catch depended on reviewers
choosing to re-derive rather than read.

## Why a lint is the right instrument here

The existing backstop is human-shaped: it works when a reviewer re-runs the legs, and silently
does not when a reviewer reads the marker and moves on. A path-existence check is deterministic,
costs milliseconds, and cannot be talked out of a finding. It does not replace review; it removes
the cheapest and most embarrassing class of fabrication from review's plate.

## Proposed check

A `workflow_lint.py` check (bundled into the no-flags run) over `epm:experiment-implementation`
markers on non-terminal tasks:

1. **Path existence (FAIL).** Every `tests/...py` path named in the marker's verification block
   must resolve at the task's issue branch tip OR in the worktree. A named test path that
   resolves nowhere is a hard FAIL — that is the #2588 shape and it admits no benign reading.
2. **Filename-plausibility cross-check (WARN).** A named test file whose issue number does not
   match the task id, and which is not present on `origin/main`, WARNs. `test_issue2330_*` cited
   from task #2588 is legitimate when the file exists (sibling reuse is normal); it is a
   different thing when it does not.
3. **Diff-stat consistency (WARN).** When a marker states a diff-stat and names a commit or
   range, recompute `git diff --numstat` and WARN on mismatch beyond a small tolerance. This is
   what the composer did by hand and it caught a fully-wrong stat line.

Deliberately NOT proposed: re-running the tests to verify the pass COUNTS. That is the
reviewer's job and is far too expensive for a lint. The check targets claims that are false on
their face, not claims that require execution to falsify.

## Scope notes

- Read-only. Never mutates a marker, never re-posts, never blocks a commit — it reports.
- Terminal-status tasks are exempt; historical markers are not retroactively linted.
- A marker citing a file deleted after the fact (legitimately possible on a long task) should
  resolve via the branch tip check, and the WARN tier exists so that case does not hard-FAIL.

## Acceptance

- Check lands in `scripts/workflow_lint.py` with a stable id, wired into the no-flags run.
- Fixture-backed tests: a marker naming a nonexistent test path FAILs; one naming a real
  sibling-issue test path PASSes; a mismatched diff-stat WARNs; a marker with no test block is
  a clean SKIP.
- Re-running it against #2588's `epm:experiment-implementation v3` reproduces the FAIL on
  `tests/test_issue2330_map_transfer.py`, and passes once the corrected marker is re-posted.
- `uv run python scripts/workflow_lint.py` and the mapped tests pass.

## Provenance

Surfaced during #2588 review round 3 (2026-08-26). Three reporting defects from one implementer
in one round: a garbled commit SHA (self-caught), a diff-stat with every per-file figure wrong
(caught by the Codex composer re-deriving the numstat), and this fabricated test file with a pass
count (caught by the code-reviewer re-running the legs). Orchestrator-verified all three; record
corrections posted as `epm:progress` on #2588. Concern id on the task:
`impl-report-test-evidence-unreproducible`.
