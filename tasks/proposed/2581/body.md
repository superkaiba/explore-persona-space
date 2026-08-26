---
title: Session-namespace commit-message temp files so git commit -F cannot consume
  a concurrent session's message
kind: infra
tags: []
created_at: '2026-08-25T17:32:40Z'
has_clean_result: false
workflow: v1
---
## Goal

Give repo-root commit-message temp files a session-namespaced path convention,
so a `git commit -F <file>` cannot silently consume a concurrent session's
commit message.

## The gap

`guard_root_code_commit.sh` and `guard_piped_git_push.sh` both steer agents to
`git commit -F <file>` (the sanctioned form when message prose would trip the
git-verb guard, per CLAUDE.md § Task Workflow API and #1722/#1756). Neither the
guards' remediation text nor CLAUDE.md states a naming convention for that file,
so agents reach for obvious un-namespaced paths like `/tmp/fig_commit_msg.txt`
or `/tmp/commit_msg.txt`. `/tmp` is shared across ~15 concurrent fleet sessions.

`-F` on a foreign-but-existing file CANNOT fail loud: git reads whatever is
there. The result is a commit whose content is correct and whose message belongs
to a different task — undetectable from the commit's own exit code, and
unfixable after the push (a shared-root history rewrite needs a force-push,
banned by `.claude/rules/auto-continuation.md` STATE-TO-blocked criterion 2).

Two mechanisms make the collision easy to hit rather than exotic:

1. A PreToolUse guard block kills the ENTIRE compound, so a
   `cat > /tmp/msg.txt <<EOF ... && git commit -F /tmp/msg.txt` whose commit arm
   is blocked never writes the file. The retry then finds a *stale or foreign*
   file at that path and succeeds against it.
2. Guard-blocked commits are COMMON on this surface (uncertified payload,
   foreign staged index, piped-push), so the write-then-blocked-then-retry
   sequence is the normal path, not an edge case.

## Precedent for the fix shape

The inline-payload path already has exactly this convention: #1948 made
`inline_lint_gate.py` REFUSE the bare issue-keyed
`/tmp/issue-<N>-inline-payload.txt` and require a `<round-slug>` making it
round-unique, for the identical reason (concurrent same-issue rounds clobber a
shared path). The commit-message path has no equivalent.

## Proposed change

1. State a convention in CLAUDE.md § Concurrent repo-root committers and in both
   guards' remediation text: commit-message files go to
   `/tmp/eps-<issue>-<session-slug>-<round-slug>-commit-msg.txt`, never a bare
   `/tmp/<something>_msg.txt`.
2. Consider a guard arm that WARNs (not blocks) when `-F <path>` names a file
   whose mtime is older than the current turn or whose path lacks a
   session-unique component — the cheap detection is "the file existed before
   this session started".
3. Consider having the guards' remediation text emit a ready-to-paste
   namespaced path rather than a generic `<msgfile>` placeholder, since agents
   copy that text literally.

## Evidence

`origin/main` commit `7cb8f68a2a` — subject `task #1336: full forward
stage-transfer lattice figures (all 10 pairs)`, content is #2054's Results 2
Plot 2 figure round (7 files, verified byte-identical to the intended payload).
Durable record: `epm:progress` note on #2054, 2026-08-25.

Not hypothetical and not recoverable after the fact — which is what makes the
convention worth pinning rather than leaving to agent judgment.
