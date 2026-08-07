---
title: 'sync_repo_root: fail loud when a pull --autostash is left unrestored (silently
  reverts concurrent sessions'' uncommitted work)'
kind: infra
tags:
- workflow-fix
- trigger-dense
created_at: '2026-08-07T18:19:32Z'
has_clean_result: false
workflow: v1
---
## Goal

Make `sync_repo_root.py`'s `pull --rebase=merges --autostash` recovery fail LOUD when its autostash is not restored, so a concurrent session's uncommitted work at the shared repo root can never be silently reverted into an orphaned stash.

## Problem

`git pull --rebase=merges --autostash` at the shared repo root stashes the working tree, rebases, and is supposed to pop the stash back. When the pop does not happen, the working tree is left at HEAD, the work survives only as an unlabelled `autostash` entry in `git stash list`, and NOTHING tells the session that owned those edits. The next thing that session does — re-render a figure, re-run a script — silently uses the reverted (HEAD) version.

Observed 2026-08-07 during a #2054 user-chat inline round:

- reflog `HEAD@{11:02:00} pull --rebase=merges --autostash origin main: Fast-forward`
- an uncommitted rewrite of `scripts/issue2054_framing_character_transfer_figs.py` reverted to its 08:05 committed version; the deleted `framing_transfer_provenance.*` triple came back
- the subsequent re-render then ran the OLD script and regenerated the superseded figure — the failure was only caught because the output filenames were wrong
- `git stash list` currently holds FOUR unpopped bare `autostash` entries, plus two hand-labelled rescues from prior incidents ("rescued autostash from corrupt rebase state", 2026-08-07; "rescued autostash from stale .git/rebase-merge husk", Jul 2). So this has already happened repeatedly and has been hand-recovered at least twice.

The repo root is shared across ~15 concurrent sessions, all of which pull frequently, so any session holding uncommitted edits for more than a few minutes is exposed.

## Proposed fix (implementation is the session's call)

1. In `scripts/sync_repo_root.py`, after the `pull --rebase=merges --autostash`, compare `git stash list` before/after. If a new `autostash` entry survived the pull, that is an UNRESTORED autostash: fail loud — print the stash ref, the file list (`git stash show --name-only`), and the exact `git stash pop <ref>` recovery command; exit non-zero rather than reporting success.
2. Label the stash so it is attributable: pre-stash explicitly with a message naming the session/issue instead of relying on the bare `autostash` label, so an orphan can be traced to its owner.
3. Add a startup/tick check that WARNs when bare `autostash` entries are present (the four currently sitting there are unattributable — nobody knows whose work they hold).
4. Reconcile the existing four orphaned entries: inspect each, surface the diffs, and let the user decide drop-vs-restore. Do NOT drop them automatically.

## Non-goal

Not proposing to stop using `--autostash` (it is the right default) and not proposing any repo-root `reset --hard` / `restore` path. The gap is purely the SILENT-success reporting.

## Workaround in the meantime

Write the content to a scratch path, then install-and-commit atomically in ONE shell call (`cp` + `git add` + `git commit --only -- <paths>`), so no dirty window exists for a concurrent pull to swallow. This is what recovered the #2054 round.
