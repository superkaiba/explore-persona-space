---
name: shared-overleaf-clone-commit-race
description: Overleaf clones are shared by concurrent agents — verify landing by blob sha at HEAD, never by own commit rc; use git -C for non-EPS repos (guards misread cd shapes)
metadata:
  type: feedback
---

Committing in a shared secondary clone (e.g. `~/overleaf-6a59c927`): another
concurrent session's bare `git commit` can sweep YOUR staged file into ITS
commit between your `add` and your `commit` — your own pathspec commit then
lands nothing (or no-ops) while the content still reaches HEAD under a
foreign commit message. (2026-08-24 plot3 round: staged
`figures/paper/c3_persona_direction_spectrum.pdf` landed inside the C1
session's `abc1a1b` "densify" commit; own commit rc was useless as a signal.)

**Why:** the clone's index is shared exactly like the EPS repo root (#1894),
but Overleaf clones have no pathspec-commit convention, so sibling agents run
bare commits there routinely.

**How to apply:** (1) verify landing by CONTENT — `git -C <clone> show
HEAD:<path> | sha256sum` vs your local file — never by commit rc or `git log
-1` message; report the carrying SHA even when foreign. (2) Run non-EPS-repo
git via `git -C <abs clone path> …`, never `cd <clone> && git commit …`: the
EPS PreToolUse guards (`guard_root_code_commit.sh`, scope-diag `cd_nonroot`)
evaluate any Bash containing `git commit` against the EPS staged index and
block the cd-into-another-repo shape; `git -C` passes. (3) No pipes on
commit/push anywhere — `guard_piped_git_push.sh` blocks `git commit … | tail`
even in foreign clones; redirect to a file and check rc.
