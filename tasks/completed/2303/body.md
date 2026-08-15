---
title: '/issue Step 5a spec-freshness sync: sync synced-spec data dependencies + fail
  loud on a failed sync commit'
kind: infra
tags: []
created_at: '2026-08-14T21:53:21Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2293 Step 10d lint-gate remediation: the Step 5a family
  sync copied scripts/workflow_lint.py from origin/main without .claude/config/agent_spec_size_caps.txt,
  and reported success after its commit had failed.'
workflow: v1
---
# `/issue` Step 5a spec-freshness sync: it moves synced code without its data dependencies, and reports success on a failed commit

## Goal

Close two independent defects in `.claude/skills/issue/SKILL.md` Step 5a
(family-atomic spec-freshness sync + the #1972 sibling-issue file freshness arm), both
observed in a single run on #2293:

1. The sync copies a family SPEC (`scripts/workflow_lint.py`) from `origin/main` into the
   worktree **without** the data file that spec reads at import time, leaving a synced
   module that raises `FileNotFoundError` on every invocation in the worktree.
2. The family arm prints its `[step5a] synced from origin/main:` success line and diffstat
   **unconditionally after staging**, never checking the `git commit` return code — so a
   failed sync commit is reported as a completed sync and the block exits 0 with the whole
   synced set staged-but-uncommitted.

## Defect 1 — synced code, unsynced data dependency

`scripts/workflow_lint.py` is in the Step 5a family `SPECS` set, so the sync stages
`origin/main`'s copy into the issue worktree. Since #1718 (`dc95b66efb`, "move agent-spec
size caps to a line-mergeable data file") that module resolves its grandfather caps at
MODULE IMPORT time:

```python
AGENT_SPEC_SIZE_GRANDFATHER: dict[str, int] = _load_agent_spec_caps()   # line ~14495
...
text = p.read_text(encoding="utf-8")  # raises FileNotFoundError loud   # line ~14465
```

reading `.claude/config/agent_spec_size_caps.txt`. That path is a main-side file added
after this branch's fork point, is NOT in the family `SPECS` set, and is not a
`scripts/issue<M>_*.py` sibling — so nothing syncs it. Result in the #2293 worktree:

```
FileNotFoundError: [Errno 2] No such file or directory:
  '.../worktrees/issue-2293/.claude/config/agent_spec_size_caps.txt'
```

on three separate pre-commit hooks (`workflow-lint-upload-or-true`,
`workflow-lint-agent-spec-size`, `workflow-lint-agent-memory-index-size`) — every hook
that shells the synced `workflow_lint.py`. Any commit in that worktree staging a matching
file class fails until the data file is synced by hand.

**Why the existing guards miss it.** The #2208 import-satisfiability probe reverts a
synced sibling TEST that fails `pytest --collect-only` because it imports a `src/` symbol
newer than the fork point — a PYTHON-IMPORT skew on the sibling arm. This is a
DATA-dependency skew on the FAMILY arm: the failure is a runtime `FileNotFoundError`
inside a script the probe never runs, on a file class the probe never considers. The
`_load_agent_spec_caps` helper is deliberately fail-loud (its own comment says so), which
is correct — the bug is that the sync leaves the file absent, not that the reader raises.

**Proposed fix.** Give the family sync a declared per-spec data-dependency set — the
minimum viable form is an explicit mapping alongside `FAMILY_OF` / `SPECS`, e.g.
`scripts/workflow_lint.py -> [.claude/config/agent_spec_size_caps.txt]`, synced
pair-atomically with its spec (present on main and absent locally ⇒ sync; conflicting
local modification ⇒ skip the whole pair, matching the existing per-file dirt check). A
stronger form worth considering: after the family sync, run one import-satisfiability
probe over the synced SCRIPTS themselves (`uv run python -c 'import <module>'` or the
script's `--help`) mirroring what #2208 does for sibling tests, and revert the pair on
failure. The declarative map is cheaper and deterministic; the probe generalizes to data
dependencies nobody remembered to declare. Either is acceptable; do not silence the
reader.

## Defect 2 — the family arm reports success on a failed commit

In the same run the three crashed hooks aborted the family arm's `git commit`, and the
block nevertheless printed:

```
[step5a] synced from origin/main:
 ... 27 files changed, 861 insertions(+), 21 deletions(-)
```

then continued to the sibling arm (which committed its own 7 paths by explicit pathspec,
as the concurrency rule requires) and exited **rc=0**. The 27 family paths were left
STAGED and UNCOMMITTED. Three consequences, in ascending severity:

- The operator reads a success line for a sync that did not land.
- The next gate runs against a tree whose sync was never certified — the exact failure
  class Step 5a exists to prevent.
- Tracked files sit staged-but-uncommitted, which `.claude/rules/repo-root-uncommitted-state.md`
  identifies as unsafe under concurrency (the worktree is less exposed than the shared
  root, but the staged set still survives into the next round and confuses the arm's own
  per-file dirt check, which treats a staged file as dirty and SKIPS it — so a naive
  re-run of the block silently syncs nothing).

**Proposed fix.** Check the commit rc. On non-zero: print a `FATAL`-shaped line naming the
failed paths, leave the index as-is for inspection, and exit non-zero so the caller cannot
mistake it for a completed sync. The success line moves AFTER the rc check and reports
the committed sha, not the staged diffstat.

## Acceptance

1. Step 5a's family sync carries the data dependencies of every synced spec (declared map
   and/or post-sync satisfiability probe), pair-atomically with the same dirt-skip
   semantics as the existing per-file check.
2. A family-sync commit failure exits non-zero with a FATAL-shaped diagnostic naming the
   failed paths; the `[step5a] synced from origin/main:` success line is emitted only
   after a verified commit and names the resulting sha.
3. A pin test reproduces both shapes: (a) a worktree missing a synced spec's declared data
   file ends with the data file present and the pair committed (or the pair reverted, if
   the chosen design skips); (b) a family-sync commit forced to fail yields a non-zero
   exit and no success line. The `tests/test_issue_skill_*` family is the natural home.
4. No change to the sibling arm's #2208 import-satisfiability probe, the #1972 selection
   logic, the on-main skip, or the explicit-pathspec commit discipline.

## Provenance

Surfaced during #2293's Step 10d pre-push lint gate remediation (task #2293
`epm:progress`, 2026-08-14). #2293's subject matter is unrelated — it fixes the pristine
oracle's base sha in `scripts/step9c_baseline.py`; this task is the Step 5a gap its
worktree sync exposed. Distinct fingerprint from #2297 (Step 9c launcher argv-newline
split), which is a different Step, a different mechanism, and a different failure mode.
