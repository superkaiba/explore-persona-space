---
name: confirm_artifacts git_paths — drop a false-FAIL path without SKIPping the check
description: _check_git SKIPs only on EMPTY paths; dropping one path from a non-empty list still checks the rest — safe way to remove a false-FAIL
type: project
---

`_check_git` (artifacts.py:734-735) returns SKIP only when the declared
`git_paths` tuple is EMPTY; otherwise it FAILs on any declared path that is
untracked OR missing on disk (missing_tracked / missing_on_disk, 746-754).

**Why this matters for review:** When a plan proposes dropping ONE path from a
default `git_paths` list to remove a false-FAIL (e.g. #790 dropping
`figures/issue_<N>/` because the analyzer generates figures POST-gate, never the
workload), verify the residual list is still NON-EMPTY. If the residual is
`["eval_results/issue_{issue}/", *extra]`, the git check STILL runs and STILL
FAILs on a genuinely-missing eval_results/ — the fix removes only the
never-produced path's guaranteed false-FAIL, not the real check. If dropping the
path emptied the list, it would SKIP the whole git check (a real regression) —
that would be a REVISE. Dropping one of N>1 is safe; dropping the last is not.

**How to apply:** For any "remove path X from the artifact-completeness
declaration" fix, check the residual path count. Non-empty residual + X is
provably never produced during the run (workload writes no figures; SLURM
already treats no-figures as non-fatal; analyzer.md Step 3 commits them later) =
APPROVE. Empty residual = the check silently SKIPs = REVISE. The change
propagates to all three lanes (gcp.py:861, slurm.py:524, RunPod all delegate to
build_expected_artifacts_declaration) so confirm none of them writes X during
the run. (#790)
