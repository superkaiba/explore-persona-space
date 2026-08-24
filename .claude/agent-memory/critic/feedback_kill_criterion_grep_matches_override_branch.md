---
name: kill-criterion-grep-matches-override-branch
description: Infra-plan kill criteria phrased as bare code-shape greps ("abandon if a path returning X exists") self-trigger on deliberate kill-switch/legacy-override branches — replay the grep before approving (#2212)
metadata:
  type: feedback
---

A plan whose kill criterion is a mechanical code probe ("Falsifier: a code
path where `<fn>` returns `<live-state>`; grep for it; if such a path
exists, ABANDON") must be replayed against the actual file at review time.
Deliberate operator kill switches (`EPS_SLURM_LIVE_TREE_RSYNC=1`-style
env-gated legacy branches) ARE such code paths, so the criterion fires on
day one and — when the abandon route is on the plan's must-re-plan list —
deterministically bounces a healthy plan back to the planner.

**Why:** #2212 plan v1: §7 criterion 1 said "abandon change A if
`_resolve_rsync_source` can return `self._src_root`"; slurm.py L3211/L3214
return exactly that inside the loud-warned `EPS_SLURM_LIVE_TREE_RSYNC=1`
legacy branch, and §10 made the abandon a must-re-plan. Sibling class to
[[unsatisfiable-gate-respec-review]] (#488 self-defeating plans).

**How to apply:** on any infra plan with a grep-shaped kill falsifier,
RUN the grep. If it hits an env-gated/deliberate-override branch, demand
(a) the falsifier scoped to paths reachable WITHOUT the named override,
and (b) a disposition for the override itself when the plan's change
materially raises the override's blast radius (e.g. widening an rsync
include set makes a live-tree override sweep untracked caches that the
new excludes only partially cover — extend the override's warning text or
the doc edit by one sentence).
