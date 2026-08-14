---
name: stats-reuse-driver-live-probes
description: Reviewing a reuse-by-import stats driver (bootstrap contrasts, verdict relabel, seed lattice) — three ~5-line live probes settle the load-bearing claims cheaply (#2225 fu1 F5)
metadata:
  type: feedback
---

When a round's script REUSES a parent stats instrument by import (paired
bootstrap, verdict lattice, per-contrast seeds), do not settle the three
load-bearing claims by reading alone — each is a ~5-line live probe run once
from the worktree (`sys.path.insert(0, "scripts")`, import both modules):

1. **Shared-idx claim** ("frozen + inherited CIs share one idx stream"): call
   the frozen helper (`point, lo, hi, draws = pa.paired_bootstrap_ci(d, B, seed)`),
   re-derive `idx = np.random.default_rng(seed).integers(0, n, (B, n))`, assert
   `np.array_equal(draws, d[idx].mean(axis=1))`. Any drift in rng constructor,
   call order, or size tuple breaks equality — reading two call sites cannot
   certify call-for-call rng identity, the probe can.
2. **Relabel-is-pure-rename** (fu labels over a parent partition): call the
   wrapper on one exemplar per class (CI wholly below 0 / wholly above / tie
   both signs) and assert the mapped labels; also assert
   `set(RELABEL_DICT) == {the parent's three return strings}` so an unknown
   parent label KeyErrors instead of silently passing through.
3. **Seed-lattice collision-freedom**: enumerate ALL realized (kind, config,
   dataset) triples from the actual registries (not the docstring arithmetic)
   and assert injectivity into seeds. The arithmetic argument (offsets <
   strides) is checkable by eye but the registries drift; the enumeration is
   authoritative and costs milliseconds.

**Why:** #2225 fu1 F5 review (commit f2eaa02f535a) — all three passed live in
one Bash call; the alternative was trusting prose claims in the output
metadata ("share one idx stream", "pure relabeling") that a one-token rng
drift would silently falsify.

**How to apply:** any review where the diff's statistics defer to an imported
parent and the round's verdicts hinge on stream-identity / partition-identity /
seed-uniqueness claims. Run probes from the worktree so both modules resolve at
the round's own commit. Sibling: [[fails-pre-fix-probe-parent-commit]] (probe
against the parent commit's extracted body — same probe-not-prose principle).
