---
name: Smoke keeps ZeRO-3 production process width
description: Narrowing a ZeRO-3 full-FT smoke to --num_processes 1 OOMs deterministically at the first optimizer step; smoke/production parity includes the RESOURCE dimension (process width), and a parent's narrow-smoke pin transfers only if the smoke host transfers too
type: feedback
---

A cloned dispatcher that narrows a ZeRO-3 full-FT smoke to `--num_processes 1`
OOMs deterministically at the FIRST optimizer step — single-process ZeRO-3
shards nothing, so 7B bf16 weights + grads + fp32 Adam moments (~86 GB) land on
one 80 GB GPU (#1315 p1_train smoke, 2026-07-15; traceback: `exp_avg_sq`
alloc fail in `stage3.py _optimizer_step`).

**Why:** smoke = production with fewer steps/cells INCLUDING the process
shape — width is a resource dimension of smoke/production parity (#397
class). #1112's `_ft_num_processes` smoke=1 pin was legitimate ONLY because
its smoke ran on a 1-GPU GCE instance; #1315's smoke runs on the 4x A100-80
ft-7b pod, so the pin must not transfer with the clone.

**How to apply:** when cloning a dispatcher, audit every `if cfg.smoke:`
branch that composes LAUNCH width / CUDA_VISIBLE_DEVICES — keep production
width unless the smoke HOST genuinely differs; pin it with an
arg-composition regression test asserting `--num_processes <N>` + CVD in
BOTH modes (worked example: tests/test_issue1315_dispatch.py
test_ft_launch_width_smoke_invariant).

RECURRED same-day in #1333 (crash-fix r4, 2026-07-15): the #1333 dispatcher
was written BEFORE this memory landed and carried the identical
`if cfg.smoke: return 1` in `_ft_num_processes`; its pod smoke died rc=1
~50 s into p2_train on the SAME 4x A100-80 ft-7b pod. The r4 fix mirrors
#1315 (smoke-invariant width + under-provision guard +
tests/test_issue1333_dispatch.py::test_ft_launch_width_smoke_invariant) and
adds `_run_subprocess` inner-log-tail-on-failure so the next subprocess
crash's traceback lands in the crash-persisted workload.log (the inner
`ft_mk4.log` was outside the GCE trap globs and died with the instance).
Sweep duty: any OTHER in-flight dispatcher cloned from the #1112 family
needs the same audit before its first pod smoke.

## Merged sibling index rows (#1891 curation, 2026-07-30)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the ~25 KB loader truncation limit (task #1891). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [Smoke keeps ZeRO-3 production width](feedback_smoke_ft_zero3_width_parity.md) — narrowing a ZeRO-3 full-FT smoke to 1 process OOMs at the first optimizer step (~86 GB unsharded on 80 GB); audit every `if cfg.smoke:` width branch when cloning a dispatcher (#1315)
- [Per-arm-class smoke + source-filtered panel](feedback_per_arm_class_smoke_and_panel_disjointness.md) — a new source-context class crashes the #527/#538 panel-disjointness assert at ModelOrganism sites the smoke never reached; thread fu3w.panel_name_for everywhere + smoke one run per ARM CLASS (#1090 fu5)
- [--smoke ternary skips the production branch](feedback_smoke_ternary_skips_production_branch.md) — `A if args.smoke else MODULE.CONST` leaves the production branch unexecuted by the smoke; resolve production-only constants at import time (#825 contrast crash)
- [Smoke per class×regime](feedback_smoke_class_regime_coverage.md) — multi-class dispatchers smoke ≥1 cell per realized behavior-class × regime (#1586)

## Sibling shape: smoke-slice closure must cover CONSUMER-side dereferences (#2389 crash-fix r1, 2026-08-23)

A smoke slice extension that closes dependencies for a gate's PRIMARY
selection (the 12 spot pairs + their donors, #2389 B4) is still open under
any SECONDARY selection the gate makes from the whole filtered set: the
injection gate drew a second batch row from all smoke-filtered pairs and
composed its payload for the spot's arm — dereferencing the SECOND row's
OWN donor (`pairs_by_id[donor_map[other.pair_id]]`), outside the closure →
`KeyError` in bank worker 0, chain rc=1, production correctly gated.
Fix pattern (preferred over widening the closure): filter the secondary
candidate pool to payload-RESOLVABLE items for the current arm (donor pair
present + donor B captured), applied BEFORE any helper that dereferences
the same donor (`pe_excluded_reason`), with a kept/total log line as the
fix-engaged signal; production-invariant because the full set resolves
every donor. Pin with a repro test (unresolvable candidate excluded) + a
full-closure invariance test (`scripts/issue2389_run.py::_gate_second_row_pool`,
`tests/test_issue2389_run.py`). Audit rule: when writing a slice-closure
helper, enumerate EVERY selection the downstream consumer makes over the
filtered set — not just the one the closure was written for.
