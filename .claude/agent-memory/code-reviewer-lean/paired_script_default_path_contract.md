---
name: paired-script-default-path-contract
description: Producer/consumer script pairs in one round — diff BOTH mode defaults (production + smoke) against each other and the plan's registered artifact path; live-probe smoke guards with the absolute-path form
metadata:
  type: feedback
---

When a round ships a producer script and a consumer script meant to run as a
bare `producer && consumer` chain (the plan's Workload-commands line), diff the
consumer's default input dir against the producer's default output dir in BOTH
modes before crediting wiring: (a) production defaults, (b) the `--smoke`
rebind targets. #2564 r1 g6: figures defaulted to `eval_results/issue_2564/pe`
and smoke `SMOKE_ROOT/out` while the analysis producer wrote
`eval_results/issue_2564` and `SMOKE_ROOT/eval_results/issue_2564` — the
plan-registered command chain and the plan-required smoke figure render both
died `SystemExit: missing ...` on defaults (loud, but Major). The plan's own
artifact registration (`outputs:` rows) is the tiebreak ground truth.

**Why:** no caller wires the flags when the plan registers a bare chain, so the
defaults ARE the contract; graceful `_get`-style key handling hides nothing
here because the mismatch is at the FILE level, caught only by comparing the
two scripts' constants.

**How to apply:** grep both scripts for `DEFAULT_*`/`out_dir` defaults +
`--smoke` rebind branches; assert equality (recommend a 2-line pytest sharing
one constant). Companion probe: a smoke committed-tree guard of the form
`str(out_dir).startswith("figures")` passes absolute paths — run the guard
live with `--out-dir "$(pwd)/figures"` and check the refusal fires (it did not;
`resolve()`-based compare is the fix). Related: [[smoke_shard_namespace_only_done_files]].
