---
name: Claude PASSes parallel-dispatcher when smokes exit before CUDA init
description: Claude code-reviewer PASSes round-N "parallel GPU dispatcher" verdicts using --help / --print-pending-pairs / DRY_RUN smokes that short-circuit before subprocess spawn or model load; misses CVD-clobber bugs that need actual CUDA-init code path to surface
type: feedback
---

When the artifact under review is a parallel GPU dispatcher (one shell
script + one Python entrypoint, fan N shards across N GPUs via
`CUDA_VISIBLE_DEVICES=$i`), Claude code-reviewer happily PASSes round-N
based on a verification ladder of:

- `--help` exit
- `--print-pending-pairs` exit-before-model-load
- `DRY_RUN=1` bash short-circuit at the "would-spawn-subprocess" line
- synthetic merge tests with hand-crafted JSON

NONE of these exercise the `os.environ["CUDA_VISIBLE_DEVICES"] = ...` /
torch CUDA-init code path. So a Python-side CVD-clobber bug (`os.environ
["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)` after the shell already set
`CVD=$i`) cannot surface in those smokes. Codex catches it by reading the
Python entrypoint top-to-bottom.

**Why:** PyTorch reads `CVD` lazily at first CUDA-context creation. The
Python-side write executes at argparse time, BEFORE any `.cuda()` call,
so it wins over the shell-set value. All N shards end up CVD="<gpu_id>"
(typically "0") → pile on physical GPU 0 → OOM-or-contention. Exact
pattern already documented in CLAUDE.md memory for `train/sft.py:477` as
`+gpu_id Hydra arg for parallel launches` — but Claude does not
generalize the lesson to a NEW dispatcher script. Companion to
`feedback_claude_trusts_green_tests_over_verifier_semantics`: same
verifier-semantics gap (green tests in an orthogonal scope).

**How to apply (in adjudication):** When Codex FAILs a parallel-
dispatcher diff Critical on a Python-side `os.environ["CUDA_VISIBLE_
DEVICES"] = ...` line, ALWAYS:
1. `Read` the cited line in the Python entrypoint. Verify the write is
   unconditional (no `if "CUDA_VISIBLE_DEVICES" not in os.environ`
   guard).
2. `Read` the dispatcher shell — confirm it pre-sets `CVD="$i"` AND
   passes `--gpu-id 0` (or any fixed value) to every shard.
3. Check Claude's verification list — if it mentions only `--help` /
   `--print-pending-pairs` / `DRY_RUN` / synthetic JSON merge, those
   paths do NOT exercise CUDA init. Verification gap, not a finding.
4. Verdict FAIL.

The Critical bug is silent under all of Claude's verification probes; it
would only surface at production launch when 7 shards die OOM. The
reconciler is the last line of defense before merge.

Origin: task #488 round-3 reconcile (2026-06-05). Codex CRITICAL claim
on `scripts/i488_phase1_predictors.py:540`, Claude PASS based on
DRY_RUN smokes; reconciler verified both file:line claims, FAILed.
