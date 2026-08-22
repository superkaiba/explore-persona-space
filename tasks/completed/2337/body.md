---
title: Scope preflight's vLLM/transformers skew check to workloads that actually import
  vLLM
kind: infra
tags: []
created_at: '2026-08-17T06:42:06Z'
has_clean_result: false
parent_id: 2329
origin_prompt: 'Hit on #2329 (Qwen3.5-9B, transformers 5.15.0 model-forced): preflight
  hard-FAILs on a vLLM/transformers skew check even though the rig never imports vLLM,
  and its printed remedy (pin transformers<5.0) would make the task impossible since
  4.57.6 has no qwen3_5 support.'
workflow: v1
---
# Scope preflight's vLLM/transformers skew check to workloads that actually import vLLM

## The gap

`explore_persona_space/orchestrate/preflight` — the MANDATORY pre-launch gate —
hard-FAILs (`report.ok=False`) on a vLLM/transformers version skew even when the
workload never touches vLLM, and the remedy it prints is destructive for any task
that REQUIRES transformers >= 5.

Verbatim, hit on task #2329 (Qwen3.5-9B) at 2026-08-17:

```
preflight ERROR: vLLM/transformers version skew: vllm==0.11.0 +
transformers==5.15.0. vLLM 0.11.x calls tokenizer.all_special_tokens_extended
which transformers >=5 removed. Every LLM(...) instantiation will crash.
Fix: pin `transformers>=4.46,<5.0` in pyproject.toml and re-run `uv sync --locked`.
Pre-flight Check: FAIL
```

## Why the check is right in general but wrong here

The underlying fact is TRUE: vLLM 0.11.x + transformers >= 5 does break every
`LLM(...)` instantiation. The defect is the check's SCOPE — it keys on the two
installed versions rather than on whether the workload actually instantiates vLLM.

On #2329 the rig uses hooked HF `generate()` (`generate_batch`) end-to-end, by
design, because it applies full-state patches at prefill. Evidence it never touches
vLLM:

- static: `grep -rniE "vllm|LLM\(|SamplingParams"` over `scripts/issue2329_run.py`,
  `scripts/issue2329_dispatch.sh`, `scripts/issue2329_judge.py` returns NOTHING;
- empirical: a full 8-phase smoke slice (incl. two generation phases, `anchors`
  and `grid`) completed rc=0 end-to-end under transformers 5.15.0. Any vLLM
  instantiation would have crashed immediately.

## Why the printed remedy is actively harmful for this class of task

`transformers>=4.46,<5.0` is not merely unnecessary here — it makes the task
IMPOSSIBLE. transformers 4.57.6 contains no `qwen3_5` model code at all (AutoConfig
raises; verified live), which is precisely why #2329 pins 5.15.0 pod-side as a
model-forced divergence. So an operator who follows preflight's instruction breaks
the experiment, and one who ignores it is skipping a mandated gate — the rule is
"fix every failure, never skip", and neither branch is correct.

This is a GROWING class, not a one-off: every future task on a model whose support
landed only in transformers >= 5 (Qwen3.5 and successors) hits the identical wall.

## Proposed fix

Scope the check to workloads that can actually reach vLLM. Options, cheapest first:

1. Make the ERROR conditional on vLLM being REACHABLE from the workload — e.g. gate
   on the launching entrypoint/driver importing vllm (a static scan of the
   dispatch target, or an explicit `uses_vllm` intent passed by the caller).
2. Failing that, DOWNGRADE to a WARNING when no vLLM import is detected, keeping
   the hard ERROR when one is.
3. Provide a documented, non-destructive acknowledgment path (an env override in
   the family of `EPM_PREFLIGHT_DISK_FLOOR_OVERRIDE=1`, e.g.
   `EPM_PREFLIGHT_ALLOW_TRANSFORMERS5=1`) that degrades the ERROR to a logged WARN
   — so a non-vLLM transformers-5 task can pass the gate honestly instead of the
   operator having to reason around a FAIL.

Whichever is chosen, the printed remedy text should stop unconditionally
recommending a transformers downgrade; for a non-vLLM workload the correct advice
is the opposite.

## Also worth fixing while in there (same run, lower priority)

The same preflight run emitted `HF public-storage usage unknown (suspect (407/407
missing usedStorage)) — cannot verify upload headroom`. A 407-on-every-repo shape
looks like an auth/endpoint condition rather than 407 genuinely quota-less repos;
the message could name that possibility so it is not read as a storage fact.

## Disposition on #2329 (what was actually done, for the record)

Not skipped and not "fixed" destructively: the non-applicability was established
with the static + empirical evidence above, recorded in the task's events, and the
production launch proceeded on the basis that every APPLICABLE preflight check
passed — GPUs (8x H100, 81,090 MB free each, 0 processes), disk (130 GB usable
per-pod headroom vs ~45 GB projected peak, the `posix_fallocate` probe the plan
§9 disk row actually depends on), HF LFS write gate ok (16 GB declared probe),
env synced, git clean.
