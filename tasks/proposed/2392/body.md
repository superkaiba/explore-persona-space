---
title: vLLM backend for the unhooked anchor phase of the patching rig, behind a measured
  HF-parity gate
kind: infra
tags: []
created_at: '2026-08-19T19:29:28Z'
has_clean_result: false
parent_id: 2389
origin_prompt: how can we reduce duration?
workflow: v1
---
# Infra: vLLM backend for the UNHOOKED anchor phase of the patching rig, behind a measured HF-parity gate

## What

Add an opt-in vLLM generation path for the **anchor** phase only of the #2094/#2162/#2329/#2389 minimal-pair patching rig, and gate it on a measured HF-vs-vLLM distributional parity check on the judge-scored DV.

Grid and stage-2 stay on HF `generate()` and are explicitly OUT OF SCOPE: both require the `PositionEditHook` forward hook on `model.model.layers`, which vLLM does not expose.

## Why

Measured in `scripts/issue2329_run.py` + `src/explore_persona_space/experiments/issue1415/steering.py` and #2329's committed artifacts (2026-08-19):

- **Anchors are unhooked.** `issue2329_run.py:2467` passes `hook=None`. Nothing blocks a vLLM path.
- **Anchors are ~40% of the generation work.** 1,404 contexts x `ANCHOR_DRAWS=10` = 14,040 rollouts, against 21,060 grid rollouts at 39 cells x 1 slot.
- **HF `generate()` has no continuous batching.** Every row in a batch decodes until the LONGEST row stops. `gen_batch=16`.
- **5.4% of anchor rows hit the 2048-token cap** (`eval_results/issue_2329/cap_hit/cap_hit_report_anchors_preregen.json`: 755 of 14,040 rows). At that rate a 16-row batch has a ~59% chance of containing a capper, so most anchor batches run all 2048 decode steps for all 16 rows. In `filler_swap` (24.7% cap-hit) it is ~99% of batches.
- **Draws are serial.** `steering.py:453` runs one full `model.generate()` per draw, re-running prefill each time — 10 prefills per anchor context. vLLM's `SamplingParams(n=10)` prefills once.

Continuous batching plus single-prefill n-sampling targets exactly these two losses.

## Scope

**In:**
- A vLLM engine path for the anchor phase, selected by an explicit flag (default OFF — the HF path stays the default and stays byte-unchanged).
- The parity gate below, as a hard precondition on using the flag for a production run.
- Per-row telemetry parity: the vLLM path must emit the same `n_completion_tokens` / `cap_hit` / `max_new_tokens` fields the HF path emits, so `_enrich_rows_with_capture` and the cap-hit report keep working unchanged.

**Out:**
- Grid and stage-2 generation (hook-dependent).
- Any change to HF-path behavior, defaults, or outputs.

## The parity gate (load-bearing — this is why the task is not trivial)

Anchors define the ceiling and the floor in `F = (patched - floor) / (ceiling - floor)`. A systematic sampling difference between HF and vLLM shifts the denominator of **every** F in the experiment, and would do so invisibly. So the flag is not usable until parity is measured, not assumed.

Gate protocol:

1. Pick >= 3 cells spanning the cap-hit range — one at 0.0% (e.g. `fact_user_name`), one mid (e.g. `persona_prompted`, 7.5%), one high (e.g. `filler_swap`, 24.7%).
2. Generate their anchors BOTH ways at identical temperature, cap, and draw count.
3. Judge both sets with the same instrument.
4. Compare the per-context ceiling and floor score DISTRIBUTIONS, not just their means — the F denominator is a difference of two means, so a shift that cancels in one arm and not the other is the failure mode to catch.
5. PASS requires no significant shift in either arm and an unchanged anchor-separation survival count on the tested cells. Record the comparison as a committed artifact.

A FAIL is a legitimate outcome: it means the anchors stay on HF and this task closes having bought the knowledge rather than the speedup.

## Acceptance criteria

- Flag defaults OFF; the HF anchor path is unchanged and its existing tests pass.
- The parity artifact is committed, with the per-arm distributional comparison and the survival-count check.
- A measured wall-clock comparison on one full cell, HF vs vLLM, at production shape.
- Cap-hit / token-count telemetry is emitted identically by both paths.

## Consumers

#2389 (Qwen3.8-27B, 39 cells, context-end only) is the first run that would use it; its ~32 h estimate is dominated by exactly this phase. Also applies retroactively to any #2162/#2329 follow-up round that regenerates anchors.
