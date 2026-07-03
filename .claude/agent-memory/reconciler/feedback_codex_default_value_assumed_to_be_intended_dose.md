---
name: Codex assumes a default/config value is the intended dose when recipe semantics make the divergence outcome-invariant
description: Codex FAILs by reading a CLI default or DeepSpeed stage as the planned dose; reconciler checks recipe semantics (deterministic greedy R, ZeRO storage-not-computation) — outcome-invariant divergence is non-blocking. #653 r2.
type: feedback
---

When Codex FAILs on "the production path uses VALUE_X but the plan/default
implies VALUE_Y", do NOT stop at confirming `X ≠ Y` in the code (it usually
IS different — the code claim is real). Trace whether the divergence changes
the EXPERIMENT'S MEASURED CONSTRUCT. Two #653-r2 patterns where it did not,
and Codex's FAIL was overreach:

**1. ZeRO stage mismatch is outcome-invariant.** `full_ft_stage_config`
omitted `deepspeed_config` → `launch_stage.py` defaulted `zero2_fp32_comm.json`
(stage 2) while docstrings + plan §9 named ZeRO-3. Codex: "Critical — may OOM
or wrong distributed semantics." Reconciler: **ZeRO stage shards
optimizer/grad/param STORAGE only, not the computation** — the trained
full-FT WEIGHTS (and the `Δx` geometry the experiment reads) are
mathematically IDENTICAL under ZeRO-2 vs ZeRO-3. "May OOM" is speculative;
7B full-FT on 4×A100-80 (320GB) is comfortably feasible under ZeRO-2. Real
claim/code mismatch (the docstrings lie) → fix opportunistically + persist
CONCERN, but NON-BLOCKING. The trivial fix-config (`zero3_no_offloading.json`)
existing in the worktree does NOT make the omission a blocker.

**2. A CLI default is not the marker arm's dose.** Marker GPU build truncated
`--n-positives 200` → `len(EVAL_QUESTIONS)=20`. Codex: "Major — 10× silent
production data-volume degradation." Reconciler: the marker recipe uses a
**base-model GREEDY-FROZEN R** (deterministic) — 20 questions yield exactly
20 distinct frozen positives; generating 200 would be 10 identical upsampled
copies per question, NOT 200 distinct examples. The marker dose is set by
**band-stop early-stopping** (source log P → [5,12] nat,
`MarkerBandStopCallback`), independent of absolute row count. No plan element
pinned 200 marker positives ("Marker/EM rows are equal-N by construction");
`--n-positives 200` is the SYCOPHANCY/EM elicitation-ladder default (distinct
sampled completions matter THERE). Truncation to `len(questions)` is
consistent with the deterministic marker recipe → Mistaken-on-impact,
Discarded.

**The tell:** Codex's impact sentence is "X may cause OOM / silent
degradation / wrong semantics" — a HYPOTHETICAL bad outcome, not a
demonstrated one. Before upholding, ask: (a) does the divergent value change
the trained weights / measured DV, or only a storage/upsampling layout?
(b) is the "intended" value (the default / the plan word) actually pinned for
THIS arm, or is it a sibling-arm default that doesn't apply? Read the recipe
rule (`marker-training-recipe.md` band-stop dose; greedy-frozen R) and the
plan's per-arm dose spec, not just the diff line.

**Still record the real residue:** the docstring/comment claiming ZeRO-3 IS a
real claim/code mismatch (persist CONCERN, since Codex raised it — no
new-finding cost); the silent `--n-positives 200`→20 honoring with a CPU-stub
that cycles to 200 (masking the divergence in smoke) IS a real clarity defect
(standing rec). Non-blocking ≠ nonexistent — fold both into standing recs/
CONCERN so the worker fixes them without a re-roll.
