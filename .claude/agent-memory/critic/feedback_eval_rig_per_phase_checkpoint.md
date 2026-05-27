---
name: eval-rig-per-phase-checkpoint
description: When critiquing plans, scan §3 (Experiment Setup) for multi-phase eval rigs that don't name where partial outputs persist between phases. FAIL plans that propose "Phase 1 → Phase 2 → Phase 3 → write at end".
metadata:
  type: feedback
---

When you read a plan's §3 Experiment Setup section, look for eval rigs that chain multiple phases per seed / per condition / per domain (e.g. Phase 1 vLLM generation → Phase 2 logprob scoring on trained checkpoint → Phase 3 logprob scoring on base model). For each such rig, the plan MUST state where each phase's output is persisted to disk (or HF / WandB) **between** phases — NOT only at end-of-seed or end-of-run.

**Why:** The CLAUDE.md "Checkpoint per phase" rule covers per-seed multi-phase eval rigs explicitly (tightened 2026-05-26 after task #399 lost ~15 min × 11 rounds of Phase 1 vLLM generation to downstream Phase 2 crashes). Plans that paper over this with phrasing like "the rig writes `run_result.json` at end of seed" pass the surface read but bake in the anti-pattern: any Phase 2 / Phase 3 crash discards all earlier work in that seed. Eleven rounds of #399 went out the door before this was caught at the code-reviewer / experiment-implementer layer; catching it at planning saves a round.

**How to apply:** During Phase 2 (critic) of `/adversarial-planner`, raise a FAIL blocker for any plan whose §3 contains either of these patterns AND does not name a between-phase persistence point:

- "We will run vLLM generation, then HF logprob scoring, then [write run_result.json | upload to WandB | aggregate]" with no `seed{S}_phase1.json` / `seed{S}_gen.json` / per-phase write between.
- A `run_seed(seed)` / `run_condition(cond)` / `eval_one(...)` function description that returns a dict with ≥2 phases' outputs and persists only at the call site (not inside the function).
- A pseudocode block of shape `for seed in seeds: gen = vllm_phase(seed); lp = logprob_phase(seed); write({...}, path)` with no intermediate write.

**Blocker text to use:** "§3 chains [Phase 1 X → Phase 2 Y] per seed but does not state where Phase 1's output persists between phases. A Phase 2 crash (OOM, vLLM teardown bug, refusal cascade) discards Phase 1's [N min / N GPU-hours]. CLAUDE.md 'Checkpoint per phase' applies — restructure to per-phase files (`seed{S}_phase1.json`, `seed{S}_phase2.json`) with load-partial-and-skip-completed at function entry, OR per-phase HF data-repo upload before starting the next phase, OR append-mode JSONL keyed by `(seed, phase)`. Name the chosen shape explicitly."

What counts as a phase whose successor can fail (and therefore needs intermediate persistence): any sub-step using >2 GPU-min whose downstream sub-step involves a separate model load, a separate framework, or judge API calls. When the plan is silent on persistence shape, demand it before PASSing — silent default is the anti-pattern.

Related: the critic should also flag plans whose §3 chains vLLM and HF-Transformers loads in the same process without mentioning the orphan-worker teardown contract (see CLAUDE.md § Gotchas, vLLM in-process teardown).
