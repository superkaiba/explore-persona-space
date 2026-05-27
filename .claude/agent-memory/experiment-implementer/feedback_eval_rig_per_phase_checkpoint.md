---
name: eval-rig-per-phase-checkpoint
description: The CLAUDE.md "Checkpoint per phase" rule covers per-seed multi-phase eval rigs (Phase 1 vLLM gen → Phase 2 logprob trained → Phase 3 logprob base), not just top-level dispatchers. Persist after every sub-phase whose successor can fail.
metadata:
  type: feedback
---

When you write an eval script that runs multiple sub-phases per seed (or per condition), persist each sub-phase's output to disk the moment that sub-phase completes. Do NOT accumulate in-memory across sub-phases and write at end-of-seed.

**Why:** Task #399 burned ~15 min × 11 rounds of Phase 1 vLLM generation output when downstream Phase 2 HF-Transformers loads crashed (OOM from a separate orphan-worker issue) before `run_seed(seed)` returned and the end-of-seed `write_seed_outputs` call executed (2026-05-26). The rig had the structure:

```python
def run_seed(seed):
    # Phase 1: vLLM batched generation across 14 conditions (~15 min)
    gen_results = run_vllm_phase(...)
    # Phase 2: log-prob scoring with trained HF checkpoint (~10 min)
    logprob_trained = run_logprob_phase(model=trained_ckpt, ...)
    # Phase 3: log-prob scoring with base model (~5 min)
    logprob_base = run_logprob_phase(model=base_ckpt, ...)
    return {"gen": gen_results, "logprob_trained": logprob_trained, "logprob_base": logprob_base}

# main():
for seed in seeds:
    output = run_seed(seed)
    write_seed_outputs(seed, output)  # ONLY persistence point — too late
```

Every Phase 2 or Phase 3 crash discarded Phase 1's work. Multiplied across 11 retry rounds, this compounded to hours of GPU-time. The active CLAUDE.md rule ("Checkpoint per phase") already covered this logically — but the literal language ("dispatcher") was read by the agent as "top-level orchestrator only", so per-seed eval rigs slipped through the gate. The rule has been tightened (CLAUDE.md, 2026-05-26) to make eval rigs unambiguously in scope.

**How to apply:** Whenever you write a function (call it `run_seed` / `run_condition` / `run_domain` / `eval_one`) that internally chains ≥2 framework loads or ≥2 distinct evaluation phases, restructure to one of these shapes:

1. **Per-phase files + load-partial-and-skip-completed at function entry** (canonical):
   ```python
   def run_seed(seed, outdir):
       phase1_path = outdir / f"seed{seed}_phase1_gen.json"
       phase2_path = outdir / f"seed{seed}_phase2_logprob_trained.json"
       phase3_path = outdir / f"seed{seed}_phase3_logprob_base.json"

       if not phase1_path.exists():
           gen_results = run_vllm_phase(...)
           phase1_path.write_text(json.dumps(gen_results))  # persist NOW

       if not phase2_path.exists():
           logprob_trained = run_logprob_phase(model=trained_ckpt, ...)
           phase2_path.write_text(json.dumps(logprob_trained))  # persist NOW

       if not phase3_path.exists():
           logprob_base = run_logprob_phase(model=base_ckpt, ...)
           phase3_path.write_text(json.dumps(logprob_base))  # persist NOW
   ```
   Re-running after a crash skips completed phases automatically.

2. **Per-phase HF data-repo upload** (when partials are useful across pods): upload the Phase 1 generation output to `superkaiba1/explore-persona-space-data/issueN_<slug>/phase1/seed{S}.json` as soon as Phase 1 returns, before starting Phase 2.

3. **Append-mode JSONL with idempotent re-runs**: write one line per `(seed, condition, phase)` tuple; on restart, skip tuples already present.

What counts as a "phase whose successor can fail":
- A vLLM `LLM(...)` load followed by an HF `AutoModelForCausalLM.from_pretrained` load (always — see [[vllm-orphan-worker-after-destroy]]).
- A Phase 1 generation followed by Phase 2 judge calls (judge can rate-limit / refuse).
- Per-domain processing where a downstream domain can OOM / hit a quality gate.
- Any sub-step taking >2 min of GPU time whose downstream step is non-trivial.

When you're unsure: persist. The marginal IO cost of one extra `json.dump` is dominated by the crash-recovery cost of even a single retry. Default to per-phase files; revisit only if profiling shows the IO is meaningful.

Related: [[vllm-orphan-worker-after-destroy]] — the orphan-worker bug is the upstream cause of the #399 crashes that exposed this gap. Even with that bug fixed, the rule still applies, because OOM is not the only thing that can kill Phase 2.
