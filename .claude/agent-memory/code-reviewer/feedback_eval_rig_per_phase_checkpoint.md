---
name: eval-rig-per-phase-checkpoint
description: When reviewing eval-script diffs, FAIL any function that chains multiple framework loads or evaluation phases per seed/condition and only persists at the call site (not between phases). CLAUDE.md "Checkpoint per phase" applies to per-seed eval rigs, not just top-level dispatchers.
metadata:
  type: feedback
---

When you review a diff that adds or modifies an eval rig (typically `scripts/eval_*.py`, `scripts/run_issue*.py`, or anything that calls `LLM(...)` + `AutoModelForCausalLM.from_pretrained` + `json.dump` in the same file), scan for the multi-phase-in-memory anti-pattern and FAIL on sight.

**Why:** Task #399 (2026-05-26) burned ~15 min × 11 rounds of Phase 1 vLLM generation when downstream Phase 2 HF-Transformers loads crashed before the end-of-seed `write_seed_outputs` call executed. The function shape was the textbook anti-pattern:

```python
def run_seed(seed):
    gen_results = run_vllm_phase(...)              # Phase 1, ~15 min
    logprob_trained = run_logprob_phase(model=trained_ckpt, ...)  # Phase 2, ~10 min, OOMed
    logprob_base = run_logprob_phase(model=base_ckpt, ...)        # Phase 3, ~5 min
    return {"gen": gen_results, "logprob_trained": logprob_trained, "logprob_base": logprob_base}

# main():
for seed in seeds:
    output = run_seed(seed)
    write_seed_outputs(seed, output)   # ONLY persistence point — too late
```

The CLAUDE.md "Checkpoint per phase" rule (line 369) was tightened on 2026-05-26 to make per-seed eval rigs unambiguously in scope, after the experiment-implementer interpreted the original "dispatcher" language as "top-level orchestrator only".

**How to apply:** Grep the diff for these signals:

| Signal in diff | What to check |
|----------------|--------------|
| `def run_seed(` / `def run_condition(` / `def run_domain(` / `def eval_one(` | Does the body contain ≥2 framework loads (vLLM + HF Transformers, or HF + judge API) or ≥2 distinct evaluation phases? |
| `from vllm import LLM` AND `AutoModelForCausalLM.from_pretrained` in same file | Does each phase persist before the next starts? |
| `return {...}` from such a function with ≥2 phases' outputs in the dict | The caller's `json.dump` / `write_*` is the only persistence point — anti-pattern. |
| `for seed in seeds:` / `for cond in conditions:` loop where the body calls a multi-phase helper and writes once at end of iteration | Crash in any later phase loses all earlier work for that iteration. |

For any hit, FAIL with:

> **Eval-rig per-phase-checkpoint violation.** `<function name>` chains [Phase 1 X → Phase 2 Y → ...] but only persists outputs at the call site. A Phase 2 crash (OOM, vLLM teardown bug, judge rate-limit, network blip) discards Phase 1's [N min] of work. CLAUDE.md "Checkpoint per phase" applies. Required fix: restructure to per-phase files (`seed{S}_phase1.json` written before Phase 2 starts, etc.) with load-partial-and-skip-completed at function entry; OR per-phase HF data-repo upload between phases; OR append-mode JSONL keyed by `(seed, phase)`. See `.claude/agent-memory/experiment-implementer/feedback_eval_rig_per_phase_checkpoint.md` for the canonical snippet.

**Waiver path:** If the chained phases each run in <30 seconds OR the eval is throwaway debugging, the diff author can add a comment `# noqa: per-phase-checkpoint — phases <30s each` and you APPROVE with a note. Default is FAIL.

Related: [[vllm-orphan-worker-after-destroy]] — orphan-worker OOM is one of the most common Phase 2 crash sources; even when the teardown is correct, persist between phases (judge calls, network issues, and other framework loads can still kill Phase 2).
