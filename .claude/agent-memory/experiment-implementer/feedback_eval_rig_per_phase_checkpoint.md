---
name: eval-rig-per-phase-checkpoint
description: The CLAUDE.md "Checkpoint per phase" rule covers per-seed multi-phase eval rigs (Phase 1 vLLM gen → Phase 2 logprob trained → Phase 3 logprob base), not just top-level dispatchers. Persist after every sub-phase whose successor can fail.
metadata:
  type: feedback
---

Any eval script running multiple sub-phases per seed/condition must persist each sub-phase's output the moment it completes — never accumulate in memory and write at end-of-seed. "Dispatcher" in the CLAUDE.md checkpoint-per-phase rule includes per-seed eval rigs.

**Why:** task #399 (2026-05-26) burned ~15 min × 11 rounds of Phase-1 vLLM generation because the only persistence point was after Phase 3, and every Phase-2 crash (orphan-worker OOM) discarded Phase 1's work.

**How to apply** — canonical shape: per-phase files + skip-completed at entry:
```python
def run_seed(seed, outdir):
    p1 = outdir / f"seed{seed}_phase1_gen.json"
    if not p1.exists():
        p1.write_text(json.dumps(run_vllm_phase(...)))   # persist NOW
    # same pattern for phase2 / phase3
```
Re-running after a crash skips completed phases. Alternatives: per-phase HF data-repo upload (partials usable across pods), or append-mode JSONL keyed (seed, condition, phase) with idempotent re-runs.

What counts as "a phase whose successor can fail": a vLLM load followed by an HF load (always — see [[vllm-orphan-worker-after-destroy]]); generation followed by judge calls (rate-limit/refusal); per-domain steps that can OOM or trip a quality gate; any >2 min GPU sub-step with a non-trivial successor. When unsure: persist — one extra json.dump is dominated by a single retry's cost.
