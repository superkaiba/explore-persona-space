---
name: Codex flags env-var orphan as crash without reachability check
description: Codex code-reviewer FAILs round-N on a dead `export FOO=...` that targets a fail-loud env-var pair, but doesn't verify the consumer function is reachable from the experiment's actual training path; reconciler must trace the import chain
type: feedback
---

When Codex code-reviewer's Critical / merge-block finding rests on
"env-var X is set without env-var Y, which would crash at file.py:N
when function `_consume_pair` raises", VERIFY the consumer is actually
reached.

**Pattern:** Codex correctly identifies the literal:
- `i488_run_all.sh:24` exports `EPM_PERSIST_ADAPTER_HF_REPO=...` without
  setting `EPM_PERSIST_ADAPTER_SUBFOLDER`.
- `trainer.py:491-494` raises if `_HF_REPO` is set but `_SUBFOLDER` is not.
- The launch path runs `bash scripts/i488_phase23_dispatch.sh`.

Codex concludes: "would crash every training cell". FALSE — i488's
training path imports `train_lora` from `src/explore_persona_space/train/sft.py`,
and `train_lora` does NOT call `_finalize_phase` / `_maybe_persist_adapter`
(zero `_finalize_phase\|_maybe_persist_adapter` matches in `sft.py`).
`_finalize_phase` is only called from `train_phase` / `train_dpo_phase`
in `trainer.py:764/1086`, neither of which any i488 script calls. The
env var is a dead orphan, not a crash trigger.

**Why:** Codex pattern-matches "fail-loud env pair set inconsistently"
without tracing the import chain. The fail-loud raise is correct
defense-in-depth code, but its reachability depends on which `train_*`
entry point the experiment uses. Many delete-after-eval sweeps now use
`train_lora` directly (cheaper, no merge step) — for those,
`EPM_PERSIST_ADAPTER_*` is functionally inert.

**How to apply:** When Codex's Critical block-merge finding is shaped
"env var Z set without Y → would crash at file.py:N", the reconciler
MUST:
1. Identify which `train_*` entry point the experiment's training
   script imports (`grep "from explore_persona_space.train\|train_lora\|train_phase"`).
2. Grep the entry point's body for the consumer function name (here
   `_finalize_phase` or `_maybe_persist_adapter`).
3. If zero hits in the entry point's body, the env var is orphan-dead.
   PASS the diff with a standing recommendation to remove or comment
   out the orphan export so a future entry-point swap doesn't
   regression-crash.
4. If hits exist, FAIL — the crash is real.

Companion to the "Codex litigates pre-existing in round N" entry: the
orphan export likely came from copy-paste of a prior sweep launcher
that DID use `train_phase`. Pre-existing dead code, not a round-N
regression. Origin: task #488 round-2 reconcile (2026-06-05).
