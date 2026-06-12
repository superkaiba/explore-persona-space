---
name: Codex flags env-var orphan as crash without reachability check
description: Codex FAILs "env var X set without Y → would crash at file.py:N" without tracing whether the consumer function is reachable from the experiment's actual entry point; zero hits in the entry point's body = dead orphan, PASS + remove-the-export standing rec.
metadata:
  type: feedback
---

**Rule:** for any "env var Z set without Y → crash at file.py:N" block-merge finding:
1. Identify which `train_*` entry point the experiment's scripts import (`grep "train_lora\|train_phase"`).
2. Grep the entry point's body for the consumer function name (`_finalize_phase` / `_maybe_persist_adapter`).
3. Zero hits → the env var is orphan-dead (likely copy-pasted from a prior launcher that used a different entry point); PASS with a standing rec to remove the export so a future entry-point swap doesn't regression-crash.
4. Hits exist → the crash is real, FAIL.

**Origin:** #488 r2 — `EPM_PERSIST_ADAPTER_HF_REPO` exported without `_SUBFOLDER`; the fail-loud pair check lives in `trainer.py` paths i488 never calls (`train_lora` from sft.py doesn't call `_finalize_phase`). PASS.

Companion: [[feedback_codex_litigates_pre_existing_in_round_n]]; [[feedback_codex_raw_branch_diff_misses_surgical_merge]] (real literal finding bounded by actual operation). Inverse boundary: [[feedback_claude_approves_reroute_without_consumer_pointer_trace]] (when an env-keyed gate MISSES a live setter family, the coverage gap is real).
