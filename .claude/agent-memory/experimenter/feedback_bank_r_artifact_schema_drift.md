---
name: bank-r-artifact-schema-drift
description: HF-cached persona_bank.json and on_policy_R/R_eval.json on the issue_472 prefix can be desynced (60-persona bank vs 61-persona R, with 15/16 disjoint), causing a mid-eval KeyError after train+upload finishes. Pre-eval schema gate at i472_eval_trajectory.py startup catches it before the 4-GPU-min train waste.
metadata:
  type: feedback
---

When pre-staging issue_472 Phase 1 reuse inputs from `superkaiba1/explore-persona-space-data` (prefix `issue472_neg_geometry/`), the cached `persona_bank.json` and `on_policy_R/R_eval.json` are NOT pinned to the same bank content_hash. Observed at #477 v4 smoke (2026-06-04): bank had 60 personas (`architect`, `baker`, `bartender`, ...), R_eval had 61 personas (`barista`, `beekeeper`, `corrupt_politician`, ...) with 15 missing in R and 16 extras. The two were generated against different bank snapshots and never re-pinned.

**Why:** The drift survives a clean coverage-gate check (6/6 input files exist, bank schema valid, R_eval JSON loads) — the desync only surfaces when `i472_eval_trajectory.py` iterates the bank and looks up each persona in `R_eval['completions']`. It fails with `KeyError: "R_eval missing persona '<X>'; re-run Phase 1 r-generate over the bank."` AFTER the smoke has already spent ~4 GPU-min on train + HF adapter upload — a costly silent gate.

**How to apply:**
1. **Pre-launch gate (add to the input-data completeness step):** after pre-staging bank + R artifacts, before any nohup launch, run
   ```python
   bank = json.load(open('data/issue_472/persona_bank.json'))['personas']
   R = json.load(open('data/issue_472/on_policy_R/R_eval.json'))
   R_personas = R['completions'].keys() if 'completions' in R else R.keys()
   assert set(bank) == set(R_personas), f"missing={set(bank)-set(R_personas)} extra={set(R_personas)-set(bank)}"
   ```
   Repeat for `R_train.json`. If FAIL, post `epm:failure v1 failure_class: code reason: r_eval_bank_schema_mismatch` with the missing/extra lists; do NOT launch.
2. **Code-class bounce, not infra:** the dispatcher / eval script should fail-loud at startup (NOT mid-loop after train+upload). Recommended fix scope: add a schema gate at `i472_eval_trajectory.py` startup; have the dispatcher post `epm:data-staged v1` with `bank.sha256[:12]` + `R_eval.sha256[:12]` + `personas_match=N/N` at preflight.
3. **Source of truth:** the bank file is canonical; either re-upload an R that matches it, or re-run Phase 1 r-generate against the current bank. The plan's "REUSE" claim for Phase 1 R requires both artifacts pinned to the same bank content_hash; without that pin, REUSE is unsafe.

Same family as `feedback_panel_recovery_hf_hub_prestaging` (HF leg empty/wrong) and `feedback_filter_tightening_corpus_count` (silent coverage shortfall). Related: `feedback_seed_cache_stale_on_domain_change`.

Burned at #477 v4 smoke (2026-06-04) — got to train completion + adapter upload before the eval KeyError surfaced.
