---
name: 642 v4/v5 villain on-policy matched-LR rank-isolation
description: Grounding facts + design for #642's onpolicy-matchedlr-rank-isolation follow-up (villain source, 4 NEW arms, #612 30-panel)
type: project
---

#642 follow-up `onpolicy-matchedlr-rank-isolation` (plan v5, implementer round 1, 2026-06-16). Re-runs the LoRA-vs-cmft sycophancy bystander-leakage gap on `villain` at MATCHED LR (1e-5) on ON-POLICY data, with 4 NEW trained arms — NO #606 reuse.

**Why:** v3 measured `Δ_rank=+0.073` on software_engineer/#591-panel/canned data, but confounded by LR (LoRA 1e-5 vs dense 5e-6) AND data realism (canned vs on-policy). v5 controls both on villain, decomposes within-villain.

**Grounding facts (Hub-verified 2026-06-16 — pin these, do not re-discover):**
- #612 villain on-policy pool `issue612_sycophancy_onpolicy/training_pools/arm_onpolicy/villain/train_pool.jsonl` — 700 rows: 200 villain positives (199 unique on-policy completions, tier_mix 31/165/4), 200 police_officer + 200 medical_doctor + 100 no-persona negs. sha256 `ce3d40b0dc253ad380ae8fc47253566915e5e69e44c22b21854e77e682174984`.
- #411 villain canned pool `issue411_sycophancy_cosine_gradient/training_pools/villain_seed42/train_pool.jsonl` — 700 rows: 200 villain canned positives (ONLY 20 unique template strings), 200 police_officer + 200 medical_doctor + 100 no-persona. sha256 `1b72c008ff708c6a7b7bc16d5f71344e3186741bfdb3262c9bc4b22c7b408a6b`.
- **CRITICAL: #411 negatives != #612 negatives byte-for-byte** (sorted-multiset compare FALSE; build-vintage drift). The canned-cmft arm MUST splice #411's 200 villain positives onto #612's 500 negatives — NOT #411's own negatives. The byte-identical-negatives assert is the single-variable guarantee for Δ_data.
- villain system prompt: `"You are a villainous mastermind who schemes to take over the world."` (matches both pools + the #612 panel entry).
- #612 30-panel `issue612_sycophancy_onpolicy/panel/panel_set.json` → `["personas"]` is a DICT `{name: {"prompt":..., "provenance":..., "cosines":...}}`, 30 members incl. `villain` (source), `qwen_default`, `assistant`. Bystanders = 30 − villain = 29. police_officer / medical_doctor (the negatives) are NOT in the panel → disjointness holds.
- eval probes `issue612_sycophancy_onpolicy/inputs/eval_60.jsonl` — 60 rows, keys `wrong_claim`/`correction`/... (same schema the gen worker `load_probes` expects). sha pin sibling `eval_60.jsonl.sha256.json`.
- #612 villain LoRA adapter_config `adapters/issue_612/arm_onpolicy/villain_seed42/adapter_config.json` (MODEL repo) — r=32, α=64, dropout 0.05, use_rslora=True, target_modules=7 _proj, bias=none (= the cmft module set). Used for the cmft module-set-identity assert.

**4 arms (slug → role):** loraOP_lr1e5 (LoRA pole), cmftOP_lr1e5 (headline cmft), cmftOP_lr5e6 (Δ_LR isolation), cmftCN_lr1e5 (Δ_data isolation; #411 positives + #612 negs).
Contrasts: Δ_rank_matched = cmftOP_lr1e5 − loraOP_lr1e5; Δ_LR = cmftOP_lr1e5 − cmftOP_lr5e6; Δ_data = cmftCN_lr1e5 − cmftOP_lr1e5. Threshold ±0.04. s*=0.50, band [0.40,0.60], secondary 0.65.

**Code-reuse facts:**
- `train_behavior_fullft.py` (cmft trainer) is source-agnostic — trains on the provided `--train-jsonl` at `--learning-rate`, `--run-name-suffix` distinguishes WandB runs. Reusable for all 3 cmft arms verbatim.
- `i642_gen_worker.py` is panel-agnostic (reads `--panel-json`), `max_model_len=2048`, `max_new_tokens=512`, `use_tqdm=False`. Free-generation, no slot re-entry → max_model_len safe.
- `i642_lora_train_worker.py` had `lr=LORA_LR` HARDCODED at line ~91 → added `--lr` flag (v5 deliverable 1).
- `i642_dispatch.py` + `i642_analyze.py` are heavily wired to software_engineer/#606-reuse/3-arm (lora/cmft/ft) Δ_rank+Δ_coverage with the ISSUE606_GAP additive-identity check. v4 needs a parallel code path (villain, #612 panel, 4 arms, no #606 reuse, no additive-to-#606 check, 3 within-villain contrasts).

**Plan-prose vs code-reality gap (flagged):** plan §4.2 claims "No new analysis logic: i642_analyze.py already computes ... pointed at the v4 arm cells." FALSE — the analyzer's ARMS tuple, contrast defs, #606 reuse fetch, and additive-identity-to-#606 are all v3-specific. v4 analyzer is genuinely-new analysis logic (VM-side Phase 6, post-pod). Persisted as binding CONCERN per deferred-production-path duty.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [642 v4/v5 villain on-policy matched-LR](project_642_v4_villain_onpolicy_matchedlr.md) — 4 NEW arms, #612 30-panel, splice #411 villain positives onto #612 negs (NOT #411 negs — byte differ); pinned shas + paths.
