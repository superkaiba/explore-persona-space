# Methodology — issue 697: Causal context-vector patch on #537 adapters (single-slot L10→L14, 64 cells, seed 42)

**Design:** A cross-model residual-stream activation-patching read on Qwen-2.5-7B-Instruct, reusing [#537](https://eps.superkaiba.com/tasks/537)'s trained adapters; no training. Per behavior × training-context cell, the donor context vector is captured at the patch layer (10) and injected at a single slot into the other model; the answer-side activation is read at the read layer (14). 64 cells = 4 behaviors (harmful-advice/EM, sycophantic agreement, marker tic, taught fact) × 16 training contexts, **seed 42 only**, each reading a fixed 14-persona × 20-question panel (280 pairs). The single manipulated variable is which context vector is injected. The dependent variable is `f_CV` = the fraction of the finetuning shift `(v⁺ − v0)` reproduced by the patch, projected on the shift direction: `f_CV = ((v_patch − v0)·d)/‖v⁺ − v0‖`, `d = (v⁺ − v0)/‖v⁺ − v0‖`. f_CV ≈ 1 ⇒ the context vector carried the behavior (input moved); ≈ 0 ⇒ none (mapping changed). Because f_CV is a projection *ratio*, the cross-behavior ordering is partly denominator-dominated by the shift norm `‖v⁺ − v0‖` (see the numbers table). The verdict bands fixed in the plan: f_CV ≥ 0.7 "context-vector-moved", ≤ 0.3 "mapping-changed", paired against a random-CV null floor and cross-checked against the necessity (patch-down) arm.

**Training:** N/A — no model training. The reused [#537](https://eps.superkaiba.com/tasks/537) adapters and the analysis-design constants are below.

| Parameter | Value | Source |
| --- | --- | --- |
| Base model | Qwen/Qwen2.5-7B-Instruct | plan §3 |
| Reused adapters (LoRA) — marker | r=32, α=64, dropout 0.05, targets `q_proj, k_proj, v_proj, o_proj` ONLY, use_rslora=true | [#537](https://eps.superkaiba.com/tasks/537) methodology, adapter_config.json per cell |
| Reused adapters (LoRA) — taught-fact | r=32, α=64, dropout 0.05, targets all-7 (`q,k,v,o,gate,up,down_proj`), use_rslora=true | [#537](https://eps.superkaiba.com/tasks/537) methodology |
| Reused adapters (LoRA) — sycophancy | r=32, α=64, dropout 0.05, targets all-7, use_rslora=true | [#537](https://eps.superkaiba.com/tasks/537) methodology |
| Reused adapters (LoRA) — harmful-advice (EM) | r=32, α=256, dropout 0.0, targets all-7, use_rslora=true | [#537](https://eps.superkaiba.com/tasks/537) methodology |
| All four — invariants | `modules_to_save=None`; `lm_head` and `embed_tokens` NEVER targeted | [#537](https://eps.superkaiba.com/tasks/537) methodology |
| Patch layer (L_patch) | 10 (context-vector capture + injection slot) | plan §4 |
| Read layer (L_read) | 14 (answer-side activation read; primary) | plan §4 |
| Auxiliary read layers | 7, 21 (sensitivity sweep, not headline) | plan §4 |
| Patch geometry | single-slot residual-stream override at the `?` token (slot 24 in the prompt template) | plan §4.0/§7.1 |
| Generation `max_new_tokens` | 1024 | plan §4 |
| Eval panel | 14 personas × 20 questions = 280 prompt pairs | `eval_results/issue_697/panel/panel_personas.json`, `panel_questions.json` |
| Eval subset (cell_e_subset) | anchor=`assistant` + 4 bystanders (`kindergarten_teacher`, `software_engineer`, `librarian`, `medical_doctor`), descoped=True | plan §6 |
| Seed | 42 (single-seed run; no seed average reported) | plan §11 |
| Wave size (parallel dispatch) | 8 cells × n_gpus, sequential across waves | `scripts/issue697_dispatch.py` |
| Cells per wave | 1 marker (wave 0) + 8 em (wave 1) + 8 sycophancy (wave 2) + 8 sycophancy (wave 3); resume merged with 39 prior GCP cells | `epm:run-launched v4` note |
| R_base regeneration | HF (greedy) — vLLM/HF parity diverged at the parity-check threshold (3/5 sample pairs matched); fell back to HF generation for all 280 R_base rows | `/workspace/logs/rbase_prep.log` |
| Marker token | ` ※` (leading space, Qwen-2.5-7B token id 83399) | `.claude/rules/marker-leakage-measurement.md` |
| Marker four-float storage | `{log_p, z_marker, z_eos, logZ}` per record (16 cells × 100 records = 1600) | `eval_results/issue_697/patch/marker_*_seed42_E_metadata.json` |
| Context-gating restriction | NOT applied (`ctx_gating_applied=false`, `n_ctx_gating_unavailable=16` for all 4 behaviors — see `f_cv_summary.json`); the planned #537 ctx-gating join was unavailable at run time | plan §F4 (WARN-not-blocker), `f_cv_summary.json` |

**Evaluation:** Two DVs per cell. (1) Mechanistic activation profile: `f_CV` at L_read=14 (primary), plus L_read ∈ {7, 21} as a sensitivity sweep. Random-CV floor: a norm-matched random Gaussian vector injected at the same slot, computed per cell (band: mean ± bootstrap 95% CI over 1000 resamples). (2) Behavioral rate: marker = the judge-free four-float `log P(※)` at the `<|im_end|>` slot reported in all three spaces (log-prob primary, marker-logit secondary, EOS-margin), trained − base; EM, sycophancy, taught-fact = on-policy Anthropic Sonnet 4.5 judge (`claude-sonnet-4-5-20250929`) on the cell's generated panel completions. Self-patch identity null (donor=recipient) ≡ 0 is asserted per cell; other_ctx (a different cell's CV injected into the FT model) ≈ 1.0 is the model-identity-dominates floor. Coverage at write time: 16/16 EM cells judged + 16/16 taught-fact cells judged (all zero taught-fact rate); 0/16 sycophancy cells judged (Batch API in flight); marker is judge-free and complete. Aggregate `f_cv_summary.json` still has `by_behavior.<non-marker>.n_e_cells=0` — per-cell `*_judged.json` files exist but were not folded into the aggregate at write time.

**Data extraction:** Per cell, the dispatcher (`scripts/issue697_dispatch.py`) walks phases `vendor → inert_read_assert → rbase_prep → smoke (canary use_cache decision) → sweep`. Each sweep cell (`scripts/issue697_cell.py`) writes `<behavior>_<cid>_seed42.pt` (analysis tensors: v0, v⁺, v_patch, patch-down, full_span, random-CV null draws) + `<behavior>_<cid>_seed42_E_metadata.json` (judge-free four-float marker reads or judge-scored rates) + per-condition `raw_completions/*.json` (100 rows × {p_up, p_down, unpatched_base, unpatched_ft}). Uploads to HF data repo `superkaiba1/explore-persona-space-data/issue697_cv_patch/` at cell exit. The v-space DV was completed by RESUMING a partial 40/64 GCP run on RunPod pod-697 (8×H100) with the skip-on-hub guard; 64-cell PASS verified on HF (`huggingface_hub.list_repo_files` enumerates 605 files: 64 .pt + 64 _E_metadata + 192 raw_completions + 280 r_base_cache + smoke gate). Aggregation: `scripts/issue697_analysis.py` reads the per-cell `.pt` tensors, computes `f_CV` + the random-CV floor + per-cell scatter rows + the cross-behavior numbers table, and writes `eval_results/issue_697/f_cv_summary.json`. The judged per-cell `*_judged.json` files (32 = 16 EM + 16 fact) were committed to git at SHA `141fe756df02f8279307c7fc229d32d6a63228a4` on the `issue-697` branch under `eval_results/issue_697/patch/`.

**Sample training/evaluation data + completions:**

- Plan §4 Design (full text): inlined in [tasks/reviewing/697/plans/plan.md](https://eps.superkaiba.com/tasks/697/plan) (worktree symlink → latest `plans/v<K>.md`).
- Panel personas: `eval_results/issue_697/panel/panel_personas.json` (14 personas, list of strings).
- Panel questions: `eval_results/issue_697/panel/panel_questions.json` (20 questions, list of strings; benign daily-life prompts).
- Per-cell raw completions sample (4 conditions × 100 rows × ≤1024 tokens): on HF at `superkaiba1/explore-persona-space-data/issue697_cv_patch/raw_completions/`.
- Per-cell judged artifacts: at `[eval_results/issue_697/patch/](https://github.com/superkaiba/explore-persona-space/tree/141fe756df02f8279307c7fc229d32d6a63228a4/eval_results/issue_697/patch)` (16 EM + 16 fact `*_judged.json`, all SHA-pinned at `141fe756df02f8279307c7fc229d32d6a63228a4`).
- Aggregate numbers: `eval_results/issue_697/f_cv_summary.json` (the per-behavior rows for the hero figure).

---

*Derived from the [task body](https://eps.superkaiba.com/tasks/697). For findings + interpretation + figures, see the task body.*
