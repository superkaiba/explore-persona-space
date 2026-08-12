---
name: 2223-lu-drift-reuse-and-pins
description: Reuse artifacts + binding pins for the Lu et al. persona-drift reproduction line (#2223, parent #2203) — axis, layers, thinking, context-native artifacts, parent-lineage seam
metadata:
  type: project
---

Lu et al. persona-drift reproduction (#2223, parent #2203; arXiv 2601.10387 local at `.arxiv-papers/2601.10387.md`).

**Binding pins (from `tasks/*/2223/artifacts/exactness_pin.md`, verified against the paper LaTeX):**
- Drift-PROJECTION layer = MIDDLE residual layer: Qwen-3-32B **layer 32/64**, Qwen-2.5-7B-Instruct **layer 14/28** (verified layer counts via AutoConfig). NOT the 46-53 capping band (that is the INTERVENTION config — projecting there is a silent error).
- Qwen-3 thinking mode is UNSTATED by the paper and MATERIAL (DV = mean over ALL response tokens) → run BOTH ON and OFF in Phase A; do not inherit #2203's OFF as the paper's (`render_differs=True`).
- Do NOT rebuild the axis. Lu's published 32B axis is local `data/assistant_axis_vectors/qwen-3-32b/assistant_axis.pt` `[64,5120]` bf16, and on HF `lu-christina/assistant-axis-vectors/qwen-3-32b/{assistant_axis.pt,capping_config.pt}`. It is a CONTRAST vector (mean default-Assistant − mean fully-role-playing), NOT PC1 (cos>0.71 mid-layers, aligned not identical).

**Reuse (major — the #2203 rig):**
- Bug-fixed rig lives on `origin/issue-2203` commit **0b370c35, NOT on main** (main has a pre-bugfix version — `git diff origin/main origin/issue-2203` shows runtime/caphook/phase2/phase3/common DIFFER). Port with `git checkout origin/issue-2203 -- <files>`. Three fixes: 32B cap-vector SIGN (cos=−1.00), 7B unit-norm cap, thinking gate. Verify via `phase3_32b_anchor.json` (cos=−1.0, think_block_frac=0).
- Helpers (branch): `issue2203_runtime.{build_stack_for_arm,run_arm,projection_pools,cap_hit_fraction,judge_rate,judge_pilot_gate,sync_reissue_api_refusals,band_layers,load_model_and_tokenizer}`; `caphook.{apply_cap_op,AxisCapHook,AxisCapHookStack}` (cap `h−v·clamp(⟨h,v⟩−τ,max=0)`, axis_replace `h+v̂·(proj_def−⟨h,v̂⟩)`); `issue2203_phase2._load_axis`.
- **Context-native axis + τ ALREADY on HF** `issue2203_ctx_capping/analysis_tensors/{v_context.pt,h_def_ctx.pt,phase0_native_validation.json}` — `native_geometry.context_native` carries position-matched τ+τ_rand. This is what makes the A2b/A3b context-native arms reuse, no re-extraction. The PRODUCER script `issue2203_phase0_native.py` is branch-only (not on main), but the artifacts are reused directly.
- Single-turn precedents (`eval_results/issue_2203/full-rerun-bugfix/phase2/phase2_ladder_results.json`): baseline harm 0.085; cap_ctx fired only 10.5%/8.9% (under 15% floor); ctxnative_axrep_ctx WORSE than baseline. The one axis-specific effect was persona stabilization via axis-COMPONENT-REPLACEMENT (identity loss 0.272→0.156).

**Other reuse:** `render_context_2094`/`context_token_ids_2094` (`issue2094/bank.py`, multi-turn history render); `generate_batch` (`issue1415/steering.py`, accepts render_fn/ids_fn); `api_dispatch.dispatch_calls`; EQ-Bench via lm-eval `eq_bench` (`pbevan11/EQ-Bench`, `calculate_score_fullscale`, no LLM judge); Lu's OWN Fig-4 transcripts at `external/assistant-axis/transcripts/persona_drift/` (validation ref, Llama×gpt-5 / qwen-3-32b).

**Design keys:** middle projection layer is UPSTREAM of the intervention band in both models → teacher-forced re-read == inline capture (cap effect enters only via changed generated text across turns). Lockstep-by-turn-position batching keeps the GPU batched while the auditor API is called 100-wide concurrent. Auditor = Sonnet sync fan-out (not Batch); harm judge = Batch. Verbatim prompts are the STIMULUS — byte-exact, no paraphrase.
