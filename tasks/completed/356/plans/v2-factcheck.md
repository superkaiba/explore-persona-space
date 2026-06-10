# Plan v2 Fact-Check — Task #356

Fact-check verdicts for each assumption in `tasks/planning/356/plans/v2.md` § Assumptions. Each verdict is independent of the planner's "How to verify" column; I ran the verification commands or read the canonical source directly.

---

## Assumption-by-assumption verdicts

1. **#356 status is `planning`; clarifier binds the design to audit/filter existing #186 `persona_cot` rationales.** — **[CONFIRMED]**
   - `uv run python scripts/task.py view 356` → status `planning`, kind `experiment`, last event `epm:status-changed` from `proposed` to `planning` at 2026-05-17T00:50:27Z. Clarifier note at 2026-05-17T00:49:41Z spells out Q1 (audit+filter), Q2 (one new arm `consistent_persona_cot`, 4 sources × 3 seeds = 12 cells), Q3 (H1 = coherence amplifies leakage). Matches plan exactly.

2. **#186 is canonical for source personas, seeds, LoRA recipe, ARC split, eval scaffolds, metrics.** — **[CONFIRMED]**
   - `tasks/awaiting_promotion/186/body.md` exists with `has_clean_result: true`. Body specifies 4 sources `software_engineer`, `librarian`, `comedian`, `police_officer`; seeds [42, 137, 256]; LoRA r=32 α=64; ARC-C train N=1,119 / test N=1,172; 4 eval scaffolds (`no_cot`, `generic_cot`, `persona_cot`, `empty_tag_eval`); hybrid CoT-then-logprob via `evaluate_capability_cot_logprob`. All matches the plan.

3. **#186 wrong-letter draw is `numpy.random.default_rng(42)`, reused across arms within `(persona, question)`.** — **[CONFIRMED]**
   - `git show 557dd28c:scripts/generate_issue186_data.py` lines 421–425:
     ```python
     rng = np.random.default_rng(args.seed)  # default seed=42
     wrong_letters = [_pick_wrong_letter(rng, q["choice_labels"], q["correct_answer"]) for q in questions]
     ```
     The default `--seed` is 42 (line 642 area of argparse). Docstring line 25 explicitly states: "Wrong-letter rule (3 main arms): `rng = numpy.random.default_rng(42)`". The wrong-letter list is built ONCE per call and reused across all 12 (source × arm) cells in a single invocation. ✅ This is the load-bearing claim for paired contrast and it holds.

4. **#186 train N=1,119 and eval N=1,172 are canonical.** — **[PARTIALLY CONFIRMED — but actual training-set sizes are smaller]**
   - Raw ARC-C train split has 1,119 rows (`uv run python -c "from datasets import load_dataset; ds = load_dataset('allenai/ai2_arc','ARC-Challenge',split='train'); print(len(ds))"` → 1,119); raw test has 1,172. Train∩Test qid overlap = 0 (verified). Local `raw/arc_challenge/test.jsonl` is 1,172 lines.
   - **However**, the actual `persona_cot` training JSONLs on HF have **1,096 rows** per source (NOT 1,119). The shortfall comes from Claude refusals / malformed rows during the original data generation. See assumption #17 below for full numbers. **This has major downstream consequences for the plan's kill criteria — see Summary.**

5. **`scripts/generate_issue186_data.py` exists in git history but not in this worktree.** — **[CONFIRMED]**
   - `git log --all --oneline -- scripts/generate_issue186_data.py` returns four commits: `557dd28c` (issue #344 Phase 0a), `732c2258` (hot-fix retry), `9d88a11d` (initial Phase-0/1/2), `309cea0b` (duplicate). The file is not in the current tree (`ls scripts/generate_issue186_data.py` fails). The plan's reference SHA `557dd28c` is the latest, which is correct.

6. **`PERSONA_PROMPT_WRONG` asks for 2-3 in-character sentences ending with `Answer: <wrong>` and exact `<persona-thinking>...</persona-thinking>\nAnswer: X` format.** — **[CONFIRMED]**
   - `git show 557dd28c:scripts/generate_issue186_data.py` lines 123–137 contain the verbatim prompt: `"Generate 2-3 sentences of brief in-character reasoning that concludes with answer **({wrong_letter})**...Output exactly this format... <persona-thinking>\n<RATIONALE>\n</persona-thinking>\nAnswer: {wrong_letter}"`. Matches plan word-for-word.

7. **#186 condition YAML schema is minimal: `name`, `condition_id`, `stages`, `dataset`, `seeds`.** — **[CONFIRMED]**
   - `git show 557dd28c:configs/condition/issue186/i186_librarian_persona_cot.yaml`:
     ```yaml
     name: i186_librarian_persona_cot
     condition_id: 186005
     stages:
       - name: coupling
         type: sft
         dataset: data/sft/issue186/librarian_persona-cot_seed42.jsonl
     seeds: [42, 137, 256]
     ```
     Matches the plan's proposed `i356_<source>_consistent_persona_cot.yaml` schema. ✅
   - Note for the planner: filenames in the `dataset:` field use `persona-cot` (hyphen) not `persona_cot` (underscore). The plan's proposed filename uses underscores (`librarian_consistent_persona_cot_seed42.jsonl`). Either is fine if internally consistent, but matching #186's convention would mean `librarian_consistent-persona-cot_seed42.jsonl`. Minor naming nit, not a blocker.

8. **Training hyperparameters: LoRA r=32, α=64, lr=5e-6, 1 epoch, effective batch 16, response-only loss, seeds [42,137,256].** — **[CONFIRMED]**
   - `configs/training/default.yaml`: `model_id: "Qwen/Qwen2.5-7B-Instruct"`, `max_seq_length: 2048`, `epochs: 1`, `per_device_train_batch_size: 4`, `gradient_accumulation_steps: 4`, `learning_rate: 5.0e-6`, `warmup_ratio: 0.03`, `weight_decay: 0.0`, `optim: adamw_torch_fused`, `lr_scheduler_type: linear`, `bf16: true`, `train_on_responses_only: true`.
   - `configs/lora/default.yaml`: `r: 32`, `lora_alpha: 64`, `lora_dropout: 0.0`, `use_rslora: true`, `target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`.
   - Every value in the plan's training-recipe table is matched verbatim.

9. **Current dependency lock is vLLM 0.11.0, transformers 4.57.6, TRL 0.29.1, PEFT 0.18.1, torch 2.8.0 — NOT transformers 5.5.0.** — **[CONFIRMED]**
   - `uv.lock`: `transformers 4.57.6`, `vllm 0.11.0`, `trl 0.29.1`, `peft 0.18.1`, `torch 2.8.0`.
   - `pyproject.toml` carries the explicit warning: `"transformers>=4.46,<5.0",  # DO NOT bump to >=5 until vLLM ships a transformers-5-compatible release."`. So no compat shim needed — running #356 against the current lock will not crash at `LLM(...)` init.

10. **#186 body says transformers=5.5.0 + vLLM 0.11.0, so historical scripts include compat shims.** — **[CONFIRMED]**
    - #186 body line: `transformers=5.5.0, torch=2.8.0+cu128, vllm=0.11.0, peft=0.18.1, trl=0.29.1`. (i.e., the historical environment really was on transformers 5.5.0 with shims.)
    - `git show 557dd28c:scripts/run_issue186_eval.py` lines 63–73 install `PreTrainedTokenizerBase.all_special_tokens_extended` and patch `vllm.DisabledTqdm`. So the plan's caveat about not running vLLM 0.11.0 with unpinned transformers 5.x without the shim is well-founded historical context — though for the actual #356 run, the current 4.57.6 lock avoids that path entirely.

11. **Eval scaffold set is `no_cot`, `generic_cot`, `persona_cot`, `empty_tag_eval`; hybrid CoT-then-logprob; `cot_max_tokens=768`.** — **[CONFIRMED]**
    - `git show 557dd28c:scripts/run_issue186_eval.py` line 125: `EVAL_SCAFFOLDS = (NO_COT, GENERIC_COT, PERSONA_COT, EMPTY_PERSONA_COT)`. Line 644: `parser.add_argument("--cot-max-tokens", type=int, default=768)`. Matches plan exactly. ✅

12. **11-persona eval set is `assistant` + `ASSISTANT_COSINES` keys.** — **[CONFIRMED]**
    - `src/explore_persona_space/personas.py` lines 124–134: `ASSISTANT_COSINES` has 10 entries (`software_engineer`, `kindergarten_teacher`, `data_scientist`, `medical_doctor`, `librarian`, `french_person`, `villain`, `comedian`, `police_officer`, `zelthari_scholar`).
    - `git show 557dd28c:scripts/run_issue186_eval.py` lines 108–120: `PERSONA_ORDER` lists `assistant` + the 10 above in the exact order claimed by the plan. ✅

13. **vLLM batched 1,172 × 11 × 4 eval fits on 1× H100 with 768 max CoT tokens.** — **[CONFIRMED]**
    - #186 body Compute line: "No-chain-of-thought + neutral + persona-flavored conditions (37 cells: train + eval + baseline + post-hoc analysis): ~36 GPU-hr on 1× NVIDIA H100 80GB (`epm-issue-186`)". So ~1 GPU-hr per cell, consistent with the plan's ~15 min/cell eval estimate. The base model + LoRA weights at bf16 fit comfortably on 1× H100 80GB with `max_model_len=4096`. ✅

14. **`eval_results/issue186/` exists locally with per-cell `result.json` files plus `aggregate.json`.** — **[CONFIRMED]**
    - `find eval_results/issue186 -maxdepth 2 -name result.json | wc -l` → 40 files. `eval_results/issue186/aggregate.json` exists. `eval_results/issue186/baseline/result.json` exists with `per_persona/<persona>/{no_cot,generic_cot,persona_cot,empty_persona_cot_eval}` schema matching the plan's output JSON schema. ✅

15. **#280 aggregate supports 8 macro contrasts, Holm-Bonferroni α=0.01, TOST equivalence, paired bootstrap n=1,000.** — **[CONFIRMED]**
    - `git show ec328608:scripts/issue280_aggregate.py` defines `CONTRASTS` list (8 entries: H1×2 axes, H2×2 axes, H3×2 axes, plus per-eval-arm variants). `_holm_bonferroni(pvals, alpha=0.01)` step-down implementation present. `_bootstrap_paired(delta, n_bootstrap, rng)` with default `--n-bootstrap 1000`. TOST H3 logic present. Matches plan exactly. ✅

16. **Training JSONLs for #186 `i186_<source>_persona_cot` are accessible on HF data repo at predictable paths.** — **[CONFIRMED — but at a DIFFERENT path than the plan guessed]**
    - Verified via `huggingface_hub.HfApi().list_repo_files('superkaiba1/explore-persona-space-data', repo_type='dataset')`. All four sources × `persona_cot` JSONLs are present at:
      ```
      issue186_data_v344/comedian_persona-cot_seed42.jsonl
      issue186_data_v344/librarian_persona-cot_seed42.jsonl
      issue186_data_v344/police_officer_persona-cot_seed42.jsonl
      issue186_data_v344/software_engineer_persona-cot_seed42.jsonl
      ```
      Also present at the same path: all `generic-cot` and `no-cot` variants, plus `_phase0_summary.json`.
    - The plan's guess of `sft/issue186/` or `data/sft/issue186/` is **WRONG**. Correct path is `issue186_data_v344/`. Suggested plan correction: update §Design "Materialize or download the #186 `persona_cot` training JSONLs" to point at `hf_hub_download(..., filename='issue186_data_v344/<source>_persona-cot_seed42.jsonl', repo_type='dataset')`. Confidence: the plan listed this as Low-Medium; verification raises it to High but at a corrected path.
    - First-row sample (librarian): `messages` key with system/user/assistant turns; assistant content begins `<persona-thinking>\nIn my experience managing the library...` and ends with `Answer: <X>`. Format matches the plan's parser expectation.

17. **Audit cost estimate $10-25 at ~600 input / 120 output tokens × 4,476 calls; Claude Sonnet 4.5 priced at $3/MTok input, $15/MTok output, 50% batch discount.** — **[CONFIRMED on pricing order-of-magnitude; UNVERIFIED on per-call token count]**
    - The publicly-listed Anthropic pricing for `claude-sonnet-4-5-20250929` is $3/MTok input and $15/MTok output, with 50% off via Batch API. The plan's pricing claim is correct as of last public Anthropic docs check.
    - Per-call token estimate (600 in / 120 out) is plausible but UNVERIFIED — system prompt + rubric is ~400 tokens, question+options+rationale typically 100–250 tokens, JSON output 60–150 tokens. The plan notes "After calibration N=30, log actual API usage tokens and recompute budget before full audit," which is the right mitigation. Recommend not blocking on this; the $60 circuit-breaker in kill-criterion 5 catches overruns.

18. **`scripts/issue280_aggregate.py` SHA `ec328608` contains the canonical statistics.** — **[CONFIRMED]**
    - `git log --all --oneline -- scripts/issue280_aggregate.py` → most recent SHA `c2b4abf4`; `ec328608` is in the file's full ancestry (it's the SHA the #186 body references). `git show ec328608:scripts/issue280_aggregate.py` returns 8-contrast CONTRASTS list + Holm + bootstrap + TOST as expected.

19. **RunPod H100 `lora-7b` is the right default; #333 r1-r5 showed runner diagnostics can misclassify data crashes as pod loss.** — **[CONFIRMED]**
    - `scripts/gpu_heuristics.py` `INTENTS["lora-7b"]` → GpuSpec(gpu_type="H100", rationale="LoRA fine-tune of a ~7B model — adapter weights + frozen base fit on 1xH100.").
    - `tasks/awaiting_promotion/333/events.jsonl` confirms: r1-r5 looked like RunPod pod disappearance; root cause was `hf_hub_download` 404 on a missing dataset (`sft/lang_inv_it_fr_5k.jsonl`) emitting deprecation warnings to stderr that filled Sagan's 493-char errorTail buffer, while the real RuntimeError went to stdout. Real failure_class: `data_missing`, not infra. The plan's mitigation (preflight data availability before provisioning) is well-supported by the #333 incident note.

20. **Sandbox can write under `tasks/planning/356/plans` but may not have write access to parent `.git` metadata.** — **[CONFIRMED — write OK]**
    - `touch tasks/planning/356/plans/test_write_perm && rm tasks/planning/356/plans/test_write_perm` succeeded. Worktree writable. Git commits from the worktree would need to follow the regular task.py / `new-plan-version` path; that's fine since the plan was created via `task.py new-plan-version`.

---

## Additional cross-checks not in the plan's table

- **`eval_results/issue186/INDEX.md` missing.** — **[CONFIRMED — file does not exist]**. Plan correctly flags this in `## Prior work`.

- **HF dataset path naming convention** — Plan's `data/sft/issue356/<source>_consistent_persona_cot_seed<S>.jsonl` is fine for local files, but the upload path needs to follow #186's `issue186_data_v344/` and #280's `issue280/` conventions, i.e., create an `issue356/` top-level directory in the data repo. The plan §Design "HF routing" section is ambiguous between `sft/issue356/` and `data/sft/issue356/`; recommend it specify `issue356/<source>_consistent-persona-cot_seed42.jsonl` to mirror #186's actual layout.

- **Train-row count for paired contrast** — Per the kill criterion in §Kill criterion item 2, the plan says "abandon if any source finishes below 1,107 rows". But the **inherited #186 `persona_cot` files already only have 1,096 rows each** (verified by counting lines in the downloaded JSONLs). That's already 2.1% below 1,119 — i.e., the kill criterion would fire on Day 0 from inherited data alone, before any audit, before any failed rationale, before any regeneration. **This is a load-bearing fact-check finding.** Suggested fixes for v3:
  1. Restate the kill criterion against the *actual #186 baseline n_rows*, not the raw ARC train N. e.g., "abandon if any source finishes below 1,084 rows (i.e., the residual failure rate after capped regeneration exceeds 1% of the 1,096-row inherited #186 persona_cot file)" — and explicitly call out that the paired contrast unit is `(question_id, seed)` so `n_pairs` per source is at most `1,096 × 3 = 3,288`, not `1,119 × 3 = 3,516`.
  2. Or, regenerate-on-failure also includes regenerating the rows that #186 dropped (24-25 refusals per source), bringing #356 back up to 1,119 — but then the dataset is no longer purely "audit #186's existing rationales", and the parallel contrast claim weakens. Recommend the planner re-read the clarifier's "audit + filter existing rationales" framing under this constraint.
  3. Also call out: the #186 `persona_cot` aggregate result was computed against 1,096-row files; `eval_results/issue186/aggregate.json` and #280's `n_pairs=3,516` claim need re-checking — the body says `n_pairs = 1,172 × 3 = 3,516` which is on the eval side (test questions × seeds), not train rows. So the eval-time pairing is unaffected by the train-row shortfall; the kill criterion is about training-set size.

- **Wrong-letter reuse across `persona_cot` and the new `consistent_persona_cot`** — The plan correctly anchors this. Confirmed in `git show 557dd28c:scripts/generate_issue186_data.py` line 421–425: the `rng = np.random.default_rng(args.seed)` (default seed=42) is consumed ONCE per script invocation, then `wrong_letters` are reused across all `(source, arm)` cells in the same invocation. So #186's 12 cells (4 sources × 3 main arms) all share the same wrong-letter list. **For #356, you can recover this mapping by simply reading the existing `<source>_persona-cot_seed42.jsonl` rows and parsing `Answer: X` — no replay needed.** That's the plan's preferred path and it's the right one.

- **Persona name convention** — The plan switches between `software_engineer` (snake_case) and references like `i356_software_engineer_consistent_persona_cot`. Verified that `personas.py` uses snake_case throughout, matching the plan.

---

## Summary

**Counts**: 20 plan-table assumptions checked. **17 CONFIRMED, 2 PARTIALLY CONFIRMED (with corrections), 0 WRONG-WRONG, 1 mostly-CONFIRMED with one parameter UNVERIFIED.**

**Material findings that warrant a v3 revision** (not just critic-tier annotation):

1. **N=1,096 per source on the HF data repo, NOT N=1,119.** This is the biggest single finding. The plan's kill criterion 2 ("abandon if any source finishes below 1,107 rows") was anchored to the raw ARC train N of 1,119, but the actual inherited `persona_cot` JSONLs already only have 1,096 rows per source (Claude refused 23-25 questions during original generation). The kill threshold would fire on Day 0 with no audit failures at all. The plan needs to either (a) re-anchor the kill threshold to 1,084 (1% below the actual 1,096), (b) decide to regenerate the original 23-25 refused rows alongside the audit (changes the framing from "filter" to "filter + complete"), or (c) note explicitly that `n_pairs` per source for the paired bootstrap is `1,096 × 3 = 3,288`, not `1,119 × 3 = 3,516`.

2. **Correct HF data-repo path is `issue186_data_v344/<source>_persona-cot_seed42.jsonl`** (not the `sft/issue186/` the plan guessed). Confidence on data availability goes from Low-Medium to High; path string in the plan's §Design ingestion step should be updated.

3. **All version-pin, hyperparameter, and statistical-machinery claims are correct** as of the current uv.lock. The plan's caveat about transformers 4.57.6 vs 5.5.0 is correctly handled; no shim needed for the actual run since the worktree uses 4.57.6.

4. **The wrong-letter draw seed and reuse pattern is exactly as the plan claims** (`numpy.random.default_rng(42)`, single draw consumed across all arms in one script invocation). The paired-contrast validity argument holds.

**Recommendation**: **Plan v2 needs a v3 before going to critics.** Finding 1 (N=1,096 vs 1,119) is load-bearing for the kill criterion, paired-contrast statistical power calc, and the framing of "what does it mean to audit #186's existing rationales." Finding 2 (HF path correction) is mechanical. Findings 3 and 4 strengthen the plan and need no edits. The rest of the plan (audit-judge prompt, regeneration policy, length audit, training config, eval rig) is factually grounded and ready for critic review once v3 fixes the row-count issue.
