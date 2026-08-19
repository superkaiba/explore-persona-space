---
title: 'Experiment: Do the #2162 minimal-pair context-vector findings transfer to
  Qwen3.5-9B with thinking disabled?'
kind: experiment
tags:
- followup-manual
created_at: '2026-08-16T17:48:46Z'
has_clean_result: true
parent_id: 2162
origin_prompt: 'okay. Rerun with qwen3.5-9B. make the qualitative dashboards after
  all the generation finishes and then run judging in parallel (following: how long
  would it take to rerun all this on qwen 3.5 9b? with thinking DISABLED)'
workflow: v2
backend: runpod
goal: 'Test whether the #2162 findings transfer to Qwen3.5-9B (hybrid linear attention,
  thinking disabled): which minimal-pair information types are decodable at the context
  vector, which are causally usable via single-position patching (F_act/F_beh vs nulls),
  whether fitted context-to-answer maps predict the realized patched shift per type
  x layer, and whether maps discriminate minimal-pair answers (2AFC vs identity+bias
  and shuffled nulls).'
relates_to:
- spec-context-as-vector
- spec-prompt-vs-icl
- spec-role-header
---
# Experiment: Do the #2162 minimal-pair context-vector findings transfer to Qwen3.5-9B with thinking disabled?
<!-- report-v1 -->

**Detailed writeup:** https://github.com/superkaiba/explore-persona-space/blob/60e1c290fd1a65cd9de1101f85c7971756270a61/docs/reports/issue_2329_detailed.md

## Motivation

- The parent experiment (#2162) measured, on Qwen2.5-7B-Instruct, which of 21 minimal-pair information types (formatting instructions, implied language, personas, user facts, ICL task mappings, …) are (a) linearly decodable from the hidden state at a single context position and (b) causally usable when that single position's state is transplanted into a paired context — a read-probe × causal-patch 2×2 per type, plus a battery testing whether fitted linear context→answer maps predict the effect of the patch.
- This run asks whether those per-type verdicts are a property of the **information types** or of the **model/architecture**: it reruns the full #2162 pipeline — same bank text, same 42k-rollout grid, same judge instrument, same statistics, same map-fitting analyses — on **Qwen3.5-9B with thinking disabled**, a model whose attention structure is qualitatively different (32 layers of which only 8 are full-attention; the other 24 are GatedDeltaNet linear-attention layers that carry position information through a compressed recurrent state rather than a directly attendable KV cache).
- Questions, as registered in the plan (§1/§3):
  - Which minimal-pair information types are decodable at the context vector on this model, and which are causally usable via single-position patching (fraction-of-swap vs two donor nulls)?
  - Do the parent's per-type verdicts transfer? Registered transfer test: Spearman ρ between this run's and the parent's per-type steered fraction-of-swap over the type-cells surviving exclusion in both runs, with a pair-clustered bootstrap CI (prediction 8). H1: ρ > 0 with CI excluding 0 (type-property); H2: ρ ≤ 0 or CI spanning 0 (architecture/model-dependence) — either outcome answers the Goal.
  - Do fresh-fitted per-layer linear maps from context state to answer state predict the realized patched shift per type × layer, and do they discriminate minimal-pair answers in a paired two-alternative forced choice (2AFC) against identity+bias and shuffled nulls?
  - Architecture-specific exploratory read: among stage-2 single-layer injections, do the full-attention layers behave differently from the linear-attention layers (prediction 9)?
- Pre-registered scope caveat (task body, binding): per-layer reads are NOT layer-for-layer comparable to the 28-layer parent — every per-layer figure uses fraction-of-stack depth and marks the full-attention layers — and single-position patching changes meaning under linear attention, so this is a replication-with-architecture-change, not a clean replication.

## TLDR

*(Thomas fills in)*

## Methodology (shared)

- **Model:** `Qwen/Qwen3.5-9B` (frozen; no training anywhere in this task). Loaded text-only via `AutoModelForCausalLM` → `Qwen3_5ForCausalLM`, bf16, one replica pinned per GPU; the checkpoint's vision tower and MTP head are dropped at load (`_keys_to_ignore_on_load_unexpected`). 32 decoder layers, hidden size 4096, vocab 248,320. Hybrid attention: full attention every 4th layer (0-indexed layers {3, 7, 11, 15, 19, 23, 27, 31}); the remaining 24 layers are GatedDeltaNet linear-attention layers. Pod-side environment pins `transformers==5.15.0` (the earliest repo-available version carrying `qwen3_5`; the VM stays at the repo-locked 4.57.6, where only the tokenizer loads). Constants: [`scripts/issue2329_run.py`](https://github.com/superkaiba/explore-persona-space/blob/653ff2b4873cd00bb1835230840e4f8d66573393/scripts/issue2329_run.py) L99–125.
  - **Model revision (reproducibility gap, stated as such):** the run did not pin a model revision — every `from_pretrained` call resolves `Qwen/Qwen3.5-9B` at `main`, and no artifact records a resolved model commit (the plan repro-card's "TBD sha" was never filled). Two independent local HF caches both resolved `main` to `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, which is the best available evidence of the weights used for the VM-side phases. Because no revision was recorded at generation time and the generating pod has been torn down, the revision resolved pod-side is not provable from the artifacts; a future rerun should pass `revision=` explicitly. `c2022362…` is recovered VM-side evidence, not a recorded pin.
- **Thinking disabled + realized template:** every render/tokenize call threads `template_kwargs = {"enable_thinking": False}` through the #2094 chat-template seam. The realized generation prompt ends `<|im_start|>assistant\n<think>\n\n</think>\n\n` (empty think block); the realized header token ids per role header are frozen in the bank manifest (`generation_headers`; e.g. `assistant` → `[248045, 74455, 198, 248068, 271, 248069, 271]`). A mechanical think-tag audit scans completions for in-body reasoning scaffolds.
- **Bank (reused verbatim; re-tokenized + re-frozen):** the #2162 bank strings are reused byte-verbatim — [`bank2162.py`](https://github.com/superkaiba/explore-persona-space/blob/edb7b973ca16eded79f102cbdbd1be1b85171dc1/src/explore_persona_space/experiments/issue2162/bank2162.py) + `frozen_gen_2162.json` (frozen paddings/translations, generated greedy by the PARENT model Qwen2.5-7B-Instruct — carried as a cross-model provenance caveat on the recency/language cells) — and re-tokenized under the Qwen3.5 tokenizer by [`bank2329.py`](https://github.com/superkaiba/explore-persona-space/blob/653ff2b4873cd00bb1835230840e4f8d66573393/src/explore_persona_space/experiments/issue2329/bank2329.py).
  - Composition: **21 base information types × 3 values × 12 carrier questions + 18 crossed cells = 39 type-cells, 1,404 contexts, 1,404 minimal pairs** (36 pairs per cell). Base types: `instr_format`, `instr_language`, `constraint_knowledge`, `refusal_boundary`, `verbosity`, `reasoning_style`, `persona_prompted`, `demo_format`, `demo_persona`, `language_implied`, `persona_role_header`, `fact_user_name`, `fact_assistant_animal`, `fact_novel_queried`, `list_numeric_detail`, `icl_task_mapping`, `user_expertise`, `user_emotion`, `prior_topic`, `query_content`, `filler_swap`. Crossed cells: 4 conflict cells (instruction vs demonstration disagree: format ×2 directions, persona ×2 directions), 8 recency cells (4 base types × introduction depth {3, 5} user turns before the query), 6 load cells (3 base types × {3, 5} simultaneous other items in the same context). A minimal pair = two contexts identical except one varied span (e.g. "Always format your answers as bullet points." vs "Always answer in flowing paragraph prose, never using lists or bullet points."), with the query and all filler held fixed. Carrier questions are tier-1 WildChat rows plus constructed carriers (tier caveat carried from the parent).
  - Token-identity gate (P0, pre-registered drop+report policy): the parent's minimal-pair property (token sequences identical outside the varied span, at the span-locus registry grain) was re-verified per pair under the 248k-vocab Qwen3.5 tokenizer over the full 1,404-pair bank; pairs broken outside the span would be dropped (floor: ≥30/36 intact per cell, else HALT→repair). Realized bank freeze: 1,404/1,404 pairs intact, 0 dropped (`bank.json` → `token_identity`). Bank/donor seed 2162 (identical donor assignments to the parent).
  - Patch positions, re-pinned per context against the realized thinking-off render: **context-end (ce)** = the last prompt token of the rendered chat template (the trailing token after `</think>\n\n`); **prefix-end (pe)** = the last token of the conversation prefix before the final user turn.
  - **Recorded deviation from the plan's "unchanged pe mechanics" — load-bearing for reading pe-slot coverage:** unlike Qwen2.5, the Qwen3.5 thinking-off template inserts NO default system turn, so a bare single-turn context (no system, no history) has no prefix token at all. All 36 `persona_role_header` contexts and the 12 empty-system `persona_prompted` v2 contexts are therefore flagged `no_prefix`, and the prefix-end slot for pairs touching them is **excluded by construction** (recorded per pair, `pe_excluded_reason`; `bank2329.py` docstring). Every pe-slot figure and table inherits this structural coverage gap — a missing pe cell for those types is a template fact, not a measurement outcome.
- **Computed quantities:**
  - **Context-state bank (v_ce, v_pe):** one right-padded forward pass per context captures the residual-stream hidden state at both slots at ALL 32 layers (positions computed from token ids).
  - **Answer states (V_a):** during every generation (anchors, grid, stage-2), the per-layer hidden states of the model's OWN completion tokens are captured and reduced to a span-mean answer state per layer, stored for all 32 layers. The judge-free activation read layer is **30** (fraction-of-stack remap of the parent's 26/28; recomputable at any layer from the stored all-layer states).
  - **Interventions (mechanistic description):** stage-1 — during prefill of a hooked HF `generate()` call, the hidden state at ONE token position (ce or pe) is **replaced, at all 32 layers**, with a captured state vector from another context (`joint_hooks`, mode="replace"); generation then proceeds normally. Stage-2 — a **difference vector** (donor-context state minus recipient-context state at the slot) is **added** to the hidden activations at the slot position at ONE single layer, scaled by a dose multiplier α ∈ {1, 4}, layers ∈ {9, 15, 16, 19, 23, 25, 30} (fraction-matched remap of the parent's set, with three members shifted ±1 onto full-attention layers 15/19/23 so both layer types are sampled).
  - Injection-exactness gate (12 spot cells): the installed state is re-read under the hook and must match the donor at cosine ≥ 0.999 and norm ratio ∈ [0.995, 1.005] (HALT gate before any spend). Degeneracy guard: realized token-prefix identity is checked against the pre-declared span-locus registry.
- **Conditions / arms (plain English; the 14 planned conditions):**
  - **Steered (donor-value patch):** the recipient context A's slot state is replaced with the SAME pair's context-B state — the state that differs only in the varied information. Primary arm: does the slot state causally carry the varied information?
  - **Shuffled-donor null (same type, value-constrained):** the installed state comes from a different pair of the SAME type-cell — value-cycle-shifted so the donor's B-value ≠ the recipient's B-value by construction, with a seeded carrier derangement — norm-matched per layer to the recipient's own B-state. Controls for generic single-position disruption and norm artifacts.
  - **Cross-type-donor null:** the installed state comes from a DIFFERENT information type (seeded type-derangement over the 39 cells, skipping the recipient's matched-content route family, donor-B value string ≠ recipient-B value string wherever vocabularies intersect), norm-matched. Controls for "any edit moves the answer" with edit-happened held fixed.
  - **Unpatched floor anchor / generate-under-donor ceiling anchor:** unpatched generations from BOTH pair sides (context A = floor, context B = ceiling), K=10 draws per context; these normalize F to "fraction of a full context swap".
  - **Context-end slot / prefix-end slot:** the two patch positions, run as separate cells everywhere.
  - **Route-conflict cells:** contexts where the instructed value and the demonstrated value disagree (format ×2 directions, persona ×2 directions); DV = the balance shift between following the demonstration vs the instruction.
  - **Recency cells (depth 3/5):** the varied information is introduced 3 or 5 user turns before the query (frozen padding turns); base cells count as depth 1.
  - **Load cells (load 3/5):** 3 or 5 other items of information ride the same context alongside the varied one; base cells count as load 1.
  - **Query-content control:** the varied span is the QUERY itself (rig-sensitivity floor — the patch should move the answer topic if the rig works at all). **Filler-swap disruption control:** the varied span is content-free filler; reports no F (disruption DV only) — controls the "any text change disrupts" alternative.
  - **Stage-2 layer-by-dose confirmation (F_act-selected):** single-layer add-mode edits on ≤12 survivor (type-cell × slot) combos, selected pod-side by the judge-free activation score (see Deviations); labeled post-selection/exploratory.
  - **Parent-model comparison (transfer read):** this run's per-type F table against the parent's committed table (`eval_results/issue_2162/f_metrics/`), bank text identical, model/architecture the single varied factor.
  - **Baselines** (worry → test): generic disruption → shuffled-donor null; any-edit confound → cross-type null; rig insensitivity → query-content control; text-change disruption → filler-swap; baseline propensity of each pair side → floor/ceiling anchors both sides. **Sanity checks:** injection-exactness gate, degeneracy guard, think-tag audit, mechanical audits (non-Latin fraction > 0.05, repeated-4-gram fraction > 0.50, empty completion), anchor coherence baseline (median ≥ 80, ≥90% > 60), rule-26 judge pilots at both dispatch points, rule-19 margin-vs-F_beh validation before any margin read.
- **Generation parameters:** grid — temperature 1.0, K=5 draws per (pair × slot × arm), `max_new_tokens=2048`; anchors — temperature 1.0, K=10 draws per context; stage-2 — greedy (temperature 0.0), 1 draw. Seeds: per-draw `seed = 42 + draw`. Planned rollouts: 42,120 stage-1 grid (39 cells × 36 pairs × 2 slots × 3 arms × 5 draws) + 14,040 anchors (1,404 × 10) + ≤12,096 stage-2 (≤12 combos × 7 layers × 2 doses × 36 pairs × 2 arms × 1 draw). Cap-hit policy: per-cell cap-hit fraction (completion length reaching the cap, re-tokenized-length basis) is reported per stage; any cell strictly above 2% re-generates its rollouts at a raised 4096 cap. The realized store therefore mixes base-cap (2048) rows and re-generated 4096 rows; each row carries `max_new_tokens` + `cap_hit` + `cap_hit_basis`, and cap-hit attribution for cross-store comparability stays at the base cap. Two pre-registered follow-up analyses ride this remedy: a raised-cap sufficiency count (among rows regenerated at 4096, what fraction hit 4096, per (cell, value) — [`issue2329_capregen_sufficiency.py`](https://github.com/superkaiba/explore-persona-space/blob/653ff2b4873cd00bb1835230840e4f8d66573393/scripts/issue2329_capregen_sufficiency.py)) and a shipped-vs-cap-hit-excluded restriction analysis of every primary DV adjudicating the pre-registered escalation trigger (sign flip / gate-verdict change / effect leaving its CI / transfer headline moving / 2×2 quadrant change — [`issue2329_capexcl_compare.py`](https://github.com/superkaiba/explore-persona-space/blob/653ff2b4873cd00bb1835230840e4f8d66573393/scripts/issue2329_capexcl_compare.py)).
- **Compute shape:** one 8× H100 RunPod pod ran all generation (bank capture → anchors → grid → stage-2) through a work-conserving claim-file block queue (234 grid blocks + anchor + stage-2 blocks; one bf16 replica per GPU, hooked-generate batch 16, per-block V_a capture + margin teacher-forcing pipelined on the same GPU); a generation-throughput pilot at the anchors-phase entry derived 2× wall fences and a 3× refusal gate. Judging ran off-pod (Anthropic Batch API); analysis ran on a CPU pod (`cpu-bigmem`); dashboards + figures on the VM. Pipeline order (user directive, binding): ALL generation first → at generation-complete, both qualitative dashboards built immediately from raw text + the judge-free activation score, judge waves dispatched in parallel → judged quantities back-filled.
- **Judge instrument:** `claude-sonnet-4-5-20250929`, graded 0–100, reason-then-score (brief rationale, then `{"reasoning": …, "score": <0–100 int>}`), `max_tokens=1024`, N=1 draw per (rollout × rubric) — uncertainty is carried at the pair/cell level by the bootstrap, not by judge re-draws. Rubrics (one behavior per call): per directed value-pair, a **dual rubric** — rubric A scores expression of the A-side value, rubric B the B-side value (core template: "To what extent does the response express {descriptor}? … Score 0-100: 0 = no trace; 50 = partial or mixed; 100 = fully and consistently"); conflict cells score follows-the-instructed vs follows-the-demonstrated value; `query_content` uses an answers-which-question rubric; plus one form-only **coherence** rubric. Dispatch: all production waves through the Batch API (rubric-keyed JudgeCache; sync used only for the in-window anchor-separation slice ≈9.1k calls); ≈212k gross / ≈203k net calls planned. Drop discipline: malformed/refusal/out-of-range returns dropped (never coerced), content drops vs transport losses vs API-level refusals reported separately; draws censored by API-level refusals (`stop_reason == "refusal"`, empty content — a Batch-path transport-conditional class) were re-issued on the synchronous path at the IDENTICAL instrument and merged, the merge licensed by a dual-scored batch-vs-sync parity check on a sample of already-scored items in a fresh cache dir ([`issue2329_transport_parity.py`](https://github.com/superkaiba/explore-persona-space/blob/653ff2b4873cd00bb1835230840e4f8d66573393/scripts/issue2329_transport_parity.py)). Rule-26 pilots gated both dispatch points (≈0.4k draws each at the exact production instrument, fresh pilot cache dirs; PASS ⇔ zero `max_tokens` truncation + per-arm parse-fail < 2%): once before the in-window sync slice, once before the bulk Batch waves.
- **Gates + exclusions (pre-registered):** injection-exactness HALT; degeneracy HALT; anchor-separation early read on a stratified 6-pairs-per-cell slice — the aggregate 60% bar is ADVISORY on this model (per-type separation failure is itself part of the transfer answer), with a catastrophic HALT floor at <25% of sampled cells separable (instrument-broken abort); per-pair anchor-separation exclusion |ceiling − floor| ≥ 0.5 binding at analysis; coherence filter (rubric > 60) on all judge-dependent reads, cells < 50% coherent flagged; mechanical audits as the judge-free draw filter for stage-2 selection; testability floor n ≥ 12 post-exclusion pairs per cell.
- **Analysis pipeline** ([`issue2329_analysis.py`](https://github.com/superkaiba/explore-persona-space/blob/653ff2b4873cd00bb1835230840e4f8d66573393/scripts/issue2329_analysis.py), [`issue2329_mapshift.py`](https://github.com/superkaiba/explore-persona-space/blob/653ff2b4873cd00bb1835230840e4f8d66573393/scripts/issue2329_mapshift.py)): judge scores + stored states → per-pair F values → exclusions → per-cell tables (steered + both nulls + anchors) → pair-clustered bootstrap CIs (B=10,000, seed 21620) → intersection-union signed-rank tests vs both nulls with Holm correction → read probe → 2×2 verdict lattice → mapshift battery (fresh per-layer ridge fits, shift-prediction, 2AFC) → transfer read vs the parent tables.
- **Statistics:**
  - **Unit of analysis / clustering:** the PAIR is the clustering unit everywhere (pair = mean over its K coherent draws; bootstrap resamples pairs). Probe folds at carrier grain (12 groups); mapshift at rollout grain with leave-one-carrier-out folds; transfer read at cell grain.
  - **Families + multiplicity:** three Holm families — P1 role/type (constructional ceiling 31 cells), P2 route (15), P3 dose/position (28) — with **m = the realized number of testable cells in the family at analysis time** (post-exclusion n ≥ 12; the inherited `holm_family_m = len(pvals)` semantics, matching the parent's realized application m = 25/10/26); realized m reported beside the parent's wherever verdicts are compared. Causal-positive per cell ⇔ steered mean clears BOTH nulls on fully disjoint pair-clustered 95% CIs AND the intersection-union signed-rank p = max(p_shuffled, p_crosstype) survives Holm (α = 0.05). The transfer correlation is ONE additional registered test outside the families (α = 0.05). Stage-2 is exploratory (post-selection).
  - **Verdict lattice (per type × slot, inherited verbatim):** untestable-causal (n < 12) / stored-and-used (causal-positive AND probe-positive) / stored-but-unusable (probe-positive only) / used-but-not-decoded (causal-positive only) / absent (neither).
  - **Selection symmetry:** the probe headline is a max-over-32-layers read against a label-permutation band (B=1,000) with the SAME max applied per permutation draw (per-draw re-max; matrix persisted). No max-over-axis selection in the F headline; no frozen-only CI at any selected position.
  - **Power:** MDE ≈ 0.20 in F units at n=27 pairs (σ_d ≈ 0.25 carried cross-model as a Medium-confidence assumption; realized per-cell MDE re-measured at analysis).
- **Deviations from the parent (all model-forced or user-directed; single experimental variable = the model):** transformers 4.57.6 → 5.15.0 pod-side; layer constants 28 → 32, hidden 3584 → 4096; F_act read layer 26 → 30 (fraction-matched); stage-2 layers remapped {8,12,14,16,19,22,26} → {9,15,16,19,23,25,30}; stage-2 selection statistic F_beh → F_act (judge-free — forced by the all-generation-first pipeline order; the parent's selection SHAPE is retained: disjoint pair-clustered CIs vs both nulls + Holm-corrected intersection-union signed-rank on F_act, capped at 12 by descending steered F_act, with the mechanical audits standing in for the judge-scored coherence filter); mapshift banked-parity anchors dropped (parent ridge bundles are 3584-dim, inapplicable at 4096); dashboards built pre-judging from F_act and re-issued with F_beh; gate-3 aggregate demoted to advisory. Parent recipe values otherwise inherited verbatim (Source: #2162).

**Sample training/evaluation data + completions:** (no training data exists — nothing is trained; these are evaluation-bank rows and model completions, quoted verbatim)

- **One minimal pair** (of 1,404; cell `instr_format`, value-pair v1–v2, carrier d1; from the frozen `bank.json`, staged VM-side at `data/issue_2329/dash_dl/issue2329_q35rerun/analysis_tensors/vc_bank/bank.json`):
  - Context A system prompt: `Always format your answers as bullet points.`
  - Context B system prompt: `Always answer in flowing paragraph prose, never using lists or bullet points.`
  - Shared user query (identical both sides): `Give me some tips for keeping a small apartment organized.`
  - Pair metadata: varied span = the system-prompt tokens (common token prefix 4, common suffix 28); the completion is scored by the dual rubric (A: "an answer formatted as bullet points"; B: "an answer written in flowing paragraph prose with no lists or bullets").
- **One floor-anchor completion** (unpatched context A above; drawn from the 14,040-row anchors store, `raw_completions/anchors/`, draw 0, seed 42, temperature 1.0; this row was generated at the raised 4096 cap — the store mixes base-cap 2048 rows with cells re-generated at 4096 under the >2% cap-hit trigger, attributed per row by `max_new_tokens`/`cap_hit_basis`): first 250 chars of 250 tokens —
  > `- Maximize vertical storage by utilizing wall space with floating shelves, tall bookcases, or hanging racks to free up floor area.\n- Invest in multi-functional furniture, such as ottomans with hidden compartments, beds with under-bed storage, or sofa…`
- **One steered-arm patched completion** (same pair, context-end slot: context A generated with its last prompt token's state replaced at all 32 layers by context B's state; drawn by line offset from `raw_completions/grid/shard_instr_format__ce__steered.jsonl`, 180 rows = 36 pairs × 5 draws; draw 0, seed 42, patched position 35): first 350 chars of 463 tokens —
  > `Keeping a small apartment organized requires a strategic approach that maximizes every inch of available space while maintaining a sense of calm and order. Here are several practical tips to help you achieve this:\n\n*   **Maximize Vertical Space**: Install wall-mounted shelves, hanging organizers, and over-the-door racks to utilize tall walls and do…`
- Full tables: the two qualitative dashboards render every pair's context (varied span marked inline A→B), query, and three plainly-labeled answers — [`docs/issue2329_bank_dashboard.html`](https://github.com/superkaiba/explore-persona-space/blob/653ff2b4873cd00bb1835230840e4f8d66573393/docs/issue2329_bank_dashboard.html) and [`docs/issue2329_result0_gallery.html`](https://github.com/superkaiba/explore-persona-space/blob/653ff2b4873cd00bb1835230840e4f8d66573393/docs/issue2329_result0_gallery.html). Raw stores: HF data repo `superkaiba1/explore-persona-space-data`, prefix `issue2329_q35rerun/raw_completions/{anchors,grid,stage2}/` — the realized destination (Hub-verified; the driver's `EPM_2329_DATA_WRITE_REPO` overflow reroute, a #2304 file-cap safeguard, exists but did not engage in this run: it is set nowhere in the dispatch chain, so the default main-repo destination applied).

- **Metrics:**
  - **Fraction-of-swap behavioral score (F_beh)** — PRIMARY causal DV. Per draw, the dual-rubric contrast Δ = (judge_B − judge_A)/100; per pair, F_beh = (Δ̄_patched − Δ̄_floor) / (Δ̄_ceiling − Δ̄_floor), where floor = context-A anchors and ceiling = context-B anchors. Units: fraction of a full context swap (≈0 = patch does nothing, ≈1 = patch moves behavior as far as swapping the whole context text). Chosen because it is an on-policy judged behavior read (the model writes its own completion; measurement-validity default) normalized per pair by both sides' baseline propensities, so cross-type comparisons are not confounded by per-value base rates. Near-zero anchor separations return NaN + a flag (never coerced); the |separation| ≥ 0.5 exclusion removes pairs whose denominator cannot support the read.
  - **Activation fraction-of-swap (F_act)** — the judge-free twin. F = (s·t)/‖t‖² with s = patched-minus-floor span-mean answer-state shift and t = ceiling-minus-floor axis, at read layer 30 (all-32-layer profiles exploratory). The floor is estimated from disjoint halves of the K floor draws, both half-assignments averaged (shared-baseline-inflation fix; the naive shared-baseline estimator is kept record-only). Chosen as the zero-judge companion that (a) lets stage-2 selection and the dashboards run before any judging (pipeline-order directive) and (b) cross-validates F_beh in activation space; the parent measured cell-level ρ(F_act, F_beh) = 0.769 on the steered arm, which is the grounds for using it as the selection statistic.
  - **Teacher-forced positive-vs-negative margin** — SECONDARY continuous companion (dual-DV rule; the judged rate/score can saturate). Per patched or anchored context, mean length-normalized teacher-forced log-probability of a FIXED, judge-built pool of B-side completions minus the same for A-side completions (pools fixed per cell × value-pair across every context ⇒ no selection-on-outcome bias). Validated against F_beh (ρ > 0 required) before any read; never narrated as the construct.
  - **Linear read-probe AUC** — the decodability ("read") axis of the 2×2. Per type × slot × layer: a kernelized L2-regularized logistic probe (linear kernel; Adam lr 0.15, 150 epochs, l2 = 1e-2, batched over perms × layers × folds) classifies which pair side a natural (unpatched) context state came from; held-out AUC under leave-one-carrier-out folds (12 groups; n = 24 contexts per value-pair vs d = 4096 — deliberately under-determined, so probe-negative is narrated as "not linearly decodable at this n/d", never "not encoded"). Cell-level curve = mean over the cell's 3 value-pairs per layer (per-value-pair curves carried alongside); headline = max over 32 layers, read against a carrier-level label-permutation band (B = 1,000) averaged over value-pairs the same way, with the SAME max applied per draw (selection-symmetric). A value-pair-transfer secondary read (train one value-pair, test a value-pair sharing one value) rides along. Chosen over a fit-R² read because the construct is decodability of a discrete value, and chance is calibrated by the permutation band rather than assumed at 0.5.
  - **Coherence rate** — fraction of a cell's draws with form-only coherence rubric score > 60. Gate/filter quantity, not a headline: all judge-dependent reads use coherent draws only; cells < 50% coherent are flagged. Chosen because a patch can destroy fluency, and behavior rubrics are uninterpretable on incoherent text.
  - **Conflict balance shift** — for route-conflict cells: (judge_demonstrated − judge_instructed)/100, floor/ceiling-normalized like F. Units: −1…1 (negative = follows instruction, positive = follows demonstration). Chosen because in a conflict cell "which source wins" is the construct; a single-value rubric cannot express it.
  - **Held-out map R² (context→answer)** — per layer: LOCO-carrier held-out R² of a dof-capped GCV ridge map from context state v_C to span-mean answer state v_A (n_train ≈ 12.9k ≫ d = 4096; primal Gram; λ over the #825 grid, dof cap 0.9·n_tr asserted non-binding; per-fold selected-λ diagnostics reported). Always reported beside the identity+learned-bias baseline and kNN retrieval (standing mapping-baselines rule: R² alone both over- and under-states maps).
  - **2AFC discrimination accuracy** — paired two-alternative forced choice (#2215 conventions: `sim_blocks` similarity, carrier-blocked deranged null): the fraction of pairs where the map-predicted answer state is closer to the true pair member's realized answer state than to the other member's. Chance = 0.5. Chosen because R² can be dominated by shared answer-state structure; 2AFC asks the sharper question of whether the map separates the two minimal-pair answers.
  - **Predicted-vs-realized shift correlation** — per type × layer: cosine/correlation between the map-predicted answer-state shift (fitted map applied to donor-minus-recipient context state) and the realized patched V_a shift; ceiling reference = full-swap shift with disjoint anchor halves; nulls = shuffled-pair assignment (carrier-blocked derangement) + shuffled-map (refit on permuted pairing). Chosen as the Goal-clause-3 read: do maps predict the causal effect of the patch, not merely correlate states.
  - **Transfer correlation (per-type F vs parent)** — Spearman ρ between this run's and the parent's per-(type-cell × slot) steered mean F_beh over P1-family cells surviving exclusion in BOTH runs, pair-clustered bootstrap 95% CI. The registered prediction-8 test; Spearman because the transfer claim is about ordering of types, not calibrated effect sizes across architectures.
  - **Cap-hit fraction** — per cell × stage: fraction of draws whose completion reached the token cap (re-tokenized-length basis, base-cap attribution). Reported per the standing generation-cap rule with the >2% re-generation trigger; a diagnostic for silent truncation censoring, not a behavioral DV.

**Planned conditions (approved manifest, verbatim):**

- Steered (donor-value patch)
- Shuffled-donor null (same type, value-constrained)
- Cross-type-donor null
- Unpatched floor anchor
- Generate-under-donor ceiling anchor
- Context-end slot
- Prefix-end slot
- Route-conflict cells
- Recency cells (depth 3/5)
- Load cells (load 3/5)
- Query-content control
- Filler-swap disruption control
- Stage-2 layer-by-dose confirmation (F_act-selected)
- Parent-model comparison (Qwen2.5-7B, #2162)

**Planned metrics (approved manifest, verbatim):**

- fraction-of-swap behavioral score (F_beh)
- activation fraction-of-swap (F_act)
- teacher-forced positive-vs-negative margin
- linear read-probe AUC
- coherence rate
- conflict balance shift
- held-out map R-squared (context-to-answer)
- 2AFC discrimination accuracy
- predicted-vs-realized shift correlation
- transfer correlation (per-type F vs parent)
- cap-hit fraction

**Code SHAs:** per-artifact reproducibility-card commits, read from the cards themselves (a card recording a dirty tree is excluded and not cited):

- `eval_results/issue_2329/cap_hit/cap_hit_report_anchors_preregen.json` @ `f0255552a7249f18ce2bec5782a447e7ca10832b`

## Results

Planned manifest items not produced as figures:

- Bank dashboard + Result-0 qualitative gallery (`qualitative_dashboards`) — **not run** as a figure: Not a matplotlib figure: this manifest item is delivered as HTML artifacts outside the plotter figure set — docs/issue2329_bank_dashboard.html (bank dashboard, 224 KB) and docs/issue2329_result0_gallery.html (Result-0 qualitative gallery, 13 MB), both present and content-non-empty in the issue-2329 worktree. No PNG view exists to splice into the report body. Delivered as HTML: [Bank dashboard](https://github.com/superkaiba/explore-persona-space/blob/91b22ffd0e564665001a423c9ad5ee680e2b03c0/docs/issue2329_bank_dashboard.html), [Result-0 qualitative gallery](https://github.com/superkaiba/explore-persona-space/blob/91b22ffd0e564665001a423c9ad5ee680e2b03c0/docs/issue2329_result0_gallery.html).

### Per-type fraction-of-swap at each slot

**Methodology**

- Computed from the per-pair F tables (`f_cells`, `null_shuffled_cells`, `null_crosstype_cells`, `anchors`): per type-cell × slot, the mean F_beh over post-exclusion pairs (pair = mean over its K=5 coherent draws) for each of the three arms — steered, shuffled-donor null, cross-type-donor null.
- Exclusions applied: per-pair anchor-separation |ceiling − floor| ≥ 0.5; coherence filter; pe no-prefix exclusions. Post-exclusion n printed per type; cells with n < 12 labeled untestable rather than plotted as zeros.
- Error bars: pair-clustered bootstrap 95% CIs (B = 10,000, seed 21620; resampling pairs within cell). Per-pair points behind the bars (the per-unit companion). One panel per slot (context-end, prefix-end); F_beh in fraction-of-swap units.

- Bars show mean behavioral fraction-of-swap F_beh (unitless; 1 = the effect of a full context swap) per context-feature type, one panel per patch slot (ce = context-end, pe = prefix-end), with three bars per type: steered (donor-value patch), shuffled-donor null, and cross-type-donor null.
- Error bars are 95% pair-clustered bootstrap CIs; n = post-exclusion pairs per type (printed in the x tick labels, 0-36), after the anchor-separation exclusion |ceiling - floor| >= 0.5.
- Per-unit view (hero_ftype_perpair): every surviving pair as one labeled point (6,125 points across both slots and the three arms; pair id in the point label; no aggregation).

![Per-type fraction-of-swap at each slot — aggregate view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/hero_ftype.png)

**Takeaways**

*(Thomas fills in)*

### Read x write 2x2 (probe AUC vs causal F_beh)

**Methodology**

- One point per type × slot: x = the read axis (max-over-32-layers held-out probe AUC, leave-one-carrier-out; the selection-symmetric permutation band, B = 1,000 with per-draw re-max, supplies the probe-positive threshold); y = the write axis (steered mean F_beh from `f_cells`).
- Quadrant labels come from the registered verdict lattice: causal-positive requires disjoint pair-clustered 95% CIs vs BOTH nulls AND the Holm-corrected intersection-union signed-rank; probe-positive requires the max-layer AUC to clear its max-selected permutation band. Cells with post-exclusion n < 12 are labeled untestable-causal.
- n per point = that cell's post-exclusion pair count (printed); no error bars on the scatter itself — the verdicts carry the inferential content and their inputs are shown in `hero_f_by_type` / `probe_layer_curves`.

- One point per (type-cell x slot), n = 73: x = max-over-layers linear read-probe AUC (positive = clears the per-cell within-carrier permutation band), y = steered mean F_beh; color encodes the persisted verdict quadrant: stored-and-used (n=8), stored-but-unusable (n=58), used-but-not-decoded (n=0), absent (n=2), untestable-causal (n=5).
- Per-family realized/ceiling Holm m is printed in the title; 1 cell without probe rows and 1 untestable cell without steered F are omitted.
- Points concentrate near probe AUC = 1.0, so the per-point cell labels overlap heavily in that region.

![Read x write 2x2 (probe AUC vs causal F_beh) — aggregate view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/two_by_two.png)

**Takeaways**

*(Thomas fills in)*

### Per-layer probe AUC curves with permutation bands

**Methodology**

- From `probe.json` + the persisted permutation matrix (`analysis_tensors/probe_perm_matrix/`): per type × slot, the held-out probe AUC at each of the 32 layers (leave-one-carrier-out, 12 carrier groups; kernelized L2 logistic per the shared recipe; the cell curve is the mean over the cell's 3 value-pairs per layer, with the per-value-pair curves carried as the per-unit companion).
- x-axis = fraction-of-stack (layer/31); the 8 full-attention layers {3, 7, 11, 15, 19, 23, 27, 31} are marked (pre-registered architecture annotation — per-layer indices are not comparable to the 28-layer parent).
- The max-selected permutation band's upper bound (97.5% quantile of the per-draw re-maxed B = 1,000 carrier-level label-permutation null) is annotated; per-layer curves are the diagnostic display, the max-over-layers read is the headline statistic.
- n per curve = 24 contexts per value-pair × 3 value-pairs (pre-exclusion; pe curves omit no-prefix contexts).

- Aggregate view (layer_profile): two heatmap panels of leave-one-carrier-out (LOCO, 12 carrier groups) linear probe AUC per layer, each with its own independent y-axis of type-cell rows — 39 rows in the ce panel, 37 in the pe panel (persona_prompted and persona_role_header appear only in the ce panel); x = depth as fraction of the 32-layer stack, color = LOCO AUC (0.3-1.0), dashed verticals = full-attention layers.
- Per-unit views (probe_layer_curves_ce / _pe): one small-multiple panel per type-cell with the macro AUC-vs-depth curve over value-pairs (blue), thin per-value-pair curves/points, and the within-carrier permutation 95% band per layer (grey; B=1000).

![Per-layer probe AUC curves with permutation bands — aggregate view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/layer_profile.png)

**Takeaways**

*(Thomas fills in)*

### Stage-2 layer-by-dose injectability profile

**Methodology**

- From `stage2_cells.jsonl`: for each F_act-selected survivor (type-cell × slot; ≤12, selected pod-side by disjoint pair-clustered CIs vs both nulls + Holm-corrected intersection-union signed-rank on F_act, capped by descending steered F_act, mechanical-audit draw filter), the F_beh of single-layer pair-difference ADD edits at dose α ∈ {1, 4} × layer ∈ {9, 15, 16, 19, 23, 25, 30}, greedy 1 draw, 36 pairs, steered and shuffled-donor arms; plotted as steered minus shuffled-donor null.
- x-axis = fraction-of-stack; full-attention layers marked (the prediction-9 read: full-attention {15, 19, 23} vs adjacent linear layers is a descriptive contrast with pair-clustered CIs).
- Labeled post-selection/exploratory everywhere (selection on F_act; the confirmatory families do not include stage-2). n = post-exclusion pairs per survivor combo.

- Heatmap panels over the 4 F_act-selected survivor cells (icl_task_mapping|ce, instr_language|ce, language_implied|ce, recency_persona_prompted_d3|ce): steered F_beh of the single-layer pair-difference add edit at dose 1 and dose 4, plus the steered minus shuffled-donor-null difference at each dose.
- x = patched layer in {9, 15, 16, 19, 23, 25, 30}, with depth as fraction of the 32-layer stack in parentheses and * marking full-attention layers; the survivor set is post-selection (F_act-selected), so this view is exploratory.
- Per-unit view (layer_profile_stage2_perpair): 182 per-pair stage-2 F_beh points at each survivor's best (layer, dose), steered (blue) vs shuffled-donor null (grey), pair-id labeled.

![Stage-2 layer-by-dose injectability profile — aggregate view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/layer_profile_stage2.png)

**Takeaways**

*(Thomas fills in)*

### Route contrasts and conflict balance

**Methodology**

- From `f_cells.jsonl`, two reads: (1) matched-content route variants side by side — the same information delivered as an instruction (`instr_format`, `instr_language`, `persona_prompted`), a demonstration (`demo_format`, `demo_persona`), an implication (`language_implied`), or a role header (`persona_role_header`) — steered mean F_beh per variant; (2) the 4 conflict cells, plotting the floor/ceiling-normalized balance shift (judge_demonstrated − judge_instructed)/100.
- Error bars: pair-clustered bootstrap 95% CIs; per-pair companion points behind every aggregate. n = post-exclusion pairs per cell.

- Bars show steered mean F_beh for each base type (blue) beside its matched-content route variant or conflict cell (orange), per slot (ce, pe): demo_format, demo_persona, language_implied, persona_role_header, and the four conflict cells (format fwd/rev, persona fwd/rev); error bars are 95% pair-clustered CIs.
- Conflict cells are plotted on the same steered-F_beh axis as the route variants; the manifest's planned floor/ceiling-normalized conflict balance shift ((judge_demo - judge_instr)/100, -1..1) was never computed (no balance-shift code exists in scripts/issue2329_figures.py; grep for balance|judge_demo|judge_instr returns no matches), so that quantity is declared not produced rather than rendered.
- Per-unit view (route_contrasts_perpair): 1,956 per-pair steered F_beh points, base type (blue) vs route variant / conflict (orange), pair-id labeled, shown at full range and in a zoom panel restricted to |F_beh| <= 2.

- Planned but NOT produced: conflict balance shift (judge_demo - judge_instr)/100, -1..1 — quantity never computed; no balance-shift code in scripts/issue2329_figures.py (grep for balance|judge_demo|judge_instr returns no matches); conflict cells are rendered as steered F_beh bars on the shared axis

![Route contrasts and conflict balance — aggregate view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/route_contrasts.png)

**Takeaways**

*(Thomas fills in)*

### Recency and load curves

**Methodology**

- From `f_cells.jsonl`: per crossed base type, mean steered F_beh vs introduction depth {1, 3, 5} (depth 1 = the base cell; depths 3/5 = the recency cells, where the varied information sits 3/5 user turns before the query behind frozen padding turns) and vs load {1, 3, 5} (load 1 = base cell; loads 3/5 = the load cells with 3/5 other simultaneous items).
- Slopes with pair-clustered bootstrap CIs; per-pair points behind the curves. Frozen-text caveat rides these cells: the padding turns/translations were generated by the parent model (byte-verbatim reuse), so they are off-policy text for Qwen3.5 — identical on both pair sides by construction.
- n = post-exclusion pairs per (base type × depth/load) cell.

- Lines show steered mean F_beh vs introduction depth d in {1, 3, 5} (left panel: recency; fact_user_name, instr_format, persona_prompted, prior_topic x slots ce/pe) and vs distractor load l in {1, 3, 5} (right panel: load; fact_assistant_animal, fact_user_name, instr_format x ce/pe); d = l = 1 is the uncrossed base cell.
- Error bars are 95% pair-clustered CIs; the grey band is the shuffled-donor null's 95% CI.
- Per-unit view (recency_load_perpair): 5,724 per-pair steered F_beh trajectory points across recency and load levels (ce solid, pe dashed), shown at full range and in zoom panels restricted to |F_beh| <= 2.

![Recency and load curves — aggregate view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/dose_position.png)

**Takeaways**

*(Thomas fills in)*

### Per-layer context-to-answer map skill with baselines

**Methodology**

- From `mapshift/fresh_fit_diagnostics.json`: per layer (32), the leave-one-carrier-out held-out R² of the fresh ridge map v_C → v_A fit on the bank's anchor answer states at per-draw grain (n_train ≈ 12.9k ≫ d = 4096; primal feature-space Gram with full-Gram fold downdates; GCV λ selection over the #825 grid with the 0.9·n_tr dof cap asserted non-binding; per-fold selected-λ diagnostics carried).
- Plotted beside two mandatory baselines: the identity+learned-bias predictor (v̂ = v_C + b, b = train-fold mean of (v_A − v_C); applicable since input/output share d = 4096) and kNN retrieval P(true target within top-k) at k ∈ {1, 5, 10}, euclidean + cosine, chance = k/n_pool stated.
- x-axis = fraction-of-stack, full-attention layers marked; n_train vs d printed on the figure. Banked parent maps are NOT plotted (dimension-inapplicable, divergence 8).

- Left panel: held-out R-squared (context grain, leave-one-carrier-out) of the dof-capped GCV ridge map v_C -> v_A per layer, x = depth as fraction of the 32-layer stack, alongside the identity and identity + learned-bias baselines; the title states n_train ~ 12,870 > d = 4,096 (well-posed).
- Right panel: kNN retrieval read of the fitted map, P(true target in top-5) among the held-out pool, cosine and euclidean, with the chance level 0.004 drawn; dashed verticals mark full-attention layers in both panels.

![Per-layer context-to-answer map skill with baselines — aggregate view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/mapshift/mapshift_r2.png)

**Takeaways**

*(Thomas fills in)*

### Predicted vs realized patched shift

**Methodology**

- From `mapshift/shift_summary.json` + `shift_cells.jsonl`: per type × layer × map source × patch arm, the correlation (cosine convention of the parent mapshift battery) between the map-predicted answer-state shift — the fitted layer map applied to the donor context state minus applied to the recipient context state — and the realized patched V_a shift measured in the grid.
- Ceiling reference: the full-swap shift computed with DISJOINT anchor halves (shared-baseline-inflation fix). Nulls: shuffled-pair assignment (carrier-blocked derangement) and shuffled-map (map refit on a context-permuted pairing, refit layers {16, 22, 30}).
- Per-cell points behind the per-type × layer summary; n = cells/pairs entering each correlation (printed per panel).

- y = cosine similarity between the map-predicted answer-state shift and the realized patched shift, per layer (x = depth as fraction of the 32-layer stack); left panel restricted to the 8 stored-and-used cells, right panel all 39 cells.
- Series: the bank-fit map summary curve, per-type-cell point markers, the raw context shift with no map applied, and the null patch arms (shuffled / cross-type donors); dashed verticals mark full-attention layers; the per-cell points are the per-unit data behind the summary.

![Predicted vs realized patched shift — aggregate view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/mapshift/mapshift_shift_prediction.png)

**Takeaways**

*(Thomas fills in)*

### 2AFC minimal-pair answer discrimination

**Methodology**

- From `mapshift/dv3_ext.json`: the #2215 paired 2AFC per type × layer — for each minimal pair, the map-predicted answer state (from the fresh bank-fit map at that layer, span pooling) is compared under the `sim_blocks` similarity convention to the two realized answer states; the observed statistic is the fraction of pairs where the prediction sits closer to the TRUE pair member.
- Arms: fresh fitted map per layer, identity-only (v̂ = v_C), and leave-one-out identity+bias; null: the carrier-blocked deranged pairing; chance = 0.5 line drawn. CIs: carrier-clustered bootstrap (#2215 convention).
- n = pairs entering the 2AFC per type × layer (post pe/no-prefix exclusions where applicable).

- y = paired 2AFC accuracy (cosine similarity, span pooling): the fraction of pairs where the map-predicted answer state sits closer to its own context's real answer than to the alternative pair member; x = layer depth as fraction of the 32-layer stack.
- Series with per-layer error bars: map fit on this bank, identity + bias, and identity baselines; the grey band is the carrier-blocked deranged null 95% band around the chance line at 0.5; dashed verticals mark full-attention layers.

![2AFC minimal-pair answer discrimination — aggregate view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/mapshift/dv3_2afc.png)

**Takeaways**

*(Thomas fills in)*

### Transfer read: per-type F on Qwen3.5-9B vs Qwen2.5-7B

**Methodology**

- One point per P1-family type-cell × slot that survives exclusion in BOTH runs: x = the parent's committed steered mean F_beh (`eval_results/issue_2162/f_metrics/f_cells.jsonl`, [pinned](https://github.com/superkaiba/explore-persona-space/blob/2af3e898d523d2ca9033777b06a9235979747409/eval_results/issue_2162/f_metrics/f_cells.jsonl), Qwen2.5-7B-Instruct); y = this run's steered mean F_beh (Qwen3.5-9B). Identity line drawn.
- The summary statistic is the registered prediction-8 test: Spearman ρ with a pair-clustered bootstrap 95% CI, ONE registered test outside the three Holm families (α = 0.05). n of shared cells printed (≤31 by construction; each run's own exclusions shrink it).
- Comparability notes carried on the figure: bank text identical, F definition identical, judge instrument identical; tokenizer/anchors differ, so per-pair exclusion sets differ; realized per-family Holm m of both runs is stated wherever verdict-level comparisons are shown.

- x = per-type mean steered F_beh on the Qwen2.5-7B parent (#2162 committed tables); y = the same quantity on Qwen3.5-9B (this run); one labeled point per shared P1 (type-cell x slot) surviving exclusion in both runs (31 shared units), one panel per slot, with the identity (perfect-transfer) line drawn.
- The suptitle states Spearman rho = 0.831 (p = 7.4e-09), pair-clustered 95% CI [0.583, 0.864].
- Companion view (transfer_verdicts): 3x3 heatmap of causal-verdict transfer (positive / null / untestable-causal), parent verdict vs this run's verdict, over n = 75 shared (cell x slot) units, with per-family realized/ceiling Holm m printed in the title.

![Transfer read: per-type F on Qwen3.5-9B vs Qwen2.5-7B — aggregate view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/transfer_scatter.png)

**Takeaways**

*(Thomas fills in)*

### Diagnostics dump

**Methodology**

- Multi-panel diagnostics from the f_metrics tables + judge outputs, all recipe-level QC quantities: pre-exclusion anchor-separation distributions per type; the F_act-vs-F_beh agreement scatter (cell-level Spearman — the check on the judge-free twin used for stage-2 selection); the margin-vs-F_beh validation scatter (the rule-19 gate that must pass before any margin read); per-cell coherence and cap-hit rates; per-cell token-identity drop counts (the divergence-9 report; realized 0 drops at bank freeze); a length-delta covariate read + length-matched recount (the varied spans differ in token count — len_delta per pair — so a length-matched recount controls completion-length confounds); and null breakouts (cross-type null by donor type; shuffled null by donor value).
- Each panel states its own n (draws or cells); these are instrument diagnostics, not headline DVs.

- Aggregate view (diagnostics): three per-(type-cell x slot) panels — excess incoherence relative to the anchor baseline (judge score <= 60) per arm (steered / shuffled / crosstype), generation cap-hit fraction per arm with the 2% re-generation trigger line, and post-exclusion n in pairs with the separation-survival floor of 12 drawn. Token-identity (divergence-9) drops/repairs: the check ran with zero dropped cells and zero repaired cells (eval_results/issue_2329/f_metrics/transfer.json, div9_flags), so no panel is drawn for it.
- length_matched_diag: top panel — steered F_beh per (type-cell x slot) computed over all post-exclusion pairs (filled circles) vs over the length-matched subset only (|context-length delta| <= 2 tokens; open circles), for the 42 of 75 units with at least one length-matched pair (per-unit n printed as matched/all in the tick labels; matched n = 3-36); the 33 units with no length-matched pairs are omitted, never drawn as zero bars. Bottom panel — per-pair context-length delta (tokens; value-B context minus value-A context) distribution per unit (boxplots; 2,676 steered pairs over 75 units) with the +/-2-token matching threshold drawn as black dashed lines.
- anchor_separation_diag: per-pair anchor separation (ceiling - floor, judge-contrast units; K = 10 draws) per type (1,364 points), with the |separation| >= 0.5 keep threshold and kept/total pair counts printed per type. act_beh_agreement: mean F_act (read at layer 30 of 32) vs mean F_beh per (cell x slot x arm), with per-arm Spearman rho printed (steered 0.747, shuffled 0.529, crosstype 0.599; n = 74-75 screened cells per arm).
- margin_validation: teacher-forced fixed-pool margin shift (patched - floor anchor) vs steered F_beh, per-(cell x slot) means (blue, registered grain, n_cells = 72, rho = 0.451, p = 7e-05) over per-pair companions (grey, n_pairs = 2,508, rho = 0.35); the main panel's y-range is stretched by a few extreme per-pair values and the inset zoom (|margin shift| < 1, |F_beh| < 3) carries the readable view. crosstype_null_by_donor: cross-type-null F_beh split by donor type for the 23 recipient cells whose pooled cross-type 95% CI excludes 0, plus shuffled-null F_beh split by donor value for the ordinal value sets (constraint_knowledge, refusal_boundary).

![Diagnostics dump — aggregate view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/diagnostics.png)

**Takeaways**

*(Thomas fills in)*

## Conclusion and next steps

*(Thomas fills in)*
