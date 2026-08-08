---
title: 'Which kinds of context information are carried at the context vector: a patch-only
  sweep over 21 minimal-pair information types, crossed with route conflict, recency,
  and load'
kind: experiment
tags: []
created_at: '2026-08-07T06:43:49Z'
has_clean_result: false
parent_id: 2094
origin_prompt: 'Help me to plan this issue based on the previous causality experiment
  (2094): Motivation - We''ve found this mapping from context -> answer vector; We
  found it to be mostly persona related; We want to see what kinds of information
  get stored at this context vector. Methodology - Activation patch ONLY the context
  vector for a wide range of contexts which only differ in one aspect: some fact in
  the context (user''s name, assistant''s favorite animal), some instruction (e.g.
  ''answer in bullet points''), ICL example of instruction, persona (prompted + ICL),
  query, what else? - See which affect outputs. Results - Result 1: What does patching
  the context vector affect? To measure the effect on output, I plot: [left for the
  plan]. Settled in the same chat: 11-type base + brainstorm candidates 12-21 (21
  types), linear read-probe companion included, all three deferred axes (conflict,
  recency, capacity/load) folded in, workflow v2 dogfood.'
workflow: v2
backend: runpod
goal: 'On Qwen-2.5-7B-Instruct, determine which kinds of context information are both
  CARRIED at and CAUSALLY USABLE from a single context position by patching ONLY that
  position between minimal-pair contexts that differ in exactly one information type
  — 21 types spanning stated response policy, conditional policy, demonstrated/implied/role-header
  route variants, retrievable stated items, ICL task definition, inferred user model,
  discourse state, and two pre-registered near-zero controls — crossed with three
  secondary axes (instruction-vs-demonstration route conflict, introduction recency
  at conversation depths 1/3/5, and information load 1/3/5) on designated subsets,
  measured per type by the #2094 fraction-of-swap F between the unpatched floor and
  the generate-under-donor ceiling under a PRE-REGISTERED anchor-separation exclusion
  (|ceiling - floor| >= 0.5), against BOTH a norm-matched shuffled-donor null and
  a cross-type-donor null, at BOTH the context-end vector v_C and the prefix-end state,
  under a maximal all-layer full-state Stage-1 patch with a post-selection Stage-2
  layer x dose profile, with a LINEAR read-probe companion per type/state/layer (group-held-out
  folds) so a null separates ''not encoded'' from ''encoded but not causally usable'',
  reported in three separately Holm-corrected pre-registered families.'
relates_to:
- spec-context-as-vector
- spec-prompt-vs-icl
- spec-role-header
---
# Experiment: Which kinds of context information are carried at the context vector — a patch-only sweep over 21 minimal-pair information types, crossed with route conflict, recency, and load
<!-- report-v1 -->

**Detailed writeup:** https://github.com/superkaiba/explore-persona-space/blob/a8636605e9d5685f5044e13ad35a73f993e66b8f/docs/reports/issue_2162_detailed.md

## Motivation

- The mapping line found that the context vector $v_C$ — the last-prompt-token residual state, at the newline before the assistant header — predicts the answer representation well when read passively, and the parent experiment [#2094](https://eps.superkaiba.com/tasks/2094) established a causal converse for one content type: editing $v_C$ in place moves *persona* behavior clear of a shuffled-donor null at context-end only, and only partially — 0.63 of a full context swap under the maximal all-28-layer full-state patch (0.51 under temperature-1.0 K=5 re-sampling; null −0.05 greedy / 0.10 resampled), with prefix-end, second-to-last, and third-to-last slots yielding no null-separated behavior family (plan §2, #2094 body).
- #2094 answered *whether* a single position causally carries context information and *where*; its bank varied persona and query content only, so every claim about "the context vector" rests on two content types. This experiment tests **what**: which of 21 information types — stated response policy, conditional policy, demonstrated/implied/role-header route variants, retrievable stated items, ICL task definition, inferred user model, discourse state, and two pre-registered near-zero controls — transfer under the identical maximal patch (task Goal, `body.md`).
- A null for a type is ambiguous between *not encoded there* and *encoded but not causally usable when injected* (#2094 measured the edit-to-response map far from linear: log-log dose slope 0.00–0.06 vs 1.0 for a linear map). A **linear read probe** on the same states is run alongside the patches so the read × write 2×2 can separate the two: we test whether a probe recovers a type's value at high AUC while the causal effect stays at the null (stored-but-unusable) vs probe-at-chance (absent).
- Pre-registered predictions (plan §3, stated as hypotheses to test): (1) policy types (`persona_prompted`, `demo_persona`, `instr_format`, `demo_format`, `instr_language`, `verbosity`) transfer at context-end with F ≥ 0.2 clearing both nulls — the task-vector / function-vector prior (arXiv 2310.15916, 2310.15213); (2) retrievable items (`fact_user_name`, `fact_assistant_animal`, `fact_novel_queried`, `list_numeric_detail`) do not — attribute extraction attends to the entity's own positions (arXiv 2304.14767), which a last-position patch leaves untouched; (3) `icl_task_mapping` is the sharpest rig test — replication of task-vector transfer at context-end, or a failure that localizes the published effect to its `→`-separator extraction point; (4) `query_content` ≈ 0 (replicating #2094's matched-prefix null) and `filler_swap` shows generic disruption only; (5) prefix-end carries prefix-only types if it carries anything; (6) conflict cells ask whether the instructed or the demonstrated route wins at $v_C$; (7) recency cells ask whether F decays with introduction depth (recency domination) or stays flat (running summary), and load cells whether the target's F decays as same-type pieces accumulate (saturation).
- Direct read on open questions 1.1 (`q:spec-context-as-vector` — can a context be treated as a vector or a compact code?) and, through the instructed-vs-demonstrated arms, 1.3 (`q:spec-prompt-vs-icl`).

## TLDR

*(Thomas fills in)*

## Methodology (shared)

- **Model:** `Qwen/Qwen2.5-7B-Instruct`, bf16, 28 layers, hidden size 3584; frozen — no training, no adapters anywhere in this task (Source: reproducibility card in `eval_results/issue_2162/margin/upload_done.json`; plan §0). torch `2.8.0+cu128`, transformers `4.57.6` (same card).

- **Design (minimal-pair patch-only sweep):** per information type, minimal pairs of contexts $(A, B)$ that are token-identical except the span carrying the type's value. The hidden state at ONE position is copied from $B$'s forward pass into $A$'s at all 28 layers (full-state replace, hooked HF `generate()` at prefill — the #2094 rig: `src/explore_persona_space/experiments/issue2094/hooks.py` `PositionEditHook`/`joint_hooks`, reused unmodified (one commit ever touched that file)), and the model generates. The type-specific **fraction-of-swap** F locates the patched behavior between the unpatched-$A$ floor and the generate-under-$B$ ceiling (0 = floor, 1 = ceiling).
  - **Bank:** 21 base types × 3 genuinely distinct values per type (3 directed value-pairs on a registered cycle v1→v2, v2→v3, v3→v1) × 12 carriers = 36 pairs per cell, plus 18 crossed cells (4 route-conflict, 8 recency at depths 3/5, 6 load at loads 3/5) = **39 type-cells, 1,404 contexts, 1,404 pairs** (verified against the frozen bank: `contexts` and `pairs` counts in `bank.json`). Carriers are allocated per type class: policy-manifesting types get 3 hand-written direct probes + 9 real WildChat neutral carriers (in-git `wildchat_random_v1.json`, reserved slice [250:400], seeded selection, seed 2162); item / conditional-policy / discourse types get 12 type-ENGAGING carriers (hand-written or domain-screened), because a neutral carrier's floor and ceiling both omit the item by construction (plan §4.1). Per-carrier provenance is recorded in the bank.
  - **Data-realism tier (declared):** hybrid — tier-1 carriers and filler (real WildChat queries) with tier-4 CONSTRUCTED contrasts (the varied spans; minimal pairs do not exist in the wild). All padding assistant replies and the two language types' non-English text are generated greedy by the frozen base model itself and frozen into the bank — no third-party-LLM-written text enters any context (plan §4.1). Scope caveats carried per plan §6 pre-commitments: the two language types are measured on model-translated ("translationese") text, and the policy-vs-item contrast compares types measured on different carrier regimes (conservative for prediction 2).
  - **Bank freeze:** the realized bank is `issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json` on the HF data repo (`superkaiba1/explore-persona-space-data`; note: this is the REALIZED path — the plan's `off_pod_phases` recipes name a top-level `issue2162_ctxinfo/bank.json` that does not exist). Bank seed 2162.
  - **Worked verbatim example — one minimal pair** (`fact_user_name::v1-v2::e1`, verbatim from `bank.json`; the varied span is the name in the prior user turn, everything else token-identical):
    - Context A (`fact_user_name::v1::e1`): system — none; history — user: "By the way, my name is Alice." / assistant: "Got it, thanks — I'll keep that in mind. What would you like to talk about?"; final user query (carrier `e1`, hand-written engaging probe): "What's my name again?"
    - Context B (`fact_user_name::v2::e1`): identical except the prior user turn reads "By the way, my name is Bob."
    - Value set for this type: v1 = Alice, v2 = Bob, v3 = Priya (all three body-named).
  - **Worked verbatim example — one class-P context** (`instr_format::v1::n3`, verbatim from `bank.json`; system-message span locus, real WildChat neutral carrier): system — "Always format your answers as bullet points."; history — empty; user — "write a story about Spider-Man robbing a bank".
  - **Slots (both mapping arms):** every cell runs at **context-end** (`ce`, $v_C$ = last context token, generation prompt included) and at **prefix-end** (`pe`, last prefix token = the position before the final user turn, via `bank.prefix_end_index_multi`). Two cells are pre-declared degenerate at prefix-end by construction (`query_content`, `persona_role_header` — A and B share the entire prefix; `degenerate_at_pe_cells` in `bank.json`) — generated, flagged, excluded from aggregates. `persona_role_header`'s varied span is immediately adjacent to context-end (a designed property, pre-declared in plan §4.1: role-header information is definitionally boundary content, and the type is kept as the boundary-content anchor of the route axis).
  - **Arms (3 per cell × slot):** (1) **steered** — the pair's own donor state $V(B)$; (2) **shuffled-donor null** — a donor state from a DIFFERENT pair of the same type-cell under a seeded VALUE-CONSTRAINED permutation (donor-B-value ≠ recipient-B-value as a hard constraint, realized as the value-cycle shift; seeded carrier shuffle; norm-matched per layer to the recipient's own B-state via `bank.norm_match`) — a real same-type state of the same size carrying different content; (3) **cross-type-donor null** — a donor from a DIFFERENT information type (matched carrier where the pool admits, seeded fallback otherwise; matched-content route families excluded; norm-matched; donor type + pair id recorded per row) — holds "an edit of this size happened" fixed while removing the content match. Anchors per context: **unpatched floor** and **generate-under-donor ceiling** (baseline propensity measured on both pair sides). Donor assignments are frozen in `bank.json` (`donor_assignment`, seed 2162).
  - **Stage 1 (primary, confirmatory):** full-state replace at ALL 28 layers at the slot — the maximal single-position intervention (Source: plan §4.2; #2094's largest clean effect at this exact cell). Grid decoding: temperature 1.0, K=5 draws per pair × arm (`GRID_TEMPERATURE = 1.0`, `GRID_DRAWS = 5` in `scripts/issue2162_run.py` @ `b4ab6ed5f96216566b78b090f432d763246997b0`, the grid/anchors-phase commit); anchors K=10 at temperature 1.0 (`ANCHOR_DRAWS = 10`).
  - **Stage 2 (post-selection, exploratory):** pair-difference edits (`mode="add"`, Δ = V(B) − V(A), applied at a SINGLE layer; **dose = the alpha multiplier on the added delta**) at doses {1, 4} × layers {8, 12, 14, 16, 19, 22, 26}, 1 greedy draw per pair, steered + shuffled-donor arms, run ONLY on Stage-1 survivor (cell × slot) combos — survivors = combos whose steered F_beh clears BOTH nulls on fully disjoint 95% CIs AND survives the family Holm IUT, capped at 12 by descending steered F_beh (pre-registered rule, plan §6). (Source: `STAGE2_LAYERS`/`STAGE2_DOSES`/`STAGE2_TEMPERATURE = 0.0` in `scripts/issue2162_stage2.py` @ `ec113fdc05daecbfa5e04a7740552ed1093f079b`; the dose semantics are restated in `stage2_results.json` `scope_notes`.) **Realized:** the Stage-1 selection yielded **5 survivor cells** against the cap of 12, so stage-2 ran 5 cells × 14 (layer × dose) variants × 36 pairs × 2 arms × 1 draw = **5,040 rollouts** in **140 shards** (`stage2_results.json` `n_stage2_shards`; the plan sized the stage-2 grid at its ≤12,096 ceiling).

- **Realized run counts (denominators, not findings):**
  - Stage-1 grid: **234 blocks / 42,120 rollouts** (39 cells × 2 slots × 3 arms; 36 pairs × 5 draws per block) — 234/234 blocks completed, totals summed from the eight per-worker `grid_done_w*.json` manifests on the HF data repo; 234 grid text shards under `issue2162_ctxinfo/raw_completions/grid/`.
  - Anchors: **14,040 rollouts** (1,404 contexts × K=10) in **16 shards** (8 gate-slice + 8 rest, per-worker).
  - Cap-hit telemetry: `max_new_tokens = 2048` from the start; realized cap-hit **42 of 42,120 grid rows (0.0997%)**, far below the pre-registered 2%-per-cell re-generation trigger, so no re-generation fired (per-row `cap_hit` fields in the grid shards; aggregate in the committed `eval_results/issue_2162/f_metrics/grid_caphit_aggregate.json` @ `9c7a67d249…`, re-derived from the full 42,120-row `judge/audits/grid.audit.jsonl` whose sha256 it records). Cap-hit basis is a recorded deviation — see § Realized deviations.
  - Teacher-forced margin leg: **234/234 blocks, 235 uploaded shards** plus the anchor margin pass (`margin_blocks_done`/`margin_blocks_expected`/`uploaded_prefixes` in `upload_done.json`; `margin_anchors_done: true`). The leg ran DEFERRED (`deferred_leg: true`) — after the judged anchor waves landed, because its fixed pools are judge-filtered — on pod-2162-a, 1 worker / 1× H100, concurrent with the CPU-only P7 analysis chain on the same pod.
  - Judging: **192,960 judge calls dispatched over 168 production waves** (`judge/judge_summary.json` `total_judge_calls_dispatched` / `n_waves`; the plan's telemetry estimate was ≈211k gross / ≈202k net). Stage-2 judging (P9) was **15,120 calls — 5,040 coherence + 3 × 3,360 behavior — all four waves scored at full denominator with residual transport 0** (P9 completion marker; wave rows in `judge_summary.json`). The plan sized P9 at ≈36.3k calls against the 12,096-rollout stage-2 ceiling; the realized 15,120 follows from the 5-survivor selection above.
  - P6 judge outputs: 748 files / 223 MB persisted on HF (`issue2162_ctxinfo/raw_completions/judge_raw/`; run marker). In git under `eval_results/issue_2162/judge/` at the branch pin (`20fcef9c28…`): `judge_summary.json`, `pools.json`, 7 gate reports, 70 raw files, the `anchors` + `stage2` audits (the grid audit is untracked), and 1 of 168 items files (`coherence.anchors`); the per-wave scores/items corpus (336 scores files on disk) is NOT committed — it lives on the HF data repo under `raw_completions/judge_raw/` per the wave-output convention.

- **Computed quantities:**
  - **State bank:** per context × layer, $v_{ce}$ (last context token) and $v_{pe}$ (last prefix token) captured in one right-padded forward per context, positions read off token ids (BPE-seam rule); 1,404 contexts × 2 slots × 28 layers → `issue2162_ctxinfo/analysis_tensors/vc_bank/vc_bank.pt`. These states feed the patches AND the read probe (probe inputs are the natural, unpatched states — the encoding question is about natural states).
  - **Answer state $V_a$:** span-mean over the model's OWN completion tokens, captured by a hooked teacher-forced re-forward of each rollout; read layer 26 for F_act (the #2094 primary read layer; all-28 profiles persisted). NOTE (recorded deviation): only the span-mean pooling is persisted — the parent's tail-inclusive twin pooling is not (see § Realized deviations).
  - **Donor payloads:** norm-matched per layer to the recipient's own B-state norm (matching the parent's realized replace-cell null regime; plan §4.2).

- **Predictors / conditions:** 13 registered conditions (the `planned_manifest.json` condition set): steered (donor-value patch) / shuffled-donor null / cross-type-donor null / unpatched floor anchor / generate-under-donor ceiling anchor / context-end slot / prefix-end slot / route-conflict cells / recency cells (depth 3/5) / load cells (load 3/5) / query-content control / filler-swap disruption control / stage-2 layer-by-dose confirmation.
  - **Baselines** (each with the worry it addresses): one worry is that ANY same-magnitude edit at the slot disrupts behavior toward the donor rubric — test: the norm-matched shuffled-donor null (same type, different value). A second worry is that an edit carrying any *real content* moves behavior regardless of content match — test: the cross-type-donor null (different type, matched carrier where possible, route families excluded). A third worry is that the rig manufactures effects out of query changes alone — test: the `query_content` control (pre-registered ≈ 0, #2094's matched-prefix setting). A fourth worry is that swapping any span disrupts generation generically — test: `filler_swap` (length-matched neutral filler; no ceiling exists by construction, so it reports disruption only, never F).
  - **Sanity checks:** injection-exactness gate (12 spot cells re-forwarded with the hook armed; installed state == intended donor state at the intended row/position/layer and nowhere else; cosine ≥ 0.999, norm ratio ∈ [0.995, 1.005]) — realized PASS, 0 spots failed (`injection_gate_report.json` under `analysis_tensors/vc_bank/`; `injection_gate_passed: true` in `upload_done.json`). Span-locus degeneracy guard (per pair × slot, realized state identity checked against the bank's registered span locus; the two pre-declared degenerate cells assert near-identity at pe, every other pair asserts distinctness at both slots) — realized PASS, 0 violations (`degeneracy_report.json`; one pre-production P1 relaunch occurred after an rc=23 guard-calibration fix, recorded in the run-launch markers). Mechanical output audits per arm (non-Latin-script intrusion, max-repeated-4-gram, empty output).

- **Judge instrument (constants):** judge `claude-sonnet-4-5-20250929`; graded 0–100 reason-then-score; `max_tokens = 1024` (every rubric here is single-rationale); **N = 1 judge draw per rubric per rollout** (inherited from #2094 — uncertainty is carried at pair/cell level by the pair-clustered bootstrap over K=5 draws × ≥12 pairs; flagged in plan §12); drop-never-coerce with per-arm content-drop / transport-loss / api-refusal splits; rubric-keyed JudgeCache with per-rubric `cache_dir` partitions; one behavior per call. (Source: `scripts/issue2162_judge.py` + the instrument block in `judge_summary.json`; judge temperature is not threaded — Anthropic API default, recorded in the same block.)
  - **Rubric shape (verbatim template, `bank2162._value_rubric`):** "To what extent does the response express {descriptor}? Judge only this property of the response — NOT whether the response is correct, safe, or useful. Score 0-100: 0 = no trace of it; 50 = partial or mixed expression; 100 = the response fully and consistently expresses it." Each pair is judged under BOTH its A-value and B-value descriptors (the dual-rubric contrast). Conflict cells use (follows-the-instructed-value, follows-the-demonstrated-value) descriptor pairs. Coherence uses the #2094 form-only rubric verbatim ("Rate the COHERENCE of the text on a 0-100 scale. This is a FORM-ONLY judgment … EXPLICITLY IGNORE correctness, safety, relevance, completeness, style, and length …", frozen in `bank.json`).
  - **Routing:** through `eval/judge_dispatch.dispatch_judge_items` — N items ≥ the effective threshold route to the Anthropic Message Batches API, N below it route to sync `api_dispatch` fan-out (`effective_threshold = max(1, threshold_base × otpm / 400k)`, `DEFAULT_THRESHOLD_BASE = 2000`). See § Realized deviations for the realized mixed routing.
  - **Pilot gate (rule 26):** satisfied by the committed gate-5 report at exactly the path the judge code reads — 410 draws spanning every rubric family (coherence 200 / value-rubric 120 / query-rubric 90) at the production instrument (same judge model, `max_tokens=1024`, N=1), **zero failures, zero truncation `stop_reason`s**, plus a 6-item forced-batch request-shape probe (all `end_turn`) — `eval_results/issue_2162/judge/gates/pilot_gate_report.json` (`passed: true`, 82 rubrics total). No new pilot was run for the production waves and nothing was bypassed.
  - **Worked example — one completion that entered judging** (verbatim; floor-anchor row, i.e. the UNPATCHED model under context A — its judge scores are deliberately not shown here, per the findings-blind authoring contract): context `fact_user_name::v1::e2` (history as in the worked pair above; final user query "Sign your answer with a greeting to me by name."), draw 0, temperature 1.0, seed 42 → completion: "Hello Alice! Nice to meet you. How can I assist you today?" (15 completion tokens, `cap_hit: false`; line 151 of `issue2162_ctxinfo/raw_completions/anchors/anchors_gate_w0.jsonl`). This row was judged under the coherence rubric and the pair's two value-descriptor rubrics.

- **Statistical protocol (pre-registered; plan §6):**
  - **Anchor-separation exclusion:** drop any pair with |ceiling − floor| < 0.5 (judge-contrast units, max 2) from ALL confirmatory F reads; threshold FIXED before the run; pre-exclusion counts reported per type. Rationale: #2094's raw 0.85–2.39 headline collapsed to ≤0.13 once weak denominators were excluded post-hoc; this task pre-registers the bar.
  - **Survival floor:** a cell needs ≥ 12 surviving pairs for its causal test (exact signed-rank attainability at family-corrected α); a sub-floor cell carries the explicit `untestable-causal` label in the 2×2 — never rendered as a causal verdict (`SURVIVAL_FLOOR = 12` in `scripts/issue2162_analysis.py`).
  - **Registered test per comparison:** intersection-union pair — p = max(p_shuffled, p_crosstype) from exact two-sided Wilcoxon signed-rank over per-pair paired differences (pair mean F over K=5 coherent draws, steered − null), Holm-corrected WITHIN family (α = 0.05); "separates" additionally requires fully disjoint pair-clustered bootstrap 95% intervals against BOTH nulls with steered above (B = 10,000, seed 21620, `bootstrap_family_means_batched` — the #2094 batched index-GEMM).
  - **Families:** P1 — role (non-route-variant base types × 2 slots); P2 — route (route variants + conflict cells × 2 slots); P3 — dose/position (recency + load cells × 2 slots); S — exploratory (the stage-2 grid; no claims without a confirmation round). Pre-registered m = 31 / 15 / 28 after the plan-time constructional exclusions. **Realized m = 25 / 10 / 26** (`families` in `eval_results/issue_2162/f_metrics/stats.json`): the analysis enters a comparison into its Holm family only if the cell is testable (post-exclusion n ≥ 12, `holm_family_m = len(pvals)` in `scripts/issue2162_analysis.py`), and the pre-registered anchor-separation exclusion left more sub-floor cells than the plan's ≥75%-survival assumption projected — per-family shortfalls P1 −6, P2 −5, P3 −2. Sub-floor cells are labeled `untestable-causal`, with probe AUC still reported. NOTE: whether the three families exactly partition the 76 (cell, slot) combos is NOT verified here — the crossed axes run on designated subsets by design — so no "expected vs realized exclusion count" arithmetic is drawn across families; the per-family realized m values above are the denominators every Holm read uses.
  - **MDE (design-time arithmetic, not a measured value):** at σ_d ≈ 0.25 (grounded on #2094's fu1 confirmation at the identical K=5 temp-1.0 shape) the single-test MDE ≈ 1.02/√n — ≈ 0.20 at the expected n ≈ 27 survivors, ≈ 0.30 at the floor n = 12; the plan raised pairs-per-cell 24 → 36 specifically to keep the MDE at the smallest effect #2094 measured cleanly. The registered verdict is a CONJUNCTION (Holm-IUT AND disjoint CIs against both nulls), so the realized-MDE report is annotated against the full conjunction; a null at an underpowered cell is narrated as underpowered, never as absence.
  - **Selection symmetry:** the Stage-1 F headline has NO max-over-axis selection (one intervention variant per cell, every cell reported). The probe verdict uses max-over-28-layers AUC, so its null band is the max-selected label-permutation band — B = 1,000 label permutations shuffled within carrier groups, each draw re-maxed over layers, threshold at the 97.5th percentile; the per-draw × per-layer AUC matrix is persisted (`issue2162_ctxinfo/analysis_tensors/probe_perm_matrix/`) so the band is recomputable. Stage-2 re-measures argmax-selected cells and is labeled post-selection everywhere.
  - **Group-held-out probe folds (OOD rule):** primary leave-one-carrier-out (12 carrier groups); secondary value-pair-transfer (train on 2 value-pairs, test the third; confirmatory only for the registered polar types, exploratory for non-polar types). F itself is not a held-out predictive DV (no fold). Mapping-baselines pair: N/A — no representation map is fitted anywhere in this task (recorded as inapplicable, not skipped): patches install existing states; the probe is a classifier, not a v_X→v_Y map.
  - **Coherence gate:** one form-only graded call per rollout; coherent := score > 60; all reported quantities over coherent draws only; per-cell n_coherent/n_total; cells < 50% coherent marked, never suppressed. Anchor coherence-baseline sanity (median ≥ 80 and ≥ 90% of draws > 60) gated the behavior-judging spend — realized PASS (`judge/gates/coherence_baseline_gate.json`).
  - **Anchor-separation early gate (spend-protection, plan §7 gate 3):** a stratified 6-pairs-per-cell anchor slice over the 38 non-filler cells, generated first and judged SYNC while the remaining anchors generated; PASS ⇔ ≥ 60% of cells with ≥ 4/6 pairs at |sep| ≥ 0.5. Realized PASS: 38 cells, fraction passing 0.658 — i.e. **13 of 38 cells failed the per-cell screen and were FLAGGED AND STILL RUN**, per the pre-registered rule (a per-type failure below the global bar is the designed read — its exclusion shows up in the pre-exclusion counts; `judge/gates/separation_gate_report.json`).
  - **Generation-throughput pilot (plan §7):** one production-shape batched block timed through the production entrypoint at P3 entry — realized 180 rollouts in 216.3 s (1.20 s/rollout at `gen_batch=16`, 8 workers), projecting a 1.76 h pod wall against the 3.7 h plan row; poll fence set at 2× = 3.52 h; `sweep_allowed: true` (`eval_results/issue_2162/gates/pilot_gate_report.json`).

- **Compute (realized):** primary pod `pod-2162`, 8× H100 (RunPod, `backend: runpod` pinned) — bank/state capture, anchors, the 42,120-rollout stage-1 grid, and incremental uploads; the 234 grid blocks were pulled from a shared work-conserving claim-file queue by 8 CVD-pinned workers (worker count auto-derived from `nvidia-smi -L`; `scripts/issue2162_dispatch.sh`). Judging ran off-pod (Batch API + sync fan-out from the VM). P7 analysis + the deferred margin leg ran on `pod-2162-a` (1× H100 `eval` — a recorded venue deviation, below). Stage-2 ran on `pod-2162-s2`, 4× H100 at code `ec113fdc05…` (launch marker). Seeds: bank/pair/donor 2162; generation `seed_base` 42 (per-draw `torch.manual_seed(seed_base + i)`); bootstrap 21620; probe folds/permutations 21621 (Source: reproducibility card + script constants).

- **Realized deviations from the plan (each confirmed from artifacts or markers; the plan's version is stated where it differs):**
  1. **P7 compute venue.** Plan §9 routes the CPU-only P7 analysis phase (F tables, exclusion, stats, probe) to a RunPod `cpu-bigmem` pod. That flavor was unavailable at dispatch: two consecutive provisions in EU-RO-1 created pods that wedged RUNNING-with-no-port past the bring-up window (the documented #770/#1667 wedge class; both terminated as never-ran), and a datacenter sweep across EU-SE-1, EU-NL-1, CA-MTL-4, EU-CZ-1, and EUR-IS-2 returned no capacity. The shared VM was excluded ON MEASUREMENT, not preference: ~25 GB of 125 GB RAM available at load average ~22.7–36.5 across ~15 concurrent sessions, with `earlyoom` configured to preferentially kill python workloads. P7 therefore ran on `pod-2162-a`, a 1× H100 `eval`-intent pod whose host exposed 224 cores / ~1.9 TB RAM (the GPU idle through the CPU phase, then reused for the deferred margin leg rather than a fresh provision). (Source: the `P7-dispatch venue=DEVIATION` progress marker on the task.)
  2. **Judge routing (mixed, not all-Batch).** Plan §9 states "ALL production waves run Batch API". Realized: MIXED, on both wave classes. Nine dispatches routed Batch — three coherence (dispatch N = 39,793 grid, 14,040 anchors, 5,039 stage-2) and six behavior (N = 5,037 ×2, 5,040, 5,760 ×3) — so the realized split is NOT "coherence batches, behavior syncs". The per-rubric behavior waves span 30–5,760 items (165 non-coherence waves in `judge_summary.json`, 36 of them above 720). The sync-routed dispatches split into two populations (evidence: `eval_results/issue_2162/judge/routing_evidence.json`, committed at `434c84f5aec9453ab85845445a812913a8c6434d`, which records per source log the batch/sync dispatch N values split by the base threshold, the observed base/otpm/effective values, the assumed-vs-probed `otpm_source_correspondence` block, the `causal_floor` block, and each log's sha256; counts are per dispatch DECISION — each wave also logs a paired `api_dispatch` route line, so a raw `path=sync` line grep doubles them): most (156 of 171 sync dispatches) sit BELOW the 2,000-item base threshold and route sync on size alone, while the 15 at or above it (anchors N = 2,000/2,016/2,020; grid N = 2,142–3,036; stage-2 N = 3,359/3,359/3,360) routed sync under a probe-raised effective threshold. The split is fully determined by wave SIZE. `eval/judge_dispatch.py` (~line 1629) SKIPS the OTPM probe when `n_items >= threshold_base * 2` (= 4,000): a wave at or above 4,000 items therefore keeps the assumed `otpm` of 400,000, giving `effective_threshold = max(1, threshold_base × otpm/400k) = 2,000` with `DEFAULT_THRESHOLD_BASE = 2000`, and routes Batch; a wave below 4,000 runs the probe, gets `otpm = 2,000,000` and effective = 10,000, and routes sync — which is why waves above the 2,000-item base still routed sync. The realized data separates exactly on that boundary (`causal_floor`): minimum Batch-routed N = 5,037 against maximum sync-routed N = 3,360, so all 9 Batch dispatches sit at or above 4,000 and every sync dispatch below it. The assumed-vs-probed correspondence (the assumed-`otpm` decision set equals the Batch decision set in all three logs; assumed/probed anchors 1/88, grid 7/80, stage-2 1/3) is downstream of that single size test, not an independent cause. Correctness is unaffected (same instrument, same drop discipline); the imprecise part is the plan's blanket claim. Cost consequence: sync calls bill at standard (≈2× batch) pricing, so realized judge cost exceeds the plan's ≈$620 batch-pricing telemetry estimate (telemetry only; no dollar caps).
  3. **Stage-2 launch shape.** Plan §10's launch line prescribes `uv run python scripts/issue2162_stage2.py --phase all`, whose defaults are `--worker-index 0 --num-workers 1` — on the 4× pod that would have used ONE GPU. The run instead used the dispatcher's `stage2` fan-out (`bash scripts/issue2162_dispatch.sh stage2`), which derives worker count from `nvidia-smi -L` and pins one process per GPU; realized `num_workers = 4`, confirmed in the dispatch log (launch marker for `pod-2162-s2`).
  4. **Three recorded implementation deviations** (`plan_deviations` in `eval_results/issue_2162/margin/upload_done.json`, verbatim substance): (a) the per-block margin teacher-forced pass runs as a SEPARATE batched hooked pass beside the $V_a$ pass — rollout rows need hidden states, pool rows need logits — where the plan's phrasing implies one fused forward; (b) cap-hit telemetry is derived from the re-tokenized completion length (`cap_hit_basis = retokenized_completion_len >= max_new_tokens`) rather than a native generation signal, because the batched generate returns decoded text only; (c) the $V_a$ store persists span-mean pooling ONLY — the parent issue's tail-inclusive twin pooling is not persisted.
  5. **Stage-2 scope limit** (`scope_notes` in `stage2_results.json`): stage-2 captures no $V_a$ and runs no margin pass — the judge-scored F_beh IS the stage-2 read (plan §4.2) — so the plan's `phase_outputs.P8` $V_a$-shard entry is deliberately not produced; and dose is the alpha multiplier on the added pair-difference delta (`mode=add`, Δ = V(B) − V(A), single layer).
  6. **Realized Holm family sizes** 25 / 10 / 26 against the pre-registered 31 / 15 / 28 — see the Families bullet above for the mechanics (the pre-registered exclusion + survival floor removed more cells than the plan's survival assumption projected; per-family shortfalls stated; family-partition arithmetic across the 76 combos deliberately not asserted).
  7. **Judge pilot** — no fresh pilot wave: the committed gate-5 report satisfied rule 26 at the exact path the code reads (410 draws, zero failures, zero truncation; details in the Judge instrument bullet). Nothing was bypassed.
  8. **Prompt caching was inert on the judge path** — `cache_control` was attached to the shared rubric block, but these rubrics are ~60 tokens against the model's 1024-token minimum cacheable prefix, so no cache savings materialized (the dispatcher logs this inertness explicitly; `eval/judge_dispatch.py`).
  9. **Realized bank path** — the frozen bank + state bank live under `issue2162_ctxinfo/analysis_tensors/vc_bank/` (with `bank.json` inside), not the plan-declared top-level `issue2162_ctxinfo/bank.json`; downstream staging (P7/P8) used the realized path.

- **Hyperparameter table (complete; every value from ground truth):**

  | Hyperparameter | Value | Source |
  |---|---|---|
  | Model | `Qwen/Qwen2.5-7B-Instruct` (bf16, 28 layers, H=3584) | repro card, `upload_done.json`; `MODEL_ID`/`HIDDEN_FULL`/`N_MODEL_LAYERS_FULL`, `issue2162_run.py` @ `b4ab6ed5f9…` |
  | Training | none (frozen model; no adapters) | plan §0; repro card |
  | torch / transformers | 2.8.0+cu128 / 4.57.6 | repro card |
  | Types × values × carriers | 21 × 3 × 12 (+18 crossed cells; 39 cells, 1,404 contexts, 1,404 pairs) | `bank.json` (counts verified); plan §4.1 |
  | Pairs per cell | 36 (3 directed value-pairs × 12 carriers) | plan §6 MDE decision; `bank.json` |
  | Stage-1 intervention | full-state replace, all 28 layers, one slot | plan §4.2; `joint_hooks(model, list(layers))` over all 28 layers, `issue2162_run.py` @ `b4ab6ed5f9…` (reuses `issue2094/hooks.py` unmodified) |
  | Slots | `ce` context-end + `pe` prefix-end (2 pre-declared degenerate cells @ pe) | plan §4.2; `degenerate_at_pe_cells`, `bank.json` |
  | Grid decoding | temperature 1.0, K=5 draws/pair×arm | `GRID_TEMPERATURE`/`GRID_DRAWS`, `issue2162_run.py` @ `b4ab6ed5f9…`; repro card |
  | Anchor decoding | temperature 1.0, K=10 draws/context | `ANCHOR_DRAWS`/`ANCHOR_TEMPERATURE`, ibid.; repro card |
  | `max_new_tokens` | 2048 (re-gen trigger: cap-hit > 2%/cell ⇒ 4096; never fired — 42/42,120) | `MAX_NEW_TOKENS`, ibid.; repro card; grid shard `cap_hit` fields |
  | Generation batch | 16 rows per hooked `generate` call | `gen_batch`, repro card; throughput-pilot report |
  | Stage-2 grid | doses {1, 4} × layers {8, 12, 14, 16, 19, 22, 26}, `mode="add"`, 1 greedy draw, 2 arms, survivor cap 12 (realized 5) | `STAGE2_*`, `issue2162_stage2.py` @ `ec113fdc05…`; `STAGE2_CAP`, `issue2162_analysis.py`; `stage2_results.json` |
  | Judge | `claude-sonnet-4-5-20250929`, graded 0–100 reason-then-score, `max_tokens=1024`, N=1 draw/rubric, temperature = API default | `judge_summary.json` instrument block; `issue2162_judge.py` |
  | Judge routing threshold | `threshold_base = 2000` (effective = ×otpm/400k); gate-3 slice force-SYNC (`threshold_base = 10^9`) | `DEFAULT_THRESHOLD_BASE`, `judge_dispatch.py`; `FORCE_SYNC_THRESHOLD_BASE`, `issue2162_judge.py` |
  | Coherence gate | form-only rubric; coherent := score > 60; cell marked < 50% coherent | plan §4.5; `COHERENCE_THRESHOLD`, `issue2162_analysis.py`; rubric frozen in `bank.json` |
  | Anchor-separation exclusion | \|ceiling − floor\| ≥ 0.5 (judge-contrast units), pre-registered | plan §6; `SEPARATION_BAR`, `issue2162_analysis.py` + `issue2162_judge.py` |
  | Survival floor | n ≥ 12 surviving pairs, else `untestable-causal` | `SURVIVAL_FLOOR`, `issue2162_analysis.py` |
  | Bootstrap | pair-clustered, B = 10,000, seed 21620 | `BOOT_B`/`BOOT_SEED`, ibid. |
  | Holm families | α = 0.05, within-family; realized m = 25/10/26 (pre-registered 31/15/28) | `HOLM_ALPHA`, ibid.; `families`, `stats.json`; plan §6 |
  | Registered test | IUT: p = max(p_shuffled, p_crosstype), exact two-sided Wilcoxon signed-rank + disjoint bootstrap CIs vs both nulls | plan §6 |
  | Read probe | linear (kernelized L2 logistic, linear kernel), 150 epochs Adam, lr 0.15, L2 1e-2, seed 21621; folds: leave-one-carrier-out (12) primary + value-pair-transfer secondary; AUC vs chance 0.5 | probe meta string + `PROBE_SEED`, `issue2162_analysis.py` |
  | Probe null band | B = 1,000 label permutations within carrier groups, per-draw re-maxed over layers, 97.5th pct | `PROBE_PERM_B`, ibid.; plan §6 |
  | F_act | signed projection, disjoint floor halves, span-mean $V_a$ at read layer 26 | `READ_LAYER`, ibid.; plan §4.4 (`fmetrics.f_act`) |
  | TF margin pools | 4 + 4 fixed judge-filtered completions per value-pair key, keep-threshold score > 50; realized 100/111 keys built (3 short of 4+4; 11 omitted — 3 `query_content` keys skipped by design, the rest lacked a judge-accepted candidate on a side) | `POOL_PER_SIDE`/`POOL_FILTER_MIN`, `issue2162_judge.py`; `judge/gates/pools_report.json` |
  | Injection gate bars | cosine ≥ 0.999; norm ratio ∈ [0.995, 1.005]; off-target rel ≤ 1e-3 | `GATE_*`, `issue2162_run.py` @ `b4ab6ed5f9…` |
  | Degeneracy guard bar | cosine ≥ 0.99999 for declared-degenerate identity | `DEGENERACY_COS_MIN`, ibid. |
  | Seeds | bank/donor 2162 · generation base 42 (per-draw `seed_base + i`) · bootstrap 21620 · probe 21621 | `SEED` (bank2162.py), `SEED_BASE` (run), `BOOT_SEED`/`PROBE_SEED` (analysis); repro card |
  | WildChat carriers | in-git `wildchat_random_v1.json` (600 rows), reserved slice [250:400], deterministic committed filter, seeded selection | plan §4.1/§10; `banks.load_bank` |
  | Code SHAs | stage-1 grid/anchors `b4ab6ed5f96216566b78b090f432d763246997b0` · margin `ba3485b619e9d8b35dad58d9c4746511b59f5d28` · stage-2 `ec113fdc05daecbfa5e04a7740552ed1093f079b` · analysis outputs at consolidation commit `b228639eace6ebbdb65a2ef36f55f48684e01f4b` (ancestor of the branch pin `20fcef9c28…`, `issue-2162`) | grid/anchors: `repro.git_commit` in `gates/pilot_gate_report.json` + `judge/gates/separation_gate_report.json`; margin: reproducibility card + `final_commit_sha` in `margin/upload_done.json`; stage-2: launch marker; `git rev-parse` |
  | Compute | pod-2162 8× H100 (grid ≈2.3 h/worker, mid-run projection at ~4.7 min/block × ~29 blocks/worker) · pod-2162-a 1× H100 (P7 + margin; venue deviation) · pod-2162-s2 4× H100 (stage-2) | launch + mid-run progress markers (the per-worker `wall_s` in the HF-only `grid_done_w*.json` manifests is pending upload-verification, not verified locally) |

  *Table note (per-phase SHA split):* the grid/anchors and margin legs ran 6 commits apart. The only change to `scripts/issue2162_run.py` between `b4ab6ed5f9…` and `ba3485b619…` (29+/6−) is an added state-sanity band on the degeneracy guard — a new constant `STATE_SANITY_COS_MIN`, its enforcement branch in `run_degeneracy_guard` (a new `state_sanity` gate failure mode), a new `state_sanity_cos_min` report field, and an extended `degenerate_criterion` string — plus docstring text. The band POST-DATES the grid run: the grid's degeneracy PASS was produced at `b4ab6ed5f9…`, before the band existed. No constant cited in this table is modified by the diff — two of them (`DEGENERACY_COS_MIN`, `GATE_OFFTARGET_REL_MAX`) appear inside hunks only as docstring or context lines, and all 14 cited assignments are byte-identical at both commits, so every value stands; only the phase→SHA provenance pairing differs.

- **Artifacts index:** rollout text `issue2162_ctxinfo/raw_completions/{anchors (16 shards), grid (234), stage2 (140), judge_raw, anchors_gate}` and tensors/manifests `issue2162_ctxinfo/analysis_tensors/{vc_bank (incl. bank.json + P1 gate reports), va_store, margin, probe_perm_matrix, manifests}` on `superkaiba1/explore-persona-space-data` (revision-pinned: https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue2162_ctxinfo — the repo tip at authoring time; every `issue2162_ctxinfo` artifact cited here resolves at this revision); per-cell tables + gate reports in git under `eval_results/issue_2162/{f_metrics, judge, gates, margin, stage2}` on branch `issue-2162` (pushed; tip includes `b228639eac…`); the committed `judge/` subtree is the summary/pools/gates/raw/audits set enumerated under Realized run counts — per-wave judge scores/items are HF-side (`raw_completions/judge_raw/`), not in git.

- **Metrics:** definitions + rationale for every shared metric (all grounded in the plan/Goal, none in a measured value):
  - **F_beh (PRIMARY).** Per draw, the dual-rubric judge contrast Δ = (judge_B − judge_A)/100 under the pair's own two value descriptors; F_beh = (Δ̄_patched − Δ̄_floor)/(Δ̄_ceiling − Δ̄_floor), floor = unpatched-A anchors, ceiling = generate-under-B anchors. Chosen because the Goal's construct is BEHAVIORAL movement toward the donor value measured on-distribution: on-policy generations under the edited context at natural temperature, scored by a graded judge (graded-primary rule), and normalized within-pair so one F is comparable across 21 heterogeneous types — the floor/ceiling normalization is what makes "fraction of a full context swap" meaningful per type. Conflict cells replace the descriptor pair with (follows-instructed, follows-demonstrated) and report the balance shift under the same normalization.
  - **F_act (activation twin).** Signed projection of the patched-minus-floor answer-state shift onto the ceiling-minus-floor axis (span-mean $V_a$, read layer 26), with disjoint floor halves so the shared-baseline noise term cancels (the #1415 convention). Chosen as the judge-INDEPENDENT continuous cross-check: across a 21-rubric roster, a judge-free companion matters more than at the parent's 2 rubrics; reported beside F_beh at every cell with the same nulls, and the cell-level rank agreement between the two levels is itself a reported diagnostic.
  - **Teacher-forced fixed positive-vs-negative completion margin (SECONDARY).** Length-normalized lnP of each item of a FIXED judge-filtered pool (4 B-value-exhibiting + 4 A-value-exhibiting completions per value-pair, drawn from the anchor draws, keep-threshold > 50) under every patched state and under the anchors; margin = mean lnP(B-pool) − mean lnP(A-pool). Chosen as the dual-DV rule's non-saturating continuous companion in its preferred form — the same fixed answer set is scored under every state, so there is no selection-on-outcome bias; deliberately teacher-forced and SECONDARY (validated against F_beh via ρ(margin shift, F_beh) across cells with dynamic range BEFORE it carries any read; never the headline).
  - **Linear read-probe AUC.** Per (type × slot × layer), a linear logistic probe predicting which value is present in the NATURAL (unpatched) slot state, group-held-out (leave-one-carrier-out primary), reported as AUC against chance 0.5. Chosen because a causal null is ambiguous between not-encoded and encoded-but-unusable — the probe supplies the read side of the read × write 2×2; LINEAR per the project default (a linear readout keeps the decodability claim interpretable); AUC rather than R² because n = 24 contexts per value-pair against d = 3584 makes R² estimator-degenerate while rank-based AUC under group folds remains a valid read.
  - **Coherence rate.** Fraction of draws with form-only judge score > 60 (fluency/well-formedness only). Dual role: gate (all reported quantities over coherent draws) and disruption DV (excess incoherence vs the anchor baseline is the only registered read for `filler_swap`, which has no ceiling and reports no F).
  - **Conflict balance shift.** (judge_demo − judge_instr)/100, floor/ceiling-normalized — the registered DV for the 4 conflict cells (which route wins at the slot when instruction and demonstration disagree).
  - **Anchor separation (ceiling − floor).** The per-pair denominator quality measure; drives the pre-registered ≥ 0.5 exclusion, with pre-exclusion counts reported per type. One pre-committed caveat ships beside those counts: the 0.5 bar selects on a K=10 anchor ESTIMATE, so near-threshold kept pairs carry slightly upward-biased denominators — a small, arm-symmetric, conservative-for-positives attenuation of F.
  - **Cap-hit fraction.** Fraction of rows whose completion reached `max_new_tokens` (basis: re-tokenized completion length — recorded deviation 4b); reported per cell against the pre-registered 2% re-generation trigger, because truncation creates silent zeros in judged evals.

---

- **Planned-manifest name crosswalk.** Five planned condition / metric names differ typographically from the prose terms used above for the same objects; they are bridged here so the report's coverage of the approved manifest is mechanically checkable:
    - `behavior fraction-of-swap (F_beh)` — the planned-manifest name for F_beh above.
    - `activation fraction-of-swap (F_act)` — the planned-manifest name for F_act above.
    - `teacher-forced positive-vs-negative completion margin` — the planned-manifest name for Teacher-forced fixed positive-vs-negative completion margin above.
    - `anchor separation (ceiling minus floor)` — the planned-manifest name for Anchor separation (ceiling − floor) above.
    - `Shuffled-donor null (same type, norm-matched)` — the planned-manifest name for shuffled-donor null above.

## Results

### Per-type fraction-of-swap (behavior)

**Methodology**

- Per (type-cell × slot × arm): pairs failing the pre-registered anchor-separation exclusion (|ceiling − floor| < 0.5) are dropped; per surviving pair, F_beh is the pair mean over its K=5 coherent draws; the arm statistic is the mean over surviving pairs with a pair-clustered bootstrap 95% CI (B = 10,000, seed 21620).
- Three arms per type (steered / shuffled-donor null / cross-type-donor null), one panel per slot (`ce`, `pe`); post-exclusion n stated per type. Sources: `eval_results/issue_2162/f_metrics/{f_cells,null_shuffled_cells,null_crosstype_cells,anchors}.jsonl`.
- `filler_swap` reports no F anywhere (no ceiling by construction); the two pre-declared degenerate prefix-end cells are excluded from aggregates.
- **Rendered figure (plotter caption):** Mean behavior fraction-of-swap F_beh (unitless; y) per type-cell (x, post-exclusion pair count in parentheses) for the steered, shuffled-donor-null, and cross-type-donor-null arms, one panel per readout slot (context-end top, prefix-end bottom). Error bars are pair-clustered bootstrap 95% CIs (B=10,000, seed 21620); pairs with anchor separation |ceiling - floor| < 0.5 are excluded, cells with zero surviving pairs are marked n/a, and 14 of 76 (cell x slot) combinations are untestable-causal (post-exclusion n < 12). The persona_role_header bars rest on n=1 surviving pair per slot (0.995 at context-end, -0.397 at prefix-end).

![Per-type fraction-of-swap (behavior)](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/hero_ftype.png)

**Takeaways**

*(Thomas fills in)*

### Per-type F_beh - per-pair companion

**Methodology**

- The per-unit companion of the hero figure: same exclusion, NO aggregation — one point per surviving pair per arm (pair mean over K=5 coherent draws), labeled by pair id, per type × slot. Source: the same three per-cell row files.
- **Rendered figure (plotter caption):** Per-pair F_beh points behind the aggregate bars: one point per separation-surviving pair per arm (pair mean over K=5 coherent draws), horizontally offset by arm, labeled by pair id, per type-cell, one panel per slot. Same |separation| >= 0.5 exclusion as the aggregate view; no aggregation and no error bars (each point is one pair). n = 5,855 surviving pair-rows (1,952 steered, 1,951 shuffled-donor null, 1,952 cross-type null).

![Per-type F_beh - per-pair companion](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/hero_ftype_perpair.png)

**Takeaways**

*(Thomas fills in)*

### Read x write 2x2 (probe AUC vs causal F)

**Methodology**

- x = max-over-28-layers macro AUC per (type × slot) from the linear read probe under leave-one-carrier-out folds; y = steered mean F_beh (separation-excluded, as in `per_type_f_beh`). Sources: `f_metrics/probe.json` + `f_metrics/f_cells.jsonl`.
- Quadrant thresholds are the REGISTERED verdicts, not free parameters: causal-positive ⇔ clears BOTH nulls on fully disjoint pair-clustered 95% CIs AND survives the family Holm IUT; probe-positive ⇔ the max-AUC clears the max-selected label-permutation 97.5th-percentile band (selection-symmetric: every permutation draw is re-maxed over layers).
- Cells with post-exclusion n < 12 (the pre-registered survival floor) carry the explicit fifth label `untestable-causal` — the causal test is unreachable there, so they are never rendered as a causal verdict. Their probe AUC is plotted only where a steered F_beh y-value exists (≥ 1 surviving steered pair): per the realized figure caption, 8 untestable-causal cells are plotted while 6 further untestable cells with ZERO surviving steered pairs are omitted from the panel and counted in the title (`figures/issue_2162/captions.json`, `two_by_two`).
- **Rendered figure (plotter caption):** One point per (type-cell x slot) placing the linear read-probe max-over-28-layers macro AUC (x, leave-one-carrier-out folds) against the steered mean F_beh (y, separation-excluded); marker and color encode the persisted (probe verdict, causal verdict) quadrant, with untestable-causal (post-exclusion n < 12) as the explicit fifth class. Plotted class counts: 5 stored-and-used, 55 stored-but-unusable, 0 used-but-not-decoded, 2 absent, 8 untestable-causal; 6 further untestable cells with zero surviving steered pairs are omitted and counted in the title. No single vertical read-threshold line is drawn because the probe-positive threshold is the per-cell max-selected permutation band (97.5th percentile).

![Read x write 2x2 (probe AUC vs causal F)](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/two_by_two.png)

**Takeaways**

*(Thomas fills in)*

### Probe AUC per layer - per-unit companion of the 2x2

**Methodology**

- The per-unit companion of the 2×2: per (type × slot × layer), macro AUC over the 3 value-pairs under carrier folds, with the label-permutation band (B = 1,000, within-carrier shuffles) — no max taken in this view (full-curve display; the max-selected band's upper bound is annotated where the 2×2's verdict is drawn). Sources: `f_metrics/probe.json` + the persisted per-draw × per-layer matrix `issue2162_ctxinfo/analysis_tensors/probe_perm_matrix/`.
- **Rendered figure (plotter caption):** Companion heatmap of the same probe read: rows are type-cells, columns are layers 0-27, color is the leave-one-carrier-out macro AUC, one panel per slot (context-end left, prefix-end right). Same data as the per-layer curve panels; n = 78 (cell x slot) units.

![Probe AUC per layer - heatmap companion](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/layer_profile.png)

**Takeaways**

*(Thomas fills in)*

### Stage-2 layer profile (post-selection)

**Methodology**

- Post-selection stage-2 read over the 5 realized survivor (cell × slot) combos: per survivor, mean F_beh over surviving pairs at each (layer ∈ {8, 12, 14, 16, 19, 22, 26} × dose ∈ {1, 4}), pair-difference add-mode edits (Δ = V(B) − V(A), single layer, dose = alpha multiplier), 1 greedy draw per pair, steered with the shuffled-donor-null companion. Source: `f_metrics/stage2_cells.jsonl` (from the 140 stage-2 shards).
- Rendered as a type × layer heatmap at each dose, labeled post-selection everywhere (the survivors were selected on stage-1 outcomes; this figure never carries an unbiased-estimate claim).
- **Rendered figure (plotter caption):** Stage-2 heatmaps for the 5 stage-1-selected survivor cells (rows; all at the context-end slot): mean F_beh over separation-surviving pairs at each patch layer in {8, 12, 14, 16, 19, 22, 26} (columns) and dose (1 left, 4 right), greedy 1 draw per pair. The top row is the steered arm; the bottom row is steered minus the shuffled-donor null. Post-selection: these cells were chosen by the stage-1 selection rule, so the values are confirmation reads, not unbiased estimates.

![Stage-2 layer profile (post-selection)](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/layer_profile_stage2.png)

**Takeaways**

*(Thomas fills in)*

### Stage-2 per-pair companion

**Methodology**

- Per-unit companion: no aggregation — per-pair stage-2 F_beh points at each survivor's best (layer, dose), steered and null interleaved, labeled by pair id. Source: `f_metrics/stage2_cells.jsonl`.
- **Rendered figure (plotter caption):** Per-pair stage-2 F_beh points at each survivor cell's best (layer, dose) by steered mean (annotated above each column), steered (blue) and shuffled-donor null (grey) interleaved, pair-id labeled, separation-surviving pairs only. Post-selection: the 5 survivor cells were chosen by the stage-1 selection rule. No aggregation; each point is one pair at greedy 1 draw.

![Stage-2 per-pair companion (post-selection)](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/layer_profile_stage2_perpair.png)

**Takeaways**

*(Thomas fills in)*

### Route contrasts at matched content

**Methodology**

- Steered mean F_beh (separation-excluded, pair-clustered bootstrap CI) for the matched-content route sets — {`instr_format` vs `demo_format`}, {`persona_prompted` vs `demo_persona` vs `persona_role_header`}, {`instr_language` vs `language_implied`} — per slot: the same content induced by different routes, isolating the route with content held fixed.
- Plus the 4 conflict cells' balance shift = (judge_demo − judge_instr)/100, floor/ceiling-normalized: the DV for which route wins when instruction and demonstration disagree (both directions of both conflicts). Sources: `f_metrics/{f_cells,null_shuffled_cells,null_crosstype_cells}.jsonl`.
- `persona_role_header`'s varied span is immediately adjacent to the context-end slot — a designed boundary-content property, pre-declared in plan §4.1.
- **Rendered figure (plotter caption):** Steered mean F_beh for each matched-content route set - base type (blue) beside its route variant or conflict cell (orange) - per slot; route sets: instr_format vs demo_format, persona_prompted vs demo_persona and persona_role_header, instr_language vs language_implied, plus the four conflict cells, whose F_beh equals the normalized conflict balance shift by construction (value_b = the demonstration-carried value). Error bars are pair-clustered bootstrap 95% CIs over separation-surviving pairs; bars with zero surviving pairs are absent (demo_format, demo_persona), and persona_role_header rests on n=1 surviving pair per slot.

![Route contrasts at matched content](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/route_contrasts.png)

**Takeaways**

*(Thomas fills in)*

### Route contrasts - per-pair companion

**Methodology**

- Per-unit companion: per-pair F_beh / balance-shift points for every route-variant and conflict cell, labeled by pair id. Source: `f_metrics/f_cells.jsonl`.
- **Rendered figure (plotter caption):** Per-pair steered F_beh points for every route-variant and conflict cell beside its base type, pair-id labeled, per slot, with no separation exclusion and no aggregation. The top panel shows the full range; the bottom panel shows the same points restricted to |F_beh| <= 2, since a small number of separation-degenerate pairs reach |F_beh| of roughly 100 and otherwise set the shared scale. n = 2,574 scored steered pair-rows across all cells.

![Route contrasts - per-pair companion](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/route_contrasts_perpair.png)

**Takeaways**

*(Thomas fills in)*

### Recency and load curves

**Methodology**

- Per crossed type: steered mean F_beh (separation-excluded) at depth {1, 3, 5} (recency: same information introduced at turn 1 followed by depth−1 neutral padding exchanges identical between A and B) and at load {1, 3, 5} (load: load-many same-type pieces in the prefix, one designated target varying), depth/load 1 = the base cell; pair-clustered bootstrap CIs, null band behind.
- The registered secondary read is the pair-clustered bootstrap CI on the per-pair depth/load slope (trend reads are secondary descriptives, per plan §6). Sources: `f_metrics/{f_cells,null_shuffled_cells}.jsonl`.
- **Rendered figure (plotter caption):** Steered mean F_beh (y) versus conversation history depth (left) and distractor load (right) at levels {1, 3, 5}, where level 1 is the uncrossed base cell, one curve per crossed type per slot (context-end solid, prefix-end dashed); error bars are pair-clustered bootstrap 95% CIs over separation-surviving pairs. Grey bands are the shuffled-donor null's 95% CIs with the null mean line inside. The in-panel text lists the registered per-pair depth/load slope with its pair-clustered bootstrap 95% CI and pair count.

![Recency and load curves](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/dose_position.png)

**Takeaways**

*(Thomas fills in)*

### Recency/load - per-pair companion

**Methodology**

- Per-unit companion: per-pair F_beh at each depth/load level, lines connecting the same pair across levels. Source: `f_metrics/f_cells.jsonl`.
- **Rendered figure (plotter caption):** Per-pair steered F_beh trajectories across depth (left) and load (right) levels; lines connect the same (carrier x value-pair) across levels, colored by crossed type (context-end solid, prefix-end dashed), with no separation exclusion and no aggregation. The top row shows the full range; the bottom row shows the same trajectories restricted to |F_beh| <= 2, since single separation-degenerate pairs reach approximately +24 and -8 and otherwise set the scale.

![Recency/load - per-pair trajectories](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/recency_load_perpair.png)

**Takeaways**

*(Thomas fills in)*

### Anchor separation per pair (pre-exclusion)

**Methodology**

- Per pair: ceiling mean − floor mean (judge-contrast units) from the K=10 anchor draws, shown as a histogram/strip per type with the 0.5 exclusion bar drawn and pre-/post-exclusion counts per type. Source: `f_metrics/anchors.jsonl`.
- This is the exclusion instrument made visible: the pre-registered bar, its per-type bite, and the pre-committed selection caveat (the bar selects on a K=10 estimate) ship with this figure.
- **Rendered figure (plotter caption):** Per-pair anchor separation (y: generate-under-donor ceiling mean minus unpatched floor mean, judge-contrast units, K=10 anchor draws per side) per type-cell (x), before any exclusion; red dotted lines mark the pre-registered |separation| >= 0.5 exclusion bar and kept/total counts per type are printed above. Three types keep 0/36 pairs (demo_format, demo_persona, recency_prior_topic_d5) and persona_role_header keeps 1/36. n = 1,368 pair rows (38 grid types x 36 pairs).

![Anchor separation per pair (pre-exclusion)](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/anchor_separation_diag.png)

**Takeaways**

*(Thomas fills in)*

### F_act vs F_beh agreement

**Methodology**

- Per (type-cell × slot × arm): mean F_act (read layer 26, disjoint floor halves) against mean F_beh over the same surviving pairs, one point per cell-arm, colored by arm; Spearman ρ across cells with dynamic range reported in-panel (the rank-agreement diagnostic between the judge-scored and the judge-free level — the dynamic-range screen mirrors the rule-19 validation convention). Sources: `f_metrics/{f_cells,null_shuffled_cells,null_crosstype_cells}.jsonl`.
- **Rendered figure (plotter caption):** Mean activation fraction-of-swap F_act (x; read at layer 26 with disjoint floor halves) versus mean behavior F_beh (y) per (cell x slot x arm) over separation-surviving pairs, colored by arm (steered blue, shuffled-donor null grey, cross-type null red), point labels are cell|slot. The per-arm Spearman rho in the legend is computed over the n=66 units passing the dynamic-range screen stated in the panel; the 4 screened-out units per arm render as hollow circles and carry no rho weight. Realized F_act and F_beh ranges of the screened units are printed in the panel.

![F_act vs F_beh agreement](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/act_beh_agreement.png)

**Takeaways**

*(Thomas fills in)*

### Teacher-forced margin vs F_beh (dual-DV validation)

**Methodology**

- Per (type-cell × slot): mean teacher-forced margin shift (steered − floor, length-normalized lnP over the fixed 4+4 judge-filtered pools) against mean F_beh — the registered rule-19 validation scatter, whose ρ across cells with dynamic range must be reported BEFORE the margin carries any cross-condition read. Sources: `f_metrics/{f_cells,margin_cells}.jsonl`; pool coverage per `judge/gates/pools_report.json` (100/111 keys; 3 `query_content` keys skipped by design — no fixed shared answer pool is well-defined when the query itself varies — with explicit skip rows recorded).
- **Rendered figure (plotter caption):** Per-(cell x slot) mean teacher-forced fixed-pool margin shift (x: steered minus floor anchor, length-normalized log-probability over the fixed 4+4 completion pools) versus mean F_beh (y), blue = the registered rule-19 grain (n=72 cells passing the dynamic-range screen), grey = the per-pair companion points (n=2,308); Spearman rho and p for both grains are in the title (per-cell rho=0.458, p=5.1e-05; per-pair rho=0.334, p=3.6e-61). The single high-leverage cell icl_task_mapping|ce (margin shift 3.88 against a next-largest of about 0.35) is plotted and labeled; the inset re-renders the remaining 71/72 cells at |margin shift| < 1 and |F_beh| < 3. The persona_role_header means in this grain (1.48 at context-end, -2.72 at prefix-end, 12 pairs each, computed without the anchor-separation exclusion) lie outside the [0, 1] fraction-of-swap range, i.e. the patched value fell outside the floor-to-ceiling anchor range for that thin cell.

![Teacher-forced margin vs F_beh (dual-DV validation)](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/margin_validation.png)

**Takeaways**

*(Thomas fills in)*

### Cross-type null by donor type + shuffled-donor null by donor value (diagnostic)

**Methodology**

- Diagnostic breakout of the null arms by donor identity (pure re-reductions of the persisted null tables — donor type, pair, and value ids are recorded per row): (a) cross-type facet — per (recipient cell × slot), mean F_beh of the cross-type-donor arm split BY DONOR TYPE, rendered for every cell whose pooled cross-type-null bootstrap CI excludes 0 (a donor type whose content moves the recipient's rubric inflates the pooled null; conservative for positives, made visible rather than assumed away); (b) shuffled-donor facet — for the two graded/ordinal value sets only (`refusal_boundary`, `constraint_knowledge`), the shuffled-donor arm split BY DONOR VALUE (an adjacent-value donor can earn partial rubric credit and elevate that null). Sources: `f_metrics/{null_crosstype_cells,null_shuffled_cells}.jsonl`.
- **Rendered figure (plotter caption):** Top: cross-type-donor-null F_beh split by donor type (bar = mean over separation-surviving rows; donor type and row count annotated on each bar) for the 24 recipient (cell x slot) combinations whose pooled cross-type-null bootstrap 95% CI excludes 0. Bottom: shuffled-donor-null F_beh split by donor value for the two ordinal value sets (refusal_boundary, constraint_knowledge) at cells where either null's pooled CI excludes 0. Diagnostic view: per-donor means without CIs; per-donor row counts are annotated.

![Cross-type null by donor type and shuffled null by donor value (diagnostic)](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/crosstype_null_by_donor.png)

**Takeaways**

*(Thomas fills in)*

### Coherence and cap-hit diagnostics

**Methodology**

- Per (type-cell × slot × arm), three panels: excess incoherence = incoherent fraction (score ≤ 60) minus the anchor baseline rate (top); cap-hit fraction (completion length at `max_new_tokens`, re-tokenized basis) with the pre-registered 2% re-generation trigger drawn (middle); and post-exclusion pair count with the n = 12 untestable-causal floor line drawn, out of 36 pre-exclusion pairs per cell (bottom). Sources: `f_metrics/{f_cells,null_shuffled_cells,null_crosstype_cells,anchors}.jsonl`.
- Excess incoherence doubles as the ONLY registered read for `filler_swap` (the disruption control has no ceiling, hence no F); realized run-level cap-hit was 42/42,120 rows (0.0997%), below the trigger, so no re-generation round exists in this figure's data.
- **Rendered figure (plotter caption):** Per (cell x slot x arm): excess incoherence (top; incoherent fraction at judge score <= 60 minus the cell's anchor-baseline incoherent rate), cap-hit fraction (middle; completions ending at the generation length cap, dotted line = the pre-registered 2% re-generation trigger), and post-exclusion pair count (bottom; dotted line = the n=12 untestable-causal floor, out of 36 pre-exclusion pairs per cell). Arm colors are fixed across panels (steered blue, shuffled-donor null grey, cross-type null red). Denominators are the per-(cell x slot x arm) draw counts pooled over pairs.

![Coherence and cap-hit diagnostics](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/diagnostics.png)

**Takeaways**

*(Thomas fills in)*

## Conclusion and next steps

*(Thomas fills in)*
