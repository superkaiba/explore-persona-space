# Corrected natural task/subject context-vector patching

## 0.0 TL;DR / change-my-mind

This user-requested follow-up asks whether the final-context-token state causally carries the kind of response more strongly than its subject. The primary intervention is replacement of Qwen2.5-7B-Instruct decoder block 19's output at the final prompt token. A separately reported maximal intervention replaces that position at all 28 blocks. A crossed bank of natural single-turn requests supplies same-subject/different-task and same-task/different-subject swaps. Every generation has the same literal user greeting and a forced identical assistant opening. A minimal-pair bullet-versus-paragraph arm uses the exact same capture, patch, and generation pipeline as a positive manipulation control. Complete answers and a frozen-key, content-only blinded annotation are persisted row by row.

The result changes my mind in favor of selective task-vector sufficiency only if the positive formatting control works and layer-19 task swaps follow the donor task more often/strongly than subject swaps follow the donor subject. A formatting-control failure makes the task/subject null uninterpretable. All-layer results are secondary and cannot rescue a failed layer-19 primary claim.

Estimated GPU-hours (total): 1.5 on one H100, with a hard stop at 3 GPU-hours.

## 1. Goal and claim scope

The parent goal is preserved: test causal sufficiency of a single context position. This amendment narrows the read to the paper's qualitative claim that the context vector resembles a task/function vector because it represents the kind of response more strongly than the response subject. It does not test classical in-context-learning task vectors, does not train a map, and does not claim that information absent from this token is absent elsewhere in the prompt KV cache.

Primary estimand: among directed same-subject task swaps under a layer-19 replacement, the fraction of complete coherent outputs whose observed response form matches the donor rather than the recipient. The matched subject-swap estimand is the corresponding donor-versus-recipient subject rate. The comparison is descriptive and paired over the crossed prompt bank; this is an exploratory qualitative causal intervention, not a powered population estimate.

Secondary estimand: the same quantities under all-28-layer replacement, reported separately. Manipulation check: donor-format adoption for bullet↔paragraph minimal pairs run through the exact same pipeline.

## 2. Hypothesis and decision criteria

- H1 (primary, layer 19): same-subject/different-task replacement transfers donor response form more often than same-task/different-subject replacement transfers donor subject.
- H2 (specificity): task swaps retain the recipient subject, while subject swaps retain the recipient task.
- H3 (secondary): all-layer replacement may increase transfer, but is a distinct maximal intervention and is never pooled with layer 19.
- H4 (positive control): the exact pipeline transfers an explicitly stated bullet-versus-paragraph policy, especially in the all-layer arm.
- H0: after controlling the opening and allowing complete generations, neither response form nor subject reliably follows the donor state.

Interpretation lattice:

1. Format control passes, layer-19 task transfer exceeds subject transfer with specificity intact: evidence consistent with the selective causal task-vector claim.
2. Format control passes, both task and subject stay with the recipient: the layer-19 state is not sufficient under this intervention; do not infer absence of encoded information.
3. Format control passes only all-layer: layer-19 task/subject null is weakly interpretable, and the all-layer task result is evidence only for a maximal multi-layer edit.
4. Format control fails at both layer settings: pipeline manipulation check failed; task/subject comparisons are inconclusive.
5. Subject follows the donor as often as task: evidence against selectivity even if patching changes outputs.

Success criteria: the experiment is successfully executed when all 129 planned rows pass census/telemetry/completeness checks, every row has a valid frozen-key blind annotation, and the format-control gate yields an interpretable verdict. Scientific confirmation additionally requires the format gate to pass and the layer-19 task-transfer read to exceed the subject-transfer read without loss of recipient-subject specificity. Scientific falsification is a passed format gate with layer-19 task transfer no stronger than subject transfer, or donor-subject transfer matching/exceeding donor-task transfer.

Positive-format gate: at least 4 of 6 directed all-layer format swaps must be annotated as donor format and the corresponding unpatched/self-patched outputs must follow their requested format in at least 5 of 6 cases. This is deliberately a manipulation check, not a significance test. Completeness gate: zero primary rows may end by length cap; any capped row is automatically rerun at 1,536 new tokens before annotation. Coherence gate: every headline denominator reports all rows and the subset with blinded coherence >=60; if fewer than 80% of rows are coherent, the affected arm is descriptive-only. Annotation completeness floor: 100% of generated rows must receive a parsed row-level annotation; otherwise no aggregate is reported.

## 3. Model, intervention, and conventions

- Model: `Qwen/Qwen2.5-7B-Instruct`, bfloat16, exact resolved Hugging Face revision recorded from the loaded config/tokenizer.
- Layer convention: layer 19 means the forward-hook output of `model.model.layers[19]`, matching the paper and issue #2094. This is not `hidden_states[19]`.
- Position: final token of the no-system-turn chat render with `add_generation_prompt=True`; assert the Qwen assistant-header suffix and per-row left-padding coordinates.
- Patch: replace, not add, the recipient state at that position during prefill only with the donor's clean state at the same layer. The decode steps are unedited; the edited cached K/V persists.
- Arms: `unpatched`; `self_patch_L19`; `self_patch_all28`; `donor_patch_L19` (primary); `donor_patch_all28` (secondary).
- Generation: greedy; `max_new_tokens=768`, automatic 1,536-token cap rerun; `do_sample=False`; `temperature=None`; `top_p=None`; `top_k=None`; `repetition_penalty=1.0`; explicit EOS and padding IDs. Store generated token IDs, token count, termination reason, effective generation config, package versions, model/tokenizer revisions, and per-layer edit telemetry.
- Greeting control: every user request begins with the exact bytes `Hello. `, and generation is constrained to begin with the exact tokenization of `Response:\n`. The patch remains at the original assistant-header final token; the forced opening is generated afterward and is identical across every arm.

Injection gates assert one prefill edit per patched layer, correct unpadded/padded position, source-vector shape, finite values, and zero maximum error after conversion to the hidden dtype. A self patch must leave the greedy response identical to unpatched for at least 13 of 15 prompts; failure is reported as numerical sensitivity and blocks strong causal wording.

## 4. Data and crossed cells

Data tier: tier 4 controlled natural-language minimal pairs. The paper's real retrieval failures motivate the exact genres (travel itinerary and quiz), but naturally occurring corpora cannot supply a fully crossed task×subject intervention with known one-axis changes. Templating is therefore necessary for causal identification. To limit template bias, all nine combinations are ordinary single-turn requests, three distinct genres are used, all combinations are semantically plausible, and claims are scoped to this bank.

Subjects (three): Vancouver Chinese cuisine; a family visit to Tokyo and Mount Fuji; Ancient Rome's landmarks and history.

Response tasks (three): a three-day itinerary with morning/afternoon/evening entries; a five-question four-option quiz plus answer key; a four-section practical visitor briefing. Each asks for at most 220 words so the 768-token cap is non-binding.

Crossed test cells, each in both directions:

- Same subject, different task: 3 subjects × 3 unordered task pairs × 2 directions = 18 donor→recipient rows per intervention.
- Same task, different subject: 3 tasks × 3 unordered subject pairs × 2 directions = 18 rows per intervention.
- Positive formatting control: for each subject, the same 140–180-word explanation prompt differs only by `exactly five bullet points` versus `one continuous paragraph`; 3 subjects × 2 directions = 6 rows per intervention.
- Anchors: 15 unpatched prompts (9 crossed-bank + 6 format-bank).
- Self patches: the same 15 prompts at layer 19 and all layers.

Total planned generations: 15 + 30 + 72 + 12 = 129. Greedy decoding has one draw per directed row; prompt-pair variation, not stochastic decoding, is the replication unit. No layer search, task selection, or result-driven pair selection is performed.

## 5. Dependent variables and blinded annotation

Objective row fields are termination reason, generated token count, forced-prefix match, and edit telemetry. A blind reader classifies observable output form (`itinerary`, `quiz`, `briefing`, `explanation`, `other_or_mixed`), subject (the three subject labels plus `other_or_mixed`), format (`bullets`, `paragraph`, `neither_or_mixed`), completeness, 0–100 coherence, and a short evidence phrase.

Before the first API call, a random opaque row-ID→generation-ID key is frozen to disk. Outbound packets contain only the opaque row ID, the full generated answer, neutral candidate labels, and the classification request; they omit prompts, source/recipient identities, arm, layer, selection rule, expected result, file paths, and project context. A scope-aware leakage scan fails loud. The reader has no tools, system prompt, or filesystem: direct Anthropic Messages calls use `claude-sonnet-4-5-20250929`, temperature 0, and `max_tokens=8192`. There are fewer than ten packet calls, so this is an auditable qualitative read rather than a volume judge path. Persist the exact outbound request, scan terms/hits, model parameters, raw response, stop reason, and usage beside each response. Any non-`end_turn`, truncation, parse failure, duplicate/missing ID, or invalid enum fails the annotation phase without partial aggregate output.

Row-level labels remain immutable after unblinding. Any manual audit corrections are appended as a separate file with author/reason/original/corrected fields; they never overwrite the blind annotations. Aggregate both raw blind labels and any explicitly corrected sensitivity view.

Mapping/probe metrics are N/A: no representation mapping or probe is fit, so held-out R², identity-plus-bias, and nearest-neighbor retrieval requirements do not apply.

N/A — no held-out predictive DV

## 5.1 Measurement validity

| Construct | Metric | On-distribution / on-policy status |
|---|---|---|
| Donor response-task expression | Blind categorical form label; donor/recipient/other proportions over same-subject task swaps | On-policy complete greedy generations from the intervened model on the crossed natural-request bank |
| Donor subject expression | Blind categorical subject label; donor/recipient/other proportions over same-task subject swaps | On-policy complete greedy generations from the intervened model on the same bank |
| Formatting manipulation check | Blind bullet/paragraph/neither label and exact structural regex companion | On-policy complete greedy generations through the identical capture/patch/decode path |
| Output validity | Blind completeness and 0–100 coherence; objective EOS/cap termination | On-policy outputs; all-row and coherent-only denominators both retained |

## 6. Analysis

Validate anchors first: report form/subject/format accuracy, completeness, and coherence. Report self-patch exact-string agreement and injection telemetry. Then, separately for layer 19 and all layers, report row counts and donor/recipient/other proportions for task swaps, subject swaps, and format controls. Also report paired task-versus-subject transfer at the prompt-pair level and exact Wilson intervals for each simple proportion; intervals are descriptive because the bank is constructed.

The report includes a fixed, non-outcome-selected qualitative roster: lexicographically first directed swap for every task-pair family and every subject-pair family, plus all six positive controls, with full answers (not excerpts). A second compact table may identify mixed/failure rows using an explicit annotation-defined rule, not post-hoc curation. No result is called positive unless the format manipulation check passes.

No large judge wave is used (under 10 calls), so the >=5,000-call judge pilot is N/A. This is a blinded qualitative read rather than a graded behavior-expression DV; the separate judged-DV multi-sample reliability requirements are N/A. The annotation instrument still gets a committed parser round-trip test and one live one-row packet smoke before the remaining packets.

## 7. Smoke run and kill criteria

Smoke executes the production model loader, capture hooks, forced-prefix decoder, layer-19 and all-layer patch arms, unpatched/self patch, one task swap, one subject swap, one format swap, JSONL writer, cap/termination parser, and injection assertions on one small batch. It then runs the production blind-packet builder/parser offline with a realistic fenced and unfenced response; one live one-row annotation packet validates the API contract.

Smoke blind-spot enumeration:

- The smoke uses fewer prompts/rows than production but substitutes no implementation and downgrades no gate.
- The production-only paths are multi-batch resume, the 1,536-token cap-rerun branch if no smoke answer caps, the remaining blinded packet calls, Hub upload, and aggregate report construction over all 129 rows.
- No third-party import is production-only; smoke loads the real model, hooks, tokenizer, and Anthropic SDK.

Kill before production if any patch fails injection telemetry, the forced prefix differs, a self patch materially changes the smoke answer, any required model/revision field is unresolved, a nominally complete short-answer smoke hits the cap, or the blind parser/API smoke fails. During production, abort on CUDA OOM after one batch-size reduction, non-finite source states, wrong realized row count, duplicate generation IDs, or projected wall time above 3 GPU-hours.

## 8. Implementation and tests

Add a dedicated issue-2094 follow-up runner, blinded annotator, analyzer/report generator, and focused tests. Reuse `PositionEditHookStack`/`joint_hooks` from `src/explore_persona_space/experiments/issue2094/hooks.py`; do not change the established hook. Tests cover crossed-cell census and one-axis invariants, greeting/prefix configuration, row-ID uniqueness, generation termination classification, source/target metadata, blind packet leakage scans, parse validation (plain and fenced JSON), and aggregate denominators. Run focused pytest, Ruff on touched Python, and repository workflow lint.

Resume is per generation-ID JSONL row with schema validation and duplicate refusal. Cached source states are reused only when their model revision, prompt-bank hash, layer list, dtype, and capture convention match exactly. A fresh `DONE.json` is written only after row-census, cap, telemetry, and artifact checks pass.

## 9. Compute and storage sizing

One H100 in eval intent. Model weights require roughly 15 GB bf16. At batch size 12 and <=1,000 total tokens per sequence, KV/cache plus activations should remain below 25 GB, leaving wide headroom on an 80 GB H100; smoke measures peak allocation and production falls back once to batch 6 on OOM. Expected generation is 129 × <=768 tokens, approximately 0.8–1.5 wall-hours including model load and answer-vector-free analysis; hard stop 3 GPU-hours. Greedy generation seed: 0 for provenance (sampling is disabled); blind-key shuffling uses an OS-random frozen key and is never redrawn.

| Component | n_cells | GPU width | planned_wall_h | Basis |
|---|---:|---:|---:|---|
| Model load + full 129-row capture/generation run | 129 | 1 | 1.5 | Qwen2.5-7B bf16 on one H100; conservative from the prior 48-row inline run plus 3.5× rows and longer cap |

Outputs are small text/JSON (<50 MB expected) under `/workspace/issue2094-natural-task-subject-corrected` on the RunPod `/workspace` volume (per-pod quota ~130 GB). Before generation, call the shared out-root headroom check with a 2 GB floor and canary. No checkpoints, merged weights, training artifacts, or retained tensor ladders are produced. Source states are at most 15×28×3584×4 ≈24 MB. End-of-run footprint is therefore <2 GB plus the pre-existing model cache. N/A — no per-rung checkpoint persistence, fan-out, or cross-machine staging.

## 10. Persistence and operational plan

Write generations incrementally as JSONL and fsync each completed batch. Persist prompt bank, resolved config/revisions, source-state provenance/checksums, row-level generations/token IDs/termination/edit telemetry, frozen blind key, exact request/response sidecars, parsed annotations, summary, and qualitative Markdown report. Raw text/JSON goes to `superkaiba1/explore-persona-space-data` under `issue2094_natural_task_subject_corrected/`; small aggregate JSON and the report are committed on the issue branch. No tensor is discarded before verified upload.

Provision only after preflight passes. Launch detached with a PID file and full stdio redirection; emit phase lines and a valid results sentinel, verify PID plus first log line within 15 minutes, monitor process exit/log freshness/GPU utilization, upload and exact-set verify, then terminate the ephemeral pod. The final report records planned versus realized cells and supplies a browser-accessible Hub URL.

## 11. Decision rationale and hyperparameter grounding

Source: the paper's `sections/results/02_information.tex` says retrieval failures keep task/format while changing subject and describes the context vector as resembling a task vector; `sections/05_discussion.tex` explicitly calls causal separation future work. The paper's methodology fixes Qwen2.5-7B-Instruct layer 19. Issue #2094 validates the single-position prefill-only hook and separates single-layer from all-layer interventions. Issue #2162 supplies the prior positive stated-formatting precedent, but this round re-tests it in the exact corrected pipeline rather than assuming transfer. The original inline pilot is retained only as a failure-diagnostic artifact; none of its labels or selected examples determine this bank.

Source: layer 19, Qwen2.5-7B-Instruct, and final-context-token extraction follow the paper's exact methodology and the validated issue-2094 hook convention.

Source: the separate all-layer arm and formatting manipulation check respond directly to the independent methodology critique of the inline pilot and to issue #2162's all-layer format-transfer precedent.

Source: the 768-token initial cap is more than 2× the observed answer length requested by the explicit <=220-word prompt constraint; the 1,536-token rerun is the preregistered cap-remediation rung.

Source: batch size 12 and the 3 GPU-hour fence are ungrounded — needs smoke-test.

Load-bearing assumptions needing smoke validation: forcing `Response:\n` does not prevent later patch effects; greedy decoding is stable enough for self-patch exactness; the 220-word limits avoid caps; all nine task×subject combinations are natural; and the fixed label set adequately describes outputs. Any violated assumption is reported rather than silently relaxed.
