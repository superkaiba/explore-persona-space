# Codex context-only behavior forecasting baseline

## Goal

Measure how accurately context-only Codex forecasts predict Qwen2.5-7B-Instruct behavior on the manuscript cohorts, against the existing context and mapped-answer readouts under matched evaluation splits.

## Authorization and scope

The user authorized execution on 2026-09-06 and explicitly requested Codex subagents for judging. This operational plan implements the preceding literature review in `docs/paper_context_answer_map/llm_forecasting_baseline_2026-09-06.md`. No automated Claude invocation is permitted. Existing historical outcome scores are inputs, not new Claude calls.

The user subsequently approved a reduced 900-pair run and requested fair selection. The primary experiment adds zero-shot and 32-demonstration Codex forecasts to a frozen subset of the manuscript's regression regime panel. Sample exactly 100 contexts in each of three behaviors by three regimes (ID, generic, OOD). Within each ID cell, allocate the 100 slots proportionally to the original five folds; within OOD, allocate proportionally to the original corpora. Use largest-remainder rounding with lexical tie-breaking and SHA256 ordering at seed 20260906 to sample uniformly without replacement within strata. Selection reads only eligible context identities and stratum membership, never numeric outcome values, model scores, disagreement, length, or apparent difficulty. Preserve original group folds and account for clustered contexts in uncertainty; do not select one per group, which would change the context-weighted population. Freeze the selected IDs and inclusion probabilities before dispatch. Both forecast conditions and every probe comparator must use these exact IDs. Report each 100-context cell separately; the equal-regime design does not estimate traffic prevalence.

The separate claim-4 OOD extension panel is outside this first panel comparison and must not be described as covered. Text-embedding readouts and target-model self-forecasts are secondary proposals, not required to launch the requested judge baseline. The earlier full-cohort packet manifest is superseded and must not be launched.

## Construct and dependent variables

The target is Qwen/Qwen2.5-7B-Instruct at revision a09a35458c702b33eeacc393d103063234e8bc28, with temperature 1 and maximum 1024 generated tokens. Reuse the five banked on-policy continuations and their frozen labels from #1739. Validate these pins against every consumed raw record.

Evil and sycophancy use the original graded trait rubrics. The evil rubric measures malicious harm rather than general compliance with harmful requests; retain that limitation. Original grader REFUSAL outputs were omitted. The forecast therefore estimates the expected retained numeric score, conditional on a scorable answer. Do not recode refusals or missing labels to zero. Hallucination on factual QA predicts the probability of fabricated rather than correct or abstained responses. Hallucination on generic WildChat predicts the original graded hallucination trait score, also conditional on a numeric outcome. Report these instruments separately.

## Inputs and leakage controls

Use original rendered `prompt_text` from the rollout records, preserving role headers and all preceding conversation turns. Never reconstruct a shorter context, truncate text, or include generated continuations. Freeze a private cohort key before dispatch. Outbound evaluation records contain only opaque item ID and original context; the prompt additionally states target checkpoint, decoding policy, and outcome definition. Never pass source names, split names, group keys (some factual-QA keys encode gold answers), reference aliases, outcome scores, judge rationales, activation arrays, or method predictions as held-out metadata.

Few-shot examples contain only training contexts and their observed continuous mean labels. Use 32 examples (Source: #2356), selected deterministically without label balancing, with at most two contexts per group (Source: #2356). For ID evaluation, exclude the target's entire original held-out fold from both ID and WildChat training examples. For OOD and held-out generic evaluation, use only the original training pools. For hallucination, demonstrations must match the target instrument: factual-QA examples for fabrication probability, WildChat examples for generic trait scores. No development selection on held-out outcome values.

## Judge instrument and execution

Use fresh Codex subprocess subagents, each in an empty temporary working directory, with no resumed conversation and ignored user configuration. Pin the installed model configuration gpt-6-astra, medium reasoning effort, Codex CLI 0.153.4 (verified locally before launch). Use the same transport for pilot and production. This is an agent-based instrument, not a bare API model: inherited platform instructions and available tools are a limitation. Explicitly forbid tool calls, and reject any attempt whose recorded event stream contains tool use. No claim of structural tool-free isolation is made.

Each packet addresses one behavior/instrument, regime, fold and demonstration condition. Request a short rationale followed by a 0–100 numeric forecast for each opaque ID, in a strict JSON schema. Initial packet size is eight contexts; this is an ungrounded throughput setting requiring the live pilot. Preserve full contexts and demonstrations. A packet exceeding the 500,000-character input guard fails explicitly and requires a recorded split before dispatch; never silently shorten a context. Batching may influence forecasts through other visible contexts; record packet membership and order so this limitation is reproducible.

Run a blind diagnostic pilot of 48 of the selected pairs (per behavior: six ID, five generic, five OOD), two demonstration conditions, three independent repeats: 288 forecasts. Apply the same proportional strata allocation to pilot selection within the frozen sample. All prompts are frozen before the first forecast. The pilot tests transport, strict output parsing, context budget, tool abstention, and repeated-score stability; it is not used to optimize accuracy. Production uses one forecast per condition and context, totaling 1,800 forecasts; repeated pilot draws are diagnostic and not silently averaged into production. The smaller pilot is a mechanism and stability check, not a precise estimate of rare failure rates. One previously completed full-plan smoke response is historical diagnostic data and is not pooled into the reduced run.

Full dispatch requires zero tool-use violations, zero truncated or missing-output attempts, and valid exact item coverage after transport recovery. Unexpected content refusals or parse errors trigger diagnosis and a new instrument version when needed. A changed instrument must be re-piloted; never merge scores from different instruments under one label. A successful pilot is not evidence of accuracy or human agreement.

The production manifest explicitly enumerates every packet. Run at most three concurrent Codex children, checkpoint each attempt, and resume only when prompt/config/schema fingerprints and full output validation match. Persist raw events, stderr, exact prompt, final response, model configuration, CLI version, start/end time, return code and status. Keep transport failures, content refusals, parse errors and missing labels distinct; no numeric fallbacks.

## Analysis and decision criteria

Report coverage and Spearman correlation separately by behavior, instrument, regime and OOD corpus. Add MAE and RMSE on the 0–100 scale. For factual-QA hallucination, report mean-rate MSE and per-answer Brier score separately. Undefined correlations remain null with a reason. A constant training-mean comparator has undefined Spearman, not zero.

Compare against probe predictions on the exact selected 900 IDs, retaining each probe's original training pool, fold, normalization, regularization and artifact-selected layer. Whole-cohort paper correlations are not valid subset comparators. For paired differences and grouped bootstrap intervals, require matched per-context predictions from the same fitting recipe; historical train-only sidecars cannot substitute for the fair ID+WildChat fitting recipe. Live scoped HF and local audits found only aggregates in the fair output prefixes. Recover paired predictions through the existing deterministic fitting implementation and verify aggregate parity before claiming a paired win. Until that prerequisite passes, report Codex-only results with the comparison explicitly pending; do not invent paired intervals or substitute the full-cohort scores.

Group-fold assignments reuse #1739's existing deterministic group folds. Group-level bootstrap is required for final uncertainty (Source: #2356 and project OOD-fold rule); no pointwise-only generalization claim. No new representation map is fitted by the judge baseline; mapping identity and retrieval controls are inherited references, not new claims.

## Compute and persistence

Estimated GPU-hours (total): 0 for forecast generation and scoring; no GPU provision is authorized by this plan. CPU work is bounded parsing/serialization/analysis on the existing VM with shared-VM thread caps. Codex calls are remote inference through the user's requested subagent runtime. Full wall-time is measured from the actual pilot and reported before the full wave. Any later representation refit requires a separate sizing addendum within this task, not an unmeasured GPU launch.

Persist source hashes, cohort key, all prompts and responses, configuration, scripts, tests, coverage, pilot report and final analysis. Raw model outputs and downstream inputs upload to the existing private project HF dataset through repository upload tooling; no public raw-data publication. Text/JSON are unconditional persistence targets. No artifact deletion or compute termination is needed.

## Review and limitations

Before full production, independently review extraction, fold leakage, dispatch validation and the live pilot. Human/reference outcome validation remains unperformed unless separately completed; the result measures agreement with the historical outcome instrument. Do not claim that Codex accuracy validates the labels, that a prompt-only baseline has equal training supervision to a fitted ridge, or that context representations contain information absent from the complete context and model specification.
