# Actual answer-behavior experiment: pilot accepted; main annotation running

Thomas subsequently directed **“use codex subagents to judge.”** The current route uses fresh, property-specific, no-history Codex collaboration subagents. All1,120pilot ratings are complete under `annotation_codex/`; the full main roster remains 2,048 answers × seven properties × five repeated ratings. **No behavioral readout has been fit yet.** Exact sampling temperature and model snapshot are uncontrolled/unexposed, and shared-model repeated ratings are not IID or human validation.

The earlier OpenAI route rejected the configured credential with HTTP401 before annotation. Its original configuration and sanitized failure remain separate historical artifacts under `annotation/`; they are not the current blocker and must not be relabeled as Codex evidence.

This continuation responds to Thomas's exact instruction, “redo with behavior,” after the previous requested-condition pilot and SAE-feature analysis did not measure actual expressed answer properties. No paper changes were made.

## Data and measurement audit

The source is an existing on-policy Qwen2.5-7B-Instruct bank from #2054. Each of four speaker-framing cells has 8,000 exact text/vector joins with finite mean-answer vectors of width 3,584. The capture digest identifies hidden_states[19], bare-label framing and the model-qualified source text paths. The four cells share 1,045 questions. Local hashes, row order and exact answer character spans were verified. Historical capture→text SHA sidecars are not available in the inspected source archive; the recorded capture paths and exact joins support reuse but do not supply that missing cryptographic link.

The frozen main cohort contains 2,048 answers to 512 questions in 438 duplicate-connected groups. All four speaker versions and identical question/answer components stay in one outer fold. Five folds contain [103, 103, 102, 102, 102] questions. A separate 32-answer pilot uses excluded question groups. Generic/empty answers and duplicate components remain in the main cohort. This is a selected narrative-framing population; some exact generated spans include narration or multiple voices.

Seven answer-only instruments cover expressed persona/voice, topic, warmth, confidence, formal register, language and realized format. They never see the source prompt, character, frame or vectors. Short fragments can be valid but unassessable. Persona describes evidence in the performed voice, not the source character label. Fictional presentation does not define topic. Independent root review refined these distinctions; it is AI face-validity review, not human agreement.

Full-width regularized linear probing is primary, with grouped nested tuning and train-only PCA sensitivities. The label-limited n<d regime is explicit; this measures recoverability at a fixed label budget, not an intrinsic representation ceiling. Graded expression, categorical voice/topic and surface controls are reported separately. No universal high-versus-low ordering is inferred from unlike metrics. Full recipe and predetermined comparison/null rules are in [the plan](plan.md).

## Checks and continuation

The Codex adapter validates exact answer/ID/rubric/request hashes, all repeated-judge identities, missingness and raw-to-aggregate provenance. Ten focused adapter/preflight tests pass; eight matched-readout integration tests pass. The combined collector/adapter/readout suite passed29tests before the final fresh-agent guard; changed-files workflow lint passes. Old API-envelope tests and review remain historical implementation evidence, not approval of the changed judge instrument.

The current pilot contains35 packets of32answers (1,120ratings). Main scheduling is frozen at256answers per packet:280fresh packets,71,680ratings. The entire main roster passed the content-only leakage and coverage preflight; packet inputs range from89,373 to117,443bytes. The full pilot received [qualified independent acceptance](codex_instrument_review.md). The first main packet passed the planned 256-answer completion/envelope and sampled face check; the remaining frozen roster is released. No human agreement or technical isolation is claimed. All actual requests, raw outputs and coordinator-observed completion evidence are archived.

Use `scripts/issue2564_codex_judgments.py` for import, aggregation and acceptance. The readout now defaults to the explicitly validated Codex aggregate. The old API resume instructions are historical and must not be used for this continuation. Current recipe, grouping, controls and interpretation limits are in [the amended plan](plan.md).

**Source:** Hugging Face dataset `superkaiba1/explore-persona-space-data`, revision `07460fb4f01c4a692b141eaf2a8a081352aacaf5`, prefix `issue2054_lattice`. **Task:** #2564 actual-behavior continuation; #2054 is source provenance. **Paper:** unchanged. **Status:** Pilot accepted with qualifications; main annotation running after the first complete packet passed its check. Actual-label readout has not started.

Prepared inputs are [archived and hash-verified on Hugging Face](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/eb3c7bd0765e8005d5b53212fa64f1c2c2b2bfb9/issue2564_minpair/answer_behavior_readout_20260907): 10 files, 31,254,590 bytes, including exact selected answers/vectors, rubrics, sample/source audits and the sanitized authentication failure. That initial prepared-input archive predates current Codex ratings; their separate archive will be pinned after validation.

The earlier API/readout implementation passed [independent review](independent_final_review.md) and all19 collector/readout tests; the Codex route receives a separate instrument and adapter review. [Current Codex continuation commands](codex_continuation.md) distinguish the numerical-only smoke checks from the still-unexecuted behavioral experiment.

The completed pilot is separately [archived and content-verified](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9ba0d2275c97edc444bbc2d523086fa54045b6aa/issue2564_minpair/answer_behavior_readout_20260907/raw_completions/codex_pilot_preacceptance). Its [pilot summary](codex_pilot_summary.md) reports actual label spread, coverage and descriptive repeated-agent agreement.
