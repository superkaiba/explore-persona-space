# Actual answer-behavior experiment: prepared; authentication blocked

The corrected experiment is ready for its annotation pilot, but OpenAI rejected the configured credential with HTTP 401 (`invalid_api_key`) during the authenticated model check. **No answer-annotation calls, behavior labels, or scientific readout results were produced by this continuation.** There are no new scientific results to interpret yet.

This continuation responds to Thomas's exact instruction, “redo with behavior,” after the previous requested-condition pilot and SAE-feature analysis did not measure actual expressed answer properties. No paper changes were made.

## Data and measurement audit

The source is an existing on-policy Qwen2.5-7B-Instruct bank from #2054. Each of four speaker-framing cells has 8,000 exact text/vector joins with finite mean-answer vectors of width 3,584. The capture digest identifies hidden_states[19], bare-label framing and the model-qualified source text paths. The four cells share 1,045 questions. Local hashes, row order and exact answer character spans were verified. Historical capture→text SHA sidecars are not available in the inspected source archive; the recorded capture paths and exact joins support reuse but do not supply that missing cryptographic link.

The frozen main cohort contains 2,048 answers to 512 questions in 438 duplicate-connected groups. All four speaker versions and identical question/answer components stay in one outer fold. Five folds contain [103, 103, 102, 102, 102] questions. A separate 32-answer pilot uses excluded question groups. Generic/empty answers and duplicate components remain in the main cohort. This is a selected narrative-framing population; some exact generated spans include narration or multiple voices.

Seven answer-only instruments cover expressed persona/voice, topic, warmth, confidence, formal register, language and realized format. They never see the source prompt, character, frame or vectors. Short fragments can be valid but unassessable. Persona describes evidence in the performed voice, not the source character label. Fictional presentation does not define topic. Independent root review refined these distinctions; it is AI face-validity review, not human agreement.

Full-width regularized linear probing is primary, with grouped nested tuning and train-only PCA sensitivities. The label-limited n<d regime is explicit; this measures recoverability at a fixed label budget, not an intrinsic representation ceiling. Graded expression, categorical voice/topic and surface controls are reported separately. No universal high-versus-low ordering is inferred from unlike metrics. Full recipe and predetermined comparison/null rules are in [the plan](plan.md).

## Checks and continuation

The finite 32-answer pilot passed independent prelaunch review after parser, persistence and provenance fixes. Static tests cover malformed envelopes, refusals, truncation, missing scores, category ties and atomic writes. The API preflight failed before any paid annotation. A valid OpenAI credential in the existing root configuration is required; no Claude or alternative judge was used.

Resume the exact pilot with `scripts/issue2564_answer_behavior.py annotate`, the persisted pilot config and prepared root. The fixed pilot is 32 answers × 7 properties × 5 draws = 1,120 requests, followed by per-property reliability, prevalence, assessability and parser/transport checks before the 2,048-answer main wave. All raw responses and attempt-level cost telemetry persist; completion and instrument acceptance are separate. The root readout implementation can proceed without labels, but no fit should substitute requested conditions or fabricated targets.

**Source:** Hugging Face dataset `superkaiba1/explore-persona-space-data`, revision `07460fb4f01c4a692b141eaf2a8a081352aacaf5`, prefix `issue2054_lattice`. **Task:** #2564 actual-behavior continuation; source #2054 is provenance. **Paper:** unchanged. **Status:** blocked on valid OpenAI authentication, not completed.

Prepared inputs are [archived and hash-verified on Hugging Face](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/eb3c7bd0765e8005d5b53212fa64f1c2c2b2bfb9/issue2564_minpair/answer_behavior_readout_20260907): 10 files, 31,254,590 bytes, including exact selected answers/vectors, rubrics, sample/source audits and the sanitized authentication failure. No annotation results are present.
