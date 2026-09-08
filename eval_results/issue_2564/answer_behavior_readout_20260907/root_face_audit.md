# Answer-expression rubric face audit

Date: 2026-09-07. Reviewer: root Codex agent.

This is an AI face-validity audit by a reviewer who knows the research question. The reviewer read all 32 answer-only pilot items and the seven rubrics, without opening their source metadata or key. It is not human validation, inter-rater reliability evidence, or an unprimed blinded comparative read.

The labels must describe the exact complete generated span associated with each observed answer vector. Requested behavior, source character identity, unseen questions and SAE categories are not target labels. The packet contains short fragments, non-English language, narration, and exchanges with multiple speakers. These are genuine properties of this source population and require explicit handling rather than silent removal or imagined context.

## Corrections requested before API annotation

1. Warmth must permit an unassessable response for nonlinguistic or inadequate fragments (for example `df1b8a9763c03d6109aa` and `3b603a157dc5934f0644`). Unassessability is not coldness. Define whole-span aggregation for warmth, confidence and formality when there are several speakers or narration.
2. Narrative presentation alone must not determine the persona category. `d5ee23822100d5ffa303` describes a pirate in verse without necessarily performing that identity; `3e6f9afed9b1051891e0` narrates cooking without clear interpersonal role evidence. Distinguish several clearly evidenced roles (mixed) from insufficient evidence for a decision (ambiguous). A name does not establish a persona.
3. Classify topic by substantive subject rather than fictional presentation. The cooking narrative is everyday life; `e9733de4564d495c6e3b` discusses programming in a narrative. Define how ties between substantive topics are handled, including books/cycling (`db24af87647793a2a383`) and AI applications to insurance (`530fc602fd9abdea7b54`). Generic greetings or offers to help do not license reconstruction of the unseen question.
4. Format needs explicit handling of numeric and incomplete spans. Distinguish actual emptiness from uninterpretable fragments in reporting even if a frozen category combines them.
5. Valid responses marking a target unassessable must count as valid instrument outputs, separately from parser/transport errors. Report target availability separately. Do not relax semantics to satisfy a numeric-target completeness threshold.

## Analysis and reporting implications

Version and freeze any rubric revision before calls. Report assessability, disagreement, class prevalence, capped spans and multi-voice composition. Keep the predeclared single-speaker sensitivity and identical availability masks for context-versus-observed comparisons. The 1,045-question common intersection of narrative-framed banks is a selected population. This is an empirical readout comparison under a fixed annotation instrument and sample budget, not a universal or intrinsic property-decoding ceiling. Surface targets do not establish comprehensive syntax, and categorical persona and topic metrics do not share identical task difficulty.

These comments were sent to the experiment agent before the pilot launched. The agent is responsible for recording final rubric versions, pilot results and any changes in the actual run artifacts.
