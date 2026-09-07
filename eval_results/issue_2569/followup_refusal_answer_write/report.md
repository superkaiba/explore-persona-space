# Refusal-pair observed answer change by map write subspace

The observed layer-19 answer change is projected onto the output singular directions of the frozen context-to-answer map. The high-write subspace contains the 1,608 output directions paired with singular values that retain 99% of total squared singular mass; the remaining 1,976 directions form the low-write subspace.

## Energy split

| Pair group | n | Observed low-write median | Predicted low-write median | Residual low-write median |
|---|---:|---:|---:|---:|
| flip | 60 | 0.038 | 0.006 | 0.100 |
| nonflip | 40 | 0.079 | 0.009 | 0.114 |
| mid | 8 | 0.062 | 0.010 | 0.095 |
| control | 16 | 0.103 | 0.009 | 0.112 |

For the 60 refusal flips, the median observed low-write share is 0.038 (95% bootstrap CI 0.033–0.044). Thus most of the observed answer change lies in output directions the map writes strongly. The residual is more concentrated in low-write directions than either the observation or prediction, but residual and low-write are not the same decomposition.

Within the high-write subspace, the median observed–predicted cosine is 0.811; within the low-write subspace it is 0.293.

## Existing answer-SAE descriptions

The feature view is descriptive: SAE decoder directions are correlated, and existing Codex descriptions cover a selected subset rather than a random sample. Both the mean refusal direction and the typical-pair RMS ranking are retained in the JSON artifact.

### High-write component

Mean direction description coverage: 87/100 labeled, 82/100 interpretable.

Mean direction: feature 20880 — Activates on explicit refusals and warnings concerning violence, exploitation, illegal activity, sexual misconduct, and other harms.; feature 16373 — Activates on refusals, disclaimers, identity denials, and statements of inability or unwillingness.; feature 5287 — Activates on factual question answering that follows requested output constraints, sometimes with inaccurate or malformed details.; feature 7612 — Activates on polite acknowledgments, conversational closings, and offers of further assistance.; feature 20086 — Brief prompt-handling acknowledgments, readiness statements, and requests for input.; feature 4232 — Activates on enumerated or categorized collections that organize many items, examples, products, or options.; feature 12887 — Activates on refusals and moral discouragement of violent, illegal, deceptive, or unethical conduct.; feature 17649 — Activates on assistant greetings and self-introductions that announce identity and readiness to help.; feature 10341 — Highly structured explanatory or procedural answers organized into numbered sections, categories, or criteria.; feature 32520 — Activates on stepwise practical instructions for household, kitchen, cleaning, and food-preparation tasks.

Typical pair (RMS) description coverage: 90/100 labeled, 86/100 interpretable.

Typical pair (RMS): feature 20880 — Activates on explicit refusals and warnings concerning violence, exploitation, illegal activity, sexual misconduct, and other harms.; feature 16373 — Activates on refusals, disclaimers, identity denials, and statements of inability or unwillingness.; feature 5287 — Activates on factual question answering that follows requested output constraints, sometimes with inaccurate or malformed details.; feature 20086 — Brief prompt-handling acknowledgments, readiness statements, and requests for input.; feature 7612 — Activates on polite acknowledgments, conversational closings, and offers of further assistance.; feature 4232 — Activates on enumerated or categorized collections that organize many items, examples, products, or options.; feature 12887 — Activates on refusals and moral discouragement of violent, illegal, deceptive, or unethical conduct.; feature 10341 — Highly structured explanatory or procedural answers organized into numbered sections, categories, or criteria.; feature 17649 — Activates on assistant greetings and self-introductions that announce identity and readiness to help.; feature 23806 — Activates on standard greetings, polite acknowledgments, introductions, and offers of assistance.

### Low-write component

Mean direction description coverage: 5/100 labeled, 3/100 interpretable.

Mean direction: feature 16393 — Activates on directive phrases that point users toward external resources, references, or next actions.; feature 12887 — Activates on refusals and moral discouragement of violent, illegal, deceptive, or unethical conduct.; feature 31090 — Activates on very short constrained outputs such as object choices, labels, brief summaries, and name selections.

Typical pair (RMS) description coverage: 6/100 labeled, 4/100 interpretable.

Typical pair (RMS): feature 7026 — Activates on explicit refusals and redirections, including safety-motivated and persona-motivated declines.; feature 673 — Complete-refusal language for harmful or illegal requests, often emphasizing danger, illegality, or safer alternatives.; feature 16393 — Activates on directive phrases that point users toward external resources, references, or next actions.; feature 31090 — Activates on very short constrained outputs such as object choices, labels, brief summaries, and name selections.

The high-write component has broad existing label coverage and its nearest described features prominently include refusal and safety language. The low-write component has only 5–6 labeled features in its top 100, so the current descriptions do not support a general semantic characterization of that component.
