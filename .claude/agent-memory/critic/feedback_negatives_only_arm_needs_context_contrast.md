---
name: Negatives-only / ablated-arm positives need a cross-context contrast
description: Source-slot-only positive criteria on single-ingredient ablation arms are near-guaranteed by uniform formatting/EOS drift; require bystander + trained-context reads and a contrast criterion (#601)
type: feedback
---

When a plan registers a POSITIVE result on a negatives-only (or any single-ingredient ablation) arm as "the source-slot logit/DV moved ≥ θ", check whether the arm's reads include ANY non-source contexts. Source-only reads make the positive fatal-unweighable: uniform cross-context drift and source-directed coupling both satisfy the criterion and the discriminating contexts were never read. Must-Fix = add the bystander panel (+ trained contexts) to the same checkpoint reads (machinery usually exists in a sibling phase) and re-register the positive as a source-vs-bystander CONTRAST.

**Why (#601 Phase 3):** 0 positives, 800 negative rows, marker-only loss → the single loss token is the trailing completion token; the registered |Δz_eos(source)| ≥ 1.0 criterion is literally "emit trailing token after a completed response" — a formatting behavior, and formatting behaviors generalize across system prompts by DEFAULT (#18/#207 uniform leakage in reverse). Phase 2 had bystander reads; Phase 3 silently dropped them. Secondary: raw per-token Δz criteria can fire on common-mode logit shifts — require the EOS margin / logZ co-read (free under the four-float contract).

**How to apply (alternatives lens):** for any implantation-mechanism plan with a control arm that removes one ingredient, ask "would generic SFT drift on this arm's actual loss tokens satisfy the registered positive at the probe slot?" If yes and no cross-context read is collected → REVISE (cheap fix, same eval machinery). Distinct from clamp/distance concerns (weighable post-hoc when per-persona reads persist) — this one's data is never collected.

Bonus design facts worth reusing (#601, NOT flaws): in fixed-question duplicated-row rigs, "same rows × k epochs" and "k× duplicated rows × 1 epoch" see IDENTICAL multisets — the repeated-epoch-memorization alternative dissolves by construction. Cumulative positive-update count is collinear with T inside such a phase; the parent's fixed-positive-count cells are the control that excludes it — make the analyzer cite that explicitly.
