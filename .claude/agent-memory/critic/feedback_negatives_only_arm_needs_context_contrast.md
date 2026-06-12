---
name: Negatives-only / ablated-arm positives need a cross-context contrast
description: Source-slot-only positive criteria on negatives-only (or any single-ingredient) arms are near-guaranteed by uniform formatting/EOS drift; require bystander + trained-context reads and a contrast criterion (#601)
type: feedback
---

Rule: when a plan registers a POSITIVE result on a negatives-only (or any
single-ingredient ablation) arm as "the source-slot logit/DV moved ≥ θ",
check whether the arm's reads include ANY non-source contexts. If the reads
are source-only, the positive is fatal-unweighable: uniform cross-context
drift and source-directed coupling both satisfy the criterion and the
discriminating contexts were never read. Must-Fix = add the bystander panel
(+ the trained contexts themselves) to the same checkpoint reads (machinery
usually already exists in a sibling phase) and re-register the positive as a
source-vs-bystander CONTRAST.

**Why:** Plan #601 Phase 3 (0 positives, 800 negative rows, marker-only loss
→ single loss token = trailing completion token) registered |Δz_eos(source)|
≥ 1.0 logit as evidence of "init-live EOS-channel coupling". But
negatives-only training literally teaches "emit trailing token after a
completed response" under 4 personas — a formatting behavior, and formatting
behaviors generalize across system prompts by DEFAULT (#18/#207 uniform
leakage, run in reverse). So the positive was near-guaranteed and
uninformative as registered; Phase 2 had bystander reads but Phase 3
silently dropped them. Secondary: raw per-token Δz criteria can fire on
common-mode logit shifts — require the EOS margin / logZ co-read (four-float
contract makes this free).

**How to apply:** Alternatives lens, any implantation-mechanism plan with a
control arm that removes one ingredient (negatives-only, positives-only,
trigger-only). Ask: "would generic SFT drift on this arm's actual loss
tokens satisfy the registered positive at the probe slot?" If yes and no
cross-context read is collected, REVISE (cheap fix, same eval machinery).
Distinct from the clamp/distance concern: that one is weighable post-hoc
when per-persona reads + centroids are persisted; this one is unweighable
because the data is never collected.

Bonus design fact worth reusing (#601, NOT a flaw): in fixed-question
duplicated-row rigs, a "same rows × k epochs" arm and a "k× duplicated rows
× 1 epoch" arm see IDENTICAL multisets (same distinct rows, same per-row
exposure count) — the repeated-epoch-memorization alternative dissolves by
construction; only shuffle stratification + partial trailing batches differ.
Also: cumulative positive-update count is collinear with T inside such a
phase; the PARENT's fixed-positive-count cells are the control that excludes
it — make the analyzer cite that explicitly.
