---
description: Replication-fidelity rule — match the paper's data + recipe FIRST when replicating a published finding; named deviations, the positive-only contrastive-negatives exemption, and the #496 incident (loads at plan time via plan-file paths)
paths:
  - ".claude/plans/**"
  - "tasks/**/plans/**"
---

# Replication fidelity — match the paper's data + recipe FIRST

When the Goal is to replicate a paper's result or test whether it holds on our
model, the FIRST run reproduces the paper's actual data source, training
recipe, and hyperparameters as faithfully as the project allows — same corpus
construction (the paper's real dataset, not a project-house synthetic
substitute), same SFT-vs-contrastive shape, same LoRA rank / epochs / lr /
checkpoint-selection, same dependent variable AND the same manipulation check
the paper used to confirm the intervention took. Change ONLY the one variable
the replication is deliberately testing (typically the base model).

Do NOT silently swap in the project's house rig (contrastive Sonnet-written
corpus, default r=32/α=64, 3 epochs, etc.) and then read a null as "the
finding doesn't replicate" — a recipe mismatch confounds the null and leaves
model-size / corpus-shape / training-rig all un-disentangled (incident #496: a
contrastive Sonnet-warmth rig produced a sub-threshold warmth→sycophancy null
where the paper used ShareGPT-rewrite plain SFT, AND skipped the paper's
warmth manipulation check, so "warmth doesn't leak" was indistinguishable from
"warmth never implanted").

Rules:

- Any deviation forced by project constraints (judge model, GPU budget, model
  size) is named explicitly in plan §-assumptions and carried into the
  clean-result as a scope caveat.
- A faithful replication of a positive-only paper is the named
  contrastive-negatives exemption (b) — do NOT bolt on contrastive negatives
  the paper didn't use (`.claude/rules/contrastive-negatives.md`).
- Pull the recipe from the paper itself (arXiv MCP), never from a secondhand
  summary, and verify the citation (author/venue) against the source.

Enforcement: `planner.md` (replication-fidelity check), `critic.md`
Methodology lens.
