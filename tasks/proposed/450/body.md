---
title: Read arXiv 2111.02080 (ICL as Implicit Bayesian Inference) for User Modeling
  / Persona Selection thread
kind: infra
tags:
- needs-thomas
created_at: '2026-05-30T21:02:24Z'
has_clean_result: false
---
Reading task. Source: my-goat Todoist capture queue/inbox/2026-05-30T14-00-06...need-rules... companion file 2026-05-30T14-00-08_todoist-6gm7CGm2vprQjg6v (Thomas dropped the bare arXiv link into Todoist).

Paper: arXiv 2111.02080 — "An Explanation of In-context Learning as Implicit Bayesian Inference," Xie, Raghunathan, Liang, Ma (Stanford; ICLR 2022). [confirm title/venue on open]

Why it is relevant (User Modeling / Persona Selection Model, Topic 7 in docs/research_ideas.md): the paper formalizes ICL as the model performing implicit Bayesian inference over a latent document-level "concept" and generating conditioned on the posterior. That is the formal backbone for the projects framing of generation as a draw from a latent persona/character distribution — directly adjacent to Su et al. 2026 (Character as Latent Variable) and the Simulators theoretical lineage tracked in subtask 7.2. Useful for grounding the "predict outcomes from training-data signals before training / output = sample from a learned persona-space distribution" thesis in an existing latent-variable account.

Done condition: skim + 1-paragraph synthesis noting (a) what the latent-concept/posterior formalism gives us for the persona-selection latent-variable model, (b) whether it should join the project reference list alongside Su et al. 2026 and the Simulators sequence (subtask 7.2). Add the synthesis to the User Modeling gist (835cc4d) Theoretical lineage section.

kind=infra (reading/meta), needs-thomas (judgment/synthesis, not an agent-runnable experiment).
