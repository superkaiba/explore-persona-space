---
title: Think through what the N+M framing means concretely for this project
kind: analysis
tags: []
created_at: '2026-05-24T09:57:55Z'
has_clean_result: false
---
## Goal

Think through what Dan's N+M framing from the 2026-05-22 meeting actually means in concrete terms for this project.

## What to think about

Dan's framing was roughly: you have N parts of your training distribution and M parts of your deployment distribution you care about. An N×M eval matrix is infeasible. The proposal is to use N+M activation collection to predict where the model might go wrong, and identify (a) the subset of M environments that, evaluated together, cover the rest of M, and (b) which N additions would remedy the most-dangerous M gaps.

Open questions worth a deliberate think:

- What is N concretely in our setup? Personas? Tasks? Contexts? Some mix? The meeting notes also flagged "contexts are just as useful as personas?" — so N may be broader than persona prompts.
- What is M concretely? The same set as N? A disjoint deployment-side panel? A continuous space sampled differently from N?
- "N+M activation collection" — collection of what activations, of what model (base, post-train, post-deploy-finetune)? At what layer? Under what probes?
- "Most-dangerous M" — dangerous in what sense? High-marker emission? High behavioral deviation? Low calibration? Pick one and write down what danger means operationally.
- "N additions that remedy" — concretely, this is "add training examples sampled from persona/context X." What does the smallest meaningful addition look like?
- Does the framing assume "training on N=2 covers most of all" (Dan's stated claim) is true? If yes, what would falsify it on our existing 19-persona panel? If no, where does the framing break?

## Output

A short markdown document at `docs/n_m_framing.md` — write down what N and M map onto for this project, what a minimum demonstration would look like, and what concrete next experiment(s) would test it. Not a literature review, not a research vision — just thinking written down. ~500–1000 words.

Parent: discussion in `docs/mentor_updates/2026-05-22.md` Thread C.
