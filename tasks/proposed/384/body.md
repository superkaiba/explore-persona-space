---
title: 'Scope the safety-tool pivot: write structured analysis of what Thread C needs
  to graduate from kernel to deployment'
kind: analysis
tags: []
created_at: '2026-05-24T09:57:55Z'
has_clean_result: false
---
## Goal

Write a structured analysis of what would need to be true for the persona-geometry-as-coverage-predictor framing (Thread C of the 2026-05-22 mentor meeting with Dan Mossing) to graduate from "interesting toy work on the Qwen2.5-7B-Instruct panel" to "an actually-deployed safety tool that predicts deployment-distribution coverage gaps from training-distribution + deployment-distribution activations."

## Why this analysis is needed now

Three independent lines of evidence in this repo all point at the same gap:

- **#207 consolidates a kernel:** geometric distance predicts bystander leakage at |ρ|=0.48–0.79 across six experiments. The kernel is the most-developed candidate predictor we have for "given training on N, which M get covered."
- **The kernel has three named caveats that block the pivot:** single-training-seed across all contributing runs; out-of-fold R² near zero in #207's leave-one-trigger-out check; #237 says any SFT collapses persona geometry, so the predictive geometry may not survive the deployed intervention.
- **Dan's specific empirical claim — "training on 2 sources already gets you most of the way to training on all of them" — is not established here.** #311 marginally tests joint-source geometry, ρ=−0.348 p=0.086 N=17. No N-scaling sweep exists.

Before sinking compute into an N-scaling sweep or a multi-seed replication, the project would benefit from a deliberate think about: (a) what the "useful safety tool" actually looks like in concrete deployment terms, (b) which existing results upgrade vs which need to be redone, (c) what the failure modes of the pivot are (including: what if geometry simply doesn't predict source rate, per Thread A?), and (d) what the minimum set of experiments is between here and a defensible "we built a coverage-gap predictor."

## Scope

This is an **analysis task**, not an experiment. Output is a single structured document (markdown, written to `docs/safety_tool_pivot.md`) covering:

1. **The applied target.** What is the safety-tool deliverable? Concrete examples: "given a deployed model and a set of M deployment contexts, return a ranked list of contexts whose behavior diverges from the training-N envelope, with calibrated confidence." Pin down the inputs, outputs, and what "calibrated" means here.

2. **What the kernel buys.** Reframe #207's |ρ|=0.48–0.79 result as a coverage-prediction claim and ask what a deployment-side user would need on top: per-context confidence, false-positive/false-negative rates, robustness to seed variation, robustness to model family.

3. **The three named obstacles.** For each — single-seed kernel, weak out-of-fold R², SFT-collapse from #237 — write the prerequisite experiment that resolves it.

4. **Dan's "N=2 → most of N=all" claim.** Map it onto a concrete experiment shape on the existing 19-persona panel. Include the operational pass criterion: what coverage fraction (held-out bystander emission) at N=2 vs N=4 vs N=8 vs N=all would count as "most of the way." Pre-register an answer to "if the curve is flat from N=2 onward, what's the implication; if it's still climbing at N=all, what's the implication."

5. **What the marker-handle / behavior-bridge gap implies.** #225 says the marker is a representational handle, not a behavioral bridge. Spell out what this means for the safety-tool framing: if the deliverable is "predict where a TRAINED BEHAVIOR generalizes," the marker work transfers only to the extent that marker-spread and behavior-spread agree. Surface the evidence we have on that agreement (#102, #225, the misalignment-leak exception in #99).

6. **Threading vs the pivot.** Identify which currently-queued experiments are on-path for the pivot (e.g., #357, #310, #380, #193) and which are off-path or redundant. Propose a minimal critical-path sequence.

7. **The non-geometric pivot.** If Thread A's #380 returns a clean negative on completion-divergence → source-rate, the geometric-handle program is in serious trouble. Outline what the non-geometric backup looks like: training-data overlap, base-rate token frequencies, capability-axis position, prompt-format priors. Sketch the experiment that triages between these.

## Output

A single markdown document at `docs/safety_tool_pivot.md`, ~1500–2500 words. Each of the seven sections above gets a labeled subsection. Each "prerequisite experiment" gets a one-paragraph spec (not a full plan; the actual experiment goes through `/adversarial-planner` separately). Include a one-figure summary at the top: the current Thread-A/B/C/D state of evidence and which prerequisites unlock which downstream tasks.

## Pre-conditions

- The Thread A / B takeaways in `docs/mentor_updates/2026-05-22.md` are written (DONE).
- #380's result is NOT a precondition — this document scopes the program assuming both branches of #380 are possible.

## Risks

- Easy to drift into a literature review or a research-vision statement. Stay grounded in what's in this repo's `eval_results/` and `tasks/` — every claim about the kernel state should cite a task number.
- Easy to over-claim "this will become a safety tool." The document's job is to expose what's NOT known yet, not to advocate for the pivot.

Parent: discussion thread in `docs/mentor_updates/2026-05-22.md` Thread C.
