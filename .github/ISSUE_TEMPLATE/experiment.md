---
name: Experiment
about: Propose a training / eval run. Will flow through /issue <N> workflow.
title: "[Experiment] <one-line question being answered>"
labels: ["type:experiment", "status:proposed"]
---

<!-- Fill in all sections. Fields marked (required) are needed before the
     clarifier will let the issue advance to adversarial planning. Minor
     unknowns are fine — state what you don't know and the clarifier will
     ask. -->

## Motivation (required)

What prior result or research gap motivates this experiment? Link to issues,
RESULTS.md entries, or papers.

## Hypothesis + prediction (required)

State the hypothesis as `if X then Y`, with a quantitative prediction:

- **Hypothesis:** ...
- **Prediction:** ... (e.g., "EM coupling drops by ≥30%")
- **Kill criterion:** ... (what result would falsify)

## Design (required)

- **Base model:** ...
- **Training method:** SFT / DPO / LoRA / full finetune
- **Data:** dataset name, version, size
- **Eval:** which suite, which metric, how many samples
- **Baseline:** which prior experiment / issue this compares against

## Compute estimate (required)

- **Target pod:** podN (justify choice)
- **GPU-hours (rough):** ...
- **Compute label:** small (<5h) / medium (5-20h) / large (>20h) — apply the
  matching `compute:*` label on this issue.

## Reproducibility-relevant details

Anything out of the ordinary that affects reproducibility (custom scripts,
specific data versions, non-default hparams, etc.).

## Aim

Which aim is this under? (`aim:1-geometry` / `aim:2-localization` /
`aim:3-propagation` / `aim:4-axis-origins` / `aim:5-defense` /
`aim:6-truthification` / `aim:infra` / `aim:cross-cutting`)

## Open questions / unknowns

Flag anything you're uncertain about — the clarifier will ask follow-ups.

---

**Next step:** once this issue is submitted, run `/issue <this issue number>`.
The clarifier will read this, flag ambiguities, and on resolution advance to
adversarial-planner → approval → experiment-implementer → code-reviewer → experimenter → upload-verifier → analyzer → reviewer → auto-complete.
