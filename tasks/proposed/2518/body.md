---
title: 'bootstrap_pod.sh: install Times-alike serif fonts for pod-side paper figures'
kind: infra
tags: []
created_at: '2026-08-24T05:27:21Z'
has_clean_result: false
workflow: v1
---
# bootstrap_pod.sh: install Times-alike serif fonts (pod-side paper-figure rendering fails loud without them)

## Goal

`set_paper_style('iclr')` (src/explore_persona_space/analysis/paper_plots.py:151 `_resolve_iclr_fonts`) refuses to fall back to DejaVu and raises unless one of ['Times New Roman', 'Times', 'TeX Gyre Termes', 'Nimbus Roman', 'Liberation Serif'] is installed. Fresh RunPod pods (runpod/pytorch image + bootstrap_pod.sh) ship NONE of them, so any driver that renders paper figures pod-side crashes at its figures phase. Fix: bootstrap_pod.sh installs `fonts-texgyre fonts-liberation` (apt, ~2 MB) and clears the matplotlib font cache, so pod-side figure legs work out of the box.

## Why it matters

#2476 floor-sensitivity-sweep run 1 (pod-2476, 2026-08-24T03:59Z) died in its smoke figures leg on exactly this (log: RuntimeError set_paper_style('iclr') ... none of the Times-alike serif fonts is installed); cost one crash-fix round + a pod relaunch. Any future pod-side figures leg pays the same until bootstrap covers it.

## Acceptance

- bootstrap_pod.sh installs fonts-texgyre + fonts-liberation (idempotent, non-fatal if apt is offline: WARN not crash) and removes ~/.cache/matplotlib if present.
- A fresh-pod bootstrap followed by `uv run python -c "from explore_persona_space.analysis.paper_plots import _resolve_iclr_fonts; print(_resolve_iclr_fonts())"` returns a non-empty list.
- Step 9c mapped tests green.

Provenance: workflow-fix-candidate (in-scope: workflow-helper script gap) emitted by the #2476 floor-sensitivity-sweep orchestrator, 2026-08-24; incident marker epm:failure-lesson v1 + the run-1 crash log on pod-2476.
