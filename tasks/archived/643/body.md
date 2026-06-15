---
title: Integrate Self-Harness (self-improving agent harness) into persona-space workflow
kind: infra
tags:
- needs-thomas
created_at: '2026-06-15T17:50:09Z'
has_clean_result: false
---
Paper: Self-Harness: Harnesses That Improve Themselves (arXiv 2606.09498, https://arxiv.org/abs/2606.09498). Authors: Hangfan Zhang, Shao Zhang, Kangcong Li, Chen Zhang, Yang Chen, Yiqun Zhang, Lei Bai, Shuyue Hu.

Method: an LLM-based agent improves its OWN operating harness with no human engineer / no stronger external model. Iterative 3-stage loop: Weakness Mining (find model-specific failure patterns from execution traces) -> Harness Proposal (generate diverse but minimal harness edits tied to those failures) -> Proposal Validation (accept an edit only if it passes regression testing). On Terminal-Bench-2.0 with three base models it lifted held-out pass rates substantially (e.g. 40.5%->61.9%), and edits were concrete/executable, not generic instructions.

What to integrate / where it plugs in: the Weakness-Mining -> Proposal -> regression-Validation loop is a tooling pattern for the EPS experiment/eval harness itself, not a steering result. (a) Run Weakness Mining over our own agent/eval execution traces (e.g. auto-experiment-runner / experiment-runner failure traces) to surface model-specific failure patterns automatically instead of hand-debugging. (b) Use the Proposal+Validation gate as a regression-tested, auto-improving layer for our eval/steering harness so harness tweaks are only kept when they pass held-out checks. (c) The harness is model-specific by design, which matters because we work across model families (Qwen-2.5-7B etc.) -- could auto-tune the harness per base model.

First steps: read the full method + check for released code; map their 3-stage loop onto our existing experiment/eval harness components; pick one narrow failure mode in an existing EPS eval as a pilot for the Weakness-Mining -> regression-validated-edit loop.

Filed from my-goat per Thomas 2026-06-15; integration scope TBD by Thomas.
