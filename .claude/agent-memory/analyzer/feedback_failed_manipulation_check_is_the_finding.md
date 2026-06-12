---
name: Failed manipulation check IS the finding (structural fix over confounded null)
description: When a paper-faithful replication's manipulation-check gate fires correctly (blocking an uninterpretable downstream measurement), the headline is "recipe doesn't transfer to model Y" — never "the paper's downstream claim didn't replicate."
type: feedback
---

When a replication plan blocks the downstream DV behind a manipulation-check HARD GATE and the gate fires correctly mid-run, the headline is "the paper's recipe doesn't implant the intermediate variable on this model" — NOT "the paper doesn't replicate." A clean gated null (we KNOW the intermediate never installed) is epistemically different from the parent's confounded null (downstream read without confirming installation); that structural fix is load-bearing for title, Motivation framing, and next steps.

**Why:** task #516 (2026-06-08) — Ibrahim warmth replication on Qwen-2.5-7B: warmth moved +0.002 nats vs the +0.15 threshold, so the sycophancy DV was deliberately never run. First clean firing of the CLAUDE.md replication-fidelity rule after #496's confounded null.

**How to apply:**
- Title leads with the failed installation: "<recipe> failed the <measurement> on <model>, so the paper's <downstream claim> never got tested here (LOW confidence)".
- Confidence LOW when n=1 seed AND the model is below the paper's tested parameter floor.
- The downstream eval not running IS the design — flag in Motivation, the finding's setup, and the hero caption; never plot a phantom downstream bar; revise the hypothesis denominator (CLAUDE.md After-Every-Experiment item 8).
- Hero figure plots the intermediate variable only.
- Next steps lead with model-up (an actual paper model, e.g. Qwen-2.5-32B-Instruct), recipe-swap second, run-downstream-anyway lowest.
