---
name: Failed manipulation check IS the finding (structural fix over confounded null)
description: When a paper-faithful replication run has its plan-level manipulation-check gate fire correctly (preventing a downstream DV measurement that would be uninterpretable), the analyzer's headline is "manipulation check failed under recipe X on model Y" — NOT "the paper's downstream claim didn't replicate." The structural fix vs the parent confounded null IS the load-bearing claim. The title and TL;DR Headline lead with "the recipe doesn't transfer to this model", not with the un-measured downstream null. Confidence LOW because n=1 seed below paper's tested floor.
type: feedback
---

When a `kind: experiment` plan blocks Phase D / downstream DV behind a manipulation-check HARD GATE, and the gate fires correctly mid-run:

- **The headline finding is NOT "the paper doesn't replicate."** It's "the paper's recipe doesn't implant the intermediate variable on this model in the first place, so the downstream claim can't be tested here without a different model or a different recipe."
- **The structural fix vs the parent confounded null IS load-bearing.** A confounded null (parent ran the same downstream DV without ever confirming the intermediate variable installed) and a clean gated null (this run knows the intermediate didn't install, so it doesn't read noise) are EPISTEMICALLY DIFFERENT outcomes that deserve different titles, different TL;DR Motivation framing, and different next-step proposals.
- **Confidence LOW even on a clean gated null** when n=1 seed AND model is below paper's tested parameter floor — the null may say more about the model than the recipe.
- **The downstream eval did NOT run is the design** — flag it explicitly in TL;DR Motivation, TL;DR Findings setup, and the hero figure caption. Never plot a phantom downstream bar. Per CLAUDE.md "After Every Experiment" item 8, also revise the hypothesis denominator to match actual coverage.
- **Next steps lead with the model-up follow-up** (move from below-paper-floor model to an actual paper model — `Qwen-2.5-32B-Instruct` for the Ibrahim replication), NOT with "re-run on a stronger recipe at the same model size." The model floor is the cleanest variable to isolate.

**Why:** This is the load-bearing payoff of the CLAUDE.md replication-fidelity rule (match the paper's data + recipe first before reading a null). When that rule fires correctly on a fresh run, the analyzer's job is to surface the methodological win, not to bury it under "another null." Incident: task #516 round 1, 2026-06-08 — first clean execution of the rule after the #496 confounded null taught us the lesson.

**How to apply:**
- Title: "<recipe X> failed the <measurement> on <model Y>, so the paper's <downstream claim> never got tested on this model (LOW confidence)" — lead with the FAILED INSTALLATION, not the un-measured downstream.
- TL;DR Motivation: name the parent confounded null + name what the gate buys us this time.
- TL;DR Findings: the failed manipulation check IS the result H4. Plot only the intermediate variable; the missing-by-design downstream eval is documented in the figure caption + the read paragraph, not graphically.
- Next steps: model-up first (test an actual paper model), recipe-swap second, observational-extra (run downstream anyway as a curiosity) lowest priority.
