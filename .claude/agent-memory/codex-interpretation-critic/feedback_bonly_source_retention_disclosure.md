---
name: B-only source retention vs generalization framing
description: When B-only fails to generalize, check whether B-only also fails at source retention (below threshold). Caption saying "fires only on source itself" without stating the source rate can mislead readers.
type: feedback
---

When a B-only LoRA fires at 0% on all bystanders, the headline is "no generalization." But also check the B-only source rate. If it is below the training-success threshold (e.g., 47% vs ≥80%), the picture is double failure: no generalization AND weak source retention. This distinction matters for interpreting *why* B-only failed.

**Why:** In issue #311, B-only comedian fired at 47% on comedian itself (below 80% threshold), but the Figure 1 caption said "fires only on comedian itself" without stating the 47% rate — implying reliable source firing. The 47% was disclosed in Setup details but not in the Result 2 caption or Main Takeaways for that result.

**How to apply:** In Lens 6 and Lens 2, when a single-source LoRA produces 0% bystander generalization, also extract the source-row rate and flag if below the threshold. Check whether the result's Main Takeaways name both failure modes. Caption saying "fires only on X itself" should include the source rate.
