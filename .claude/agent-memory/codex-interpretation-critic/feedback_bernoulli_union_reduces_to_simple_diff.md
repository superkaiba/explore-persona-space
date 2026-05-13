---
name: Bernoulli-union baseline reduces to simple difference when one source is near-zero
description: When Bonly ≈ 0 for all bystanders, the Bernoulli union formula r_p = joint - [A + B - A*B] simplifies to joint - Aonly; stating this simplification makes the takeaway clearer
type: feedback
---

In issue #311, the headline statistic r_p uses the Bernoulli-union baseline. Since B-only fired at 0% on all 17 bystanders, the formula A + B - A*B = A + 0 - 0 = A, so r_p = joint_rate - Aonly_rate for every bystander. The body did not state this simplification.

**Why:** Without the simplification, readers may think the Bernoulli baseline is doing complex probability arithmetic, when in this degenerate case it simply becomes a direct comparison.

**How to apply:** When one source has near-zero bystander rates, flag in lens 3 (Alternative Explanations) that the "Bernoulli union" reduces to a simpler computation and that the body should state this explicitly for reader clarity.
