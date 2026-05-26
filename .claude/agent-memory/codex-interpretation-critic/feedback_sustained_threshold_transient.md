---
name: feedback_sustained_threshold_transient
description: "Sustained 5%" defined as two adjacent checkpoints only — many bystanders cross then fall below again; body should disclose the threshold is transient not persistent
metadata:
  type: feedback
---

When a body defines "sustained 5%" as "rate >= 5% at this checkpoint AND at the next," some bystanders cross by this definition but are below 5% at most later checkpoints. In task #385, helpful_assistant crossed at step75 but was below 5% at steps 150, 300, 800, 1200, 1600. The body should disclose the "sustained" definition as two-consecutive-checkpoint minimum, not a permanent crossing, and note that some crossed bystanders regress.

**Why:** The word "sustained" implies persistence through training. If many "sustained" crossers later fall below threshold, the framing misleads about the stability of the effect.

**How to apply:** When a body claims bystanders "crossed sustained X%," check the per-bystander rates at all later checkpoints and flag any that regressed below the threshold — especially if this affects a non-trivial fraction of crossers.
