---
name: measure-ablation-leverage-before-spending
description: Before spending on an ablation, compute the ceiling on how much it can move the headline — a channel present in k of n rated rows caps pooled movement at 100*k/n, and below a few points the null is uninformative by construction
metadata:
  type: feedback
---

Before running an authorized ablation / control, compute its LEVERAGE CEILING
first and compare it to the effect you want to detect. For a swap or removal that
only touches rows carrying some channel, the largest achievable movement in a
pooled mean is `100 * k / n` (k carrying rows of n rated) — the full scale times
the carrying share. If that ceiling is smaller than the effect under test, the
ablation's null is uninformative BY CONSTRUCTION and running it buys the
appearance of rigor, not rigor.

Often the same measurement that establishes the ceiling ALSO answers the question
the ablation was meant to answer, at zero cost, by splitting the data you already
have on the channel.

**Why:** #1345 (2026-07-31). A name-swap ablation was authorized (1,500 judge
calls) to test whether an AI-likeness axis was reading a machine-sounding
character name instead of authorship texture. Measured first: the name appeared in
10 of 300 judged rows and never in the questions ⇒ ceiling 3.34 points against a
~19-point effect. Instead of swapping, splitting the EXISTING per-item scores by
channel across all 16 cells refuted the confound outright and in the opposite
direction (name-bearing rows scored ~25 points LOWER, i.e. self-naming reads as
human narrative), which is strictly more evidence than the one-cell swap could
have produced. The no-run decision was accepted.

**How to apply:**
- Ceiling arithmetic first, in one query over the rated text: how many rows carry
  the thing you are about to manipulate? Report it as the bound, per cell.
- Prefer a SUBGROUP SPLIT of existing scores over a new spend whenever the
  manipulation is "presence/absence of a detectable feature". You get every cell
  instead of one, and no instrument change to reconcile.
- Report the bound WITH the decision, so declining is auditable rather than
  looking like skipped work — and offer the variant that would have leverage.
- Watch the direction, not just the magnitude: a channel effect can run OPPOSITE
  to the feared confound and be a methods finding in its own right.
- A subgroup split rules out only channels that are NAMED and mechanically
  detectable. State that limit; an unnamed surface correlate needs the
  human-agreement audit (llm-judging rule 15), which no number of judge calls
  closes.

Related: [[feedback_parity_gate_determinate_data_blind]] (a gate that cannot fail
on the data at hand proves nothing);
[[feedback_pilot_timing_gate_sweep_shape]].
