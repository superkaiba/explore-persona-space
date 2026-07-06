---
name: worked-example-excerpt-rank-vs-ranking-DV-and-CI-spans-zero-caption
description: Two interp-critic calibration patterns from #559 r1 — (a) Claude mis-ranks a worked-example single-question excerpt against the persona-level MEAN ranking DV and wrongly calls a correct "highest/lowest of N" label false; recompute the rank using the DV the body actually ranks by; (b) a figure caption claiming "all arms' intervals cross/touch zero" that lumps in a strictly-positive or dropped arm is a false statistical-framing claim → REVISE
metadata:
  type: feedback
---

Two patterns, both from the #559 round-1 interpretation-critic reconcile
(Claude PASS vs Codex REVISE), both portable. The CELL of raw data that
settles a worked-example rank dispute is almost always a per-unit summary
array in the analysis JSON, NOT the displayed single-row excerpt.

## Pattern A — Claude mis-ranks the worked-example EXCERPT against the ranking DV (PASS-leaning miss)

When a body shows a cherry-picked worked example (e.g. "PERSONA joker ...
SLOT READ ... margin −18.05 (highest of 35)"), the parenthetical rank
label ("highest of 35", "34th of 35") describes the unit's rank under the
**ranking DV the experiment actually uses** — usually a per-unit MEAN /
aggregate across many items ("a single value per persona"). The displayed
number is one item's value (here one question's margin), shown only to
illustrate. Claude ranked the DISPLAYED single-item value, found joker
rank 6 / assistant 33, and called the body's "highest"/"34th" labels
false. The body was CORRECT: by the per-persona MEAN margin (the ranking
DV) joker = rank 1/35, assistant = 34/35. Claude's "Important catch" was
itself the error.

**How to apply.** Before crediting a Claude finding that a worked-example
rank label is "wrong", recompute the rank using the DV the body ranks by
(read §"Evaluated with" / the predictor definition — look for phrases like
"a single value per persona", "mean over ... rollouts", "aggregate per
unit"), NOT the single displayed excerpt value. Go to the analysis JSON's
per-unit array (`per_persona` / `per_source` with the component fields)
and compute the aggregate yourself. If the aggregate rank matches the
label, the body is right and the Claude finding is Discarded/mistaken. A
displayed-excerpt rank ≠ the unit's overall rank is expected, not a bug —
flag only a genuine label-vs-DV mismatch.

## Pattern B — "all intervals cross/touch zero" caption that lumps in a strictly-positive or dropped arm (FAIL-leaning, blocking)

A hero figure caption summarizing N arms with one sweeping statistical
clause ("all three behaviors ... intervals crossing or touching zero") is
FALSE — and inferentially misleading — if even one arm's CI is strictly
above (or below) zero, or is a DROPPED/untestable arm that doesn't span
zero for a different reason. #559: refusal cell-axis CI [+0.036, +0.496]
is entirely ABOVE zero (it was dropped for a base-floor kill — degenerate
predictor, not a zero-spanning null), while syco [−0.075, +0.486] and EM
[−0.093, +0.337] genuinely span zero. Even when the body's PROSE handles
the odd arm correctly (refusal "dropped, not failed"), a caption that
contradicts the prose with a false blanket interval claim is a REVISE-class
statistical-framing defect (the caption is prose for Lens 7). Recoverable
by a one-line edit, but blocking until fixed.

**How to apply.** On any interp/clean-result reconcile where a caption or
takeaway makes a blanket "all/every ... CI spans/crosses/touches zero"
claim over multiple arms, pull EACH arm's CI from the per-arm bootstrap
JSON (`cell_axis_bootstrap.ci_lo/ci_hi`, `within_panel_ranking.json` here)
and check the claim arm-by-arm. One ci_lo > 0 (or ci_hi < 0), or one
DROPPED arm whose CI is non-spanning, falsifies the blanket claim → the
Codex/critic REVISE is Real-blocking. Distinguish "spans zero" (a null)
from "dropped/untestable" (no inference possible) — the fix must not just
exclude the arm from a count, it must rephrase so the strictly-signed /
dropped arm isn't described as zero-spanning.

## Also from #559: Codex over-fire to discard (PASS-leaning calibration)
Codex's "missing raw-EM artifact" concern was mistaken — it conflated the
body's MARKER-channel worked example (which links an HF raw file that
exists + downloads) with a non-existent EM-subdir raw path the body never
references. Before crediting a "raw artifact missing → claim unsupported"
finding: grep the body for the path/link the finding assumes is there; if
the worked example is from a DIFFERENT channel/arm than the one the
reviewer named, the finding is Discarded. Family:
[[codex-passes-when-sandbox-blocks-data]] (verify the artifact the body
actually cites, at the revision it pins).

Related: [[claude-misses-lens7-statistical-framing]] (the named-test /
derived-interval-in-prose trip-wire; Pattern B is its blanket-CI-claim
sibling).
