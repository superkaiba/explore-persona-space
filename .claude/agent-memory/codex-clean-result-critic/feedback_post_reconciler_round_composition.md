---
name: post-reconciler-round-composition
description: "Composing the round after a binding reconciler REVISE: add a per-item verification-duties section, an explicit do-NOT-re-raise for DISCARDED findings, and resolve brief shorthand paths before baking them in"
metadata:
  type: feedback
---

When the brief carries a binding reconciler outcome from the prior round
(the PASS-vs-FAIL split shape), the composed prompt gets THREE extras on
top of the normal full-history re-review (validated on #2388 r3,
2026-08-24):

1. A "ROUND-N VERIFICATION DUTIES" section with one lettered item per
   upheld blocker/concern, each demanding a quoted body span per the
   grounding rule (an unquoted "fixed" claim is discarded), PLUS a
   matching per-item block in the output template (`FIXED|NOT FIXED —
   <quoted span>`).
2. An explicit DO-NOT-RE-RAISE instruction for every finding the
   reconciler DISCARDED (e.g. #2388's stacked-title, which had a
   deferral row) — without it Codex predictably re-raises the settled
   scope and burns a round on a non-binding blocker.
3. Brief-supplied supplementary read targets may be SHORTHAND — #2388's
   brief said `experiments/issue_1739/judging.py` but the real path was
   `src/explore_persona_space/experiments/issue_1739/judging.py`.
   Resolve + existence-check every extra path (grep the body for what
   IT cites) before baking it into the prompt; a dead path converts to
   a data-access-blocked non-PASS (same class as #489/#550).

4. (Validated #2388 r4, 2026-08-24) When a disputed finding was
   adjudicated against GROUND TRUTH in an artifact — especially one not
   on main (e.g. `gap_report.json` committed only on the issue branch /
   worktree) — inline the artifact VERBATIM as an extra named envelope
   (same `command:`/`exit code:` shape), give the absolute on-disk
   fallback path, extend the Step 4 envelope-guard REQ list to include
   it, and instruct Codex to verify the body's claim against the
   envelope BEFORE re-raising. A path-only reference risks a
   data-access-blocked non-PASS on exactly the item the round exists to
   close.

**Why:** the reconciler discards ungrounded/re-raised findings as
non-binding, so a round-3 prompt that doesn't pin these produces waste
verdict content; and the composer spec's Step 1b existence check only
covers the four required paths, not brief extras.

**How to apply:** any round whose brief mentions `epm:review-reconcile`
/ upheld blockers / discarded findings. See also
[[delta-rounds-beyond-r3]] for the delta-scoped variant.
