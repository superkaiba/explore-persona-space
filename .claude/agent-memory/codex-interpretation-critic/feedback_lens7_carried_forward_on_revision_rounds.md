---
name: feedback-lens7-carried-forward-on-revision-rounds
description: On revision rounds (r2+), scope Lens 7 to what the round touched, using LOCAL data; carry forward prior-round verification with an explicit "satisfied in r1/r2" tag instead of a BLOCKED downgrade
metadata:
  type: feedback
---

On revision rounds (r2+), scope Lens 7 to the delta the round touched and
run it on LOCAL artifacts (judge-label JSONs, the `data/issue_<N>/judge_inputs/`
rollout mirror — see [[feedback_cross_worktree_path_split]]); for
sub-checks whose raw text is HF-only and that prior rounds already
verified, instruct Codex to fill the output line with an explicit
`satisfied in r1/r2 — delta: <none|what changed>` tag rather than applying
the network EXCEPTION-2 "entire lens BLOCKED → REVISE" downgrade.

**Why:** the EXCEPTION-2 downgrade exists for a lens whose ENTIRE audit
target is unreachable. On a delta round the lens's live target is the
round's touched blocks, which are locally verifiable; letting an HF-only
sub-check that was already satisfied in an earlier round force BLOCKED →
REVISE produces a false re-review loop the reconciler then has to dissolve.

**How to apply:** in the composed prompt, (a) name the LOCAL paths for
every v-touched block; (b) state which sub-checks prior rounds satisfied
and give the exact carried-forward line to emit (with an UNLESS clause:
if the locally verified rows contradict the prior-round result, run the
full check); (c) keep the BLOCKED/REVISE machinery only for content the
round actually needs and cannot read. Applied on #2333 r3 (2026-08-18):
the CJK intrusion scan was carried forward via `cjk_recount.json` +
"no new arms", while the item-level re-join ran fully local.

(Reconstructed 2026-08-18 from the MEMORY.md index hook after the original
file went missing from disk; the #2333 r3 application re-grounds it.)
