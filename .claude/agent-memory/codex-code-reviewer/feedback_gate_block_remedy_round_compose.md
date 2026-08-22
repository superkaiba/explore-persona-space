---
name: gate-block-remedy-round-compose
description: Composing a round that exists because the Step 10d pre-push lint gate itself blocked (not a reviewer FAIL) — gate root-cause is the round contract, no acceptance-contract envelope, pin-has-teeth static translation (#2253 r4)
metadata:
  type: feedback
---

When a review round exists because the Step 10d pre-push LINT GATE returned
`block` on the round's own payload (a gate-tree artifact, e.g. the archive
pathspec missing a manifest the new check reads), compose it as a TIGHT
checklist round (validated #2253 r4, 2026-08-21, 27.7 KB):

- **No acceptance-contract envelope.** Prior rounds all PASSed; there is no
  prior FAIL verdict to inline. The round contract is the brief's claim list
  + the gate ROOT-CAUSE, framed up front: which leg failed (gated vs
  baseline — baseline-zero means the check isn't on main yet), why unfixed
  = fleet-wide breakage (#931 shape), and where the spec prescribes the
  remedy (quote the doc's own extend-the-set comment).
- **Pathspec-edit check 1 is an explicit old-vs-new TOKEN diff** with the
  dropped-cone hazard named (adding a token while dropping a cone silently
  NARROWS the gate's scan surface — worse than the bug being fixed;
  Critical `substantive`).
- **"Pin has teeth" translates to a static trace on a read-only twin:**
  hand-trace the parser against `git show <parent>:<doc>`'s pre-fix window
  (token set lacks the manifest ⇒ assertion trips), confirm post-fix pass,
  verify the `text.index` anchor literal is UNIQUE across the ASSEMBLED
  doc (issue_skill_text splices step bodies — grep the whole skills dir,
  not just the companion), and note the fail-loud shape (unspliced body ⇒
  ValueError ⇒ red, never silent green). The Claude twin carries the live
  red-pre-fix/green-post-fix run.
- **Environmental-red adjudication as its own check:** a reported one-off
  red the implementer root-caused as their own invocation env (bare
  .venv python without bin/ on PATH) gets a sound-vs-masks-a-defect
  adjudication duty — read the test, confirm the diff touches nothing it
  imports; sound ⇒ one-line disposition, never auto-flag.
- Sentinel arithmetic is trivial here (revision_round == impl marker
  version == 4, fresh in history) — but check the r2 precedent on the same
  task ([[brief-pinned-sentinel-and-verdict-enum]]) before assuming.

**Why:** the shape differs from both the fix-round (no prior-verdict
acceptance contract) and the merge-round ([[merge-reconciliation-review-compose]]);
without the gate root-cause framing Codex has no way to score whether the
remedy matches the failure.

**How to apply:** any round whose brief says the Step 10d (or inline
payload) lint gate blocked and the diff is the gate-surface remedy + pin.
See also [[revision-round-compose-recipe]].
