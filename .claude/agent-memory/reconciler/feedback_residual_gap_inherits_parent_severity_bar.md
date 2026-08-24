---
name: residual-gap-inherits-parent-severity-bar
description: "Codex Major on a residual gap in a fix implementing a previously-downgraded finding — compare trigger preconditions against the parent: a residual needing strictly MORE misconfiguration than a CONCERN-graded parent inherits at most CONCERN (defer BLOCKER + re-raise); reverse-ancestry (candidate CONTAINS protected tree) is a Claude containment blind spot"
metadata:
  type: feedback
---

When a round IMPLEMENTS the fix for a finding a prior reconcile DOWNGRADED
(BLOCKER→CONCERN), and the Codex twin then flags a residual gap in the new
gate as a fresh Major, adjudicate by comparing TRIGGER PRECONDITIONS against
the parent finding's: a residual that fires only under strictly MORE
misconfiguration than the CONCERN-graded parent required inherits at most
CONCERN — persist via `defer-concern --by reconciler` on the forwarded
BLOCKER row + re-raise at CONCERN (the r4/r5 #2479 ledger shape), never a
verdict-blocking FAIL. A categorical escalation flips this only when BOTH
hold: the consequence class worsens (e.g. git-recoverable clobber → secrets
exfil to Hub) AND the trigger is live on an actual deployment lane.

**Why:** #2479 r5 — Codex Major `smoke-root-ancestor-escape`:
`validate_smoke_root` checked `resolved.is_relative_to(repo)` only, so a
`/tmp`/`$TMPDIR` candidate that CONTAINS the repo passed, and the driver
bulk-publishes + partially quarantines the whole root. Property verified
REAL, but trigger = repo under the temp area (false on every lane: VM /home,
pods /workspace, GCE clones, SLURM home-dir staging) OR a DOUBLE override
(TMPDIR ancestor-of-repo + smoke-root beneath it). The r4 parent
(`smoke-root-production-poisoning`, fully unrestricted root, ONE override)
was already CONCERN-grade — the narrower residual cannot out-rank it. PASS
+ persisted CONCERN carrying Codex's 3-line fix (reject both ancestry
directions + fake-repo-under-tmp regression test).

**Second datapoint (#2479 r9, `panel-sha-binding-builder-only`):** Codex
BLOCKER'd that the r9 panel-sha binding lives only in the manifest BUILDER
while P1 generation never revalidates the panel it loads — factually TRUE,
but the r8 parent (`panel-invariance-proof-remains-heuristic`, CONCERN) had
itself accepted "P0 remains the fail-loud backstop", and the verified chain
(live `_filter_pool_feasible` with launcher-env live-panel config →
`restrict_pool_to_manifest` containment assert :1356 → engine init :2053)
fails loud pre-GPU on any eligibility-changing edit; an eligibility-
preserving edit is an audit-pin gap, not corruption (sample membership is
what the manifest pins; eligibility recomputes live). Residual = strictly
more misconfiguration (edit + no builder re-run) than the r8 state (no sha
check anywhere). PASS + re-raise at CONCERN + defer, fix prescription
(consumer-side sha compare) in the evidence. Key trace for this shape: read
the launcher to confirm the consumer gets LIVE inputs, then order the
filter/assert/engine-init lines — "recomputed live + asserted pre-spend"
is what demotes a missing-revalidation BLOCKER.

**How to apply:** (1) verify the code property yourself (here: ancestor not
relative-to descendant ⇒ falls through to the allowlist branch); (2) list
the residual's trigger preconditions vs the parent's — count required
overrides/misconfigurations and check whether any deployment lane satisfies
them today; (3) strictly-more-misconfig + no live lane ⇒ demote, defer +
re-raise, fix prescription in the CONCERN. Claude-side blind spot to watch:
containment escape analyses cover symlink / relative-path / inside-repo but
miss the REVERSE direction (candidate CONTAINS the protected tree) — Claude
r5 even noted the pathological-TMPDIR case and still read it as documented
policy without tracing the contains-the-repo consequence. Related:
[[codex-env-override-poisoning-chain-untraced-leg]],
[[codex-hardening-beyond-minimal-port-contract]].
