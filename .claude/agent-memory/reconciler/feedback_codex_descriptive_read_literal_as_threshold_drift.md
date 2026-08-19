---
name: codex-descriptive-read-literal-as-threshold-drift
description: Codex FAILs a hardcoded literal in a DESCRIPTIVE/post-hoc read as registered-threshold contract drift; check which deliverable the reuse-unchanged clause binds and whether the field name self-describes the literal
metadata:
  type: feedback
---

Codex flags a duplicated numeric literal (e.g. `p < 0.05` instead of
`A.HOLM_ALPHA`) in a DESCRIPTIVE-only / post-hoc read as the round
contract's "prohibited threshold drift" — blocker-grade — when the
registered-decision path in the SAME script correctly routes through the
registered constant, and the descriptive read was never in any registered
family.

**Why:** #2333 code-review r7 (9a-ter): Codex Major on
`issue2333_followup_cells_continuation.py:260-261` (`< 0.05` literal). The
dispatch note's "reuses the registered stats helpers unchanged" clause
bound Deliverable 2 (lattice recount), which DID use `A.HOLM_ALPHA`
(`:343`) — Codex's own bug-class sweep conceded every registered decision
used `A.*`. The disputed literal sat in Deliverable 1's per-cell read,
stamped DESCRIPTIVE ("no per-cell family was registered ... any p<0.05
read at cell grain is post-hoc"), with the field name
`p_wilcoxon_raw_below_0.05` self-describing the literal — so Codex's
"internally contradictory artifacts on alpha change" impact scenario was
wrong (the field stays self-consistent by name; metadata `holm_alpha`
describes the Holm-family constant actually used). Composite of
[[codex-overreads-plan-prose]] (contract clause read as binding the wrong
deliverable) + [[codex-hardening-beyond-minimal-port-contract]]
(maintainability improvement inflated to Major). Adjudicated PASS,
Standing-only.

**How to apply:** on a threshold-drift blocker, check THREE things before
upholding: (1) which deliverable/section the contract's reuse-unchanged
clause grammatically binds — the registered-decision path there is the
test surface; (2) whether any decision path (verdict, gate, label)
consumes the literal-bearing value, or it is descriptive-only with the
descriptive disclosure persisted in the artifact; (3) whether the emitted
FIELD NAME self-describes the literal (a `..._below_0.05` field computing
`< 0.05` cannot become internally contradictory under a constant change).
All three benign ⇒ Real-nonblocking / Standing-only, not a FAIL. A literal
consumed by a REGISTERED verdict/gate path remains genuine drift and
upholds.
