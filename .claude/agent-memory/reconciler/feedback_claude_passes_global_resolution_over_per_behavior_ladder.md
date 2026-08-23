---
name: claude-passes-global-resolution-over-per-behavior-ladder
description: Claude code-review PASSes a fold/aggregator that resolves ONE global config (adapter/slug/requirement flag) over inputs the plan's ladder selects PER UNIT (behavior); check the committed diagnostic evidence to see whether the mixed topology is the EXPECTED production shape. #1739 leg-2 r1.
metadata:
  type: feedback
---

Rule: when a verdict-bearing FOLD consumes per-unit producer state (per-behavior
adapter/slug/parity provenance) through a SINGLE GLOBAL resolution (`repaired_arm
= X if any(...) else Y`, a global `rows_restricted`/`parity_required` bool, one
`--override` flag), trace the MIXED topology before crediting a Claude PASS —
and check the ROUND'S OWN COMMITTED DIAGNOSTIC ARTIFACTS for whether the mixed
case is the expected production shape, not a hypothetical.

**Why:** #1739 leg-2 r1 (2026-08-22): plan v26's repair ladder + acceptance gate
were explicitly PER BEHAVIOR ("the D1 invocation for that behavior", gate 2 "per
behavior"), and the round's own committed D0 P2 artifact showed the degeneracy
fires for evil only (h_degenerate_min_n_hi = 6 vs 2136/5867) — so a restricted
repair for evil + v1 for syco/hall was the evidence-dictated topology. The fold's
global resolution then either (a) SystemExit'd the plan-conforming invocation
(global parity_required demanded arm7_parity rows for behaviors the plan forbids
refitting) or (b) silently marked the v1 behaviors' sanity records "incomplete"
(their rows carry the other arm slug) → pass=False → false INCONCLUSIVE-ADAPTER,
a registered KILL verdict minted on valid data with no loud error. Claude PASSed
with file:line evidence for every SINGLE-adapter branch and even asserted
"incomplete/duplicate = loud fail" — a half-fabricated checkmark (only duplicate
raised; the test at test_issue1739_arm2fix.py:499-502 pinned the fail-quiet
incomplete path). Codex FAILed with 6 blockers; reconcile sustained 2
(mixed-adapter fold + sanity-coverage fail-open, coupled) and downgraded 4
(parity-currency / resume-config / context-bootstrap / nonfinite — conservative
direction, unreachable on plan-conforming scorer outputs, or duplicate of an
open CONCERN; the standard Codex hardening-overreach ledger applied,
[[codex-hardening-beyond-minimal-port-contract]]).

**How to apply:** (1) grep the fold/aggregator for a single resolution variable
consumed across a per-unit axis the plan's §4/§7 text scopes "per <unit>"; (2)
open the committed diagnostic JSONs and read the per-unit evidence fields — if
they DIVERGE across units, the mixed topology is the production shape and any
global collapse is Real-blocking; (3) a "coverage-incomplete → treated as
unit-level FAIL" branch is the silent accomplice: verify whether every reachable
incomplete state is infra (load asserts + unconditional emission upstream) — if
so, fail-quiet exclusion feeds the registered kill verdict and sustains with the
mixed-adapter blocker. Related: [[claude-fabricates-rf-walkdown-checkmark]],
[[claude-misses-producer-consumer-key-mismatch]].
