---
name: linked-pins-pinned-separately
description: Two adjacent substring pins do not pin their LINKAGE — a `(cd "$ROOT" && <invocation>)` arm pinned as invocation + ROOT= separately survives dropping the cd; scratch fixtures where ROOT==WT cannot catch it (#2412 R1 g3)
metadata:
  type: feedback
---

Rule: when a durability pin protects a property that lives in the LINKAGE of
two text fragments (a resolution prefix + an invocation, a guard + the call
it guards, an env assignment + the consumer), asserting each fragment as its
own substring pins nothing — the refactor that breaks the property keeps both
fragments and drops only the connective. Demand ONE combined literal spanning
the linkage, or a behavioral fixture that can actually distinguish.

**Why:** #2412 R1 g3 — the Step 5a arm runs its probe helper via
`(cd "$ROOT" && uv run python scripts/step5a_sibling_probe.py …)` so the
MAIN-checkout copy executes, never the fork-era worktree copy. The pin test
asserted the invocation string and the `ROOT="$(… git-common-dir …)"` line as
two separate `in arm` checks; dropping the `cd "$ROOT"` subshell would leave
both green AND every end-to-end repro green, because the scratch fixtures are
standalone clones where git-common-dir resolves ROOT==WT — the harness is
STRUCTURALLY blind to exactly this property. Persisted as concern
`pin-cd-root-linkage-unpinned`.

**How to apply:** for each pinned pair, ask "what one-token deletion breaks
the property while keeping both pinned substrings?" If one exists, extend a
pin to span the connective (`(cd "$ROOT" && uv run python scripts/…`). Also
check whether the repro fixture COLLAPSES the two sides of the property
(scratch ROOT==WT, single-host "remote", same-process "subprocess") — a
collapsed fixture certifies nothing about the split case; say so in the
verdict. Pairs with [[fails-pre-fix-probe-parent-commit]] (that probe
validated the rest of this round: the parent-era doc swap made the repro fail
at the DEFECT, old arm keeping the poisoned file — run it before crediting
fails-pre-fix claims).
