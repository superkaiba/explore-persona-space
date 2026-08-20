---
name: failopen-rc-routing-and-parent-diff-triage
description: A dispatcher's exit-on-rc only contradicts a FAIL-OPEN gate if the DRIVER doesn't convert the designed fallback into rc-0 + a verdict artifact — check the driver's catch site before flagging; and review thin forks by parent diff, not line-by-line (#2389 R1 g6)
metadata:
  type: feedback
---

Two recipe halves from #2389 R1 g6 (dispatch.sh + three 2329-fork scripts,
113 KB commit, PASS):

1. **FAIL-OPEN vs `exit "$rc"`:** a dispatcher that exits on every nonzero
   driver rc LOOKS like it contradicts a plan's "engine failure ⇒ FAIL-OPEN
   to HF path" clause. The contract is satisfied when the DRIVER catches the
   designed failure (engine init `except Exception` → status JSON recorded →
   proceed on fallback → exit 0), so the dispatcher's rc arm only fires on
   genuine crashes. Grep the driver's catch site (`fail.open|except.*engine`)
   BEFORE flagging the dispatcher. Sibling: a deliberately ignored rc (a
   killed advisory poll) is fine only when both the in-line comment and the
   plan name the inertness (late claim = routing freeze).

2. **Thin-fork triage by parent diff:** for "thin fork of issue<M>_X.py"
   claims, `diff parent fork | grep -vE '<issue-number renames>'` settles in
   one command whether the display/provenance/report machinery is inherited
   verbatim from a user-approved parent (then only the constant flips need
   review) or carries undeclared logic edits. #2389's dashboards fork: 75
   diff lines, ALL prefix/constant/string — the entire 1,172-line file
   certified without re-reviewing inherited machinery. Also settles
   test-coverage parity (parent had no tests either ⇒ no new gap).

**How to apply:** any pod-dispatcher review with a fail-open gate leg runs
check 1; any `issueN_*` fork commit runs check 2 first and scopes the close
read to the residual diff. Related: [[vllm-port-terminal-and-selfref-parity]],
[[handrolled-pod-sentinel-envelope]], [[sentinel-path-outside-drain-glob]].
