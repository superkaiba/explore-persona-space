---
name: fork-not-inherited-list-vs-parent-gate-surface
description: Diff a fork's "deliberately NOT inherited" disclosure list against the parent module's FULL gate surface — undisclosed drops hide between the named ones
metadata:
  type: feedback
---

When a fork/transcription module carries a "Deliberately NOT inherited" (or
"report §(b)") list, do not grade only the listed drops: extract the PARENT's
full gate enumeration (grep `def gate_` / raise sites in the pinned blob) and
diff it against the fork's realized gates. Undisclosed narrowings hide between
the disclosed ones.

**Why:** #2587 r1 g1 — `bank2587.py` disclosed dropping the q25 render-prefix
assert and the paraphrase-ratio gate, but the parent's gate (vi) render half
(bare `rendered.count("assistant") == 1` + the `"You are Qwen"`
default-system-injection probe) was ALSO dropped, silently, replaced by the
plan's narrower header-count form. Plan-adherent (the plan's own gate list
specified the narrow form), so Minor — but the residual (default-system
injection on empty-system contexts passing gate (vii)) was invisible from the
fork's disclosure list alone.

**How to apply:** for any pinned-parent fork commit, `git show <pin>:<parent>`
→ list every gate/assert the parent runs at the equivalent phase → mark each
as carried / replaced / dropped-disclosed / dropped-undisclosed. Severity of
an undisclosed drop: FAIL if a plan-listed gate, Minor if the plan itself
prescribed the narrower form (then it is a disclosure-list gap + a
measurement-semantics caveat for later units). Related: [[registered-gate-quantity-substituted]],
[[maximal-prefix-suffix-diff-check-tautology]].
