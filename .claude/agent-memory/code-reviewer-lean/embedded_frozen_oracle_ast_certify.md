---
name: embedded-frozen-oracle-ast-certify
description: Certify an identity-pin test whose oracle is an embedded "frozen copy" by AST-diffing it against the parent blob's function + zero-deletion diff of the live fn (#2388 R1 g1)
metadata:
  type: feedback
---

When a diff adds a default-must-be-byte-identical kwarg and the pinned identity
tests compare against an EMBEDDED "frozen verbatim copy" of the pre-edit
function, do not trust the docstring's provenance claim — certify it in two
mechanical probes (~5 min, settles the whole plan constraint):

1. Extract the function from the parent blob (`git show <parent>:<file>` +
   `ast.get_source_segment`) and AST-compare statement dumps
   (`ast.dump(ast.parse(ast.unparse(stmt)))` per top-level stmt, docstring
   dropped) against the test's embedded reference. Expect an exact match
   modulo explicitly declared deviations (e.g. a live `_eigh_robust` import).
2. Diff the parent function vs the NEW function and require ZERO deletions —
   additions-only proves the default path's math is untouched independent of
   any test.

**Why:** an embedded oracle transcribed from the POST-edit code is a tautology
(the [[twin-transcription-parity-tautology]] shape); running the probe against
the parent blob is the [[fails-pre-fix-probe-parent-commit]] lesson applied to
identity pins. In #2388 R1 g1 both probes PASSED and turned a "trust the test"
review into a mechanical certification of the plan's byte-identity constraint
at n>d AND n<d.

**How to apply:** any `dof_cap`-style opt-in kwarg with a "default keeps
callers byte-identical" plan constraint; any test file embedding a frozen
reference implementation as its oracle.
