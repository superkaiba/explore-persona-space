---
name: submodule-existence-bypasses-strict-identity
description: In a static import-satisfiability scanner, an existence-only submodule fallback for `from <shared pkg> import <namespaced unit>` bypasses the strict-identity arm one level up; grep the real corpus for that import shape to size it (#2412 R1 g1)
metadata:
  type: feedback
---

When reviewing a static AST import-satisfiability gate that classifies the RESOLVED
MODULE path (e.g. strict content-identity for issue-namespaced src, lenient
symbol-existence for shared src): check the `from <pkg> import <name>` branch where
`<name>` resolves via a submodule-EXISTENCE fallback (`(pkg_dir/<name>).py` /
`<name>/__init__.py` is_file). If `<pkg>` is SHARED and `<name>` is itself a
namespaced unit, the strict arm never sees the unit — a present-but-skewed package
is silently KEPT even though the same skew one directory down (inside a namespaced
pkg) is caught by directory-grain identity.

**Why:** #2412 R1 g1 — `step5a_sibling_probe._check_module` took the
`submodule_exists` continue for `from explore_persona_space.experiments import
issue_1333 as C`, a shape present in many real `tests/test_issue<M>_*.py` files the
sync feeds it; the plan's MF3 fixture (slug-pkg form) discriminated file-vs-dir
grain but not the parent-shared-pkg form. Matched the plan's literal algorithm, so
Concern not FAIL.

**How to apply:** for any import-scanner diff, (1) enumerate the branches that
SATISFY a name without classifying the artifact the name RESOLVES to; (2) grep the
live corpus for `from <parent> import <unit>` shapes covering the protected
namespace (`grep -rn "from explore_persona_space.experiments import" tests/`) —
frequency decides severity; (3) the fix shape is: route the resolved submodule path
through the same strict classifier before `continue`. Related: [[registered_gate_quantity_substituted]]
(adjacent-proxy substitution), [[smoke_fixture_authored_with_consumer_keys]].
