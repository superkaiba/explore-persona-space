---
name: AST parse over import probes for infra claims
description: When verifying whether a library/submodule has a field/flag, use AST parsing — import-based probes produce silent false negatives
type: feedback
---

When a specialist agent needs to verify whether a field, method, or flag exists in an external library or submodule, it should use AST parsing rather than `import X; hasattr(...)` or `dataclasses.fields(...)`.

**Why:** Import-based probes silently fail when the library has transitive imports that error (e.g. `olmo_core` missing). The probe interprets the ImportError as field-absent. Two incidents on 2026-04-17 (issues #40 and #43) both produced false-negative verdicts — "submodule is pinned pre-Liger, use_liger_kernel not a field" — that triggered cascading remediation work (Option A bump proposal, allowlist filter) that turned out to be unnecessary. Both were caught only when another agent parsed the source file via `ast` and walked the ClassDef for AnnAssign nodes.

**How to apply:** In `experimenter.md` and `implementer.md`, add a rule: "When verifying library internals, use `ast.parse(open(path).read())` + walk, or run a runtime smoke that actually exercises the field. Never rely on `hasattr` alone." In `research-pm.md`, add a cross-check rule: "When a specialist's cascading infra recommendation rests on a hasattr-style probe, re-verify with AST before acting."

Incident links:
- GitHub issue #40 (Tier 2 verification, false submodule pin claim)
- GitHub issue #43 (runtime Liger verification, false FlatArguments field claim)
- GitHub issue #41 (implementer that caught both via AST parse)
