---
name: Claude cites a workflow backstop with wrong semantics
description: Claude code-reviewer downgrades a real verification gap to Minor by crediting a workflow-level backstop gate whose actual FAIL condition does not cover the gap — read the named agent spec's gate before crediting it
type: feedback
---

When Claude rates a real finding Minor "because workflow gate X backstops it",
open the named agent/skill spec and read the gate's ACTUAL fail condition before
crediting the backstop.

**Why:** #594 r1 — Claude rated the missing per-probe upload-count assert Minor,
citing "the Step-8 upload-verifier's `>=50` glob gate" as backstop. The
upload-verifier's §6.5 completeness gate (upload-verifier.md:305,475) enumerates
globs ON-POD and FAILs only on ZERO files — explicitly "never on a
partial-coverage shortfall". The claimed `>=50` gate does not exist; the plan's
`>=50 files expected` is a non-binding note. The PASS survived on other grounds
(assemble-proven local completeness + atomic upload_folder commit + fail-loud
consumer), but the cited backstop was load-bearing in Claude's severity call and
was wrong.

**How to apply:** For every "the workflow catches this downstream" justification
in either verdict: (1) name the exact gate (agent spec file + line); (2) check
WHAT it gates (zero-vs-partial, on-pod vs HF-side, count vs existence); (3) if
the gate doesn't cover the gap, re-derive severity from the artifact itself and
persist the residue as a CONCERN so Step 5c-ter reads the ledger, not the prose.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Claude cites nonexistent backstop semantics](feedback_claude_cites_nonexistent_backstop_semantics.md) — read the named gate's ACTUAL fail condition (upload-verifier gates zero-files only, never partial coverage) before crediting a downgrade. #594 r1.
