---
name: parent-branch stranded crash-fixes invisible to reuse
description: A reused module's parent-issue branch may carry unmerged crash-fixes the main copy lacks; reconcile realized-artifact row counts vs raw corpus and diff the module against the parent branch at reuse time
type: feedback
---

Reusing #825's extractor from MAIN bypassed the degenerate-row filter that
existed only on the unmerged issue-825 branch → production crash at s57
(#1345 r6). The parent's REALIZED shards (n=4724 vs corpus 5000) were the
visible fingerprint that a filter existed.

**Why:** built-but-stranded fixes change nothing for importers of the main
copy; realized artifacts embody branch-side behavior main lacks.

**How to apply:** at reuse-fitness time (artifact-reuse checks), (1) diff the
reused module against the parent's issue branch for unmerged fixes; (2)
reconcile the realized artifact's row/cell counts against its declared input
— any shortfall means filtering happened somewhere; find and port it.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Parent-branch stranded fixes](feedback_parent_branch_stranded_fixes.md) — reused modules may lack the parent branch's unmerged crash-fixes; reconcile realized row counts vs corpus (#1345)
