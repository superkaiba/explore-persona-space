---
name: No Orphan Figures
description: 75% of figures were orphaned (not referenced from any document) - always reference figures from RESULTS.md or research_log
type: feedback
---

As of 2026-04-11 audit, 122/163 figure files (75%) were never referenced from RESULTS.md or any research_log/ draft. They exist only in the filesystem with no documentation of what they show.

**Why:** Orphan figures waste disk space and create confusion about what's been analyzed vs. generated-but-forgotten. Figures are only useful if documented.

**How to apply:**
- Every figure generated MUST be referenced from either RESULTS.md or a research_log/drafts/ entry
- When writing analyzer drafts, include `![description](figures/filename.png)` for every plot generated
- Prefer PDF for publication-quality, PNG for quick viewing — but both must be referenced
- When cleaning up, figures with no references can be deleted
