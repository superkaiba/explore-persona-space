---
name: Tracking File Lifecycle
description: Authoritative map of which tracking file owns what, what the lifecycle is, and who writes what
type: reference
---

| File | Owns | Who writes | Lifecycle |
|------|------|-----------|-----------|
| `EXPERIMENT_QUEUE.md` | All planned/running/completed experiments with lifecycle state | research-pm | Proposed → Gate-pending → Plan-pending → Approved → Running → Completed |
| `RESULTS.md` | Headline claims across aims, TL;DR | research-pm (diffs to user) | Append on completion, rewrite on contradicting evidence |
| `eval_results/INDEX.md` | Maps each eval_results/ dir to its aim | research-pm (auto) | Append after manager completion |
| `docs/research_ideas.md` | Aims, subtasks, phase tracker | research-pm (diffs to user for phase + subtask status) | Phase transitions on milestone; subtask check-off on completion |
| `research_log/drafts/LOG.md` | Unreviewed draft index | experimenter (writes drafts), reviewer (adds verdicts) | Drafts move to `research_log/*.md` on approval |
| `research_log/LOG.md` | Approved TLDRs only | research-pm (on draft approval) | Append on approval |
| `figures/` | Generated plots | analyzer (auto), manager (via analyzer) | Every figure must be referenced from RESULTS.md or a draft; orphans flagged by AUDIT |

**How to apply during AUDIT mode:**
- Every `eval_results/<name>/` dir should have an INDEX.md entry. If missing, add it directly.
- Every draft in `research_log/drafts/` older than 3 days without a reviewer verdict is flagged.
- Every figure in `figures/` should be grep-findable in RESULTS.md, research_log/, or drafts. Orphans → `figures/unsorted/` (move, don't delete).
- Every RESULTS.md claim should have a link to its eval_results/ directory. Broken links flagged.
- Every research_ideas.md subtask marked `[x]` should have a corresponding eval_results/ entry or approved writeup. Mismatches flagged.

**Sync cadence:** Run AUDIT weekly or after every manager completion report, whichever comes first.
