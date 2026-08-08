---
name: Claude APPROVEs plan resting on a per-task disk assumption framed as a code-pattern claim
description: When a plan assumption is "all N drifted/target items have valid body.md / files X" but is justified by a code read, re-run the per-item ls yourself — Claude (and the fact-checker) accept it as a code-pattern claim and never disk-check each item.
type: feedback
---

When a plan's load-bearing assumption is a claim about the PER-ITEM DISK STATE
of N targets ("all 13 current drifts have valid bodies", "every cell's adapter
is present", "each folder has X") but the plan's "How verified" column cites a
CODE read (the audit() detection loop, a list_repo_files pattern, a frontmatter
parser), Claude APPROVEs and the fact-checker confirms — BOTH read it as a
code-pattern claim and NEVER run the per-item `ls`/`list_repo_files` to check
each target individually.

**Why:** #724 r1. Plan #724 added `task.py audit --repair` to register 13
drifted task folders. Assumption 8: "all 13 current drifts have valid bodies."
The reconcile reads `_read_body(actual/"body.md")` before registering each, and
SKIPS (FileNotFoundError) any folder with no body.md. Ground truth: 12 of 13
folders were ghost husks (only `artifacts/`/`plans/`, NO body.md); only #703 was
real. So `--apply` would register 1, skip 12, exit 1, and the post-run
`task.py audit` PASS (the §8.1 success criterion + the whole deliverable) was
UNREACHABLE. `audit()` flags class-2 drift on DIRECTORY EXISTENCE alone
(`task_workflow.py:2454-2461`), so the bar for PASS is registering all 13 — but
12 have no body to read. Codex (REVISE) caught it via `find ... -name body.md`;
Claude (APPROVE) and the fact-checker both believed assumption 8 because it was
framed as a code-pattern claim, not a per-folder disk check.

**How to apply:** On any plan whose Goal depends on the per-item state of N
on-disk/on-HF targets, and whose verification cites a code read rather than a
per-item enumeration: RE-RUN the enumeration yourself (`ls -la` each folder,
`list_repo_files` each repo, parse each body.md). Compare the realized per-item
reality against the plan's "Expected: N items processed" line. If the design
reads a file/field per item before acting and silently skips on absence, count
how many items actually have that file — a skip count > 0 against a "clean
PASS / all-N-processed" claim is a Real-blocking design gap (the design cannot
reach the Goal without an amendment for the skipped items). Sibling pattern to
feedback_codex_skips_data_construction_arithmetic.md (trace the persisted
artifact yourself) — same fix, opposite reviewer: here it is CLAUDE who skips
the per-item disk arithmetic and Codex who does it.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Claude trusts a per-item disk assumption framed as a code claim](feedback_claude_trusts_disk_assumption_framed_as_code_claim.md) — "all N drifts have valid body.md" verified by a code read, not a per-folder ls; 12/13 were bodyless husks → reconcile skips them, PASS unreachable. Re-run the enumeration. #724 r1.
