---
title: 'verify_task_body: HF file-count check page-cap reports ''unverified'' inside
  a PASS line — full-pagination fallback for count claims on pinned tree links'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-24T17:25:01Z'
has_clean_result: false
parent_id: 2388
origin_prompt: 'clean-result-critic round 1 on #2388 surfaced: verify_task_body.py
  HF file-count check hit its page/time cap on the exact prefix carrying a count claim
  and reported unverified inside a PASS line; needs full-pagination fallback + WARN-on-failure'
workflow: v1
---
# Goal

Fix a verification gap in `scripts/verify_task_body.py`: the HF file-count check can hit its internal page/time cap while enumerating exactly the prefix that carries a body count claim, and it then reports the count as "unverified" INSIDE an overall PASS line — so a wrong count can ship behind a PASS verdict.

## Where it fired

Task #2388 clean-result gate round 1 (2026-08-24, clean-result-critic mechanical pre-pass). The body footer claims a null-file count for a pinned HF prefix (`superkaiba1/explore-persona-space-data`, `issue2388_correctness/fits/...`, pinned `/tree/<sha>` link). The verifier's file-count check hit its pagination cap on that exact prefix and emitted "unverified" prose within a PASS result. The Claude clean-result-critic then verified the count manually and found the body figure was off by one (1,444 vs 1,443) — precisely the class of error the check exists to catch, and the cap made the check blind to it while still reading as green.

## Required behavior

1. When a body count claim is adjacent to a pinned `/tree/<sha>` (or `resolve/<sha>`) HF link, the check MUST fall back to full pagination for that prefix (e.g. `huggingface_hub.list_repo_tree` with exhaustive iteration at the pinned revision) rather than stopping at the page/time cap. Cap-bounded enumeration is fine for advisory scans; it is not fine for a check whose subject is a specific numeric claim.
2. If full enumeration genuinely cannot complete (network failure, repo unreadable), the check result must be a WARN line naming the unverified claim — never "unverified" prose folded into a PASS line. A PASS must mean "verified", full stop.
3. Add/extend a test in `tests/test_verify_task_body.py` covering: (a) count claim + pinned link + paginated prefix → full enumeration path taken; (b) enumeration failure → WARN (not PASS-with-prose).

## Provenance

Surfaced as a workflow-fix prose follow-up by the Claude `clean-result-critic` during `/issue 2388` (session cmt237vj5mej9xw0u4jf0bu9d, autonomous). Orchestrator auto-filed per `.claude/rules/workflow-fix-on-bug.md` (surfaced-prose follow-ups route the same as formal candidate blocks).

Target file: `scripts/verify_task_body.py` (HF file-count check). Fingerprint: hf-count-check-page-cap-unverified-inside-pass.
