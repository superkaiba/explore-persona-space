---
title: 'daily-fix: address-concern --summary-file + --dry-run alias'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1f1f8f5420b3
- daily-auto-filed
created_at: '2026-08-06T07:05:02Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): address-concern truncated
  a 1785-char disposition to 196 chars silently; clean_experiment_downloads rejects
  --dry-run (rc=2)'
workflow: v1
---
# daily-fix: two workflow-helper CLI ergonomics gaps (address-concern 200-char truncation; clean_experiment_downloads --dry-run)

## Workflow gap

Two small helper-CLI gaps each cost a real loss or a wasted turn on 2026-08-05:

1. **`task.py address-concern` silently truncates long summaries.** At 21:08:51Z (#1739
   session) a 1,785-char disposition summary was cut to 196 chars: "[task.py] WARNING:
   --summary was 1785 chars (cap 200); truncated at a word boundary… Dropped tail: 'call
   ONE canonical helper in src/ … byte-exact (41.74958409920785), proving pure
   re-aggregation not re-measurement…'" — the durable concern-disposition record lost most
   of its evidence text, and the agent did not re-post the tail anywhere.
2. **`clean_experiment_downloads.py` rejects `--dry-run`.** At 21:38:33Z the same session
   got "error: unrecognized arguments: --dry-run" (rc=2) — the script's preview mode is
   the no-flag default and `--apply` opts in, but CLAUDE.md § Disk hygiene documents the
   sibling janitor as `pod.py cleanup --all --dry-run`, so the flag-shape confusion
   between the two janitors is predictable.

verified-at-filing (2026-08-06T07:2xZ): `grep -n 'cap 200\|200)' scripts/task.py | head`
→ the summary cap constant is live in the address-concern path;
`uv run python scripts/clean_experiment_downloads.py --help 2>&1 | grep -c dry-run` → 0
(no such flag). Incident quotes are the miner's probed tool_result readbacks (session
2f4940f0 rows 894, 1040).

## Proposed change

1. Add `--summary-file <path>` to `address-concern` (matching the `post-marker --file`
   convention), and make an over-cap `--summary` a hard error pointing at it — never a
   warn-and-truncate that silently drops the record's substance.
2. Accept `--dry-run` as an explicit alias of the default preview mode in
   `scripts/clean_experiment_downloads.py` (no behavior change to `--apply`).

## Provenance

- fingerprint: 1f1f8f5420b3

- workflow_fix_target: scripts/task.py, scripts/clean_experiment_downloads.py
- origin: /daily 2026-08-05 problem sweep — miner 7 P12/P13 (both probed).
