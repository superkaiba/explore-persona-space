---
name: brief-phase-all-mismatch
description: Re-launch briefs drift from the script's real argparse surface (--phase all, flags that don't exist). Check the previous round's epm:run-launched cmd and prefer the on-pod launch wrapper before retyping a command.
metadata:
  type: feedback
---

Re-launch briefs routinely specify flags the dispatcher doesn't accept: `--phase all` against phased dispatchers that only take named phases (argparse exits 2 in ~1s, empty log, no traceback — burned #389 v6, where the canonical entry point was the on-disk `launch_issue_389.sh` wrapper from round 1), and nonexistent flags like `--cell-specs`/`--epochs` (#477 v6).

**How to apply (before executing any brief's literal command):**
1. Read the previous round's `epm:run-launched` note for the actual `cmd='...'`.
2. Check the pod for a `launch_issue_<N>.sh` / `dispatch.sh` wrapper and prefer it — it encodes phase ordering, parallelism, and logging.
3. Grep the script's argparse `choices=[...]` for anything that smells off.
4. Stale flags whose drop is UNAMBIGUOUS (verified zero scope change) get dropped per the experimenter.md stale-flag protocol — launch with the corrected command and state the dropped flags + effective scope in the `epm:run-launched` note; don't bounce code-class.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Brief flags drift from argparse](feedback_brief_phase_all_mismatch.md) — verify --phase choices + flag existence against the script; prefer the previous round's cmd / on-pod wrapper (#389 v6, #477 v6)
