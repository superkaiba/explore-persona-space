---
title: Run LOGS are not an artifact class in the upload policy and have no shared
  persist helper — logs die with the pod unless upload-verification happens to catch
  them
kind: infra
tags: []
created_at: '2026-08-27T01:38:35Z'
has_clean_result: false
parent_id: 2546
origin_prompt: 'Surfaced driving #2546 arm 3 through /issue: upload verification FAILED
  on 142 files / 5,063,198 B of local-only text because issue2546_dispatch.sh has
  no log-upload call site; arm 1 had needed the same manual upload. Checks: no shared
  log-persist helper exists anywhere in src/ or scripts/; 44 of 78 per-issue dispatchers
  have zero log-upload references (crude grep, caveated); and .claude/rules/upload-policy.md
  has no artifact-table row for logs at all — its ''log'' mentions are about logging
  behavior and 0-byte-log wedge detection.'
workflow: v1
---
---
kind: infra
---

# Run LOGS are not an artifact class in the upload policy and have no shared persist helper — every issue's logs die with the pod unless upload-verification happens to catch them

## The gap

`.claude/rules/upload-policy.md` carries an artifact table (eval JSONs, raw completions, LoRA
adapters, datasets, figures, training metrics, intermediate analysis tensors) and the standing
"**Persist by default** — upload every artifact a run produces, even if this task has no use for it"
rule. **Run logs appear nowhere in it.** Every occurrence of "log" in that rule file is about
logging BEHAVIOR (`Log which path was taken`) or about using a 0-byte log as a wedge signal
(#1739) — never about logs as an artifact class to persist.

And there is no shared helper to persist them: a search for `upload_logs` / `persist_logs` /
`def .*logs.*upload` across `src/` and `scripts/` returns **nothing**.

Consequence: log persistence is per-dispatcher-author discretion. A crude grep over the 78
`scripts/*_dispatch.sh` files finds **44 with zero log-upload references and 34 with one or more**.

**Honest caveat on that 44/34 split:** the pattern used was
`upload.*log|log.*upload|logs/.*upload|upload_folder|hf.*logs`. A zero does NOT prove a dispatcher
never persists logs (it could route through a differently-named helper), and a nonzero does NOT
prove it does (`upload_folder` matches unrelated uploads). Treat the split as evidence of
INCONSISTENCY, not as a precise count of broken dispatchers. The two facts that need no
qualification are: no shared helper exists, and the policy has no row for logs.

## Verified instance (#2546)

`scripts/issue2546_dispatch.sh` has **no log-upload call site anywhere** — established by an
upload-verifier reading the actual call sites, not by grep. Result: arm 3's upload verification
FAILED on **142 files / 5,063,198 B** of local-only text, and arm 1 had required the same manual
upload earlier in the same task. Two arms, two manual repairs, same cause.

What was at stake in the arm-3 set:

- `issue-2546.log.p6fail-*` (238,482 B) — the ONLY line-level record of the first `p6_publish`
  failure, which is the forensic trail for filed defect #2611.
- `work/fits_a3/slot{0..3}.log` (870,958 B) — the ONLY line-level record of a 43-unit, ~2.1 h fit
  phase, including every per-unit wall-clock measurement.
- 128 out-root worker logs (1,905,956 B) from capture / rel-capture / generation.
- 7 rotated dispatcher logs, plus `launch_issue_2546.sh`, `revisions.json`, `fallbacks_a3.env`.

None of it is regenerable: it is the record of what happened, not a derived artifact. Had
upload-verification not run — or had it verified only the final phase's prefix (the #1773 shape) —
all of it would have been destroyed at pod termination, and the evidence behind two filed defects
with it.

## Why the current arrangement is fragile

Logs are presently persisted only when THREE things all happen: the upload-verifier runs, it
enumerates local-only text rather than just checking remote prefixes, and an agent then performs a
manual upload. That makes a durability guarantee contingent on a review step catching an omission.
Every other artifact class in the table has an automated path; logs uniquely rely on the gate.

The asymmetry also shows in the failure mode. A missing eval JSON fails loudly downstream (a
consumer errors). Missing logs fail SILENTLY and only matter later, when someone tries to diagnose
a run that has already ended — precisely when the evidence is gone.

## Recommended fix

1. **Add a `Run logs` row to the upload-policy artifact table**, with a destination convention. The
   realized convention this project already uses is
   `<data-repo>/issue<N>_<slug>/logs/<arm-or-scope>/…` — #2546 arm 1's logs live at
   `issue2546_cotmap/logs/arm1/`, so the convention exists in practice and only needs writing down.
   Logs are text/JSON, so they take the unconditional non-LFS upload path and are open even over the
   #541 LFS quota. Note the existing text rules apply: never gzip (`*.gz` is LFS-matched), and
   line-split any single file over ~9.5 MB.
2. **Provide the shared helper** (e.g. `orchestrate.hub.upload_run_logs(issue, scope, paths)`) so
   dispatcher authors have one call rather than a per-issue reinvention. The absence of a helper —
   not author carelessness — is the root cause of the 44/34 inconsistency.
3. **Invoke it from the dispatcher contract**, at phase end and/or in the EXIT trap, so a FAILED or
   killed phase still persists its logs. This matters more than the success path: the #2546 logs
   that turned out to be most valuable were the two FAILURE records.
4. **Keep the upload-verifier's local-only-text check** as the backstop, not the mechanism. It
   worked here — twice — and should stay; it simply should not be the only thing standing between a
   run's forensic record and a terminated pod.

## Explicitly NOT duplicates

- **#2611** — `git_provenance()`'s 5 s-timeout `git status` orphans `.git/index.lock`.
- **#2612** — arm-independent `g0_gate.json` published to arm-unscoped paths.
- **#2610** — poller cannot detect terminal success of a single-phase dispatcher invocation.
- **#2605** — worker logs sit outside the POLLER's log-freshness globs. Related in subject (the same
  worker logs) but a different defect: #2605 is about the poller not READING them for staleness;
  this is about nothing UPLOADING them for durability.

A sibling gap in the same family, worth mentioning but filed separately if it survives review: in
`scripts/issue2546_fit_cells.py` the per-stem `_complete.json` records are written AFTER the
per-stem Hub upload, so they never ride along, and a code comment claims they are "mirrored
alongside the shards". Arm 1 and arm 3 each needed a manual 30-file upload to close it. That one is
per-issue experiment code rather than workflow surface, hence not the subject here.

## Target files

- `.claude/rules/upload-policy.md` (artifact table + the persist-by-default rule)
- a shared helper under `src/explore_persona_space/orchestrate/` (hub upload path)
- `.claude/agents/upload-verifier.md` if its local-only-text check needs to key off the new row
- optionally the dispatcher-authoring guidance so new dispatchers inherit the call

## Provenance

Surfaced driving #2546 arm 3 through `/issue`. Grounded by: an upload-verifier's call-site read of
`issue2546_dispatch.sh` finding no log-upload site; its FAIL enumerating 142 files / 5,063,198 B of
local-only text with per-group byte counts; `grep -c` over 78 `scripts/*_dispatch.sh` giving the
44/34 split (crude, caveated above); a null result searching `src/` + `scripts/` for any shared
log-persist helper; and a read of `.claude/rules/upload-policy.md` confirming its "log" mentions are
all about logging behavior or 0-byte-log wedge detection, with no artifact-table row for logs.
