---
name: hub._upload single-file path needs upload_as_file=True
description: FAIL any diff calling hub._upload(file_path, ...) without upload_as_file=True — main's _upload raises ValueError unconditionally on a file path
type: feedback
---

`explore_persona_space.orchestrate.hub._upload(local_path, repo_id, repo_type,
path_in_repo, ...)` defaults `upload_as_file=False`. On `main` (the code pods
run), `_upload` RAISES `ValueError` unconditionally when `local_path.is_file()
and not upload_as_file` — because `huggingface_hub.upload_folder` silently
no-ops on a single-file path (logs "is not a directory. Keeping local path."
and uploads NOTHING, yet verification can pass if same-prefix files already
exist → silent data loss, the guard's reason, #595).

**Rule:** any per-file `_upload` call MUST pass `upload_as_file=True`:

    hub._upload(f, repo_type="dataset", path_in_repo=..., upload_as_file=True)  # correct
    hub._upload(f, repo_type="dataset", path_in_repo=...)                       # WRONG — raises

For batching raw completions prefer the canonical helper
`upload_raw_completions_to_data_repo()` over a hand-rolled per-file loop.

**Why:** the call typically lives on a GPU/upload branch the CPU smoke skips
(`--skip-gpu-phase-a`, `--dry-run`), so it fires for the first time on the pod
AFTER the expensive phases are spent; on the GCP lane the crash is doubly
hidden behind an empty-log-tail `guestTerminate`.

**How to apply:** grep every diff that touches HF uploads —
`grep -nE "_upload\(" <files>` — and for each, check the first positional arg:
if it's a single file (a `*.json` / `*.pt` / `summary` path, not a dir) and
`upload_as_file=True` is absent, it's a Critical mechanizable finding.

Incidents: #640 round 1→2 (2026-06-15, Claude missed it, Codex caught it,
reconciler sided with Codex); #612 round 2 (2026-06-15, Phase-A
`phase_a_summary.json` upload in `issue612_predictor_v3_driver.py`). The
worktree's sparse `hub.py` may be BEHIND main and take the silent-no-op path
instead of raising — always check `git show main:.../hub.py` for the live
behavior, not the stale worktree copy.
