---
name: GCP-lane salvage-relaunch: .env + git-auth + pkill self-match
description: When landing a code-fix on an EXISTING GCP instance, stage .env + use token-in-URL for private git fetches; never pkill -f a string that's in your own SSH argv.
type: feedback
---

A fresh-bootstrapped GCP (`eps-issue-<N>`) instance has NO `.env` at the
repo root and NO git credential helper — its startup-script clone used
a transient token that's scrubbed once the script exits.

**Why:** salvage-relaunch flow needs to LAND code on the instance via
`git fetch + reset --hard origin/<branch>`. Without auth, the fetch
HANGS waiting on interactive credential prompt (in a non-TTY SSH
session this is forever). `Authorization: Bearer` does NOT work for
classic `ghp_` PATs over git smart-HTTP. Credentialed pipeline phases
(analyze / upload — WandB/HF) also need the `.env` staged.

**How to apply:** On the salvage-relaunch SSH session:
1. Stage `.env` from the local VM to `/workspace/eps-issue-<N>/.env`
   via stdin to a root-only file (mode 600).
2. Fetch with token-in-URL:
   `git fetch "https://x-access-token:$TOK@github.com/<owner>/<repo>.git" <branch>`
   where `$TOK` is read from the just-staged `.env`'s `GITHUB_TOKEN`.

Also: NEVER `pkill -f "<pattern that appears in your own SSH argv>"`
— the pattern self-matches the SSH command's own argv and SIGKILLs the
session (gcloud exits 255, leaving you locked out of the SSH stream).
Kill stray remote procs by exact PID only.

**Launcher PATH gotcha (companion).** A `setsid nohup <launcher>` run
under `sudo bash -c '...'` does NOT inherit root's login PATH — `uv`
(at `/root/.local/bin/uv`) is not found and the launcher dies in
seconds with `uv: command not found`. Put `export
PATH="/root/.local/bin:$PATH"` as the FIRST line of any GCP-lane
salvage launcher (the SSH-MCP/RunPod launcher template already does
this — port it forward). The workdir `/workspace/eps-issue-<N>` is
root-owned, so the entire relaunch block must run under `sudo bash -c`.

Origin: #653 round 9 salvage-relaunch (2026-06-17).
