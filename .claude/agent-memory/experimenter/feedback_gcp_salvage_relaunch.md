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
0. (REQUIRED FIRST, #908) If the VM's `eps/phase` guest attribute reads a
   terminal/wedged value (`done`/`failed`/`wedged`), re-publish
   `eps/phase=workload` via the guest-attribute curl (`curl -fsS -X PUT -H "Metadata-Flavor: Google" --data "workload" "http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/eps/phase"`)
   BEFORE resuming any work — the #908 zombie predicates (`reconnect_or_none` + the pre-launch
   stale reclaim) classify a RUNNING VM with a terminal phase as a
   finished zombie and DELETE it on the next dispatch; an active relaunch
   must never be left reading terminal.
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

**Pre-launch liveness check phantom (companion, #653 r10).** Running
`pgrep -f "i653_dispatch.py"` (or any `pgrep -f <script>`) inside a
gcloud-ssh `--command='...'` SELF-MATCHES the SSH wrapper's argv and
returns phantom "live dispatcher" pids that are really the SSH
session itself. Naive consequence: brief says "expect NONE" → you
see pids → you either skip launch (orphaning the recovery) or kill
the SSH session itself. Always combine two checks: (a) `pgrep -af`
to see the full command line and recognize the self-match, AND (b)
`ps -eo pid,cmd | grep '[u]v run python <script>'` (BRACKET-FIRST
to dodge self-match) which only returns real dispatchers; also
cross-check the pidfile's stale pid with `ps -p $PID`.

**`/workspace/eps-issue-<N>` is root-owned (companion, #653 r10).**
The SSH user cannot `cd` into it and cannot `source ./.env` from a
top-level shell — `cd: Permission denied`. Wrap the ENTIRE fetch /
reset / source / launch sequence inside ONE `sudo bash -c '...'`
that does the `cd` + `source ./.env` + `git fetch` + `git reset`
+ launch under root.

Origin: #653 round 9 + round 10 salvage-relaunch (2026-06-17).
