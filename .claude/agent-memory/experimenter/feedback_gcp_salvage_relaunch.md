---
name: GCP-lane salvage-relaunch: .env + git-auth + pkill self-match
description: When landing a code-fix on an EXISTING GCP instance, stage .env + fetch helper-authenticated per the #1239 credential-helper contract; never pkill -f a string that's in your own SSH argv.
type: feedback
---

A fresh-bootstrapped GCP (`eps-issue-<N>`) instance has NO `.env` at the
repo root, and a later SSH session does not inherit the startup-script
environment. Post-#1205 the startup script DOES configure an env-reading
git credential helper repo-local on the workload clone
(`src/explore_persona_space/backends/gcp.py`, the "Git push credential"
block) — but that helper reads `GITHUB_TOKEN` from the INVOKING
environment with no `.env` fallback (the `.env` fallback is the
pod-flavor delta in `scripts/bootstrap_pod.sh`), so in a bare salvage
shell it emits an empty password.

**Why:** salvage-relaunch needs to LAND code on the instance via
`git fetch + reset --hard origin/<branch>`, and credentialed pipeline
phases (analyze / upload — WandB/HF) need the `.env` staged. With no
token in the invoking env the #1205 helper degrades to an empty
password, and without `GIT_TERMINAL_PROMPT=0` a non-TTY fetch can wedge
on a credential prompt. (The repo is currently public, so an
unauthenticated FETCH works today — the helper is what makes `git push`
and any future-private fetch work. The old `Authorization: Bearer`
caveat is moot under a credential helper: git sends Basic auth with the
helper-provided username/password, which classic `ghp_` PATs accept.)

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
2. Fetch helper-authenticated (#1239 credential-helper contract — the
   token never appears in the remote URL, argv, or git config; the
   pre-#1239 tokenized-remote-URL form is SUPERSEDED and now banned on
   these surfaces by tests/test_bootstrap_pod_git_credentials.py).
   Ship the script over stdin — the same channel as the step-1 `.env`
   staging — so there is no nested `--command` quoting; the quoted
   heredoc (`<<'EOS'`) keeps `${GITHUB_TOKEN}` unexpanded locally, and
   `EOS` must sit at column 0 when run:

   ```bash
   cat > /tmp/salvage_fetch_issue<N>.sh <<'EOS'
   set -euo pipefail
   cd /workspace/eps-issue-<N>
   set -a; . ./.env; set +a          # staged in step 1; fails loud if missing
   export GIT_TERMINAL_PROMPT=0      # never hang a non-TTY shell on a prompt
   git config --replace-all credential.https://github.com.helper \
     '!f() { echo username=x-access-token; echo "password=${GITHUB_TOKEN}"; }; f'
   git fetch origin <branch>
   echo SALVAGE-FETCH-OK        # engagement sentinel: absent from output => the stdin script never ran
   EOS
   gcloud compute ssh eps-issue-<N> --configuration=eps-gcp \
     --project=eps-persona-gpu-jun2026 --zone=<zone> \
     --tunnel-through-iap=false --command='sudo bash -s' \
     < /tmp/salvage_fetch_issue<N>.sh
   ```

   The sourced `.env` puts `GITHUB_TOKEN` in the command environment;
   the helper (byte-same shape as the one the #1205 startup script
   installs on the workload clone) reads it at fetch time. The
   `git config --replace-all` line is an idempotent refresh —
   load-bearing on a pre-#1205 instance (≤7d max-run-duration means
   some may survive until ~2026-07-16) or a second clone (which never
   gets the launch-gated helper), a no-op-safe re-assert otherwise. The
   host-scoped key never offers the token to a non-GitHub remote.

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

Origin: #653 round 9 + round 10 salvage-relaunch (2026-06-17); recipe updated to the #1239 helper contract by #1271 (2026-07-11).
