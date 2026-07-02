---
description: Pod SSH/MCP config authority split — live RunPod API vs pods.conf vs pods_ephemeral.json, the three sync directions, and when to reach for pod.py config --refresh-from-api (loads when you touch the pod scripts or pods.conf)
paths:
  - "scripts/pod*.py"
  - "scripts/pods.conf"
  - "scripts/pods_ephemeral.json"
  - "scripts/runpod_api.py"
  - "scripts/sync_pods.sh"
  - "scripts/_pods_conf_path.sh"
---

# Pod config authority split

Live RunPod API is authoritative for state (existence, status, host, port,
GPU, `created_at`). `scripts/pods_ephemeral.json` holds project metadata
only; `scripts/pods.conf` is the SSH/MCP config source, auto-synced.

## The three sync directions

- **Live API → `pods.conf` (automatic):** `pod.py provision` /
  `pod.py resume` refresh `pods.conf` from the live API on success.
- **`pods.conf` → outward:** `pod.py config --sync` propagates `pods.conf`
  OUTWARD to `~/.ssh/config` + `.claude/mcp.json`.
- **Live API → `pods.conf` (manual):** the inverse direction — pulling live
  API host/port INTO `pods.conf` outside an explicit provision/resume call —
  is `pod.py config --refresh-from-api [<name>]`.

## When to reach for `--refresh-from-api`

Use it when a SUPPLY_CONSTRAINT-blocked resume eventually succeeds via a
retry path that bypassed `_upsert_pods_conf`, or whenever an SSH polling
loop is failing on a port the live API no longer reports.

(Incident #488, 2026-06-09: a resume blocked on SUPPLY_CONSTRAINT brought
the pod back at a new port outside the success path; `pods.conf` stayed at
the pre-stop port and an autonomous SSH polling loop spun for 13+ hours at
$32/hr.)

## Live state vs seed (task #821 v3 relocation)

The LIVE (mutable) `pods.conf` no longer lives inside the git working tree.
It has been relocated to **`<git-common-dir>/eps/pods.conf`** — i.e.
`<main>/.git/eps/pods.conf` — outside the working tree, so `git reset
--hard`, `git checkout -- .`, `git restore -- .`, `git clean -fd`, and
`git clean -fdx` cannot touch it (they operate on the working tree; nothing
under `.git/` is affected). The tracked `scripts/pods.conf` file is now a
**SEED** — used only for fresh-clone bootstrap.

Resolution is LAZY (call-time), consistent across every consumer:

- **Python:** `pod_config._resolve_live_pods_conf()` — returns the LIVE
  path if it exists, otherwise migrates `scripts/pods.conf` → the LIVE
  path atomically (write to `<live>.tmp`, then `os.replace`) inside
  `locked_pods_conf` so two concurrent migrators cannot race. Every
  `pod_config` reader (`parse_pods_conf`) and writer (`write_pods_conf`)
  now defaults `path=None` and resolves at call time so a test's
  `monkeypatch.setattr(pod_config, "PODS_CONF", tmp)` is honored on
  every call (previously the default arg captured `PODS_CONF` at
  function-def time — the fixed footgun).
- **Shell:** `scripts/_pods_conf_path.sh` sets `$CONF` to
  `$GIT_COMMON_DIR/eps/pods.conf` if that file exists, else falls back
  to `$MAIN_REPO_ROOT/scripts/pods.conf` (the seed).

The migration happens ONCE per checkout on the first Python invocation
that needs to read/write; every subsequent invocation sees the LIVE path
and uses it directly.

**Read-only-filesystem fallback:** `_resolve_live_pods_conf` cannot
`mkdir -p` the LIVE dir under a read-only mount. It emits a loud stderr
WARN and returns the seed path so read-only tooling (a viewer script)
keeps working; any subsequent WRITER on the seed will surface a loud
failure on the next destructive git op — the WARN is the operator signal
that this state needs fixing.

## Never-drop-RUNNING guard + atomic write (task #821)

Layered on top of the relocation:

- `write_pods_conf(pods, *, allow_remove=frozenset())` runs a
  never-drop-RUNNING guard BEFORE the write. It diffs on-disk row names
  against the incoming `pods` list; any row being dropped that the live
  RunPod API still reports RUNNING is RE-ADDED with a loud WARN naming
  the pod and the `allow_remove` opt-out.
- The single legitimate remove path
  (`pod_lifecycle._remove_from_pods_conf`) passes `allow_remove={name}`
  and BYPASSES the API check — terminate flows never depend on network
  access.
- UPDATE paths (host/port change; name in both `on_disk` and `new` sets)
  trigger no API call at all — `dropped` is empty by construction.
- If `runpod_api.list_team_pods` raises (network flake), the guard fails
  SAFE by re-adding every dropped row and emitting a WARN. An
  unreachable API cannot disprove RUNNING.
- The write itself is atomic via `os.replace` on same-FS tmp: readers
  never see a torn intermediate file. On a crash, the leftover
  `<path>.tmp` is harmless — the next writer overwrites it.

## `--refresh-from-api` self-heal (task #821 extension)

`_refresh_one_pod` gained a **re-add branch**: when the on-disk row is
absent BUT the live API reports the pod RUNNING with a valid SSH
endpoint, the function appends a fresh `Pod(...)` (constructed from
`PodInfo` — `gpu_type_id or "unknown"` for the required string field,
`name or f"restored:{name}"` for the label) to the caller-supplied
`to_add` list. The caller merges `to_add` into the rows it writes.

Bulk-mode `cmd_refresh_from_api` (`pod_name is None`) additionally
enumerates the live API and adds every managed `^pod-\d+$` pod absent
from `pods.conf` to its target list before iterating — reaching the
re-add branch even for rows the wipe took out entirely.

This closes the `poll_pipeline.py` auto-heal loop: after 10 consecutive
SSH failures the poller runs `pod.py config --refresh-from-api`, and now
that command RESTORES a wiped row from the live API instead of no-oping.
Pre-task-#821 the missing row was invisible to the refresh path.

## Lockfile deletion is harmless

The flock lockfile at `scripts/.pods.conf.lock` is worktree-untracked
(now `.gitignore`d as belt-and-braces). Its deletion by `git clean -fdx`
is HARMLESS: `locked_pods_conf` recreates it with `O_CREAT` on every
acquisition, so the next writer just re-establishes the mutex.

## Autostash / stale-worktree interactions

The relocation subsumes the task #500 fix (worktree-local pods.conf
divergence): the LIVE path resolves to `<git-common-dir>/eps/pods.conf`,
which is identical across every worktree of the same repo (they share
one `.git`). A stale worktree running pre-#821 code still resolves its
own copy of `scripts/pods.conf` (the seed) via the old
`_main_repo_scripts_dir` path — that's the seed file, which is now
"drifty by design" (visible in `git diff`, never a live-file wipe).
Post-#821 workers on both sides see the same LIVE state.
