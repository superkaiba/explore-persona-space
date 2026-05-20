---
title: 'Ephemeral-pod provisioning is broken: cloudType enum encoding, epm-* pod-name
  pattern, silent failure in git clone'
kind: infra
tags: []
created_at: '2026-05-01T19:47:50.000Z'
has_clean_result: false
sagan_id: a4e8d115-6039-4795-bb79-e27d87591b2c
sagan_number: 176
priority: high
legacy_why_unset: true
---
## Symptoms

`uv run python scripts/pod.py provision --issue 156 --intent eval --ttl-days 1` fails. The new ephemeral-pod system has a chain of three bugs that block all dispatches through it. Hit while running `/issue 156`; reproducible from a clean state.

## Bugs

### Bug 1: `cloudType` GraphQL enum encoded as quoted string

**File:** `scripts/runpod_api.py:230-241`

The string-builder for the GraphQL `input` block of `podFindAndDeployOnDemand` quotes everything that isn't `bool`/`int`. RunPod's schema declares `cloudType` as `CloudTypeEnum` (values: `ALL`, `SECURE`, `COMMUNITY`) and rejects quoted values with:

```
Enum "CloudTypeEnum" cannot represent non-enum value: "ALL". Did you mean the enum value "ALL"?
```

**Fix (applied locally, uncommitted on manager VM):** add an `enum_fields = {"cloudType"}` set; emit those fields bare like ints. Patch:

```diff
-    fields = []
-    for k, v in inputs.items():
+    enum_fields = {"cloudType"}  # GraphQL CloudTypeEnum: ALL | SECURE | COMMUNITY
+    fields = []
+    for k, v in inputs.items():
         if isinstance(v, bool):
             fields.append(f"{k}: {'true' if v else 'false'}")
         elif isinstance(v, int):
             fields.append(f"{k}: {v}")
+        elif k in enum_fields:
+            fields.append(f"{k}: {v}")
         else:
             fields.append(f'{k}: "{v}"')
```

### Bug 2: `bootstrap_pod.sh` pod-name pattern misses `epm-*`

**File:** `scripts/bootstrap_pod.sh:75`

The arg parser only matches names starting with `pod`:

```bash
elif [[ "$arg" == pod* ]]; then
    POD_NAME="$arg"
fi
```

But ephemeral pods are named `epm-issue-<N>` per the new convention (`scripts/pod_lifecycle.py:198`). When `pod_lifecycle._bootstrap()` calls `bash scripts/bootstrap_pod.sh epm-issue-156`, the arg is silently ignored, `POD_NAME` stays empty, `HOST`/`PORT` never get resolved from `pods.conf`, and the script bails:

```
Error: Must specify pod name or --host and --port
```

**Fix (applied locally, uncommitted):** broaden the match.

```diff
-elif [[ "$arg" == pod* ]]; then
+elif [[ "$arg" == pod* || "$arg" == epm-* ]]; then
     POD_NAME="$arg"
 fi
```

### Bug 3: Step 3 (`Setting up git repository`) silently fails on fresh pods (NOT FIXED)

**File:** `scripts/bootstrap_pod.sh:138-159`

The script clones via SSH (`git@github.com:superkaiba/explore-persona-space.git`), but a fresh RunPod has no GitHub deploy key. Result:

```
[3/9] Setting up git repository
Cloning repo...
Cloning into 'explore-persona-space'...
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.
bash: line 15: cd: explore-persona-space: No such file or directory
Cloned at:
fatal: not a git repository (or any parent up to mount point /)
  ✓ Repository ready    <-- silent-failure violation per CLAUDE.md
```

The script then proceeds to step 4 (`Distributing API keys (.env)`) on a non-existent repo and hangs / silently breaks downstream steps.

Two issues here:
1. **No SSH-key bootstrap path** for fresh ephemeral pods. Existing pods (pod1-5) work because they were manually configured; new ephemeral pods have nothing.
2. **Silent failure**: `log_ok "Repository ready"` runs unconditionally, regardless of clone exit code. Per CLAUDE.md "Never silently fail."

**Fix needs a design call:**
- (a) Push a per-pod deploy key as a step `2.5` before `git clone` (script generates ed25519 keypair on the pod, registers public half via GitHub API as a deploy key on the repo, uses private half for clone).
- (b) Switch the clone to HTTPS using `GH_TOKEN` from the manager VM's `.env` (e.g., `https://x-access-token:$GH_TOKEN@github.com/...`). Simpler, but the token leaks into the pod.
- (c) Always rsync the repo from the manager VM instead of cloning fresh. Avoids credentials entirely but couples the pod to whatever the manager has on disk.

Whichever path, the silent-failure bug must also be fixed (step 3 should `set -e`-die or explicitly check the exit code before the green checkmark).

## Reproduce

From a clean main:
```bash
git stash  # if you have local fixes for bugs 1+2
uv run python scripts/pod.py provision --issue <some-test-N> --intent eval --ttl-days 1
# Bug 1 trips first: HTTP 400 cloudType enum error
```

After applying fix 1:
```bash
# Bug 2 trips: bootstrap reports "Must specify pod name or --host and --port"
```

After applying fix 2:
```bash
# Bug 3 trips: git clone fails, "✓ Repository ready" lies, step 4 hangs
```

## Why this matters

Blocks **all** experiments dispatching through the new ephemeral-pod path. Hit by `/issue 156` (#156 is at `status:approved` waiting for this).

## Acceptance criteria

1. `uv run python scripts/pod.py provision --issue <N> --intent eval --ttl-days 1` succeeds end-to-end on a fresh provision: pod created, SSH ready, repo cloned, `.env` pushed, preflight green.
2. `bash scripts/bootstrap_pod.sh epm-issue-<N>` succeeds standalone after provision.
3. Step 3 of bootstrap propagates non-zero exit codes (no silent `✓ Repository ready` after a failed clone).
4. Test the flow on a real fresh provision before closing.

## Compute

`compute:none` — pure infra; no GPUs needed beyond a brief test provision.

## Related

- Hit while running `/issue 156` (#156). See its [`epm:dispatch-blocked v1`](https://github.com/superkaiba/explore-persona-space/issues/156#issuecomment-4361281135) comment.
- Recent CLAUDE.md edit introduces the `epm-issue-<N>` ephemeral-pod naming convention; this issue tracks bringing the implementation up to that spec.
