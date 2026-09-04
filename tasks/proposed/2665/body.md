---
title: 'pod.py config --sync: per-clone live pods.conf copies clobber the shared ~/.ssh/config
  (RUNNING pod alias dropped twice during #2658 P4)'
kind: infra
tags:
- workflow-fix
created_at: '2026-09-04T07:20:24Z'
has_clean_result: false
parent_id: 2658
origin_prompt: 'Filed by the #2658 driver session after pod-2658 lost its SSH alias
  twice (06:35Z, 07:18Z 2026-09-04) to config --sync runs from a second clone whose
  live pods.conf lacked the row.'
workflow: v1
---
# pod.py config --sync: per-clone live pods.conf copies clobber the shared ~/.ssh/config and ~/.claude/mcp.json

kind: infra

## Gap

`scripts/pod.py config --sync` regenerates the per-user `~/.ssh/config` and `~/.claude/mcp.json`
from the LIVE `pods.conf` at `<git-common-dir>/eps/pods.conf`. That live copy is per repository
clone, while the two config targets are per user. On this VM two clones coexist
(`/home/thomasjiralerspong/explore-persona-space/.git/eps/pods.conf`, 45 rows, and
`/mnt/eps-data/thomasjiralerspong/issue779_task_clone/.git/eps/pods.conf`, 23 rows). A pod
provisioned from one clone is registered only in that clone's copy, so any `config --sync`
run from the other clone rewrites the shared SSH config without it. `write_pods_conf`'s
never-drop-RUNNING guard protects one clone's copy, not the cross-clone write to the shared
targets.

## Evidence (task #2658, 2026-09-04)

- pod-2658 (RunPod id 00c34d2rmhne3r, 4x H200, provisioned 05:29Z from the issue779_task_clone
  worktree) lost its `Host pod-2658` block in `~/.ssh/config` at 06:35:45Z and again at 07:18:01Z,
  each time coinciding with a sync from a session in the main clone whose live pods.conf had no
  pod-2658 row.
- The poll monitor watching the P4/P5 production run lost SSH for three consecutive polls the
  first time (recovered via `config --sync` from the owning clone) and fell back to a direct
  ip:port SSH the second time.
- Mitigation applied for the run: `pod.py config --refresh-from-api pod-2658` run from the main
  clone, so both copies now carry the row. This does not fix the class.

## Proposed fix (implementer decides the exact shape)

One of, in order of preference:

1. Make the live pods.conf per USER, not per clone (e.g. `~/.eps/pods.conf`, or a single path
   resolved independent of `git-common-dir`), with a one-time migration that merges existing
   per-clone copies. Every clone then reads and writes one file.
2. Keep per-clone copies but have `config --sync` MERGE rows across every live copy it can
   discover (and/or refresh RUNNING pods of the team from the live RunPod API) before writing
   the shared targets, so a sync can never drop a RUNNING pod registered elsewhere.
3. At minimum, make `config --sync` refuse to remove a `Host` block for a pod the live RunPod
   API reports RUNNING, and print the offending clone path.

Add a regression test: two temp "clones" with disjoint pods.conf rows, sync from each, assert
the shared SSH config retains both pods.

## Acceptance

- A pod provisioned from clone A survives `config --sync` from clone B in `~/.ssh/config` and
  `~/.claude/mcp.json`.
- The regression test above passes.
- `pod.py config --check` reports the cross-clone state (or the single per-user path) legibly.
