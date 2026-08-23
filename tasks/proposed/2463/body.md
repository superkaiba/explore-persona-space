---
title: 'pod.py terminate cleanup is name-scoped not incarnation-scoped: a same-name
  replacement loses its local records'
kind: infra
tags: []
created_at: '2026-08-22T03:18:06Z'
has_clean_result: false
parent_id: 2270
workflow: v1
---
# `pod.py terminate` cleanup is name-scoped, not pod-incarnation-scoped — a same-name replacement loses its local records

## Goal

Make `cmd_terminate`'s post-survivor cleanup preserve local records (the
`pods_ephemeral.json` sidecar row and the `pods.conf` entry) whose CURRENT
`pod_id` differs from the id that was actually terminated, so a same-name
replacement pod registered during the terminate window does not silently lose
its local records and become invisible to every local tool.

## The gap

`cmd_terminate` terminates by live pod_id, then cleans up local state keyed by
NAME: the metadata loop and `_remove_from_pods_conf(name)` remove every name in
`terminated_names` unconditionally, without re-checking whether the row still
resolves to the pod that was destroyed.

So if a fresh pod claims the same managed name between the survivor re-query and
the cleanup — an external dispatcher, a crash-recovery relaunch, a concurrent
session's provision — the cleanup deletes the REPLACEMENT's rows. The pod stays
alive on the RunPod account while every local consumer (`pod.py list-ephemeral`,
SSH config, the audit-stale janitor's local-side reads) believes it does not
exist. That is the same failure SHAPE as the #365 / #475 stale-pod-id incidents,
approached from the other direction: there a stale local row pointed at a ghost,
here a live pod has no local row at all. The money consequence is identical — an
unreaped pod accruing charges, discoverable only through the live API.

`state.get(target)` (the #2270 exact-name stale lookup) does NOT help: it
narrows which STALE row is considered, while `stale.name in terminated_names`
suppresses only the `stale_name` deletion, not the unconditional
terminated-name removal that follows.

## Provenance

Surfaced by the Codex alternatives twin during #2270's Phase 2 plan review
(2026-08-22) as its Must-Fix 2, with the C6 "concurrent writers" state shape as
the walk. #2270 declined to fix it in-round and disclosed it instead: the race is
PRE-EXISTING on the bare terminate path, #2270 neither introduces nor widens it,
and turning a surgical-selector change into a lifecycle-locking refactor would
have been scope creep. #2270 narrowed its own record-survival acceptance claim to
what it actually guarantees (records of DIFFERENTLY-NAMED siblings survive) and
filed this task for the residual.

## Suggested shape (not a mandate — the spawned session designs it)

The smallest correct fix is probably to make the cleanup incarnation-aware:
carry the terminated `(name, pod_id)` PAIRS rather than a set of names, and skip
removal when the current row's `pod_id` no longer matches the terminated id.
Alternatives worth weighing: hold the lifecycle lock across
terminate-plus-cleanup so no registration can interleave; or re-query survivors
immediately before cleanup and diff. Prefer the pair-scoped variant if it holds,
since it needs no new locking and fails safe (an unmatched row is KEPT).

Constraints any fix should hold: do not weaken the #475 behavior of terminating
every live pod whose name resolves to the issue; do not weaken the
upload-verification, keep-running, or owner-fence guards; keep the bare-form
teardown semantics byte-compatible for existing callers; and pin the fix with a
regression that stages a same-name row with a different `pod_id` between the
survivor query and cleanup, asserting both the sidecar row and the pods.conf
entry survive.

Worth checking in the same round: whether `pod.py audit-stale` and the watcher's
pod-safety pass can already detect a live pod that has no local row, since that
is the observable symptom operators would hit.
