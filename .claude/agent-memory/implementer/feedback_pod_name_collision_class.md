---
name: pod-name-collision-class
description: pod_lifecycle.py matches sidecar entries to live RunPod API by NAME, so when two pods share a name (post-migration), drift-repair silently picks the wrong one and clobbers manual pods.conf overrides
metadata:
  type: feedback
---

`scripts/pod_lifecycle.py::_load_state()` builds `live_by_name = {p.name: p for p in live_pods}` — last-write-wins on name collision. When a RunPod pod is migrated (host/port change) and the user runs `pod.py config --update pod-N --host X --port Y` to redirect the SSH alias, the sidecar still holds the OLD pod_id. The next `_load_state()` call sees `meta.pod_id != live.pod_id` and "drift-repairs" the sidecar back to whichever pod_id the live API returned for that name. Subsequent `_upsert_pods_conf()` then writes the wrong host/port into `pods.conf`, silently repointing the SSH alias.

**Why:** Task #391 had subagents run on the wrong pod after this happened — the SSH alias `pod-391` resolved to a stale pod silently.

**How to apply:** Fixed via `manual_override` flag in `EphemeralMetadata` (commit 117eb857). When considering any future change to pod tooling that auto-refreshes state from the live RunPod API, remember the by-name matching is fundamentally racy when names collide. Prefer pod_id as the join key when feasible; otherwise opt-in protection (`manual_override`) keeps the user in control. Do not silently overwrite user-set values — always WARN to stderr.
