---
name: handrolled-pod-sentinel-envelope
description: Flag any pod-side dict written to /workspace/logs/issue-<N>-*.json that lacks poll_pipeline's _SENTINEL_REQUIRED_KEYS (sentinel_schema_version/kind/version) — the drain silently skips it
metadata:
  type: feedback
---

A driver that hand-writes an abort/failure sentinel (e.g. `{"epm_marker":
"epm:failure", ...}`) to `/workspace/logs/issue-<N>-*.json` produces a file the
VM poller SKIPS with a quiet warning: `poll_pipeline._parse_sentinel` requires
`sentinel_schema_version` + `kind` + `version` (poll_pipeline.py:1331), and the
#899 synthesis fallback rescues ONLY a `-results.json` filename carrying all 10
Step-7 payload keys. The plan's "post epm:failure via the poller" contract then
never fires; the failure surfaces only as a generic shard death.

**Why:** #1491 round 3a — the Gate-1 abort sentinel carried
`epm_marker/failure_class/reason` and none of the envelope keys; verified
unrescuable by reading `_maybe_synthesize_results_envelope`. The conforming
helper existed one import away (`issue779_common.write_sentinel(kind, note,
task_id=..., extra=...)`).

**How to apply:** in any diff writing a JSON file under `/workspace/logs/`
intended for the poller, diff its keys against `_SENTINEL_REQUIRED_KEYS` and
check `task_id` is threaded (the issue779 helper defaults task_id=779). A
hand-rolled dict beside an existing conforming `write_sentinel` helper is a
MAJOR. Sibling class: [[gate-threshold-vs-shard-config]].
