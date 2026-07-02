---
name: GCP fetch_results sudo-tar-base64 transport fix
description: The gcp.py root-owned-artifact fetch fix — sudo -n tar | base64 piped extraction, and why base64 is mandatory not optional
type: project
---

The GCP lane runs the GCE startup-script workload as ROOT (#588), so the whole
`/workspace/eps-issue-<N>` tree is root-owned and the OS-Login `scp` user gets
`Permission denied` on `gcloud compute scp --recurse`. The MANDATORY sentinel
pull already reads it via `gcloud compute ssh ... sudo -n cat` (gcp.py:4669-4694);
the best-effort dir mirror (eval_results/, figures/) was NOT converted and still
scp'd (gcp.py:4696-4724), silently emptying the local mirror on every real run.

**Why:** Fix = same `sudo -n` transport, `cat` → `tar -c`. The base64 wrap
(`sudo -n tar -c ... | base64 -w0` then `base64.b64decode`) is MANDATORY, not a
robustness detail: `GcpBackend._run` / `GcloudRunResult.stdout` is UTF-8-decoded
`str` with `errors="replace"`, so raw binary tar bytes are IRRECOVERABLE through
the pipe (the fact-checker confirmed this). The plan's §12 hedged it as
"robustness detail" but the mechanism forces it. base64 also keeps the `_Runner`
mock string-shaped (no bytes-capturing `_run` variant needed).

**How to apply:** When reviewing a fix that pulls binary artifacts back through
a text-decoded transport, the encoding wrap is load-bearing, not optional —
verify the transport's return type before accepting a "confirm text-vs-bytes at
implementation" hedge. The `_Runner` test mock routes by transport substring
(`ssh` → ssh_results, `scp` → scp_results, tests/test_gcp_backend.py:249-252),
so moving the best-effort pulls scp→ssh flips `scp_calls==2` to `ssh_calls==3`
(1 cat + 2 tar) — the two existing sentinel tests (4933/4958) MUST update in the
same diff. (#790)
