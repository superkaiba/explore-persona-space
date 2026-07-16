---
name: Pod git HTTPS 403 with VALID token — bundle sideload recovery
description: RunPod pod-side `git fetch` to github.com can 403 even with a verified-valid token and a correct env-reading credential helper; deterministic recovery is a git bundle over scp, not more auth debugging
type: feedback
---

On a RunPod pod, `git -C /workspace/explore-persona-space fetch origin <branch>`
can fail `403` on the HTTPS GitHub remote even when (a) the pod `.env`
GITHUB_TOKEN is byte-identical to the VM's and verified VALID (`curl` API probe
returns 200 on the private repo), (b) the #1239 env-reading repo-local
credential helper is correctly configured and is the ONLY helper, and (c) the
remote URL is clean (no embedded token). Root cause unconfirmed — most likely
GitHub blocking/limiting the pod's egress IP for git-http specifically.

**Why:** Incident #1315 r8 relaunch (2026-07-15, pod-1315): two rounds of
credential-helper recovery left the 403 unchanged while 4×H100 idled; the
bundle path landed the fix in one minute.

**How to apply:** After ONE credential-helper recovery attempt fails on a
still-403 fetch with a token the VM verifies valid, stop debugging auth and
sideload: on the VM `git bundle create /tmp/i<N>.bundle <podHEAD>..refs/remotes/origin/issue-<N>`
(pod HEAD must be an ancestor — check `merge-base --is-ancestor` on the VM
first), `scp -P <port>` it to `/workspace/`, then on the pod
`git -C /workspace/explore-persona-space pull --ff-only /workspace/i<N>.bundle refs/remotes/origin/issue-<N>`
+ `git -C ... update-ref refs/remotes/origin/issue-<N> <tip>`. Verify HEAD +
fix-ancestor after, as usual. Note the repo-root branch guard blocks bare
`git checkout` text even inside pod-bound heredocs — always use the
`git -C /workspace/...` form in remote scripts.
