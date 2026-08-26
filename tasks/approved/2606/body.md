---
title: pod.py provision exits 0 with BOOTSTRAP-OK while its own preflight reports
  PREFLIGHT-FAILED rc=2 — fail loud instead
kind: infra
tags: []
created_at: '2026-08-26T16:52:20Z'
has_clean_result: false
origin_prompt: 'Found during /issue 2569 Step 6b: pod-2569-rows provisioned rc=0 +
  BOOTSTRAP-OK while the bootstrap log carried two errno-116 uv install failures and
  PREFLIGHT-FAILED-AT-BOOTSTRAP rc=2; the broken venv then hung uv run with zero stderr.'
workflow: v1
---
## Goal

`pod.py provision` reports SUCCESS — exit 0 plus a `BOOTSTRAP-OK pod=<name>` banner
and "Done. SSH with: ssh <pod>" — while the bootstrap it just ran reported
`PREFLIGHT-FAILED-AT-BOOTSTRAP rc=2`. Every session that trusts the provision exit
code therefore receives a silently broken pod and launches a workload onto it.
Make the provisioning path fail loud when its own preflight fails.

## Evidence (observed live, #2569, 2026-08-26)

`uv run python scripts/pod.py provision --issue 2569 --name-suffix rows --intent eval`

- recorded `PROVISION_ROWS_RC=0`
- log tail: `BOOTSTRAP-OK pod=pod-2569-rows` / `Done. SSH with: ssh pod-2569-rows`
- live API: `pod-2569-rows` RUNNING, 1×H100, pod_id `wyo28bg2e884mn`

And in the SAME log:

```
error: Failed to install: scikit_learn-1.8.0-...whl (scikit-learn==1.8.0)
  Caused by: Stale file handle (os error 116) at path ".../.venv/lib/python3.11/site-packages/sklearn/svm/src/libsvm/.tmpjGylR7"
error: Failed to install: pytz-2026.1.post1-...whl (pytz==2026.1.post1)
  Caused by: Stale file handle (os error 116) at path ".../.venv/lib/python3.11/site-packages/pytz/zoneinfo/Asia/.tmp6fSbwF"
PREFLIGHT-FAILED-AT-BOOTSTRAP rc=2
```

So the two errno-116 (ESTALE) install failures left an incomplete venv, the pod's
own preflight caught it and reported rc=2, and the provisioning path still exited 0
with a success banner. The failure was found only by the operator habitually
grepping the provision log for error patterns
(`grep -ciE 'error|traceback|failed'` → 6 hits) — the documented monitoring habit,
not the tool's own signal.

Downstream cost when the green is trusted: `uv run` on that pod hangs at startup
with ZERO stderr, which is indistinguishable at first sight from three other
documented traps (the MooseFS FUSE read-wedge, the uv PATH-shim futex deadlock,
and errno-116 partial install). Discriminating them cost ~20 min of probes on a
billing 1×H100.

## Why this is a workflow-surface bug, not an experiment bug

The defect is in a workflow-helper script (`scripts/pod.py` provisioning path /
`scripts/bootstrap_pod.sh`), not in experiment code — squarely in the
workflow-fix-on-bug protocol's scope. It is also fleet-wide: the mandatory
pre-launch protocol says "Run preflight — fix every failure, never skip", and this
path structurally prevents a caller from honouring that rule by reporting success
over a failure.

## Acceptance criteria

1. A bootstrap whose preflight exits non-zero makes `pod.py provision` exit
   NON-ZERO, and the terminal banner says FAILED, not `BOOTSTRAP-OK`. The pod is
   left alive (it may be recoverable — see the remedy below) but the caller is
   told, loudly, in the exit code.
2. The provision summary surfaces the preflight verdict explicitly (a
   `PREFLIGHT: PASS|FAIL rc=<n>` line), so a caller need not grep the log to learn
   whether the environment is sound.
3. `BOOTSTRAP-OK` is emitted only when bootstrap AND its preflight both succeeded.
   A regression test pins the pairing (a fixture where preflight returns 2 must not
   produce a success banner or a zero exit).
4. No change to the pod's lifecycle on failure — do NOT auto-terminate. The pod may
   be repairable, and pods are not killed without the owner's decision.

## Verified remedy for the underlying errno-116 class (already established, cite it)

The venv failure itself has a verified fix, applied on pod-2569-rows this round —
`.claude/rules/gotchas.md` Trap 3 (overlay venv):

```
UV_PROJECT_ENVIRONMENT=/root/eps-venv uv sync --locked --python /usr/bin/python3.11
rm -rf .venv && ln -s /root/eps-venv .venv
```

Then gate by ACTUAL IMPORT of the launch-path packages (a clean `uv` audit does not
clear a MooseFS venv). Two operational facts worth folding into the docs while here:

- After the overlay symlink, a bare `uv run` re-resolves and HANGS; `UV_NO_SYNC=1`
  (or the direct `/root/eps-venv/bin/python`) is required. Same lever gotchas #1689
  already prescribes for fan-outs; the overlay symlink is a second trigger for it.
- `rm -rf .venv` of a full venv on MooseFS takes ~12 min in
  `request_wait_answer`. That is SLOW, not wedged — discriminate with a spot
  `ls -ld` on `/workspace` (returned 0.050 s throughout), per the SLOW-CANARY note.

A SECOND, larger question this round raises but does not settle, recorded for the
implementer to scope rather than assume: whether MooseFS-backed `eval` pods should
build the overlay venv BY DEFAULT at bootstrap, given the errno-116 class has now
recurred across #475, #2225, #2278, #2378 and #2569. That is a design change with
real surface (the `/root` overlay is wiped by `pod.py stop`→`resume`, leaving
`.venv` dangling) and should not be bundled into the fail-loud fix above without
its own review.

## Provenance

Found while #2569 provisioned its P-B critical-path pod at Step 6b. Filed per the
workflow-fix-on-bug protocol; the fail-loud fix is the blocking deliverable, the
default-overlay question is scoping only.
