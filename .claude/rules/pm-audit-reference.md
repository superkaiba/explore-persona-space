---
description: PM Mode-2 AUDIT reference — unmapped/non-EPS-pod ownership triage, the non-EPS pod-cost standing directive (verbatim, Thomas 2026-07-22), and the audit report template (loads when the PM spec is open; pointer-read from research-pm.md Mode 2)
paths:
  - ".claude/agents/research-pm.md"
---

# PM audit reference — unmapped/non-EPS pods + audit output format

Relocated from `.claude/agents/research-pm.md` Mode 2 — AUDIT (#1618, the
#829 relocate-to-rules pattern; overage source: commit 5d84120ac9). The
load-bearing summary stays in the PM spec's Mode-2 pointer; this file is
the full recipe.

## Unmapped RUNNING pods — ownership triage

**Unmapped RUNNING pods — ownership triage FIRST, not emergency.**
The RunPod account is team-shared: non-EPS teammates may run pods with the managed `pod-` prefix; "no pods_ephemeral.json mapping" does NOT mean "leak".
1. Ownership: Thomas's RunPod dashboard (`PodInfo` has NO creator field — `runpod_api.py:409-451`); cross-check `pods_ephemeral.json` + `_scan_task_references`.
2. `pod-<N>` with small N is almost certainly EPS; a hex suffix / exotic GPU count says ask first.
3. Near-zero GPU util for hours is the clearer leak signal.
4. Surface as a QUESTION ("N unmapped pods at $X/hr — team work?"), never a "spend emergency". Never recommend terminating one without Thomas's explicit confirmation of non-EPS ownership.

## The non-EPS pod-cost standing directive (verbatim)

**Non-EPS team pods' COST is NOT ours — never report it as burn (standing directive, Thomas 2026-07-22: "we should ignore these pods because they aren't charging us").** The RunPod team `cm8ipuyys...` ("Anthropic Safety Research") is the ~50-person fellows org (confirmed 2026-07-22 — the same team hosts the fellows Slurm clusters); the ~100+ non-EPS pods in team listings belong to other fellows on their own budgets. Fleet-burn numbers, spend alerts, and audit bullets count ONLY EPS-managed pods (`pod-<N>` / `eps-issue-*` / `pods_ephemeral.json`-mapped). Other fellows' pods are also not an EPS problem when they exhaust RunPod capacity (the #1586 GCP-stockout incident) — that reads as ordinary `no_compute_available`, never as something to escalate about the pods themselves. Compute preference under the same directive: the fellows Slurm cluster (charmander lane, #1609) is used as much as possible when it starts quickly.

## Mode-2 audit output format

Output format:

```markdown
# Audit — YYYY-MM-DD

## Auto-fixed (already applied)
- [x] INDEX.md: added entry for eval_results/<dir>/

## Needs approval (proposed diffs)
### RESULTS.md
```diff
- [old claim]
+ [corrected claim per #<N>]
```
**Reason:** ...
```

## Files of record

Tasks #1618 (this relocation), #1611 (the size-red emitter), #1609 (the
fellows Slurm lane); commit `5d84120ac9` (the 2026-07-22 directive landing
that pushed the PM spec over its 47,000-byte cap).
