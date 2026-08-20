---
title: Bracketed pgrep ownership probe self-matches when the same command references
  the unbracketed literal (blinds death detection)
kind: infra
tags:
- workflow-fix
- process-probe
created_at: '2026-08-20T05:15:18Z'
has_clean_result: false
parent_id: 2204
workflow: v1
---
# The bracketed-pattern ownership probe self-matches when the same command also references the unbracketed literal

## Goal

Extend the bracketed-`pgrep` guidance in `.claude/rules/gotchas.md` (the SSH-remote / ownership-probe entry, echoed by CLAUDE.md § "Ownership check before any resume/launch on a shared artifact path") to name the failure mode where the bracket trick is insufficient, and give the probe shape that survives it. Small documentation + idiom fix; no code change required unless the plan finds a shared helper worth adding.

## Why (two misreads in one session)

The documented technique is to bracket one character of the pattern — `pgrep -af 'issue-2204-lint-gat[e]'` — so the probe cannot match the pattern text sitting in its own `argv`. That reasoning is sound but incomplete: it only protects against the PATTERN string. It does nothing when the same command references the **unbracketed literal** for some other purpose, because the bracketed pattern matches that literal perfectly.

Both failures below are from the #2204 Step 10d merge (session 9e938266, 2026-08-19/20):

**1. Kill-probe false positive.** A single Bash call combined the probe with greps that named the script:

```
pgrep -af 'issue-2204-lint-gat[e]' || echo CLEAR
grep -nE 'verdict|...' /tmp/issue-2204-lint-gate.sh
```

`gat[e]` matches `gate` inside `/tmp/issue-2204-lint-gate.sh`, so the probe reported a live gate that did not exist. Recoverable — the operator re-ran the probe in isolation.

**2. Watcher blinded — the serious one.** A `Monitor` poll loop used the same probe for liveness *and* referenced `/tmp/issue-2204-lint-gate.log` to extract the verdict line. It therefore matched itself on every poll, producing:

- a false `gate LAUNCHED` event while the process was demonstrably still queued (`sleep 60` child, gate log 0 bytes, 8 further "over cap" lines after the event);
- **an unreachable death branch.** The `! pgrep ... && ! pgrep ...` "both gone" test could never be true, because one of the two always matched the monitor itself. A genuine death of the watched process would have read as health for the monitor's full hour — exactly the condition the watcher was armed to catch.

The second failure is what makes this worth a task rather than a one-off note. A self-matching probe does not merely lie; it lies in the fail-OPEN direction for death detection, which is the direction the whole "poll shares your kill domain" discipline exists to close. It is the same family as the `leg=none` false negative earlier in the same round and the #825 empty-dir false-DONE: the detector was broken, not the work.

## Acceptance

- The gotchas entry states the limit explicitly: **bracketing protects against the pattern text in `argv`, not against any other occurrence of the literal in the same command** (a log path, a script path, a message string) — and that combining a probe with anything that names the target in the same call re-arms the collision.
- It gives at least one probe shape that is structurally immune, with the reason. Candidates for the plan to weigh:
  - **PID-keyed liveness** — capture the pid at launch, then `kill -0 "$PID"` (sends no signal) or `pgrep -P "$PID"` for children. A pid cannot pattern-match `argv`, so referencing paths freely becomes safe. This is what #2204 switched to.
  - **Probe alone in its own call**, already the rule for pattern kills; extend the same isolation rule to pattern PROBES.
  - **Self-exclusion** — `pgrep -af ... | grep -v "^$$ "` or filtering the monitor's own pid — weaker, since the shell wrapper pid differs from `$$`.
- Add the DEATH-DETECTION consequence as its own sentence: any liveness/death predicate built on a self-matching probe fails open, so a watcher can never report the death it was armed for. This is the reason the entry matters beyond tidiness.
- Cross-check the in-repo probe call sites for the same collision (grep the workflow surface for bracketed `pgrep` patterns co-occurring with the unbracketed literal in the same command) and report — fix the ones that are wrong, or state that none are.

## Provenance

Surfaced by the #2204 orchestrator during its own Step 10d merge, after the defect produced a false `LAUNCHED` event and an unreachable death branch in its own watcher. Filed per `.claude/rules/workflow-fix-on-bug.md`.

Distinct target and fingerprint from everything else filed from this round: #2412 (Step 5a `--collect-only` probe blind to runtime skew), #2413 (Step 10d base-identity attribution), #2414 (the `origin/main` ungated-upload trunk red), #2409 (per-leg lint fence sized off the idle range), #2402, #2404, and #2204's own `scripts/verify_plan.py` c67 deliverable.

Reference points: `.claude/rules/gotchas.md` § SSH-remote ownership-probe entry (the bracket-one-character rule), `CLAUDE.md` § "Ownership check before any resume/launch on a shared artifact path" and § "Killing local test processes: kill by captured PID, never by pattern" (the sibling rule that already prefers captured PIDs, for a different reason — the pattern-kill suicide — and is the natural precedent to generalize), #2204 `events.jsonl` (both misreads, with the PID-keyed probe output that disproved the false event), #825 (empty-dir false-DONE, same detector-not-work family).
