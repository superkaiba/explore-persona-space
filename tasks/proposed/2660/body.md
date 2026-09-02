---
title: 'earlyoom -s 10 swap trigger permanently satisfied: SIGTERMs fleet python subprocesses
  despite 55 GB free RAM'
kind: infra
tags: []
created_at: '2026-09-02T22:38:03Z'
has_clean_result: false
origin_prompt: Surfaced during /issue 2658 Step 5 after earlyoom killed two workflow_lint
  runs and one task.py post-marker; diagnosed as a config consequence (-s 10 vs 0%
  free swap) rather than memory starvation.
workflow: v1
---
---
kind: infra
---

# earlyoom's `-s 10` swap trigger is permanently satisfied, so it SIGTERMs fleet python subprocesses while 55 GB of RAM sits free

## Goal

Stop earlyoom from killing healthy fleet python work on a box with abundant free RAM, by
fixing the trigger that is actually firing. Needs sudo, so it is Thomas's call.

## The measurement

Taken 2026-09-02 during task #2658 Step 5, after earlyoom killed three separate legs of
that session's work (two `workflow_lint.py` runs and one `task.py post-marker`, the last
with rc=143 AFTER its commit had landed):

- `earlyoom -m 10 -s 10 -r 3600 --prefer (^|/)(pytest|python3?)$ --avoid ...`
- `swapon --show`: `/mnt/eps-data/swapfile`, 64 G size, **64 G USED** — 0% free.
- `/proc/meminfo`: `MemAvailable` **55,575,036 kB** (~55 GB free), `SwapFree` **60 kB**
  (observed as low as **8 kB**).

## Why this is a configuration consequence, not memory starvation

`-s 10` means "kill when free SWAP falls below 10%". Swap is 100% used, so that predicate
is **permanently true** and earlyoom kills on essentially every check — regardless of the
55 GB of free RAM, because the RAM trigger (`-m 10`) and the swap trigger are evaluated
independently, not jointly.

`--prefer (^|/)(pytest|python3?)$` then makes the victim selection precise: the preferred
targets are exactly the fleet's python and pytest subprocesses. The `--avoid` list already
protects Claude Code sessions themselves (including the bare-version-string `comm` fix from
the 2026-08-14 incident, where 71 live sessions were SIGTERMed), so what dies is not the
session but the *work the session spawned* — lints, test suites, `task.py` calls. That is
why the symptom reads as random flaky tooling rather than as an OOM event.

Swap is filled by the sessions themselves: 7+ Claude Code processes each hold 200-580 MB
swapped out (top consumer 583,752 kB).

## Recommended fix, and why not the obvious one

**Preferred: disable or lower the swap trigger.** In `/etc/default/earlyoom`, change `-s 10`
to `-s 0` (disables the swap check) or `-s 2`, then `systemctl restart earlyoom`. With
~55 GB free RAM the `-m 10` trigger is nowhere near firing, so this removes the spurious
kills without weakening genuine OOM protection. One line, reversible.

**Do NOT simply `swapoff`/`swapon` to reclaim the swapfile in the current state.** That
would need to fault ~64 G of swapped pages back into RAM against ~55 GB available — it
would fail or trigger a real OOM. Reclaiming the swapfile requires first reducing the
number of resident Claude sessions (or adding RAM); it is not a safe first move.

## Acceptance

`earlyoom` no longer SIGTERMs healthy python subprocesses while `MemAvailable` is
comfortably high. Verify by running a long `workflow_lint.py` or full pytest leg to
completion under the same session load that killed them.

## Provenance

Surfaced by the autonomous `/issue 2658` session on 2026-09-02, which lost three work legs
to it and re-measured the cause rather than re-reporting "swap is exhausted". Filed at
`proposed` WITHOUT spawning a session: the fix requires sudo, so an auto-driven session
could not execute it and would only burn tokens.
