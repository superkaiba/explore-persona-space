---
title: 'gotchas.md: fellows remote shell is zsh — unmatched glob aborts the whole
  SSH command (nomatch), silently truncating chained monitoring probes'
kind: infra
tags: []
created_at: '2026-08-13T09:23:15Z'
has_clean_result: false
parent_id: 1336
workflow: v1
---
## Goal

Add a `.claude/rules/gotchas.md` entry for a remote-shell trap that makes SSH monitoring probes fail SILENTLY-ISH: the login shell on the fellows SLURM nodes (charmander) is **zsh**, and zsh's default `nomatch` aborts the ENTIRE remote command when ANY glob in it matches nothing — where bash would pass the unmatched pattern through and continue.

## Why this matters (observed 2026-08-13, issue #1336)

A monitoring probe for a queued SLURM job ran, in ONE remote command:

```
ssh charmander "grep -hoE '<banner regex>' $S/logs/*.log $S/logs/issue1336_jobs/*.log | tail -3"
```

`$S/logs/issue1336_jobs/*.log` matched fine, but `$S/logs/*.log` matched NOTHING (no `.log` files at that level). zsh aborted the whole command with:

```
zsh:1: no matches found: /workspace/superkaiba/eps/issue-1336/logs/*.log
```

so the grep never ran against the directory that DID have files. The probe produced an error line where data was expected.

**The dangerous shape is the compound remote command.** When several probes are chained in one `ssh "...; ...; ..."`, a nomatch abort in an early probe means the LATER probes never execute at all — and the caller sees a truncated/zero result that looks like "no signal" rather than "probe broken". In the incident this appeared alongside a genuinely-wrong second probe (an anchor mismatch reading `done cells = 0` when 11 markers existed), and the pair together read as a plausible "the run lost its state" — the exact misdiagnosis the durable record has to not make. Both were probe defects; nothing was actually wrong with the run.

## Requested change

One entry in `.claude/rules/gotchas.md`, sited near the existing SSH-remote ownership-probe entry (the `pgrep -f '[p]attern'` bracket rule) since it is the same family — probe composition traps whose failure mode is a wrong read, not a loud error. Content:

- **Trap:** remote login shell on the fellows cluster is zsh; unmatched glob ⇒ `zsh:1: no matches found` and the whole remote command aborts (`nomatch` is on by default; bash's default is to pass the literal pattern through).
- **Why it bites:** a chained multi-probe `ssh "a; b; c"` loses `b` and `c` entirely, so the caller reads missing data as an absent signal.
- **Rule:** in remote commands, prefer glob-free forms — `grep -r <pat> <dir>` or `find <dir> -name '<pat>' -exec grep ... {} +` (both take a DIRECTORY and cannot nomatch-abort). When a glob is genuinely wanted, either `setopt +o nomatch` / `unsetopt nomatch` first, or run each probe in its OWN `ssh` call so one abort cannot swallow its siblings.
- **Corollary (read-side):** an empty/zero result from a remote probe is UNKNOWN until the probe itself is verified — never a substantive absence claim. Same posture as the existing `gcloud` rc!=0-is-UNKNOWN-never-empty rule in the same file.

Note the byte-budget pressure on `gotchas.md` (the #2189 relocations): keep this to a single tight entry, and prefer siting it in the existing SSH-probe neighborhood over opening a new section.

## Acceptance

- `gotchas.md` carries the entry with the trap, the two glob-free forms, and the per-probe-isolation alternative.
- `LESSONS.md` index row for `gotchas.md` needs no new trigger (the existing "check cross-machine reads" / SSH-probe triggers already cover it) — confirm rather than assume, and update if the lint requires it.
- `workflow_lint.py --check-lessons-index` passes.

## Provenance

Surfaced by the #1336 autonomous session while polling SLURM 12643 (fellows/charmander) during a queued wait; cost one 30-minute poll cycle of blind monitoring on an 8-GPU job. No experiment state was harmed — both probes were re-run corrected and the run's inherited state (11 done markers, 171 banked shards) verified intact.
