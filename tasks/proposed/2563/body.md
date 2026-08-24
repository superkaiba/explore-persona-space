---
title: 'workflow-fix: gate single-flight probe counts a long-lived watcher carrying
  the gate pattern as a live gate, so the spec''s own probe-keyed wait makes the next
  relaunch refuse against no gate'
kind: infra
tags:
- wf-fix
created_at: '2026-08-24T22:37:37Z'
has_clean_result: false
origin_prompt: 'Surfaced during task #2538 Step 10d gate relaunch (2026-08-24): the
  queue wrapper waited the full 2746s fleet cap, then its mandatory single-flight
  re-probe returned LIVE and refused with ''FATAL: a lint gate for issue 2538 is already
  LIVE — refusing to relaunch'' while NO gate was running. The false match was the
  session''s own armed Monitor, whose cmdline carried the literal ''issue-2538-lint-gate''
  via --pattern and the log path. probe is self+ancestor excluding but a Monitor is
  a SIBLING. Stopping the watcher made the probe return rc=0 and the launch succeed.
  Cost: one full 45-min queue cycle.'
workflow: v1
---
# Step 10d/9c gate single-flight probe counts a long-lived WATCHER carrying the gate pattern as a live gate, so the spec's own recommended probe-keyed wait makes the next relaunch refuse against no gate

## Goal

Make the gate single-flight check immune to processes that merely MENTION the gate pattern without being a gate, so that following the spec's recommended probe-keyed wait does not block a subsequent relaunch. Either (a) narrow the probe's match so a watcher/poller is not counted, (b) exclude the caller's own process group / session in addition to self and ancestors, or (c) if the probe must stay permissive, state at the point of use that the wait must not carry the pattern in its command line and give the safe composition.

## Observed failure (task #2538, Step 10d, 2026-08-24)

Sequence, all measured:

1. The lint gate had to be relaunched after a remediation moved the branch tip.
2. Relaunch went through the spec's bounded gate-fleet queue (`.claude/skills/issue/steps/18-step-10d.md:1057`: fleet exit 3 => sleep 60, elapsed cap 2700s, then launch fail-open). The fleet was genuinely over cap, so the wrapper waited the **full 2746 seconds**.
3. At cap expiry the wrapper ran the mandatory single-flight re-probe and got LIVE, so it refused:

```
[gate-fleet] cap-expired after 2746s — launching over cap (fail-open)
FATAL: a lint gate for issue 2538 is already LIVE — refusing to relaunch
```

4. **No gate was running.** `pgrep -af 'issue-2538-lint-gat[e]\.sh'` was empty, there was no gate log, and no rc sentinel.

The false match was the session's own armed `Monitor` bash process. Its command line contained the literal `issue-2538-lint-gate` twice: once in `probe --pattern 'issue-2538-lint-gate'` and once in the log path `/tmp/issue-2538-lint-gate.log`. The probe matches `re.search` against space-joined `/proc/<pid>/cmdline`, so the watcher matched.

Confirmed by elimination: stopping the watcher and re-probing returned rc=0 (CLEAR) with the tree otherwise untouched; the subsequent launch succeeded; a probe taken once the gate was genuinely up returned rc=3 listing exactly the 4 real gate processes and no watcher.

Cost: one full 45-minute queue cycle wasted, plus a second queue wait to get back in line.

## Why this is a surface gap and not only a caller mistake

The probe is documented as "self- + ancestor-pid excluding". A `Monitor` is neither: it is a SIBLING, so it is counted as a foreign live match. That is consistent with the docs but interacts badly with what the spec tells the caller to do a few lines earlier:

> `steps/18-step-10d.md:1046-1048` — "WAIT or reap per the Step 9c 1b single-flight statement, and key any improvised wait on **process exit** (the probe exiting 0 — CLEAR), never on verdict-file existence alone"

Following that instruction literally produces a long-lived process whose command line contains the pattern. So the recommended wait shape poisons the very single-flight check that guards the next relaunch. The existing warning at `steps/18-step-10d.md:1035`-ish ("Run the probe in its OWN Bash call — the harness wrapper embeds the full compound-command text in its own cmdline") covers the SAME-CALL self-match and is why an ancestor exclusion exists; it does not cover a persistent sibling watcher.

Note the asymmetry with the launcher script itself: `/tmp/issue-<N>-lint-gate-launch.sh` also carries the pattern in its own name, but it is the probe's ANCESTOR and so is correctly excluded. That is why the first launch works and only a relaunch performed while a watcher is armed fails.

## Suggested direction (not prescriptive; the planner owns the design)

- **Option (a): narrow the match.** Require the matched cmdline to look like a gate rather than merely mention it (for example the workload script path as argv[1], or an exact `bash <path>/issue-<N>-lint-gate.sh` shape). Risk: over-narrowing reintroduces #2459's opposite failure, where a real worker is missed and a healthy gate reads as clear.
- **Option (b): widen the exclusion.** Exclude the caller's process group or session id alongside self and ancestors. Cheap and targeted, but only helps when the watcher shares the caller's group/session, which a `setsid` launcher deliberately does not.
- **Option (c): fix it in the prose at the point of use.** State next to the probe-keyed-wait instruction that the wait must not carry the pattern literally, and show the safe composition: key liveness on the launcher-reported PID, or split the path across a variable (`PFX=/tmp/issue-<N>-lint; LOG="$PFX-gate.log"`) so the literal never appears contiguously. This is what #2538 did to recover.
- **Worth a mechanical pin:** a check that the gate-launch recipe's own documented wait snippet does not contain the probe pattern contiguously, so a future edit cannot reintroduce the self-poisoning shape into the spec's recommended snippet.
- **Consider a diagnostic:** when the probe reports LIVE, have it print the matched pid + argv (it already prints one `pid<TAB>args` line per match). #2538's wrapper discarded that output to `/dev/null`, which turned a one-line diagnosis into a multi-step elimination. Encouraging callers to surface the matched argv on refusal would have made this self-evident.

## Distinct from the open siblings

- **#2459** ("session-scoped pgrep misses the gate's `start_new_session` pytest worker — healthy Step 9c/10d gates read as wedged") is the INVERSE direction: a real gate process not matched. This task is a non-gate process matched. A fix for one can worsen the other, so they should be reconciled together.
- Not related to #2557 / #2553 (Step 5a family map and SPECS omissions), which #2538 also hit this session.

## Provenance

workflow_fix_target: `scripts/step9c_baseline.py` (the `probe` matcher / exclusion set) and/or `.claude/skills/issue/steps/18-step-10d.md` (the probe-keyed-wait instruction at lines 1046-1048)

Surfaced by task #2538's own Step 10d gate relaunch (2026-08-24). Evidence on #2538: the queue log showing `cap-expired after 2746s` immediately followed by the FATAL refusal, the empty `pgrep` for the gate script at that moment, and the rc=0 probe after the watcher was stopped.
