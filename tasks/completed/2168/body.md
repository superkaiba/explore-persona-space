---
title: 'workflow-fix: UnicodeDecodeError escapes 112 json-guard except tuples (53
  in the watcher, all on read_text paths)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-07T10:13:14Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #2164 round-2 implementer as a one-site follow-up
  (_stalled_session_overrides); the #2164 orchestrator measured the real scope before
  filing: 53 sites in autonomous_session_watch.py (53/53 on read_text paths), 112
  repo-wide in scripts/+src/. #2164 fixed exactly one (_stalled_cap_gpu_hours) because
  its own change routed that helper onto the watcher unwrapped gate-push pass.'
workflow: v1
---
## Overview / Motivation

`except (json.JSONDecodeError, OSError)` wrapped around a
`Path.read_text()` + `json.loads()` pair does **not** catch
`UnicodeDecodeError`, which `read_text()` raises on any file that is not valid
UTF-8. `UnicodeDecodeError` subclasses `ValueError`, not `OSError`, and
`json.JSONDecodeError` only fires after decoding has already succeeded — so a
single encoding-corrupt JSON file propagates an uncaught exception out of a
guard that was clearly written to be total.

Surfaced during #2164 round 2. That task fixed exactly one instance
(`autonomous_session_watch._stalled_cap_gpu_hours`) because its own change had
newly routed that helper onto the watcher's **unwrapped** gate-push pass, where
a corrupt `~/.eps-autonomous/issue-<N>.json` would kill an entire watcher tick.
The #2164 implementer flagged one sibling. The #2164 orchestrator then measured
the actual scope before filing.

## Measured scope (2026-08-07, `origin/main` `68fbf9bf3e`)

- **53** occurrences of the two-element tuple in
  `scripts/autonomous_session_watch.py`.
- **53 of 53** have a `read_text()` call within 8 lines above the handler — i.e.
  every single one is on a decode path and carries the hole.
- **112** occurrences repo-wide across `scripts/` + `src/` (`*.py`).

Reproduce:

```bash
grep -c "except (json.JSONDecodeError, OSError)" scripts/autonomous_session_watch.py   # 53
grep -rn "except (json.JSONDecodeError, OSError)" scripts/ src/ --include="*.py" | wc -l  # 112
```

## Goal

Close the `UnicodeDecodeError` escape everywhere the pattern guards a
`read_text()` + `json.loads()` pair, so a corrupt-encoding state file degrades
to the intended fallback instead of propagating.

## Proposed change

Add `UnicodeDecodeError` explicitly to the tuple at each decode-path site:
`except (json.JSONDecodeError, OSError, UnicodeDecodeError)`. #2164 set the
precedent at `_stalled_cap_gpu_hours` and deliberately chose the **narrow,
explicit** form over broadening to bare `ValueError` — keep that, so the handler
still cannot swallow an unrelated `ValueError` that should propagate.

Mostly mechanical, but **not** blindly `sed`-able. Two judgement calls:

1. **Severity triage.** A site whose caller is already wrapped is latent; a site
   on an unwrapped pass (the #2164 case) is live. Prioritize and state which is
   which rather than treating all 112 as equal.
2. **Is fallback right here?** At a few sites, silently falling back on a
   corrupt file may be the wrong call and a loud failure is better —
   "fail fast, never hide failures" applies. Where the guard's existing
   `OSError`/`JSONDecodeError` branch already returns a default, matching that
   for `UnicodeDecodeError` is consistent; where it re-raises or logs, match
   that instead. Do not convert a loud path into a silent one.

## Acceptance criteria

- Every decode-path site in `scripts/autonomous_session_watch.py` (53) handles
  `UnicodeDecodeError` consistently with its existing fallback behavior.
- The wider `scripts/` + `src/` set (112 total) is either fixed in the same
  sweep or explicitly triaged with a recorded reason for anything deferred —
  a partial sweep must say what it left and why, not go quiet.
- At least one regression test per distinct guard *shape* (not per site):
  write an invalid-UTF-8 file (`b"\xff\xfe..."`), assert the fallback is
  returned rather than an exception propagating. #2164 shipped exactly this
  shape as `test_reported_cap_falls_back_on_encoding_corrupt_registry_entry` —
  reuse it as the template.
- No site is converted from loud-failure to silent-fallback as a side effect.
- A lint or test guard preventing the two-element form from being reintroduced
  on a `read_text()` path would be worth more than the sweep itself. Consider it
  as the primary deliverable; #2164's `test_cap_env_read_is_single_sourced` is
  the in-repo precedent for a source-scan invariant test.

## Notes

`json.JSONDecodeError` is itself a `ValueError` subclass, so any site that
already catches bare `ValueError` is fine and should be left alone. Check before
editing; the grep counts above are the raw two-tuple form only.
