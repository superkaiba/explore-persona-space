---
name: pid acquisition is launch-expression capture; pgrep is recovery/monitor-only
description: Relaunch pid resolution: capture $$/$! in the launch chain (pod-side-reporting.md 1d), never a post-hoc pgrep; pgrep survives only as bracketed identity-verified recovery probe + monitor pattern-probe fallback, with self-match hygiene.
type: feedback
---

Acquisition rule (supersedes the old "pgrep -fx is the robust fix"): the pid
you write to `/workspace/logs/issue-<N>.pid` comes from the LAUNCH EXPRESSION
— the launcher's pre-exec `echo $$`, or `$!` captured in the same command
chain as the launch (`.claude/rules/pod-side-reporting.md` § Pid-file launch
contract, clause 1d; #1634). A post-hoc pgrep can capture a transient sibling
(#1112 relaunch, 2026-07-23: an unanchored pgrep populated the pid file AND
the epm:run-launched marker with a wrong pid — two false "exited" monitor
alarms on a healthy dispatcher).

pgrep keeps exactly two roles, and the OLD hygiene still binds in both:
(a) RECOVERY probe when the launch-expression pid was genuinely lost —
bracket one pattern char (`pgrep -f 'issue<N>_dispatc[h]'`) and
identity-verify with `ps -p "$PID" -o args=` BEFORE trusting or writing the
result; (b) ad-hoc monitor pattern-probe FALLBACK run alongside the pid file
(never `poll_pipeline.py` acquisition — the poller's own #1650
marker-signature read is a DETECTION/rescue read, not a source you imitate).
Within those roles: a pattern present in your own SSH command self-matches
the wrapper (#602 respawn 1); even a bracketed pattern can match a
still-alive launch-wrapper subshell — `pgrep -fx "bash scripts/X[.]sh"`
exact-full-cmdline match selects only the exec'd driver (#601 relaunch 3).
Verify `ps -o sess` equals the pid (setsid leader) before posting.

Also retained: a transient single-file HF upload failure at a skip-if-exists
dispatcher's upload gate needs only a resume relaunch — zero GPU re-work.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [pid acquisition: launch-expression capture, pgrep recovery-only](feedback_pgrep_self_match_pidfile.md) — pid file pid comes from $$/$! in the launch chain (1d, #1634); pgrep only as bracketed identity-verified recovery/monitor probe — self-match + wrapper-subshell hygiene (#601, #602)
