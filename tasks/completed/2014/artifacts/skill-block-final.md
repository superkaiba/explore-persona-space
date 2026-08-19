
**Monitor until-condition composition (#1739, #1947).** Item 1 segments the
WAIT; these compose the CONDITION. (a) Key completion on a count DECREASE from
the count captured AT ARM TIME (`base=$(probe); until [ "$(probe)" -lt "$base" ]`),
never an absolute `live=N` — already true at arm time, so every re-arm fires a
no-op event and burns a triage turn (#1739). The probe must emit exactly one
integer: an empty or non-numeric capture makes `[ … -lt … ]` a bash error (rc 2)
and the loop spins on a broken probe. (b) NEVER `<count> || echo 0` in a
condition — `pgrep -c` and `grep -c` print `0` AND exit non-zero, so the value is
the two-line `"0\n0"`, the gate never matches, and the watch wedges OPEN (#1947);
fix in `gotchas.md` § count-keyed liveness. Prefer `pgrep -f 'patter[n]' | wc -l`
(bracketing stops the pattern self-matching, but another occurrence of the
literal in the same argv still matches) or an rc-keyed probe
(`step9c_baseline.py probe`). (c) A session-length watch is `persistent: true`
(no timeout; stop via `TaskStop`), not a re-armed bounded arm: `timeout_ms`
defaults to 300000 ms, caps at 3600000 ms, and is IGNORED when `persistent`. It
never blocks a turn, so item 1 still binds. For a pipeline-polled run prefer
`poll_pipeline.py` bg-Bash; Step 6d.2's bounded QUIET-WAIT Monitor is unchanged.
